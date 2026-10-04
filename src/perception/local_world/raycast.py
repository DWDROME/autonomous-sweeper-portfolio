"""Bounded scalar DDA compiled without fast-math; NumPy reference stays in world."""
import numpy as np

try:
    from numba import njit
except ImportError:
    traverse = None
else:
    @njit(cache=True, nogil=True, boundscheck=True)
    def traverse(origin, endpoints, resolution, lower, extent):
        start = np.floor(origin/resolution).astype(np.int64)
        targets = np.floor(endpoints/resolution).astype(np.int64)
        visited = np.zeros(int(extent[0]*extent[1]*extent[2]),dtype=np.bool_)
        for index in range(len(endpoints)):
            # Scalar state avoids allocating temporary NumPy arrays at every
            # crossed voxel; the tie and endpoint rules match the reference.
            x,y,z=start[0],start[1],start[2]
            tx,ty,tz=targets[index,0],targets[index,1],targets[index,2]
            dx,dy,dz=endpoints[index,0]-origin[0],endpoints[index,1]-origin[1],endpoints[index,2]-origin[2]
            sx,sy,sz=int(np.sign(dx)),int(np.sign(dy)),int(np.sign(dz))
            ix=resolution/abs(dx) if dx!=0. else np.inf
            iy=resolution/abs(dy) if dy!=0. else np.inf
            iz=resolution/abs(dz) if dz!=0. else np.inf
            cx=((x+(sx>0))*resolution-origin[0])/dx if dx!=0. else np.inf
            cy=((y+(sy>0))*resolution-origin[1])/dy if dy!=0. else np.inf
            cz=((z+(sz>0))*resolution-origin[2])/dz if dz!=0. else np.inf
            for _ in range(abs(tx-x)+abs(ty-y)+abs(tz-z)+1):
                if x==tx and y==ty and z==tz:
                    break
                rx,ry,rz=x-lower[0],y-lower[1],z-lower[2]
                if rx<0 or rx>=extent[0] or ry<0 or ry>=extent[1] or rz<0 or rz>=extent[2]:
                    raise ValueError('ray escaped its bounded volume')
                key = (rx*extent[1]+ry)*extent[2]+rz
                visited[key] = True
                minimum = min(cx,cy,cz)
                if minimum >= 1.-1e-12:
                    break
                if cx<=minimum+1e-12:
                    x+=sx
                    cx+=ix
                if cy<=minimum+1e-12:
                    y+=sy
                    cy+=iy
                if cz<=minimum+1e-12:
                    z+=sz
                    cz+=iz
        indices=np.flatnonzero(visited)
        cells=np.empty((len(indices),3),dtype=np.int64)
        for i in range(len(indices)):
            key=indices[i]
            cells[i,0]=key//(extent[1]*extent[2])+lower[0]
            cells[i,1]=(key//extent[2])%extent[1]+lower[1]
            cells[i,2]=key%extent[2]+lower[2]
        return cells
