"""Competing physical profiles, unsupported gaps and prior re-ranking on synthetic scans."""
from dataclasses import replace
import unittest

import numpy as np

from perception.curb_geometry import CurbConfig, CurbResult, CurbSegment, detect_curb

ROAD_BAND = (0., 2., 1.4, 1.6, .95)


class CurbGeometryTest(unittest.TestCase):
    def cloud(self, profile, *, ground=None, gap=None, side=1, grade=(.03, .02)):
        x = np.arange(.0125, 1.9876, .025)
        y = np.arange(.5125, 2.4876, .025)
        x, y = np.meshgrid(x, y, indexing='ij')
        x, y = x.ravel(), y.ravel()
        height = profile(x, y)
        points = np.column_stack((x, side*y, 1.4+grade[0]*x+grade[1]*y+height))
        mask = np.abs(height) < 1e-9 if ground is None else ground(x, y)
        keep = np.ones(len(points), bool) if gap is None else ~gap(x, y)
        return points[keep], mask[keep]

    def run_geometry(self, cloud, side=1, **changes):
        changes.setdefault('sensor_origin', (0., .75*side, .4))
        return detect_curb(cloud[0], cloud[1], (0., 2., .5, 2.5), side, **changes)

    def test_drain_cover_leaves_do_not_replace_raised_curb_on_sloped_ground(self):
        def profile(x, y):
            return np.select([y >= 1.8, (y >= 1.4) & (y < 1.55),
                              (y >= 1.1) & (y < 1.25), (y >= .8) & (y < .95)],
                             [.12, .05, .02, -.03], default=0.)
        for side in (-1, 1):
            with self.subTest(side=side):
                result = self.run_geometry(self.cloud(profile, side=side), side=side)
                self.assertIsInstance(result, CurbResult)
                self.assertEqual(result.reason, 'MEASURED_CHAINS')
                self.assertIsNone(result.diagnostics)
                self.assertTrue(result.segments)
                for segment in result.segments:
                    self.assertIsInstance(segment, CurbSegment)
                    self.assertEqual(segment.boundary_type, 'RAISED')
                    self.assertEqual(segment.xyz.shape, (len(segment.sigma_m), 3))
                    self.assertEqual(segment.score.shape, segment.sigma_m.shape)
                    self.assertTrue(np.all(np.diff(segment.xyz[:, 0]) > 0))
                    np.testing.assert_allclose(np.abs(segment.xyz[:, 1]), 1.8, atol=.026)
                    self.assertTrue(np.all(segment.sigma_m > 0))

    def test_road_band_and_prior_choose_second_sustained_candidate(self):
        cloud = self.cloud(lambda x, y: np.select([y >= 1.8, (y >= 1.) & (y < 1.2)], [.12, .10], default=0.))
        result = self.run_geometry(cloud, prior_xy=[np.array([[0., 1.8], [2., 1.8]])],
                                   road_bands=[(0., 2., 1.7, 1.9, .95)])
        self.assertGreater(result.counts['candidates'], 40)
        self.assertTrue(result.segments)
        for segment in result.segments:
            np.testing.assert_allclose(segment.xyz[:, 1], 1.8, atol=.026)

    def test_flat_road_band_is_flush_and_missing_upper_ground_is_unsupported(self):
        cloud = self.cloud(lambda x, y: np.zeros_like(x))
        result = self.run_geometry(cloud, road_bands=[ROAD_BAND])
        self.assertTrue(result.segments)
        self.assertEqual({segment.boundary_type for segment in result.segments}, {'FLUSH'})
        self.assertFalse(self.run_geometry(cloud).segments)
        self.assertFalse(self.run_geometry(cloud, prior_xy=[np.array([[0., 1.5], [2., 1.5]])]).segments)
        missing = self.cloud(lambda x, y: np.zeros_like(x), gap=lambda x, y: y >= 1.4)
        result = self.run_geometry(missing, road_bands=[ROAD_BAND])
        self.assertEqual(result.reason, 'NO_SUPPORTED_BOUNDARY')
        self.assertFalse(result.segments)

    def test_gradual_ramp_is_not_raised(self):
        cloud = self.cloud(lambda x, y: np.clip((y-1.3)*.3, 0., .12), ground=lambda x, y: y < 1.3)
        result = self.run_geometry(cloud)
        self.assertTrue(result.segments)
        self.assertEqual({segment.boundary_type for segment in result.segments}, {'RAMP'})

    def test_upper_platform_cannot_extend_a_short_rise_into_a_ramp(self):
        cloud = self.cloud(lambda x, y: np.select([y >= 1.35, y >= 1.3], [.067, .0335], default=0.),
                           ground=lambda x, y: y < 1.3)
        result = self.run_geometry(cloud)
        self.assertFalse(result.segments)

    def test_local_steep_rise_is_not_diluted_by_platform_average_slope(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                cloud = self.cloud(lambda x, y: np.clip((y-1.3)*1.03, 0., .103),
                                   ground=lambda x, y: y < 1.3, side=side)
                result = self.run_geometry(cloud, side=side)
                self.assertFalse(result.segments)

    def test_mixed_tall_returns_are_not_clipped_to_a_legal_curb_height(self):
        cloud = self.cloud(lambda x, y: np.where(y < 1.5, 0.,
                           np.where(np.floor((y-.5)/.025).astype(int) % 2 == 0, .12, .6)),
                           ground=lambda x, y: y < 1.5)
        self.assertFalse(self.run_geometry(cloud).segments)

    def test_mixed_vertical_face_uses_its_supported_top_not_its_median(self):
        cloud = self.cloud(lambda x, y: np.where(y >= 1.55, .103, 0.),
                           gap=lambda x, y: (y >= 1.5) & (y < 1.55))
        x, h = np.meshgrid(np.arange(.0125, 1.9876, .025), [0., .01, .02, .035, .05, .08, .103, .103])
        x, h = x.ravel(), h.ravel()
        face = np.column_stack((x, np.full(len(x), 1.523), 1.4+.03*x+.02*1.523+h))
        result = self.run_geometry((np.vstack((cloud[0], face)), np.r_[cloud[1], np.zeros(len(face), bool)]))
        self.assertTrue(result.segments)
        self.assertEqual({segment.boundary_type for segment in result.segments}, {'RAISED'})
        for segment in result.segments:
            np.testing.assert_allclose(segment.xyz[:, 1], 1.523, atol=.026)

    def test_low_face_returns_in_ground_mask_do_not_break_observed_curb(self):
        for side, grade in ((-1, (.08, -.06)), (1, (.08, -.06)), (1, (0., 0.))):
            with self.subTest(side=side, grade=grade):
                points, mask = self.cloud(lambda x, y: np.where(y >= 1.5, .103, 0.),
                                          side=side, grade=grade)
                x, z = np.meshgrid(np.arange(.9125, 1.3376, .025), [.006, .012, .018])
                x, z = x.ravel(), z.ravel()
                face = np.column_stack((x, np.full(len(x), side*1.48),
                                        1.4+grade[0]*x+grade[1]*1.48+z))
                result = self.run_geometry((np.vstack((points, face)),
                                            np.r_[mask, np.ones(len(face), bool)]), side=side)
                self.assertTrue(any(s.boundary_type == 'RAISED' and
                                    s.xyz[0, 0] < .9 and s.xyz[-1, 0] > 1.35
                                    for s in result.segments))
                if grade == (0., 0.):
                    for segment in result.segments:
                        np.testing.assert_allclose(segment.xyz[:, 2], 1.4, atol=1e-9)

    def test_mixed_cell_uses_nearest_ground_and_raised_returns_for_root(self):
        for side in (-1, 1):
            with self.subTest(side=side):
                cloud = self.cloud(lambda x, y: np.where(y >= 1.48, .103, 0.),
                    side=side, grade=(0., 0.), gap=lambda x, y: (x >= 1.) &
                    (y > 1.45) & (y < 1.48) & (np.floor(x/.025).astype(int) % 2 == 1))
                result = self.run_geometry(cloud, side=side)
                self.assertEqual(len(result.segments), 1)
                segment = result.segments[0]
                self.assertEqual(segment.boundary_type, 'RAISED')
                self.assertLess(segment.xyz[0, 0], .1)
                self.assertGreater(segment.xyz[-1, 0], 1.9)
                # Ground at 1.4625 m and raised returns at 1.4875 m bracket the root.
                np.testing.assert_allclose(side*segment.xyz[:, 1], 1.475, atol=1e-9)
                np.testing.assert_allclose(segment.xyz[:, 2], 1.4, atol=1e-9)

    def test_ground_classified_ramp_is_not_mislabelled_flush_at_road_band_seam(self):
        for direction in (-1., 1.):
            with self.subTest(direction=direction):
                cloud = self.cloud(lambda x, y: direction*np.clip((y-1.3)*.3, 0., .12),
                                   ground=lambda x, y: np.ones(len(x), bool))
                result = self.run_geometry(cloud, road_bands=[(0., 2., 1.25, 1.35, .95)])
                self.assertTrue(result.segments)
                self.assertEqual({segment.boundary_type for segment in result.segments}, {'RAMP'})

    def test_curved_curb_keeps_measured_vertices_without_straight_line_extrapolation(self):
        cloud = self.cloud(lambda x, y: np.where(y >= 1.2+.12*x*x, .12, 0.))
        result = self.run_geometry(cloud, debug=True)
        self.assertTrue(result.segments)
        points = np.concatenate([segment.xyz for segment in result.segments])
        np.testing.assert_allclose(points[:, 1], 1.2+.12*points[:, 0]**2, atol=.055)
        self.assertLess(points[:, 0].min(), .1)
        self.assertGreater(points[:, 0].max(), 1.9)
        candidates = {tuple(row['point']) for row in result.diagnostics['candidates']}
        self.assertTrue(all(tuple(point) in candidates for point in points))

    def test_missing_longitudinal_columns_split_polylines(self):
        cloud = self.cloud(lambda x, y: np.where(y >= 1.5, .12, 0.), gap=lambda x, y: (x > .8) & (x < 1.2))
        result = self.run_geometry(cloud)
        self.assertEqual(len(result.segments), 2)
        for segment in result.segments:
            self.assertTrue(segment.xyz[:, 0].max() <= .8 or segment.xyz[:, 0].min() >= 1.2)

    def test_raised_to_ramp_transition_splits_types(self):
        cloud = self.cloud(lambda x, y: np.where(x < 1., np.where(y >= 1.3, .12, 0.),
                                                np.clip((y-1.3)*.3, 0., .12)),
                           ground=lambda x, y: y < 1.3)
        result = self.run_geometry(cloud)
        self.assertEqual({segment.boundary_type for segment in result.segments}, {'RAISED', 'RAMP'})
        for segment in result.segments:
            x = segment.xyz[:, 0]
            self.assertTrue(np.all(x < 1.) if segment.boundary_type == 'RAISED' else np.all(x >= 1.))

    def test_empty_and_degenerate_ground_do_not_create_flush_or_raised_support(self):
        cloud = self.cloud(lambda x, y: np.where(y >= 1.5, .12, 0.))
        result = self.run_geometry((cloud[0], np.zeros(len(cloud[0]), bool)), road_bands=[ROAD_BAND])
        self.assertFalse(result.segments)
        self.assertEqual(result.reason, 'NO_SUPPORTED_BOUNDARY')
        self.assertEqual(result.counts['candidates'], 0)
        empty = self.run_geometry((np.empty((0, 3)), np.empty(0, bool)), debug=True)
        self.assertFalse(empty.segments)
        self.assertEqual(empty.counts, dict(input_points=0, band_points=0, cells=0, plane_fits=0, candidates=0))
        self.assertEqual(empty.diagnostics, dict(candidates=[], cells=[]))

    def test_mixed_height_cells_do_not_count_ground_returns_as_raised_support(self):
        cloud = self.cloud(lambda x, y: np.where((y >= 1.5) & (np.floor((y-.5)/.025).astype(int) % 2 == 0), .24, 0.),
                           ground=lambda x, y: y < 1.5)
        keep = np.floor(cloud[0][:, 0]/.025).astype(int) % 2 == 0
        result = self.run_geometry(tuple(values[keep] for values in cloud))
        self.assertFalse(result.segments)

    def test_budget_excess_is_explicit_and_never_silently_downsamples(self):
        cloud = self.cloud(lambda x, y: np.where(y >= 1.5, .12, 0.))
        for config, reason, band_points in ((replace(CurbConfig(), max_points=20), 'POINT_BUDGET_EXCEEDED', len(cloud[0])),
                                            (replace(CurbConfig(), max_cells=20), 'CELL_BUDGET_EXCEEDED', 0)):
            with self.subTest(reason=reason):
                result = self.run_geometry(cloud, config=config, debug=True)
                self.assertEqual(result.reason, reason)
                self.assertFalse(result.segments)
                self.assertIsNone(result.diagnostics)
                self.assertEqual(result.counts, dict(input_points=len(cloud[0]), band_points=band_points,
                                                     cells=0, plane_fits=0, candidates=0))

    def test_debug_diagnostics_index_the_input_cloud_and_expose_adaptive_thresholds(self):
        cloud = self.cloud(lambda x, y: np.where(y >= 1.5, .12, 0.))
        for origin, near in (((0., .75, .4), True), ((0., .75, 12.), False)):
            with self.subTest(origin=origin):
                result = self.run_geometry(cloud, sensor_origin=origin, debug=True)
                candidates = result.diagnostics['candidates']
                self.assertEqual(len(candidates), result.counts['candidates'])
                self.assertTrue(candidates)
                for row in candidates:
                    self.assertEqual(row['boundary_type'], 'RAISED')
                    self.assertEqual(set(row['score_terms']), {'height', 'support', 'persistence', 'history', 'semantic'})
                    self.assertGreater(row['thresholds']['step_min_m'], .06)
                    self.assertEqual(row['thresholds']['range_m'] < 4., near)
                    self.assertTrue(cloud[1][row['ground_ids']].all())
                    self.assertGreaterEqual(len(row['raised_ids']), 3)
                    self.assertFalse(cloud[1][row['raised_ids']].any())
                cells = result.diagnostics['cells']
                self.assertEqual(len(cells), result.counts['cells'])
                self.assertEqual(sum(cell['point_count'] for cell in cells), result.counts['band_points'])
                self.assertTrue(all(len(cell['point_ids']) == cell['point_count']
                                    and cell['point_ids'].max() < len(cloud[0]) for cell in cells))


if __name__ == '__main__':
    unittest.main()
