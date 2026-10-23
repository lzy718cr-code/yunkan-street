import unittest

from traffic_app.analytics import AnalyticsConfig, Detection, TrafficAnalytics


def detection(track_id=1, bottom_y=40, class_name="car", confidence=0.9, x=50):
    return Detection(class_id=0, class_name=class_name, confidence=confidence,
                     bbox=(x - 5, bottom_y - 10, x + 5, bottom_y), track_id=track_id)


class TrafficAnalyticsTests(unittest.TestCase):
    def test_current_counts_do_not_masquerade_as_cumulative_tracks(self):
        analytics = TrafficAnalytics()
        first = analytics.update([detection(1), detection(None), detection(2, class_name="bus")], 0, 0, 100, 100)
        self.assertEqual(first.current_count, 3)
        self.assertEqual(first.tracked_current_count, 2)
        self.assertEqual(first.cumulative_unique_count, 2)
        self.assertEqual(first.counts_by_class, {"car": 2, "bus": 1})
        second = analytics.update([detection(1, class_name="truck"), detection(None)], 1, 1, 100, 100)
        self.assertEqual(second.cumulative_unique_count, 2)
        self.assertEqual(second.cumulative_by_class, {"car": 1, "bus": 1})
        self.assertEqual(analytics.summary()["detection_observations"], 5)

    def test_drawn_section_counts_crossings_in_any_orientation(self):
        # A straight line on the ground stays a straight line under perspective,
        # so a segment drawn in the image is a valid cross-section.
        flat = AnalyticsConfig(section=(0.0, 0.5, 1.0, 0.5), line_hysteresis=.02)
        analytics = TrafficAnalytics(flat)
        analytics.update([detection(1, bottom_y=40)], 0, 0, 100, 100)
        crossed = analytics.update([detection(1, bottom_y=60)], 1, 1, 100, 100)
        self.assertEqual(sum(crossed.line_counts.values()), 1)

        diagonal = AnalyticsConfig(section=(0.0, 0.0, 1.0, 1.0), line_hysteresis=.02)
        slant = TrafficAnalytics(diagonal)
        slant.update([detection(1, bottom_y=40, x=70)], 0, 0, 100, 100)
        result = slant.update([detection(1, bottom_y=70, x=40)], 1, 1, 100, 100)
        self.assertEqual(sum(result.line_counts.values()), 1)

        stay = TrafficAnalytics(flat)
        stay.update([detection(1, bottom_y=30)], 0, 0, 100, 100)
        still = stay.update([detection(1, bottom_y=35)], 1, 1, 100, 100)
        self.assertEqual(sum(still.line_counts.values()), 0)

    def test_hysteresis_and_one_crossing_per_direction_per_id(self):
        analytics = TrafficAnalytics(AnalyticsConfig(line_y=.5, line_hysteresis=.02))
        results = []
        for index, bottom in enumerate([40, 49, 51, 53, 51, 49, 47, 55, 40]):
            results.append(analytics.update([detection(bottom_y=bottom)], index, index, 100, 100))
        self.assertEqual(results[2].line_counts, {"up": 0, "down": 0})
        self.assertEqual(results[3].line_counts, {"up": 0, "down": 1})
        self.assertEqual(results[5].line_counts, {"up": 0, "down": 1})
        self.assertEqual(results[-1].line_counts, {"up": 1, "down": 1})
        events = [event for result in results for event in result.events]
        self.assertEqual([event.direction for event in events], ["down", "up"])

    def test_tracker_id_zero_is_valid_and_duplicate_ids_use_best_confidence(self):
        analytics = TrafficAnalytics()
        result = analytics.update([detection(0, confidence=.2), detection(0, class_name="bus", confidence=.9)], 0, 0, 100, 100)
        self.assertEqual(result.current_count, 1)
        self.assertEqual(result.cumulative_unique_count, 1)
        self.assertEqual(result.counts_by_class, {"bus": 1})

    def test_roi_dwell_uses_source_seconds_and_resets_on_missing_frame(self):
        analytics = TrafficAnalytics(AnalyticsConfig(roi=(.2, .2, .8, .8), dwell_threshold_seconds=2))
        self.assertEqual(analytics.update([detection()], 10, 0, 100, 100).dwell_current_count, 0)
        second = analytics.update([detection()], 12, 1, 100, 100)
        self.assertEqual(second.dwell_current_count, 1)
        self.assertEqual(second.events[0].event_type, "roi_dwell")
        self.assertEqual(second.events[0].duration_seconds, 2)
        analytics.update([], 13, 2, 100, 100)
        returned = analytics.update([detection()], 20, 3, 100, 100)
        self.assertEqual(returned.dwell_seconds_by_track, {1: 0})
        self.assertEqual(returned.dwell_current_count, 0)
        qualifying = analytics.update([detection()], 22, 4, 100, 100)
        self.assertEqual(qualifying.dwell_current_count, 1)
        self.assertEqual(analytics.summary()["roi_dwell_events"], 2)

    def test_leaving_roi_resets_visit_and_untracked_detections_have_no_dwell(self):
        analytics = TrafficAnalytics(AnalyticsConfig(roi=(.2, .2, .8, .8), dwell_threshold_seconds=1))
        analytics.update([detection(), detection(None)], 0, 0, 100, 100)
        inside = analytics.update([detection(), detection(None)], 1, 1, 100, 100)
        self.assertEqual(inside.dwell_current_count, 1)
        analytics.update([detection(x=90)], 2, 2, 100, 100)
        result = analytics.update([detection()], 3, 3, 100, 100)
        self.assertEqual(result.dwell_current_count, 0)
        self.assertEqual(result.dwell_seconds_by_track, {1: 0})

    def test_expired_states_do_not_inflate_counts_or_make_phantom_crossings(self):
        analytics = TrafficAnalytics(AnalyticsConfig(track_ttl_seconds=2))
        analytics.update([detection(bottom_y=40)], 0, 0, 100, 100)
        analytics.update([detection(bottom_y=60)], 1, 1, 100, 100)
        analytics.update([], 4, 2, 100, 100)
        self.assertEqual(analytics.active_track_count, 0)
        reappeared = analytics.update([detection(bottom_y=40)], 5, 3, 100, 100)
        self.assertEqual(reappeared.cumulative_unique_count, 1)
        self.assertEqual(reappeared.line_counts, {"up": 0, "down": 1})
        crossed_again = analytics.update([detection(bottom_y=60)], 6, 4, 100, 100)
        self.assertEqual(crossed_again.line_counts, {"up": 0, "down": 1})

    def test_new_source_reset_clears_all_statistics(self):
        analytics = TrafficAnalytics()
        analytics.update([detection()], 20, 100, 100, 100)
        analytics.reset()
        result = analytics.update([], 0, 0, 100, 100)
        self.assertEqual(result.cumulative_unique_count, 0)
        self.assertEqual(analytics.summary()["analyzed_frames"], 1)
        self.assertEqual(analytics.summary()["source_duration_seconds"], 0)

    def test_backward_timestamps_and_repeated_frames_are_rejected(self):
        analytics = TrafficAnalytics()
        analytics.update([], 10, 2, 100, 100)
        with self.assertRaises(ValueError):
            analytics.update([], 9, 3, 100, 100)
        with self.assertRaises(ValueError):
            analytics.update([], 11, 2, 100, 100)
        with self.assertRaises(ValueError):
            analytics.update([], float("nan"), 3, 100, 100)
        with self.assertRaises(ValueError):
            analytics.update([], 11, 3, 0, 100)

    def test_invalid_roi_and_nonfinite_boxes_rejected(self):
        with self.assertRaises(ValueError):
            AnalyticsConfig(roi=(.8, .1, .2, .9))
        with self.assertRaises(ValueError):
            Detection(0, "car", .9, (0, 0, float("nan"), 1), 1)


if __name__ == "__main__":
    unittest.main()
