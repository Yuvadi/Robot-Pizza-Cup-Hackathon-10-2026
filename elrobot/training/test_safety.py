import unittest

from safety import Joint, ensure_fresh, goals, motor_ids, verify_station_config


class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.joints = [Joint(2000, 0.5, 1000, 3000)] * 8

    def test_eight_joint_mapping(self):
        self.assertEqual(goals([0.52] * 8, self.joints, 100), [2040] * 8)

    def test_movement_threshold_exact(self):
        self.assertEqual(goals([0.55] * 8, self.joints, 100), [2100] * 8)
        with self.assertRaises(ValueError):
            goals([0.551] * 8, self.joints, 100)

    def test_no_silent_clipping_or_truncation(self):
        for actions in ([0.5] * 6, [0.5] * 9, [float("nan")] * 8, [float("inf")] * 8,
                        [-0.01] * 8, [1.01] * 8):
            with self.subTest(actions=actions), self.assertRaises(ValueError):
                goals(actions, self.joints, 100)

    def test_invalid_calibration(self):
        for joint in (Joint(2000, 0.5, 1000, 1000), Joint(500, 0.5, 1000, 3000),
                      Joint(2000, 0.1, 1000, 3000), Joint(5000, 0.5, 1000, 3000)):
            with self.subTest(joint=joint), self.assertRaises(ValueError):
                goals([0.5] * 8, [joint] * 8, 100)

    def test_wraparound_calibration(self):
        joints = [Joint(0, 196 / 396, 3900, 200)] * 8
        self.assertEqual(goals([206 / 396] * 8, joints, 100), [10] * 8)

    def test_positive_bounded_limit_required(self):
        for limit in (0, -1, 201):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                goals([0.5] * 8, self.joints, limit)

    def test_motor_id_order_and_count(self):
        self.assertEqual(motor_ids("1,2,3,4,5,6,7,8"), list(range(1, 9)))
        for value in ("1,2,3,4,5,6", "1,2,3,4,5,6,7,7", "2,1,3,4,5,6,7,8", "0,2,3,4,5,6,7,8"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                motor_ids(value)

    def test_bus_must_be_explicit_not_auto_or_another_arm(self):
        for bus in ("auto", "LEADER"):
            with self.subTest(bus=bus), self.assertRaises(ValueError):
                verify_station_config(
                    {"inference": [{"queue-id": "inference/normvla", "format": "normvla", "st3215-bus": bus}]},
                    "FOLLOWER", "inference/normvla",
                )
        verify_station_config(
            {"inference": [{"queue-id": "inference/normvla", "format": "normvla", "st3215-bus": "FOLLOWER"}]},
            "FOLLOWER", "inference/normvla",
        )

    def test_freshness_all_sensors_and_no_future_clocks(self):
        now = 10_000_000_000
        ensure_fresh(now - 100, [now - 200], now, 2)
        for stamps in ([now - 3_000_000_000], [now + 1], [0], []):
            with self.subTest(stamps=stamps), self.assertRaises(ValueError):
                ensure_fresh(now, stamps, now, 2)

    def test_stale_error_reports_age_and_frame(self):
        with self.assertRaisesRegex(ValueError, r"frame age is 290\.000s"):
            ensure_fresh(1_000_000_000, [1_000_000_000], 291_000_000_000, 2)


if __name__ == "__main__":
    unittest.main()
