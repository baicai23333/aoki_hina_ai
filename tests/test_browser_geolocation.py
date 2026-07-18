import math
import unittest

from browser_geolocation import (
    BrowserGeolocationError,
    normalize_geolocation_result,
)


class BrowserGeolocationTests(unittest.TestCase):
    def test_success_is_rounded_and_accuracy_is_bucketed(self):
        result = normalize_geolocation_result(
            {
                "status": "success",
                "latitude": 23.129123,
                "longitude": 113.264456,
                "accuracy_m": 143.2,
                "ignored": "not persisted",
            }
        )

        self.assertEqual(
            result,
            {
                "status": "success",
                "latitude": 23.13,
                "longitude": 113.26,
                "accuracy_m": 100,
            },
        )

    def test_errors_are_reduced_to_an_allowlist(self):
        self.assertEqual(
            normalize_geolocation_result({"status": "error", "code": "evil"}),
            {"status": "error", "code": "unknown"},
        )
        self.assertEqual(
            normalize_geolocation_result(
                {"status": "error", "code": "permission_denied", "message": "secret"}
            ),
            {"status": "error", "code": "permission_denied"},
        )

    def test_invalid_coordinates_fail_closed(self):
        invalid_values = (
            {"status": "success", "latitude": 91, "longitude": 0},
            {"status": "success", "latitude": 0, "longitude": -181},
            {"status": "success", "latitude": math.nan, "longitude": 0},
            {"status": "success", "latitude": True, "longitude": 0},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(BrowserGeolocationError):
                    normalize_geolocation_result(value)

    def test_none_means_no_interaction_yet(self):
        self.assertIsNone(normalize_geolocation_result(None))


if __name__ == "__main__":
    unittest.main()
