import os
import tempfile
import unittest
from pathlib import Path

from paperdrm.cache_identity import build_cache_fingerprint


class CacheIdentityTests(unittest.TestCase):
    def test_same_inputs_and_parameters_produce_same_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "0_30.jpg"
            image.write_bytes(b"image-a")
            params = {"angle_slice": [2, 2], "subtract_background": True}

            first = build_cache_fingerprint([image], params)
            second = build_cache_fingerprint([image], params)

            self.assertEqual(first, second)

    def test_file_change_invalidates_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "0_30.jpg"
            image.write_bytes(b"image-a")
            first = build_cache_fingerprint([image], {"crop_roi": None})

            image.write_bytes(b"image-b-longer")
            os.utime(image, None)
            second = build_cache_fingerprint([image], {"crop_roi": None})

            self.assertNotEqual(first, second)

    def test_preprocessing_change_invalidates_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = Path(tmp) / "0_30.jpg"
            image.write_bytes(b"image")

            first = build_cache_fingerprint(
                [image],
                {"crop_roi": [0, 0, 100, 100], "theta_min_deg": 30},
            )
            second = build_cache_fingerprint(
                [image],
                {"crop_roi": [10, 0, 100, 100], "theta_min_deg": 30},
            )

            self.assertNotEqual(first, second)

    def test_selected_file_set_and_order_are_part_of_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            first_image = Path(tmp) / "0_30.jpg"
            second_image = Path(tmp) / "0_45.jpg"
            first_image.write_bytes(b"a")
            second_image.write_bytes(b"b")

            one = build_cache_fingerprint([first_image], {})
            two = build_cache_fingerprint([first_image, second_image], {})
            reversed_order = build_cache_fingerprint(
                [second_image, first_image],
                {},
            )

            self.assertNotEqual(one, two)
            self.assertNotEqual(two, reversed_order)
