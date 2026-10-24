import io
import unittest

import numpy as np
from PIL import Image

from prepare_demo_cheese import image_values, joint_values


class ConversionTests(unittest.TestCase):
    def test_normalized_joint_slots_preserved(self):
        values = np.linspace(0, 1, 8, dtype=np.float32)
        np.testing.assert_array_equal(joint_values(values, "action"), values)

    def test_bad_joints_rejected(self):
        for values in ([0] * 7, [float("nan")] * 8, [-0.1] * 8, [1.1] * 8):
            with self.subTest(values=values), self.assertRaises(ValueError):
                joint_values(values, "action")

    def test_embedded_image_decoded_without_camera_swap(self):
        buffer = io.BytesIO()
        Image.new("RGB", (224, 224), (255, 0, 0)).save(buffer, format="PNG")
        image = image_values({"bytes": buffer.getvalue()})
        self.assertEqual(image.shape, (224, 224, 3))
        self.assertEqual(image[0, 0].tolist(), [255, 0, 0])

    def test_external_image_paths_rejected(self):
        with self.assertRaises(ValueError):
            image_values({"bytes": None, "path": "missing.png"})


if __name__ == "__main__":
    unittest.main()
