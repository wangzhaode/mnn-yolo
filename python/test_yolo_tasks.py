import unittest

import numpy as np

from yolo_tasks import dense_output, instance_masks, preprocess, resize_align_corners, select_predictions


class TaskTests(unittest.TestCase):
    def test_class_aware_nms(self):
        # Three identical boxes: same-class suppression must not remove another class.
        raw = np.array([[[50, 50, 50], [50, 50, 50], [40, 40, 40], [40, 40, 40],
                         [.9, .8, .01], [.01, .02, .85]]], dtype=np.float32)
        rows = select_predictions(raw, 'detect', 'raw', 2, 0, .25, .45)
        self.assertEqual(rows.shape, (2, 6))
        self.assertEqual(set(rows[:, 5]), {0, 1})

    def test_end_to_end_boxes_are_not_suppressed(self):
        rows = np.array([[[1, 2, 3, 4, .9, 0], [1, 2, 3, 4, .8, 0]]], dtype=np.float32)
        self.assertEqual(len(select_predictions(rows, 'detect', 'end2end', 2, 0, .25, .45)), 2)

    def test_instance_mask_restores_padding_and_crops_box(self):
        rows = np.array([[2, 3, 6, 5, .9, 0, 1]], dtype=np.float32)
        proto = np.ones((1, 1, 4, 4), dtype=np.float32)
        mask = instance_masks(rows, proto, (4, 8, 3), (8, 8), (1, 0, 2))[0]
        expected = np.zeros((4, 8), dtype=bool)
        expected[1:3, 2:6] = True
        np.testing.assert_array_equal(mask, expected)

    def test_semantic_class_map_restores_padding(self):
        classes = np.zeros((1, 8, 8), dtype=np.float32)
        classes[:, 2:6, 4:] = 2
        actual = dense_output(classes, 'semantic', (4, 8, 3), (8, 8), (1, 0, 2))
        np.testing.assert_array_equal(actual[:, :4], 0)
        np.testing.assert_array_equal(actual[:, 4:], 2)

    def test_preprocess_uses_rgb_and_centered_letterbox(self):
        image = np.zeros((2, 4, 3), dtype=np.uint8)
        image[:] = (10, 20, 30)
        tensor, transform = preprocess(image, 4, 4, 'detect')
        self.assertEqual(transform, (1, 0, 1))
        np.testing.assert_allclose(tensor[0, :, 1, 0], np.array([30, 20, 10]) / 255)
        np.testing.assert_allclose(tensor[0, :, 0, 0], np.array([114, 114, 114]) / 255)

    def test_depth_bilinear_alignment(self):
        a = np.array([[0, 1], [2, 3]], dtype=np.float32)
        expected = np.array([[0, .5, 1], [1, 1.5, 2], [2, 2.5, 3]])
        np.testing.assert_allclose(resize_align_corners(a, (3, 3)), expected)


if __name__ == '__main__':
    unittest.main()
