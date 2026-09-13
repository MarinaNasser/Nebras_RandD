import unittest
import numpy as np
from pipeline import CLASSES, crop_boxes, split_records


class PipelineTests(unittest.TestCase):
    def test_patient_split(self):
        rows = [dict(patient=f'{c}_{p}', label=c, path=f'{c}_{p}_{i}')
                for c in CLASSES for p in range(10) for i in range(3)]
        result = split_records(rows, 42)
        self.assertEqual(result, split_records(rows, 42))
        groups = {s: {r['patient'] for r in result if r['split'] == s}
                  for s in ['train', 'val', 'test']}
        self.assertFalse(groups['train'] & groups['val'])
        self.assertFalse(groups['train'] & groups['test'])
        self.assertFalse(groups['test'] & groups['val'])
        for s in groups:
            self.assertEqual({r['label'] for r in result if r['split'] == s}, set(CLASSES))

    def test_masks_and_bounds(self):
        mask = np.zeros((100, 100), np.uint8)
        self.assertEqual(crop_boxes(mask), [])
        mask[20:40, 30:50] = 1
        self.assertEqual(crop_boxes(mask), [[27, 17, 53, 43]])
        mask[:10, :10] = 1
        self.assertEqual(crop_boxes(mask)[1], [0, 0, 12, 12])
        mask[80, 80] = 1
        self.assertEqual(len(crop_boxes(mask)), 2)


if __name__ == '__main__': unittest.main()
