import ast
import random
import unittest
from collections import Counter, defaultdict
from pathlib import Path

# Exercise split logic without importing the model or image-processing stack.
source = Path(__file__).with_name('REAL-Colon_characterize.py').read_text()
tree = ast.parse(source)
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'video_level_split')
namespace = dict(random=random, Counter=Counter, defaultdict=defaultdict, MIN_VIDEOS_PER_CLASS=3)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(__file__), 'exec'), namespace)
split = namespace['video_level_split']


class SplitTests(unittest.TestCase):
    def test_coverage_and_video_isolation(self):
        records = [dict(video=str(v), label=c, path=f'{v}_{c}_{i}')
                   for v in range(15) for c in (['common', 'rare'] if v < 3 else ['common'])
                   for i in range(2)]
        for seed in range(20):
            rows, classes = split(records, ['common', 'rare'], seed)
            self.assertEqual((rows, classes), split(records, classes, seed))
            self.assertEqual(len(rows), len(records))
            groups = []
            for s in ('train', 'val', 'test'):
                subset = [r for r in rows if r['split'] == s]
                self.assertEqual({r['label'] for r in subset}, set(classes))
                groups.append({r['video'] for r in subset})
            self.assertEqual([len(g) for g in groups], [9, 3, 3])
            self.assertEqual(sum(map(len, groups)), len(set.union(*groups)))

    def test_insufficient_class_is_not_dropped(self):
        with self.assertRaisesRegex(ValueError, 'rare.*2'):
            split([dict(video=str(i), label='rare') for i in range(2)], ['rare'], 42)

    def test_impossible_multilabel_overlap(self):
        # Each triple must use three colors; all four triples on four videos cannot.
        records = [dict(video=str(v), label=str(c))
                   for c in range(4) for v in range(4) if v != c]
        with self.assertRaisesRegex(ValueError, 'overlap'):
            split(records, list(map(str, range(4))), 42)

    def test_missing_class(self):
        with self.assertRaisesRegex(ValueError, 'No extracted crops'):
            split([], ['absent'], 42)

    def test_absent_metadata_classes_do_not_block_observed_classes(self):
        observed = ['AD', 'NO POLYP', 'HP']
        records = [dict(video=str(v), label=c) for v in range(12) for c in observed]
        rows, classes = split(records, observed + ['OTHER', 'SSL', 'TSA'], 42)
        self.assertEqual(classes, observed)
        self.assertEqual(len(rows), len(records))
        for s in ('train', 'val', 'test'):
            self.assertEqual({r['label'] for r in rows if r['split'] == s}, set(observed))


if __name__ == '__main__':
    unittest.main()
