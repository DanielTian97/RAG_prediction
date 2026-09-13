"""Check final summary selection against Tables 1 and 2 of arXiv:2601.14546.

These are artifact consistency tests, not a rerun of the published experiment.
Run: python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'analysis'))
from report_results import build_tables


class PaperResults(unittest.TestCase):
    def setUp(self):
        self.single = pd.read_csv(ROOT/'analysis/ecir_res/single_output_v2.csv')
        self.union = pd.read_csv(ROOT/'analysis/ecir_res/union_output_v2.csv')

    def test_published_table_cells(self):
        first, second, flags, figure = build_tables(self.single, self.union)
        expected1 = [
            [.0694,-.0643,.0994,.0610,-.1288,.1210],
            [.1485,.1625,.1288,.1609,.2218,.2344],
            [.1739,.2027,.1025,.2166,.2992,.1922],
            [.1565,.0136,.1913,.1353,-.0483,.2470],
            [.1898,.1474,.1352,.2079,.1641,.1630]]
        expected2 = [
            [.2146,.2008,.2155,.2454,.2861,.2952],
            [.2365,.2001,.2110,.2821,.3040,.3094],
            [.2419,.2136,.2101,.2948,.3071,.3075],
            [.2380,.2350,.2093,.2945,.3253,.3088],
            [.1468,.1351,.1470,.3037,.2579,.2873],
            [.2387,.2283,.2513,.3507,.3616,.3915],
            [.2534,.2280,.2474,.3653,.3726,.3988],
            [.2573,.2404,.2476,.3729,.3737,.3977],
            [.2539,.2563,.2472,.3727,.3856,.3974]]
        self.assertEqual(first.iloc[:,1:].round(4).values.tolist(), expected1)
        self.assertEqual(second.iloc[:,2:].round(4).values.tolist(), expected2)
        self.assertEqual(len(figure), 40)
        self.assertFalse(figure.duplicated(['Task','k','Method']).any())
        self.assertEqual(set(figure.k), {2,3,5,7,10})

    def test_duplicate_or_missing_results_are_rejected(self):
        selected = self.single.query('task=="nq" and k==2 and QPP_Method=="nqc"').iloc[[0]]
        with self.assertRaises(ValueError):
            build_tables(pd.concat([self.single, selected]), self.union)
        with self.assertRaises(ValueError):
            build_tables(self.single.drop(selected.index), self.union)


if __name__ == '__main__':
    unittest.main()
