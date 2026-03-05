import unittest

import pandas as pd

from scripts.run_superperformance_custom_walkforward import _build_selection_schedule, _membership_by_day


class SuperperformanceCustomWalkforwardTests(unittest.TestCase):
    def test_selection_schedule_prefers_stronger_symbol_and_membership_carries(self) -> None:
        dates = pd.bdate_range('2019-01-02', periods=520)
        strong = pd.DataFrame({
            'close': [100.0]*len(dates),
            'adv50': [5e6]*len(dates),
            'adr20': [4.0]*len(dates),
            'ret63': [0.12]*len(dates),
            'ret126': [0.22]*len(dates),
            'prox252': [0.92]*len(dates),
        }, index=dates)
        weak = pd.DataFrame({
            'close': [50.0]*len(dates),
            'adv50': [5e6]*len(dates),
            'adr20': [4.0]*len(dates),
            'ret63': [0.01]*len(dates),
            'ret126': [0.02]*len(dates),
            'prox252': [0.60]*len(dates),
        }, index=dates)
        metrics = {'STRONG': strong, 'WEAK': weak}
        cfg = {
            'custom_universe': {
                'rebalance_months': 3,
                'max_names': 1,
                'min_price': 8,
                'max_price': 250,
                'min_adv50': 1e6,
                'max_adv50': 1e8,
                'min_adr20_pct': 3.0,
                'min_ret63': 0.05,
                'min_ret126': 0.10,
                'min_proximity_52w': 0.75,
                'ret126_weight': 0.55,
                'ret63_weight': 0.25,
                'proximity_52w_weight': 0.20,
            }
        }
        schedule = _build_selection_schedule(metrics, cfg, '2020-04-01', '2020-12-31')
        self.assertTrue(schedule)
        first_nonempty = next((item for item in schedule if item[1]), None)
        self.assertIsNotNone(first_nonempty)
        self.assertEqual(first_nonempty[1], ['STRONG'])
        membership = _membership_by_day(pd.to_datetime(['2020-05-15','2020-08-15','2020-11-15']), schedule)
        self.assertEqual(membership[0], {'STRONG'})
        self.assertEqual(membership[1], {'STRONG'})
        self.assertEqual(membership[2], {'STRONG'})


if __name__ == '__main__':
    unittest.main()
