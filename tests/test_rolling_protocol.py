"""The expanded evaluation must stay prequential and preserve the July boundary."""
import importlib.util
from pathlib import Path
import unittest
import pandas as pd
from pp_data.io import ModelSplit

ROOT=Path(__file__).resolve().parents[1]


class RollingProtocolTests(unittest.TestCase):
    def api(self):
        path=ROOT/'pp_models/rolling_protocol.py';self.assertTrue(path.exists(),'rolling protocol missing')
        spec=importlib.util.spec_from_file_location('pp_models.rolling_protocol',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

    def pool(self):
        times=['2026-02-28T00:00Z','2026-03-02T00:00Z','2026-03-31T00:00Z',
               '2026-04-01T01:00Z','2026-04-30T23:00Z','2026-05-01T01:00Z','2026-06-30T20:00Z']
        close=['2026-02-28T01:00Z','2026-03-02T01:00Z','2026-04-01T00:00Z',
               '2026-04-01T02:00Z','2026-05-01T02:00Z','2026-05-01T03:00Z','2026-06-30T21:00Z']
        x=pd.DataFrame({'v':range(7)});y=pd.Series(range(7),dtype=float)
        meta=pd.DataFrame({'created_at_utc':pd.to_datetime(times,utc=True),'closed_at_utc':pd.to_datetime(close,utc=True)})
        return ModelSplit(x,y,meta)

    def test_april_uses_only_prior_thirty_days_settled_strictly_before_day(self):
        m=self.api();p=self.pool();b=list(m.rolling_batches(p,m.load_protocol()))
        d,h,q=b[0]
        self.assertEqual(d,pd.Timestamp('2026-04-01',tz='UTC'))
        self.assertEqual(h.tolist(),[1]);self.assertEqual(q.tolist(),[3])

    def test_month_boundary_labels_scored_once_and_wait_for_settlement_before_training(self):
        m=self.api();p=self.pool();b=list(m.rolling_batches(p,m.load_protocol()))
        byday={d:(h,q) for d,h,q in b}
        self.assertEqual(byday[pd.Timestamp('2026-04-30',tz='UTC')][1].tolist(),[4])
        self.assertNotIn(4,byday[pd.Timestamp('2026-05-01',tz='UTC')][0])
        self.assertEqual([int(i) for _,_,q in b for i in q],[3,4,5,6])

    def test_prohibits_july_in_input_and_invalid_period_extension(self):
        m=self.api();p=self.pool();p.metadata.loc[6,'created_at_utc']=pd.Timestamp('2026-07-01',tz='UTC')
        with self.assertRaises(ValueError):list(m.rolling_batches(p,m.load_protocol()))
        config=m.load_protocol();config['validation_end']='2026-08-01T00:00:00Z'
        with self.assertRaises(ValueError):m.validate_protocol(config)


if __name__=='__main__':unittest.main()
