import copy
import json
from pathlib import Path
import tempfile
import unittest

from pp_calendar.calendar import build_calendar, events_asof, assert_coverage, publish_calendar, query_events, select_snapshot

SOURCE = Path(__file__).resolve().parents[1] / 'external_data/macro_events/sources/2026-10-14/verified_schedule.json'


class MacroCalendarTests(unittest.TestCase):
    def setUp(self):
        self.bundle = json.loads(SOURCE.read_text())

    def test_dst_and_unique_events(self):
        c = build_calendar(self.bundle)
        rows = {r['event_id']: r for r in c['events']}
        self.assertEqual(len(rows), 82)
        self.assertEqual(rows['US_CPI_2025-12']['event_time_utc'], '2026-01-13T13:30:00Z')
        self.assertEqual(rows['US_CPI_2026-03']['event_time_utc'], '2026-04-10T12:30:00Z')
        self.assertEqual(rows['US_FOMC_2026-04-29_statement']['event_time_utc'], '2026-04-29T18:00:00Z')
        self.assertEqual(rows['US_FOMC_2026-04-29_press_conference']['event_time_utc'], '2026-04-29T18:30:00Z')
        self.assertEqual(rows['US_CPI_2026-09']['status'], 'scheduled')

    def test_known_information_and_coverage(self):
        c = build_calendar(self.bundle)
        april = events_asof(c, '2026-04-01T00:00:00Z')
        self.assertIn('US_CPI_2026-03', {r['event_id'] for r in april})
        self.assertNotIn('US_CPI_2026-04', {r['event_id'] for r in april})
        assert_coverage(c, '2026-04-01T00:00:00Z', '2026-08-01T00:00:00Z')
        with self.assertRaises(ValueError):
            assert_coverage(c, '2028-01-02T00:00:00Z', '2028-01-03T00:00:00Z')

    def test_duplicate_source_and_bad_date_rejected(self):
        b = copy.deepcopy(self.bundle); b['records'].append(b['records'][0])
        with self.assertRaises(ValueError): build_calendar(b)
        b = copy.deepcopy(self.bundle); b['records'][0]['source_id'] = 'missing'
        with self.assertRaises(ValueError): build_calendar(b)

    def test_immutable_version(self):
        with tempfile.TemporaryDirectory() as d:
            publish_calendar(self.bundle, Path(d))
            original = (Path(d) / 'calendar.json').read_bytes()
            b = copy.deepcopy(self.bundle); b['records'][0]['local_time'] = '09:00:00'
            with self.assertRaises(FileExistsError): publish_calendar(b, Path(d))
            self.assertEqual(original, (Path(d) / 'calendar.json').read_bytes())

    def test_unknown_history_is_not_an_empty_calendar(self):
        c = build_calendar(self.bundle)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            query_events(c, '2026-08-07T12:00:00Z', '2026-08-07T14:00:00Z', '2026-08-07T12:00:00Z')
        known = query_events(c, '2026-04-10T12:00:00Z', '2026-04-10T14:00:00Z', '2026-04-10T12:00:00Z')
        self.assertEqual([e['event_id'] for e in known], ['US_CPI_2026-03'])
        # Even an apparently empty historical interval needs a then-published snapshot.
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            query_events(c, '2026-08-08T12:00:00Z', '2026-08-08T14:00:00Z', '2026-08-08T12:00:00Z')
        self.assertEqual(query_events(c, '2026-11-08T12:00:00Z', '2026-11-08T14:00:00Z', '2026-11-08T12:00:00Z'), [])

    def test_select_snapshot_and_no_silent_event_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); publish_calendar(self.bundle, root)
            b = copy.deepcopy(self.bundle); b['as_of_utc'] = '2026-10-15T12:00:00Z'
            publish_calendar(b, root)
            self.assertEqual(select_snapshot(root, '2026-10-14T20:00:00Z')['as_of_utc'], self.bundle['as_of_utc'])
            with self.assertRaisesRegex(ValueError, 'No snapshot'):
                select_snapshot(root, '2026-10-13T20:00:00Z')
            b['as_of_utc'] = '2026-10-16T12:00:00Z'; b['records'].pop()
            with self.assertRaisesRegex(ValueError, 'delete'):
                publish_calendar(b, root)
