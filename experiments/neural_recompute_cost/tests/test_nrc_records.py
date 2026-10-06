"""Focused tests for nrc_records (stdlib only): python tests/test_nrc_records.py"""

from __future__ import annotations

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nrc_records as R  # noqa: E402


def ev(t, stage, event, kind="point", **extra):
    return {"t": t, "pc": t + 1000.0, "stage": stage, "event": event, "kind": kind, **extra}


def epoch_events(stage="s", epochs=2, t0=100.0):
    """Synthetic epochs: 10 s each = 1 path gen + 0.5 resample + 6 opt + 1.5 val + 0.4 ckpt + 0.1 snap + 0.5 other."""
    out, t = [], t0
    for e in range(epochs):
        out.append(ev(t, stage, "epoch", "begin", key=e))
        for name, d in (("path_generation", 1.0), ("resample", 0.5), ("optimization", 6.0), ("validation", 1.5),
                        ("checkpoint_write", 0.4), ("snapshot_write", 0.1)):
            key = {"file": "last.ckpt", "epoch": e} if name == "checkpoint_write" else e
            out.append(ev(t, stage, name, "begin", key=key))
            t += d
            out.append(ev(t, stage, name, "end", key=key))
        out.append(ev(t, stage, "val_metrics", epoch=e, metrics={"val/loss": 1.0 / (e + 1)}))
        t += 0.5
        out.append(ev(t, stage, "epoch", "end", key=e))
    return out


class TestEventsAndPhases(unittest.TestCase):
    def test_writer_roundtrip_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "e.jsonl"
            log = R.EventLog(p, "stage_a")
            log.begin("fit")
            log.emit("note", value=3)
            log.end("fit")
            log.close()
            rows = R.load_events(p)
            self.assertEqual([r["kind"] for r in rows], ["begin", "point", "end"])
            self.assertTrue(all(r["stage"] == "stage_a" for r in rows))
            iv = R.intervals(rows)
            self.assertEqual(len(iv), 1)
            self.assertGreaterEqual(iv[0]["seconds"], 0.0)
            # a malformed line is refused
            p.write_text('{"t": 1, "stage": "x", "event": "y", "kind": "sideways"}\n')
            with self.assertRaises(ValueError):
                R.load_events(p)

    def test_nan_refused(self):
        with tempfile.TemporaryDirectory() as d:
            log = R.EventLog(Path(d) / "e.jsonl", "s")
            with self.assertRaises(ValueError):
                log.emit("bad", x=float("nan"))
            log.close()
        with self.assertRaises(ValueError):
            R.dumps({"x": math.inf})

    def test_pairing_errors(self):
        with self.assertRaises(ValueError):
            R.intervals([ev(1, "s", "a", "end")])
        with self.assertRaises(ValueError):
            R.intervals([ev(1, "s", "a", "begin"), ev(2, "s", "a", "begin")])
        with self.assertRaises(ValueError):
            R.intervals([ev(5, "s", "a", "begin"), ev(2, "s", "a", "end")])
        # same event, different keys pair independently; an unclosed begin stays visible
        iv = R.intervals([ev(1, "s", "a", "begin", key=0), ev(2, "s", "a", "begin", key=1), ev(4, "s", "a", "end", key=0)])
        closed = [r for r in iv if r["seconds"] is not None]
        self.assertEqual(len(closed), 1)
        self.assertAlmostEqual(closed[0]["seconds"], 3.0)
        self.assertEqual(sum(r["end"] is None for r in iv), 1)

    def test_phase_accounting_closes(self):
        events = epoch_events(epochs=3)
        iv = R.intervals(events)
        rows = R.epoch_table(iv, "s", events)
        R.check_epoch_accounting(rows)
        self.assertEqual(len(rows), 3)
        for r in rows:
            self.assertAlmostEqual(r["wall_s"], 10.0)
            self.assertAlmostEqual(r["optimization_s"], 6.0)
            self.assertAlmostEqual(r["path_generation_s"], 1.0)
            self.assertAlmostEqual(r["other_s"], 0.5)
            parts = sum(r[k + "_s"] for k in R.EPOCH_LEAVES) + r["other_s"]
            self.assertAlmostEqual(parts, r["wall_s"])
            self.assertIn("val/loss", r["val_metrics"]["metrics"])
        tot = R.phase_totals(iv, "s")
        self.assertAlmostEqual(tot["optimization"]["seconds"], 18.0)
        self.assertEqual(tot["epoch"]["count"], 3)

    def test_double_counting_detected(self):
        events = [ev(0, "s", "epoch", "begin", key=0), ev(0, "s", "optimization", "begin", key=0),
                  ev(5, "s", "optimization", "end", key=0), ev(0, "s", "validation", "begin", key=0),
                  ev(5, "s", "validation", "end", key=0), ev(6, "s", "epoch", "end", key=0)]
        rows = R.epoch_table(R.intervals(events), "s", events)
        with self.assertRaises(ValueError):
            R.check_epoch_accounting(rows)


class TestTimeline(unittest.TestCase):
    def test_cumulative(self):
        c = R.cumulative(100.0, {"a": 100.0, "b": 162.5, "c": None})
        self.assertEqual(c, {"a": 0.0, "b": 62.5, "c": None})
        with self.assertRaises(ValueError):
            R.cumulative(100.0, {"early": 99.0})

    def test_stage_table_and_gaps(self):
        chain = [ev(10, "chain", "stage", "begin", key="a"), ev(20, "chain", "stage", "end", key="a", exit_code=0),
                 ev(20.25, "chain", "stage", "begin", key="b"), ev(50, "chain", "stage", "end", key="b", exit_code=0)]
        st = R.stage_table(chain)
        self.assertEqual([s["stage"] for s in st], ["a", "b"])
        self.assertIsNone(st[0]["gap_before_s"])
        self.assertAlmostEqual(st[1]["gap_before_s"], 0.25)
        self.assertAlmostEqual(st[0]["wall_s"] + st[1]["gap_before_s"] + st[1]["wall_s"], 50 - 10)

    def test_checkpoint_mapping(self):
        events = epoch_events(epochs=2, t0=100.0)
        rows = R.epoch_table(R.intervals(events), "s", events)
        m = R.map_checkpoints(rows, t0=90.0, export_s=0.2)
        # epoch 0 checkpoint ends at 100 + 1 + 0.5 + 6 + 1.5 + 0.4 = 109.4
        self.assertAlmostEqual(m[0]["checkpoint_available_elapsed_s"], 19.4)
        self.assertAlmostEqual(m[0]["usable_elapsed_s"], 19.6)
        self.assertAlmostEqual(m[1]["checkpoint_available_elapsed_s"], 29.4)
        # a WSL clock running 3 s ahead is corrected by the measured offset
        m2 = R.map_checkpoints(rows, t0=90.0, offset_s=-3.0)
        self.assertAlmostEqual(m2[0]["checkpoint_available_elapsed_s"], 16.4)
        with self.assertRaises(ValueError):
            R.map_checkpoints(rows, t0=200.0)


class TestRecoveryAndBudget(unittest.TestCase):
    def test_first_recovery(self):
        trace = [{"usable_elapsed_s": 30.0, "recovers": True, "epoch": 2},
                 {"usable_elapsed_s": 10.0, "recovers": False, "epoch": 0},
                 {"usable_elapsed_s": 20.0, "recovers": True, "epoch": 1},
                 {"usable_elapsed_s": None, "recovers": True, "epoch": 9}]
        self.assertEqual(R.first_recovery(trace)["epoch"], 1)
        self.assertIsNone(R.first_recovery([{"usable_elapsed_s": 1.0, "recovers": False}]))
        # only an explicit True counts (a missing verdict is not a recovery)
        self.assertIsNone(R.first_recovery([{"usable_elapsed_s": 1.0}]))

    def test_budget_and_category(self):
        b = R.frame_budget_ratios(1.0)
        self.assertAlmostEqual(b["x_30fps_33.3ms"], 30.0)
        self.assertAlmostEqual(b["x_60fps_16.7ms"], 60.0)
        self.assertEqual(R.frame_budget_ratios(None)["x_30fps_33.3ms"], None)
        self.assertEqual(R.latency_category(0.05), "A")
        self.assertEqual(R.latency_category(0.1), "B")
        self.assertEqual(R.latency_category(9.99), "B")
        self.assertEqual(R.latency_category(10.0), "C")
        self.assertEqual(R.latency_category(7200.0), "C")


class TestProvenanceAndRecord(unittest.TestCase):
    def test_require_clean(self):
        R.require_clean({"commit": "abc", "dirty": []})
        with self.assertRaises(SystemExit):
            R.require_clean({"commit": "abc", "dirty": [" M x.py"]})
        with self.assertRaises(SystemExit):
            R.require_clean({"commit": None, "dirty": []})
        R.require_clean({"commit": "abc", "dirty": [" M x.py"]}, allow_dirty=True)

    def test_git_state_on_this_repo(self):
        st = R.git_state(Path(__file__).resolve().parents[3])
        self.assertRegex(st["commit"], r"^[0-9a-f]{40}$")
        self.assertIsInstance(st["dirty"], list)

    def _record(self):
        rec = {k: None for k in R.RECORD_FIELDS}
        rec.update(schema=R.SCHEMA, project_dirty=[], total_schedule_s=100.0,
                   first_recovery={"usable_elapsed_s": 50.0, "epoch": 3})
        return rec

    def test_record_validation_and_roundtrip(self):
        rec = self._record()
        R.validate_record(rec)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "r.json"
            R.write_json(p, rec)
            back = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(back, rec)
            R.validate_record(back)
        bad = dict(rec)
        del bad["inference"]
        with self.assertRaises(ValueError):
            R.validate_record(bad)
        with self.assertRaises(ValueError):
            R.validate_record(dict(rec, project_dirty=[" M a"]))
        with self.assertRaises(ValueError):
            R.validate_record(dict(rec, first_recovery={"usable_elapsed_s": 101.0}))
        with self.assertRaises(ValueError):
            R.validate_record(dict(rec, schema="other"))

    def test_process_clock(self):
        import time

        t = R.process_create_time()
        self.assertIsNotNone(t)
        self.assertLess(t, time.time())
        self.assertGreater(t, time.time() - 3600)
        self.assertGreater(R.host_peak_memory_mb(), 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
