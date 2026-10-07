"""Candidate-level checkpoints containing only statistics, guarded by a process lock."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from pp_data.config import DEFAULT_OUTPUT
from .runs import canonical_json, fingerprint, run_destination


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(canonical_json(value))
            stream.flush()
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TrialStore:
    def __init__(self, path, identity):
        self.path, self.identity = Path(path), identity

    def status(self, state, **extra):
        atomic_json(self.path / "provenance/manifest.json", {
            "producer": "pp_model_study", "identity": self.identity, "status": state, **extra})

    def trial(self, resolved, phase, period, compute):
        spec = {"resolved": resolved, "phase": phase, "period": period}
        trial_id = fingerprint(spec)
        model = resolved["selected_models"][0]
        path = self.path / "trials" / model / trial_id
        if path.exists():
            record = json.loads((path / "record.json").read_text(encoding="utf-8"))
            if record.get("spec") != spec or set(record.get("files", {})) != {"summary.csv", "daily.csv"}:
                raise ValueError("Candidate checkpoint identity mismatch")
            captured = {}
            for name, digest in record["files"].items():
                from io import BytesIO
                payload = (path / name).read_bytes()
                if hashlib.sha256(payload).hexdigest() != digest:
                    raise ValueError("Candidate checkpoint checksum mismatch")
                captured[name] = pd.read_csv(BytesIO(payload), float_precision="round_trip",
                                             dtype={"value": str})
            summary, daily = captured["summary.csv"], captured["daily.csv"]
        else:
            summary, daily = compute()
            overall = summary.loc[(summary.cohort == resolved["cohort"]) & (summary.group == "overall")]
            if len(overall) != 1 or not np.isfinite(overall.rmse_per_btc.iloc[0]):
                raise ValueError("Candidate has no valid overall score")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=".candidate_", dir=path.parent) as temp:
                stage = Path(temp)
                for name, frame in (("summary.csv", summary), ("daily.csv", daily)):
                    frame.to_csv(stage / name, index=False, encoding="utf-8-sig", float_format="%.17g")
                (stage / "record.json").write_text(canonical_json({"spec": spec,
                    "files": {n: file_digest(stage / n) for n in ("summary.csv", "daily.csv")}}), encoding="utf-8")
                # The study process lock excludes concurrent publication. Rename the
                # complete directory atomically so interrupted candidates never look cached.
                os.rename(stage, path)
        overall = summary.loc[(summary.cohort == resolved["cohort"]) & (summary.group == "overall")]
        if len(overall) != 1 or not np.isfinite(overall.rmse_per_btc.iloc[0]):
            raise ValueError("Invalid cached candidate score")
        return {"id": trial_id, "resolved": resolved, "phase": phase,
                "score": float(overall.rmse_per_btc.iloc[0]), "summary": summary, "daily": daily}


@contextmanager
def study_workspace(root, run_id, identity, resume=False, dataset_dir=DEFAULT_OUTPUT):
    if resume:
        path = Path(root).resolve() / run_id
        # Reuse the normal name/path checks, without permitting existing unrelated runs.
        try:
            run_destination(root, run_id, dataset_dir)
        except FileExistsError:
            pass
        if not path.is_dir() or path.is_symlink():
            raise ValueError("No resumable study directory")
    else:
        path = run_destination(root, run_id, dataset_dir)
        path.mkdir(parents=True, exist_ok=False)
        (path / "provenance").mkdir()
    with (path / "provenance/.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Study is already running in another process") from error
        store = TrialStore(path, identity)
        if resume:
            previous = json.loads((path / "provenance/manifest.json").read_text(encoding="utf-8"))
            if previous.get("producer") != "pp_model_study" or previous.get("identity") != identity:
                raise ValueError("Study configuration/code/data/environment changed; cannot resume")
            if previous.get("status") == "complete":
                for name, digest in previous.get("files", {}).items():
                    if file_digest(path / name) != digest:
                        raise ValueError("Completed study summary checksum mismatch")
        store.status("running")
        try:
            yield store
        except BaseException as error:
            store.status("failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
