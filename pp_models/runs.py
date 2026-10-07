"""Compact run artifacts and reproducible validation-to-test handoff."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import tempfile
from pp_data.config import DEFAULT_DATA, DEFAULT_OUTPUT, PROJECT
from pp_data.io import checksum

OUTPUT_ROOT = PROJECT / "outputs" / "model_runs"
ARTIFACTS = {"metrics/summary.csv", "metrics/daily.csv", "provenance/config.json"}


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"


def fingerprint(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def snapshot(dataset_dir=DEFAULT_OUTPUT):
    files = list((PROJECT / "pp_models").glob("*.py")) + [PROJECT / "requirements-models.txt", PROJECT / "requirements.txt"]
    packages = ("numpy", "pandas", "scipy", "pygam", "interpret-core", "scikit-learn", "joblib", "threadpoolctl")
    environment = {"python": platform.python_version()}
    for package in packages:
        try:
            environment[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            environment[package] = None
    return {"dataset_sha256": checksum(Path(dataset_dir) / "manifest.json"),
            "code_sha256": {str(p.relative_to(PROJECT)): checksum(p) for p in sorted(files)},
            "environment": environment}


def run_destination(root, run_id, dataset_dir=DEFAULT_OUTPUT):
    if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", run_id):
        raise ValueError("run-id must start with a lowercase letter and contain only letters, digits, _ or -")
    root = Path(root).resolve()
    protected = [DEFAULT_DATA, Path(dataset_dir), PROJECT / "pp_data", PROJECT / "pp_audit",
                 PROJECT / "pp_models", PROJECT / "config", PROJECT / "docs"]
    for protected_dir in protected:
        protected_dir = protected_dir.resolve()
        if root == protected_dir or protected_dir in root.parents or root in protected_dir.parents:
            raise ValueError("Run output overlaps source, configuration or canonical data")
    destination = root / run_id
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Run already exists: {destination}; use a new run-id")
    return destination


def validate_reference(path, resolved, provenance):
    root = Path(path)
    manifest = json.loads((root / "provenance/manifest.json").read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or manifest.get("producer") != "pp_models"
            or manifest.get("status") != "complete" or manifest.get("split") != "validation"):
        raise ValueError("Test requires a completed validation run")
    if manifest.get("configuration_sha256") != fingerprint(resolved) or manifest.get("source") != provenance:
        raise ValueError("Validation configuration, code, data or dependency versions differ")
    artifacts = manifest.get("artifacts", {})
    if set(artifacts) != ARTIFACTS:
        raise ValueError("Invalid validation artifact list")
    captured = {}
    for name in sorted(ARTIFACTS):
        payload = (root / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifacts[name]:
            raise ValueError(f"Validation artifact checksum mismatch: {name}")
        captured[name] = payload
    if json.loads(captured["provenance/config.json"]) != resolved:
        raise ValueError("Validation config does not match resolved settings")
    return {"path": str(root.resolve()), "manifest_sha256": fingerprint(manifest)}


def write_run(root, run_id, split, resolved, provenance, summary, daily, reference=None,
              dataset_dir=DEFAULT_OUTPUT):
    if split not in ("validation", "test") or (split == "test" and reference is None):
        raise ValueError("Invalid split or missing validation reference")
    destination = run_destination(root, run_id, dataset_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".pp_models_", dir=destination.parent) as temporary:
        stage = Path(temporary)
        (stage / "metrics").mkdir()
        (stage / "provenance").mkdir()
        for name, table in (("summary", summary), ("daily", daily)):
            table.to_csv(stage / f"metrics/{name}.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
        (stage / "provenance/config.json").write_text(canonical_json(resolved), encoding="utf-8")
        manifest = {"producer": "pp_models", "version": "model_run_v1", "status": "complete",
                    "split": split, "configuration_sha256": fingerprint(resolved), "source": provenance,
                    "validation_reference": reference,
                    "artifacts": {name: checksum(stage / name) for name in sorted(ARTIFACTS)},
                    "limitations": ["Internal daily prequential evaluation, not official batch scoring",
                                    "Only compact statistics saved; no model checkpoints or row predictions",
                                    "No demand estimation, pricing margin or profit optimization"]}
        (stage / "provenance/manifest.json").write_text(canonical_json(manifest), encoding="utf-8")
        # Reserve exclusively at publication time; never replace an existing run.
        destination.mkdir(exist_ok=False)
        (destination / "metrics").mkdir()
        (destination / "provenance").mkdir()
        for name in sorted(ARTIFACTS) + ["provenance/manifest.json"]:
            os.replace(stage / name, destination / name)
    return destination
