"""Fit or independently evaluate a prediction certificate.

With TrajZoo on PYTHONPATH:
    python -m calibration.certificate fit --method bootstrap \
        --checkpoint best.pth --dataset heldout.pkl --output certificate.json
    python -m calibration.certificate evaluate --checkpoint best.pth \
        --certificate certificate.json --dataset evaluation.pkl --output metrics.json

Fitting uses supplied calibration scenes, not predictor-training data. No
prediction cache is written. Confidence concerns approximate group moments.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

if __package__:
    from .bootstrap import (BOOTSTRAP_DRAWS, CONFIDENCE, MIN_VALID_ROWS, SEED,
                            estimate_bootstrap_profiles, estimate_motion_profiles)
    from .inference import (BATCH_SIZE, calibration_rows, collect_predictions,
                            create_predictor, load_dataset, validate_predictor)
    from .metrics import evaluate_predictions, require_disjoint_groups, wire_covariances
    from .runtime import Certificate
    from .motion import MOTION_DEFINITION, MOTION_TAGS, classify_motion
else:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from calibration.bootstrap import (BOOTSTRAP_DRAWS, CONFIDENCE, MIN_VALID_ROWS, SEED,
                                       estimate_bootstrap_profiles, estimate_motion_profiles)
    from calibration.inference import (BATCH_SIZE, calibration_rows, collect_predictions,
                                      create_predictor, load_dataset, validate_predictor)
    from calibration.metrics import evaluate_predictions, require_disjoint_groups, wire_covariances
    from calibration.runtime import Certificate
    from calibration.motion import MOTION_DEFINITION, MOTION_TAGS, classify_motion


def _file_identity(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(Path(path).resolve()), "sha256": digest.hexdigest()}


def _save_json(document, output_path):
    """Publish a complete report without overwriting an existing artifact."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".{}.".format(path.name)) as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.link(handle.name, path)


def _ensure_new_output(output_path):
    if Path(output_path).exists() or Path(output_path).is_symlink():
        raise FileExistsError("Output already exists: {}".format(output_path))


def _settings(method):
    return {
        "bootstrap_draws": BOOTSTRAP_DRAWS, "seed": SEED,
        "min_fit_rows": MIN_VALID_ROWS, "confidence": CONFIDENCE,
        "grouping": ("object category x predicted motion x native horizon" if method == "motion_bootstrap"
                     else "object category x native horizon x fit-only covariance-trace bin"),
        "uncertainty_bins": (None if method == "motion_bootstrap"
                             else "fit-only covariance-trace quintiles; duplicate edges removed"),
        "resampling": "whole trajectories within each category; one draw shared across horizons",
        "moment": "uncentered raw error second moment",
        "radius": "95th percentile bootstrap spectral-norm deviation",
        "confidence_scope": "approximate 95% per group; not simultaneous",
    }


def _assumptions(layout, method):
    return {
        "sample_independence_verified": False,
        "source_group_ids_available": True,
        "population": ("calibration target distribution within category, predicted motion and native horizon"
                       if method == "motion_bootstrap" else
                       "calibration target distribution within category, native horizon and trace bin"),
        "bootstrap": "empirical trajectory resampling; overlapping windows and cross-CAV dependence are not removed",
        "conditional_moments": "a group moment is an approximation for inputs assigned to that group",
        "transfer_to_runtime": "group second-moment transfer is assumed, not verified",
        "cross_source_error_moments": "zero cross-error moments are required by the gate, not established here",
        "input_scope": "observed histories and {} at native forecast timestamps".format(
            "preserved scene context" if layout == "scene" else "independent tracks"),
        "confidence_scope": "approximate per group; no simultaneous or deployment guarantee",
        "interpolation_transfer": "nearest native covariance/risk-bound pairing does not certify interpolated means",
    }


def _motion_tags(rows, means, obs_len):
    return [classify_motion(row["history"], prediction, obs_len)
            for row, prediction in zip(rows, means)]


def create_certificate(checkpoint_path, dataset_path, output_path, method="bootstrap"):
    """Run inference and fit the chosen estimator on complete calibration targets."""
    if method not in ("bootstrap", "motion_bootstrap"):
        raise ValueError("Unknown certificate method: {}".format(method))
    _ensure_new_output(output_path)
    dataset = load_dataset(dataset_path)
    rows = calibration_rows(dataset)
    predictor = create_predictor(checkpoint_path)
    contract = validate_predictor(predictor, dataset["metadata"])
    means, covariances = collect_predictions(predictor, dataset, contract["layout"])
    targets = np.stack([row["future"] for row in rows])
    categories = [row["category"] for row in rows]
    motion_tags = None
    if method == "motion_bootstrap":
        motion_tags = _motion_tags(rows, means, contract["obs_len"])
        profiles, exclusions = estimate_motion_profiles(
            means, targets, covariances, categories, motion_tags, contract["sample_fps"])
        usable = sum(profile["usable_for_gate"] for profile in profiles)
    else:
        profiles, exclusions = estimate_bootstrap_profiles(
            means, targets, covariances, categories, contract["sample_fps"])
        usable = sum(any(item["usable_for_gate"] for item in profile["bins"]) for profile in profiles)
    result = {
        "schema_version": 3 if method == "motion_bootstrap" else 2,
        "estimator": "motion_bootstrap_raw_moment" if method == "motion_bootstrap" else "bootstrap_raw_moment",
        "method": method,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "bootstrap_group_moment_approximate" if usable else "no_usable_profiles",
        "usable_profile_count": usable,
        "checkpoint": {**_file_identity(checkpoint_path), "model_name": predictor.config["model"]["name"]},
        "dataset": {**_file_identity(dataset_path), "metadata": dataset["metadata"],
                    "n_samples": len(rows), "n_scenes": len(dataset["samples"])},
        "inference": {**contract, "seed": SEED, "batch_size": BATCH_SIZE,
                      "device": str(predictor.device), "checkpoint_config": predictor.config},
        "settings": _settings(method), "assumptions": _assumptions(contract["layout"], method),
        "fit_provenance": {
            "purpose": "calibration_fit", "selection": "complete scoring targets in supplied archive",
            "n_fit": len(rows),
            "source_group_ids": sorted({row["source_group_id"] for row in rows}),
            "source_groups": "physical scenarios; all CAV views retain their shared source_group_id",
            "predictor_training_disjointness_verified": False,
            "evaluation_used_for_fitting": False,
        },
        "exclusions": exclusions, "profiles": profiles,
    }
    if motion_tags is not None:
        counts = Counter(zip(categories, motion_tags))
        result.update({
            "risk_unit": "m2", "motion": MOTION_DEFINITION,
            "motion_counts": {
                category: {**{tag: counts[category, tag] for tag in MOTION_TAGS},
                           "unavailable": counts[category, None]}
                for category in sorted(set(categories))
            },
        })
    Certificate(result)
    _save_json(result, output_path)
    print("Certificate saved to: {}".format(Path(output_path).resolve()), flush=True)
    return result


def evaluate_certificate(checkpoint_path, dataset_path, certificate_path, output_path):
    """Evaluate a frozen certificate, rejecting physical fit/evaluation overlap."""
    _ensure_new_output(output_path)
    dataset = load_dataset(dataset_path)
    rows = calibration_rows(dataset)
    certificate = Certificate.from_file(certificate_path, checkpoint_path=checkpoint_path)
    groups = require_disjoint_groups(certificate, rows)
    predictor = create_predictor(checkpoint_path)
    contract = validate_predictor(predictor, dataset["metadata"])
    certificate = Certificate.from_file(certificate_path, checkpoint_path=checkpoint_path, expected=contract)
    means, covariances = collect_predictions(predictor, dataset, contract["layout"])
    targets = np.stack([row["future"] for row in rows])
    categories = [row["category"] for row in rows]
    motion_tags = (_motion_tags(rows, means, contract["obs_len"])
                   if certificate.document.get("method") == "motion_bootstrap" else None)
    native = evaluate_predictions(certificate, means, targets, covariances, categories,
                                  contract["sample_fps"], motion_tags=motion_tags)
    decoded = evaluate_predictions(
        certificate, means, targets, wire_covariances(covariances), categories,
        contract["sample_fps"], grouping_covariances=covariances, slice_name="wire_covariance",
        motion_tags=motion_tags)
    result = {
        "schema_version": 1, "purpose": "independent_certificate_evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "certificate": _file_identity(certificate_path), "checkpoint": _file_identity(checkpoint_path),
        "dataset": {**_file_identity(dataset_path), "metadata": dataset["metadata"]},
        "source_group_ids": groups, "physical_fit_evaluation_disjoint": True,
        "inference": contract, "slices": {"native": native, "wire_covariance": decoded},
        "scope": "wire_covariance retains native means; position codec and interpolation need aligned runtime captures",
    }
    _save_json(result, output_path)
    print("Certificate metrics saved to: {}".format(Path(output_path).resolve()), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("fit", "evaluate"):
        command = commands.add_parser(name, allow_abbrev=False)
        if name == "fit":
            command.add_argument("--method", choices=("bootstrap", "motion_bootstrap"), default="bootstrap")
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        if name == "evaluate":
            command.add_argument("--certificate", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "fit":
        create_certificate(args.checkpoint, args.dataset, args.output, args.method)
    else:
        evaluate_certificate(args.checkpoint, args.dataset, args.certificate, args.output)


if __name__ == "__main__":
    main()
