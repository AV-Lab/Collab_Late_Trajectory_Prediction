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
import json
import os
from pathlib import Path
import inspect
import tempfile

import numpy as np

if not __package__:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration.estimators.bootstrap import (BOOTSTRAP_DRAWS, CONFIDENCE, MIN_VALID_ROWS, SEED,
                                   estimate_bootstrap_profiles, estimate_motion_profiles)
from calibration.data.inference import (BATCH_SIZE, calibration_rows, collect_predictions,
                                   create_predictor, load_dataset, validate_predictor)
from calibration.metrics import evaluate_predictions, require_disjoint_groups, wire_covariances
from calibration.runtime import Certificate, file_identity as _file_identity
from calibration.estimators.motion import MOTION_DEFINITION, MOTION_TAGS, classify_motion
from calibration.estimators.motion_covariance import (DEFAULT_COV_BINS, ESTIMATOR,
                                           estimate_motion_covariance_profiles, validate_cov_bins)


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


def _settings(method, cov_bins=DEFAULT_COV_BINS):
    settings = {
        "bootstrap_draws": BOOTSTRAP_DRAWS, "seed": SEED,
        "min_fit_rows": MIN_VALID_ROWS, "confidence": CONFIDENCE,
        "grouping": ("object category x predicted motion x native horizon" if method == "motion_bootstrap"
                     else "object category x native horizon"),
        "uncertainty_bins": None,
        "resampling": "whole trajectories within each category; one draw shared across horizons",
        "moment": "uncentered raw error second moment",
        "radius": "95th percentile bootstrap spectral-norm deviation",
        "confidence_scope": "approximate 95% per group; not simultaneous",
    }
    if method == "bootstrap":
        settings.update(covariance_bins=1,
                        factor_conversion="generalized eigenvalue factors against the supplied covariance")
    if method == "motion_covariance":
        settings.update(
            grouping="object category x predicted motion x native horizon x fit-only covariance-trace bin",
            uncertainty_bins="fit-only covariance-trace quantiles; duplicate edges removed",
            covariance_bins=int(cov_bins),
            sparse_bin_fallback="parent category/motion/native-horizon risk; not a conditional-bin guarantee")
    return settings


def _assumptions(layout, method):
    return {
        "sample_independence_verified": False,
        "source_group_ids_available": True,
        "population": ("calibration target distribution within category, predicted motion and native horizon"
                       if method == "motion_bootstrap" else
                       "calibration target distribution within category, predicted motion, native horizon and trace bin"
                       if method == "motion_covariance" else
                       "calibration target distribution within category and native horizon"),
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


def _training_groups(predictor):
    """Read the checkpoint's prepared training view when it declares one."""
    specification = predictor.config.get("datasets", {}).get("train", {})
    if specification.get("name") != "prepared_dataset":
        return set(), None
    parameters = specification["params"]
    root = Path(parameters["root"])
    if not root.is_absolute():
        root = Path(inspect.getfile(type(predictor))).resolve().parent / root
    path = root / (parameters["split"] + ".dataset.json")
    view = json.loads(path.read_text())
    groups = {str(group) for source in view["storage"]["sources"] for group in source.get("included_groups", [])}
    if not groups:
        raise ValueError("Predictor training view lacks physical source groups.")
    return groups, _file_identity(path)


def create_certificate(checkpoint_path, dataset_path, output_path, method="bootstrap", *, cov_bins=None,
                       histories="clean", noise_model_path=None, device=None, batch_size=BATCH_SIZE):
    """Run inference and fit the chosen estimator on complete calibration targets."""
    if method not in ("bootstrap", "motion_bootstrap", "motion_covariance"):
        raise ValueError("Unknown certificate method: {}".format(method))
    if cov_bins is not None and method != "motion_covariance":
        raise ValueError("cov_bins is only valid for method=motion_covariance.")
    if histories not in ("clean", "gaussian") or (histories == "gaussian") != (noise_model_path is not None):
        raise ValueError("Gaussian histories require --noise-model; clean histories do not use it.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    cov_bins = DEFAULT_COV_BINS if cov_bins is None else cov_bins
    validate_cov_bins(cov_bins)
    _ensure_new_output(output_path)
    inputs = {"checkpoint": _file_identity(checkpoint_path), "dataset": _file_identity(dataset_path)}
    dataset = load_dataset(dataset_path)
    rows = calibration_rows(dataset)
    predictor = create_predictor(checkpoint_path) if device is None else create_predictor(checkpoint_path, device=device)
    contract = validate_predictor(predictor, dataset["metadata"])
    source_groups = {row["source_group_id"] for row in rows}
    trained, training_view = _training_groups(predictor)
    if source_groups & trained:
        raise ValueError("Certificate calibration overlaps predictor-training physical scenarios.")
    if training_view is not None:
        inputs["training_view"] = training_view
    noise_model, noise_groups = None, set()
    if histories == "gaussian":
        from calibration.data.history_noise import HistoryNoiseModel
        inputs["noise_model"] = _file_identity(noise_model_path)
        noise_model = HistoryNoiseModel.from_file(noise_model_path)
        metadata = noise_model.fit_metadata
        noise_groups = set(metadata.get("source_group_ids", []))
        if not noise_groups or metadata.get("evaluation_used_for_fitting") is not False:
            raise ValueError("Noise model needs calibration-only physical source groups; fit with calibration.data.history_errors.")
        if noise_groups & trained:
            raise ValueError("History-noise calibration overlaps predictor-training physical scenarios.")
    stats = {}
    means, covariances = collect_predictions(predictor, dataset, contract["layout"], batch_size,
        rows=rows, noise_model=noise_model, stats=stats)
    targets = np.stack([row["future"] for row in rows])
    categories = [row["category"] for row in rows]
    motion_tags = None
    if method in ("motion_bootstrap", "motion_covariance"):
        motion_tags = _motion_tags(rows, means, contract["obs_len"])
        estimator = (estimate_motion_covariance_profiles if method == "motion_covariance"
                     else estimate_motion_profiles)
        options = {"cov_bins": cov_bins} if method == "motion_covariance" else {}
        known = np.array([tag is not None for tag in motion_tags])
        if not known.any():
            raise ValueError("No calibration forecasts have a complete observed history for motion assignment.")
        profiles, exclusions = estimator(
            means[known], targets[known], covariances[known], np.asarray(categories)[known],
            np.asarray(motion_tags, dtype=object)[known], contract["sample_fps"], **options)
        usable = sum(profile["usable_for_gate"] for profile in profiles)
    else:
        profiles, exclusions = estimate_bootstrap_profiles(
            means, targets, covariances, categories, contract["sample_fps"])
        usable = sum(any(item["usable_for_gate"] for item in profile["bins"]) for profile in profiles)
    result = {
        "schema_version": {"bootstrap": 2, "motion_bootstrap": 3, "motion_covariance": 4}[method],
        "estimator": {"bootstrap": "bootstrap_raw_moment", "motion_bootstrap": "motion_bootstrap_raw_moment",
                      "motion_covariance": ESTIMATOR}[method],
        "method": method,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "bootstrap_group_moment_approximate" if usable else "no_usable_profiles",
        "usable_profile_count": usable,
        "checkpoint": {**inputs["checkpoint"], "model_name": predictor.config["model"]["name"]},
        "dataset": {**inputs["dataset"], "metadata": dataset["metadata"],
                    "n_samples": len(rows), "n_scenes": len(dataset["samples"])},
        "inference": {**contract, "seed": SEED, "batch_size": batch_size,
                      "device": str(predictor.device), "checkpoint_config": predictor.config},
        "settings": _settings(method, cov_bins), "assumptions": _assumptions(contract["layout"], method),
        "fit_provenance": {
            "purpose": "calibration_fit", "selection": "complete scoring targets in supplied archive",
            "n_input": len(rows), "n_fit": int(known.sum()) if motion_tags is not None else len(rows),
            "excluded_unavailable_motion": int((~known).sum()) if motion_tags is not None else 0,
            "source_group_ids": sorted(source_groups | noise_groups),
            "source_groups": "physical scenarios; all CAV views retain their shared source_group_id",
            "predictor_training_disjointness_verified": training_view is not None,
            "evaluation_used_for_fitting": False,
            "inputs": inputs, "history_kind": histories, "history_statistics": stats,
            "prediction_adapter": "COLTP runtime track/scene wrapper; original native timestamps",
        },
        "exclusions": exclusions, "profiles": profiles,
    }
    if noise_model is not None:
        result["fit_provenance"].update(noise_model=inputs["noise_model"], generation_seed=SEED,
            one_sample_per_original_window=True, parameter_estimation_uncertainty_in_bootstrap=False)
        result["assumptions"].update(input_scope="Gaussian-corrupted XY histories; clean scene membership and headings",
            history_length_sampling="measured category/length mixture; short contexts retain their available length",
            input_error_distribution="joint Gaussian with measured bias and full temporal covariance",
            noise_model_estimation_uncertainty_covered=False)
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
    for item in inputs.values():
        if _file_identity(item["path"]) != item:
            raise ValueError("Calibration input changed during certificate fitting.")
    _save_json(result, output_path)
    print("Certificate saved to: {}".format(Path(output_path).resolve()), flush=True)
    return result


def evaluate_certificate(checkpoint_path, dataset_path, certificate_path, output_path, *, device=None,
                         batch_size=BATCH_SIZE):
    """Evaluate a frozen certificate, rejecting physical fit/evaluation overlap."""
    _ensure_new_output(output_path)
    dataset = load_dataset(dataset_path)
    rows = calibration_rows(dataset)
    certificate = Certificate.from_file(certificate_path, checkpoint_path=checkpoint_path)
    groups = require_disjoint_groups(certificate, rows)
    predictor = create_predictor(checkpoint_path) if device is None else create_predictor(checkpoint_path, device=device)
    contract = validate_predictor(predictor, dataset["metadata"])
    certificate = Certificate.from_file(certificate_path, checkpoint_path=checkpoint_path, expected=contract)
    means, covariances = collect_predictions(predictor, dataset, contract["layout"], batch_size, rows=rows)
    targets = np.stack([row["future"] for row in rows])
    categories = [row["category"] for row in rows]
    motion_tags = (_motion_tags(rows, means, contract["obs_len"])
                   if certificate.needs_motion else None)
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
            command.add_argument("--method", choices=("bootstrap", "motion_bootstrap", "motion_covariance"),
                                 default="bootstrap")
            command.add_argument("--cov-bins", type=int,
                                 help="Requested trace-quantile bins for motion_covariance (default: 10).")
            command.add_argument("--histories", choices=("clean", "gaussian"), default="clean")
            command.add_argument("--noise-model", type=Path)
        command.add_argument("--checkpoint", required=True, type=Path)
        command.add_argument("--dataset", required=True, type=Path)
        command.add_argument("--output", required=True, type=Path)
        command.add_argument("--device", help="Predictor device, e.g. cpu or cuda:1.")
        command.add_argument("--batch-size", type=int, default=BATCH_SIZE)
        if name == "evaluate":
            command.add_argument("--certificate", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "fit":
        create_certificate(args.checkpoint, args.dataset, args.output, args.method, cov_bins=args.cov_bins,
                           histories=args.histories, noise_model_path=args.noise_model,
                           device=args.device, batch_size=args.batch_size)
    else:
        evaluate_certificate(args.checkpoint, args.dataset, args.certificate, args.output,
                             device=args.device, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
