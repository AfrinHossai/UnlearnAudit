"""Offline SmolLM2 memory assessment, never a runtime admission authority.

No Torch import, model construction, training, download or telemetry collection.
The original physical readiness command continues to use the Llama SGD guard.
Evidence can produce a proposal only; adopting it requires a separate reviewed
code/policy change. An evidence JSON is an assertion, not authenticated proof.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.small_model_experiment import MODEL_ID, PROFILE, REVISION, ROOT, digest, validate_profile

GIB = 1024**3
POLICY = ROOT / "experiments/smollm2_135m_memory_preflight.json"
GUARD_ID = "smollm2_135m_fp32_sgd_uncalibrated_v1"
PROFILE_SHA256 = "260db810b3b1fbd4da0870725bcb7cac9a2ae4a472dbdb50c089007d35b711eb"
STAGES = ("baseline", "loaded", "forward_actual", "forward_512", "backward_512",
          "clipping", "serialization", "teardown")


def select_memory_guard(profile: Mapping[str, Any]) -> str:
    """Select by exact experiment identity/settings, never the Llama class name."""
    validate_profile(profile)
    if digest(profile) != PROFILE_SHA256:
        raise ValueError("memory assessment requires the exact frozen experiment profile")
    return GUARD_ID


def validate_policy(profile: Mapping[str, Any], policy: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": "1.0", "guard_id": select_memory_guard(profile),
        "status": "assessment_only_not_runtime_admission",
        "experiment_profile_sha256": PROFILE_SHA256,
        "active_runtime_guard": "unchanged_graddiff_real_train_sgd",
        "approved_before_load_available_bytes": None, "approved_measurement_sha256": None,
        "training_authorized": False, "minimum_total_host_gib": 48,
        "continuing_available_lower_bound_gib": 6, "minimum_system_memory_free_percentage": 20,
        "minimum_uncertainty_gib": 4, "uncertainty_fraction": 0.25,
        "minimum_trials": 3, "required_run_kinds": ["cold", "warm"],
        "required_stages": list(STAGES),
    }
    # Strict schema: unknown switches, booleans posing as numbers, or numeric
    # thresholds supplied by a caller cannot silently enable execution.
    if (set(policy) != set(expected)
            or any(type(policy[k]) is not type(v) or policy[k] != v for k, v in expected.items())):
        raise ValueError("unsupported small-model memory policy; runtime approval is disabled")


def memory_estimates(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Analytical tensor sizes and explicit speculative allowances, all unmeasured."""
    select_memory_guard(profile)
    p = profile["model"]["parameter_count"]
    b, s = profile["training"]["batch_size"], profile["training"]["max_sequence_length"]
    m = profile["model"]
    return {
        "measured": False, "units": "bytes; GiB = 2**30 bytes",
        "assumptions": "FP32, batch 1, length 512, SDPA, checkpointing, no KV cache, one model",
        "fp32_weights_bytes": 4 * p,
        "fp32_gradients_bytes_training_only": 4 * p,
        "sgd_zero_momentum_optimizer_state_bytes": 0,
        "adamw_moments_bytes_comparison_only": 8 * p,
        "inference_persistent_floor_bytes": 4 * p,
        "sgd_training_persistent_floor_bytes": 8 * p,
        "adamw_training_persistent_floor_bytes_comparison_only": 16 * p,
        "single_fp32_logits_tensor_bytes": 4 * b * s * m["vocab_size"],
        "single_hidden_tensor_bytes": 4 * b * s * m["hidden_size"],
        "one_hidden_checkpoint_per_layer_bytes_illustrative_only":
            4 * b * s * m["hidden_size"] * m["num_hidden_layers"],
        "inference_activation_workspace_allowance_bytes_unmeasured": [GIB // 2, 2 * GIB],
        "training_activation_workspace_allowance_bytes_unmeasured": [2 * GIB, 4 * GIB],
        "runtime_overhead_allowance_bytes_unmeasured": [GIB, 2 * GIB],
        "inference_working_plan_bytes_not_bound": [4 * p + 3 * GIB // 2, 4 * p + 4 * GIB],
        "sgd_training_working_plan_bytes_not_bound": [8 * p + 3 * GIB, 8 * p + 6 * GIB],
        "additional_host_os_other_process_reserve_bytes_unmeasured": 12 * GIB,
        "additional_planning_uncertainty_bytes_unmeasured": 4 * GIB,
        "loading_and_save_transients_measured": False,
        "training_readiness_proven": False,
    }


def _integer(values: Mapping[str, Any], key: str) -> int:
    value = values.get(key)
    if type(value) is not int or value < 0:
        raise ValueError(f"missing or invalid measurement: {key}")
    return value


def _snapshot(sample: Mapping[str, Any], total: int) -> None:
    keys = ("timestamp_ns", "available_physical_memory_lower_bound_bytes",
            "system_memory_free_percentage", "process_current_rss_bytes", "process_peak_rss_bytes",
            "mps_tensor_allocated_bytes", "mps_driver_allocated_bytes", "swap_used_bytes", "pageouts")
    for key in keys:
        _integer(sample, key)
    available = sample["available_physical_memory_lower_bound_bytes"]
    if (not 6 * GIB <= available <= total
            or not 20 <= sample["system_memory_free_percentage"] <= 100
            or not sample["process_current_rss_bytes"] <= sample["process_peak_rss_bytes"] <= total
            or not sample["mps_tensor_allocated_bytes"] <= sample["mps_driver_allocated_bytes"] <= total):
        raise ValueError("unsafe or inconsistent measured memory telemetry")


def measurement_proposal(profile: Mapping[str, Any], policy: Mapping[str, Any],
                         evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate future evidence and calculate an unapproved planning proposal.

    RSS/MPS/system availability overlap: use their maximum baseline-relative
    excursion, never their sum. Sampled peaks can miss transients; human review
    of traces and a separate implementation change remain necessary.
    """
    validate_policy(profile, policy)
    for key, expected in {"schema_version": "1.0", "profile_sha256": digest(profile),
                          "policy_sha256": digest(policy), "model_id": MODEL_ID,
                          "model_revision": REVISION, "device": "mps",
                          "fallback_enabled": False, "optimizer_updates_performed": 0}.items():
        if type(evidence.get(key)) is not type(expected) or evidence[key] != expected:
            raise ValueError(f"measurement provenance/configuration mismatch: {key}")
    for key in ("hardware_id", "os_version", "torch_version", "transformers_version"):
        if not isinstance(evidence.get(key), str) or not evidence[key].strip():
            raise ValueError(f"missing measurement environment: {key}")
    total = _integer(evidence, "total_physical_memory_bytes")
    if total < 48 * GIB:
        raise ValueError("measurement host must retain the current 48 GiB capacity protection")
    trials = evidence.get("trials")
    if not isinstance(trials, list) or len(trials) < policy["minimum_trials"]:
        raise ValueError("at least three measured cold/warm trials are required")
    worst = 0
    for trial in trials:
        if set(trial) != {"cold", "warm"}:
            raise ValueError("each measured trial requires cold and warm runs")
        for run in trial.values():
            _integer(run, "sampling_interval_ms")
            if not 0 < run["sampling_interval_ms"] <= 100:
                raise ValueError("measurement sampling interval must be 1..100 ms")
            samples = run.get("samples")
            if not isinstance(samples, list) or not samples or samples[0].get("stage") != "baseline":
                raise ValueError("missing baseline measurements")
            # Require ordered stage boundaries; extra samples within a stage
            # retain brief peaks instead of reducing the trace to one reading.
            stages = [sample.get("stage") for sample in samples]
            observed_stages = [stage for i, stage in enumerate(stages) if i == 0 or stage != stages[i - 1]]
            if observed_stages != list(STAGES):
                raise ValueError("missing or unordered measured stages")
            base = samples[0]
            previous_time = -1
            for sample in samples:
                _snapshot(sample, total)
                if sample["timestamp_ns"] <= previous_time:
                    raise ValueError("measurement timestamps must increase within each run")
                if (previous_time >= 0
                        and sample["timestamp_ns"] - previous_time > run["sampling_interval_ms"] * 1_000_000):
                    raise ValueError("measurement trace has a gap exceeding its sampling interval")
                previous_time = sample["timestamp_ns"]
                if sample["swap_used_bytes"] != base["swap_used_bytes"] or sample["pageouts"] != base["pageouts"]:
                    raise ValueError("swap/page-out changes invalidate calibration")
                worst = max(worst,
                            base["available_physical_memory_lower_bound_bytes"]
                            - sample["available_physical_memory_lower_bound_bytes"],
                            sample["process_peak_rss_bytes"] - base["process_current_rss_bytes"],
                            sample["mps_driver_allocated_bytes"] - base["mps_driver_allocated_bytes"])
            if base["available_physical_memory_lower_bound_bytes"] < 32 * GIB:
                raise ValueError("measurements must have been admitted under the unchanged 32 GiB guard")
    uncertainty = max(4 * GIB, math.ceil(worst * policy["uncertainty_fraction"]))
    proposal = math.ceil((worst + uncertainty + 6 * GIB) / GIB) * GIB
    return {"measurement_sha256": digest(evidence), "observed_incremental_bytes": worst,
            "uncertainty_bytes": uncertainty, "unapproved_before_load_proposal_bytes": proposal,
            "review_required": True, "sampled_peaks_are_not_upper_bounds": True}


def assess_memory(profile: Mapping[str, Any], policy: Mapping[str, Any],
                  measurements: Mapping[str, Any] | None = None) -> dict[str, Any]:
    validate_policy(profile, policy)
    proposal = measurement_proposal(profile, policy, measurements) if measurements is not None else None
    return {"status": "STOP_uncalibrated_small_model_memory_policy", "guard_id": select_memory_guard(profile),
            "active_runtime_guard": policy["active_runtime_guard"],
            "active_before_load_available_bytes": profile["safety"]["before_load_available_lower_bound_gib"] * GIB,
            "approved_small_model_threshold_bytes": None, "runtime_admission_allowed": False,
            "training_authorized": False, "model_loaded": False,
            "blockers": ["missing_measurements" if proposal is None else "evidence_requires_independent_review",
                         "no_approved_small_model_threshold", "no_small_model_training_entrypoint"],
            "estimates": memory_estimates(profile), "measurement_proposal": proposal}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, help="optional local evidence JSON; never grants admission")
    args = parser.parse_args(argv)
    try:
        profile, policy = json.loads(PROFILE.read_text()), json.loads(POLICY.read_text())
        evidence = json.loads(args.measurements.read_text()) if args.measurements else None
        if args.measurements and not isinstance(evidence, dict):
            raise ValueError("measurement file must contain an evidence object")
        print(json.dumps(assess_memory(profile, policy, evidence), indent=2))
        return 2  # Assessment can succeed; admission remains STOP even with evidence.
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        parser.exit(2, f"STOP: small-model memory assessment failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
