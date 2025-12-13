"""
Artifact store for expensive gradient-analysis outputs.

Goal: decouple expensive compute (SVDs, echoes, etc.) from visualization iteration.

Layout:
  research_logs/analysis_artifacts/{run_id}/step_{step:06d}/rank{rank}.pt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple, Iterable
import torch

from empirical.research.analysis.constants import FIELD_NAMES
from empirical.research.analysis.model_utilities import GPTLayerProperty


def artifact_root(run_id: str) -> Path:
    return Path("research_logs/analysis_artifacts") / run_id


def artifact_step_dir(run_id: str, step: int) -> Path:
    return artifact_root(run_id) / f"step_{step:06d}"


def artifact_shard_path(run_id: str, step: int, rank: int) -> Path:
    return artifact_step_dir(run_id, step) / f"rank{rank}.pt"


def _to_cpu_compact(x: Any) -> Any:
    # Keep scalars as-is; move tensors to CPU and compact dtype where safe.
    if isinstance(x, torch.Tensor):
        t = x.detach().cpu()
        if t.is_floating_point():
            # Compact large arrays; preserve scalars/short vectors as fp32 if you want later.
            if t.numel() >= 1024:
                return t.to(torch.float16)
            return t.to(torch.float32)
        return t
    return x


def _filter_layer_props_for_artifact(props: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    # Store shape without storing full weights
    if "checkpoint_weights" in props and hasattr(props["checkpoint_weights"], "shape"):
        sh = tuple(int(x) for x in props["checkpoint_weights"].shape[-2:])
        out["shape"] = sh
    elif "shape" in props:
        out["shape"] = tuple(int(x) for x in props["shape"])

    # Keep only whitelisted fields (computed artifacts)
    for k in FIELD_NAMES:
        if k in props:
            out[k] = _to_cpu_compact(props[k])

    # Backward/compat convenience: some viz code expects this name.
    # In your pipeline replicate_singular_values is already aligned.
    if "replicate_singular_values" in out and "aligned_replicate_singular_values" not in out:
        out["aligned_replicate_singular_values"] = out["replicate_singular_values"]

    return out


def save_rank_artifact_shard(
    run_id: str,
    step: int,
    rank: int,
    layer_props: GPTLayerProperty,
) -> Path:
    """Save one rank's shard for one checkpoint step."""
    step_dir = artifact_step_dir(run_id, step)
    step_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_shard_path(run_id, step, rank)

    filtered: GPTLayerProperty = {}
    for key, props in layer_props.items():
        if isinstance(props, dict):
            filtered[key] = _filter_layer_props_for_artifact(props)

    payload = {
        "meta": {"run_id": run_id, "step": int(step), "rank": int(rank)},
        "layers": filtered,
    }
    torch.save(payload, path)
    return path


def list_artifact_steps(run_id: str) -> list[int]:
    root = artifact_root(run_id)
    if not root.exists():
        return []
    steps: list[int] = []
    for p in root.glob("step_*"):
        try:
            steps.append(int(p.name.split("_", 1)[1]))
        except Exception:
            continue
    return sorted(set(steps))


def load_step_artifacts(run_id: str, step: int) -> GPTLayerProperty:
    """Load and merge all rank shards for a given step."""
    d = artifact_step_dir(run_id, step)
    if not d.exists():
        raise FileNotFoundError(f"Artifact step dir not found: {d}")

    merged: GPTLayerProperty = {}
    for shard in sorted(d.glob("rank*.pt")):
        obj = torch.load(shard, map_location="cpu")
        layers = obj.get("layers", {})
        # Keys should be disjoint by construction (param sharding).
        for k, v in layers.items():
            merged[k] = v
    return merged
