"""Rift decode maps and pathway traces.

RiftMap is the first decode-native indexing layer above the current expert
packs. The four storage coordinates are:

    layer -> expert -> projection lane -> tile

RiftPathway adds the fifth coordinate, ``tau`` (decode step). It is a route
replay that labels each 4D tile access as hot or cold for a specific decode
step. This is a model-free contract: it does not execute MLX, but it tells the
runtime which contiguous byte spans a future Rift pack or page-backed QMM
primitive should be able to serve.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from smarttensor.packstore import ExpertPackReader, PackTensorRecord


_LAYER_EXPERT_PATTERNS = (
    re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\..*?\.experts\.(?P<lane>.+)$"),
    re.compile(r"(?:^|\.)layers\.(?P<layer>\d+)\..*?\.switch_mlp\.(?P<lane>.+)$"),
    re.compile(r"^backbone\.layers\.(?P<layer>\d+)\..*?\.switch_mlp\.(?P<lane>.+)$"),
)


@dataclass(frozen=True, order=True)
class RiftCoordinate:
    """Four-dimensional address of one decode tile."""

    layer: int
    expert: int
    projection: str
    tile: int

    def page_id(self) -> str:
        return f"L{self.layer}.E{self.expert}.{self.projection}.T{self.tile}"


@dataclass(frozen=True)
class RiftTile:
    """Physical byte span backing one 4D Rift coordinate."""

    coord: RiftCoordinate
    pack_path: str
    source: str | None
    tensor_name: str
    dtype: str
    shape: tuple[int, ...]
    file_offset: int
    nbytes: int
    expert_nbytes: int
    expert_stride: int
    tile_offset: int

    @property
    def file_end(self) -> int:
        return self.file_offset + self.nbytes


@dataclass(frozen=True)
class RiftSpan:
    """A contiguous physical read span after coalescing tiles."""

    pack_path: str
    offset: int
    nbytes: int
    coordinates: tuple[RiftCoordinate, ...]

    @property
    def end(self) -> int:
        return self.offset + self.nbytes


@dataclass(frozen=True)
class RiftRouteStep:
    """Selected experts for one routed layer at one decode step."""

    step_index: int
    layer: int
    experts: tuple[int, ...]
    hot_experts: tuple[int, ...] = ()


@dataclass(frozen=True, order=True)
class RiftExpertNode:
    """Binary on/off expert activation node inside the Rift manifold."""

    layer: int
    expert: int

    def id(self) -> str:
        return f"L{self.layer}.E{self.expert}"


@dataclass(frozen=True)
class RiftEntanglementEdge:
    """Causal correlation edge between two expert activations."""

    source: RiftExpertNode
    target: RiftExpertNode
    support: int
    source_count: int
    target_count: int
    confidence: float
    lift: float
    layer_delta: int


@dataclass(frozen=True)
class RiftManifoldAttentionScore:
    """Runtime attention score over a remote manifold node."""

    node: RiftExpertNode
    score: float
    confidence: float
    lift: float
    layer_delta: int
    resident: bool
    read_bytes: int


@dataclass(frozen=True)
class RiftPathwayAccess:
    """Five-dimensional pathway access: tau + 4D tile coordinate."""

    step_index: int
    coord: RiftCoordinate
    role: str
    nbytes: int

    @property
    def is_hot(self) -> bool:
        return self.role == "hot"


@dataclass(frozen=True)
class RiftPathwaySummary:
    steps: int
    accesses: int
    hot_accesses: int
    cold_accesses: int
    logical_bytes: int
    hot_bytes: int
    cold_bytes: int
    coalesced_spans: int
    cold_coalesced_spans: int

    @property
    def hot_byte_rate(self) -> float:
        if self.logical_bytes <= 0:
            return 0.0
        return self.hot_bytes / self.logical_bytes


@dataclass(frozen=True)
class RiftPathway:
    accesses: tuple[RiftPathwayAccess, ...]
    summary: RiftPathwaySummary

    def cold_coordinates(self) -> tuple[RiftCoordinate, ...]:
        return tuple(access.coord for access in self.accesses if not access.is_hot)


@dataclass(frozen=True)
class RiftRequestPacket:
    """A coalesced physical request against the Rift manifold."""

    request_id: str
    role: str
    tau_range: tuple[int, int] | None
    coordinates: tuple[RiftCoordinate, ...]
    spans: tuple[RiftSpan, ...]
    logical_bytes: int

    @property
    def read_bytes(self) -> int:
        return sum(span.nbytes for span in self.spans)

    @property
    def span_count(self) -> int:
        return len(self.spans)


@dataclass(frozen=True)
class RiftWormholePlan:
    """Predicted remote pathway folded into one manifold request packet."""

    observed: tuple[RiftExpertNode, ...]
    predicted_by_layer: dict[int, tuple[int, ...]]
    pathway: RiftPathway
    packet: RiftRequestPacket


@dataclass(frozen=True)
class RiftRoutePriorEntry:
    """One compiled route prediction for a same-layer route signature."""

    layer: int
    previous_route: tuple[int, ...]
    experts: tuple[int, ...]
    examples: int
    fed_token: int | None = None

    def to_qwen_dict(self, *, mode: str) -> dict[str, Any]:
        item: dict[str, Any] = {
            "layer": int(self.layer),
            "previous_route": list(self.previous_route),
            "experts": list(self.experts),
            "examples": int(self.examples),
        }
        if mode == "rolling-recent":
            item["fed_token"] = int(self.fed_token) if self.fed_token is not None else None
        return item


@dataclass(frozen=True)
class RiftRoutePrior:
    """Compiled route prior that can be loaded by the Qwen Rift runtime."""

    mode: str
    entries: tuple[RiftRoutePriorEntry, ...]
    profile: dict[str, Any]

    def to_qwen_dict(self) -> dict[str, Any]:
        return {
            "format": "qwen-rift-map-v1",
            "product": "Rift",
            "model_type": "qwen3_5_moe",
            "mode": self.mode,
            "signature": (
                "previous_same_layer_route + layer"
                if self.mode == "recent-route"
                else "fed_token + previous_same_layer_route + layer"
            ),
            "profile": self.profile,
            "routes": [
                entry.to_qwen_dict(mode=self.mode)
                for entry in self.entries
            ],
        }


@dataclass(frozen=True)
class RiftRouteEvaluation:
    """Held-out route-prior quality metrics."""

    observed_layers: int
    predicted_layers: int
    full_hits: int
    true_rows: int
    predicted_true_rows: int
    predicted_rows: int
    hit_rows: int
    missing_rows: int
    wasted_rows: int
    skipped_no_state: int
    skipped_unseen_signature: int
    skipped_low_examples: int

    @property
    def prediction_rate(self) -> float:
        return self.predicted_layers / self.observed_layers if self.observed_layers else 0.0

    @property
    def full_hit_rate(self) -> float:
        return self.full_hits / self.predicted_layers if self.predicted_layers else 0.0

    @property
    def expert_recall(self) -> float:
        return self.hit_rows / self.predicted_true_rows if self.predicted_true_rows else 0.0

    @property
    def overall_expert_recall(self) -> float:
        return self.hit_rows / self.true_rows if self.true_rows else 0.0

    @property
    def precision(self) -> float:
        return self.hit_rows / self.predicted_rows if self.predicted_rows else 0.0

    @property
    def fallback_rate(self) -> float:
        return self.missing_rows / self.predicted_true_rows if self.predicted_true_rows else 0.0

    @property
    def waste_rate(self) -> float:
        return self.wasted_rows / self.predicted_rows if self.predicted_rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_layers": self.observed_layers,
            "predicted_layers": self.predicted_layers,
            "prediction_rate": self.prediction_rate,
            "full_hits": self.full_hits,
            "full_hit_rate": self.full_hit_rate,
            "true_rows": self.true_rows,
            "predicted_true_rows": self.predicted_true_rows,
            "predicted_rows": self.predicted_rows,
            "hit_rows": self.hit_rows,
            "missing_rows": self.missing_rows,
            "wasted_rows": self.wasted_rows,
            "expert_recall": self.expert_recall,
            "overall_expert_recall": self.overall_expert_recall,
            "precision": self.precision,
            "fallback_rate": self.fallback_rate,
            "waste_rate": self.waste_rate,
            "skipped_no_state": self.skipped_no_state,
            "skipped_unseen_signature": self.skipped_unseen_signature,
            "skipped_low_examples": self.skipped_low_examples,
        }


@dataclass(frozen=True)
class RiftRouteBlockEvaluation:
    """Block-union quality for route priors feeding no-host preload plans.

    Per-step route prediction can look good while still being unsafe for the
    block verifier: every actual expert selected anywhere inside the block must
    be resident before no-host routing can be used. Extra predicted experts only
    waste bytes; missing experts force a fault or must fail closed.
    """

    block_count: int
    block_layer_count: int
    predicted_block_layers: int
    full_layer_hits: int
    full_block_hits: int
    actual_rows: int
    predicted_rows: int
    hit_rows: int
    missing_rows: int
    wasted_rows: int
    skipped_no_state: int
    skipped_unseen_signature: int
    skipped_low_examples: int

    @property
    def full_layer_hit_rate(self) -> float:
        return self.full_layer_hits / self.block_layer_count if self.block_layer_count else 0.0

    @property
    def full_block_hit_rate(self) -> float:
        return self.full_block_hits / self.block_count if self.block_count else 0.0

    @property
    def expert_recall(self) -> float:
        return self.hit_rows / self.actual_rows if self.actual_rows else 0.0

    @property
    def precision(self) -> float:
        return self.hit_rows / self.predicted_rows if self.predicted_rows else 0.0

    @property
    def waste_rate(self) -> float:
        return self.wasted_rows / self.predicted_rows if self.predicted_rows else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_count": self.block_count,
            "block_layer_count": self.block_layer_count,
            "predicted_block_layers": self.predicted_block_layers,
            "full_layer_hits": self.full_layer_hits,
            "full_layer_hit_rate": self.full_layer_hit_rate,
            "full_block_hits": self.full_block_hits,
            "full_block_hit_rate": self.full_block_hit_rate,
            "actual_rows": self.actual_rows,
            "predicted_rows": self.predicted_rows,
            "hit_rows": self.hit_rows,
            "missing_rows": self.missing_rows,
            "wasted_rows": self.wasted_rows,
            "expert_recall": self.expert_recall,
            "precision": self.precision,
            "waste_rate": self.waste_rate,
            "skipped_no_state": self.skipped_no_state,
            "skipped_unseen_signature": self.skipped_unseen_signature,
            "skipped_low_examples": self.skipped_low_examples,
        }


@dataclass(frozen=True)
class RiftMap:
    """A 4D logical map over one or more current expert packs."""

    tiles: tuple[RiftTile, ...]
    tile_bytes: int

    def __post_init__(self) -> None:
        if self.tile_bytes <= 0:
            raise ValueError("tile_bytes must be positive")

    @property
    def tile_count(self) -> int:
        return len(self.tiles)

    @property
    def logical_bytes(self) -> int:
        return sum(tile.nbytes for tile in self.tiles)

    @property
    def layers(self) -> tuple[int, ...]:
        return tuple(sorted({tile.coord.layer for tile in self.tiles}))

    def tiles_for(
        self,
        *,
        layer: int,
        experts: Iterable[int],
        projections: Iterable[str] | None = None,
    ) -> tuple[RiftTile, ...]:
        expert_set = {int(expert) for expert in experts}
        projection_set = None if projections is None else {str(projection) for projection in projections}
        return tuple(
            tile
            for tile in self.tiles
            if tile.coord.layer == int(layer)
            and tile.coord.expert in expert_set
            and (projection_set is None or tile.coord.projection in projection_set)
        )

    def coalesced_spans(self, tiles: Iterable[RiftTile] | None = None) -> tuple[RiftSpan, ...]:
        selected = list(self.tiles if tiles is None else tiles)
        selected.sort(key=lambda tile: (tile.pack_path, tile.file_offset, tile.file_end, tile.coord))
        spans: list[RiftSpan] = []
        for tile in selected:
            if (
                spans
                and spans[-1].pack_path == tile.pack_path
                and spans[-1].end == tile.file_offset
            ):
                previous = spans[-1]
                spans[-1] = RiftSpan(
                    pack_path=previous.pack_path,
                    offset=previous.offset,
                    nbytes=previous.nbytes + tile.nbytes,
                    coordinates=previous.coordinates + (tile.coord,),
                )
                continue
            spans.append(
                RiftSpan(
                    pack_path=tile.pack_path,
                    offset=tile.file_offset,
                    nbytes=tile.nbytes,
                    coordinates=(tile.coord,),
                )
            )
        return tuple(spans)

    def request_packet(
        self,
        pathway: RiftPathway,
        *,
        role: str = "cold",
        request_id: str = "rift-request",
    ) -> RiftRequestPacket:
        """Collapse a 5D pathway into one physical manifold request packet.

        ``role='cold'`` is the normal exact pager packet. ``role='hot'`` lets a
        runtime prewarm a resident pathway. ``role='all'`` is useful for a
        predicted/frozen geodesic where the request wants every tile in the
        planned pathway up front.
        """

        if role not in {"cold", "hot", "all"}:
            raise ValueError("role must be 'cold', 'hot', or 'all'")
        if role == "cold":
            accesses = [access for access in pathway.accesses if not access.is_hot]
        elif role == "hot":
            accesses = [access for access in pathway.accesses if access.is_hot]
        else:
            accesses = list(pathway.accesses)

        coordinates = tuple(dict.fromkeys(access.coord for access in accesses))
        coord_set = set(coordinates)
        tiles = [tile for tile in self.tiles if tile.coord in coord_set]
        spans = self.coalesced_spans(tiles)
        steps = [access.step_index for access in accesses]
        return RiftRequestPacket(
            request_id=request_id,
            role=role,
            tau_range=(min(steps), max(steps)) if steps else None,
            coordinates=coordinates,
            spans=spans,
            logical_bytes=sum(tile.nbytes for tile in tiles),
        )

    def wormhole_plan(
        self,
        *,
        tau: int,
        observed: Iterable[RiftExpertNode],
        edges: Sequence[RiftEntanglementEdge],
        projections: Iterable[str] | None = None,
        hotset_by_layer: Mapping[int, Iterable[int]] | None = None,
        min_confidence: float = 0.5,
        max_predictions: int | None = None,
        request_id: str = "rift-wormhole",
    ) -> RiftWormholePlan:
        """Fold predicted remote expert weights into one request packet."""

        observed_nodes = tuple(dict.fromkeys(observed))
        predicted = predict_entangled_experts(
            observed_nodes,
            edges,
            min_confidence=min_confidence,
            max_predictions=max_predictions,
        )
        steps = tuple(
            RiftRouteStep(step_index=int(tau), layer=layer, experts=experts)
            for layer, experts in predicted.items()
        )
        pathway = self.pathway(
            steps,
            hotset_by_layer=hotset_by_layer,
            projections=projections,
        )
        packet = self.request_packet(
            pathway,
            role="all",
            request_id=request_id,
        )
        return RiftWormholePlan(
            observed=observed_nodes,
            predicted_by_layer=predicted,
            pathway=pathway,
            packet=packet,
        )

    def pathway(
        self,
        steps: Sequence[RiftRouteStep],
        *,
        hotset_by_layer: Mapping[int, Iterable[int]] | None = None,
        projections: Iterable[str] | None = None,
    ) -> RiftPathway:
        hotsets = {
            int(layer): {int(expert) for expert in experts}
            for layer, experts in (hotset_by_layer or {}).items()
        }
        accesses: list[RiftPathwayAccess] = []
        for step in steps:
            step_hot = set(hotsets.get(int(step.layer), set())) | {
                int(expert) for expert in step.hot_experts
            }
            for tile in self.tiles_for(
                layer=step.layer,
                experts=step.experts,
                projections=projections,
            ):
                role = "hot" if tile.coord.expert in step_hot else "cold"
                accesses.append(
                    RiftPathwayAccess(
                        step_index=int(step.step_index),
                        coord=tile.coord,
                        role=role,
                        nbytes=tile.nbytes,
                    )
                )

        cold_lookup = {access.coord for access in accesses if not access.is_hot}
        cold_tiles = [tile for tile in self.tiles if tile.coord in cold_lookup]
        cold_spans = self.coalesced_spans(cold_tiles)
        all_lookup = {access.coord for access in accesses}
        all_tiles = [tile for tile in self.tiles if tile.coord in all_lookup]
        all_spans = self.coalesced_spans(all_tiles)
        hot_bytes = sum(access.nbytes for access in accesses if access.is_hot)
        cold_bytes = sum(access.nbytes for access in accesses if not access.is_hot)
        summary = RiftPathwaySummary(
            steps=len({step.step_index for step in steps}),
            accesses=len(accesses),
            hot_accesses=sum(1 for access in accesses if access.is_hot),
            cold_accesses=sum(1 for access in accesses if not access.is_hot),
            logical_bytes=hot_bytes + cold_bytes,
            hot_bytes=hot_bytes,
            cold_bytes=cold_bytes,
            coalesced_spans=len(all_spans),
            cold_coalesced_spans=len(cold_spans),
        )
        return RiftPathway(accesses=tuple(accesses), summary=summary)


def parse_expert_tensor_name(name: str) -> tuple[int, str] | None:
    """Return ``(layer, projection_lane)`` for supported expert tensor names."""

    for pattern in _LAYER_EXPERT_PATTERNS:
        match = pattern.search(name)
        if match is not None:
            return int(match.group("layer")), match.group("lane")
    return None


def extract_qwen_route_steps_from_result(
    payload: Mapping[str, Any],
    *,
    pass_kind: str = "decode",
) -> tuple[RiftRouteStep, ...]:
    """Extract Qwen route steps from a SmartTensor benchmark JSON payload."""

    events = payload.get("events")
    if not isinstance(events, list):
        return ()
    steps: list[RiftRouteStep] = []
    for event in events:
        if not isinstance(event, Mapping):
            continue
        if event.get("pass") != pass_kind:
            continue
        token_step = event.get("token_step")
        layer = event.get("layer")
        experts = event.get("experts")
        if token_step is None or layer is None or not isinstance(experts, list):
            continue
        steps.append(
            RiftRouteStep(
                step_index=int(token_step),
                layer=int(layer),
                experts=tuple(sorted(int(expert) for expert in experts)),
            )
        )
    return tuple(sorted(steps, key=lambda step: (step.step_index, step.layer)))


def generated_tokens_from_result(payload: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the single-request generated token stream from a benchmark payload."""

    tokens = payload.get("generated_tokens")
    if not isinstance(tokens, list):
        return ()
    if tokens and all(isinstance(token, int) for token in tokens):
        return tuple(int(token) for token in tokens)
    if len(tokens) == 1 and isinstance(tokens[0], list):
        return tuple(int(token) for token in tokens[0])
    return ()


def load_qwen_route_trace(path: str | Path) -> tuple[tuple[RiftRouteStep, ...], tuple[int, ...]]:
    """Load route steps and generated tokens from one benchmark JSON file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return extract_qwen_route_steps_from_result(payload), generated_tokens_from_result(payload)


def compile_qwen_route_prior(
    traces: Sequence[tuple[Sequence[RiftRouteStep], Sequence[int]]],
    *,
    mode: str = "rolling-recent",
) -> RiftRoutePrior:
    """Compile route traces into the runtime-compatible ``qwen-rift-map-v1`` prior."""

    if mode not in {"rolling-recent", "recent-route"}:
        raise ValueError("mode must be 'rolling-recent' or 'recent-route'")
    route_counts: dict[
        tuple[int, int, tuple[int, ...]],
        Counter[tuple[int, ...]],
    ] = {}
    observations = 0
    skipped_no_state = 0
    skipped_no_token = 0
    for steps, generated_tokens in traces:
        previous_by_layer: dict[int, tuple[int, ...]] = {}
        token_tuple = tuple(int(token) for token in generated_tokens)
        for step in sorted(steps, key=lambda item: (item.step_index, item.layer)):
            previous = previous_by_layer.get(step.layer)
            actual = tuple(sorted(int(expert) for expert in step.experts))
            if previous is None:
                skipped_no_state += 1
            else:
                fed_token = (
                    int(token_tuple[step.step_index])
                    if 0 <= step.step_index < len(token_tuple)
                    else None
                )
                if mode == "rolling-recent" and fed_token is None:
                    skipped_no_token += 1
                else:
                    signature_token = fed_token if mode == "rolling-recent" else -1
                    key = (int(step.layer), int(signature_token), previous)
                    route_counts.setdefault(key, Counter()).update([actual])
                    observations += 1
            previous_by_layer[int(step.layer)] = actual

    entries: list[RiftRoutePriorEntry] = []
    ambiguous = 0
    examples_total = 0
    for (layer, signature_token, previous), counts in sorted(
        route_counts.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        if len(counts) > 1:
            ambiguous += 1
        experts, examples = max(counts.items(), key=lambda item: (item[1], item[0]))
        examples_total += int(examples)
        entries.append(
            RiftRoutePriorEntry(
                layer=int(layer),
                fed_token=int(signature_token) if mode == "rolling-recent" else None,
                previous_route=tuple(int(expert) for expert in previous),
                experts=tuple(int(expert) for expert in experts),
                examples=int(examples),
            )
        )

    profile = {
        "trace_count": len(traces),
        "mode": mode,
        "observations": observations,
        "routes": len(entries),
        "ambiguous_signatures": ambiguous,
        "avg_examples_per_route": examples_total / len(entries) if entries else 0.0,
        "skipped_no_state": skipped_no_state,
        "skipped_no_token": skipped_no_token,
    }
    return RiftRoutePrior(mode=mode, entries=tuple(entries), profile=profile)


def evaluate_qwen_route_prior(
    prior: RiftRoutePrior,
    traces: Sequence[tuple[Sequence[RiftRouteStep], Sequence[int]]],
    *,
    min_examples: int = 1,
) -> RiftRouteEvaluation:
    """Evaluate a compiled route prior without online adaptation."""

    if min_examples < 1:
        raise ValueError("min_examples must be positive")
    lookup = {
        (
            entry.layer,
            int(entry.fed_token) if prior.mode == "rolling-recent" else -1,
            entry.previous_route,
        ): (entry.experts, entry.examples)
        for entry in prior.entries
    }
    observed_layers = 0
    predicted_layers = 0
    full_hits = 0
    true_rows = 0
    predicted_true_rows = 0
    predicted_rows = 0
    hit_rows = 0
    missing_rows = 0
    wasted_rows = 0
    skipped_no_state = 0
    skipped_unseen_signature = 0
    skipped_low_examples = 0

    for steps, generated_tokens in traces:
        previous_by_layer: dict[int, tuple[int, ...]] = {}
        token_tuple = tuple(int(token) for token in generated_tokens)
        for step in sorted(steps, key=lambda item: (item.step_index, item.layer)):
            actual = tuple(sorted(int(expert) for expert in step.experts))
            previous = previous_by_layer.get(step.layer)
            observed_layers += 1
            true_rows += len(actual)
            if previous is None:
                skipped_no_state += 1
            else:
                fed_token = (
                    int(token_tuple[step.step_index])
                    if 0 <= step.step_index < len(token_tuple)
                    else None
                )
                signature_token = fed_token if prior.mode == "rolling-recent" else -1
                if prior.mode == "rolling-recent" and signature_token is None:
                    skipped_unseen_signature += 1
                else:
                    prediction = lookup.get((int(step.layer), int(signature_token), previous))
                    if prediction is None:
                        skipped_unseen_signature += 1
                    else:
                        predicted, examples = prediction
                        if examples < min_examples:
                            skipped_low_examples += 1
                        else:
                            actual_set = set(actual)
                            predicted_set = set(predicted)
                            hits = actual_set & predicted_set
                            missing = actual_set - predicted_set
                            wasted = predicted_set - actual_set
                            predicted_layers += 1
                            predicted_true_rows += len(actual_set)
                            predicted_rows += len(predicted_set)
                            hit_rows += len(hits)
                            missing_rows += len(missing)
                            wasted_rows += len(wasted)
                            if predicted_set == actual_set:
                                full_hits += 1
            previous_by_layer[int(step.layer)] = actual

    return RiftRouteEvaluation(
        observed_layers=observed_layers,
        predicted_layers=predicted_layers,
        full_hits=full_hits,
        true_rows=true_rows,
        predicted_true_rows=predicted_true_rows,
        predicted_rows=predicted_rows,
        hit_rows=hit_rows,
        missing_rows=missing_rows,
        wasted_rows=wasted_rows,
        skipped_no_state=skipped_no_state,
        skipped_unseen_signature=skipped_unseen_signature,
        skipped_low_examples=skipped_low_examples,
    )


def evaluate_qwen_route_prior_blocks(
    prior: RiftRoutePrior,
    traces: Sequence[tuple[Sequence[RiftRouteStep], Sequence[int]]],
    *,
    block_size: int,
    min_examples: int = 1,
) -> RiftRouteBlockEvaluation:
    """Evaluate whether a route prior can cover block-union preload plans.

    This is stricter than :func:`evaluate_qwen_route_prior`: a no-host route
    block is safe only when the predicted expert union covers every selected
    expert in every layer in the block. Missing rows are therefore fatal even
    when most individual per-step predictions were correct.
    """

    if block_size < 1:
        raise ValueError("block_size must be positive")
    if min_examples < 1:
        raise ValueError("min_examples must be positive")

    lookup = {
        (
            entry.layer,
            int(entry.fed_token) if prior.mode == "rolling-recent" else -1,
            entry.previous_route,
        ): (entry.experts, entry.examples)
        for entry in prior.entries
    }
    actual_by_block_layer: dict[
        tuple[int, int, int],
        set[int],
    ] = {}
    predicted_by_block_layer: dict[
        tuple[int, int, int],
        set[int],
    ] = {}
    skipped_no_state = 0
    skipped_unseen_signature = 0
    skipped_low_examples = 0

    for trace_index, (steps, generated_tokens) in enumerate(traces):
        previous_by_layer: dict[int, tuple[int, ...]] = {}
        token_tuple = tuple(int(token) for token in generated_tokens)
        for step in sorted(steps, key=lambda item: (item.step_index, item.layer)):
            actual = tuple(sorted(int(expert) for expert in step.experts))
            block_start = (int(step.step_index) // int(block_size)) * int(block_size)
            key = (int(trace_index), block_start, int(step.layer))
            actual_by_block_layer.setdefault(key, set()).update(actual)

            previous = previous_by_layer.get(step.layer)
            if previous is None:
                skipped_no_state += 1
            else:
                fed_token = (
                    int(token_tuple[step.step_index])
                    if 0 <= step.step_index < len(token_tuple)
                    else None
                )
                signature_token = fed_token if prior.mode == "rolling-recent" else -1
                if prior.mode == "rolling-recent" and signature_token is None:
                    skipped_unseen_signature += 1
                else:
                    prediction = lookup.get((int(step.layer), int(signature_token), previous))
                    if prediction is None:
                        skipped_unseen_signature += 1
                    else:
                        predicted, examples = prediction
                        if examples < min_examples:
                            skipped_low_examples += 1
                        else:
                            predicted_by_block_layer.setdefault(key, set()).update(
                                int(expert) for expert in predicted
                            )
            previous_by_layer[int(step.layer)] = actual

    block_keys = {
        (trace_index, block_start)
        for trace_index, block_start, _layer in actual_by_block_layer
    }
    full_block_hits = 0
    block_layer_count = 0
    predicted_block_layers = 0
    full_layer_hits = 0
    actual_rows = 0
    predicted_rows = 0
    hit_rows = 0
    missing_rows = 0
    wasted_rows = 0

    for block_key in sorted(block_keys):
        block_full_hit = True
        layer_keys = sorted(
            key
            for key in actual_by_block_layer
            if key[0] == block_key[0] and key[1] == block_key[1]
        )
        for key in layer_keys:
            actual_set = actual_by_block_layer[key]
            predicted_set = predicted_by_block_layer.get(key, set())
            hits = actual_set & predicted_set
            missing = actual_set - predicted_set
            wasted = predicted_set - actual_set
            block_layer_count += 1
            actual_rows += len(actual_set)
            predicted_rows += len(predicted_set)
            hit_rows += len(hits)
            missing_rows += len(missing)
            wasted_rows += len(wasted)
            if predicted_set:
                predicted_block_layers += 1
            if missing:
                block_full_hit = False
            else:
                full_layer_hits += 1
        if block_full_hit and layer_keys:
            full_block_hits += 1

    return RiftRouteBlockEvaluation(
        block_count=len(block_keys),
        block_layer_count=block_layer_count,
        predicted_block_layers=predicted_block_layers,
        full_layer_hits=full_layer_hits,
        full_block_hits=full_block_hits,
        actual_rows=actual_rows,
        predicted_rows=predicted_rows,
        hit_rows=hit_rows,
        missing_rows=missing_rows,
        wasted_rows=wasted_rows,
        skipped_no_state=skipped_no_state,
        skipped_unseen_signature=skipped_unseen_signature,
        skipped_low_examples=skipped_low_examples,
    )


def learn_rift_entanglement(
    steps: Sequence[RiftRouteStep],
    *,
    min_support: int = 2,
    min_confidence: float = 0.5,
    max_layer_delta: int | None = None,
) -> tuple[RiftEntanglementEdge, ...]:
    """Learn causal expert co-activation edges from route steps.

    This is the classical version of the user's "particle on/off" idea:
    experts are binary activations, and a source expert can predict a distant
    future-layer expert inside the same decode step. Edges are causal only:
    source.layer must be lower than target.layer.
    """

    if min_support < 1:
        raise ValueError("min_support must be positive")
    if min_confidence < 0:
        raise ValueError("min_confidence must be non-negative")
    by_tau: dict[int, dict[int, set[int]]] = {}
    for step in steps:
        layer_bucket = by_tau.setdefault(int(step.step_index), {})
        layer_bucket.setdefault(int(step.layer), set()).update(int(expert) for expert in step.experts)

    source_counts: dict[RiftExpertNode, int] = {}
    target_counts: dict[RiftExpertNode, int] = {}
    pair_counts: dict[tuple[RiftExpertNode, RiftExpertNode], int] = {}
    tau_count = len(by_tau)

    for layers in by_tau.values():
        activations = [
            (int(layer), int(expert))
            for layer, experts in layers.items()
            for expert in experts
        ]
        for layer, expert in activations:
            node = RiftExpertNode(layer, expert)
            source_counts[node] = source_counts.get(node, 0) + 1
            target_counts[node] = target_counts.get(node, 0) + 1
        for source_layer, source_expert in activations:
            source = RiftExpertNode(source_layer, source_expert)
            for target_layer, target_expert in activations:
                layer_delta = target_layer - source_layer
                if layer_delta <= 0:
                    continue
                if max_layer_delta is not None and layer_delta > max_layer_delta:
                    continue
                target = RiftExpertNode(target_layer, target_expert)
                pair_counts[(source, target)] = pair_counts.get((source, target), 0) + 1

    edges: list[RiftEntanglementEdge] = []
    for (source, target), support in pair_counts.items():
        source_count = source_counts[source]
        target_count = target_counts[target]
        confidence = support / source_count if source_count else 0.0
        target_rate = target_count / tau_count if tau_count else 0.0
        lift = confidence / target_rate if target_rate else 0.0
        if support < min_support or confidence < min_confidence:
            continue
        edges.append(
            RiftEntanglementEdge(
                source=source,
                target=target,
                support=support,
                source_count=source_count,
                target_count=target_count,
                confidence=confidence,
                lift=lift,
                layer_delta=target.layer - source.layer,
            )
        )

    return tuple(
        sorted(
            edges,
            key=lambda edge: (
                -edge.confidence,
                -edge.support,
                -edge.lift,
                edge.source.layer,
                edge.source.expert,
                edge.target.layer,
                edge.target.expert,
            ),
        )
    )


def predict_entangled_experts(
    observed: Iterable[RiftExpertNode],
    edges: Sequence[RiftEntanglementEdge],
    *,
    min_confidence: float = 0.5,
    max_predictions: int | None = None,
) -> dict[int, tuple[int, ...]]:
    """Predict remote experts from currently observed activation nodes."""

    observed_set = set(observed)
    scored: dict[RiftExpertNode, float] = {}
    for edge in edges:
        if edge.source not in observed_set:
            continue
        if edge.confidence < min_confidence:
            continue
        previous = scored.get(edge.target, 0.0)
        scored[edge.target] = max(previous, edge.confidence * max(edge.lift, 1.0))

    ranked = sorted(scored, key=lambda node: (-scored[node], node.layer, node.expert))
    if max_predictions is not None:
        ranked = ranked[: max(0, int(max_predictions))]
    by_layer: dict[int, list[int]] = {}
    for node in ranked:
        by_layer.setdefault(node.layer, []).append(node.expert)
    return {
        layer: tuple(sorted(experts))
        for layer, experts in sorted(by_layer.items())
    }


def score_rift_manifold_attention(
    rift_map: RiftMap,
    *,
    observed: Iterable[RiftExpertNode],
    edges: Sequence[RiftEntanglementEdge],
    projections: Iterable[str] | None = None,
    hotset_by_layer: Mapping[int, Iterable[int]] | None = None,
    min_confidence: float = 0.5,
    layer_decay: float = 0.02,
    read_cost_per_gib: float = 0.0,
    max_nodes: int | None = None,
) -> tuple[RiftManifoldAttentionScore, ...]:
    """Score remote experts as a multidimensional runtime attention problem.

    The score is intentionally simple and deterministic:

    ``confidence * max(lift, 1)`` attends to statistically entangled nodes,
    layer decay discounts very distant jumps, and read cost can penalize large
    physical payloads. A future learned scorer can replace this without
    changing the RiftMap/Pathway contract.
    """

    if layer_decay < 0:
        raise ValueError("layer_decay must be non-negative")
    if read_cost_per_gib < 0:
        raise ValueError("read_cost_per_gib must be non-negative")
    observed_set = set(observed)
    hotsets = {
        int(layer): {int(expert) for expert in experts}
        for layer, experts in (hotset_by_layer or {}).items()
    }
    best_by_node: dict[RiftExpertNode, RiftManifoldAttentionScore] = {}
    for edge in edges:
        if edge.source not in observed_set or edge.confidence < min_confidence:
            continue
        tiles = rift_map.tiles_for(
            layer=edge.target.layer,
            experts=[edge.target.expert],
            projections=projections,
        )
        if not tiles:
            continue
        read_bytes = sum(tile.nbytes for tile in tiles)
        resident = edge.target.expert in hotsets.get(edge.target.layer, set())
        base = edge.confidence * max(edge.lift, 1.0)
        distance_penalty = 1.0 + layer_decay * max(edge.layer_delta, 0)
        read_penalty = read_cost_per_gib * (read_bytes / 1024**3)
        score = (base / distance_penalty) - read_penalty
        candidate = RiftManifoldAttentionScore(
            node=edge.target,
            score=score,
            confidence=edge.confidence,
            lift=edge.lift,
            layer_delta=edge.layer_delta,
            resident=resident,
            read_bytes=read_bytes,
        )
        current = best_by_node.get(edge.target)
        if current is None or candidate.score > current.score:
            best_by_node[edge.target] = candidate

    ranked = sorted(
        best_by_node.values(),
        key=lambda item: (-item.score, item.node.layer, item.node.expert),
    )
    if max_nodes is not None:
        ranked = ranked[: max(0, int(max_nodes))]
    return tuple(ranked)


def build_rift_map_from_pack_readers(
    readers: Mapping[str, ExpertPackReader],
    *,
    tile_bytes: int,
) -> RiftMap:
    """Build a 4D RiftMap from already-open expert pack readers."""

    if tile_bytes <= 0:
        raise ValueError("tile_bytes must be positive")
    tiles: list[RiftTile] = []
    for _source_key, reader in sorted(readers.items(), key=lambda item: str(item[0])):
        tiles.extend(_tiles_from_reader(reader, tile_bytes=tile_bytes))
    return RiftMap(tiles=tuple(sorted(tiles, key=lambda tile: tile.coord)), tile_bytes=tile_bytes)


def build_rift_map_from_pack_dir(pack_dir: str | Path, *, tile_bytes: int) -> RiftMap:
    """Open ``pack_dir`` and build a 4D RiftMap."""

    from smarttensor.packstore import open_model_packs

    readers = open_model_packs(pack_dir, access_mode="pread")
    try:
        return build_rift_map_from_pack_readers(readers, tile_bytes=tile_bytes)
    finally:
        for reader in readers.values():
            reader.close()


def _tiles_from_reader(reader: ExpertPackReader, *, tile_bytes: int) -> list[RiftTile]:
    tiles: list[RiftTile] = []
    pack_path = str(reader.path)
    for tensor_name, record in sorted(reader.records.items()):
        parsed = parse_expert_tensor_name(tensor_name)
        if parsed is None:
            continue
        layer, projection = parsed
        tiles.extend(
            _tiles_from_record(
                record,
                layer=layer,
                projection=projection,
                pack_path=pack_path,
                source=reader.source,
                tensor_name=tensor_name,
                tile_bytes=tile_bytes,
            )
        )
    return tiles


def _tiles_from_record(
    record: PackTensorRecord,
    *,
    layer: int,
    projection: str,
    pack_path: str,
    source: str | None,
    tensor_name: str,
    tile_bytes: int,
) -> list[RiftTile]:
    tiles: list[RiftTile] = []
    for expert in range(record.expert_count):
        expert_start = record.expert_offset(expert)
        tile_index = 0
        tile_offset = 0
        while tile_offset < record.expert_nbytes:
            nbytes = min(tile_bytes, record.expert_nbytes - tile_offset)
            tiles.append(
                RiftTile(
                    coord=RiftCoordinate(
                        layer=int(layer),
                        expert=int(expert),
                        projection=str(projection),
                        tile=int(tile_index),
                    ),
                    pack_path=pack_path,
                    source=source,
                    tensor_name=tensor_name,
                    dtype=record.dtype,
                    shape=record.shape,
                    file_offset=expert_start + tile_offset,
                    nbytes=nbytes,
                    expert_nbytes=record.expert_nbytes,
                    expert_stride=record.expert_stride,
                    tile_offset=tile_offset,
                )
            )
            tile_index += 1
            tile_offset += nbytes
    return tiles


def rift_map_summary(rift_map: RiftMap) -> dict[str, Any]:
    """Return JSON-friendly map shape telemetry."""

    projection_count = len({tile.coord.projection for tile in rift_map.tiles})
    expert_count = len({(tile.coord.layer, tile.coord.expert) for tile in rift_map.tiles})
    spans = rift_map.coalesced_spans()
    return {
        "format": "rift-map-v0",
        "manifold": {
            "storage_axes": ["layer", "expert", "projection", "tile"],
            "pathway_axes": ["tau", "layer", "expert", "projection", "tile"],
            "tau": "decode_step",
        },
        "tile_bytes": rift_map.tile_bytes,
        "layers": len(rift_map.layers),
        "layer_ids": list(rift_map.layers),
        "layer_experts": expert_count,
        "projection_lanes": projection_count,
        "tiles": rift_map.tile_count,
        "logical_bytes": rift_map.logical_bytes,
        "coalesced_spans": len(spans),
    }


def rift_map_to_dict(rift_map: RiftMap, *, include_tiles: bool = False) -> dict[str, Any]:
    """Return a JSON-friendly RiftMap payload."""

    payload = rift_map_summary(rift_map)
    if include_tiles:
        payload["tile_index"] = [
            {
                "coord": {
                    "layer": tile.coord.layer,
                    "expert": tile.coord.expert,
                    "projection": tile.coord.projection,
                    "tile": tile.coord.tile,
                    "page_id": tile.coord.page_id(),
                },
                "pack_path": tile.pack_path,
                "source": tile.source,
                "tensor_name": tile.tensor_name,
                "dtype": tile.dtype,
                "shape": list(tile.shape),
                "file_offset": tile.file_offset,
                "nbytes": tile.nbytes,
                "expert_nbytes": tile.expert_nbytes,
                "expert_stride": tile.expert_stride,
                "tile_offset": tile.tile_offset,
            }
            for tile in rift_map.tiles
        ]
    return payload
