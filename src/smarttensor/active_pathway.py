"""Active-pathway runtime contracts for low-memory execution.

These primitives do not execute model math. They define the scheduling and
accounting contract for a future low-memory executor: keep a small hot pathway
resident, emit compact page faults for cold misses, certify safe cold skips
where a margin bound permits it, and render symbolic output bytecode.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class PathwayTraceStep:
    """One exact decode step's page requirements and local argmax margin."""

    required_pages: tuple[str, ...]
    margin: float
    visible_bytes: int = 0


@dataclass(frozen=True)
class ActivePathwayPlan:
    hot_pages_by_step: tuple[tuple[str, ...], ...]
    hit_rate: float
    peak_resident_pages: int


@dataclass(frozen=True)
class ProofCertificate:
    skipped_pages: tuple[str, ...]
    total_bound: float
    margin: float

    @property
    def valid(self) -> bool:
        return self.total_bound < self.margin


@dataclass(frozen=True)
class PageFaultPacket:
    step_index: int
    missing_pages: tuple[str, ...]
    resident_pages: tuple[str, ...]
    skipped_pages: tuple[str, ...] = ()


@dataclass(frozen=True)
class RuntimeStepResult:
    step_index: int
    packet: PageFaultPacket | None
    hot_hits: int
    skipped_pages: int
    loaded_bytes: int


@dataclass(frozen=True)
class RuntimeSummary:
    fault_packets: tuple[PageFaultPacket, ...]
    page_faults: int
    loaded_bytes: int
    hot_hits: int
    skipped_pages: int
    visible_bytes: int
    resident_peak_pages: int

    @property
    def useful_bytes_per_loaded_mb(self) -> float:
        if self.loaded_bytes <= 0:
            return float(self.visible_bytes)
        return self.visible_bytes / (self.loaded_bytes / 1_000_000)


@dataclass(frozen=True)
class BytecodeOp:
    op: str
    span_id: str | None = None
    text: str = ""


@dataclass(frozen=True)
class ActivePathwayStackResult:
    plan: ActivePathwayPlan
    summary: RuntimeSummary
    rendered_output: str


class ActivePathwayCompiler:
    """Compile a bounded hot-page plan from recent exact routing history."""

    def __init__(
        self,
        *,
        capacity_pages: int,
        history_window: int,
        seed_pages: Sequence[str] = (),
    ) -> None:
        if capacity_pages <= 0:
            raise ValueError("capacity_pages must be positive")
        if history_window < 0:
            raise ValueError("history_window must be non-negative")
        self.capacity_pages = capacity_pages
        self.history_window = history_window
        self.seed_pages = tuple(dict.fromkeys(seed_pages))

    def compile(self, steps: Sequence[PathwayTraceStep]) -> ActivePathwayPlan:
        hot_pages_by_step: list[tuple[str, ...]] = []
        hit_count = 0
        required_count = 0

        for index, step in enumerate(steps):
            counts: dict[str, int] = {}
            for seed in self.seed_pages:
                counts[seed] = self.history_window + 1

            start = max(0, index - self.history_window)
            for prior in steps[start:index]:
                for page in dict.fromkeys(prior.required_pages):
                    counts[page] = counts.get(page, 0) + 1

            ranked = sorted(counts, key=lambda page: (-counts[page], page))
            hot = tuple(ranked[: self.capacity_pages])
            hot_pages_by_step.append(hot)

            required = set(step.required_pages)
            hit_count += len(required & set(hot))
            required_count += len(required)

        return ActivePathwayPlan(
            hot_pages_by_step=tuple(hot_pages_by_step),
            hit_rate=hit_count / required_count if required_count else 0.0,
            peak_resident_pages=max((len(pages) for pages in hot_pages_by_step), default=0),
        )


class ProofSkipCertifier:
    """Certify cold pages whose cumulative contribution bound cannot flip argmax."""

    def __init__(self, contribution_bounds: Mapping[str, float]) -> None:
        for page, bound in contribution_bounds.items():
            if bound < 0:
                raise ValueError(f"negative contribution bound for {page!r}")
        self.contribution_bounds = dict(contribution_bounds)

    def certify(
        self,
        step: PathwayTraceStep,
        *,
        resident_pages: Sequence[str],
    ) -> ProofCertificate:
        resident = set(resident_pages)
        cold_pages = [page for page in dict.fromkeys(step.required_pages) if page not in resident]
        total_bound = 0.0
        skipped: list[str] = []

        for page in sorted(cold_pages, key=lambda p: (self.contribution_bounds.get(p, 1.0), p)):
            bound = self.contribution_bounds.get(page, 1.0)
            if total_bound + bound < step.margin:
                total_bound += bound
                skipped.append(page)

        return ProofCertificate(
            skipped_pages=tuple(skipped),
            total_bound=total_bound,
            margin=step.margin,
        )


class PageFaultRuntime:
    """Simulate the exact low-memory runtime's page-fault contract."""

    def __init__(self, *, page_bytes: int, certifier: ProofSkipCertifier | None = None) -> None:
        if page_bytes <= 0:
            raise ValueError("page_bytes must be positive")
        self.page_bytes = page_bytes
        self.certifier = certifier

    def run(
        self,
        steps: Sequence[PathwayTraceStep],
        plan: ActivePathwayPlan,
        *,
        visible_bytes: int | None = None,
    ) -> RuntimeSummary:
        if len(plan.hot_pages_by_step) != len(steps):
            raise ValueError("plan length must match steps length")

        packets: list[PageFaultPacket] = []
        page_faults = 0
        hot_hits = 0
        skipped_count = 0
        resident_peak = 0

        for index, (step, resident_pages) in enumerate(zip(steps, plan.hot_pages_by_step)):
            required_unique = tuple(dict.fromkeys(step.required_pages))
            resident_set = set(resident_pages)
            hot_hits += len(set(required_unique) & resident_set)

            skipped_pages: tuple[str, ...] = ()
            if self.certifier is not None:
                certificate = self.certifier.certify(step, resident_pages=resident_pages)
                if certificate.valid:
                    skipped_pages = certificate.skipped_pages

            skipped_set = set(skipped_pages)
            missing_pages = tuple(
                page for page in required_unique if page not in resident_set and page not in skipped_set
            )
            skipped_count += len(skipped_set & set(required_unique))
            page_faults += len(missing_pages)
            resident_peak = max(resident_peak, len(resident_pages) + len(missing_pages))

            if missing_pages:
                packets.append(
                    PageFaultPacket(
                        step_index=index,
                        missing_pages=missing_pages,
                        resident_pages=resident_pages,
                        skipped_pages=skipped_pages,
                    )
                )

        output_bytes = sum(step.visible_bytes for step in steps) if visible_bytes is None else visible_bytes
        return RuntimeSummary(
            fault_packets=tuple(packets),
            page_faults=page_faults,
            loaded_bytes=page_faults * self.page_bytes,
            hot_hits=hot_hits,
            skipped_pages=skipped_count,
            visible_bytes=output_bytes,
            resident_peak_pages=resident_peak,
        )


class BytecodeVM:
    """Render symbolic output ops into normal user-visible text."""

    def __init__(self, spans: Mapping[str, str]) -> None:
        self.spans = dict(spans)

    def render(self, ops: Sequence[BytecodeOp]) -> str:
        chunks: list[str] = []
        for op in ops:
            if op.op == "copy":
                if op.span_id is None:
                    raise ValueError("copy op requires span_id")
                try:
                    chunks.append(self.spans[op.span_id])
                except KeyError as exc:
                    raise KeyError(f"unknown span_id {op.span_id!r}") from exc
            elif op.op == "insert":
                chunks.append(op.text)
            else:
                raise ValueError(f"unknown bytecode op {op.op!r}")
        return "".join(chunks)


def run_active_pathway_stack(
    steps: Sequence[PathwayTraceStep],
    *,
    compiler: ActivePathwayCompiler,
    runtime: PageFaultRuntime,
    bytecode_ops: Sequence[BytecodeOp] = (),
    spans: Mapping[str, str] | None = None,
) -> ActivePathwayStackResult:
    plan = compiler.compile(steps)
    rendered = BytecodeVM(spans or {}).render(bytecode_ops) if bytecode_ops else ""
    visible_bytes = len(rendered.encode("utf-8")) if rendered else None
    summary = runtime.run(steps, plan, visible_bytes=visible_bytes)
    return ActivePathwayStackResult(plan=plan, summary=summary, rendered_output=rendered)
