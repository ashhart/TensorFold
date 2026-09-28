from types import SimpleNamespace
from threading import RLock

import numpy as np

from tensorfold.families.qwen3_5.family import Qwen35Family
from tensorfold.families.qwen3_5.vision import VisionPrompt, merge_vision_prompts
from tensorfold.server.memory_budget import CacheMemory
from tensorfold.server.prompt_memory import PromptMemory
from tensorfold.server.scheduler import ChatJob, Scheduler
from tests.lane_fakes import FakeEngine


def vision_prompt(tokens, *, patches=2, digest="image"):
    return VisionPrompt(
        input_ids=list(tokens),
        model_inputs={
            "input_ids": np.asarray([tokens], dtype=np.int64),
            "mask": np.ones((1, len(tokens)), dtype=np.int64),
            "pixel_values": np.arange(patches * 4, dtype=np.float32).reshape(patches, 4),
            "image_grid_thw": np.asarray([[1, 1, patches]], dtype=np.int64),
        },
        image_digest=digest,
        reserved_bytes=128,
    )


def image_job(name, tokens):
    return ChatJob(name, list(tokens), 4, 0.0, multimodal=vision_prompt(tokens, digest=name), drafts=False)


class BatchFakeEngine(FakeEngine):
    def __init__(self, *, fail_batch=False, fail_single=False, on_batch=None):
        super().__init__()
        self.vision_batches = []
        self.fail_batch = fail_batch
        self.fail_single = fail_single
        self.on_batch = on_batch

    def _add_as_text(self, stream, **kwargs):
        multimodal, stream.multimodal = stream.multimodal, None
        try:
            return super().add_stream(stream, **kwargs)
        finally:
            stream.multimodal = multimodal

    def add_stream(self, stream, **kwargs):
        if stream.multimodal is not None:
            if self.fail_single:
                raise RuntimeError("synthetic singleton failure")
            return self._add_as_text(stream, **kwargs)
        return super().add_stream(stream, **kwargs)

    def add_vision_streams(self, streams):
        self.vision_batches.append([stream.stream_id for stream in streams])
        if self.on_batch is not None:
            self.on_batch()
        if self.fail_batch:
            self.fail_batch = False
            raise RuntimeError("synthetic batch failure")
        for stream in streams:
            self._add_as_text(stream)


def scheduler(engine, lanes=3):
    return Scheduler(engine, lanes=lanes, eos_ids=frozenset({-1}))


def test_adjacent_image_jobs_prefill_as_one_batch():
    engine = BatchFakeEngine()
    service = scheduler(engine)
    jobs = [image_job(f"image-{index}", [index + 1, index + 2]) for index in range(3)]
    for job in jobs:
        service.submit(job)

    service._admit()

    assert engine.vision_batches == [["image-0", "image-1", "image-2"]]
    assert engine.active_count == 3
    assert set(service._jobs) == {job.job_id for job in jobs}


def test_text_job_is_not_reordered_to_form_an_image_batch():
    engine = BatchFakeEngine()
    service = scheduler(engine)
    jobs = [
        image_job("image-1", [1, 2]),
        ChatJob("text", [3, 4], 4, 0.0),
        image_job("image-2", [5, 6]),
    ]
    for job in jobs:
        service.submit(job)

    service._admit()

    assert engine.vision_batches == []
    assert [stream_id for stream_id, _ in engine.prefill_calls] == ["image-1", "text", "image-2"]


def test_a_batch_failure_retries_each_image_separately():
    engine = BatchFakeEngine(fail_batch=True)
    service = scheduler(engine, lanes=2)
    jobs = [image_job("image-1", [1, 2]), image_job("image-2", [3, 4])]
    for job in jobs:
        service.submit(job)

    service._admit()

    assert engine.vision_batches == [["image-1", "image-2"]]
    assert [stream_id for stream_id, _ in engine.prefill_calls] == ["image-1", "image-2"]
    assert all(job.error is None for job in jobs)
    assert engine.active_count == 2


def test_cancellation_after_batch_prefill_discards_only_that_row():
    jobs = [image_job("kept", [1, 2]), image_job("cancelled", [3, 4])]
    engine = BatchFakeEngine(on_batch=jobs[1].cancellation.cancel)
    service = scheduler(engine, lanes=2)
    for job in jobs:
        service.submit(job)

    service._admit()

    assert set(service._jobs) == {"kept"}
    assert jobs[1].done.is_set()
    assert jobs[1].error is not None
    assert engine.active_count == 1


def test_cancelled_queued_image_releases_host_buffers_and_slot():
    job = image_job("cancelled", [1, 2])
    releases = []
    job.multimodal.release_slot = lambda: releases.append(True)
    job.cancellation.cancel()
    service = scheduler(BatchFakeEngine())
    service.submit(job)

    service._admit()

    assert job.done.is_set()
    assert job.multimodal.model_inputs == {}
    assert releases == [True]


def test_failed_batch_and_singleton_fallback_release_every_image_slot():
    jobs = [image_job("first", [1, 2]), image_job("second", [3, 4])]
    releases = []
    for job in jobs:
        job.multimodal.release_slot = lambda: releases.append(True)
    service = scheduler(BatchFakeEngine(fail_batch=True, fail_single=True), lanes=2)
    for job in jobs:
        service.submit(job)

    service._admit()

    assert all(job.done.is_set() and job.error is not None for job in jobs)
    assert all(job.multimodal.model_inputs == {} for job in jobs)
    assert releases == [True, True]


def test_batch_admission_reserves_live_stream_reply_growth():
    service = scheduler(BatchFakeEngine(), lanes=3)
    calls = []
    service.prompt_memory = SimpleNamespace(
        would_fit_batch=lambda prompts, replies, **kwargs: calls.append(kwargs) or True
    )
    service.admission = SimpleNamespace(memory=SimpleNamespace(per_token=5))
    active = ChatJob("active", list(range(10)), 4, 0.0)
    active.stream = SimpleNamespace(context=list(range(8)), finished=False)
    service._jobs[active.job_id] = active

    assert service._fits_batch([image_job("first", [1, 2]), image_job("second", [3, 4])])

    # Six remaining live tokens plus both prepared-image reservations.
    assert calls == [{"extra_bytes": 2 * 128 + 6 * 5}]


def test_vision_prompt_merge_right_pads_text_and_concatenates_patches():
    first = vision_prompt([7, 8], patches=2, digest="a")
    second = vision_prompt([9, 10, 11], patches=3, digest="b")

    batch = merge_vision_prompts([first, second])

    assert batch.lengths == (2, 3)
    assert batch.input_ids.tolist() == [[7, 8, 0], [9, 10, 11]]
    assert batch.model_inputs["mask"].tolist() == [[1, 1, 0], [1, 1, 1]]
    assert batch.model_inputs["pixel_values"].shape == (5, 4)
    assert batch.model_inputs["image_grid_thw"].tolist() == [[1, 1, 2], [1, 1, 3]]


def test_vision_prompt_discards_host_buffers_and_releases_its_slot_once():
    prompt = vision_prompt([1, 2])
    releases = []
    prompt.release_slot = lambda: releases.append(True)

    prompt.discard_buffers()
    prompt.discard_buffers()

    assert prompt.model_inputs == {}
    assert releases == [True]


class ExtractableCache:
    keys = 0
    state = []

    def __init__(self, row=None):
        self.row = row

    def extract(self, index):
        return ExtractableCache(index)

    def finalize(self):
        return None


def test_split_vision_batch_preserves_row_rope_delta_and_digest():
    import mlx.core as mx

    family = Qwen35Family.__new__(Qwen35Family)
    prompts = [
        vision_prompt([1, 2], digest="first"),
        vision_prompt([3, 4, 5], digest="second"),
    ]

    rows = family.split_vision_batch(prompts, [ExtractableCache()], mx.array([[7], [11]]))

    assert [rows[index][0].row for index in rows] == [0, 1]
    assert [rows[index].vision_state for index in rows] == [
        {"rope_delta": 7, "image_digest": "first"},
        {"rope_delta": 11, "image_digest": "second"},
    ]


def test_make_vision_batch_cache_uses_mlx_vlm_right_padding_metadata():
    from mlx_vlm.models.cache import ArraysCache, BatchKVCache, KVCache

    family = Qwen35Family.__new__(Qwen35Family)
    family.vision_language = SimpleNamespace(
        make_cache=lambda: [KVCache(), ArraysCache(size=2)]
    )
    prompts = [vision_prompt([1, 2]), vision_prompt([3, 4, 5])]

    cache = family.make_vision_batch_cache(prompts)

    assert isinstance(cache[0], BatchKVCache)
    assert cache[0].left_padding.tolist() == [0, 0]
    assert cache[0]._right_padding.tolist() == [1, 0]
    assert isinstance(cache[1], ArraysCache)
    assert cache[1].lengths.tolist() == [2, 3]


def test_vision_prefill_batch_runs_one_embedding_and_language_forward():
    import mlx.core as mx

    calls = []

    def embeddings(*, input_ids, **kwargs):
        calls.append(("vision", tuple(input_ids.shape), tuple(kwargs["pixel_values"].shape)))
        batch, length = input_ids.shape
        return SimpleNamespace(
            inputs_embeds=mx.broadcast_to(
                mx.arange(batch * length).reshape(batch, length, 1),
                (batch, length, 4),
            ),
            position_ids=mx.zeros((3, batch, length), dtype=mx.int32),
            rope_deltas=mx.array([[2], [3]], dtype=mx.int32),
        )

    def language(input_ids, *, inputs_embeds, cache, position_ids):
        calls.append(("language", tuple(input_ids.shape), tuple(position_ids.shape), len(cache)))
        return inputs_embeds

    family = Qwen35Family.__new__(Qwen35Family)
    family.inner = SimpleNamespace(get_input_embeddings=embeddings)
    family.vision_core = language
    prompts = [
        vision_prompt([1, 2], patches=2, digest="first"),
        vision_prompt([3, 4, 5], patches=3, digest="second"),
    ]

    hidden, rope_deltas = family.vision_prefill_batch(prompts, [ExtractableCache()])

    assert hidden.shape == (2, 1, 4)
    assert hidden[:, 0, 0].tolist() == [1, 5]
    assert rope_deltas.tolist() == [[2], [3]]
    assert calls == [
        ("vision", (2, 3), (5, 4)),
        ("language", (2, 3), (3, 2, 3), 1),
    ]


def test_prompt_memory_batch_keeps_each_row_for_independent_rounding():
    memory = PromptMemory.__new__(PromptMemory)
    memory._memory_lock = RLock()
    memory.runtime = SimpleNamespace(
        reset_peak_memory=lambda: None,
        get_active_memory=lambda: 0,
        get_cache_memory=lambda: 0,
    )
    memory.store = memory.profile = None
    memory.bootstrap = 0
    memory.budget = 10_000
    memory.require = lambda: None

    memory.begin_batch([10, 20], [3, 4], extra_bytes=500)
    assert memory.would_fit_batch([10, 20], [3, 4], extra_bytes=500)

    assert memory.prompt == 40
    assert memory.reply == 0
    assert memory._batch_prompts == (10, 20)
    assert memory._batch_replies == (3, 4)
    assert memory.cache_copies == memory.cache_instances == 2


def test_batch_projection_scales_fixed_recurrent_state_per_row_and_copy():
    memory = PromptMemory.__new__(PromptMemory)
    memory.profile = CacheMemory(fixed_bytes=100, bytes_per_token=10)
    memory.cache_instances = 3
    memory.cache_copies = 2
    memory.extra_bytes = memory.reply = 0
    memory._batch_prompts = memory._batch_replies = None
    memory._used = lambda: 0
    memory._work = lambda tokens: 0

    assert memory.projected(5) == 2 * (3 * 100 + 256 * 10)


def test_batch_projection_rounds_source_and_each_extracted_row_independently():
    memory = PromptMemory.__new__(PromptMemory)
    memory.profile = CacheMemory(fixed_bytes=100, bytes_per_token=10, step=256)
    memory.cache_instances = 3
    memory.cache_copies = 2
    memory.extra_bytes = memory.reply = 0
    memory._batch_prompts = (1, 257, 10)
    memory._batch_replies = (0, 0, 0)
    memory._used = lambda: 0
    memory._work = lambda tokens: 0

    padded_source = 3 * (100 + 512 * 10)
    extracted_rows = (100 + 256 * 10) + (100 + 512 * 10) + (100 + 256 * 10)
    assert memory.projected(sum(memory._batch_prompts)) == padded_source + extracted_rows


def test_batch_work_growth_rounds_each_padded_row_independently():
    memory = PromptMemory.__new__(PromptMemory)
    memory.profile = CacheMemory(fixed_bytes=100, bytes_per_token=10, step=256)
    memory._batch_prompts = (1, 1, 1)
    memory.bootstrap = memory.observed_work = memory.workspace_per_token = memory.heads = 0
    memory.score_rows = 144

    assert memory._work(3) == 3 * (100 + 256 * 10)
