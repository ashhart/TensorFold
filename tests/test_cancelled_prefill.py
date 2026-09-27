"""A prefill the client stops keeps its progress: a retry resumes there and equals fresh; a taken prefix survives."""

import pytest

from tensorfold.server.cancellation import Cancellation, RequestCancelled
from tests.test_lane_server import expected_reply, make_app

TURN = [{"role": "user", "content": "abcdefgh"}, {"role": "assistant", "content": "ijklmnop"},
        {"role": "user", "content": "qrstuvwx"}]


def test_a_prefill_stopped_after_a_checkpoint_resumes_there_on_the_retry():
    app = make_app(lanes=1)
    try:
        prompt, history_len = app.render(TURN)
        copies = {"n": 0}
        copy = app.engine.copy_single_cache

        def counted(cache):
            copies["n"] += 1
            return copy(cache)

        app.engine.copy_single_cache = counted       # the prefill's checkpoint copy; the store keeps its own copier
        with pytest.raises(RequestCancelled):
            app.chat(TURN, max_tokens=4, cancellation=Cancellation(disconnected=lambda: copies["n"] > 0))
        app.engine.copy_single_cache = copy
        retry = app.chat(TURN, max_tokens=4)
        assert retry["cached_tokens"] == history_len
        assert retry["content"] == expected_reply(app, TURN, 4)[1]       # the same reply a fresh prefill gives
    finally:
        app.close()


def test_a_stopped_resume_keeps_the_prefix_it_took():
    from tensorfold.server.checkpoints import CheckpointStore
    from tensorfold.server.memory_budget import cache_nbytes
    from tensorfold.server.scheduler import ChatJob
    from tests.test_resume_memory import refuse_copy, scheduler_with, sized, stored

    prompt = [3] * 64
    store = CheckpointStore(3, copier=refuse_copy, sizer=cache_nbytes)
    store.insert([7] * 8, sized([7] * 8, 400), last_prompt=[7] * 8)
    store.insert(prompt[:48], sized(prompt[:48], 400), last_prompt=prompt[:48])
    scheduler, engine = scheduler_with(store, budget=2000)       # a copy would not fit: the prefix is taken
    polls = {"n": 0}

    def disconnected():
        polls["n"] += 1
        return polls["n"] > 1                                    # gone once the prefix was taken

    job = ChatJob("resume", prompt, 4, 0.0, cancellation=Cancellation(disconnected))
    scheduler._start_job(job)
    assert isinstance(job.error, RequestCancelled) and engine.prefill_calls == [("resume", 48)]
    assert prompt[:48] in stored(store)
