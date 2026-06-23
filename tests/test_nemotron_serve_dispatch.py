import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _nemotron_serve_args(**overrides):
    """Full serve args namespace defaulting to a nemotron_h dispatch.

    Mirrors the attribute set the gpt_oss/glm ``run_server`` dispatch tests
    build (see tests/test_safetensors_runtime.py::
    test_glm_server_defaults_to_direct_qmm) so ``run_server`` can build the
    runner kwargs without touching real model files.
    """

    args = SimpleNamespace(
        model_dir=Path("/tmp/nemotron"),
        served_name=None,
        host="127.0.0.1",
        port=0,
        retain_layers=None,
        resident_budget="160GiB",
        loader_backend="native",
        pin_policy="phase",
        expert_hot_set="",
        sliding_cache="rotating",
        native_layers=0,
        pack_dir=None,
        pack_read_workers=1,
        weight_page_budget=None,
        weight_page_policy="auto",
        weight_page_rows=1,
        decode_scheduler="auto",
        expert_compute_mode="table",
        expert_prefetch="off",
        expert_prefetch_cap=32,
        expert_slot_capacity=None,
        page_experts=True,
        draft="off",
        draft_model=None,
        draft_command=None,
        max_draft=8,
        speculative_scheduler="linear",
        max_branches=16,
        draft_margin=0.5,
        reasoning_effort="low",
        max_batch_size=1,
        batch_wait_ms=0,
        exact_mode="target-verified",
        max_tokens_default=8,
        enable_thinking=False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def _dispatch_nemotron_runner_kwargs(args):
    """Drive ``run_server`` for a nemotron_h model and capture runner kwargs.

    Mirrors the gpt_oss/glm dispatch test harness: stub the config to
    ``nemotron_h``, swap in a capturing ``FakeRunner`` for the live adapter
    class, and short-circuit the HTTP server so ``run_server`` returns 0.
    """

    from smarttensor.server import run_server

    captured: dict[str, object] = {}

    class FakeRunner:
        def __init__(self, model_dir, **kwargs) -> None:
            captured["model_dir"] = model_dir
            captured["kwargs"] = kwargs
            self.base_retain_layers = set()
            self.tokenizer = object()
            self.model_type = "nemotron_h"

        def close(self) -> None:
            captured["closed"] = True

    class FakeServer:
        def __init__(self, address, handler) -> None:
            captured["address"] = address

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            captured["server_closed"] = True

    with mock.patch(
        "smarttensor.adapters.mlx.load_mlx_config",
        return_value={"model_type": "nemotron_h"},
    ), mock.patch(
        "smarttensor.adapters.mlx.NemotronHStreamingForwardRunner",
        FakeRunner,
    ), mock.patch(
        "smarttensor.server.ThreadingHTTPServer",
        FakeServer,
    ):
        assert run_server(args) == 0

    return captured["kwargs"]


class ServeDispatchTests(unittest.TestCase):
    def test_nemotron_h_resolves_to_runner(self):
        from smarttensor import server
        from smarttensor.adapters.mlx import NemotronHStreamingForwardRunner

        self.assertIs(server.RUNNERS["nemotron_h"], NemotronHStreamingForwardRunner)

    def test_existing_runners_unchanged(self):
        from smarttensor import server

        for mt in ("deepseek_v3", "glm_moe_dsa", "qwen3_5_moe", "gpt_oss"):
            self.assertIn(mt, server.RUNNERS)

    def test_nemotron_weight_page_kwargs(self):
        """nemotron_h serve routes --weight-page-* into the runner kwargs.

        Mirrors the gpt_oss dispatch (server.py): a ``--weight-page-budget``
        is parsed via ``parse_bytes`` and a non-default policy/rows are passed
        through; with no budget the cache stays off (None) so default serving
        is unchanged.
        """

        from smarttensor.planner import parse_bytes

        # Cache requested: budget parsed, non-default policy + rows passed through.
        kwargs = _dispatch_nemotron_runner_kwargs(
            _nemotron_serve_args(
                weight_page_budget="150GiB",
                weight_page_policy="frequency",
                weight_page_rows=1,
            )
        )
        self.assertEqual(
            kwargs["weight_page_budget_bytes"], parse_bytes("150GiB")
        )
        self.assertEqual(kwargs["weight_page_policy"], "frequency")
        self.assertEqual(kwargs["weight_page_rows"], 1)

        # Default/unset: no --weight-page-budget => cache off (None).
        default_kwargs = _dispatch_nemotron_runner_kwargs(_nemotron_serve_args())
        self.assertIsNone(default_kwargs["weight_page_budget_bytes"])


if __name__ == "__main__":
    unittest.main()
