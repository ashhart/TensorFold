"""Flash Next rank settings from the real source methods, with no GPU runtime imports."""

import ast
from pathlib import Path
import types
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold"


def function(path, name, namespace):
    tree = ast.parse((SOURCE / path).read_text())
    node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)
    node.body = [item for item in node.body if not isinstance(item, ast.ImportFrom)]
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    exec(compile(module, str(SOURCE / path), "exec"), namespace)
    return namespace[name]


class Tensor(list):
    def numel(self):
        return len(self)

    def view(self, rows, width):
        width = len(self) // rows if width == -1 else width
        return Tensor(Tensor(self[start:start + width]) for start in range(0, len(self), width))

    def cpu(self):
        return self

    def tolist(self):
        return list(self)

    def __getitem__(self, index):
        if isinstance(index, tuple):
            return list.__getitem__(self, index[0])[index[1]]
        return list.__getitem__(self, index)


class Comm:
    def __init__(self, peers=None):
        self.peers, self.sent = peers, None

    def all_gather(self, mine, both):
        self.sent = list(mine)
        both[:] = [value for row in (self.peers or [mine, mine]) for value in row]


torch = types.SimpleNamespace(tensor=lambda values, **kw: Tensor(values),
                              empty=lambda size, **kw: Tensor([0] * size[0]),
                              equal=lambda left, right: left == right, int64=None)
precision = types.SimpleNamespace(fp8=lambda: False,
                                  same_on_ranks=function("cuda/prompt_precision.py", "same_on_ranks", {}))
same_settings = function("families/qwen4_exp/cuda/engine.py", "_same_settings",
                         {"BITS_OF": {"bf16": 16}, "prompt_precision": precision})
prefill_rows = function("cuda/geometry.py", "indexed_prefill_rows", {})


def engine(value, peers=None):
    with patch.dict("os.environ", {}, clear=True):
        if value is not None:
            with patch.dict("os.environ", {"TENSORFOLD_PREFILL_ROWS": value}):
                rows = prefill_rows()
        else:
            rows = prefill_rows()
    return types.SimpleNamespace(depth=6, confidence=.7, max_len=8192, kv_dtype="bf16",
                                 prefill_rows=rows or 2048, comm=Comm(peers))


def settings(value):
    item = engine(value)
    same_settings(item, torch, None)
    return item.comm.sent


class FlashNextRankSettingsTests(unittest.TestCase):
    def test_equal_explicit_rows_pass_on_both_ranks(self):
        peers = [settings("3072"), settings(" 3072 ")]
        for value in ("3072", " 3072 "):
            same_settings(engine(value, peers), torch, None)

    def test_default_and_same_explicit_rows_pass(self):
        peers = [settings(None), settings("2048")]
        for value in (None, "2048"):
            same_settings(engine(value, peers), torch, None)

    def test_different_explicit_rows_refuse_on_both_ranks(self):
        peers = [settings("3072"), settings("4096")]
        for value in ("3072", "4096"):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "prompt rows"):
                same_settings(engine(value, peers), torch, None)

    def test_default_and_different_explicit_rows_refuse(self):
        peers = [settings(None), settings("4096")]
        for value in (None, "4096"):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "prompt rows"):
                same_settings(engine(value, peers), torch, None)

    def test_prompt_precision_still_uses_its_existing_refusal(self):
        peers = [settings("3072"), settings("3072")]
        peers[1][-1] = 1
        with self.assertRaisesRegex(RuntimeError, "different prompt precision"):
            same_settings(engine("3072", peers), torch, None)


if __name__ == "__main__":
    unittest.main()
