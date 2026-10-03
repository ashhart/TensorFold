"""Rank-local memory facts become a shared conservative decision before admission or round agreement."""

import ast
import hashlib
import json
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Callable

import pytest


def _methods():
    root = Path(__file__).resolve().parents[1] / 'src/tensorfold'
    module = ast.parse((root / 'families/qwen4_exp/cuda/multi_tp.py').read_text())
    owner = next(node for node in module.body if isinstance(node, ast.ClassDef) and node.name == 'TwoRanks')
    methods = [node for node in owner.body if isinstance(node, ast.FunctionDef)
               and node.name in ('_agree', '_prepare_admission', '_prepare_round')]
    for method in methods:
        method.body = [node for node in method.body if not isinstance(node, ast.ImportFrom)]
    body = ast.ClassDef(name='Ranks', bases=[], keywords=[], body=methods, decorator_list=[], type_params=[])
    capacity = ast.parse((root / 'cuda/capacity.py').read_text())
    gather = next(node for node in capacity.body if isinstance(node, ast.FunctionDef) and node.name == 'gather_ints')
    return ast.fix_missing_locations(ast.Module(body=[gather, body], type_ignores=[]))


@pytest.mark.parametrize('rooms,expected', [([9, 5], 'wait'), ([5, 9], 'wait'), ([9, 9], 'admit')])
def test_skewed_rank_memory_produces_identical_admission_and_rounds(rooms, expected):
    barrier = threading.Barrier(2)
    values, results = {}, [None, None]
    plan = {'slot': 0, 'cached': 0, 'resume_slot': None, 'error': None, 'points': [], 'need': 7,
            'ended': [], 'solo': None, 'passes': [], 'mixed': [], 'pass_width': 512}

    class OutOfStep(RuntimeError):
        pass

    class NoRoom(RuntimeError):
        pass

    class Tensor:
        def __init__(self, data):
            self.data, self.rows = list(data), None

        def view(self, rows, unused):
            self.rows = rows
            return self

        def tolist(self):
            width = len(self.data) // self.rows
            return [self.data[start:start + width] for start in range(0, len(self.data), width)]

    torch = SimpleNamespace(int64=None, tensor=lambda data, **kwargs: Tensor(data),
                            empty=lambda shape, **kwargs: Tensor([0] * shape[0]))
    namespace = {'hashlib': hashlib, 'json': json, 'torch': torch, 'Callable': Callable,
                 'OutOfStep': OutOfStep, 'NoRoom': NoRoom,
                 'shape': lambda dec: ['shared-state'], '_pack': lambda sampling: None,
                 'admission': lambda dec, stream: dict(plan), 'round_plan': lambda dec: dict(plan),
                 'ready': lambda dec, proposed: dec.room >= proposed['need'],
                 'apply': lambda dec, proposed: setattr(dec, 'applied', True)}

    def gather(rank, send, receive):
        values[rank] = list(send.data)
        barrier.wait(timeout=2)
        receive.data = [*values[0], *values[1]]
        barrier.wait(timeout=2)
    exec(compile(_methods(), 'actual-tp-planning-methods', 'exec'), namespace)

    def run(rank):
        decoder = namespace['Ranks']()
        decoder.w = SimpleNamespace(comm=SimpleNamespace(all_gather=lambda send, receive: gather(rank, send, receive)))
        decoder.room, decoder.next_id, decoder.link = rooms[rank], 0, None
        decoder.slots, decoder.kept, decoder.streams, decoder.filling = [object()], [], {1: object()}, []
        stream = SimpleNamespace(prompt=[1, 2], count=4, sampling=None, draft=True,
                                 stop_eos=False, background=False)
        outcome = []
        for operation in ('admission', 'round'):
            try:
                if operation == 'admission':
                    decoder._prepare_admission(stream, dict(plan))
                else:
                    decoder._prepare_round(dict(plan))
                outcome.append('admit')
            except NoRoom:
                outcome.append('wait')
            except Exception as error:
                outcome.append(type(error).__name__)
        results[rank] = outcome

    threads = [threading.Thread(target=run, args=(rank,)) for rank in (0, 1)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
        assert not thread.is_alive()
    assert results == [[expected, expected], [expected, expected]]
