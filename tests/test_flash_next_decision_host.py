"""Decision admission and collector tests using source seams without GPU-runtime imports."""

import ast
import os
from pathlib import Path
import queue
import sys
import threading
import types
import unittest

ROOT = Path(os.environ.get('TF_DECISION_TEST_SRC', Path(__file__).resolve().parents[1] / 'src'))


def load_source(name, path, injected=None):
    module = types.ModuleType(name)
    sys.modules[name] = module
    module.__dict__.update(injected or {})
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or not node.level]
    exec(compile(tree, str(ROOT / path), 'exec'), module.__dict__)
    return module


probabilities = load_source('decision_host_probabilities', 'tensorfold/engine/probabilities.py')
streams = load_source('decision_host_streams', 'tensorfold/cuda/streams.py')
memory_gate = load_source('decision_host_memory', 'tensorfold/cuda/memory_gate.py')
scheduler = load_source('decision_host_scheduler', 'tensorfold/cuda/scheduler.py',
                        {'Stream': streams.Stream, 'NoRoom': memory_gate.NoRoom})
prefixes = load_source('decision_host_prefixes', 'tensorfold/families/qwen4_exp/cuda/prefixes.py')


class Sampling:
    def __init__(self, **values):
        self.__dict__.update(values)


def source_class(path, name, methods, injected):
    tree = ast.parse((ROOT / path).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    for node in ast.walk(cls):
        if isinstance(node, ast.FunctionDef):
            node.body = [item for item in node.body if not isinstance(item, ast.ImportFrom)]
    namespace = {'__name__': 'decision_host_methods', **injected}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, cls], type_ignores=[]))
    exec(compile(module, str(ROOT / path), 'exec'), namespace)
    return namespace[name]


Engine = source_class('tensorfold/families/qwen4_exp/cuda/engine.py', 'FlashNextEngine',
                      {'_limit', 'supports_logprobs', 'score_labels', 'score_labels_many'},
                      {'Sampling': Sampling, 'LabelProbabilities': probabilities.LabelProbabilities})


class Jobs:
    def __init__(self, finite=True):
        self.requests, self.finite = [], finite

    def submit_many(self, requests):
        self.requests.extend(requests)
        for item in requests:
            probe = item['probabilities']
            values = [float(token) for token in probe.labels]
            probe.add_labels(len(item['prompt']), values if self.finite else [float('nan')] * len(values), 10.)
        return [{} for _ in requests]


def engine(grouped=True):
    instance = Engine()
    instance.tp, instance.depth, instance.max_len = 1, 6, 64
    instance.w = types.SimpleNamespace(cfg=types.SimpleNamespace(vocab=32))
    instance.scheduler, instance.cache = Jobs() if grouped else None, [('kept', object())]
    instance.calls = []
    def generate(prompt, count, sampling, emit, **options):
        instance.calls.append((prompt, count, options))
        probe = options['probabilities']
        probe.add_labels(len(prompt), [float(token) for token in probe.labels], 10.)
    instance.generate = generate
    return instance


class DecisionHostTests(unittest.TestCase):
    def test_collector_only_accepts_its_own_position(self):
        a, b = probabilities.LabelProbabilities([2, 3], 5), probabilities.LabelProbabilities([7], 6)
        a.add_labels(6, [9.], 11.)
        self.assertIsNone(a.label_logits)
        a.add_labels(5, [2., 3.], 10.)
        self.assertEqual(a.label_logits, [2., 3.])
        self.assertIsNone(b.label_logits)

    def test_grouped_scores_are_fresh_one_shots_and_ordered(self):
        item = engine()
        kept = list(item.cache)
        self.assertEqual(item.score_labels_many([([1, 2], [3, 4]), ([5], [7])]),
                         [([3., 4.], 10.), ([7.], 10.)])
        jobs = item.scheduler.requests
        self.assertEqual([job['prompt'] for job in jobs], [[1, 2], [5]])
        self.assertTrue(all(job['count'] == 1 and job['draft'] is False for job in jobs))
        self.assertIsNot(jobs[0]['probabilities'], jobs[1]['probabilities'])
        self.assertEqual(item.cache, kept)
        self.assertEqual(item.calls, [])

    def test_solo_collector_does_not_request_a_decode_loop(self):
        item = engine(False)
        self.assertEqual(item.score_labels([1], [7]), ([7.], 10.))
        self.assertEqual(item.calls[0][1], 1)
        self.assertIs(item.calls[0][2]['draft'], False)

    def test_bad_second_prompt_is_rejected_before_any_group_is_queued(self):
        item = engine()
        with self.assertRaises(ValueError):
            item.score_labels_many([([1], [7]), ([2] * 64, [3])])
        self.assertEqual(item.scheduler.requests, [])

    def test_bad_label_is_rejected_before_queueing(self):
        item = engine()
        with self.assertRaises(ValueError):
            item.score_labels_many([([1], [7]), ([2], [32])])
        self.assertEqual(item.scheduler.requests, [])

    def test_nonfinite_group_score_is_rejected(self):
        item = engine()
        item.scheduler.finite = False
        with self.assertRaises(ValueError):
            item.score_labels_many([([1], [7]), ([2], [8])])

    def test_two_ranks_are_rejected_before_queueing(self):
        item = engine()
        item.tp = 2
        with self.assertRaises(ValueError):
            item.score_labels_many([([1], [7]), ([2], [8])])
        self.assertEqual(item.scheduler.requests, [])

    def test_bulk_queue_keeps_priority_and_arrival(self):
        waiting = scheduler.Waiting()
        pairs = [(streams.Stream([1], 1, background=True), queue.Queue()),
                 (streams.Stream([2], 1), queue.Queue()), (streams.Stream([3], 1), queue.Queue())]
        waiting.put_many(pairs)
        self.assertEqual([waiting.get_nowait()[0].prompt for _ in pairs], [[2], [3], [1]])

    def test_worker_cannot_see_a_partial_group(self):
        waiting = scheduler.Waiting()
        first, attempted = threading.Event(), threading.Event()
        original = waiting._put
        count = [0]
        def put(value):
            original(value)
            count[0] += 1
            if count[0] == 1:
                first.set()
                self.assertTrue(attempted.wait(2))
        waiting._put = put
        seen = []
        def consumer():
            if not first.wait(2):
                return
            attempted.set()
            seen.append(waiting.get()[0].prompt)
            seen.append(waiting.qsize())
        worker = threading.Thread(target=consumer)
        worker.start()
        waiting.put_many([(streams.Stream([n], 1), queue.Queue()) for n in (1, 2, 3)])
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertEqual(seen, [[1], 2])

    def test_submit_many_drains_a_request_error_and_finishes_other_jobs(self):
        class Decoder:
            def __init__(self):
                self.streams, self.finished = {}, []
            def live(self):
                return len(self.streams)
            def admit(self, stream):
                if stream.prompt == [31]:
                    raise ValueError('refused fixture')
                self.streams[id(stream)] = stream
            def round(self):
                done = list(self.streams.values())
                for stream in done:
                    stream.take([0])
                return done
            def finish(self, done):
                for stream in done:
                    self.finished.append(stream.prompt)
                    self.streams.pop(id(stream), None)
        decoder = Decoder()
        worker = scheduler.Scheduler(decoder, max_streams=4)
        try:
            with self.assertRaisesRegex(ValueError, 'refused fixture'):
                worker.submit_many([{'prompt': [n], 'count': 1, 'sampling': None} for n in (31, 2, 3)])
            self.assertEqual(decoder.finished, [[2], [3]])
        finally:
            worker.close()
        self.assertFalse(worker.thread.is_alive())

    def test_fresh_score_can_reclaim_an_idle_kept_slot(self):
        idle = object()
        owner = types.SimpleNamespace(kept=[([1], idle, {}, None)], free=[], depth=6,
                                      _busy=lambda: set())
        owner._drop_kept = lambda state: setattr(owner, 'kept', [k for k in owner.kept if k[1] is not state])
        state, hit, cached = prefixes.slot_for(owner, [2], False)
        self.assertIs(state, idle)
        self.assertIsNone(hit)
        self.assertEqual(cached, 0)
        self.assertEqual(owner.kept, [])

    def test_decision_prompt_respects_the_existing_live_decode_budget(self):
        seen = []
        def budget(rows, live, *rest):
            seen.append(live)
            return 64 if live else rows
        Decoder = source_class('tensorfold/families/qwen4_exp/cuda/multi.py', 'MultiDecoder', {'_pass_rows'},
                               {'pass_limit': budget, 'PASS_MIN': 64})
        item = Decoder()
        item.streams = {1: types.SimpleNamespace(done=False)}
        item.filling = [types.SimpleNamespace(probabilities=probabilities.LabelProbabilities([1], 2))]
        item.prefill_rows, item.share, item.round_s, item.row_s = 1024, .25, .02, .0001
        self.assertEqual(item._pass_rows(), 64)
        self.assertEqual(seen, [True])


if __name__ == '__main__':
    unittest.main()
