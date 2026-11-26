"""Synthetic complete-round records; none are GPU measurements."""
import copy
import hashlib
import json
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cuda_round_attribution as m


def fixture(mt2=False, wall=100):
    header = dict(schema='tensorfold.cuda-round-identity.v1', source='a'*40+':'+'b'*40,
                  binary_sha256='c'*64, checkpoint={'checkpoint_sha256': 'd'*64}, drafter=None,
                  quantization={'format': 'NVFP4'}, kv8=True, geometry={}, conditions='fixed',
                  operation_names=m.OPS, phase_names=m.PHASES, driver=13000, sm_major=12, sm_minor=0,
                  checkpoint_config_sha256='f'*64)
    raw = json.dumps(header, separators=(',', ':'))
    key = hashlib.sha256(raw.encode()).hexdigest()
    q = dict(m=8, n=17408, k=5120, sk=4, cluster=[4, 1, 1], reduce=True, mode='nvfp4')
    devices = [dict(correlation=2, parent_scope=1, graph_replay=None, stream=17,
                    phase='forward', op='qmmf', symbol='projection', begin_ns=10, end_ns=50, qmmf=[q]),
               dict(correlation=3, parent_scope=1, graph_replay=7, stream=17,
                    phase='forward', op='attention', symbol='attention', begin_ns=30, end_ns=60, qmmf=[])]
    hosts = [dict(correlation=1, parent_scope=0, phase='forward', kind='span', begin_ns=0, end_ns=70),
             dict(correlation=2, parent_scope=1, phase='forward', kind='span', begin_ns=0, end_ns=5),
             dict(correlation=7, parent_scope=1, phase='forward', kind='span', begin_ns=5, end_ns=10),
             dict(correlation=8, parent_scope=1, phase='token_readback', kind='wait', begin_ns=40, end_ns=70)]
    request = dict(stream='stream', prompt_sha256='e'*64, sampling_json=json.dumps(dict(sampling={'seed': 7}, penalties={}, drafts=True)), cached=8)
    window = dict(stream='stream', kind='head', offered_rows=5, proposal_nodes=4, connected_accepted_depth=2,
                  kept_path=[0, 2, 4], committed_ids=[101, 37], cached_after=11, pending_after=5, rolled_back_rows=2)
    r = dict(schema='tensorfold.cuda-round.v1', round=1, identity=key, complete=True, problem=None,
             requests=[request], windows=[window], frames=[dict(correlation=1, live_rows=5, windows=1, shape_bucket=8, shape_key=42 | ((1 << 29) if mt2 else 0), mt2=mt2)],
             host=hosts, device=devices, budget=dict(complete_round_wall_ns=wall, functional_return_ns=80, device_envelope_ns=90))
    r['budget'] = m.calculate(r)
    return header, raw, key, r


def lines(f):
    _, raw, key, r = f
    return ['unrelated startup line\n', f'TF_ROUND_IDENTITY {key} {raw}\n', 'TF_ROUND '+json.dumps(r)+'\n']


class Attribution(unittest.TestCase):
    def test_round_and_overlap_have_separate_denominators(self):
        f = fixture()
        data = m.read(lines(f))
        b = data[1][0]['budget']
        self.assertEqual(100, b['complete_round_wall_ns'])
        self.assertEqual(50, b['device_event_span_union_ns'])
        self.assertEqual(70, b['kernel_event_span_sum_ns'])
        self.assertEqual(40, b['device_envelope_gap_ns'])
        self.assertEqual(30, b['host_wait_union_ns'])
        self.assertIn('UNQUALIFIED', m.summary(data)['qmmf_60_percent_gate'])

    def test_event_sum_substitution_is_a_failing_control(self):
        f = fixture()
        f[3]['budget']['complete_round_wall_ns'] = 70
        with self.assertRaises(m.InvalidTrace):
            m.read(lines(f))

    def test_budget_corruption_is_refused(self):
        for field in fixture()[3]['budget']:
            f = fixture()
            if isinstance(f[3]['budget'][field], list):
                f[3]['budget'][field][0] += 1
            else:
                f[3]['budget'][field] += 1
            # The three measured boundaries are inputs; check invalid ordering separately.
            if field in ('complete_round_wall_ns', 'functional_return_ns', 'device_envelope_ns'):
                continue
            with self.subTest(field=field), self.assertRaises(m.InvalidTrace):
                m.read(lines(f))

    def test_unjoined_or_partial_data_never_pass(self):
        changes = [lambda r: r.update(complete=False), lambda r: r.update(problem='lost graph'),
                   lambda r: r.update(device=[]), lambda r: r['device'][0].update(parent_scope=999),
                   lambda r: r['device'][0].update(correlation=999),
                   lambda r: r['device'][0].update(qmmf=[]),
                   lambda r: r['windows'][0].update(connected_accepted_depth=4),
                   lambda r: r['windows'][0].update(rolled_back_rows=0),
                   lambda r: r['frames'][0].update(live_rows=4),
                   lambda r: r['windows'][0].update(committed_ids=[1, 2, 3, 4]),
                   lambda r: r['device'][0].update(end_ns=100)]
        for change in changes:
            f = fixture(); change(f[3])
            with self.subTest(change=change), self.assertRaises(m.InvalidTrace):
                m.read(lines(f))

    def test_missing_header_and_digest_mismatch(self):
        for text in (lines(fixture())[2:], ['TF_ROUND_IDENTITY '+'f'*64+' {}\n'], []):
            with self.assertRaises(m.InvalidTrace):
                m.read(text)

    def test_duplicate_round_or_launch(self):
        f = fixture()
        with self.assertRaises(m.InvalidTrace):
            m.read(lines(f)+[lines(f)[-1]])
        f[3]['device'][1]['correlation'] = 2
        with self.assertRaises(m.InvalidTrace):
            m.read(lines(f))

    def test_ab_compares_actual_ids_and_published_state(self):
        on, off = m.read(lines(fixture(True, 100))), m.read(lines(fixture(False, 102)))
        result = m.compare(on, off)
        self.assertEqual('do not invest in fusion', result['experiment_decision'])
        self.assertEqual('UNQUALIFIED', result['release_gate'])
        for field, bad in [('committed_ids', [101, 38]), ('cached_after', 12), ('pending_after', 6)]:
            wrong = copy.deepcopy(off); wrong[1][0]['windows'][0][field] = bad
            with self.subTest(field=field), self.assertRaises(m.InvalidTrace):
                m.compare(on, wrong)

    def test_ab_rejects_false_mt2_on_and_different_binary(self):
        data = m.read(lines(fixture()))
        with self.assertRaises(m.InvalidTrace):
            m.compare(data, data)
        on = m.read(lines(fixture(True)))
        wrong = copy.deepcopy(data)
        next(iter(wrong[0].values()))['binary_sha256'] = 'f'*64
        with self.assertRaises(m.InvalidTrace):
            m.compare(on, wrong)

    def test_ab_allows_request_uuids_to_change(self):
        on, off = m.read(lines(fixture(True))), m.read(lines(fixture(False)))
        off[1][0]['requests'][0]['stream'] = 'other UUID'
        off[1][0]['windows'][0]['stream'] = 'other UUID'
        self.assertEqual('UNQUALIFIED', m.compare(on, off)['release_gate'])

    def test_forced_rows_are_not_accepted_proposals(self):
        f = fixture(); w = f[3]['windows'][0]
        w.update(kind='forced', proposal_nodes=0, connected_accepted_depth=0)
        self.assertEqual(0, m.summary(m.read(lines(f)))['accepted_depth'])
        w['connected_accepted_depth'] = 2
        with self.assertRaises(m.InvalidTrace):
            m.read(lines(f))

    def test_nan_and_explicit_errors_refused(self):
        for text in (['TF_ROUND_ERROR round=1 budget=lost\n'], ['TF_ROUND {"round":NaN}\n']):
            with self.assertRaises(m.InvalidTrace):
                m.read(text)


class ReviewAB(unittest.TestCase):
    def test_round_device_config_and_frame_mismatches(self):
        on, off = m.read(lines(fixture(True))), m.read(lines(fixture(False)))
        changes = [('round', lambda h, r: r.update(round=2))]
        for key, value in [('driver', 13001), ('sm_major', 11), ('sm_minor', 1), ('checkpoint_config_sha256', '0'*64)]:
            changes.append((key, lambda h, r, k=key, v=value: h.update({k: v})))
        for key, value in [('live_rows', 6), ('windows', 2), ('shape_bucket', 16), ('shape_key', 43)]:
            changes.append((key, lambda h, r, k=key, v=value: r['frames'][0].update({k: v})))
        for key, change in changes:
            wrong = copy.deepcopy(off)
            change(next(iter(wrong[0].values())), wrong[1][0])
            with self.subTest(key=key), self.assertRaises(m.InvalidTrace):
                m.compare(on, wrong)

    def test_only_correlation_and_mt2_bit_may_differ(self):
        on, off = m.read(lines(fixture(True))), m.read(lines(fixture(False)))
        off[1][0]['frames'][0]['correlation'] = 99
        self.assertEqual('UNQUALIFIED', m.compare(on, off)['release_gate'])


if __name__ == '__main__':
    unittest.main()
