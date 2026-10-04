"""Record deployed ATen sort, floating scan and sum choices before replacing them."""

import json
from pathlib import Path

import torch


def probe_nemotron_nucleus_dispatch(config, out):
    out = Path(out)
    vocab = int(config['vocab_size'])
    ids = torch.arange(vocab, device='cuda', dtype=torch.int64)
    values = (((ids * 47 + 11) % 257 - 128).float() / 13).bfloat16().double()[None]
    reference_rank = torch.argsort(values, descending=True, stable=True)
    reference = torch.gather(values, 1, reference_rank)
    probabilities = torch.exp(reference - reference[:, :1])
    first_scan = probabilities.cumsum(dim=-1)
    first_sum = probabilities.sum(dim=-1, keepdim=True)
    records = []
    for rows in (1, 2, 4, 8, 16):
        scores = values.expand(rows, -1).contiguous()
        p = probabilities.expand(rows, -1).contiguous()
        torch.argsort(scores, descending=True, stable=True)
        p.cumsum(dim=-1)
        p.sum(dim=-1, keepdim=True)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA], record_shapes=True) as profiler:
            with torch.profiler.record_function('nemotron/sort/' + str(rows)):
                order = torch.argsort(scores, descending=True, stable=True)
            with torch.profiler.record_function('nemotron/scan/' + str(rows)):
                scan = p.cumsum(dim=-1)
            with torch.profiler.record_function('nemotron/sum/' + str(rows)):
                sums = p.sum(dim=-1, keepdim=True)
            torch.cuda.synchronize()
        trace = out / ('nucleus-dispatch-' + str(rows) + '.json')
        profiler.export_chrome_trace(str(trace))
        data = json.loads(trace.read_text())
        kernels = [{'name': event['name'], 'duration_us': event.get('dur', 0)}
                   for event in data.get('traceEvents', []) if event.get('cat') == 'kernel']
        scan_bits = scan[0].view(torch.int64) != first_scan[0].view(torch.int64)
        sum_bits = sums[0].view(torch.int64) != first_sum[0].view(torch.int64)
        records.append({'rows': rows, 'vocab': vocab, 'dtype': 'float64', 'kernels': kernels,
                        'sort_ids_equal_to_one_row': bool(torch.equal(order[0], reference_rank[0])),
                        'scan_different_words_from_one_row': int(scan_bits.sum().item()),
                        'sum_different_words_from_one_row': int(sum_bits.sum().item())})
    result = {'oracle_torch': torch.__version__, 'git_version': torch.version.git_version,
              'cuda': torch.version.cuda, 'model_loaded': False, 'records': records,
              'statement': 'Dispatch measurement only, not native scan/sort/sum qualification; no code copied'}
    (out / 'nucleus-dispatch.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
