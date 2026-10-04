"""Keep raw-bit failures visible while independent operator cells continue."""

import json
from pathlib import Path


class RawCells:
    def __init__(self, out, receipt):
        self.out, self.receipt = Path(out), receipt
        self.receipt.setdefault('cells', [])

    def __call__(self, name, actual, expected, status=0):
        import torch

        if status:
            raise RuntimeError(f'{name}: native CUDA status {status}')
        if actual.is_cuda or expected.is_cuda:
            torch.cuda.current_stream().synchronize()
        got = actual.contiguous().reshape(-1).view(torch.uint8)
        want = expected.contiguous().reshape(-1).view(torch.uint8)
        if got.shape != want.shape:
            raise AssertionError(f'{name}: native byte extent differs from Torch')
        different = int((got != want).sum().item())
        cell = {'id': name, 'shape': list(actual.shape), 'dtype': str(actual.dtype),
                'bytes': got.numel(), 'different_bytes': different, 'exact': different == 0}
        if different:
            cell['first_different_byte'] = int(torch.nonzero(got != want)[0].item())
        self.receipt['cells'].append(cell)
        self.write()
        print(f'CELL {name} exact={cell["exact"]} different_bytes={different}', flush=True)

    def write(self):
        (self.out / 'receipt.json').write_text(json.dumps(self.receipt, indent=2) + '\n')

    def finish(self, started):
        import time
        import torch

        self.receipt.update(exact=all(cell['exact'] for cell in self.receipt['cells']),
                            cells_checked=len(self.receipt['cells']), elapsed_s=time.monotonic() - started,
                            peak_allocated_bytes=torch.cuda.max_memory_allocated())
        self.write()
        if not self.receipt['exact']:
            raise AssertionError('Native operator bytes differ from Torch; all cell receipts retained')
