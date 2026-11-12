#!/usr/bin/env python3
"""Inspect pinned Hugging Face configs and safetensors headers without downloading tensor data."""

import argparse
import collections
import json
import math
import os
import re
import struct
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

MAX_JSON = 16 * 1024 * 1024
MAX_HEADER = 16 * 1024 * 1024


class InspectionError(Exception):
    pass


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name:
        raise InspectionError(f'Unsafe checkpoint file name: {name!r}')
    return name


def read_json_bytes(data, description):
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        raise InspectionError(f'Invalid JSON in {description}: {exc}') from exc


class Remote:
    def __init__(self, repository, revision):
        if not re.fullmatch(r'[\w.-]+/[\w.-]+', repository):
            raise InspectionError('Repository must have owner/name form')
        api = 'https://huggingface.co/api/models/' + repository + '/revision/'
        metadata = self.json_url(api + urllib.parse.quote(revision, safe=''))
        self.revision = metadata.get('sha', '')
        if not re.fullmatch(r'[0-9a-f]{40}', self.revision):
            raise InspectionError('Hugging Face did not resolve a full commit revision')
        self.repository = repository
        self.files = {entry['rfilename'] for entry in metadata.get('siblings', [])}
        self.base = f'https://huggingface.co/{repository}/resolve/{self.revision}/'

    @staticmethod
    def fetch(url, limit, byte_range=None, with_total=False):
        headers = {'User-Agent': 'TensorFold-checkpoint-inspector', 'Accept-Encoding': 'identity'}
        if byte_range is not None:
            start, end = byte_range
            headers['Range'] = f'bytes={start}-{end}'
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
                if byte_range is not None:
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                    if response.status != 206 or not match:
                        raise InspectionError('Server refused an exact HTTP byte range; refusing tensor download')
                    first, last, total = map(int, match.groups())
                    if (first, last) != byte_range or total <= last:
                        raise InspectionError('Server returned an unexpected byte range')
                    if response.headers.get('Content-Encoding', 'identity') != 'identity':
                        raise InspectionError('Compressed range response is not supported')
                length = response.headers.get('Content-Length')
                if length is not None and int(length) > limit:
                    raise InspectionError(f'Response exceeds {limit}-byte body limit')
                data = response.read(limit + 1)
                if len(data) > limit:
                    raise InspectionError(f'Response exceeds {limit}-byte body limit')
                if byte_range is not None and len(data) != byte_range[1] - byte_range[0] + 1:
                    raise InspectionError('Truncated byte-range response')
                return (data, total) if with_total else data
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise InspectionError(f'Cannot fetch {url}: {exc}') from exc

    @classmethod
    def json_url(cls, url):
        return read_json_bytes(cls.fetch(url, MAX_JSON), url)

    def json(self, name):
        return self.json_url(self.base + urllib.parse.quote(safe_name(name), safe='/'))

    def header(self, name):
        url = self.base + urllib.parse.quote(safe_name(name), safe='/')
        prefix, total = self.fetch(url, 8, (0, 7), with_total=True)
        size = header_size(prefix)
        if total < 8 + size:
            raise InspectionError('Safetensors header extends beyond shard size')
        data, second_total = self.fetch(url, size, (8, 7 + size), with_total=True)
        if total != second_total:
            raise InspectionError('Shard size changed between header requests')
        return read_json_bytes(data, name), total - 8 - size


class Local:
    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self.repository = str(self.directory)
        self.revision = None
        self.files = {str(path.relative_to(self.directory)) for path in self.directory.rglob('*.safetensors')}
        if (self.directory / 'model.safetensors.index.json').is_file():
            self.files.add('model.safetensors.index.json')

    def path(self, name):
        path = (self.directory / safe_name(name)).resolve()
        if not path.is_relative_to(self.directory):
            raise InspectionError('Checkpoint file escapes local directory')
        return path

    def json(self, name):
        with self.path(name).open('rb') as stream:
            data = stream.read(MAX_JSON + 1)
        if len(data) > MAX_JSON:
            raise InspectionError('Local JSON exceeds body limit')
        return read_json_bytes(data, name)

    def header(self, name):
        with self.path(name).open('rb') as stream:
            size = header_size(stream.read(8))
            data = stream.read(size)
            total = os.fstat(stream.fileno()).st_size
        if len(data) != size:
            raise InspectionError('Truncated local safetensors header')
        return read_json_bytes(data, name), total - 8 - size


def header_size(prefix):
    if len(prefix) != 8:
        raise InspectionError('Truncated safetensors size prefix')
    size = struct.unpack('<Q', prefix)[0]
    if not 2 <= size <= MAX_HEADER:
        raise InspectionError(f'Invalid or excessive safetensors header size: {size}')
    return size


def validate_tensor(name, tensor, data_length):
    widths = {
        'BOOL': 1, 'U8': 1, 'I8': 1, 'F8_E4M3': 1, 'F8_E5M2': 1,
        'U16': 2, 'I16': 2, 'F16': 2, 'BF16': 2,
        'U32': 4, 'I32': 4, 'F32': 4, 'U64': 8, 'I64': 8, 'F64': 8,
    }
    if not isinstance(tensor, dict) or not all(k in tensor for k in ('dtype', 'shape', 'data_offsets')):
        raise InspectionError(f'Invalid tensor descriptor: {name}')
    shape, offsets = tensor['shape'], tensor['data_offsets']
    if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
        raise InspectionError(f'Invalid tensor shape: {name}')
    if not isinstance(offsets, list) or len(offsets) != 2 or any(type(d) is not int for d in offsets):
        raise InspectionError(f'Invalid tensor offsets: {name}')
    start, end = offsets
    if not 0 <= start <= end <= data_length:
        raise InspectionError(f'Tensor offsets exceed shard data bounds: {name}')
    dtype = tensor['dtype']
    if not isinstance(dtype, str) or dtype not in widths:
        raise InspectionError(f'Unsupported safetensors dtype: {name}')
    if math.prod(shape) * widths[dtype] != end - start:
        raise InspectionError(f'Tensor shape/dtype byte count disagrees with offsets: {name}')


def inspect(source):
    config = source.json('config.json')
    index = None
    if 'model.safetensors.index.json' in source.files:
        index = source.json('model.safetensors.index.json')
        shard_names = sorted(set(index['weight_map'].values()))
    else:
        shard_names = sorted(name for name in source.files if name.endswith('.safetensors'))
    if not shard_names:
        raise InspectionError('No safetensors files found')
    tensors = {}
    shard_data_lengths = {}
    for shard in shard_names:
        header, data_length = source.header(shard)
        shard_data_lengths[shard] = data_length
        ranges = []
        for name, tensor in header.items():
            if name == '__metadata__':
                continue
            if name in tensors:
                raise InspectionError(f'Duplicate tensor: {name}')
            validate_tensor(name, tensor, data_length)
            start, end = tensor['data_offsets']
            if start != end:
                ranges.append((start, end, name))
            tensors[name] = dict(tensor, shard=shard)
        previous_end = 0
        for start, end, name in sorted(ranges):
            if start < previous_end:
                raise InspectionError(f'Overlapping tensor data: {name}')
            previous_end = end
    if index is not None:
        if set(index['weight_map']) != set(tensors):
            raise InspectionError('Tensor index and shard headers disagree')
        if any(index['weight_map'][name] != tensor['shard'] for name, tensor in tensors.items()):
            raise InspectionError('Tensor index points to the wrong shard')
    return {
        'repository': source.repository,
        'revision': source.revision,
        'config': config,
        'tensor_count': len(tensors),
        'dtype_counts': dict(collections.Counter(t['dtype'] for t in tensors.values())),
        'quantized_module_count': sum(name.endswith('.scales') for name in tensors),
        'shards': shard_names,
        'shard_data_lengths': shard_data_lengths,
        'tensors': tensors,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('repository', nargs='?', help='Hugging Face owner/name')
    parser.add_argument('--revision', default='main')
    parser.add_argument('--local-dir', type=Path)
    parser.add_argument('--output', type=Path, help='Write complete verified metadata as JSON')
    args = parser.parse_args()
    if bool(args.repository) == bool(args.local_dir):
        parser.error('Choose exactly one repository or --local-dir')
    try:
        source = Local(args.local_dir) if args.local_dir else Remote(args.repository, args.revision)
        result = inspect(source)
        if args.output:
            args.output.write_text(json.dumps(result, indent=2) + '\n')
        summary = {key: value for key, value in result.items() if key not in ('tensors', 'config')}
        summary['quantization'] = result['config'].get('quantization', result['config'].get('quantization_config'))
        print(json.dumps(summary, indent=2))
    except (InspectionError, OSError, KeyError, TypeError) as exc:
        print(f'Checkpoint inspection failed: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
