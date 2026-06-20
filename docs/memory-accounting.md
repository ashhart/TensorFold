# Memory Accounting

TensorFold uses explicit memory accounting because low-memory inference is easy
to misread.

## Numbers To Track

- **Process RSS:** memory the operating system reports for the process.
- **Resident weight bytes:** model weight bytes TensorFold intentionally keeps
  resident.
- **MLX/Metal peak:** allocations held by MLX and the Metal backend.
- **Pending read bytes:** in-flight pack or shard reads that have not been
  released yet.
- **KV/cache bytes:** attention and recurrent state that grows with prompt and
  generated token count.

## Why RSS Can Exceed The Resident Budget

The resident budget controls TensorFold's planned weight residency. RSS can also
include memory maps, allocator caches, Python objects, temporary expert tables,
pending read buffers, KV cache, and OS page-cache effects.

For tight profiles, use:

```bash
tensorfold serve /path/to/model \
  --resident-budget 2GiB \
  --loader-backend native \
  --pin-policy phase
```

## Comparing Runs

Compare warm decode to warm decode, or full-load session to full-load session.
Do not compare a native run with model loading outside the timer against a
streaming run that includes first-touch reads inside the timed region.

Good benchmark reports include:

- model name and quantization
- prompt token count
- generated token count
- tok/s
- RSS after run
- resident peak
- MLX/Metal peak
- bytes read per token
- exactness status
