//! A contiguous sequence whose conv windows, states and tail are views of one reused block (cuda_lane_parts.zig).
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const prompt = @import("cuda_prompt.zig");
const Engine = prompt.Engine;
const Seq = prompt.Seq;
const linear_n = wt.linear_n;
const attn_n = wt.attn_n;
const ple_tail_n = wt.ple_tail_n;
const wide = wt.wide;
const vocab = wt.vocab;
const sizes = @import("cuda_seq_bytes.zig");
const Heads = @import("cuda_heads.zig").Heads;

/// A sequence's fixed-size buffers in one block (newSeqIn) at heads `h`: its conv windows, states and n-gram tail.
pub fn seqPartsBytes(h: Heads) usize {
    const a = std.mem.alignForward;
    return linear_n * a(usize, sizes.convBytes(h), 256) + linear_n * a(usize, sizes.stateBytes(h), 256) + a(usize, ple_tail_n * wide * 2, 256);
}

/// `n` borrowed views of `len` bytes from `p` on, 256-byte aligned (the block stays its owner's).
fn viewsAt(gpa: std.mem.Allocator, d: *const cuda.Driver, p: *u64, n: usize, len: usize) ![]cuda.DeviceBuffer {
    const bs = try gpa.alloc(cuda.DeviceBuffer, n);
    for (bs) |*b| b.* = viewAt(d, p, len);
    return bs;
}

fn viewAt(d: *const cuda.Driver, p: *u64, len: usize) cuda.DeviceBuffer {
    const b: cuda.DeviceBuffer = .{ .d = d, .ptr = p.*, .len = len, .borrowed = true };
    p.* += std.mem.alignForward(usize, len, 256);
    return b;
}

/// A contiguous sequence whose conv windows, states and tail are views of `block` (zeroed here); its caches its own.
pub fn newSeqIn(e: *Engine, cap: usize, block: cuda.DeviceBuffer) !Seq {
    if (block.len < seqPartsBytes(e.heads)) return error.UnexpectedTensor;
    e.round_graph.clear();
    if (cap == 0 or cap > prompt.max_tokens) return error.UnexpectedTensor;
    _ = try prompt.seqBytesPaged(cap, 0, e.heads);
    var seq: Seq = .{
        .gen = @import("cuda_window_graph.zig").nextGen(),
        .stream = e.stream.*,
        .heads = e.heads,
        .conv = &.{},
        .state = &.{},
        .kc = &.{},
        .vc = &.{},
        .ikc = &.{},
        .pooled = &.{},
        .tail = .{ .d = e.driver, .ptr = 0, .len = 0 },
        .history = .{ e.eos, e.eos },
        .fed = 0,
        .logits = &.{},
        .cap = cap,
    };
    errdefer seq.deinit(e.gpa);
    seq.logits = try e.gpa.alloc(u8, vocab * 2);
    try e.stream.d.check(e.stream.d.api.cuMemsetD8Async(block.ptr, 0, seqPartsBytes(e.heads), e.stream.handle), "cuMemsetD8Async");
    var at = block.ptr;
    seq.conv = try viewsAt(e.gpa, e.driver, &at, linear_n, sizes.convBytes(e.heads));
    seq.state = try viewsAt(e.gpa, e.driver, &at, linear_n, sizes.stateBytes(e.heads));
    seq.tail = viewAt(e.driver, &at, ple_tail_n * wide * 2);
    // TF_CUDA_CARVEOUT: the planes the load plan counts into the room are claimed here, as newSeqOn does.
    if (e.carve) |c| try c.take(&seq.carved, cap);
    seq.kc = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, sizes.kvBytes(cap, e.heads), seq.carved.of(.key));
    seq.vc = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, sizes.kvBytes(cap, e.heads), seq.carved.of(.value));
    seq.ikc = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, sizes.indexBytes(cap), seq.carved.of(.index));
    seq.pooled = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, sizes.pooledBytes(cap), seq.carved.of(.pooled));
    return seq;
}
