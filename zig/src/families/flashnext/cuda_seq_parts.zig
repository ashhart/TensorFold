//! A contiguous sequence whose conv windows, states and tail are views of one reused block (cuda_lane_parts.zig).
const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const wt = @import("cuda_prompt_wt.zig");
const prompt = @import("cuda_prompt.zig");
const Engine = prompt.Engine;
const Seq = prompt.Seq;
const linear_n = wt.linear_n;
const attn_n = wt.attn_n;
const state_n = wt.state_n;
const ple_tail_n = wt.ple_tail_n;
const wide = wt.wide;
const vocab = wt.vocab;
const kv_heads = wt.kv_heads;
const head_dim = wt.head_dim;
const index_dim = wt.index_dim;

const conv_part = (gdn.taps - 1) * gdn.conv_dim * 2;

/// A sequence's fixed-size buffers in one block (newSeqIn): its conv windows, DeltaNet states and n-gram tail.
pub fn seqPartsBytes() usize {
    const a = std.mem.alignForward;
    return linear_n * a(usize, conv_part, 256) + linear_n * a(usize, state_n * 4, 256) + a(usize, ple_tail_n * wide * 2, 256);
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
    if (block.len < seqPartsBytes()) return error.UnexpectedTensor;
    e.round_graph.clear();
    if (cap == 0 or cap > prompt.max_tokens) return error.UnexpectedTensor;
    _ = try prompt.seqBytesPaged(cap, 0);
    var seq: Seq = .{
        .stream = e.stream.*,
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
    try e.stream.d.check(e.stream.d.api.cuMemsetD8Async(block.ptr, 0, seqPartsBytes(), e.stream.handle), "cuMemsetD8Async");
    var at = block.ptr;
    seq.conv = try viewsAt(e.gpa, e.driver, &at, linear_n, conv_part);
    seq.state = try viewsAt(e.gpa, e.driver, &at, linear_n, state_n * 4);
    seq.tail = viewAt(e.driver, &at, ple_tail_n * wide * 2);
    seq.kc = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * kv_heads * head_dim * 2);
    seq.vc = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * kv_heads * head_dim * 2);
    seq.ikc = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * index_dim * 2);
    seq.pooled = try prompt.zeros(e.gpa, e.driver, e.stream.*, attn_n, prompt.pooledBytes(cap));
    return seq;
}
