const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const pages = @import("cuda_pages.zig");
const kvpage = @import("exl3/kv/cuda_exl3_kvpage.zig");
const weights = @import("cuda_prompt_wt.zig");
const prompt = @import("cuda_prompt.zig");

const linear_n = weights.linear_n;
const attn_n = weights.attn_n;
const wide = weights.wide;
const vocab = weights.vocab;
const head_dim = weights.head_dim;
const index_dim = weights.index_dim;
const ple_tail_n = weights.ple_tail_n;
const state_n = weights.state_n;
const kv_heads = weights.kv_heads;
const max_tokens = prompt.max_tokens;
const zeros = prompt.zeros;
const freeOwned = prompt.freeOwned;
const pooledBytes = prompt.pooledBytes;
const seqBytesPaged = prompt.seqBytesPaged;

pub fn newPlacedSeq(engine: *prompt.Engine, cap: usize) !prompt.Seq {
    var sequence = try @import("exl3/kv/cuda_exl3_kv.zig").newSeq(engine, prompt.Seq, cap);
    sequence.recycle_owner = engine;
    return sequence;
}

pub fn deinit(sequence: *prompt.Seq, allocator: std.mem.Allocator) void {
    if (@import("exl3/offload/cuda_exl3_recycle.zig").giveSeq(sequence)) return;
    if (sequence.stream) |stream| @import("cuda").completion.drain(stream.d, stream, "Flash Next sequence work did not drain before release");
    if (sequence.paged != null or sequence.invalidated) sequence.tail.d.synchronize() catch @panic("sequence consumers did not drain before freeing allocations");
    @import("exl3/graph/cuda_exl3_graph.zig").releaseSeq(sequence);
    if (sequence.media) |*media| media.deinit();
    @import("exl3/kv/cuda_exl3_kv.zig").deinitSeq(allocator, &sequence.kv8, &sequence.ikc_host);
    for ([_][]@import("cuda").DeviceBuffer{ sequence.conv, sequence.state, sequence.kc, sequence.vc, sequence.ikc, sequence.pooled }) |buffers| {
        for (buffers) |*buffer| buffer.free();
        allocator.free(buffers);
    }
    if (sequence.tail.ptr != 0) sequence.tail.free();
    if (sequence.paged) |pager| pager.destroy();
    if (sequence.ptable) |*table| table.deinit(allocator);
    allocator.free(sequence.logits);
}

/// A fresh cache whose attention caches are `given` pages (owned by the cache from here), else contiguous.
pub fn newSeqOn(e: *prompt.Engine, cap: usize, given: ?pages.Paged) !prompt.Seq {
    if (given != null) return newSeqPageTable(e, cap, given);
    e.round_graph.clear();
    if (cap == 0) return error.UnexpectedTensor;
    if (e.page_config) |cfg| if (cfg.shared_lanes != 0) {
        const arena = e.shared_arena orelse return error.SharedKvArenaRequired;
        if (cap != arena.layout.options.cap) return error.SharedKvCommonCap;
    };
    if (try @import("exl3/offload/cuda_exl3_recycle.zig").takeSeq(e, cap)) |kept| return kept;
    if (@import("exl3/kv/cuda_exl3_kv.zig").of(e).placed()) return newPlacedSeq(e, cap);
    if (e.page_config == null) return newSeqPageTable(e, cap, null); // contiguous caches: upstream's flow, which drains before it frees
    const logits = try e.gpa.alloc(u8, vocab * 2);
    errdefer e.gpa.free(logits);
    const conv = try zeros(e.gpa, e.driver, e.stream.*, linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
    errdefer freeOwned(e.gpa, conv);
    const state = try zeros(e.gpa, e.driver, e.stream.*, linear_n, state_n * 4);
    errdefer freeOwned(e.gpa, state);
    // K/V paged to host memory (TF_FLASHNEXT_KVPAGE): kc/vc stay one row (cuda_exl3_kvpage.zig holds the rest)
    const paged = if (e.page_config) |cfg| if (cfg.shared_lanes != 0) try kvpage.Paged.initShared(e.gpa, e.driver, cap, cfg.forChunk(e.chunk), e.shared_arena orelse return error.SharedKvArenaRequired) else try kvpage.Paged.init(e.gpa, e.driver, cap, cfg.forChunk(e.chunk)) else null;
    errdefer if (paged) |pg| pg.destroy();
    const kc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, (if (paged != null) 1 else cap) * kv_heads * head_dim * 2);
    errdefer freeOwned(e.gpa, kc);
    const vc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, (if (paged != null) 1 else cap) * kv_heads * head_dim * 2);
    errdefer freeOwned(e.gpa, vc);
    const ikc = try zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * index_dim * 2);
    errdefer freeOwned(e.gpa, ikc);
    const pooled = try zeros(e.gpa, e.driver, e.stream.*, attn_n, pooledBytes(cap));
    errdefer freeOwned(e.gpa, pooled);
    var tail = try cuda.DeviceBuffer.alloc(e.driver, ple_tail_n * wide * 2);
    errdefer tail.free();
    try tail.fill8(0, e.stream.handle);
    return .{
        .stream = e.stream.*,
        .conv = conv,
        .state = state,
        .kc = kc,
        .vc = vc,
        .ikc = ikc,
        .pooled = pooled,
        .tail = tail,
        .history = .{ e.eos, e.eos },
        .fed = 0,
        .logits = logits,
        .cap = cap,
        .paged = paged,
        .recycle_owner = e,
    };
}

/// A fresh cache whose attention caches are the given page table's pages (owned by the cache from here).
fn newSeqPageTable(e: *prompt.Engine, cap: usize, given: ?pages.Paged) !prompt.Seq {
    var table = given;
    errdefer if (table) |*p| p.deinit(e.gpa);
    e.round_graph.clear();
    if (cap == 0 or cap > max_tokens) return error.UnexpectedTensor;
    _ = try seqBytesPaged(cap, if (table) |p| p.tokens else 0);
    var seq: prompt.Seq = .{
        .stream = e.stream.*,
        .ptable = table,
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
        .recycle_owner = e,
    };
    table = null; // seq owns the pages from here
    // One rollback owns every published array, and drains before releasing any destination.
    errdefer seq.deinit(e.gpa);
    seq.logits = try e.gpa.alloc(u8, vocab * 2);
    seq.conv = try zeros(e.gpa, e.driver, e.stream.*, linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
    seq.state = try zeros(e.gpa, e.driver, e.stream.*, linear_n, state_n * 4);
    seq.kc = if (seq.ptable) |*p| try p.views(e.gpa, e.driver, .key) else try zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * kv_heads * head_dim * 2);
    seq.vc = if (seq.ptable) |*p| try p.views(e.gpa, e.driver, .value) else try zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * kv_heads * head_dim * 2);
    seq.ikc = if (seq.ptable) |*p| try p.views(e.gpa, e.driver, .index) else try zeros(e.gpa, e.driver, e.stream.*, attn_n, cap * index_dim * 2);
    seq.pooled = if (seq.ptable) |*p| try p.views(e.gpa, e.driver, .pooled) else try zeros(e.gpa, e.driver, e.stream.*, attn_n, pooledBytes(cap));
    seq.tail = try cuda.DeviceBuffer.alloc(e.driver, ple_tail_n * wide * 2);
    try seq.tail.fill8(0, e.stream.handle);
    return seq;
}
