//! Dynamic KV paging for long contexts (docs/plans/exl3-kv-paging.md): a sequence's full K/V of the 12 attention

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const parts = @import("../../cuda_batch_state.zig");
pub const shared_contract = @import("../offload/exl3_shared_kv.zig");
pub const shared_device = @import("../offload/cuda_exl3_shared_kv.zig");
const configs = @import("cuda_exl3_kvpage_config.zig");
const traces = @import("cuda_exl3_kvpage_trace.zig");
const snaps = @import("cuda_exl3_kvpage_snapshot.zig");
const paged = @import("cuda_exl3_kvpage_paged.zig");

const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const att_rows = parts.att_rows;

const cdiv = configs.cdiv;
const cint = configs.cint;

test "exl3: shared KV host ownership contracts" {
    _ = shared_contract;
    _ = shared_device;
}

/// The active trace, if any (a test sets it; the pass reads it).
pub var trace: ?*Trace = null;

/// Whether attnRows runs here: a trace is on, or the sequence's K/V are paged.
pub fn takes(seq: anytype) bool {
    return trace != null or seq.paged != null;
}

/// cuda_batch.zig attnRows (rope, cache writes, pooling, then each 256-row block's scores, selection and attention)
pub fn attnRows(p: anytype, lw: anytype, ai: usize, pa: u64, n: usize, at: anytype) !void {
    if (p.seq.media != null) return error.KvPageMedia;
    if (p.seq.paged) |pg| return pg.attnRows(p, lw, ai, pa, n, at);
    const s = p.s;
    const kc = p.seq.kc[ai].ptr;
    const vc = p.seq.vc[ai].ptr;
    const ikc = p.seq.ikc[ai].ptr;
    const pooled = p.seq.pooled[ai].ptr;
    try p.tri.attnPrep(pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kc, vc, at.iq, ikc, s.dummy, s.dummy, wt.eps, n, 0, .{});
    try p.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, wt.eps, n, .{});
    const nb = parts.scoreStride(p.seq.cap);
    var r0: usize = 0;
    var b: usize = 0;
    while (r0 < n) : ({
        r0 += att_rows;
        b += 1;
    }) {
        const m = @min(att_rows, n - r0);
        const ends = p.pos0 + r0 + m;
        if (ends > p.seq.cap) return error.UnexpectedTensor;
        const pos = at.pos + b * 16;
        try select(p, at.iq + r0 * wt.index_heads * wt.index_dim * 2, pooled, pos, nb, m, ends);
        if (trace) |t| try t.record(p.stream, ai, p.pos0 + r0, m, p.lanes.len > 0, s.qids, s.qnk, s.qspr);
        const chunks = cdiv(@min(ends, wt.dense_keys), 512);
        try p.tri.attnChunks(at.q + r0 * wt.q_heads * head_dim * 2, kc, vc, pos, s.po, s.pm, s.pl, s.qids, s.qnk, s.qspr, m, kv_heads, chunks, true, .{});
        try p.tri.attnMerge(s.po, s.pm, s.pl, pos, at.ao + r0 * wt.out_k * 2, s.qnk, s.qspr, m, kv_heads, true);
    }
}

/// Block scores (rows past `top` complete blocks) and fn_qsa_select into s.qids / qnk / qspr, as upstream.
pub fn select(p: anytype, iq: u64, pooled: u64, pos: u64, nb: usize, m: usize, ends: usize) !void {
    const s = p.s;
    if (ends / ratio > top) try p.tri.qsaScores(iq, pooled, pos, p.seq.scores, nb, m, @min(nb, cdiv(ends, ratio)), .{});
    var a: cuda.Args = .{};
    a.add(p.seq.scores);
    a.add(pos);
    a.add(s.qids);
    a.add(s.qnk);
    a.add(s.qspr);
    for ([_]usize{ nb, idw, top, ratio }) |v| a.add(cint(v));
    try cuda.launch.launch(p.k.select, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 1024 } }, p.stream, &a);
}

/// A test's override of the environment (null: read it).
pub var force: ?Config = null;
pub var force_off: bool = false;

fn envUsize(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

pub fn configFromEnv() ?Config {
    if (force_off) return null;
    if (force) |c| return c;
    const v = std.c.getenv("TF_FLASHNEXT_KVPAGE") orelse return null;
    if (!std.mem.eql(u8, std.mem.span(v), "1")) return null;
    const d: Config = .{};
    var rows = d.rows;
    if (std.mem.eql(u8, std.mem.span(std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL") orelse "1"), "1")) {
        rows = @max(rows, envUsize("TF_FLASHNEXT_EXL3_PREFILL_ROWS", 4096));
    }
    const shared = std.mem.eql(u8, std.mem.span(std.c.getenv("TF_FLASHNEXT_KVPAGE_SHARED") orelse "0"), "1");
    const shared_lanes = if (shared) envUsize("TF_FLASHNEXT_KVPAGE_SHARED_LANES", 8) else 0;
    return .{ .window = envUsize("TF_FLASHNEXT_KVPAGE_WINDOW", d.window), .slots = d.slots, .rows = rows, .shared_lanes = if (shared and shared_lanes == 0) std.math.maxInt(usize) else shared_lanes };
}

/// A new sequence's paged store when the environment (or a test) turns paging on, else null.
pub fn forSeq(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, chunk: usize) !?*Paged {
    const cfg = configFromEnv() orelse return null;
    return try Paged.init(gpa, d, cap, cfg.forChunk(chunk));
}

pub const row_bytes = configs.row_bytes;
pub const Config = configs.Config;
pub const configIdentity = configs.configIdentity;
pub const admissionRoom = configs.admissionRoom;
pub const pagingRows = configs.pagingRows;
pub const windowPlan = configs.windowPlan;
pub const deviceBytesFor = configs.deviceBytesFor;
pub const sharedBytesFor = configs.sharedBytesFor;
pub const ownedSnapshotBytes = configs.ownedSnapshotBytes;
pub const Trace = traces.Trace;
pub const SnapshotLayout = snaps.SnapshotLayout;
pub const Snapshot = snaps.Snapshot;
pub const Stats = snaps.Stats;
pub const Paged = paged.Paged;
