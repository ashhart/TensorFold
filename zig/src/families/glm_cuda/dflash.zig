//! DFlash2 drafts (TF_GLM_DRAFTER=dflash2): incoai/GLM-5.3-Flash-DFlash2 (CC BY-NC-ND 4.0, unmodified), z-lab (MIT).

const std = @import("std");
const cuda = @import("cuda");
const kern = @import("kern.zig");
const forward = @import("forward.zig");
const blocks = @import("blocks.zig");
const draw_mod = @import("draw.zig");
const weights_mod = @import("weights.zig");
const dl = @import("dflash_load.zig");
const ds = @import("dflash_select.zig");

pub const TAPS = [_]usize{ 5, 14, 24, 33, 42 };
pub const BLOCK: usize = 8;
pub const MASK: u32 = 154856;
pub const WIN: i64 = 2048;
pub const LAYERS: usize = 5;
pub const D: usize = 4096;
pub const KVW: usize = 1024;
pub const QKV: usize = 4096 + 2 * KVW;
pub const FF: usize = 12288;
pub const RANK: usize = 256;
pub const TOPK: usize = 16;
const THETA: f32 = 10000;
const EPS: f32 = 1e-5;
const SCALE: f32 = 0.08838834764831845; // 128^-0.5
pub const VOCAB: usize = 154880;
pub const MAX_M: usize = forward.MAX_STREAMS * BLOCK; // block rows at most
pub const MAX_CTX: usize = @intCast(forward.PREFILL_ROWS_MAX); // context rows one ingest takes at most
pub const CW: usize = 2 * TOPK; // a row's gathered ints: 16 ids, then 16 values' bits
pub const CHUNKS: usize = 19; // 4096-column chunks of a rank's head rows at most
pub const QH: usize = 32;
pub const HD: usize = 128;
pub const CH: usize = 16; // key chunks of the chunked attention (df_attn_part)

fn cdiv(a: i64, b: i64) i64 {
    return @divFloor(a + b - 1, b);
}

/// Device bytes a stream's context ring takes (K and V of every layer).
pub const ring_bytes: u64 = LAYERS * 2 * @as(u64, @intCast(WIN)) * KVW * 2;

pub const B16 = kern.B16;
const LayerW = struct {
    in_norm: u64,
    post_norm: u64,
    qn: u64,
    kn: u64,
    a_base: u64,
    m_base: u64,
    qkv: B16,
    kv: B16,
    o: B16,
    gu: B16,
    down: B16,
    a_proj: B16,
    m_proj: B16,
};

/// One stream's draft request: ``n`` last-verify rows into its context first (0: prefill's), the anchor, ``out``.
pub const Req = struct { s: *forward.Seq, n: usize, anchor: u32, out: ?[]u32 = null, probs: ?[]f64 = null, drafts: bool = true };

pub const Drafter = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    weights: cuda.DeviceBuffer,
    bits: u8,
    layers: [LAYERS]LayerW = undefined,
    fc: B16 = undefined,
    hproj: B16 = undefined,
    hnorm: u64 = 0,
    norm: u64 = 0,
    pred: []u16, // the selector's codebooks, bf16 [VOCAB][RANK] (host: it reads 16 rows a draft)
    succ: []u16,
    scratch: cuda.DeviceBuffer,
    // block buffers [MAX_M][..]
    x: u64 = 0,
    y: u64 = 0,
    y2: u64 = 0,
    dyn: u64 = 0,
    qkv: u64 = 0,
    q: u64 = 0,
    kb: u64 = 0,
    vb: u64 = 0,
    att: u64 = 0,
    a: u64 = 0,
    gu: u64 = 0,
    act: u64 = 0,
    hn: u64 = 0,
    g: u64 = 0,
    // context buffers [MAX_CTX][..]
    cin: u64 = 0,
    c1: u64 = 0,
    ctx: u64 = 0,
    ckv: u64 = 0,
    // the target's taps: a verify step's rows and a prompt chunk's, [rows][5 D]
    taps_step: u64 = 0,
    taps_prefill: u64 = 0,
    // tables
    ids: u64 = 0,
    bpos: u64 = 0,
    rings: u64 = 0,
    starts: u64 = 0,
    froms: u64 = 0,
    cpos: u64 = 0,
    crings: u64 = 0,
    invt: u64 = 0,
    tk: u64 = 0,
    gtk: u64 = 0,
    pv: u64 = 0,
    pi: u64 = 0,
    pm: u64 = 0,
    pl: u64 = 0,
    pacc: u64 = 0, // the chunked attention's partials
    attn_chunked: bool = true, // TF_GLM_DFLASH2_ATTN=serial: df_attn's one block a kv head (the chunked kernel otherwise)
    h_ids: []i32,
    h_bpos: []i32,
    h_rings: []u64,
    h_starts: []i64,
    h_froms: []i64,
    h_cpos: []i64,
    h_crings: []u64,
    cand: []i32, // both ranks' gathered candidates, [2][MAX_M][W]
    gh: []f32, // the selector's hidden projections [MAX_M][RANK]
    forward_ms: f64 = 0,
    calls: u64 = 0,
    lost: u64 = 0, // streams whose last verify rows were gone at their draft (their context restarts)
    prof: bool = false, // TF_GLM_DFLASH2_PROF: each phase synchronized and timed (no graphs)
    graphs: [forward.MAX_STREAMS + 1]?cuda.graph.Exec = @splat(null), // the block's pass a stream count
    dump: ?[*:0]const u8 = null, // TF_GLM_DFLASH2_DUMP (development): a one-chunk prompt's taps and its first round, written there
    dump_state: u8 = 0, // 0: waiting for a prompt, 1: its taps written, 2: done
    dump_round: u32 = 0, // the dumped stream's rounds so far, and the one written (TF_GLM_DFLASH2_DUMP_ROUND, 10)
    dump_at: u32 = 10,
    tap_input: bool = false,
    tap_s0: bool = false,
    full_head: bool = false, // TF_GLM_DFLASH2_HEAD=full: the candidates from the verify head (8-bit) instead of the drafts' (4-bit)
    temp: f64 = 1.0, // TF_GLM_DFLASH2_TEMP: the selector's softmax temperature for the drafts' chances (the lane core's depth rule)
    phase_ms: [5]f64 = @splat(0), // context, layers, head and gather, selector, tables
    prof_fine: bool = false, // TF_GLM_DFLASH2_PROF=2: every kernel kind of the block pass synchronized and timed
    split: bool = false, // the MLP linears split across the two ranks (gate and up by columns, down by K; TF_GLM_DFLASH2_SPLIT=0: whole on each)
    ff: usize = FF, // the MLP width this rank holds
    fine_ms: [FINE.len]f64 = @splat(0),
    fine_t: std.Io.Timestamp = undefined,

    pub fn deinit(dr: *Drafter) void {
        for (&dr.graphs) |*g| if (g.*) |*ex| ex.deinit();
        dr.scratch.free();
        dr.weights.free();
        const g = dr.gpa;
        g.free(dr.pred);
        g.free(dr.succ);
        g.free(dr.h_ids);
        g.free(dr.h_bpos);
        g.free(dr.h_rings);
        g.free(dr.h_starts);
        g.free(dr.h_froms);
        g.free(dr.h_cpos);
        g.free(dr.h_crings);
        g.free(dr.cand);
        g.free(dr.gh);
        g.destroy(dr);
    }
};

/// The block pass's kernel kinds (TF_GLM_DFLASH2_PROF=2), in launch order.
pub const FINE = [_][]const u8{ "embed", "rmsnorm", "a_proj", "conv", "qkv", "block_qkv", "attn", "o_proj", "m_proj", "gate_up", "swiglu", "down", "head_mm", "top16", "cand_gather", "hproj", "down_gather", "down_sum" };

fn fine(dr: *Drafter, e: *forward.Engine, which: usize) !void {
    if (!dr.prof_fine) return;
    try e.k.stream.synchronize();
    const now = std.Io.Timestamp.now(e.io, .awake);
    dr.fine_ms[which] += @as(f64, @floatFromInt(dr.fine_t.durationTo(now).toNanoseconds())) / 1e6;
    dr.fine_t = now;
}

/// The kernel kinds' mean ms a round, logged (TF_GLM_DFLASH2_PROF=2) and cleared.
pub fn logFine(dr: *Drafter, rounds: u64) void {
    if (!dr.prof_fine) return;
    var buf: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    const n: f64 = @floatFromInt(@max(rounds, 1));
    for (FINE, dr.fine_ms) |name, ms| w.print(" {s} {d:.3}", .{ name, ms / n }) catch break;
    std.log.info("glm dflash2 kernels (ms a round):{s}", .{w.buffered()});
    dr.fine_ms = @splat(0);
}

/// out[m, n] = x[m, k] . w^T through the engine's own matmul (the drafter's linears packed as the dense layers are).
fn lin(e: *forward.Engine, x: u64, w: B16, out: u64, m: usize, f32_out: bool) !void {
    try kern.mm(&e.k, x, w.k, w, out, f32_out, @intCast(m), e.pbuf.sk); // the prompt buffers' split-K scratch: no prefill runs meanwhile
}

// -- loading: dflash_load.zig ---------------------------------------------------------------------------------------

pub const load = dl.load;

// -- the target's side ---------------------------------------------------------------------------------------------

/// After target layer ``li``: its rows' 4-stream means into the taps; TF_GLM_DFLASH2_TAP=mean|input|s0 (development).
pub fn tap(dr: *const Drafter, k: *kern.K, x: u64, taps: u64, li: usize, R: i64) !void {
    const at = li + @intFromBool(dr.tap_input);
    const slot: i32 = for (TAPS, 0..) |t, i| {
        if (t == at) break @intCast(i);
    } else return;
    var a: cuda.Args = .{};
    a.add(x);
    a.add(taps);
    a.add(slot);
    a.add(@as(i32, @intFromBool(dr.tap_s0)));
    try k.go(try k.ext(.dflash, "df_tap"), .{ .x = @intCast(R), .y = @intCast(D / 1024) }, 256, 0, &a);
}

pub fn isTap(dr: *const Drafter, li: usize) bool {
    const at = li + @intFromBool(dr.tap_input);
    for (TAPS) |t| if (t == at) return true;
    return false;
}

/// Context runs: ``n`` tap rows from ``taps`` (consecutive) at positions pos0 .. pos0 + n, into ``s``'s ring.
pub const Run = struct { s: *forward.Seq, taps: u64, n: usize, pos0: i64 };

/// The runs' rows through fc, the norm and each layer's k, v into their rings; TF_GLM_DFLASH2_DUMP dumps one prompt.
pub fn dumpPrompt(dr: *Drafter, e: *forward.Engine, n: usize) !void {
    if (dr.dump == null or dr.dump_state == 2 or n > MAX_CTX) return; // the latest prompt until its first round is written
    try e.k.stream.synchronize();
    const buf = try dr.gpa.alloc(u8, n * 5 * D * 2);
    defer dr.gpa.free(buf);
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(buf.ptr), dr.taps_prefill, buf.len), "dump taps");
    try writeFile(dr, "taps.bin", buf);
    try writeFile(dr, "ingest.bin", "");
    dr.dump_state = 1;
    dr.dump_round = 0;
}

/// While dumping: the ingest's rows (their positions, then their taps) appended to ingest.bin.
fn dumpIngest(dr: *Drafter, e: *forward.Engine, N: usize) !void {
    try e.k.stream.synchronize();
    const buf = try dr.gpa.alloc(u8, N * 5 * D * 2);
    defer dr.gpa.free(buf);
    try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(buf.ptr), dr.cin, buf.len), "dump ingest");
    var pb: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pb, "{s}/ingest.bin", .{std.mem.span(dr.dump.?)}, 0);
    const f = std.c.fopen(path, "ab") orelse return error.DumpOpen;
    defer _ = std.c.fclose(f);
    const hdr = [2]i64{ @intCast(N), 0 };
    _ = std.c.fwrite(std.mem.asBytes(&hdr).ptr, 1, 16, f);
    _ = std.c.fwrite(std.mem.sliceAsBytes(dr.h_cpos[0..N]).ptr, 1, N * 8, f);
    _ = std.c.fwrite(buf.ptr, 1, buf.len, f);
}

fn writeFile(dr: *Drafter, name: []const u8, bytes: []const u8) !void {
    var pb: [512]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ std.mem.span(dr.dump.?), name }, 0);
    const f = std.c.fopen(path, "wb") orelse return error.DumpOpen;
    defer _ = std.c.fclose(f);
    if (std.c.fwrite(bytes.ptr, 1, bytes.len, f) != bytes.len) return error.DumpWrite;
}

pub fn ingest(dr: *Drafter, e: *forward.Engine, runs: []const Run) !void {
    const k = &e.k;
    var N: usize = 0;
    for (runs) |r| N += r.n;
    if (N == 0) return;
    if (N > MAX_CTX) return error.DrafterContextTooWide;
    var row: usize = 0;
    for (runs) |r| {
        if (r.n == 0) continue;
        try k.copy(dr.cin + row * 5 * D * 2, r.taps, r.n * 5 * D * 2);
        for (0..r.n) |j| {
            dr.h_cpos[row + j] = r.pos0 + @as(i64, @intCast(j));
            dr.h_crings[row + j] = r.s.draft;
        }
        row += r.n;
    }
    if (dr.dump != null and dr.dump_state == 1) try dumpIngest(dr, e, N);
    try k.upload(dr.cpos, std.mem.sliceAsBytes(dr.h_cpos[0..N]));
    try k.upload(dr.crings, std.mem.sliceAsBytes(dr.h_crings[0..N]));
    try lin(e, dr.cin, dr.fc, dr.c1, N, false);
    try kern.rmsnorm(k, dr.c1, D, dr.hnorm, dr.ctx, D, null, @intCast(N), D);
    for (dr.layers, 0..) |l, li| {
        try lin(e, dr.ctx, l.kv, dr.ckv, N, false);
        var a: cuda.Args = .{};
        a.add(dr.ckv);
        a.add(l.kn);
        a.add(dr.cpos);
        a.add(dr.crings);
        a.add(@as(i32, @intCast(li)));
        a.add(EPS);
        a.add(THETA);
        try k.go(try k.ext(.dflash, "df_ctx_kv"), .{ .x = @intCast(N), .y = 8 }, 64, 0, &a);
    }
    try k.stream.synchronize();
}

// -- drafting ------------------------------------------------------------------------------------------------------

/// Kept rows into each context, then the block pass for drafting streams (both ranks); rank 0 gets 7 drafts each.
pub fn draft(dr: *Drafter, e: *forward.Engine, all: []const Req) !void {
    const k = &e.k;
    if (all.len == 0) return;
    if (all.len > forward.MAX_STREAMS) return error.TooManyStreams;
    const t0 = std.Io.Timestamp.now(e.io, .awake);
    var tp = t0;
    // the last verify's kept rows into each stream's context
    var runs: [forward.MAX_STREAMS]Run = undefined;
    var nr: usize = 0;
    for (all) |q| {
        if (q.n == 0) continue;
        const s = q.s;
        if (s.v_step != e.verifies or q.n > s.v_rows) { // its rows were overwritten (or never verified): restart its context
            s.d_from = s.pos;
            dr.lost += 1;
            continue;
        }
        runs[nr] = .{ .s = s, .taps = dr.taps_step + @as(u64, @intCast(s.v_row0)) * 5 * D * 2, .n = q.n, .pos0 = s.v_pos0 };
        nr += 1;
    }
    try ingest(dr, e, runs[0..nr]);
    try phase(dr, e, &tp, 0);
    var sub: [forward.MAX_STREAMS]Req = undefined;
    var S: usize = 0;
    for (all) |q| if (q.drafts) {
        sub[S] = q;
        S += 1;
    };
    if (S == 0) return;
    const reqs = sub[0..S];
    // the block: [anchor, MASK x 7] a stream, at positions pos .. pos + 7
    const M = S * BLOCK;
    if (dr.split and M > e.buf.rows) return error.DrafterRowsExceedBuffers; // the split's partials use the decode buffers
    for (reqs, 0..) |q, i| {
        dr.h_rings[i] = q.s.draft;
        dr.h_starts[i] = q.s.pos;
        dr.h_froms[i] = q.s.d_from;
        for (0..BLOCK) |t| {
            dr.h_ids[i * BLOCK + t] = @intCast(if (t == 0) q.anchor else MASK);
            dr.h_bpos[i * BLOCK + t] = @intCast(q.s.pos + @as(i64, @intCast(t)));
        }
    }
    try k.upload(dr.ids, std.mem.sliceAsBytes(dr.h_ids[0..M]));
    try k.upload(dr.bpos, std.mem.sliceAsBytes(dr.h_bpos[0..M]));
    try k.upload(dr.rings, std.mem.sliceAsBytes(dr.h_rings[0..S]));
    try k.upload(dr.starts, std.mem.sliceAsBytes(dr.h_starts[0..S]));
    try k.upload(dr.froms, std.mem.sliceAsBytes(dr.h_froms[0..S]));
    try phase(dr, e, &tp, 4);
    const rank0 = reqs[0].out != null;
    const nw = M * CW;
    if (dr.prof or !e.use_graphs or e.sync_each) {
        try blockBody(dr, e, S, rank0, &tp);
    } else if (dr.graphs[S]) |ex| {
        try ex.launchOn(k.stream);
    } else { // the block's pass for S streams, captured once (its tables at fixed places, uploaded above)
        try cuda.graph.beginCapture(k.stream, .thread_local);
        blockBody(dr, e, S, rank0, &tp) catch |err| {
            if (cuda.graph.endCapture(k.stream)) |g| {
                var gg = g;
                gg.deinit();
            } else |_| {}
            return err;
        };
        var g = try cuda.graph.endCapture(k.stream);
        defer g.deinit();
        dr.graphs[S] = try g.instantiate();
        try dr.graphs[S].?.launchOn(k.stream);
    }
    try k.stream.synchronize();
    try e.ctx.gathered(); // the candidates and projections below come through gathers
    try phase(dr, e, &tp, 2);
    if (!rank0 and dr.prof_fine and dr.calls % 256 == 255) logFine(dr, 256); // rank 1 logs its own table (rank 0's goes with the trace)
    if (rank0) {
        try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(dr.cand.ptr), dr.gtk, 2 * nw * 4), "dflash2 candidates");
        try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(dr.gh.ptr), dr.g, M * RANK * 4), "dflash2 projections");
        for (reqs, 0..) |q, i| ds.select(dr, i, M, q);
        if (dr.dump != null and dr.dump_state == 1 and S == 1) dr.dump_round += 1;
        if (dr.dump != null and dr.dump_state == 1 and S == 1 and dr.dump_round == dr.dump_at) {
            const hn = try dr.gpa.alloc(u8, M * D * 2);
            defer dr.gpa.free(hn);
            try e.ctx.d.check(e.ctx.d.api.cuMemcpyDtoH_v2(@ptrCast(hn.ptr), dr.hn, hn.len), "dump hidden");
            try writeFile(dr, "hidden.bin", hn);
            try writeFile(dr, "cand.bin", std.mem.sliceAsBytes(dr.cand[0 .. 2 * nw]));
            try writeFile(dr, "proj.bin", std.mem.sliceAsBytes(dr.gh[0 .. M * RANK]));
            var meta: [64]u8 = undefined;
            const q = reqs[0];
            const text = try std.fmt.bufPrint(&meta, "{d} {d}", .{ q.anchor, q.s.pos });
            try writeFile(dr, "meta.txt", text);
            try writeFile(dr, "drafts.bin", std.mem.sliceAsBytes(q.out.?[0 .. BLOCK - 1]));
            try writeFile(dr, "probs.bin", std.mem.sliceAsBytes(q.probs.?[0 .. BLOCK - 1]));
            dr.dump_state = 2;
            std.log.info("dflash2: dumped a prompt's taps and its first round to {s}", .{std.mem.span(dr.dump.?)});
        }
    }
    try phase(dr, e, &tp, 3);
    dr.forward_ms += @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(e.io, .awake)).toNanoseconds())) / 1e6;
    dr.calls += 1;
}

/// The block pass for S streams: embed, 5 layers, norm, both ranks' 16 best a row, rank 0's selector. Capturable.
fn blockBody(dr: *Drafter, e: *forward.Engine, S: usize, rank0: bool, tp: *std.Io.Timestamp) !void {
    const k = &e.k;
    const M = S * BLOCK;
    const nw = M * CW;
    if (dr.prof_fine) dr.fine_t = std.Io.Timestamp.now(e.io, .awake);
    try kern.embed(k, dr.ids, e.w.embed, dr.x, @intCast(M), 1);
    try fine(dr, e, 0);
    const R: i64 = @intCast(M);
    const grid: cuda.Dim3 = .{ .x = @intCast(M), .y = @intCast(D / 1024) };
    for (dr.layers, 0..) |l, li| {
        // attention, wrapped in the attention conv
        try kern.rmsnorm(k, dr.x, D, l.in_norm, dr.y, D, null, R, D);
        try fine(dr, e, 1);
        try lin(e, dr.y, l.a_proj, dr.dyn, M, false);
        try fine(dr, e, 2);
        try conv(k, dr.y, dr.dyn, 0, l.a_base, dr.y2, false, grid);
        try fine(dr, e, 3);
        try lin(e, dr.y2, l.qkv, dr.qkv, M, false);
        try fine(dr, e, 4);
        {
            var a: cuda.Args = .{};
            a.add(dr.qkv);
            a.add(@as(i32, @intCast(QKV)));
            a.add(l.qn);
            a.add(l.kn);
            a.add(dr.bpos);
            a.add(EPS);
            a.add(THETA);
            a.add(dr.q);
            a.add(dr.kb);
            a.add(dr.vb);
            try k.go(try k.ext(.dflash, "df_block_qkv"), .{ .x = @intCast(M), .y = 32 + 8 }, 64, 0, &a);
        }
        try fine(dr, e, 5);
        if (dr.attn_chunked) {
            var a: cuda.Args = .{};
            a.add(dr.q);
            a.add(dr.kb);
            a.add(dr.vb);
            a.add(dr.rings);
            a.add(dr.starts);
            a.add(dr.froms);
            a.add(@as(i32, @intCast(li)));
            a.add(SCALE);
            a.add(dr.pm);
            a.add(dr.pl);
            a.add(dr.pacc);
            try k.go(try k.ext(.dflash, "df_attn_part"), .{ .x = @intCast(S), .y = 8, .z = CH }, 256, 0, &a);
            var b: cuda.Args = .{};
            b.add(dr.pm);
            b.add(dr.pl);
            b.add(dr.pacc);
            b.add(dr.att);
            try k.go(try k.ext(.dflash, "df_attn_merge"), .{ .x = @intCast(M), .y = QH }, HD, 0, &b);
        } else {
            var a: cuda.Args = .{};
            a.add(dr.q);
            a.add(dr.kb);
            a.add(dr.vb);
            a.add(dr.rings);
            a.add(dr.starts);
            a.add(dr.froms);
            a.add(@as(i32, @intCast(li)));
            a.add(SCALE);
            a.add(dr.att);
            try k.go(try k.ext(.dflash, "df_attn"), .{ .x = @intCast(S), .y = 8 }, 256, 0, &a);
        }
        try fine(dr, e, 6);
        try lin(e, dr.att, l.o, dr.a, M, false);
        try fine(dr, e, 7);
        try conv(k, dr.a, dr.dyn, 1, l.a_base + 2 * D * 2, dr.x, true, grid);
        try fine(dr, e, 3);
        // the MLP, wrapped in the MLP conv
        try kern.rmsnorm(k, dr.x, D, l.post_norm, dr.y, D, null, R, D);
        try fine(dr, e, 1);
        try lin(e, dr.y, l.m_proj, dr.dyn, M, false);
        try fine(dr, e, 8);
        try conv(k, dr.y, dr.dyn, 0, l.m_base, dr.y2, false, grid);
        try fine(dr, e, 3);
        try lin(e, dr.y2, l.gu, dr.gu, M, false);
        try fine(dr, e, 9);
        {
            var a: cuda.Args = .{};
            a.add(dr.gu);
            a.add(dr.act);
            a.add(@as(i32, @intCast(dr.ff)));
            try k.go(try k.ext(.dflash, "df_swiglu"), .{ .x = @intCast(M), .y = @intCast(dr.ff / 1024) }, 256, 0, &a);
        }
        try fine(dr, e, 10);
        if (dr.split) { // each rank's half of down's K as fp32 partials, gathered and summed
            const b = &e.buf;
            try lin(e, dr.act, l.down, b.part, M, true);
            try fine(dr, e, 11);
            try e.ctx.allGather(b.part, b.gath, M * D, .f32, "dflash2 down");
            try fine(dr, e, 16);
            var a: cuda.Args = .{};
            a.add(b.gath);
            a.add(dr.a);
            a.add(@as(i32, @intCast(M)));
            try k.go(try k.ext(.dflash, "df_sum2"), grid, 256, 0, &a);
            try fine(dr, e, 17);
        } else {
            try lin(e, dr.act, l.down, dr.a, M, false);
            try fine(dr, e, 11);
        }
        try conv(k, dr.a, dr.dyn, 1, l.m_base + 2 * D * 2, dr.x, true, grid);
        try fine(dr, e, 3);
    }
    try kern.rmsnorm(k, dr.x, D, dr.norm, dr.hn, D, null, R, D);
    try fine(dr, e, 1);
    try phase(dr, e, tp, 1);
    // every row through the drafts' packed head (the anchor row skipped by the host), 16 best a rank, selector proj
    try blocks.mm(e, &e.mbuf, dr.hn, D, if (dr.full_head) e.w.head else e.w.draft_head, e.mbuf.logits, false, R);
    try fine(dr, e, 12);
    {
        const chunks = cdiv(forward.HEAD_N, 4096);
        var a: cuda.Args = .{};
        a.add(e.mbuf.logits);
        a.add(@as(i32, @intCast(forward.HEAD_N)));
        a.add(@as(i32, @intCast(e.w.vocab_offset)));
        a.add(dr.pv);
        a.add(dr.pi);
        try k.go(try k.ext(.dflash, "df_top16_part"), .{ .x = @intCast(M), .y = @intCast(chunks) }, 256, 0, &a);
        var b: cuda.Args = .{};
        b.add(dr.pv);
        b.add(dr.pi);
        b.add(@as(i32, @intCast(chunks)));
        b.add(dr.tk);
        try k.go(try k.ext(.dflash, "df_top16_merge"), .{ .x = @intCast(M) }, 256, 0, &b);
    }
    try fine(dr, e, 13);
    try e.ctx.allGather(dr.tk, dr.gtk, nw, .i32, "dflash2 candidates");
    try fine(dr, e, 14);
    if (rank0) try lin(e, dr.hn, dr.hproj, dr.g, M, true);
    try fine(dr, e, 15);
}

fn phase(dr: *Drafter, e: *forward.Engine, t: *std.Io.Timestamp, which: usize) !void {
    if (!dr.prof) return;
    try e.k.stream.synchronize();
    const now = std.Io.Timestamp.now(e.io, .awake);
    dr.phase_ms[which] += @as(f64, @floatFromInt(t.durationTo(now).toNanoseconds())) / 1e6;
    t.* = now;
}

fn conv(k: *kern.K, z: u64, dyn: u64, which: i32, base: u64, out: u64, add: bool, grid: cuda.Dim3) !void {
    var a: cuda.Args = .{};
    a.add(z);
    a.add(dyn);
    a.add(which);
    a.add(base);
    a.add(out);
    a.add(@as(i32, @intFromBool(add)));
    try k.go(try k.ext(.dflash, "df_conv"), grid, 256, 0, &a);
}
