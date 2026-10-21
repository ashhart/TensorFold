//! The MTP draft chain as one CUDA graph (TF_FLASHNEXT_EXL3_GRAPH=1, greedy drafts).
//!
//! Upstream's chain (cuda_mtp_ops.Ops.draft) picks a draft on the host after every head step: fn_row_best, a blocking
//! read of four floats, then the next step uploads that token. Here the whole chain is captured once per shape: each
//! pick's record stays on the device, fn_exl3_draft_token turns its column into the next step's input token, and the
//! head step (the same launches as forwardHead for one row, its experts on the device planner) runs on it; the host
//! reads the chain's records once and applies upstream's confidence rule to them. Every step past an early stop still
//! runs (its cache entries are past the chain and are trimmed by the next absorb, as drafted entries always are), so
//! the drafts and their probabilities are upstream's bit for bit (tf-cuda-test flashnext-exl3-graph and the serve
//! gate's drafted == undrafted).
//!
//! `enter`/`leave` (cuda_mtp_ops.forwardHead's hook) also put every head step of up to the planner's rows, eager or
//! captured, on the device planner, so the head's pool stays device-owned between decode steps.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const batch = @import("cuda_batch.zig");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");
const dplan = @import("cuda_exl3_dplan.zig");
const graph = @import("cuda_exl3_graph.zig");

const dims = wt.dims;
const streams = wt.streams;
const wide = wt.wide;
const eps = wt.eps;
const out_sk = wt.out_sk;
const max_chain = mtp.max_chain;
const fc_sk = mtp.fc_sk;
const piece_rows = mtp.piece_rows;

/// forwardHead's hook: the head's pool on the device planner for a step of `rows` rows; returns what to restore.
pub fn enter(e: *prompt.Engine, rows: usize) !?*dplan.DevPlan {
    const was = dplan.active;
    if (!graph.enabled() or rows == 0) return was;
    const dp = (try dplan.forLayers(e.gpa, e.driver, e, e.layers[0..e.n_layers])) orelse return was;
    const mp = dplan.mtp_pool orelse return was;
    if (rows > dp.max_rows or dp.indexOf(mp) == null) return was;
    try dp.toDevice(e.stream.*);
    dplan.active = dp;
    return was;
}

pub fn leave(was: ?*dplan.DevPlan) void {
    dplan.active = was;
}

const Key = struct {
    picks: usize,
    heads: usize,
    /// Attention seams of the first and the last step (chunk counts and score tiles move with the head's length).
    seams: [4]u32,
    head: usize,
    kc: u64,
    out: u64,
    dlogits: u64,
    cap: usize,
    scratch: u64,
    scratch_cap: usize,
    scores: u64,
    plan: u64,
    confidence: u64,
};

const Entry = struct { key: Key, exec: cuda.graph.Exec = undefined, used: u64, pending: ?*graph.Pending = null, admission: ?graph.memory.Ticket = null, stream: ?cuda.Stream = null };

const State = struct {
    head: usize,
    ids: cuda.DeviceBuffer,
    token: cuda.DeviceBuffer,
    best: cuda.DeviceBuffer,
    pos_host: cuda.HostBuffer,
    best_host: cuda.HostBuffer,
    module: cuda.Module,
    token_fn: cuda.Function,
    next_fn: cuda.Function,
    /// [0] the chain's running decision, [1 + j] the decision after pick j (read back with the records).
    alive: cuda.DeviceBuffer,
    alive_host: cuda.HostBuffer,
    /// Where a step's conditional body is captured (cuStreamBeginCaptureToGraph).
    side: cuda.Stream,
    cond: Cond,
};

/// The conditional-node driver entry points (CUDA 12.4+), looked up from libcuda.
const NodeParams = extern struct {
    type: c_int,
    reserved0: [3]c_int = .{ 0, 0, 0 },
    u: extern union { cond: CondParams, pad: [29]i64 },
    reserved2: i64 = 0,
};
const CondParams = extern struct { handle: u64, type: c_int, size: c_uint, graphs: ?[*]?*anyopaque, ctx: ?*anyopaque };
const Cond = struct {
    lib: std.DynLib,
    handle_create: *const fn (*u64, ?*anyopaque, ?*anyopaque, c_uint, c_uint) callconv(.c) c_int,
    add_node: *const fn (*?*anyopaque, ?*anyopaque, ?[*]const ?*anyopaque, ?*const anyopaque, usize, *NodeParams) callconv(.c) c_int,
    capture_info: *const fn (?*anyopaque, *c_int, *u64, *?*anyopaque, *?[*]const ?*anyopaque, *?*const anyopaque, *usize) callconv(.c) c_int,
    update_deps: *const fn (?*anyopaque, ?[*]const ?*anyopaque, ?*const anyopaque, usize, c_uint) callconv(.c) c_int,
    begin_to_graph: *const fn (?*anyopaque, ?*anyopaque, ?[*]const ?*anyopaque, ?*const anyopaque, usize, c_int) callconv(.c) c_int,
    ctx_current: *const fn (*?*anyopaque) callconv(.c) c_int,

    fn open() !Cond {
        var lib = try std.DynLib.open("libcuda.so.1");
        errdefer lib.close();
        var c: Cond = undefined;
        c.lib = lib;
        c.handle_create = lib.lookup(@TypeOf(c.handle_create), "cuGraphConditionalHandleCreate") orelse return error.MissingSymbol;
        c.add_node = lib.lookup(@TypeOf(c.add_node), "cuGraphAddNode_v2") orelse return error.MissingSymbol;
        c.capture_info = lib.lookup(@TypeOf(c.capture_info), "cuStreamGetCaptureInfo_v3") orelse return error.MissingSymbol;
        c.update_deps = lib.lookup(@TypeOf(c.update_deps), "cuStreamUpdateCaptureDependencies_v2") orelse return error.MissingSymbol;
        c.begin_to_graph = lib.lookup(@TypeOf(c.begin_to_graph), "cuStreamBeginCaptureToGraph") orelse return error.MissingSymbol;
        c.ctx_current = lib.lookup(@TypeOf(c.ctx_current), "cuCtxGetCurrent") orelse return error.MissingSymbol;
        return c;
    }

    fn ok(r: c_int, what: []const u8) !void {
        if (r == 0) return;
        std.log.err("flashnext exl3 chain: {s}: CUDA error {d}", .{ what, r });
        return error.CudaFailed;
    }

    /// The capturing graph of `s` (main capture) and a new conditional handle on it.
    fn handle(c: *const Cond, s: cuda.Stream) !u64 {
        var status: c_int = 0;
        var id: u64 = 0;
        var g: ?*anyopaque = null;
        var deps: ?[*]const ?*anyopaque = null;
        var edges: ?*const anyopaque = null;
        var n: usize = 0;
        try ok(c.capture_info(@ptrCast(s.handle), &status, &id, &g, &deps, &edges, &n), "cuStreamGetCaptureInfo");
        var ctx: ?*anyopaque = null;
        try ok(c.ctx_current(&ctx), "cuCtxGetCurrent");
        var h: u64 = 0;
        try ok(c.handle_create(&h, g, ctx, 0, 1), "cuGraphConditionalHandleCreate");
        return h;
    }

    /// An IF node on `h` after `s`'s captured work; `s` continues after it; returns its body graph.
    fn ifNode(c: *const Cond, s: cuda.Stream, h: u64) !?*anyopaque {
        var status: c_int = 0;
        var id: u64 = 0;
        var g: ?*anyopaque = null;
        var deps: ?[*]const ?*anyopaque = null;
        var edges: ?*const anyopaque = null;
        var n: usize = 0;
        try ok(c.capture_info(@ptrCast(s.handle), &status, &id, &g, &deps, &edges, &n), "cuStreamGetCaptureInfo");
        var ctx: ?*anyopaque = null;
        try ok(c.ctx_current(&ctx), "cuCtxGetCurrent");
        var params: NodeParams = .{ .type = 13, .u = .{ .pad = @splat(0) } };
        params.u.cond = .{ .handle = h, .type = 0, .size = 1, .graphs = null, .ctx = ctx };
        var node: ?*anyopaque = null;
        try ok(c.add_node(&node, g, deps, null, n, &params), "cuGraphAddNode(conditional)");
        const one = [1]?*anyopaque{node};
        try ok(c.update_deps(@ptrCast(s.handle), &one, null, 1, 1), "cuStreamUpdateCaptureDependencies");
        return params.u.cond.graphs.?[0];
    }
};

var state: ?State = null;
var entries: [16]?Entry = @splat(null);
var clock: u64 = 0;
var broken = false;

pub var stats: struct { chains: u64 = 0, replays: u64 = 0, captures: u64 = 0, updates: u64 = 0, eager: u64 = 0 } = .{};

fn stateFor(e: *prompt.Engine, h: *const mtp.Head) !*State {
    if (state) |*s| if (s.head == @intFromPtr(h)) return s;
    if (state) |*s| release(s);
    var st: State = undefined;
    st.head = @intFromPtr(h);
    st.ids = try cuda.DeviceBuffer.fromHost(e.driver, std.mem.sliceAsBytes(h.ids));
    errdefer st.ids.free();
    st.token = try cuda.DeviceBuffer.alloc(e.driver, 256);
    errdefer st.token.free();
    st.best = try cuda.DeviceBuffer.alloc(e.driver, max_chain * 16);
    errdefer st.best.free();
    st.pos_host = try cuda.HostBuffer.alloc(e.driver, max_chain * 16);
    errdefer st.pos_host.free();
    st.best_host = try cuda.HostBuffer.alloc(e.driver, max_chain * 16);
    errdefer st.best_host.free();
    st.module = try cuda.Module.load(e.driver, cuda.kernels.flashnext_exl3_dplan);
    errdefer st.module.unload();
    st.token_fn = try st.module.function("fn_exl3_draft_token");
    st.next_fn = try st.module.function("fn_exl3_draft_next");
    st.alive = try cuda.DeviceBuffer.alloc(e.driver, (1 + max_chain) * 4 + 256);
    errdefer st.alive.free();
    st.alive_host = try cuda.HostBuffer.alloc(e.driver, (1 + max_chain) * 4);
    errdefer st.alive_host.free();
    st.side = try cuda.Stream.init(e.driver, true);
    errdefer st.side.deinit();
    st.cond = try Cond.open();
    state = st;
    return &state.?;
}

fn release(s: *State) void {
    for (&entries) |*en| if (en.*) |*x| {
        dropEntry(x);
        en.* = null;
    };
    s.cond.lib.close();
    s.side.deinit();
    s.alive_host.free();
    s.alive.free();
    s.module.unload();
    s.best_host.free();
    s.pos_host.free();
    s.best.free();
    s.token.free();
    s.ids.free();
    state = null;
}

fn seams(end: usize, chunks: *u32, tiles: *u32) void {
    chunks.* = @intCast((@min(end, wt.dense_keys) + 511) / 512);
    tiles.* = if (end / wt.index_ratio > wt.index_top) @intCast(((end + 3) / 4 + 63) / 64) else 0;
}

/// One head step for one row whose token is on the device: forwardHead's launches for `next.len == 1` (streams from
/// d.out, scores into d.dlogits), its position from the pinned row `j`.
fn step(comptime Tri: type, o: anytype, tri: Tri, p: *batch.Pass(Tri), d: *mtp.Draft, st: *State, j: usize, len0: usize) !void {
    const e = o.e;
    const h = o.h;
    const s = p.s;
    const stream = p.stream;
    p.rows = 1;
    p.pos0 = len0 + j;
    p.pending = null;
    try stream.d.check(stream.d.api.cuMemcpyDtoDAsync_v2(s.ids, st.token.ptr, 4, stream.handle), "cuMemcpyDtoDAsync");
    try graph.pinnedCopy(stream, s.pos, st.pos_host.bytes[j * 16 ..][0..16]);
    if (e.table.dense) {
        try tri.embedDense(s.ids, e.table.w.ptr, s.ple_keys, dims, streams, 1);
    } else {
        try tri.embed(s.ids, e.table.w.ptr, e.table.s.ptr, e.table.b.ptr, s.ple_keys, dims, streams, 1);
    }
    try tri.rmsnorm(s.ple_keys, h.norm_e.ptr, s.mixed, s.xsm, eps, wide, 1, dims, dims);
    try p.lane(s.mixed, s.xsm, h.fc_e, s.dummy, s.part, 1, fc_sk);
    try p.sumSlices(s.part, s.branch, true, dims, fc_sk);
    try tri.rmsnorm(d.out.ptr, h.norm_h.ptr, s.norm, s.gq, eps, wide, 1, wide, wide);
    try p.lane(s.norm, s.gq, h.fc_h, s.dummy, s.part, streams, fc_sk);
    try p.sumSlices(s.part, s.ple_gated, true, streams * dims, fc_sk);
    try tri.addStreams(s.branch, s.ple_gated, s.h, 1, dims, streams);
    try p.writeback(0, 1);
    try p.readout(h.layer.attn_hc, 0, 1);
    try p.attnBlock(&h.layer, 0);
    try tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", dims, 1, dims, streams, 4, 1, 1, out_sk);
    try p.readout(h.layer.mlp_hc, 0, 1);
    try p.moeBlock(&h.layer);
    p.pending = 0;
    try p.writeback(0, 1);
    try p.copy(d.out.ptr, s.h, wide * 2);
    try p.readout(h.mixer, 0, 1);
    try p.lane(s.mixed, s.xsm, h.draft, d.dlogits.ptr, s.dummy, 1, 1);
}

/// The chain's launches on `p`'s stream: the decision reset, every pick's record into st.best; after a pick that
/// continues, its token and decision (fn_exl3_draft_next) and a conditional node whose body (captured on st.side
/// through `ps`, a pass on that stream) is the next head step: a chain that stops skips its remaining steps.
fn chainBody(comptime Tri: type, o: anytype, p: *batch.Pass(Tri), ps: *batch.Pass(Tri), d: *mtp.Draft, st: *State, picks: usize, heads: usize, len0: usize, confidence: f64) !void {
    const h = o.h;
    const stream = p.stream;
    try stream.d.check(stream.d.api.cuMemsetD32Async(st.alive.ptr, 1, 1 + max_chain, stream.handle), "cuMemsetD32Async");
    for (0..picks) |j| {
        try h.best(stream, d.dlogits.ptr, h.ids.len, h.ids.len, 1, st.best.ptr + j * 16);
        if (j >= heads) continue;
        const handle = try st.cond.handle(stream);
        var a: cuda.Args = .{};
        a.add(st.best.ptr + j * 16);
        a.add(st.ids.ptr);
        a.add(st.token.ptr);
        a.add(st.alive.ptr);
        a.add(st.alive.ptr + (1 + j) * 4);
        a.add(handle);
        a.add(confidence);
        try cuda.launch.launch(st.next_fn, .{ .grid = .{ .x = 1 }, .block = .{ .x = 32 } }, stream, &a);
        const body = try st.cond.ifNode(stream, handle);
        try Cond.ok(st.cond.begin_to_graph(@ptrCast(st.side.handle), body, null, null, 0, 1), "cuStreamBeginCaptureToGraph");
        step(Tri, o, ps.tri, ps, d, st, j, len0) catch |err| {
            var discarded: cuda.abi.Graph = null;
            _ = stream.d.api.cuStreamEndCapture(st.side.handle, &discarded);
            return err;
        };
        var g: cuda.abi.Graph = null;
        try stream.d.check(stream.d.api.cuStreamEndCapture(st.side.handle, &g), "cuStreamEndCapture(body)");
    }
}

/// Ops.draft's hook (cuda_mtp_ops.zig): the chain for greedy drafts as one graph; false leaves it to upstream.
pub fn draft(comptime Tri: type, o: anytype, d: *mtp.Draft, next: []const u32, rows: u64, position: u64, count: usize, sm: ?mtp.Sampling, confidence: f64, out: *mtp.Chain) !bool {
    _ = position;
    if (graph.enabled() and broken) return error.GraphCaptureFailed;
    if (!graph.enabled() or count < 2 or d.media != null) return false;
    if (sm) |s| if (s.temperature > 0) return false;
    const e = o.e;
    const dp = (try dplan.forLayers(e.gpa, e.driver, e, e.layers[0..e.n_layers])) orelse return false;
    const mp = dplan.mtp_pool orelse return false;
    if (dp.indexOf(mp) == null) return false;
    const picks: usize = @min(count, max_chain);
    // upstream runs a head step after pick j while j + 1 < count
    const heads: usize = if (count > max_chain) max_chain else count - 1;
    if (heads > picks) return false;
    out.n = 0;
    try o.absorb(d, next, rows, true);
    if (d.len + heads > d.cap) {
        // upstream's loop, which reports PromptTooLong where it would
        stats.eager += 1;
        return eagerChain(o, d, count, picks, confidence, out);
    }
    const st = try stateFor(e, o.h);
    const len0 = d.len;
    const pos: []i32 = @alignCast(std.mem.bytesAsSlice(i32, st.pos_host.bytes[0 .. max_chain * 16]));
    @memset(pos, 0);
    for (0..heads) |j| pos[4 * j] = @intCast(len0 + j);
    const s = try e.reserve(@max(@as(usize, 32), 2));
    const piece = @max(@as(usize, 1), @min(piece_rows, s.cap / 2));
    const score_rows = try e.reserveScores(d.cap, piece);
    var p: batch.Pass(Tri) = .{
        .gpa = e.gpa,
        .io = e.io,
        .stream = e.stream.*,
        .tri = o.tri,
        .k = &e.batch_k,
        .s = s,
        .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = (&o.h.layer)[0..1], .mixer = o.h.mixer, .head = o.h.draft, .ngram = &e.ngram },
        .seq = .{ .media = null, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows },
        .hooks = .{ .logits = .none },
    };
    var key: Key = .{ .picks = picks, .heads = heads, .seams = undefined, .head = @intFromPtr(o.h), .kc = d.kc.ptr, .out = d.out.ptr, .dlogits = d.dlogits.ptr, .cap = d.cap, .scratch = s.ids, .scratch_cap = s.cap, .scores = score_rows, .plan = dp.state.ptr, .confidence = @bitCast(confidence) };
    seams(len0 + 1, &key.seams[0], &key.seams[1]);
    seams(len0 + @max(heads, 1), &key.seams[2], &key.seams[3]);
    try dp.toDevice(p.stream);
    const was = dplan.active;
    dplan.active = dp;
    defer dplan.active = was;
    stats.chains += 1;
    var tri_side = o.tri;
    tri_side.s = st.side;
    var ps = p;
    ps.stream = st.side;
    ps.tri = tri_side;
    const launched = run(Tri, key, o, &p, &ps, d, st, picks, heads, len0, confidence) catch |err| {
        std.log.err("flashnext exl3 chain: the draft chain failed ({t})", .{err});
        broken = true;
        return err;
    };
    if (!launched) {
        stats.eager += 1;
        return eagerChain(o, d, count, picks, confidence, out);
    }
    // the chain's records, once
    const stream = p.stream;
    try stream.d.check(stream.d.api.cuMemcpyDtoHAsync_v2(st.best_host.bytes.ptr, st.best.ptr, picks * 16, stream.handle), "cuMemcpyDtoHAsync");
    try stream.d.check(stream.d.api.cuMemcpyDtoHAsync_v2(st.alive_host.bytes.ptr, st.alive.ptr, (1 + max_chain) * 4, stream.handle), "cuMemcpyDtoHAsync");
    try stream.synchronize();
    const alive: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, st.alive_host.bytes[0 .. (1 + max_chain) * 4]));
    d.len += heads;
    d.drafted += heads;
    const rec: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, st.best_host.bytes[0 .. picks * 16]));
    for (0..picks) |j| {
        const top = rec[4 * j];
        const lse = rec[4 * j + 1];
        const col: usize = @intCast(@as(i32, @bitCast(rec[4 * j + 2])));
        const got: mtp.Pick = .{ .token = o.h.ids[col], .prob = @exp(@as(f64, top) - @as(f64, lse)) };
        const low = confidence > 0 and got.prob < confidence;
        if (low and j > 0) break;
        out.tokens[out.n] = got.token;
        out.probs[out.n] = got.prob;
        out.n += 1;
        if (low) break;
        // the device stopped the chain here (its decision is the same rule; a step past it did not run)
        if (j < heads and alive[1 + j] == 0) break;
    }
    return true;
}

/// upstream's loop (Ops.draft after its absorb), for chains that leave the graph path.
fn eagerChain(o: anytype, d: *mtp.Draft, count: usize, picks: usize, confidence: f64, out: *mtp.Chain) !bool {
    var j: usize = 0;
    while (j < picks) : (j += 1) {
        const got = try o.pick(d, 0, null);
        const low = confidence > 0 and got.prob < confidence;
        if (low and j > 0) break;
        out.tokens[out.n] = got.token;
        out.probs[out.n] = got.prob;
        out.n += 1;
        if (low) break;
        if (j + 1 < count) {
            try o.forwardHead(d, &.{got.token}, d.out.ptr, true);
            d.drafted += 1;
        }
    }
    return true;
}

fn sameShape(a: Key, b: Key) bool {
    var x = a;
    x.kc = b.kc;
    x.out = b.out;
    x.dlogits = b.dlogits;
    return std.meta.eql(x, b);
}

/// Replays the chain's graph (true), or captures it; a new graph instantiates on a worker thread and this chain runs
/// eagerly (false), as does every chain until it landed.
fn run(comptime Tri: type, key: Key, o: anytype, p: *batch.Pass(Tri), ps: *batch.Pass(Tri), d: *mtp.Draft, st: *State, picks: usize, heads: usize, len0: usize, confidence: f64) !bool {
    clock += 1;
    for (&entries) |*en| if (en.*) |*x| if (std.meta.eql(x.key, key)) {
        x.used = clock;
        if (x.pending) |pd| {
            const got = graph.takePending(pd) catch |err| {
                if (x.admission) |ticket| graph.memory.release(ticket) catch unreachable;
                en.* = null;
                return err;
            };
            var exec = got orelse return false;
            x.pending = null;
            graph.uploadResident(&exec, p.stream) catch |err| {
                exec.deinit();
                if (x.admission) |ticket| graph.memory.release(ticket) catch unreachable;
                en.* = null;
                return err;
            };
            x.exec = exec;
            if (x.admission) |ticket| try graph.memory.installed(ticket);
        }
        stats.replays += 1;
        try x.exec.launchOn(p.stream);
        return true;
    };
    try p.stream.synchronize();
    const admission = try graph.reserveMemory(p.stream, .chain, 0);
    var admitted = true;
    defer if (admitted) graph.memory.release(admission) catch unreachable;
    try cuda.graph.beginCapture(p.stream, .thread_local);
    graph.capturing = true;
    p.prepared = true;
    chainBody(Tri, o, p, ps, d, st, picks, heads, len0, confidence) catch |err| {
        graph.capturing = false;
        if (cuda.graph.endCapture(p.stream)) |g| {
            var x = g;
            x.deinit();
        } else |_| {}
        return err;
    };
    graph.capturing = false;
    var g = try cuda.graph.endCapture(p.stream);
    var owned = true;
    defer if (owned) g.deinit();
    stats.captures += 1;
    // a stale graph of the same shape (another stream's buffers) is updated in place; else a new one
    var oldest: usize = 0;
    var oldest_used: u64 = std.math.maxInt(u64);
    for (&entries, 0..) |*en, i| {
        if (en.* == null) {
            oldest = i;
            oldest_used = 0;
            break;
        }
        if (en.*.?.used < oldest_used) {
            oldest_used = en.*.?.used;
            oldest = i;
        }
    }
    for (&entries) |*en| if (en.*) |*x| if (x.pending == null and sameShape(x.key, key) and clock - x.used > 64) {
        if ((try x.exec.update(g)) == .success) {
            try graph.uploadResident(&x.exec, p.stream);
            x.key = key;
            x.used = clock;
            stats.updates += 1;
            try x.exec.launchOn(p.stream);
            return true;
        }
    };
    if (entries[oldest]) |*x| dropEntry(x);
    entries[oldest] = null;
    if (try graph.startInstantiate(g, false)) |pd| {
        owned = false;
        entries[oldest] = .{ .key = key, .used = clock, .pending = pd, .admission = admission, .stream = p.stream };
        admitted = false;
        return false;
    }
    try installSync(&entries[oldest], g, .{ .key = key, .used = clock, .admission = admission, .stream = p.stream });
    admitted = false;
    try entries[oldest].?.exec.launchOn(p.stream);
    return true;
}

fn installSync(target: *?Entry, template: cuda.graph.Graph, entry: Entry) !void {
    var exec = try template.instantiate();
    errdefer exec.deinit();
    try graph.uploadResident(&exec, entry.stream.?);
    try graph.memory.installed(entry.admission.?);
    target.* = entry;
    target.*.?.exec = exec;
}

fn dropEntry(x: *Entry) void {
    if (x.stream) |stream| stream.synchronize() catch @panic("chain retirement synchronization failed");
    defer if (x.admission) |ticket| {
        graph.memory.release(ticket) catch unreachable;
        x.admission = null;
    };
    if (x.pending) |pd| {
        while (true) {
            const got = graph.takePending(pd) catch break;
            if (got) |e| {
                var ex = e;
                ex.deinit();
                break;
            }
            std.Thread.yield() catch {};
        }
        return;
    }
    x.exec.deinit();
}

pub fn evictOne() bool {
    var oldest: ?usize = null;
    for (entries, 0..) |maybe, index| if (maybe) |entry| {
        if (oldest == null or entry.used < entries[oldest.?].?.used) oldest = index;
    };
    const index = oldest orelse return false;
    dropEntry(&entries[index].?);
    entries[index] = null;
    return true;
}

test "exl3: synchronous chain launch failure retains ownership until retirement" {
    const Mock = struct {
        var drops: usize = 0;

        fn instantiate(out: *cuda.abi.GraphExec, _: cuda.abi.Graph, _: u64) callconv(.c) c_int {
            out.* = @ptrFromInt(1);
            return 0;
        }

        fn upload(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }

        fn synchronize(_: cuda.abi.Stream) callconv(.c) c_int {
            return 0;
        }

        fn launch(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
            return cuda.abi.error_not_ready;
        }

        fn destroy(_: cuda.abi.GraphExec) callconv(.c) c_int {
            drops += 1;
            return 0;
        }

        fn installAndLaunch(target: *?Entry, template: cuda.graph.Graph, entry: Entry) !void {
            try installSync(target, template, entry);
            try target.*.?.exec.launchOn(entry.stream.?);
        }
    };
    Mock.drops = 0;
    var driver: cuda.Driver = undefined;
    driver.api.cuGraphInstantiateWithFlags = Mock.instantiate;
    driver.api.cuGraphUpload = Mock.upload;
    driver.api.cuStreamSynchronize = Mock.synchronize;
    driver.api.cuGraphLaunch = Mock.launch;
    driver.api.cuGraphExecDestroy = Mock.destroy;
    const stream: cuda.Stream = .{ .d = &driver, .handle = @ptrFromInt(1) };
    try graph.memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
    const ticket = try graph.memory.admit(.chain, 0, 100);
    var ticket_owned = true;
    defer if (ticket_owned) graph.memory.release(ticket) catch unreachable;
    var entry: ?Entry = null;
    defer if (entry) |*cached| dropEntry(cached);
    const result = Mock.installAndLaunch(&entry, .{ .d = &driver, .handle = @ptrFromInt(1) }, .{ .key = undefined, .used = 1, .admission = ticket, .stream = stream });
    if (entry != null) ticket_owned = false;
    try std.testing.expectError(error.NotReady, result);
    try std.testing.expect(entry != null);
    try std.testing.expectEqual(@as(usize, 0), Mock.drops);
    try std.testing.expectEqual(@as(u64, 30), graph.memory.charged());
    dropEntry(&entry.?);
    entry = null;
    try std.testing.expectEqual(@as(usize, 1), Mock.drops);
    try std.testing.expectEqual(@as(u64, 0), graph.memory.charged());
    try std.testing.expectError(error.StaleGraphTicket, graph.memory.release(ticket));
}
