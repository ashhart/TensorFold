//! DSpark's pass with candidates on the GPU, in CUDA graphs (TF_DSV41_DRAFT_GRAPHS=1); self-checked against the host.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const graphs = @import("graphs.zig");
const rows_mod = @import("dspark_rows.zig");
const dspark = @import("draft/dspark.zig");
const ph = @import("phases.zig");
const emit = @import("dspark_emit.zig");

pub const Mode = enum { host, device, graphs };

/// TF_DSV41_DRAFT_GRAPHS: 1 = device candidates + graphed passes, `device` = eager device candidates, 0 = host step.
pub fn modeFromEnv() !Mode {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_GRAPHS") orelse return .host);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return .host;
    if (std.mem.eql(u8, v, "1")) return .graphs;
    if (std.mem.eql(u8, v, "device")) return .device;
    return error.BadDraftGraphs;
}

pub const check_passes = 8;

pub const Dev = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    comm: tp.collective.Collective,
    stream: cuda.Stream,
    /// what TF_DSV41_DRAFT_GRAPHS asked; `active` is what runs (host after a failed check)
    want: Mode,
    active: Mode,
    k: usize,
    world: usize,
    hidden: usize,
    vocab_lo: u32,
    /// rows of every pass of a round together (the regions' size)
    rows_max: usize,
    conf: bool,
    send: cuda.DeviceBuffer,
    gathered: cuda.DeviceBuffer,
    /// pinned: gathered fp32 [rows_max][W][2 k] region-major, then the head hidden bf16 [rows_max][D]
    host: cuda.HostBuffer,
    done: cuda.Event,
    agreement: graphs.Agreement,
    engine: graphs.gc.CudaEngine,
    cache: graphs.gc.Cache(u32),
    /// pinned statics staging: `regions` regions of `region_bytes` (a pass of the round each)
    stage: cuda.HostBuffer,
    region_bytes: usize,
    regions: usize,
    checked: u32 = 0,
    mismatches: u64 = 0,
    stats: struct { passes: u64 = 0, graphed: u64 = 0 } = .{},
    /// TF_DSV41_DRAFT_CHAIN=1: the chain on the device (Chain); null: the host's
    chain: ?*Chain = null,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, comm: tp.collective.Collective, stream: cuda.Stream, want: Mode, k: usize, hidden: usize, rows_max: usize, conf: bool, region_bytes: usize, regions: usize, vocab_lo: u32) !*Dev {
        const x = try gpa.create(Dev);
        errdefer gpa.destroy(x);
        const W: usize = comm.world();
        const pk = rows_max * 2 * k * 4;
        var max: u32 = 32;
        if (std.c.getenv("TF_DSV41_DRAFT_GRAPHS_MAX")) |v| max = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        var send = try cuda.DeviceBuffer.alloc(d, pk);
        errdefer send.free();
        var gathered = try cuda.DeviceBuffer.alloc(d, pk * W);
        errdefer gathered.free();
        var host = try cuda.HostBuffer.alloc(d, pk * W + 2 * rows_max * hidden);
        errdefer host.free();
        var done = try cuda.Event.init(d, false);
        errdefer done.deinit();
        var agreement = try graphs.Agreement.init(d, comm, stream);
        errdefer agreement.deinit();
        var stage = try cuda.HostBuffer.alloc(d, region_bytes * regions);
        errdefer stage.free();
        x.* = .{
            .gpa = gpa,
            .d = d,
            .comm = comm,
            .stream = stream,
            .want = want,
            .active = .device,
            .k = k,
            .world = W,
            .hidden = hidden,
            .vocab_lo = vocab_lo,
            .rows_max = rows_max,
            .conf = conf,
            .send = send,
            .gathered = gathered,
            .host = host,
            .done = done,
            .agreement = agreement,
            .engine = .{ .d = d },
            .cache = undefined,
            .stage = stage,
            .region_bytes = region_bytes,
            .regions = regions,
        };
        const hold = if (std.c.getenv("TF_DSV41_GRAPH_FLOOR")) |v| std.mem.eql(u8, std.mem.span(v), "hold") else false;
        x.cache = graphs.gc.Cache(u32).init(gpa, x.engine.engine(), .{ .on = true, .max = max, .min_keep = 2, .hold = hold });
        x.cache.agree = x.agreement.agree();
        x.cache.room = graphs.room;
        x.cache.meter = graphs.meter;
        x.cache.name = "draft passes";
        std.log.scoped(.dsv41).info("draft graphs: {t} (candidates on the device, the first {d} passes checked against the host's)", .{ want, check_passes });
        return x;
    }

    pub fn deinit(x: *Dev) void {
        if (x.chain) |ch| ch.deinit();
        const cs = x.cache.stats;
        std.log.scoped(.dsv41).info("draft graphs: {t}, {d} passes, {d} graphed; {d} captured, {d} replayed, {d} eager; {d} checked, {d} mismatched", .{ x.active, x.stats.passes, x.stats.graphed, cs.captured, cs.replayed, cs.eager, x.checked, x.mismatches });
        x.cache.deinit();
        x.send.free();
        x.gathered.free();
        x.host.free();
        x.done.deinit();
        x.agreement.deinit();
        x.stage.free();
        x.gpa.destroy(x);
    }

    /// The device path is on (candidates on the GPU); false: the host step.
    pub fn on(x: *const Dev) bool {
        return x.active != .host;
    }

    /// Region `g`'s pinned staging for the statics: (int64 [n64], int32 [n32]); free once the previous round waited.
    pub fn statics(x: *Dev, g: usize, n64: usize, n32: usize) !struct { []i64, []i32 } {
        if (g >= x.regions or 8 * n64 + 4 * n32 > x.region_bytes) return error.Statics;
        try x.agreement.stream.d.synchronize();
        const b = x.stage.bytes[g * x.region_bytes ..][0..x.region_bytes];
        return .{ @alignCast(std.mem.bytesAsSlice(i64, b[0 .. 8 * n64])), @alignCast(std.mem.bytesAsSlice(i32, b[8 * n64 ..][0 .. 4 * n32])) };
    }

    /// ds_out on the device for `rows` rows at offset `at` (no sync or alloc; graphable): candidates, gather, copies.
    pub fn launch(x: *Dev, r: *run.Runner, logits_t: calls.Tensor, hid_t: calls.Tensor, rows: usize, at: usize) !void {
        if (at + rows > x.rows_max) return error.Rows;
        const k = x.k;
        const V: usize = @intCast(logits_t.shape[1]);
        const pk = rows * 2 * k * 4;
        const ops = r.kernels.others(r.stream);
        const send = x.send.ptr + at * 2 * k * 4;
        const dst = x.gathered.ptr + at * 2 * k * 4 * x.world;
        try ops.dsCandidates(try r.tensorAddr(logits_t), V, rows, V, k, x.vocab_lo, send);
        try x.comm.allGather(send, dst, rows * 2 * k, .f32, r.stream.handle);
        if (x.chain) |ch| try ch.layout(r, x.gathered.ptr, x.world, rows, k, at, try r.tensorAddr(hid_t));
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.host.bytes.ptr + at * 2 * k * 4 * x.world, dst, pk * x.world, r.stream.handle), "cuMemcpyDtoHAsync");
        if (x.conf) {
            const ho = x.rows_max * 2 * k * 4 * x.world + at * 2 * x.hidden;
            try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(x.host.bytes.ptr + ho, try r.tensorAddr(hid_t), 2 * rows * x.hidden, r.stream.handle), "cuMemcpyDtoHAsync");
        }
    }

    /// After the round's launches: one event the host waits on.
    pub fn mark(x: *Dev) !void {
        try x.done.record(x.stream);
    }

    pub fn wait(x: *Dev) !void {
        try x.done.synchronize();
    }

    /// Region `at`'s gathered candidates into rows of [W k] (`cand` / `cval`) and the head hidden as fp32 (`hid`).
    pub fn unpack(x: *Dev, rows: usize, at: usize, vocab: usize, cand: []i32, cval: []f32, hid: ?[]f32) !void {
        const k = x.k;
        const all: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, x.host.bytes[at * 2 * k * 4 * x.world ..][0 .. rows * 2 * k * 4 * x.world]));
        rows_mod.ungather(all, x.world, rows, k, cand, cval);
        if (hid) |h| {
            const ho = x.rows_max * 2 * k * 4 * x.world + at * 2 * x.hidden;
            const h16: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, x.host.bytes[ho..][0 .. 2 * rows * x.hidden]));
            for (h16, h[0 .. rows * x.hidden]) |b, *o| o.* = dspark.bf16(b);
        }
        for (cand[0 .. rows * x.world * k], 0..) |id, i| if (id < -1 or id >= @as(i64, @intCast(vocab))) {
            std.log.err("dspark: gathered candidate {d} at {d} is not a token id", .{ id, i });
            return error.BadCandidates;
        };
    }

    /// A check pass: this rank's logits rows on the host's rule against its part of region `at`; differences counted.
    pub fn check(x: *Dev, logits: u64, V: usize, rows: usize, at: usize) !void {
        if (x.checked >= check_passes) return;
        x.checked += 1;
        const k = x.k;
        const host = try x.gpa.alloc(f32, rows * V);
        defer x.gpa.free(host);
        try cuda.DeviceBuffer.download(.{ .d = x.d, .ptr = logits, .len = 4 * host.len }, 0, std.mem.sliceAsBytes(host));
        const ids = try x.gpa.alloc(i32, k);
        defer x.gpa.free(ids);
        const vals = try x.gpa.alloc(f32, k);
        defer x.gpa.free(vals);
        const rank = x.comm.rank();
        const all: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, x.host.bytes[at * 2 * k * 4 * x.world ..][0 .. rows * 2 * k * 4 * x.world]));
        for (0..rows) |i| {
            try dspark.candidates(x.gpa, host[i * V ..][0..V], x.vocab_lo, ids, vals);
            const got = all[(rank * rows + i) * 2 * k ..][0 .. 2 * k];
            for (0..k) |j| {
                const gid: i32 = @bitCast(got[k + j]);
                if (gid != ids[j] or @as(u32, @bitCast(got[j])) != @as(u32, @bitCast(vals[j]))) {
                    if (x.mismatches < 4) std.log.err("draft graphs: device candidates differ from the host's (row {d} slot {d}: {d} {e} vs {d} {e})", .{ i, j, gid, got[j], ids[j], vals[j] });
                    x.mismatches += 1;
                    break;
                }
            }
        }
    }

    /// After a pass (every rank): at the last check pass the ranks agree; all equal -> the device mode, else host step.
    pub fn settle(x: *Dev) !void {
        x.stats.passes += 1;
        if (x.active != .device or x.stats.passes != check_passes) return;
        const ag = x.agreement.agree();
        const ok = try ag.all(ag.ctx, x.mismatches == 0);
        if (ok) {
            x.active = x.want;
            std.log.scoped(.dsv41).info("draft graphs: {d} passes' device candidates equal the host's on every rank: {t}", .{ check_passes, x.active });
        } else {
            x.active = .host;
            std.log.err("draft graphs: device candidates differ on some rank ({d} rows here); the host step from here on", .{x.mismatches});
        }
    }

    /// The pass's launches as the key's graph (once checks passed and graphs are asked), else eagerly.
    pub fn issue(x: *Dev, key: u32, fp: u64, body: graphs.gc.Body) !void {
        if (x.active != .graphs) return body.run(body.ctx, x.stream);
        const out = try x.cache.run(key, x.stream, fp, body);
        if (out != .eager) x.stats.graphed += 1;
    }

    /// A speculative pass's launch: the key's graph when already captured, else eagerly; never a capture.
    pub fn issueReplay(x: *Dev, key: u32, fp: u64, body: graphs.gc.Body) !void {
        if (x.active == .graphs and !x.cache.failed and x.cache.has(key) and x.cache.fingerprint == fp) {
            _ = try x.cache.run(key, x.stream, fp, body);
            x.stats.graphed += 1;
            return;
        }
        return body.run(body.ctx, x.stream);
    }

    /// The device buffers a graphed pass bakes in besides the runner's roles.
    pub fn words(x: *const Dev, out: []u64) usize {
        const w = [_]u64{ x.send.ptr, x.gathered.ptr, @intFromPtr(x.host.bytes.ptr), @intFromPtr(x.stage.bytes.ptr) };
        @memcpy(out[0..w.len], &w);
        var m: usize = w.len;
        if (x.chain) |ch| for (ch.bufs) |b| {
            out[m] = b.ptr;
            m += 1;
        };
        return m;
    }
};

/// TF_DSV41_DRAFT_CHAIN: 1 = the chain on the device (with TF_DSV41_DRAFT_GRAPHS' device path), unset / 0 = the host's.
pub fn chainFromEnv() !bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_DRAFT_CHAIN") orelse return false);
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadDraftChain;
}

/// One strided copy of ds_out's reorder: `height` rows of `width` bytes; `ids`: into cand, else cval.
pub const Copy2D = struct { src: usize, src_pitch: usize, dst: usize, dst_pitch: usize, width: usize, height: usize, ids: bool };

/// The gathered region at round row `at` ([W][rows][2 k]) laid out as cand / cval [rows, W k]: 2 W copies.
pub fn reorder(world: usize, rows: usize, k: usize, at: usize, out: []Copy2D) []Copy2D {
    const c = world * k;
    const region = at * 2 * k * 4 * world;
    for (0..world) |w| for (0..2) |half| {
        const src = region + w * rows * 2 * k * 4 + half * k * 4;
        out[2 * w + half] = .{ .src = src, .src_pitch = 2 * k * 4, .dst = (at * c + w * k) * 4, .dst_pitch = c * 4, .width = k * 4, .height = rows, .ids = half == 1 };
    };
    return out[0 .. 2 * world];
}

/// The chain on the device (TF_DSV41_DRAFT_CHAIN=1): `_chain`'s tensors, pinned staging, a program per slot count.
pub const Chain = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    /// rows a slot (the block), the most slots a round, candidates a row (W k), the confidence head loaded
    n: usize,
    slots: usize,
    c: usize,
    hidden: usize,
    conf: bool,
    /// cand, cval, hid, anchor, ipar, fpar, draft, conf, scr (dspark_emit.chain_roles' order)
    bufs: [9]cuda.DeviceBuffer,
    /// pinned: ipar i64 [slots][4], anchor i32 [slots], fpar f32 [slots][3]; draft i32 [slots][n], conf f32 [slots][n]
    host: cuda.HostBuffer,
    arena: std.heap.ArenaAllocator,
    programs: [16]?[]const calls.Call = @splat(null),
    /// the slots of the round whose chain is in flight (0: none; the host chain runs)
    launched: usize = 0,
    /// a launch failed (no AOT variant, ...): the host's chain from then on
    off: bool = false,
    stats: struct { rounds: u64 = 0, slots: u64 = 0, failed: u64 = 0 } = .{},
    /// the check rounds' drafts (host chain over the same candidates) and how many differ from the device's
    checked: u64 = 0,
    differ: u64 = 0,

    /// Rounds whose drafts are also chained on the host and compared (the device path's own check passes come first).
    pub const check_rounds = 8;

    /// After the check rounds: past a quarter of drafts differing, the host's chain from here on (rank 0 alone).
    pub fn verdict(x: *Chain) void {
        std.log.scoped(.dsv41).info("draft chain on the device: {d} check drafts, {d} differ from the host chain (fp32 order)", .{ x.checked, x.differ });
        if (4 * x.differ > x.checked) {
            x.off = true;
            std.log.err("draft chain on the device: {d} of {d} check drafts differ from the host's: the host's chain from here on", .{ x.differ, x.checked });
        }
    }

    const roles = [_][]const u8{ emit.chain_roles.cand, emit.chain_roles.cval, emit.chain_roles.hid, emit.chain_roles.anchor, emit.chain_roles.ipar, emit.chain_roles.fpar, emit.chain_roles.draft, emit.chain_roles.conf, emit.chain_roles.scr };

    pub fn init(gpa: std.mem.Allocator, r: *run.Runner, n: usize, slots: usize, c: usize, hidden: usize, conf: bool) !*Chain {
        if (c > emit.chain_kp or slots > 16) return error.Shape;
        const x = try gpa.create(Chain);
        errdefer gpa.destroy(x);
        const rows = n * slots;
        const sizes = [9]usize{ rows * c * 4, rows * c * 4, rows * hidden * 2, slots * 4, slots * 32, slots * 12, slots * n * 4, slots * n * 4, slots * 2 * emit.chain_kp * 8 };
        var bufs: [9]cuda.DeviceBuffer = undefined;
        var made: usize = 0;
        errdefer for (bufs[0..made]) |*b| b.free();
        for (&bufs, sizes) |*b, sz| {
            b.* = try cuda.DeviceBuffer.alloc(r.d, @max(sz, 256));
            try b.fill8(0, null);
            made += 1;
        }
        x.* = .{ .gpa = gpa, .d = r.d, .n = n, .slots = slots, .c = c, .hidden = hidden, .conf = conf, .bufs = bufs, .host = try cuda.HostBuffer.alloc(r.d, hostBytes(n, slots)), .arena = std.heap.ArenaAllocator.init(gpa) };
        for (roles, bufs) |role, b| try r.external(role, b.ptr);
        return x;
    }

    fn hostBytes(n: usize, slots: usize) usize {
        return slots * (32 + 4 + 12) + 2 * slots * n * 4;
    }

    pub fn deinit(x: *Chain) void {
        if (x.stats.rounds + x.stats.failed > 0) std.log.scoped(.dsv41).info("draft chain on the device: {d} rounds, {d} slots, {d} failed launches", .{ x.stats.rounds, x.stats.slots, x.stats.failed });
        for (&x.bufs) |*b| b.free();
        x.host.free();
        x.arena.deinit();
        x.gpa.destroy(x);
    }

    /// ds_out's part (graphable): the region at round row `at` as cand / cval rows, the head hidden kept at those rows.
    pub fn layout(x: *Chain, r: *run.Runner, gathered: u64, world: usize, rows: usize, k: usize, at: usize, hid: u64) !void {
        var cs: [32]Copy2D = undefined;
        for (reorder(world, rows, k, at, &cs)) |cp| {
            const p: cuda.abi.Memcpy2D = .{
                .src = gathered + cp.src,
                .src_pitch = cp.src_pitch,
                .dst = (if (cp.ids) x.bufs[0].ptr else x.bufs[1].ptr) + cp.dst,
                .dst_pitch = cp.dst_pitch,
                .width = cp.width,
                .height = cp.height,
            };
            try r.d.check(r.d.api.cuMemcpy2DAsync_v2(&p, r.stream.handle), "cuMemcpy2DAsync");
        }
        if (x.conf) try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.bufs[2].ptr + at * x.hidden * 2, hid, rows * x.hidden * 2, r.stream.handle), "cuMemcpyDtoDAsync");
    }

    /// The first slot count whose `_chain` launch the AOT set cannot serve (null: every one), probed at boot.
    pub fn probe(x: *Chain, r: *const run.Runner, cfg: anytype, w: anytype, o: anytype) !?usize {
        for (1..x.slots + 1) |S| if (try r.missingTriton(try x.program(cfg, w, o, S)) != null) return S;
        return null;
    }

    fn program(x: *Chain, cfg: anytype, w: anytype, o: anytype, S: usize) ![]const calls.Call {
        if (x.programs[S - 1]) |p| return p;
        const p = try emit.emitChain(x.arena.allocator(), cfg, w, o, @intCast(x.n), @intCast(S), @intCast(x.c), x.conf);
        x.programs[S - 1] = p;
        return p;
    }

    /// The round's chain after its passes on the runner's stream; false: not launched (off, or failed: host chain).
    pub fn launch(x: *Chain, r: *run.Runner, cfg: anytype, w: anytype, o: anytype, anchors: []const u32, params: []const dspark.Params) !bool {
        const S = anchors.len;
        x.launched = 0;
        if (x.off or S == 0) return false;
        if (S > x.slots) return error.Slots;
        const a = stageArgs(x.host.bytes, x.slots, anchors, params);
        x.issue(r, cfg, w, o, S, a) catch |e| {
            x.off = true;
            x.stats.failed += 1;
            std.log.err("draft chain on the device: the launch failed ({t}); the host's chain from here on", .{e});
            return false;
        };
        x.launched = S;
        x.stats.rounds += 1;
        x.stats.slots += S;
        return true;
    }

    fn issue(x: *Chain, r: *run.Runner, cfg: anytype, w: anytype, o: anytype, S: usize, a: Args) !void {
        const s = r.stream.handle;
        try cuda.DeviceBuffer.uploadAsync(x.bufs[4], 0, a.ipar, s);
        try cuda.DeviceBuffer.uploadAsync(x.bufs[3], 0, a.anchor, s);
        try cuda.DeviceBuffer.uploadAsync(x.bufs[5], 0, a.fpar, s);
        try r.window(try x.program(cfg, w, o, S));
        const res = x.results();
        try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(res.draft.ptr), x.bufs[6].ptr, 4 * S * x.n, s), "cuMemcpyDtoHAsync");
        if (x.conf) try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(res.conf.ptr), x.bufs[7].ptr, 4 * S * x.n, s), "cuMemcpyDtoHAsync");
    }

    pub const Args = struct { ipar: []const u8, anchor: []const u8, fpar: []const u8 };

    /// Stages the chain inputs a slot each: ipar (seed, POS0, top_k, sampled), anchor, fpar (T, top_p, min_p) as fp32.
    pub fn stageArgs(host: []u8, slots: usize, anchors: []const u32, params: []const dspark.Params) Args {
        const S = anchors.len;
        const ipar: []i64 = @alignCast(std.mem.bytesAsSlice(i64, host[0 .. slots * 32]));
        const anchor: []i32 = @alignCast(std.mem.bytesAsSlice(i32, host[slots * 32 ..][0 .. slots * 4]));
        const fpar: []f32 = @alignCast(std.mem.bytesAsSlice(f32, host[slots * 36 ..][0 .. slots * 12]));
        for (anchors, params, 0..) |an, p, j| {
            ipar[4 * j ..][0..4].* = .{ @bitCast(p.seed), @intCast(p.pos0), p.top_k, @intFromBool(p.sampled) };
            anchor[j] = @intCast(an);
            fpar[3 * j ..][0..3].* = .{ @floatCast(p.temperature), @floatCast(p.top_p), @floatCast(p.min_p) };
        }
        return .{ .ipar = std.mem.sliceAsBytes(ipar[0 .. 4 * S]), .anchor = std.mem.sliceAsBytes(anchor[0..S]), .fpar = std.mem.sliceAsBytes(fpar[0 .. 3 * S]) };
    }

    /// The pinned results (valid after the round's event): draft int32 [slots][n], conf fp32 [slots][n].
    pub fn results(x: *const Chain) struct { draft: []i32, conf: []f32 } {
        const base = x.slots * 48;
        const m = x.slots * x.n;
        return .{
            .draft = @alignCast(std.mem.bytesAsSlice(i32, x.host.bytes[base..][0 .. 4 * m])),
            .conf = @alignCast(std.mem.bytesAsSlice(f32, x.host.bytes[base + 4 * m ..][0 .. 4 * m])),
        };
    }
};

/// The fingerprint of a pass: the runner's owned buffers, its scratch, the pass's own and `extra`.
pub fn fingerprint(r: *const run.Runner, x: *const Dev, extra: []const u64) u64 {
    var words: [80]u64 = undefined;
    var m: usize = 0;
    for (r.owned.items) |b| {
        if (m + 2 > 60) break;
        words[m] = b.ptr;
        words[m + 1] = b.len;
        m += 2;
    }
    if (r.scratch) |s| {
        words[m] = s.ptr;
        m += 1;
    }
    m += x.words(words[m..]);
    for (extra) |e| if (m < words.len) {
        words[m] = e;
        m += 1;
    };
    return graphs.gc.fingerprintOf(words[0..m]);
}

test "draft graphs: the knob" {
    try std.testing.expectEqual(Mode.host, try modeFromEnv());
}
