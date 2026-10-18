//! Row windows' program, row table, glue and CUDA graphs.

const std = @import("std");
const ph = @import("phases.zig");
const cuda = @import("cuda");
const kv = @import("kv");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");
const graphs = @import("graphs.zig");
const rowtab = @import("rowtab.zig");
const rowmode = @import("rowmode.zig");
const batch = @import("batch.zig");
const Batch = batch.Batch;
const rowOptions = batch.rowOptions;

pub fn body(ctx: *anyopaque, stream: cuda.Stream) anyerror!void {
    const b: *Batch = @ptrCast(@alignCast(ctx));
    _ = stream; // the runner's stream: the one the cache captures
    try b.f.runner.window(b.current[b.span[0]..b.span[1]]);
}

/// The key's program: block.emit at R rows ending at the context bucket's end, in row mode.
pub fn program(b: *Batch, key: graphs.Key, s: graphs.Settings) ![]const calls.Call {
    if (b.programs.get(key)) |cs| return cs;
    const t0 = ph.nowNs();
    defer {
        b.stats.builds += 1;
        b.stats.build_ns += ph.nowNs() - t0;
    }
    const f = b.f;
    const a = f.arena.allocator();
    const start = graphs.emitStart(key.rows, key.ctx, s, @intCast(f.opts.limit));
    const one = try block.emit(a, f.cfg, f.widths, rowOptions(f), try f.backbone(a), key.rows, start, true);
    const cs = try rowmode.transform(a, one, key.rows, .{ .slots = b.ss.n, .rmax = b.cap, .pts = f.kv.?.slot.max_pages });
    try b.taps_at.put(b.gpa, key, run.lastTaps(cs));
    try b.programs.put(b.gpa, key, cs);
    return cs;
}

pub fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const b: *Batch = @ptrCast(@alignCast(ctx));
    const f = b.f;
    const step = c.name["glue.".len..];
    const k = r.kernels.others(r.stream);
    const s = r.stream.handle;
    const R: usize = b.R;
    const D: usize = f.cfg.hidden;
    const W: usize = f.comm.world();
    if (std.mem.eql(u8, step, "positions")) {
        // positions come from the table; the SWA lower bounds are zeros (Context.lo)
        const lo = try r.tensorAddr(c.args[2].arg.t);
        return r.d.check(r.d.api.cuMemsetD32Async(lo, 0, R, s), "cuMemsetD32Async");
    }
    if (std.mem.eql(u8, step, "embed")) {
        const embed = r.weights.get("embed") orelse return error.MissingWeight;
        const V: usize = f.cfg.vocab / W;
        const rows = try f.sendBuffer(r, 2 * R * D);
        try k.embedSend(b.tab.dev.ptr + rowtab.colOffset(.ids, b.cap), R, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
        const recv = try r.tensorAddr(c.args[1].arg.t);
        try f.comm.allGather(rows, recv, R * D, .bf16, s);
        return k.embedSum(recv, W, R, D, try r.tensorAddr(c.args[0].arg.t));
    }
    if (std.mem.eql(u8, step, "engram_rows")) {
        const L: u32 = @intCast(c.args[1].arg.i);
        for (b.stages.items) |*st| if (st.layer == L) {
            if (f.gated) try f.gateCopy(r, L, st.dev.ptr, R);
            try f.comm.allGather(st.dev.ptr, f.gathered.?.ptr, R * st.cols, .bf16, s);
            return k.gatherCols(f.gathered.?.ptr, W, R, st.cols, try r.tensorAddr(c.args[0].arg.t));
        };
        return error.NotEngram;
    }
    if (std.mem.eql(u8, step, "kx_dense")) return b.exchange(r, c);
    if (std.mem.eql(u8, step, "trace")) return error.TracedRows;
    return fwd.Forward.glueFn(f, r, c);
}

/// Split KV's dense exchange in row mode: each row's selection through its slot's row of the stacked split table.
pub fn exchange(b: *Batch, r: *run.Runner, c: *const calls.Call) !void {
    const kx = b.f.kv orelse return error.GlueNotBuilt;
    const x = c.args;
    if (x.len != 7) return error.BadGlue;
    const sel = x[0].arg.t;
    const da: kv.split.DenseArgs = .{
        .sel = try r.tensorAddr(sel),
        .rows = @intCast(sel.shape[0]),
        .k = @intCast(sel.shape[1]),
        .table = try r.tensorAddr(x[2].arg.t),
        .pts = kx.slot.max_pages,
        .rslot = b.tab.dev.ptr + rowtab.rslot32Offset(b.cap),
        .psh = @intCast(x[6].arg.i),
        .base = try r.tensorAddr(x[1].arg.t),
        .row_bytes = block.Pool.row_bytes,
        .world = kx.world,
        .send = try r.tensorAddr(x[3].arg.t),
        .tok = try r.tensorAddr(x[5].arg.t),
    };
    try kx.x.window(kv.split.PackArgs.of(da, kx.rank, kx.lens.?.ptr), try r.tensorAddr(x[4].arg.t), r.stream.handle);
    kx.stats.dense += 1;
}

pub fn fingerprint(b: *const Batch) u64 {
    const f = b.f;
    var h = std.hash.Wyhash.init(0x5107);
    const r = f.runner;
    for (r.owned.items) |x| {
        h.update(std.mem.asBytes(&x.ptr));
        h.update(std.mem.asBytes(&x.len));
    }
    for ([_]?cuda.DeviceBuffer{ r.scratch, f.send, f.gathered, b.tab.dev }) |x| if (x) |y| h.update(std.mem.asBytes(&y.ptr));
    for (b.stages.items) |st| h.update(std.mem.asBytes(&st.dev.ptr));
    for (b.ss.roles.values()) |st| h.update(std.mem.asBytes(&st.base));
    if (f.kv) |k| for (k.dev.tensors) |t| h.update(std.mem.asBytes(&t.ptr));
    // the taps mark's event: a graph captured without it (or with another) records no mark, so it is dropped
    if (r.taps_mark) |m| h.update(std.mem.asBytes(&m.ev.handle));
    return h.final();
}

/// The row table on the device and two pinned staging halves; a half is rewritten only after its copy finished.
pub const Table = struct {
    rmax: u32,
    dev: cuda.DeviceBuffer,
    stages: [2]@import("host_stage.zig").Stage = @splat(.{}),
    half: u1 = 0,
    pub fn init(d: *const cuda.Driver, rmax: u32) !Table {
        return .{ .rmax = rmax, .dev = try cuda.DeviceBuffer.alloc(d, rowtab.tableBytes(rmax)) };
    }
    pub fn drain(t: *Table) !void {
        for (&t.stages) |*st| try st.wait();
    }
    pub fn deinit(t: *Table) void {
        for (&t.stages) |*st| st.close() catch unreachable;
        t.dev.free();
    }
    pub fn stage(t: *Table, stream: cuda.Stream, mix: []const rowtab.Seg, ids: []const u32, R: u32, nslots: u32) !void {
        const st = &t.stages[t.half];
        const bytes = try st.take(stream.d, rowtab.tableBytes(t.rmax));
        @memset(bytes, 0);
        const cols: []i64 = @alignCast(std.mem.bytesAsSlice(i64, bytes[0 .. 8 * rowtab.ncol * @as(usize, t.rmax)]));
        const r32: []i32 = @alignCast(std.mem.bytesAsSlice(i32, bytes[rowtab.rslot32Offset(t.rmax)..][0 .. 4 * @as(usize, t.rmax)]));
        try rowtab.plan(mix, ids, R, t.rmax, nslots, cols, r32);
        try st.copyTo(t.dev, 0, stream);
        t.half +%= 1;
    }
};

/// TF_DSV41_GRAPH_LOG=1: every row-graph capture logged with its key and host ms (else the powers of two).
pub fn graphLog() bool {
    const v = std.c.getenv("TF_DSV41_GRAPH_LOG") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

pub fn outcomeLog(out: graphs.gc.Outcome, g: *const Graphs) bool {
    return out == .captured and (g.cache.stats.captured & (g.cache.stats.captured - 1)) == 0;
}

/// Row graphs' state: the cache over the runner's stream, keyed (bucket rows, context bucket), every rank agreeing.
pub const Graphs = struct {
    engine: graphs.gc.CudaEngine,
    agreement: graphs.Agreement,
    cache: graphs.Cache,

    /// `split`: a window's two graphs (head, tail) held in one cache of twice the cap (a head is ~1/43 of a window)
    pub fn init(g: *Graphs, f: *fwd.Forward, s: graphs.Settings, split: bool) !void {
        const r = f.runner;
        g.* = .{ .engine = .{ .d = r.d }, .agreement = try graphs.Agreement.init(r.d, f.comm, r.stream), .cache = undefined };
        g.cache = graphs.Cache.init(f.gpa, g.engine.engine(), .{ .on = true, .max = if (split) 2 * s.max else s.max, .hold = s.floor_hold });
        g.cache.agree = g.agreement.agree();
        g.cache.room = graphs.room;
        g.cache.meter = graphs.meter;
        g.cache.name = "row graphs";
        graphs.rule = s.room;
    }

    pub fn deinit(g: *Graphs) void {
        const cs = g.cache.stats;
        std.log.scoped(.dsv41).info("row graphs: {d} captured, {d} replayed, {d} eager, {d} keys held, {d} evicted, {d} dropped{s}", .{ cs.captured, cs.replayed, cs.eager, g.cache.count(), cs.evicted, cs.dropped, if (g.cache.failed) " (off after a failed capture)" else "" });
        g.cache.deinit();
        g.agreement.deinit();
    }
};
