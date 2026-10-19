//! The forward's Engram rows: hashing, staging, the gate, read-ahead and the all-gather.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const eh = @import("engram_host.zig");
const er = @import("engram_rows.zig");
const hooks = @import("forward_hooks.zig");
const ft = @import("forward_types.zig");
const ph = @import("phases.zig");
const hs = @import("host_stage.zig");
const fwd = @import("forward.zig");
const fwd_glue = @import("forward_glue.zig");
const Forward = fwd.Forward;
const glueMissing = fwd_glue.glueMissing;

/// A window's Engram rows of layer L: hashes, this rank's heads' rows, the all-gather and the column reorder.
pub fn engramRows(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const h = f.engram orelse return glueMissing("engram_rows (no Engram tables)");
    const L: u32 = @intCast(c.args[1].arg.i);
    const n = f.ids.len; // a multi-segment run: every segment's rows (each with its slot's tail)
    const W = f.comm.world();
    const cols = f.engramCols(h); // this rank's columns of a row
    const bytes = 2 * n * cols;
    if (f.gated) {
        // the gate: the GPU waits for the worker's rows and copies them where the staged copy would land
        const send = try f.sendBuffer(r, bytes);
        try f.gateCopy(r, L, send, n);
        if (f.gathered == null or f.gathered.?.len < W * bytes) {
            if (f.gathered) |*g| g.free();
            f.gathered = null;
            f.gathered = try cuda.DeviceBuffer.alloc(r.d, W * bytes);
        }
        try f.comm.allGather(send, f.gathered.?.ptr, n * cols, .bf16, r.stream.handle);
        return r.kernels.others(r.stream).gatherCols(f.gathered.?.ptr, W, n, cols, try r.tensorAddr(c.args[0].arg.t));
    }
    // an earlier copy out of this layer's staging must have left: a segment waits for it, smaller ones the stream
    const li = std.mem.indexOfScalar(u32, h.layer_ids, L) orelse return error.NotEngram;
    if (n <= hs.chunk) try r.stream.synchronize();
    const stage = &f.rows_stage[li];
    const host = try stage.take(r.d, bytes);
    try f.fillEngramRows(h, L, @alignCast(std.mem.bytesAsSlice(u16, host)));
    const send = try f.sendBuffer(r, bytes);
    const s = r.stream.handle;
    try stage.copyTo(.{ .d = r.d, .ptr = send, .len = bytes }, 0, r.stream);
    if (f.gathered == null or f.gathered.?.len < W * bytes) {
        if (f.gathered) |*g| g.free();
        f.gathered = null;
        f.gathered = try cuda.DeviceBuffer.alloc(r.d, W * bytes);
    }
    try f.comm.allGather(send, f.gathered.?.ptr, n * cols, .bf16, s);
    return r.kernels.others(r.stream).gatherCols(f.gathered.?.ptr, W, n, cols, try r.tensorAddr(c.args[0].arg.t));
}

/// This rank's columns of an Engram row (its heads' share).
pub fn engramCols(f: *const Forward, h: *const eh.Host) usize {
    return h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
}

/// Layer L's rows of the window's ids (with the slot's tail): the hashes, this rank's heads' index rows.
pub fn fillEngramRows(f: *Forward, h: *const eh.Host, L: u32, out: []u16) !void {
    const tw = ph.start(f.phases, .@"engram.wait");
    defer tw.stop();
    if (f.multi) |m| {
        // a multi-segment run: each segment's rows after its own slot's tail, at its rows of the run
        const cols = out.len / f.ids.len;
        for (m.segs, m.row0, m.tails) |sg, r0, t| try h.windowRows(f.gpa, try f.engramSource(h), sg.ids, t.items(), L, f.comm.rank(), f.comm.world(), f.cfg.engram_head_dim, out[r0 * cols ..][0 .. sg.ids.len * cols]);
        return;
    }
    return h.windowRows(f.gpa, try f.engramSource(h), f.ids, f.slot.tail[0..f.slot.tail_len], L, f.comm.rank(), f.comm.world(), f.cfg.engram_head_dim, out);
}

/// The Engram rows' source, opened on first use (TF_DSV41_ENGRAM_DIR; TF_DSV41_ENGRAM_CACHE records a layer).
pub fn engramSource(f: *Forward, h: *const eh.Host) !?eh.Source {
    if (!f.engram_opened) {
        f.engram_opened = true;
        if (std.c.getenv("TF_DSV41_ENGRAM_PREFETCH")) |v| f.engram_prefetch = !std.mem.eql(u8, std.mem.span(v), "0");
        if (std.c.getenv("TF_DSV41_ENGRAM_DIR")) |dir| {
            const cap: u32 = if (std.c.getenv("TF_DSV41_ENGRAM_CACHE")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 65536;
            f.engram_rows = try er.Rows.open(f.gpa, std.mem.span(dir), f.comm.rank(), f.comm.world(), h.layer_ids, cap);
            std.log.scoped(.dsv41).info("engram: rows from {s} (io_uring {s}, {d} records a layer cached)", .{ std.mem.span(dir), if (@hasField(er.Rows, "ring") and f.engram_rows.?.ring != null) "on" else "off", cap });
        } else if (!quiet) std.log.scoped(.dsv41).warn("engram: TF_DSV41_ENGRAM_DIR unset: synthetic index rows (the forward gate's), not the model's table", .{});
        try (try hooks.gate()).open(f, h);
    }
    if (f.gate) |g| return hooks.set.gate.?.source(g);
    return if (f.engram_rows) |x| x.source() else null;
}

/// TF_DSV41_ENGRAM_GATE: arms the gate for a window over `items`; false when off or too wide (rows staged instead).
pub fn armEngram(f: *Forward, items: []const ft.Item, fill: u32) !bool {
    const h = f.engram orelse return false;
    _ = try f.engramSource(h); // the table and the gate open on the first Engram window
    const g = f.gate orelse return false;
    const gh = hooks.set.gate.?;
    if (!gh.eligible(g, fill)) return false;
    const ts = ph.start(f.phases, .@"engram.wait");
    defer ts.stop();
    try gh.arm(g, items, fill);
    return true;
}

/// After a window's launches (one-slot or row): a gated window's end recorded for the next arm (`syncGatedEnd`).
pub fn gatedEnd(f: *Forward) !void {
    if (!f.gated or f.gate == null or !(try (try hooks.roundGraph()).current()).gate) return;
    if (f.gated_end == null) f.gated_end = try cuda.Event.init(f.runner.d, false);
    try f.gated_end.?.record(f.runner.stream);
}

/// The armed round's errors (a failed read, a GPU wait past the timeout); nothing when no window was gated.
pub fn engramCheck(f: *Forward) !void {
    if (f.gate) |g| if (f.gated) try hooks.set.gate.?.check(g);
}

/// The gate kernel of Engram layer L: `n` rows of the armed round's staging into `dst` once published.
pub fn gateCopy(f: *Forward, r: *run.Runner, L: u32, dst: u64, n: usize) !void {
    return hooks.set.gate.?.copy(f, r, L, dst, n);
}

/// Every Engram layer's reads of a window in flight at once (one read latency); off: TF_DSV41_ENGRAM_PREFETCH=0.
pub fn issueEngram(f: *Forward, ids: []const u32, tail: []const u32) void {
    const h = f.engram orelse return;
    const src = (f.engramSource(h) catch return) orelse return;
    if (!f.engram_prefetch or h.layer_ids.len < 2) return;
    h.prefetch(f.gpa, src, ids, tail, f.comm.rank(), f.comm.world());
}

/// The coming window's rows `ids` at `start` read ahead on this rank; a wrong guess costs a read, never a bit.
pub fn prefetchEngram(f: *Forward, start: u64, ids: []const u32) void {
    const h = f.engram orelse return;
    const src = (f.engramSource(h) catch return) orelse return;
    if (!f.engram_prefetch) return;
    const tp_ = ph.start(f.phases, .prefetch);
    defer tp_.stop();
    var buf: [2 * eh.max_ngram + 128]u32 = undefined;
    const pend: ?[]const u32 = if (f.slot.pending) |p| f.ids[0..p.n] else null;
    const lb = eh.lookbackAt(f.slot.tail[0..f.slot.tail_len], f.slot.pos, pend, start, h.max_ngram - 1, &buf) orelse return;
    // the gate's worker owns the table between windows: a warm job, not a read on this thread
    if (f.gate) |g| return hooks.set.gate.?.warm(g, &.{.{ .ids = ids, .tail = lb }});
    h.prefetch(f.gpa, src, ids, lb, f.comm.rank(), f.comm.world());
}
