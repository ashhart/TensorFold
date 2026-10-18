//! The decode window's glue steps: positions, embedding, exchanges, selections, traces.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const hs = @import("host_stage.zig");
const fwd = @import("forward.zig");
const hooks = @import("forward_hooks.zig");
const Error = fwd.Error;
const Forward = fwd.Forward;

pub fn glueMissing(name: []const u8) Error {
    if (!quiet) std.log.scoped(.dsv41).warn("glue step {s}: its kernel is not built yet (no glue kernel binds it)", .{name});
    return error.GlueNotBuilt;
}

/// The materialised selection's glue steps ("topk", "counts", "block_pos"); false: not one of them.
pub fn selectionGlue(k: anytype, r: *run.Runner, c: *const calls.Call, step: []const u8) !bool {
    const x = c.args;
    if (std.mem.eql(u8, step, "topk")) {
        const keys = x[0].arg.t;
        const out = x[1].arg.t;
        const count: usize = @intCast(x[2].arg.i);
        if (out.shape[0] != keys.shape[0] or out.shape[1] != count) return error.BadGlue;
        try k.topPositions(try r.tensorAddr(keys), @intCast(keys.stride[0]), @intCast(keys.shape[0]), @intCast(keys.shape[1]), count, try r.tensorAddr(out));
        return true;
    }
    if (std.mem.eql(u8, step, "counts")) {
        const sel = x[0].arg.t;
        try k.visibleCounts(try r.tensorAddr(sel), @intCast(sel.shape[1]), try r.tensorAddr(x[2].arg.t), @intCast(x[3].arg.i), @intCast(sel.shape[0]), try r.tensorAddr(x[1].arg.t));
        return true;
    }
    if (std.mem.eql(u8, step, "block_pos")) {
        const v: i32 = @intCast(x[1].arg.i);
        try r.d.check(r.d.api.cuMemsetD32Async(try r.tensorAddr(x[0].arg.t), @bitCast(v), 1, r.stream.handle), "cuMemsetD32Async");
        return true;
    }
    return false;
}

pub fn numel(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |x| n *= @intCast(x);
    return n;
}

pub fn sendBuffer(f: *Forward, r: *run.Runner, bytes: usize) !u64 {
    if (f.send == null or f.send.?.len < bytes) {
        if (f.send) |*b| b.free();
        f.send = null;
        f.send = try cuda.DeviceBuffer.alloc(r.d, bytes);
    }
    return f.send.?.ptr;
}

/// The decode window's glue (a prefill segment's handler hands it the steps both have).
pub fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const step = c.name["glue.".len..];
    if (std.mem.startsWith(u8, step, "kx_")) return (f.kv orelse return error.GlueNotBuilt).glue(r, c, step);
    const s = r.stream.handle;
    const k = r.kernels.others(r.stream);
    if (try selectionGlue(k, r, c, step)) return;
    const p = f.slot.pending.?;
    const D: usize = f.cfg.hidden;
    const W: usize = f.comm.world();
    const g = f.activeGraph();
    if (std.mem.eql(u8, step, "positions")) {
        const t = [_]calls.Tensor{ c.args[0].arg.t, c.args[1].arg.t, c.args[2].arg.t };
        // graphed: the start staged in the statics (one graph serves every window of its key)
        if (g) |gr| return k.positionsDev(gr.statics.startDev(), p.n, try r.tensorAddr(t[0]), try r.tensorAddr(t[1]), try r.tensorAddr(t[2]));
        try k.positions(@intCast(p.start), p.n, try r.tensorAddr(t[0]), try r.tensorAddr(t[1]), try r.tensorAddr(t[2]));
        // CED's decoder replay: every row's window starts at the pass's first row (replay.finish's ctx.lo)
        if (c.args.len == 4) try r.d.check(r.d.api.cuMemsetD32Async(try r.tensorAddr(t[2]), @bitCast(@as(i32, @intCast(c.args[3].arg.i))), p.n, s), "cuMemsetD32Async");
        return;
    }
    if (g != null and std.mem.eql(u8, step, "embed")) {
        // graphed: the ids staged before the launch; no host step inside the window
        const n: usize = p.n;
        const embed = r.weights.get("embed") orelse return error.MissingWeight;
        const V: usize = f.cfg.vocab / W;
        const rows = try f.sendBuffer(r, 2 * n * D);
        try k.embedSend(g.?.statics.idsDev(), n, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
        const recv = try r.tensorAddr(c.args[1].arg.t);
        try f.comm.allGather(rows, recv, n * D, .bf16, s);
        return k.embedSum(recv, W, n, D, try r.tensorAddr(c.args[0].arg.t));
    }
    if (std.mem.eql(u8, step, "embed")) {
        // this rank's vocabulary rows (others +0), their all-sum (the window's first exchange), x4 into the streams
        const n: usize = p.n;
        if (f.dev_ids == null or f.dev_ids.?.len < 8 * n) {
            if (f.dev_ids) |*b| b.free();
            f.dev_ids = null;
            f.dev_ids = try cuda.DeviceBuffer.alloc(r.d, @max(8 * n, 256));
        }
        // the previous copy out of this staging finished: a segment waits for it alone, smaller windows the stream
        if (n <= hs.chunk) try r.stream.synchronize();
        const h = try f.ids_stage.take(r.d, 8 * n);
        for (f.ids, 0..) |id, i| std.mem.writeInt(i64, h[8 * i ..][0..8], id, .little);
        try f.ids_stage.copyTo(f.dev_ids.?, 0, r.stream);
        const embed = r.weights.get("embed") orelse return error.MissingWeight;
        const V: usize = f.cfg.vocab / W;
        const rows = try f.sendBuffer(r, 2 * n * D);
        try k.embedSend(f.dev_ids.?.ptr, n, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
        if (f.vision) |vr| try hooks.set.vision.?.fill(vr, r.stream, f.ids, rows, D); // rank 0: the image positions' rows (vision.fill)
        const recv = try r.tensorAddr(c.args[1].arg.t);
        try f.comm.allGather(rows, recv, n * D, .bf16, s);
        return k.embedSum(recv, W, n, D, try r.tensorAddr(c.args[0].arg.t));
    }
    if (std.mem.eql(u8, step, "exchange")) {
        // attention's bf16 partial gathered into the receive buffer, rank 0's first
        const send = c.args[0].arg.t;
        return f.comm.allGather(try r.tensorAddr(send), try r.tensorAddr(c.args[1].arg.t), numel(send), .bf16, s);
    }
    if (std.mem.eql(u8, step, "exchange_rows")) {
        // TF_DSV41_PF_OVERLAP: piece rows all-gathered into "L.xrows", copied into place: same bytes
        const send = c.args[0].arg.t;
        const into = c.args[1].arg.t;
        const r0: usize = @intCast(c.args[2].arg.i);
        const n: usize = @intCast(into.shape[1]);
        const m = numel(send) / D;
        if (r0 + m > n or c.args.len < 4) return error.BadGlue;
        const tmp = try r.tensorAddr(c.args[3].arg.t);
        try f.comm.allGather(try r.tensorAddr(send), tmp, m * D, .bf16, s);
        const base = try r.tensorAddr(into);
        for (0..W) |q| try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(base + ((q * n + r0) * D) * 2, tmp + q * m * D * 2, m * D * 2, s), "cuMemcpyDtoDAsync");
        return;
    }
    if (std.mem.eql(u8, step, "exchange_f32")) {
        // the MoE's fp32 partial rounded to bf16, then gathered
        const part = c.args[0].arg.t;
        const n = numel(part);
        const half = try f.sendBuffer(r, 2 * n);
        try k.castBf16(try r.tensorAddr(part), half, n);
        return f.comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
    }
    if (std.mem.eql(u8, step, "kit_weights"))
        return k.kitWeights(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
    if (std.mem.eql(u8, step, "kit_logits")) return k.kitLogits(try r.tensorAddr(c.args[0].arg.t), numel(c.args[0].arg.t));
    if (std.mem.eql(u8, step, "engram_rows")) {
        if (g) |gr| return f.engramGathered(gr, r, c);
        return f.engramRows(r, c);
    }
    if (std.mem.eql(u8, step, "trace")) {
        const map = f.trace orelse return;
        const t = c.args[0].arg.t;
        const host = try f.gpa.alloc(u8, 2 * numel(t));
        defer f.gpa.free(host);
        try r.stream.synchronize();
        try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(t), .len = host.len }, 0, host);
        var sha: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(host, &sha, .{});
        return map.put(f.gpa, @intCast(c.args[1].arg.i), std.fmt.bytesToHex(sha, .lower));
    }
    return glueMissing(step);
}
