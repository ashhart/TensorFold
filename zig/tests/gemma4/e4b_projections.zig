//! Gemma 4 E4B projections on real rows and weights through the decode path and affine_mm's tiles, for the FP64 check.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const g4 = tf.gemma4;
const fwd = g4.forward;
const shared = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

fn npy(io: std.Io, gpa: std.mem.Allocator, dir: []const u8, name: []const u8, vals: []const u16, rows: usize, cols: usize) !void {
    var dict: [128]u8 = undefined;
    const d = try std.fmt.bufPrint(&dict, "{{'descr': '<f4', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ rows, cols });
    var head: [192]u8 = undefined;
    const len = (10 + d.len + 1 + 63) / 64 * 64;
    @memcpy(head[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, head[8..10], @intCast(len - 10), .little);
    @memcpy(head[10..][0..d.len], d);
    @memset(head[10 + d.len .. len - 1], ' ');
    head[len - 1] = '\n';
    var path: [512]u8 = undefined;
    const file = try std.Io.Dir.cwd().createFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.npy", .{ dir, name }), .{});
    defer file.close(io);
    var wb: [1 << 16]u8 = undefined;
    var w = file.writer(io, &wb);
    try w.interface.writeAll(head[0..len]);
    const flat = try gpa.alloc(f32, rows * cols);
    defer gpa.free(flat);
    for (flat, vals[0 .. rows * cols]) |*f, b| f.* = @bitCast(@as(u32, b) << 16);
    try w.interface.writeAll(std.mem.sliceAsBytes(flat));
    try w.interface.flush();
}

fn finish(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("gpu failed: {s}\n", .{why});
        return error.GpuFailed;
    }
}

/// A copy of `from`'s first `bytes` bytes on the host (its own command buffer: untracked buffers order only by waits).
fn grab(m: *g4.Model, gpa: std.mem.Allocator, from: mtl.Buffer, bytes: usize) ![]u16 {
    const stage = try m.device.buffer(bytes, shared);
    defer stage.deinit();
    const cb = m.queue.commandBuffer();
    const b = cb.blit();
    b.copy(from, 0, stage, 0, bytes);
    b.end();
    try finish(cb);
    return gpa.dupe(u16, stage.slice(u16, bytes / 2));
}

const Class = struct {
    name: []const u8,
    x: []const u16,
    k: usize,
    n: usize,
};

/// One quantized stack/run set: decode-style rows one at a time, then the tiles over all rows.
fn quantized(m: *g4.Model, io: std.Io, gpa: std.mem.Allocator, dir: []const u8, tag: []const u8, runs: []const g4.weights.Linear, pitch: usize, x: []const u16, k: usize, rows: usize) !void {
    const xb = try m.device.buffer(rows * k * 2, shared);
    defer xb.deinit();
    @memcpy(xb.slice(u16, rows * k), x[0 .. rows * k]);
    const y_win = try m.device.buffer(rows * pitch * 2, shared);
    defer y_win.deinit();
    const y_tile = try m.device.buffer(rows * pitch * 2, shared);
    defer y_tile.deinit();
    {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (runs) |r| for (0..rows) |row|
            try g4.weights.projectRowAt(&m.projections, m.device, e, r.raw.?, r, .{ .buf = xb, .off = row * k * 2 }, y_win, row * pitch * 2, 1);
        e.end();
        try finish(cb);
    }
    {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (runs) |r| {
            const raw = r.raw.?;
            const q: g4.weights.Mlx = .{ .w = .{ .buf = raw.w }, .s = .{ .buf = raw.scales }, .b = .{ .buf = raw.biases }, .n = @intCast(r.n), .k = @intCast(r.k), .bits = r.format.bits };
            g4.affine_mm.denseBatch(e, m.pipes(q.bits), g4.weights.Ref{ .buf = xb }, q, g4.weights.Ref{ .buf = y_tile, .off = r.col * 2 }, @intCast(rows), 1, .{ .x_row = @intCast(k), .y_row = @intCast(pitch) });
        }
        e.end();
        try finish(cb);
    }
    var name: [128]u8 = undefined;
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.x", .{tag}), x, rows, k);
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.win", .{tag}), y_win.slice(u16, rows * pitch), rows, pitch);
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.tile", .{tag}), y_tile.slice(u16, rows * pitch), rows, pitch);
}

/// A bf16 linear through gemma4/dense.metal, all rows at once (its row bits do not depend on the window).
fn dense(m: *g4.Model, io: std.Io, gpa: std.mem.Allocator, dir: []const u8, tag: []const u8, d: g4.weights.Dense, x: []const u16, rows: usize) !void {
    const xb = try m.device.buffer(rows * d.k * 2, shared);
    defer xb.deinit();
    @memcpy(xb.slice(u16, rows * d.k), x[0 .. rows * d.k]);
    const yb = try m.device.buffer(rows * d.n * 2, shared);
    defer yb.deinit();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(m.kernels.ple.?.dense);
    e.setBuffer(xb, 0, 0);
    e.setBuffer(d.w.buf, d.w.off, 1);
    e.setBuffer(yb, 0, 2);
    e.setValue([3]i32{ @intCast(rows), @intCast(d.n), @intCast(d.k) }, 3);
    e.dispatchGroups(.{ .width = (d.n / 4 + 1) / 2 }, .{ .width = 64 });
    e.end();
    try finish(cb);
    var name: [128]u8 = undefined;
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.x", .{tag}), x, rows, d.k);
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.y", .{tag}), yb.slice(u16, rows * d.n), rows, d.n);
}

/// The per-layer gate with its gelu epilogue (what the decode runs), on the layer's real inputs and per-layer rows.
fn gateFused(m: *g4.Model, io: std.Io, gpa: std.mem.Allocator, dir: []const u8, tag: []const u8, d: g4.weights.Dense, x: []const u16, pli: []const u16, layer: usize, rows: usize) !void {
    const geo = m.config.geo;
    const xb = try m.device.buffer(rows * d.k * 2, shared);
    defer xb.deinit();
    @memcpy(xb.slice(u16, rows * d.k), x[0 .. rows * d.k]);
    const pb = try m.device.buffer(pli.len * 2, shared);
    defer pb.deinit();
    @memcpy(pb.slice(u16, pli.len), pli);
    const yb = try m.device.buffer(rows * d.n * 2, shared);
    defer yb.deinit();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(m.kernels.ple.?.gate);
    e.setBuffer(xb, 0, 0);
    e.setBuffer(d.w.buf, d.w.off, 1);
    e.setBuffer(yb, 0, 2);
    e.setValue([3]i32{ @intCast(rows), @intCast(d.n), @intCast(d.k) }, 3);
    e.setBuffer(pb, 0, 4);
    e.setValue([2]i32{ @intCast(layer * geo.layer_input), @intCast(geo.layers * geo.layer_input) }, 5);
    if (@hasDecl(g4.kernels, "gate_ksplit")) e.dispatchGroups(.{ .width = d.n / 4, .height = (rows + g4.kernels.gate_rows - 1) / g4.kernels.gate_rows }, .{ .width = 32 * g4.kernels.gate_ksplit }) else e.dispatchGroups(.{ .width = (d.n / 4 + 1) / 2 }, .{ .width = 64 });
    e.end();
    try finish(cb);
    var name: [128]u8 = undefined;
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.x", .{tag}), x, rows, d.k);
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.pli", .{tag}), pli, rows, geo.layers * geo.layer_input);
    try npy(io, gpa, dir, try std.fmt.bufPrint(&name, "{s}.y", .{tag}), yb.slice(u16, rows * d.n), rows, d.n);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const ids_file = args.next() orelse return error.Usage;
    const dir = args.next() orelse return error.Usage;
    const rows: usize = 64;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, ids_file, gpa, .limited(1 << 24));
    defer gpa.free(bytes);
    const all = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer all.deinit();
    try std.Io.Dir.cwd().createDirPath(io, dir);
    const m = try g4.Model.load(gpa, io, model_dir);
    defer m.deinit();
    const geo = m.config.geo;
    var cache = try g4.state.Cache.init(m.device, geo, 256);
    defer cache.deinit();
    var s = try g4.state.Scratch.init(m.device, geo, 256);
    defer s.deinit();
    const ids = all.value[0..rows];
    const segs = [_]fwd.Segment{.{ .cache = &cache, .rows = rows }};
    var chunks: fwd.Chunks = undefined;
    fwd.prepare(&s, &segs, ids, &chunks);
    {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        fwd.embedRows(m, e, &s, rows);
        e.setPipeline(m.kernels.sliding.rms);
        e.setBuffer(s.h[0], 0, 0);
        e.setBuffer(m.weights.layers[0].input_norm.buf, m.weights.layers[0].input_norm.off, 1);
        e.setBuffer(s.normed[0], 0, 2);
        e.setValue(g4.config.eps, 3);
        e.setBuffer(s.sums, 0, 4);
        e.setValue(@as(i32, @intCast(rows)), 5);
        e.dispatchGroups(.{ .width = rows }, .{ .width = 1024 });
        fwd.pleInputs(m, e, &s, &m.kernels.ple.?, rows, m.config.geo.layers);
        e.end();
        try finish(cb);
    }
    const t = m.weights.ple.?;
    {
        const emb = try grab(m, gpa, s.h[0], rows * geo.hidden * 2);
        defer gpa.free(emb);
        try dense(m, io, gpa, dir, "model_projection", t.model_projection, emb, rows);
    }
    const wanted = [_]usize{ 0, 3, 5, 11, 17, 23, 24, 35, 41 };
    var cur: usize = 0;
    for (0..geo.layers) |i| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        try fwd.layer(m, e, &s, &segs, &chunks, i, cur, rows, .{}, false);
        e.end();
        try finish(cb);
        if (std.mem.indexOfScalar(usize, &wanted, i) != null) {
            const l = &m.weights.layers[i];
            var tag: [64]u8 = undefined;
            const qkv_in = try grab(m, gpa, s.normed[cur], rows * geo.hidden * 2);
            defer gpa.free(qkv_in);
            try quantized(m, io, gpa, dir, try std.fmt.bufPrint(&tag, "qkv_L{d}", .{i}), l.qkv.list(), l.qkv.n, qkv_in, geo.hidden, rows);
            const qd = geo.query_heads * g4.config.headDim(i);
            const o_in = try grab(m, gpa, s.attn, rows * qd * 2);
            defer gpa.free(o_in);
            try quantized(m, io, gpa, dir, try std.fmt.bufPrint(&tag, "o_L{d}", .{i}), &.{l.o}, l.o.n, o_in, qd, rows);
            const gu_in = try grab(m, gpa, s.n_mlp, rows * geo.hidden * 2);
            defer gpa.free(gu_in);
            try quantized(m, io, gpa, dir, try std.fmt.bufPrint(&tag, "gateup_L{d}", .{i}), l.gate_up.list(), l.gate_up.n, gu_in, geo.hidden, rows);
            const down_in = try grab(m, gpa, s.act, rows * geo.intermediate * 2);
            defer gpa.free(down_in);
            try quantized(m, io, gpa, dir, try std.fmt.bufPrint(&tag, "down_L{d}", .{i}), &.{l.down}, l.down.n, down_in, geo.intermediate, rows);
            const h_out = try grab(m, gpa, s.h[1 - cur], rows * geo.hidden * 2);
            defer gpa.free(h_out);
            try dense(m, io, gpa, dir, try std.fmt.bufPrint(&tag, "plegate_L{d}", .{i}), l.ple_gate.?, h_out, rows);
            const pli = try grab(m, gpa, s.ple.?.pli, rows * geo.layers * geo.layer_input * 2);
            defer gpa.free(pli);
            try gateFused(m, io, gpa, dir, try std.fmt.bufPrint(&tag, "gatefused_L{d}", .{i}), l.ple_gate.?, h_out, pli, i, rows);
            const mid = try grab(m, gpa, s.ple.?.mid, rows * geo.layer_input * 2);
            defer gpa.free(mid);
            try dense(m, io, gpa, dir, try std.fmt.bufPrint(&tag, "pleproj_L{d}", .{i}), l.ple_projection.?, mid, rows);
            std.debug.print("layer {d} captured\n", .{i});
        }
        cur = 1 - cur;
    }
}
