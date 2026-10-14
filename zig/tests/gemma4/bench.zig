//! Gemma 4 dense step costs: the 60 layers' projections and head alone by rows, against whole forwards.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");

const g4 = tf.gemma4;
const cfg = g4.config;
const lane = tf.lane_projection;

fn gpuMs(cb: mtl.CommandBuffer) f64 {
    return cb.gpuSeconds() * 1e3;
}

/// Every projection of the 60 layers and the head at `rows` rows, chained in one command buffer: GPU ms.
var given_sums: ?mtl.Buffer = null;

fn projections(m: *g4.Model, rows: usize, x: mtl.Buffer, y: mtl.Buffer, reg: bool) !f64 {
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (m.weights.layers) |l| for ([_]g4.weights.Linear{ l.qkv, l.o, l.gate_up, l.down }) |lin| {
        const p = try m.projections.get(m.gpa, m.device, lin.n, lin.k);
        if (reg) try p.encodeSums(e, .{ .buf = x }, .{ .buf = lin.w }, .{ .buf = lin.sb }, .{ .ref = .{ .buf = given_sums.? }, .stride = @intCast(rows) }, .{ .buf = y }, @intCast(rows)) else try p.encode(e, .{ .buf = x }, .{ .buf = lin.w }, .{ .buf = lin.sb }, .{ .buf = y }, @intCast(rows));
    };
    const head = try m.projections.get(m.gpa, m.device, m.weights.head.n, m.weights.head.k);
    try head.encode(e, .{ .buf = x }, .{ .buf = m.weights.head.w }, .{ .buf = m.weights.head.sb }, .{ .buf = y }, @intCast(rows));
    e.end();
    cb.commit();
    cb.wait();
    return gpuMs(cb);
}

const Q = struct { w: lane.Ref, s: lane.Ref, b: lane.Ref, n: u32, k: u32 };

fn member(m: *g4.Model, name: []const u8) !Q {
    var buf: [3][160]u8 = undefined;
    const w = try m.checkpoint.get(try std.fmt.bufPrint(&buf[0], "{s}.weight", .{name}));
    const sc = try m.checkpoint.get(try std.fmt.bufPrint(&buf[1], "{s}.scales", .{name}));
    const bi = try m.checkpoint.get(try std.fmt.bufPrint(&buf[2], "{s}.biases", .{name}));
    return .{ .w = .{ .buf = w.buffer, .off = w.offset }, .s = .{ .buf = sc.buffer, .off = sc.offset }, .b = .{ .buf = bi.buffer, .off = bi.offset }, .n = @intCast(w.shape[0]), .k = @intCast(w.shape[1] * 8) };
}

/// Every projection of the 60 layers on core/affine_mm (prompt GEMMs from the checkpoint's own tensors): GPU ms.
fn affineAll(m: *g4.Model, pipes: tf.affine_mm.Pipes, rows: u32, x: mtl.Buffer, sums: mtl.Buffer, y: mtl.Buffer) !f64 {
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    var name: [128]u8 = undefined;
    for (0..cfg.layers) |i| {
        const members = if (cfg.global(i)) &[_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" } else &[_][]const u8{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj" };
        for (members) |mm| {
            const q = try member(m, try std.fmt.bufPrint(&name, "language_model.model.layers.{d}.{s}", .{ i, mm }));
            tf.affine_mm.rowSums(e, pipes, 64, lane.Ref{ .buf = x }, lane.Ref{ .buf = sums }, rows, q.k);
            tf.affine_mm.dense(e, pipes, lane.Ref{ .buf = x }, lane.Ref{ .buf = sums }, q, lane.Ref{ .buf = y }, rows);
        }
    }
    const head = try member(m, "language_model.model.embed_tokens");
    tf.affine_mm.rowSums(e, pipes, 64, lane.Ref{ .buf = x }, lane.Ref{ .buf = sums }, rows, head.k);
    tf.affine_mm.dense(e, pipes, lane.Ref{ .buf = x }, lane.Ref{ .buf = sums }, head, lane.Ref{ .buf = y }, 1);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| std.debug.print("GPU failure: {s}\n", .{msg});
    return gpuMs(cb);
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const m = try g4.Model.load(gpa, io, model_dir);
    defer m.deinit();
    var bytes: usize = 0;
    for (m.weights.layers) |l| for ([_]g4.weights.Linear{ l.qkv, l.o, l.gate_up, l.down }) |lin| {
        bytes += lin.n * lin.k / 2 + lin.n * lin.k / 64 * 4;
    };
    bytes += m.weights.head.n * m.weights.head.k / 2 + m.weights.head.n * m.weights.head.k / 64 * 4;
    const x = try m.device.buffer(128 * cfg.intermediate * 2, mtl.ResourceOptions.private);
    defer x.deinit();
    const y = try m.device.buffer(128 * cfg.vocab * 2, mtl.ResourceOptions.private);
    defer y.deinit();
    std.debug.print("weights read a step: {d:.2} GB\n", .{@as(f64, @floatFromInt(bytes)) / 1e9});
    for ([_]usize{ 1, 16, 64, 128 }) |rows| {
        var best: f64 = 1e9;
        for (0..5) |_| best = @min(best, try projections(m, rows, x, y, false));
        std.debug.print("projections, {d} rows: {d:.2} ms GPU ({d:.0} GB/s)\n", .{ rows, best, @as(f64, @floatFromInt(bytes)) / best / 1e6 });
        if (rows <= 16) {
            const sb = try m.device.buffer(16 * (cfg.intermediate / 64) * 4, mtl.ResourceOptions.private);
            defer sb.deinit();
            given_sums = sb;
            var best2: f64 = 1e9;
            for (0..5) |_| best2 = @min(best2, try projections(m, rows, x, y, true));
            std.debug.print("  with sums given (no sums dispatches): {d:.2} ms GPU\n", .{best2});
        }
    }
    {
        const text = try tf.affine_mm.source(m.device, gpa, .{});
        defer gpa.free(text);
        const lib = try mtl.Library.fromSource(m.device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: tf.affine_mm.Pipes = undefined;
        for (&pipes, tf.affine_mm.names) |*p, nm| p.* = try mtl.Pipeline.init(m.device, lib, nm, false);
        const big_x = try m.device.buffer(512 * cfg.intermediate * 2, mtl.ResourceOptions.private);
        defer big_x.deinit();
        const big_y = try m.device.buffer(512 * 2 * cfg.intermediate * 2, mtl.ResourceOptions.private);
        defer big_y.deinit();
        const sums = try m.device.buffer(512 * (cfg.intermediate / 64) * 4, mtl.ResourceOptions.private);
        defer sums.deinit();
        for ([_]u32{ 128, 256, 512 }) |rows| {
            var best: f64 = 1e9;
            for (0..3) |_| best = @min(best, try affineAll(m, pipes, rows, big_x, sums, big_y));
            std.debug.print("affine_mm projections, {d} rows: {d:.2} ms GPU ({d:.0} tok/s)\n", .{ rows, best, @as(f64, @floatFromInt(rows)) * 1e3 / best });
        }
    }
    {
        var ss = try g4.state.Scratch.init(m.device, 4096);
        defer ss.deinit();
        var ps = try g4.prompt.Scratch.init(m.device, 4096);
        defer ps.deinit();
        var tok: [40]u32 = undefined;
        for (&tok, 0..) |*t, i| t.* = @intCast(2000 + i * 37);
        {
            var ca = try g4.state.Cache.init(m.device, 4096);
            defer ca.deinit();
            var cbk = try g4.state.Cache.init(m.device, 4096);
            defer cbk.deinit();
            const step: usize = if (std.c.getenv("GEMMA_STEP")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 40 else 40;
            var at: usize = 0;
            while (at < tok.len) : (at += step) try g4.prompt.run(m, &ps, &ca, tok[at..@min(at + step, tok.len)]);
            try g4.forward.run(m, &ss, &.{.{ .cache = &cbk, .rows = 16 }}, tok[0..16]);
            try g4.forward.run(m, &ss, &.{.{ .cache = &cbk, .rows = 16 }}, tok[16..32]);
            try g4.forward.run(m, &ss, &.{.{ .cache = &cbk, .rows = 8 }}, tok[32..40]);
            const a = ps.logits.slice(u16, cfg.vocab);
            const b = ss.logits.slice(u16, 16 * cfg.vocab)[7 * cfg.vocab ..][0..cfg.vocab];
            var worst: f32 = 0;
            for (a, b) |x0, y0| worst = @max(worst, @abs(@as(f32, @bitCast(@as(u32, x0) << 16)) - @as(f32, @bitCast(@as(u32, y0) << 16))));
            std.debug.print("prompt path vs decode path, 40 tokens, last-row logits: worst |diff| {d}\n", .{worst});
        }
        if (std.c.getenv("GEMMA_DEBUG_ONLY") != null) return;
    }
    var s = try g4.state.Scratch.init(m.device, 33024);
    defer s.deinit();
    var c = try g4.state.Cache.init(m.device, 33024);
    defer c.deinit();
    var ids: [128]u32 = undefined;
    for (&ids, 0..) |*t, i| t.* = @intCast(1000 + i);
    for ([_]usize{ 0, 8000, 32000 }) |depth| {
        c.len = depth;
        for ([_]usize{ 1, 128 }) |rows| {
            var best: f64 = 1e9;
            for (0..5) |_| {
                c.len = depth;
                const t0 = std.c.mach_absolute_time();
                try g4.forward.run(m, &s, &.{.{ .cache = &c, .rows = rows }}, ids[0..rows]);
                best = @min(best, @as(f64, @floatFromInt(std.c.mach_absolute_time() - t0)) / 24e3);
            }
            std.debug.print("whole forward, {d} rows at position {d}: {d:.2} ms wall ({d:.0} tok/s)\n", .{ rows, depth, best, @as(f64, @floatFromInt(rows)) * 1e3 / best });
        }
    }
}
