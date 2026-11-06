//! The trellis tile as int8 with per-tile scales, on the tensor units, against the CPU int8 oracle.
//! `projection.applyF32TrellisInt8` decodes each 16x16 tile, scales it to int8 by its own peak, quantizes the
//! rotated x a 16-block, and accumulates `q_w q_x * tile_scale * x_scale`; this drives the same formula through
//! the GPU's int8 MMA and compares the whole projection (with the engine's own rotate_out on top).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const exl3 = tf.exl3_projection;
const oracle = @import("projection.zig");
const h = @import("operator_csa2_fp64.zig");
const math = @import("math.zig");

fn view(v: anytype) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

// TF_INT8_TILE=1: one 16x16 output tile a threadgroup, k in 16-blocks, int8 codes and int8 x.
test "the int8 trellis tile against the CPU int8 oracle (TF_INT8_TILE=1)" {
    if (std.c.getenv("TF_INT8_TILE") == null) return error.SkipZigTest;
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const queue = try device.queue();
    defer queue.deinit();
    const a = std.testing.allocator;
    // The routed experts' shape, and a small one, so the tile core is checked where the path will use it.
    const shapes = [_][2]usize{ .{ 256, 128 }, .{ 5120, 2304 } };
    for (shapes) |shape| {
    const k: usize = shape[0];
    const n: usize = shape[1];
    var rng = h.Lcg{ .state = 0x00c0de01 };
    const width = exl3.format.Width{ .halves = 6 };
    const tile_bytes = width.tileWords() * 2;
    var b = h.Buffers{ .device = device };
    defer b.deinit();
    const trellis = try b.get(u8, (k / 16) * (n / 16) * tile_bytes);
    for (trellis.slice(u8, (k / 16) * (n / 16) * tile_bytes)) |*byte| byte.* = @truncate(rng.next());
    const suh = try b.get(u16, k);
    for (suh.slice(u16, k)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const svh = try b.get(u16, n);
    for (svh.slice(u16, n)) |*cell| cell.* = @bitCast(@as(f16, @floatCast(0.02 * (@as(f64, rng.scale(1.0)) + 1.0))));
    const x_gpu = try b.get(u16, k);
    for (x_gpu.slice(u16, k)) |*cell| cell.* = math.b(0.05 * rng.scale(1.0));
    const y_gpu = try b.get(u16, n);
    const scratch = try b.get(u8, try exl3.scratchBytes(k, n, 1));
    const matrix = exl3.Matrix{ .trellis = view(h.view(trellis)), .suh = view(h.view(suh)), .svh = view(h.view(svh)), .k = k, .n = n, .width = width, .codebook = .inst3 };
    try matrix.validate();

    var p = try exl3.Pipelines.loadWith(device, a);
    defer p.deinit();
    {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        errdefer enc.end();
        try exl3.encodeInt8Tile(&p, enc, matrix, view(h.view(x_gpu)), view(h.view(y_gpu)), view(h.view(scratch)), .{ .count = 1, .x_stride = k, .y_stride = n, .output = .bf16 });
        enc.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |t| {
            std.debug.print("INT8TILE gpu failure: {s}\n", .{std.mem.span(t)});
            return error.Int8TileGpuFailure;
        }
    }
    // The oracle's own formula over the same bytes.
    var s1 = try oracle.Scratch.init(a, k, n);
    defer s1.deinit();
    const ideal = try a.alloc(f32, n);
    defer a.free(ideal);
    try oracle.applyF32TrellisInt8(matrix, x_gpu.slice(u16, k), ideal, &s1);
    var unequal: usize = 0;
    var worst: f64 = 0;
    var energy: f64 = 0;
    var magnitude: f64 = 0;
    for (0..n) |i| {
        const got = math.f(y_gpu.slice(u16, n)[i]);
        const want = ideal[i];
        if (math.b(got) != math.b(want)) unequal += 1;
        const scale = @max(@abs(want), 0x1p-126);
        worst = @max(worst, @abs(@as(f64, got) - @as(f64, want)) / scale);
        energy += (@as(f64, got) - @as(f64, want)) * (@as(f64, got) - @as(f64, want));
        magnitude += @as(f64, want) * @as(f64, want);
    }
    std.debug.print("INT8TILE k={d} n={d} cells={d} unequal={d} worst={d} nrms={d}\n", .{ k, n, n, unequal, worst, @sqrt(energy / magnitude) });
    }
}
