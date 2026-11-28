//! A real Ling bf16 projection through unchanged Kimi rows, with an independent float64 sum and row seam.
const std = @import("std");
const mtl = @import("metal");
const ck = @import("ling_checkpoint");
const kernels = @import("kimi_kernels");
fn f(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}
fn round(v: f64) f64 {
    const bits: u32 = @bitCast(@as(f32, @floatCast(v)));
    return f(@truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16));
}
pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2 or args.len > 3) return error.ModelDirectoryRequired;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const library = try mtl.Library.fromBytes(device, @import("kimi_metallib").bytes);
    defer library.deinit();
    var k = try kernels.Kernels.init(device, library);
    defer k.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var model = try ck.Checkpoint.open(init.gpa, init.io, device, args[1]);
    defer model.deinit();
    const K = 1536;
    const N = 2048;
    const embed = try model.get("model.word_embeddings.weight", .bf16, &.{ 157184, K });
    const weight = try model.get("model.layers.0.attention.q_proj.weight", .bf16, &.{ N, K });
    const options = mtl.ResourceOptions.shared;
    const x = try device.buffer(2 * K * 2, options);
    defer x.deinit();
    const w = try device.buffer(N * K * 2, options);
    defer w.deinit();
    const y = try device.buffer(2 * N * 2, options);
    defer y.deinit();
    const split = try device.buffer(2 * N * 2, options);
    defer split.deinit();
    const scratch = try device.buffer(kernels.partFloats(2) * 4, options);
    defer scratch.deinit();
    for ([_]usize{ 11, 48 }, 0..) |id, row| {
        const at = embed.offset + id * K * 2;
        @memcpy(x.contents()[row * K * 2 ..][0 .. K * 2], embed.buffer.contents()[at..][0 .. K * 2]);
    }
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    k.interleaved(e, .{ .buf = weight.buffer, .off = weight.offset }, .{ .buf = w }, N, K);
    const a = kernels.RowsArgs{ .K = K, .N = N, .rows = 2, .x_stride = K, .y_stride = N };
    k.rows(e, .{ .buf = x }, .{ .buf = w }, .{ .buf = y }, .{ .buf = scratch }, false, a);
    for (0..2) |row| {
        var one = a;
        one.rows = 1;
        k.rows(e, .{ .buf = x, .off = row * K * 2 }, .{ .buf = w }, .{ .buf = split, .off = row * N * 2 }, .{ .buf = scratch }, false, one);
    }
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    try std.testing.expectEqualSlices(u8, y.contents()[0 .. 2 * N * 2], split.contents()[0 .. 2 * N * 2]);
    var rows = try @import("ling_rows").Rows.init(init.gpa, device, 2);
    defer rows.deinit();
    try rows.linear(&model, "model.layers.0.attention.q_proj.weight", .{ .buf = x }, .{ .buf = split }, 2, K, N, false);
    try std.testing.expectEqualSlices(u8, y.contents()[0 .. 2 * N * 2], split.contents()[0 .. 2 * N * 2]);
    const xv = x.slice(u16, 2 * K);
    const got = y.slice(u16, 2 * N);
    const raw = std.mem.bytesAsSlice(u16, weight.buffer.contents()[weight.offset..][0 .. N * K * 2]);
    var err: f64 = 0;
    var baseline: f64 = 0;
    var max_err: f64 = 0;
    var max_round: f64 = 0;
    for (0..2) |row| for (0..N) |n| {
        var sum: f64 = 0;
        for (0..K) |i| sum += f(xv[row * K + i]) * f(raw[n * K + i]);
        const delta = f(got[row * N + n]) - sum;
        const rounding = round(sum) - sum;
        err += delta * delta;
        baseline += rounding * rounding;
        max_err = @max(max_err, @abs(delta));
        max_round = @max(max_round, @abs(rounding));
    };
    std.debug.print("Ling q projection rows=2 whole/split=exact Metal_rms={d} bf16_rms={d} Metal_max={d} bf16_max={d}\n", .{ @sqrt(err / (2 * N)), @sqrt(baseline / (2 * N)), max_err, max_round });
    if (err > baseline or max_err > max_round) return error.WorseThanBf16;
    const activation = try rows.buffer(2 * N * 2);
    try rows.siluMul(.{ .buf = y }, .{ .buf = split }, .{ .buf = activation }, 2 * N);
    for (activation.slice(u16, 2 * N), got) |value, gate| {
        const g = f(gate);
        const expected = round(round(g / (1.0 + @exp(-g))) * g);
        if (f(value) != expected) return error.SiluRoundingMismatch;
    }
    std.debug.print("Ling SiLU reused lfm3_swiglu: 4096 real projection values equal bf16-point FP64 formula\n", .{});
    const gate = try rows.buffer(2 * 4608 * 2);
    const up = try rows.buffer(2 * 4608 * 2);
    const act = try rows.buffer(2 * 4608 * 2);
    const dense = try rows.buffer(2 * K * 2);
    const chunked = try rows.buffer(2 * K * 2);
    try rows.mlp(&model, "model.layers.0.mlp.", .{ .buf = x }, .{ .buf = dense }, .{ .buf = gate }, .{ .buf = up }, .{ .buf = act }, 2, K, 4608);
    if (args.len == 3) {
        for ([_]struct { suffix: []const u8, buffer: mtl.Buffer }{
            .{ .suffix = ".gate", .buffer = gate },
            .{ .suffix = ".up", .buffer = up },
            .{ .suffix = ".act", .buffer = act },
        }) |tap| {
            const path = try std.fmt.allocPrint(init.arena.allocator(), "{s}{s}", .{ args[2], tap.suffix });
            try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = tap.buffer.contents()[0 .. 2 * 4608 * 2] });
        }
    }
    for (0..2) |row| try rows.mlp(&model, "model.layers.0.mlp.", .{ .buf = x, .off = row * K * 2 }, .{ .buf = chunked, .off = row * K * 2 }, .{ .buf = gate }, .{ .buf = up }, .{ .buf = act }, 1, K, 4608);
    try std.testing.expectEqualSlices(u8, dense.contents()[0 .. 2 * K * 2], chunked.contents()[0 .. 2 * K * 2]);
    if (args.len == 3) try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = dense.contents()[0 .. 2 * K * 2] });
    std.debug.print("Ling layer0 dense MLP two rows vs1+1: exact\n", .{});
    const Kda = @import("ling_kda").Kda;
    var whole_kda = try Kda.init(&rows, &model, "model.layers.0.attention.", 16);
    var split_kda = try Kda.init(&rows, &model, "model.layers.0.attention.", 16);
    try whole_kda.forward(&rows, &model, "model.layers.0.attention.", .{ .buf = x }, .{ .buf = dense }, 2, K, -5, 1e-6);
    for (0..2) |row| try split_kda.forward(&rows, &model, "model.layers.0.attention.", .{ .buf = x, .off = row * K * 2 }, .{ .buf = chunked, .off = row * K * 2 }, 1, K, -5, 1e-6);
    try std.testing.expectEqualSlices(u8, dense.contents()[0 .. 2 * K * 2], chunked.contents()[0 .. 2 * K * 2]);
    try std.testing.expectEqualSlices(u8, whole_kda.state.contents()[0 .. 16 * 128 * 128 * 4], split_kda.state.contents()[0 .. 16 * 128 * 128 * 4]);
    try std.testing.expectEqualSlices(u8, whole_kda.conv.contents()[0 .. 9 * 2048 * 2], split_kda.conv.contents()[0 .. 9 * 2048 * 2]);
    if (args.len == 3) {
        const path = try std.fmt.allocPrint(init.arena.allocator(), "{s}.kda", .{args[2]});
        try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = path, .data = dense.contents()[0 .. 2 * K * 2] });
    }
    std.debug.print("Ling layer0 KDA full attention whole2 vs1+1: output/state/conv exact\n", .{});
    const Mla = @import("ling_mla").Mla;
    var whole_mla = try Mla.init(&rows, 16, 16);
    defer whole_mla.deinit();
    var split_mla = try Mla.init(&rows, 16, 16);
    defer split_mla.deinit();
    try whole_mla.forward(&rows, &model, "model.layers.3.attention.", .{ .buf = x }, .{ .buf = dense }, 2, K, 1e-6, 6000000);
    for (0..2) |row| try split_mla.forward(&rows, &model, "model.layers.3.attention.", .{ .buf = x, .off = row * K * 2 }, .{ .buf = chunked, .off = row * K * 2 }, 1, K, 1e-6, 6000000);
    try std.testing.expectEqualSlices(u8, dense.contents()[0 .. 2 * K * 2], chunked.contents()[0 .. 2 * K * 2]);
    try std.testing.expectEqualSlices(u8, whole_mla.cache.contents()[0 .. 2 * 576 * 2], split_mla.cache.contents()[0 .. 2 * 576 * 2]);
    std.debug.print("Ling layer3 MLA whole2 vs1+1: output/cache exact\n", .{});
    var moe = try @import("ling_moe").Moe.init(&rows, K, 512);
    defer moe.deinit();
    try moe.forward(&rows, &model, "model.layers.1.mlp.", .{ .buf = x }, .{ .buf = dense }, 2);
    for (0..2) |row| try moe.forward(&rows, &model, "model.layers.1.mlp.", .{ .buf = x, .off = row * K * 2 }, .{ .buf = chunked, .off = row * K * 2 }, 1);
    try std.testing.expectEqualSlices(u8, dense.contents()[0 .. 2 * K * 2], chunked.contents()[0 .. 2 * K * 2]);
    std.debug.print("Ling layer1 MoE whole2 vs1+1: output exact\n", .{});
}
