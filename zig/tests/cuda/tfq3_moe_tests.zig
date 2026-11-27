//! TFQ3 mixed-bit routed gate/up/down uses the same row arithmetic at every runtime width.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const data = @import("tfq3_test_data.zig");
const fmt = cuda.tfq3_format;
const tf = cuda.tfq3;

pub fn run(gpu: check.Gpu, kernels: *tf.Kernels, stream: cuda.Stream, tables: u64) !void {
    const a = gpu.gpa;
    var metas: [3]fmt.TfpwExpertMeta = undefined;
    var wc: u64 = 0;
    for (&metas, 0..) |*m, i| {
        const bits: u32 = @intCast(i + 3);
        const words = fmt.tensorWords(128, 128, bits);
        m.* = std.mem.zeroes(fmt.TfpwExpertMeta);
        m.b_bits = bits;
        m.gate_word_off = wc;
        m.up_word_off = wc + words;
        m.down_word_off = wc + 2 * words;
        m.gate_sc_off = @intCast(i * 3);
        m.up_sc_off = @intCast(i * 3 + 1);
        m.down_sc_off = @intCast(i * 3 + 2);
        wc += 3 * words;
    }
    const words = try data.words(a, @intCast(wc));
    defer a.free(words);
    const scales = try data.scales(a, 9);
    defer a.free(scales);
    const x = try data.inputs(a, 128);
    defer a.free(x);
    const picks = try a.alloc(i32, data.max_rows * 3);
    defer a.free(picks);
    for (picks, 0..) |*p, i| p.* = @intCast((i + i / 3) % 3);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(words));
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    var dm = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.asBytes(&metas));
    defer dm.free();
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    var dp = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(picks));
    defer dp.free();
    const layer: tf.MoeLayer = .{ .weights = dw.ptr, .scales = ds.ptr, .metas = dm.ptr, .tables = tables, .experts = 3, .dims = 128, .inter = 128 };
    for ([_]u32{ 0, 2 }) |mode| for ([_]bool{ false, true }) |f32_out| {
        const elem: usize = if (f32_out) 4 else 2;
        const row_bytes = 3 * 128 * elem;
        var serial = try cuda.DeviceBuffer.alloc(gpu.d, data.max_rows * row_bytes);
        defer serial.free();
        var batch = try cuda.DeviceBuffer.alloc(gpu.d, serial.len);
        defer batch.free();
        try serial.fill8(0x5a, stream.handle);
        for (0..data.max_rows) |row| try kernels.moe(stream, layer, dx.ptr + row * 128 * 2, 128, dp.ptr + row * 3 * 4, 3, serial.ptr + row * row_bytes, 3, 1, 2, 10, mode, f32_out);
        try stream.synchronize();
        const want = try check.download(gpu, serial);
        defer a.free(want);
        for (data.widths) |rows| for (0..5) |_| {
            try batch.fill8(0x5a, stream.handle);
            try kernels.moe(stream, layer, dx.ptr, 128, dp.ptr, 3, batch.ptr, 3, rows, 2, 10, mode, f32_out);
            try stream.synchronize();
            const got = try a.alloc(u8, rows * row_bytes);
            defer a.free(got);
            try batch.download(0, got);
            try check.sameBytes("TFQ3 MoE serial/window/unused-slot", got, want[0..got.len]);
        };
    };
    check.pass("TFQ3 mixed MoE: 516 widths, both activation modes and output types equal serial at five repeats", .{});
}
