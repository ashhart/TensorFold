//! CUDA TFQ3: every served decode window and prefill width equals the same rows launched one at a time.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const data = @import("tfq3_test_data.zig");
const fmt = cuda.tfq3_format;
const tf = cuda.tfq3;
const Gpu = check.Gpu;

pub fn run(gpu: Gpu) !void {
    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var kernels = try tf.Kernels.load(gpu.d);
    defer kernels.unload();
    try kernels.reserve(512 * 5120, 64 * 34816 * 8, 512 * 2 * 128);
    try @import("tfq3_rounding_tests.zig").run(gpu, &kernels, stream);
    try @import("tfq3_sign_tests.zig").run(gpu, &kernels, stream);
    const table_bytes = data.tables();
    try fmt.validateTables(&table_bytes);
    var tables = try cuda.DeviceBuffer.fromHost(gpu.d, &table_bytes);
    defer tables.free();
    for ([_]struct { n: u32, k: u32 }{
        .{ .n = 128, .k = 128 },  .{ .n = 128, .k = 1024 },
        .{ .n = 128, .k = 5120 }, .{ .n = 34944, .k = 128 },
    }) |shape| {
        for ([_]u8{ 3, 4, 5 }) |bits| {
            try dense(gpu, &kernels, stream, tables.ptr, shape.n, shape.k, bits);
        }
    }
    try @import("tfq3_moe_tests.zig").run(gpu, &kernels, stream, tables.ptr);
    check.pass("CUDA TFQ3 contract: dense 3/4/5-bit, BF16/FP32, split-K/fused/BM64, mixed MoE, widths 1..512 and chunked 513/1024/1536/4218; five byte-identical repeats", .{});
}

fn dense(gpu: Gpu, kernels: *tf.Kernels, stream: cuda.Stream, tables: u64, n: u32, k: u32, bits: u8) !void {
    const a = gpu.gpa;
    const words = try data.words(a, @intCast(fmt.tensorWords(n, k, bits)));
    defer a.free(words);
    const scales = try data.scales(a, @intCast(fmt.tensorScales(n, k)));
    defer a.free(scales);
    const x = try data.inputs(a, k);
    defer a.free(x);
    var dw = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(words));
    defer dw.free();
    var ds = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(scales));
    defer ds.free();
    var dx = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(x));
    defer dx.free();
    const w: tf.DenseWeight = .{ .words = dw.ptr, .scales = ds.ptr, .tables = tables, .n = n, .k = k, .bits = bits, .row_off = 128, .sr_split = 256 };
    for ([_]bool{ false, true }) |f32_out| {
        const elem: usize = if (f32_out) 4 else 2;
        const stride = n + 128;
        var serial = try cuda.DeviceBuffer.alloc(gpu.d, data.max_rows * stride * elem);
        defer serial.free();
        var batch = try cuda.DeviceBuffer.alloc(gpu.d, serial.len);
        defer batch.free();
        try serial.fill8(0x5a, stream.handle);
        for (0..data.max_rows) |row| {
            try kernels.dense(stream, w, dx.ptr + row * k * 2, k, serial.ptr + row * stride * elem, stride, 1, f32_out);
        }
        try stream.synchronize();
        const want = try check.download(gpu, serial);
        defer a.free(want);
        var nonzero = false;
        for (0..n) |col| {
            if (f32_out) {
                const v: f32 = @bitCast(std.mem.readInt(u32, want[col * 4 ..][0..4], .little));
                try check.expect(std.math.isFinite(v), "TFQ3 serial nonfinite", .{});
                nonzero = nonzero or v != 0;
            } else {
                const b = std.mem.readInt(u16, want[col * 2 ..][0..2], .little);
                try check.expect(b & 0x7f80 != 0x7f80, "TFQ3 serial nonfinite", .{});
                nonzero = nonzero or b & 0x7fff != 0;
            }
        }
        try check.expect(nonzero, "TFQ3 nonzero control: fixture collapsed to zero", .{});
        for (data.widths) |rows| {
            const used = rows * stride * elem;
            for (0..5) |_| {
                try batch.fill8(0x5a, stream.handle);
                try kernels.dense(stream, w, dx.ptr, k, batch.ptr, stride, rows, f32_out);
                try stream.synchronize();
                const got = try a.alloc(u8, used);
                defer a.free(got);
                try batch.download(0, got);
                try check.sameBytes("TFQ3 dense serial/window/padding", got, want[0..used]);
            }
        }
        const changed = try a.dupe(u8, want[0 .. stride * elem]);
        defer a.free(changed);
        changed[0] ^= 1;
        try check.expect(!std.mem.eql(u8, changed, want[0..changed.len]), "TFQ3 byte-comparison planted control", .{});
    }
    check.pass("TFQ3 dense {d}-bit [{d},{d}]: 516 widths, BF16/FP32, padding and repeats equal serial", .{ bits, n, k });
}
