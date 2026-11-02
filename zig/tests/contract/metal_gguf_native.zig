//! Exercise the owning 27B preparation, selector, row windows and borrowed draft-head metadata spans.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const c = contract.spec;
const r = @import("rig.zig");
const family = tf.qwen27;
const projection = family.projection;
const Tensor = tf.checkpoint_metal.Tensor;
fn tensor(ref: r.Ref, dtype: tf.checkpoint_metal.DType, n: usize, k: usize) Tensor {
    return .{ .buffer = ref.buf, .offset = ref.off, .bytes = n * k * dtype.size(), .dtype = dtype, .rank = 2, .shape = .{ n, k, 1, 1, 1 } };
}
pub fn run(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, format: [2]usize) !void {
    for ([_]usize{ 48, 64 }) |n| try matrix(a, device, queue, format, n);
    for ([_]bool{ false, true }) |raw| try head(a, device, format, raw);
    std.debug.print("PASS GGUF native q{d} g{d}: actual 27B row/lane preparation, 17-row == solo, FP64 and both draft-head spans\n", .{ format[0], format[1] });
}
fn matrix(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, format: [2]usize, n: usize) !void {
    const shape = c.Shape{ .rows = 17, .n = n, .k = 5120 };
    const f = try contract.affine_f32.Fixture.init(a, shape, @intCast(format[0]), format[1]);
    defer f.deinit();
    try f.fill(.random, 0x443);
    var rng = std.Random.DefaultPrng.init(0x17);
    for (f.f.x) |*v| v.* = c.toBf16(rng.random().float(f32) * 4 - 2);
    var rig = r.Rig{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    const words = try a.alloc(u32, n * shape.k * format[0] / 32);
    defer a.free(words);
    f.f.words(words);
    const w = try rig.input(std.mem.sliceAsBytes(words));
    const s = try rig.input(std.mem.sliceAsBytes(f.scales));
    const b = try rig.input(std.mem.sliceAsBytes(f.biases));
    const x = try rig.input(std.mem.sliceAsBytes(f.f.x));
    const batch = try rig.output(2 * shape.rows * n);
    const solo = try rig.output(2 * shape.rows * n);
    const original = try a.dupe(u8, try rig.inputBytes());
    defer a.free(original);
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (owned.items) |v| v.deinit();
        owned.deinit(a);
    }
    const p = try projection.prepare(device, a, &owned, &.{.{ .weight = tensor(w, .u32, n, shape.k * format[0] / 32), .scales = tensor(s, .f32, n, shape.k / format[1]), .biases = tensor(b, .f32, n, shape.k / format[1]), .bits = @intCast(format[0]), .group = @intCast(format[1]) }}, 32);
    if (p.metadata != .f32 or (n == 48 and p.raw == null)) return error.NativeMetadataLost;
    var kernels = family.quant_gpu.Kernels{ .allocator = a, .device = device, .raw_tiles = !device.tensorUnits() };
    defer kernels.deinit();
    const pass = rig.begin();
    var open = true;
    errdefer if (open) pass.e.end();
    kernels.prompt = true;
    try kernels.quant(pass.e, p, .{ .buffer = x.buf, .offset = x.off }, null, .{ .buffer = x.buf, .offset = x.off }, .{ .buffer = batch.buf, .offset = batch.off }, @intCast(shape.rows));
    kernels.prompt = false;
    for (0..shape.rows) |row| try kernels.quant(pass.e, p, .{ .buffer = x.buf, .offset = x.off + row * shape.k * 2 }, null, .{ .buffer = x.buf, .offset = x.off }, .{ .buffer = solo.buf, .offset = solo.off + row * n * 2 }, 1);
    open = false;
    try rig.finish(pass);
    const actual = rig.bytes(batch, 2 * shape.rows * n);
    if (!std.mem.eql(u8, actual, rig.bytes(solo, actual.len))) return error.NativeRowsDiffer;
    if (!try rig.intact() or !std.mem.eql(u8, original, try rig.inputBytes())) return error.NativeGuardOrInputChanged;
    const spec = @import("metal_affine.zig").spec(.{ .kind = if (p.raw != null) .row else .lane, .bits = @intCast(format[0]), .group = format[1] }, "native FP32 affine");
    for (0..shape.rows) |row| for (0..n) |col| {
        const value = f.one(row, col);
        const got = c.read(.bf16, actual, row * n + col);
        var bound = spec;
        if (format[1] == 16 and p.raw == null) bound.depth = depth16;
        if (@abs(got - value[0]) > c.bound(bound, shape, value[0], value[1])) return error.NativeFloat64Mismatch;
    };
}
fn depth16(s: c.Shape) usize {
    return 128 + 2 * (s.k / 16) + 32;
}
fn head(a: std.mem.Allocator, device: mtl.Device, format: [2]usize, raw: bool) !void {
    const n = 248320;
    const k = 32;
    const ng = k / format[1];
    const width = 98592;
    const w = try device.buffer(n * k * format[0] / 8, mtl.ResourceOptions.shared);
    defer w.deinit();
    for (w.slice(u8, w.length()), 0..) |*v, i| v.* = @truncate(i *% 13);
    const sb = try device.buffer(n * ng * 8, mtl.ResourceOptions.shared);
    defer sb.deinit();
    for (sb.slice(u32, n * ng * 2), 0..) |*v, i| v.* = 0x3f800001 + @as(u32, @intCast(i % 0x10000));
    const biases = try device.buffer(n * ng * 4, mtl.ResourceOptions.shared);
    defer biases.deinit();
    for (biases.slice(u32, n * ng), 0..) |*v, i| v.* = 0xbf800003 + @as(u32, @intCast(i % 0x10000));
    const source = projection.Linear{ .words = .{ .buffer = w }, .pairs = .{ .buffer = sb }, .n = n, .k = k, .tile = 32, .slices = 1, .bits = @intCast(format[0]), .group = @intCast(format[1]), .metadata = .f32, .raw = if (raw) .{ .w = w, .scales = sb, .biases = biases, .n = n, .k = k, .bits = @intCast(format[0]), .group = format[1], .metadata = .f32 } else null };
    var h = try family.dflash.runtime_backend.Head.init(a, device, source);
    defer h.deinit();
    if (h.linear.metadata != .f32 or h.linear.n != width) return error.DraftHeadMetadataLost;
    const spans = [_][2]usize{ .{ 0, 98304 }, .{ 248032, 248320 } };
    var dst: usize = 0;
    for (spans) |span| {
        const bytes = (span[1] - span[0]) * k * format[0] / 8;
        if (!std.mem.eql(u8, h.linear.words.buffer.contents()[dst * k * format[0] / 8 ..][0..bytes], w.contents()[span[0] * k * format[0] / 8 ..][0..bytes])) return error.DraftHeadWordsDiffer;
        for (0..ng) |g| for (span[0]..span[1], 0..) |col, j| {
            if (raw) {
                const got = h.linear.raw.?;
                if (got.scales.slice(u32, width * ng)[(dst + j) * ng + g] != sb.slice(u32, n * ng)[col * ng + g] or got.biases.slice(u32, width * ng)[(dst + j) * ng + g] != biases.slice(u32, n * ng)[col * ng + g]) return error.DraftHeadNativeRowsDiffer;
            } else {
                const want = sb.slice(u32, n * ng * 2);
                const got = h.linear.pairs.buffer.slice(u32, width * ng * 2);
                if (got[(g * width + dst + j) * 2] != want[(g * n + col) * 2] or got[(g * width + dst + j) * 2 + 1] != want[(g * n + col) * 2 + 1]) return error.DraftHeadNativePairsDiffer;
            }
        };
        dst += span[1] - span[0];
    }
}
