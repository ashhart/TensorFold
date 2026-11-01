//! Prompt passes past 128 rows through raw mixed-width projections equal their rows decoded one at a time.
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const projection = @import("projection.zig");
const quant = @import("quant_gpu.zig");

const opts = mtl.ResourceOptions.shared;
const group = 64;
const k = 256;
const rows = 300; // two full 128-row dispatches and a partial one

fn toBf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn tensor(buffer: mtl.Buffer, dtype: checkpoint.DType, r: usize, c: usize, size: usize) checkpoint.Tensor {
    return .{ .buffer = buffer, .offset = 0, .bytes = r * c * size, .dtype = dtype, .shape = .{ r, c, 1, 1, 1 }, .rank = 2 };
}

/// Random affine codes [n, k] at `bits` with bf16 scales and biases, as the checkpoint stores them.
fn synth(device: mtl.Device, owned: *std.ArrayList(mtl.Buffer), n: usize, bits: u8, seed: u64) !projection.Triple {
    var prng = std.Random.DefaultPrng.init(seed);
    const r = prng.random();
    const code_bits: usize = bits;
    const g = n * (k / group);
    const w = try device.buffer(n * k * code_bits / 8, opts);
    try owned.append(std.testing.allocator, w);
    const s = try device.buffer(g * 2, opts);
    try owned.append(std.testing.allocator, s);
    const b = try device.buffer(g * 2, opts);
    try owned.append(std.testing.allocator, b);
    r.bytes(w.slice(u8, n * k * code_bits / 8));
    for (s.slice(u16, g), b.slice(u16, g)) |*sv, *bv| {
        sv.* = toBf((r.float(f32) + 0.5) / 64 / @as(f32, @floatFromInt(@as(u32, 1) << @intCast(bits - 2))));
        bv.* = toBf(-(r.float(f32) + 0.5) * 0.15);
    }
    return .{ .weight = tensor(w, .u32, n, k * code_bits / 32, 4), .scales = tensor(s, .bf16, n, k / group, 2), .biases = tensor(b, .bf16, n, k / group, 2), .bits = bits, .group = group };
}

/// One call over every row, then one call a row; both outputs' bytes must match.
fn expectRowsEqual(device: mtl.Device, kern: *quant.Kernels, lin: projection.Linear) !void {
    const queue = try device.queue();
    defer queue.deinit();
    const n: usize = lin.n;
    const x = try device.buffer(rows * k * 2, opts);
    defer x.deinit();
    var prng = std.Random.DefaultPrng.init(0x128);
    for (x.slice(u16, rows * k)) |*v| v.* = toBf(prng.random().float(f32) * 2 - 1);
    const whole = try device.buffer(rows * n * 2, opts);
    defer whole.deinit();
    const single = try device.buffer(rows * n * 2, opts);
    defer single.deinit();
    @memset(whole.slice(u8, rows * n * 2), 0xff);
    @memset(single.slice(u8, rows * n * 2), 0xff);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    var open = true;
    errdefer if (open) e.end(); // a refused call still ends its encoder
    kern.prompt = true;
    try kern.quant(e, lin, .{ .buffer = x }, null, .{ .buffer = x }, .{ .buffer = whole }, rows);
    kern.prompt = false;
    for (0..rows) |r| try kern.quant(e, lin, .{ .buffer = x, .offset = r * k * 2 }, null, .{ .buffer = x }, .{ .buffer = single, .offset = r * n * 2 }, 1);
    e.end();
    open = false;
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.debug.print("GPU failed: {s}\n", .{why});
        return error.GpuFailed;
    }
    try std.testing.expectEqualSlices(u16, single.slice(u16, rows * n), whole.slice(u16, rows * n));
}

/// The device's buffers and the kernels as the 27B's model sets them up on this chip.
const Rig = struct {
    device: mtl.Device,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    kern: quant.Kernels,

    fn init() !Rig {
        const device = mtl.Device.init() catch return error.SkipZigTest;
        return .{ .device = device, .kern = .{ .allocator = std.testing.allocator, .device = device, .raw_tiles = !device.tensorUnits() } };
    }

    fn deinit(r: *Rig) void {
        r.kern.deinit();
        for (r.owned.items) |b| b.deinit();
        r.owned.deinit(std.testing.allocator);
        r.device.deinit();
    }

    fn matrix(r: *Rig, n: usize, bits: u8, seed: u64) !projection.Linear {
        return projection.prepare(r.device, std.testing.allocator, &r.owned, &.{try synth(r.device, &r.owned, n, bits, seed)}, 32);
    }
};

test "a 300-row prompt pass through a 4/5-bit stack equals its rows one at a time" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var r = try Rig.init();
    defer r.deinit();
    // One product a member, each into its columns; the 16-row member stays a raw row matrix on every chip.
    var runs = [_]projection.Run{ .{ .linear = try r.matrix(64, 4, 2), .first = 0 }, .{ .linear = try r.matrix(16, 5, 3), .first = 64 } };
    var stack = runs[0].linear;
    stack.n = 80;
    stack.gn = runs[0].linear.gn + runs[1].linear.gn;
    stack.raw = null;
    stack.runs = &runs;
    try expectRowsEqual(r.device, &r.kern, stack);
}

test "a 300-row prompt pass through a 5-bit matrix on simd_qmm_bits tiles equals its rows one at a time" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var r = try Rig.init();
    defer r.deinit();
    // Tensor units take a non-4-bit matrix of whole tiles on lane kernels, not this raw path.
    if (r.device.tensorUnits()) return error.SkipZigTest;
    try expectRowsEqual(r.device, &r.kern, try r.matrix(64, 5, 1));
}
