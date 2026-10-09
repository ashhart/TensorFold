//! Core lane NT32 packing and fixed member K split, independent of active rows.
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const row = @import("core").row_projection;

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
pub const Linear = struct { words: Ref, pairs: Ref, n: u32, k: u32, tile: u32, slices: u32, raw: ?row.Weights = null };
pub const Triple = struct { weight: checkpoint.Tensor, scales: checkpoint.Tensor, biases: checkpoint.Tensor };

pub fn split(n: usize, k: usize) !u32 {
    if (n == 0 or k == 0 or n % 4 != 0 or k % 64 != 0) return error.BadProjectionShape;
    var slices: u32 = 1;
    while (slices < 8 and (n + 31) / 32 * slices < 1024 and k / 64 / (2 * slices) >= 8) slices *= 2;
    return slices;
}

/// K splits fastest for the reg kernel on an M5: an equal-precision reorder of the fp32 slice sums.
fn m5Split(n: usize, k: usize) ?u32 {
    const table = [_]struct { n: usize, k: usize, sk: u32 }{ .{ .n = 10240, .k = 5120, .sk = 2 }, .{ .n = 6240, .k = 5120, .sk = 2 }, .{ .n = 5120, .k = 6144, .sk = 4 }, .{ .n = 2048, .k = 5120, .sk = 4 }, .{ .n = 5120, .k = 17408, .sk = 4 } };
    for (table) |t| if (t.n == n and t.k == k) return t.sk;
    return null;
}

/// Prompt-row K splits fastest at 128 rows on an M5; prompt rows may differ in bits from decoded rows.
pub fn promptSplit(n: usize, k: usize) ?u32 {
    const table = [_]struct { n: usize, k: usize, sk: u32 }{ .{ .n = 34816, .k = 5120, .sk = 1 }, .{ .n = 10240, .k = 5120, .sk = 1 }, .{ .n = 5120, .k = 17408, .sk = 2 }, .{ .n = 5120, .k = 6144, .sk = 2 }, .{ .n = 12288, .k = 5120, .sk = 2 } };
    for (table) |t| if (t.n == n and t.k == k) return t.sk;
    return null;
}

pub fn prepare(device: mtl.Device, allocator: std.mem.Allocator, owned: *std.ArrayList(mtl.Buffer), parts: []const Triple, tile: u32) !Linear {
    if (tile != 32 or parts.len == 0) return error.BadProjectionShape;
    const k = parts[0].weight.shape[1] * 8;
    var slices = try split(parts[0].weight.shape[0], k);
    var n: usize = 0;
    for (parts) |p| {
        const rows = p.weight.shape[0];
        if (p.weight.rank != 2 or p.weight.dtype != .u32 or p.weight.shape[1] * 8 != k or rows % 4 != 0 or try split(rows, k) != slices) return error.BadProjectionShape;
        if (p.scales.dtype != .bf16 or p.biases.dtype != .bf16 or p.scales.rank != 2 or p.biases.rank != 2 or p.scales.shape[0] != rows or p.biases.shape[0] != rows or p.scales.shape[1] != k / 64 or p.biases.shape[1] != k / 64) return error.BadAffineMetadata;
        n = try std.math.add(usize, n, rows);
    }
    if (n % tile != 0) return error.BadProjectionShape;
    if (!device.tensorUnits()) return prepareRows(device, allocator, owned, parts, n, k, tile, slices);
    slices = m5Split(n, k) orelse slices;
    const w = try device.buffer(n * k / 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    owned.append(allocator, w) catch |err| {
        w.deinit();
        return err;
    };
    const sb = try device.buffer(n * (k / 64) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    owned.append(allocator, sb) catch |err| {
        sb.deinit();
        return err;
    };
    const dst = w.slice(u32, n * k / 8);
    const pair = sb.slice(u16, n * (k / 64) * 2);
    var base: usize = 0;
    for (parts) |p| {
        const rows = p.weight.shape[0];
        const words = p.weight.host(u32);
        const scales = p.scales.host(u16);
        const biases = p.biases.host(u16);
        for (0..rows) |r| for (0..k / 64) |g| {
            const src = r * (k / 8) + g * 8;
            const to = (((base + r) / tile) * (k / 64) + g) * tile * 8 + ((base + r) % tile) * 8;
            @memcpy(dst[to..][0..8], words[src..][0..8]);
            pair[(g * n + base + r) * 2] = scales[r * (k / 64) + g];
            pair[(g * n + base + r) * 2 + 1] = biases[r * (k / 64) + g];
        };
        base += rows;
    }
    return .{ .words = .{ .buffer = w }, .pairs = .{ .buffer = sb }, .n = @intCast(n), .k = @intCast(k), .tile = tile, .slices = slices };
}

test "stacked gate-up keeps the member K split rather than recomputing it from stacked width" {
    try std.testing.expectEqual(@as(u32, 2), try split(17408, 5120));
    try std.testing.expectEqual(@as(u32, 1), try split(34816, 5120));
    try std.testing.expectEqual(@as(u32, 8), try split(48, 5120));
    try std.testing.expectEqual(@as(u32, 8), try split(6144, 5120));
    try std.testing.expectEqual(@as(u32, 8), try split(5120, 17408));
}

fn prepareRows(device: mtl.Device, allocator: std.mem.Allocator, owned: *std.ArrayList(mtl.Buffer), parts: []const Triple, n: usize, k: usize, tile: u32, slices: u32) !Linear {
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const w = try device.buffer(n * k / 2, options);
    owned.append(allocator, w) catch |err| {
        w.deinit();
        return err;
    };
    const scale = try device.buffer(n * (k / 64) * 2, options);
    owned.append(allocator, scale) catch |err| {
        scale.deinit();
        return err;
    };
    const bias = try device.buffer(n * (k / 64) * 2, options);
    owned.append(allocator, bias) catch |err| {
        bias.deinit();
        return err;
    };
    var base: usize = 0;
    for (parts) |p| {
        const rows = p.weight.shape[0];
        @memcpy(w.contents()[base * k / 2 .. (base + rows) * k / 2], std.mem.sliceAsBytes(p.weight.host(u32)));
        @memcpy(scale.contents()[base * (k / 64) * 2 .. (base + rows) * (k / 64) * 2], std.mem.sliceAsBytes(p.scales.host(u16)));
        @memcpy(bias.contents()[base * (k / 64) * 2 .. (base + rows) * (k / 64) * 2], std.mem.sliceAsBytes(p.biases.host(u16)));
        base += rows;
    }
    const raw = row.Weights{ .w = w, .scales = scale, .biases = bias, .n = n, .k = k, .group = 64, .bits = 4, .sum = .f32 };
    try raw.validate();
    return .{ .words = .{ .buffer = w }, .pairs = .{ .buffer = scale }, .n = @intCast(n), .k = @intCast(k), .tile = tile, .slices = slices, .raw = raw };
}
