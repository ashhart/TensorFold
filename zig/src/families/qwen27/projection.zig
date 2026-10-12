//! Core lane NT32 packing and fixed member K split, independent of active rows.
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const row = @import("core").row_projection;
const lane = @import("core").lane_projection;

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };
/// A device matrix; `runs` non-empty means fused members of different widths, each its own product into these rows.
pub const Linear = struct { words: Ref, pairs: Ref, n: u32, k: u32, tile: u32, slices: u32, raw: ?row.Weights = null, bits: u8 = 4, group: u32 = 64, runs: []const Run = &.{} };
/// Members of one width inside a fused matrix, writing output columns [first, first + linear.n).
pub const Run = struct { linear: Linear, first: u32 };
pub const Triple = struct { weight: checkpoint.Tensor, scales: checkpoint.Tensor, biases: checkpoint.Tensor, bits: u8 = 4, group: u32 = 64 };

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
    const bits = parts[0].bits;
    const group = parts[0].group;
    if (bits == 0 or bits > 8 or group == 0) return error.BadAffineMetadata;
    const k = parts[0].weight.shape[1] * 32 / bits;
    var slices = try split(parts[0].weight.shape[0], k);
    var n: usize = 0;
    for (parts) |p| {
        const rows = p.weight.shape[0];
        if (p.bits != bits or p.group != group or k % group != 0) return error.BadAffineMetadata;
        if (p.weight.rank != 2 or p.weight.dtype != .u32 or p.weight.shape[1] * 32 != k * bits or rows % 4 != 0 or try split(rows, k) != slices) return error.BadProjectionShape;
        if (p.scales.dtype != .bf16 or p.biases.dtype != .bf16 or p.scales.rank != 2 or p.biases.rank != 2 or p.scales.shape[0] != rows or p.biases.shape[0] != rows or p.scales.shape[1] != k / group or p.biases.shape[1] != k / group) return error.BadAffineMetadata;
        n = try std.math.add(usize, n, rows);
    }
    if (!device.tensorUnits()) return prepareRows(device, allocator, owned, parts, n, k, tile, slices);
    if (n % tile != 0) return error.BadProjectionShape;
    if (bits != 4 or group != 64) return prepareLanes(device, allocator, owned, parts, n, k, tile, slices);
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
    const bits: usize = parts[0].bits;
    const ng = k / parts[0].group;
    const row_bytes = k * bits / 8;
    const w = try own(device, allocator, owned, n * row_bytes, options);
    const scale = try own(device, allocator, owned, n * ng * 2, options);
    const bias = try own(device, allocator, owned, n * ng * 2, options);
    var base: usize = 0;
    for (parts) |p| {
        const rows = p.weight.shape[0];
        @memcpy(w.contents()[base * row_bytes .. (base + rows) * row_bytes], std.mem.sliceAsBytes(p.weight.host(u32)));
        @memcpy(scale.contents()[base * ng * 2 .. (base + rows) * ng * 2], std.mem.sliceAsBytes(p.scales.host(u16)));
        @memcpy(bias.contents()[base * ng * 2 .. (base + rows) * ng * 2], std.mem.sliceAsBytes(p.biases.host(u16)));
        base += rows;
    }
    const raw = row.Weights{ .w = w, .scales = scale, .biases = bias, .n = n, .k = k, .group = parts[0].group, .bits = parts[0].bits, .sum = .f32 };
    try raw.validate();
    return .{ .words = .{ .buffer = w }, .pairs = .{ .buffer = scale }, .n = @intCast(n), .k = @intCast(k), .tile = tile, .slices = slices, .raw = raw, .bits = parts[0].bits, .group = parts[0].group };
}

/// Widths other than 4/64 on tensor units: lane_projection.weight_layout byte copies for the non-reg lane kernel.
fn prepareLanes(device: mtl.Device, allocator: std.mem.Allocator, owned: *std.ArrayList(mtl.Buffer), parts: []const Triple, n: usize, k: usize, tile: u32, slices: u32) !Linear {
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const layout = lane.Layout{ .n = n, .k = k, .format = .{ .bits = parts[0].bits, .group = parts[0].group }, .sk = slices };
    try layout.validate();
    const w = try own(device, allocator, owned, layout.weightWords() * 4, options);
    const sb = try own(device, allocator, owned, layout.metadataElements() * 2, options);
    var base: usize = 0;
    for (parts) |p| {
        packTiles(.{ .words = p.weight.host(u32), .scales = p.scales.host(u16), .biases = p.biases.host(u16), .rows = p.weight.shape[0] }, base, n, k, parts[0].bits, parts[0].group, w.slice(u32, layout.weightWords()), sb.slice(u16, layout.metadataElements()));
        base += p.weight.shape[0];
    }
    return .{ .words = .{ .buffer = w }, .pairs = .{ .buffer = sb }, .n = @intCast(n), .k = @intCast(k), .tile = tile, .slices = slices, .bits = parts[0].bits, .group = parts[0].group };
}

/// One member as stored: words [rows][k*bits/32], scales and biases [rows][k/group].
pub const Member = struct { words: []const u32, scales: []const u16, biases: []const u16, rows: usize };

/// Copies a member into lane_projection.weight_layout rows [base, base + rows) of an n-row stack, bytes unchanged.
pub fn packTiles(p: Member, base: usize, n: usize, k: usize, bits: usize, group: usize, out_w: []u32, out_sb: []u16) void {
    const ng = k / group;
    const wp = group * bits / 32;
    for (0..p.rows) |r| for (0..ng) |g| {
        const at = base + r;
        @memcpy(out_w[((at / 32 * ng + g) * 32 + at % 32) * wp ..][0..wp], p.words[(r * ng + g) * wp ..][0..wp]);
        out_sb[(g * n + at) * 2] = p.scales[r * ng + g];
        out_sb[(g * n + at) * 2 + 1] = p.biases[r * ng + g];
    };
}

test "a stack packed member by member equals the core lane pack of the whole matrix" {
    const a = std.testing.allocator;
    const n = 64;
    const k = 256;
    var rng = std.Random.DefaultPrng.init(7);
    for ([_]u8{ 4, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| {
        const l = lane.Layout{ .n = n, .k = k, .format = .{ .bits = bits, .group = group } };
        const w = try a.alloc(u32, l.weightWords());
        defer a.free(w);
        const sb = try a.alloc(u16, n * (k / group) * 2);
        defer a.free(sb);
        rng.random().bytes(std.mem.sliceAsBytes(w));
        rng.random().bytes(std.mem.sliceAsBytes(sb));
        const scales = sb[0 .. n * (k / group)];
        const biases = sb[n * (k / group) ..];
        const want_w = try a.alloc(u32, w.len);
        defer a.free(want_w);
        const want_sb = try a.alloc(u16, l.metadataElements());
        defer a.free(want_sb);
        try lane.pack(l, w, scales, biases, want_w, want_sb);
        const got_w = try a.alloc(u32, w.len);
        defer a.free(got_w);
        const got_sb = try a.alloc(u16, want_sb.len);
        defer a.free(got_sb);
        const row_words = w.len / n;
        const ng = k / group;
        for ([_][2]usize{ .{ 0, 32 }, .{ 32, 16 }, .{ 48, 16 } }) |part| packTiles(.{ .words = w[part[0] * row_words ..][0 .. part[1] * row_words], .scales = scales[part[0] * ng ..][0 .. part[1] * ng], .biases = biases[part[0] * ng ..][0 .. part[1] * ng], .rows = part[1] }, part[0], n, k, bits, group, got_w, got_sb);
        try std.testing.expectEqualSlices(u32, want_w, got_w);
        try std.testing.expectEqualSlices(u16, want_sb, got_sb);
    };
}

fn own(device: mtl.Device, allocator: std.mem.Allocator, owned: *std.ArrayList(mtl.Buffer), bytes: usize, options: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, options);
    owned.append(allocator, b) catch |err| {
        b.deinit();
        return err;
    };
    return b;
}
