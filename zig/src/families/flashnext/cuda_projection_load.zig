//! Projection loading preserves dense BF16 bits and MXFP8 codes/scales; affine layouts keep their existing packer.
const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const projection = @import("cuda_projection.zig");

pub const Micro = struct {
    n: usize,
    k: usize,
    codes: []u8,
    scales: []u8,

    pub fn deinit(self: *Micro, a: std.mem.Allocator) void {
        a.free(self.codes);
        a.free(self.scales);
        self.* = undefined;
    }
};

/// Own each plane before the store advances to a possibly different mapped shard.
pub fn readMicro(a: std.mem.Allocator, store: anytype, prefix: []const u8) !Micro {
    var buf: [512]u8 = undefined;
    const w = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}));
    if (w.dtype != .f8_e4m3 or w.rank != 2 or w.dim(0) == 0 or w.dim(1) == 0 or w.dim(1) % 64 != 0)
        return error.UnexpectedTensor;
    const n = w.dim(0);
    const k = w.dim(1);
    const count = try std.math.mul(usize, n, k);
    if (w.bytes.len != count) return error.UnexpectedTensor;
    const codes = try a.dupe(u8, w.bytes);
    errdefer a.free(codes);
    const s = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{prefix}));
    if (!s.is(.u8, &.{ n, k / 32 }) or s.bytes.len != count / 32) return error.UnexpectedTensor;
    return .{ .n = n, .k = k, .codes = codes, .scales = try a.dupe(u8, s.bytes) };
}

/// Concatenate output rows of compatible projections without converting or recalibrating either plane.
pub fn readStitched(a: std.mem.Allocator, store: anytype, prefix: []const u8, names: []const []const u8) !Micro {
    if (names.len == 0 or names.len > 6) return error.UnexpectedTensor;
    var parts: [6]Micro = undefined;
    var held: usize = 0;
    defer for (parts[0..held]) |*part| part.deinit(a);
    var n: usize = 0;
    var k: usize = 0;
    var buf: [512]u8 = undefined;
    for (names) |name| {
        parts[held] = try readMicro(a, store, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }));
        held += 1;
        const part = parts[held - 1];
        if (k != 0 and k != part.k) return error.UnexpectedTensor;
        k = part.k;
        n = try std.math.add(usize, n, part.n);
    }
    const count = try std.math.mul(usize, n, k);
    const codes = try a.alloc(u8, count);
    errdefer a.free(codes);
    const scales = try a.alloc(u8, count / 32);
    var row: usize = 0;
    for (parts[0..held]) |part| {
        @memcpy(codes[row * k ..][0..part.codes.len], part.codes);
        @memcpy(scales[row * (k / 32) ..][0..part.scales.len], part.scales);
        row += part.n;
    }
    return .{ .n = n, .k = k, .codes = codes, .scales = scales };
}

fn uploadMicro(a: std.mem.Allocator, d: *const cuda.Driver, value: Micro) !wt.Q4 {
    return projection.mxfp8(a, d, .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ value.n, value.k, 1, 1, 1 }, .bytes = value.codes }, .{ .dtype = .u8, .rank = 2, .shape = .{ value.n, value.k / 32, 1, 1, 1 }, .bytes = value.scales });
}

pub fn load(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, prefix: []const u8, n: usize, k: usize) !wt.Q4 {
    var buf: [512]u8 = undefined;
    const t = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}));
    switch (t.dtype) {
        .bf16 => {
            if (!t.is(.bf16, &.{ n, k })) return error.UnexpectedTensor;
            return projection.dense(a, d, t);
        },
        .f8_e4m3 => {
            var value = try readMicro(a, store, prefix);
            defer value.deinit(a);
            if (value.n != n or value.k != k) return error.UnexpectedTensor;
            return uploadMicro(a, d, value);
        },
        .u32 => {
            if (!t.is(.u32, &.{ n, k / 8 })) return error.UnexpectedTensor;
            const planes = try wt.join3(a, store, &buf, prefix, "");
            defer a.free(planes.w);
            defer a.free(planes.s);
            defer a.free(planes.b);
            return wt.uploadQ4(a, d, planes.w, planes.s, planes.b, n, k / 8);
        },
        else => return error.UnsupportedQuantization,
    }
}

pub fn loadStitched(a: std.mem.Allocator, d: *cuda.Driver, store: *wt.Store, prefix: []const u8, names: []const []const u8, n: usize, k: usize) !wt.Q4 {
    if (names.len == 0) return error.UnexpectedTensor;
    var buf: [512]u8 = undefined;
    const first = try store.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, names[0] }));
    if (first.dtype == .f8_e4m3) {
        var value = try readStitched(a, store, prefix, names);
        defer value.deinit(a);
        if (value.n != n or value.k != k) return error.UnexpectedTensor;
        return uploadMicro(a, d, value);
    }
    if (first.dtype != .u32) return error.UnsupportedQuantization;
    const planes = try wt.stitch(a, store, prefix, names);
    defer a.free(planes.w);
    defer a.free(planes.s);
    defer a.free(planes.b);
    if (planes.w.len != n * (k / 8) * 4) return error.UnexpectedTensor;
    return wt.uploadQ4(a, d, planes.w, planes.s, planes.b, n, k / 8);
}

test {
    _ = @import("cuda_projection_load_test.zig");
}
