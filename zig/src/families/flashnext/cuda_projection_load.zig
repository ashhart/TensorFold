//! Projection loading preserves dense BF16 bits and MXFP8 codes/scales; affine layouts keep their existing packer.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const wt = @import("cuda_prompt_wt.zig");
const projection = @import("cuda_projection.zig");
const Cut = @import("cuda_modelopt_experts.zig").Cut;

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
    if (first.dtype == .bf16) {
        const bytes = try a.alloc(u8, try std.math.mul(usize, try std.math.mul(usize, n, k), 2));
        defer a.free(bytes);
        var at: usize = 0;
        for (names) |name| {
            const t = try store.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, name }));
            if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) != k or t.bytes.len != t.dim(0) * k * 2 or t.bytes.len > bytes.len - at)
                return error.UnexpectedTensor;
            @memcpy(bytes[at..][0..t.bytes.len], t.bytes);
            at += t.bytes.len;
        }
        if (at != bytes.len) return error.UnexpectedTensor;
        return projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });
    }
    if (first.dtype != .u32) return error.UnsupportedQuantization;
    const planes = try wt.stitch(a, store, prefix, names);
    defer a.free(planes.w);
    defer a.free(planes.s);
    defer a.free(planes.b);
    if (planes.w.len != n * (k / 8) * 4) return error.UnexpectedTensor;
    return wt.uploadQ4(a, d, planes.w, planes.s, planes.b, n, k / 8);
}

/// What a rank's cut of a projection keeps: output rows, or each row's input columns.
pub const Axis = enum { rows, cols };

/// A rank's own BF16 [cut, k] rows or [n, cut] input columns, copied before the store moves on.
pub fn cutBytes(a: std.mem.Allocator, t: core.safetensors.Tensor, cut: Cut, axis: Axis) ![]u8 {
    if (t.dtype != .bf16 or t.rank != 2 or t.dim(0) == 0 or t.dim(1) == 0) return error.UnexpectedTensor;
    const n = t.dim(0);
    const k = t.dim(1);
    if (t.bytes.len != try std.math.mul(usize, try std.math.mul(usize, n, k), 2)) return error.UnexpectedTensor;
    if (cut.from >= cut.to or cut.to > (if (axis == .rows) n else k)) return error.UnexpectedTensor;
    const w = cut.width();
    switch (axis) {
        .rows => return a.dupe(u8, t.bytes[cut.from * k * 2 ..][0 .. w * k * 2]),
        .cols => {
            if (w % 64 != 0) return error.UnexpectedTensor;
            const out = try a.alloc(u8, n * w * 2);
            for (0..n) |row| @memcpy(out[row * w * 2 ..][0 .. w * 2], t.bytes[(row * k + cut.from) * 2 ..][0 .. w * 2]);
            return out;
        },
    }
}

/// A BF16 [n, k] projection's cut on this rank, packed as `load` packs a whole one; other formats are refused.
pub fn loadCut(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, n: usize, k: usize, cut: Cut, axis: Axis) !wt.Q4 {
    var buf: [512]u8 = undefined;
    const t = try store.tensor(try std.fmt.bufPrint(&buf, "{s}.weight", .{prefix}));
    if (t.dtype != .bf16) return error.UnsupportedQuantization;
    if (!t.is(.bf16, &.{ n, k })) return error.UnexpectedTensor;
    const bytes = try cutBytes(a, t, cut, axis);
    defer a.free(bytes);
    const rows = if (axis == .rows) cut.width() else n;
    const cols = if (axis == .rows) k else cut.width();
    var value = try projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ rows, cols, 1, 1, 1 }, .bytes = bytes });
    if (axis == .rows) value.native.?.sk = cuda.qmmf.splitK(n, k);
    return value;
}

/// Up to four [from, to) output-row ranges kept of one stitched part, copied in order.
pub const Ranges = struct {
    cuts: [4]Cut = undefined,
    len: usize = 0,

    pub fn of(cuts: []const Cut) Ranges {
        std.debug.assert(cuts.len <= 4);
        var out: Ranges = .{ .len = cuts.len };
        @memcpy(out.cuts[0..cuts.len], cuts);
        return out;
    }

    pub fn items(self: *const Ranges) []const Cut {
        return self.cuts[0..self.len];
    }

    pub fn rows(self: Ranges) usize {
        var n: usize = 0;
        for (self.items()) |c| n += c.width();
        return n;
    }
};

/// Copy `kept` rows of a row-major `src` into the front of `dst`, in order; returns the bytes written.
pub fn copyRanges(dst: []u8, src: []const u8, row_bytes: usize, kept: Ranges) !usize {
    var at: usize = 0;
    for (kept.items()) |c| {
        if (c.from >= c.to or try std.math.mul(usize, c.to, row_bytes) > src.len) return error.UnexpectedTensor;
        const len = c.width() * row_bytes;
        if (len > dst.len - at) return error.UnexpectedTensor;
        @memcpy(dst[at..][0..len], src[c.from * row_bytes ..][0..len]);
        at += len;
    }
    return at;
}

/// One rank's BF16 [n, k] rows of a stitched projection: each part's ranges in order, copied before the store moves on.
pub fn stitchCut(a: std.mem.Allocator, store: anytype, prefix: []const u8, parts: []const []const u8, ranges: []const Ranges, n: usize, k: usize) ![]u8 {
    if (parts.len == 0 or parts.len != ranges.len) return error.UnexpectedTensor;
    const out = try a.alloc(u8, try std.math.mul(usize, try std.math.mul(usize, n, k), 2));
    errdefer a.free(out);
    var buf: [512]u8 = undefined;
    var at: usize = 0;
    for (parts, ranges) |name, kept| {
        const t = try store.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, name }));
        if (t.dtype != .bf16) return error.UnsupportedQuantization;
        if (t.rank != 2 or t.dim(0) == 0 or t.dim(1) != k or t.bytes.len != try std.math.mul(usize, t.dim(0) * k, 2))
            return error.UnexpectedTensor;
        at += try copyRanges(out[at..], t.bytes, k * 2, kept);
    }
    if (at != out.len) return error.UnexpectedTensor;
    return out;
}

/// `stitchCut` packed as `loadStitched` packs, its K slices pinned to `sk` (the whole stitched shape's) for equal sums.
pub fn loadStitchedCut(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, parts: []const []const u8, ranges: []const Ranges, n: usize, k: usize, sk: u32) !wt.Q4 {
    // the pin is checked on the shape before anything is read, so a bad one never reaches the driver
    const view = try cuda.qmmf.pinned(cuda.modelopt.bf16Weight(0, n, k), sk);
    const bytes = try stitchCut(a, store, prefix, parts, ranges, n, k);
    defer a.free(bytes);
    var p = try projection.dense(a, d, .{ .dtype = .bf16, .rank = 2, .shape = .{ n, k, 1, 1, 1 }, .bytes = bytes });
    p.native.?.sk = view.sk;
    return p;
}

test {
    _ = @import("cuda_projection_load_test.zig");
}
