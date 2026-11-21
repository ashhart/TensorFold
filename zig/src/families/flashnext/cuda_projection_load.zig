//! Projection loading preserves dense BF16 bits, MXFP8 codes/scales and block-FP8 planes; affine keeps its packer.
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

/// Block-FP8 (e4m3 weight, fp32 weight_scale_inv per 128x128 block) parts of one projection, rows stacked in order.
pub const Block8 = struct {
    n: usize,
    k: usize,
    codes: []u8,
    sinv: []f32,

    pub fn deinit(self: *Block8, a: std.mem.Allocator) void {
        a.free(self.codes);
        a.free(self.sinv);
        self.* = undefined;
    }
};

/// Whether `{prefix}.weight_scale_inv` is there: the one tensor a block-FP8 projection carries and the rest do not.
pub fn isBlock8(store: anytype, prefix: []const u8) bool {
    var buf: [512]u8 = undefined;
    return store.has(std.fmt.bufPrint(&buf, "{s}.weight_scale_inv", .{prefix}) catch return false);
}

/// How many leading parts of a stitched projection are block FP8, and so where a BF16 tail could start.
pub fn block8Parts(store: anytype, prefix: []const u8, names: []const []const u8) usize {
    var buf: [512]u8 = undefined;
    var heads: usize = 0;
    while (heads < names.len and isBlock8(store, std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, names[heads] }) catch return heads)) heads += 1;
    return heads;
}

/// A projection's leading block-FP8 parts: e4m3 codes [n, k] and fp32 scales [n / 128, k / 128], stacked in order.
pub fn readBlock8(a: std.mem.Allocator, store: anytype, prefix: []const u8, names: []const []const u8, why: ?*core.modelopt.Why) !Block8 {
    var buf: [512]u8 = undefined;
    var n: usize = 0;
    var k: usize = 0;
    for (names) |name| {
        const w = try store.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, name }));
        if (w.dtype != .f8_e4m3 or w.rank != 2 or w.dim(0) % 128 != 0 or w.dim(1) % 128 != 0) return error.UnexpectedTensor;
        if (k != 0 and k != w.dim(1)) return error.UnexpectedTensor;
        k = w.dim(1);
        n = std.math.add(usize, n, w.dim(0)) catch return error.UnexpectedTensor;
    }
    const kb = k / 128;
    const codes = try a.alloc(u8, std.math.mul(usize, n, k) catch return error.UnexpectedTensor);
    errdefer a.free(codes);
    const sinv = try a.alloc(f32, (n / 128) * kb);
    errdefer a.free(sinv);
    var row: usize = 0;
    for (names) |name| {
        defer store.release();
        const pn = (try store.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, name }))).dim(0);
        const h = try core.quant.fp8.read(store, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, name }), pn, k, .f32, why);
        if (h.weight.bytes.len != pn * k) return error.UnexpectedTensor;
        @memcpy(codes[row * k ..][0 .. pn * k], h.weight.bytes);
        try core.quant.fp8.widen(h.scale_inv, sinv[(row / 128) * kb ..][0 .. (pn / 128) * kb], true);
        row += pn;
    }
    if (row != n) return error.UnexpectedTensor;
    return .{ .n = n, .k = k, .codes = codes, .sinv = sinv };
}

/// A projection's trailing BF16 parts [nt, k], stacked in order: the exact stored rows the tail's own matmul runs on.
pub fn readTail(a: std.mem.Allocator, store: anytype, prefix: []const u8, names: []const []const u8, k: usize, nt: usize) ![]u8 {
    var buf: [512]u8 = undefined;
    const bytes = std.math.mul(usize, nt, k * 2) catch return error.UnexpectedTensor;
    const out = try a.alloc(u8, bytes);
    errdefer a.free(out);
    var at: usize = 0;
    for (names) |name| {
        const t = try store.tensor(try std.fmt.bufPrint(&buf, "{s}{s}.weight", .{ prefix, name }));
        if (t.dtype != .bf16 or t.rank != 2 or t.dim(1) != k) return error.UnexpectedTensor;
        if (t.bytes.len > bytes - at) return error.UnexpectedTensor;
        @memcpy(out[at..][0..t.bytes.len], t.bytes);
        at += t.bytes.len;
    }
    if (at != bytes) return error.UnexpectedTensor;
    return out;
}

/// A projection whose leading parts are block FP8 and whose trailing parts are BF16, as INT4-AutoRound's mixers are.
pub fn loadBlock8(a: std.mem.Allocator, d: *cuda.Driver, store: anytype, prefix: []const u8, names: []const []const u8, n: usize, k: usize) !wt.Q4 {
    const heads = block8Parts(store, prefix, names);
    if (heads == 0) return error.UnexpectedTensor;
    var head = try readBlock8(a, store, prefix, names[0..heads], null);
    defer head.deinit(a);
    if (head.k != k or head.n > n) return error.UnexpectedTensor;
    const nt = n - head.n;
    if (nt == 0) return projection.fp8g(a, d, head.codes, head.sinv, head.n, k, null, 0);
    // the scratch's tail holds wt.tail_n columns; a wider one has no buffer to be copied out of
    if (nt > wt.tail_n) return error.UnexpectedTensor;
    const tail = try readTail(a, store, prefix, names[heads..], k, nt);
    defer a.free(tail);
    return projection.fp8g(a, d, head.codes, head.sinv, head.n, k, tail, nt);
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
            if (isBlock8(store, prefix)) return loadBlock8(a, d, store, prefix, &.{""}, n, k);
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
    if (first.dtype == .f8_e4m3 and isBlock8(store, try std.fmt.bufPrint(&buf, "{s}{s}", .{ prefix, names[0] }))) return loadBlock8(a, d, store, prefix, names, n, k);
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
