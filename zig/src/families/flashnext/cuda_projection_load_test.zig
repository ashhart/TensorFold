//! MXFP8 stitching through a store that moves on every read, like mapped shards; BF16 cuts by rows or input columns.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Cut = @import("cuda_modelopt_experts.zig").Cut;
const source = @import("cuda_projection_load.zig");
const Tensor = core.safetensors.Tensor;

const Fake = struct {
    scratch: [1024]u8 = undefined,
    bad: enum { none, weight, scale, extent, width, missing } = .none,

    pub fn tensor(self: *Fake, name: []const u8) !Tensor {
        const part: usize = if (std.mem.startsWith(u8, name, "root.a.")) 0 else if (std.mem.startsWith(u8, name, "root.b.")) 1 else return error.MissingTensor;
        const n: usize = if (part == 0) 3 else 5;
        const k: usize = if (self.bad == .width and part == 1) 128 else 64;
        const scale = std.mem.endsWith(u8, name, ".weight_scale");
        if (scale and self.bad == .missing) return error.MissingTensor;
        if (!scale and !std.mem.endsWith(u8, name, ".weight")) return error.MissingTensor;
        const cols = if (scale) k / 32 else k;
        const len = n * cols;
        @memset(&self.scratch, 0xee);
        for (self.scratch[0..len], 0..) |*v, i| v.* = @truncate(i * 17 + part * 53 + @as(usize, if (scale) 129 else 7));
        return .{
            .dtype = if (scale) (if (self.bad == .scale) .f8_e4m3 else .u8) else (if (self.bad == .weight) .bf16 else .f8_e4m3),
            .rank = 2,
            .shape = .{ n, cols, 1, 1, 1 },
            .bytes = self.scratch[0 .. len - @as(usize, if (self.bad == .extent) 1 else 0)],
        };
    }
};

fn trial(a: std.mem.Allocator) !void {
    var fake: Fake = .{};
    var value = try source.readStitched(a, &fake, "root", &.{ ".a", ".b" });
    defer value.deinit(a);
    try std.testing.expectEqual(@as(usize, 8), value.n);
    try std.testing.expectEqual(@as(usize, 64), value.k);
    for (0..2) |part| {
        const start: usize = if (part == 0) 0 else 3;
        const n: usize = if (part == 0) 3 else 5;
        for (0..n * 64) |i| try std.testing.expectEqual(@as(u8, @truncate(i * 17 + part * 53 + 7)), value.codes[start * 64 + i]);
        for (0..n * 2) |i| try std.testing.expectEqual(@as(u8, @truncate(i * 17 + part * 53 + 129)), value.scales[start * 2 + i]);
    }
}

test "MXFP8 fusion owns each shard plane and preserves exact output row order" {
    try trial(std.testing.allocator);
}

test "MXFP8 fusion cleans every partial allocation on failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, trial, .{});
}

test "MXFP8 source refuses incompatible shapes, dtypes, extents and missing scales" {
    const a = std.testing.allocator;
    for ([_]@FieldType(Fake, "bad"){ .weight, .scale, .extent, .width }) |bad| {
        var fake: Fake = .{ .bad = bad };
        try std.testing.expectError(error.UnexpectedTensor, source.readStitched(a, &fake, "root", &.{ ".a", ".b" }));
    }
    var fake: Fake = .{ .bad = .missing };
    try std.testing.expectError(error.MissingTensor, source.readMicro(a, &fake, "root.a"));
    try std.testing.expectError(error.UnexpectedTensor, source.readStitched(a, &fake, "root", &.{}));
    try std.testing.expectError(error.UnexpectedTensor, source.readStitched(a, &fake, "root", &.{ ".a", ".a", ".a", ".a", ".a", ".a", ".a" }));
}

const rows6 = 6;
const cols256 = 256;

/// A [6, 256] BF16 tensor whose every element holds a distinct value.
fn dense(bytes: *[rows6 * cols256 * 2]u8) Tensor {
    for (0..rows6 * cols256) |i| std.mem.writeInt(u16, bytes[i * 2 ..][0..2], @intCast(i + 1), .little);
    return .{ .dtype = .bf16, .rank = 2, .shape = .{ rows6, cols256, 1, 1, 1 }, .bytes = bytes };
}

fn at(bytes: []const u8, i: usize) u16 {
    return std.mem.readInt(u16, bytes[i * 2 ..][0..2], .little);
}

test "BF16 row and column cuts equal the source, and two ranks cover every element once" {
    const a = std.testing.allocator;
    var bytes: [rows6 * cols256 * 2]u8 = undefined;
    const t = dense(&bytes);
    var seen: [rows6 * cols256]u8 = @splat(0);
    for (0..2) |r| {
        const cut = try Cut.init(cols256, 2, r);
        const cols = try source.cutBytes(a, t, cut, .cols);
        defer a.free(cols);
        try std.testing.expectEqual(rows6 * 128 * 2, cols.len);
        for (0..rows6) |row| for (0..128) |c| {
            const v = at(cols, row * 128 + c);
            try std.testing.expectEqual(at(t.bytes, row * cols256 + cut.from + c), v);
            seen[v - 1] += 1;
        };
        const half: Cut = .{ .from = 3 * r, .to = 3 * r + 3 };
        const rows = try source.cutBytes(a, t, half, .rows);
        defer a.free(rows);
        try std.testing.expectEqualSlices(u8, t.bytes[half.from * cols256 * 2 ..][0 .. 3 * cols256 * 2], rows);
        for (0..rows.len / 2) |i| seen[at(rows, i) - 1] += 1;
    }
    for (seen) |count| try std.testing.expectEqual(@as(u8, 2), count);
}

fn cutTrial(a: std.mem.Allocator) !void {
    var bytes: [rows6 * cols256 * 2]u8 = undefined;
    const t = dense(&bytes);
    for ([_]source.Axis{ .rows, .cols }) |axis| {
        const out = try source.cutBytes(a, t, .{ .from = 2, .to = if (axis == .rows) 4 else 66 }, axis);
        a.free(out);
    }
}

test "BF16 cuts free every allocation on failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, cutTrial, .{});
}

const Plain = struct {
    t: Tensor,

    pub fn tensor(self: *Plain, _: []const u8) !Tensor {
        return self.t;
    }
};

test "BF16 cuts refuse other formats, a wrong extent and a column width not a multiple of 64" {
    const a = std.testing.allocator;
    var bytes: [rows6 * cols256 * 2]u8 = undefined;
    const t = dense(&bytes);
    const half: Cut = .{ .from = 128, .to = 256 };
    var other = t;
    other.dtype = .f8_e4m3;
    try std.testing.expectError(error.UnexpectedTensor, source.cutBytes(a, other, half, .cols));
    var short = t;
    short.bytes = t.bytes[0 .. t.bytes.len - 2];
    try std.testing.expectError(error.UnexpectedTensor, source.cutBytes(a, short, half, .cols));
    try std.testing.expectError(error.UnexpectedTensor, source.cutBytes(a, t, .{ .from = 128, .to = 320 }, .cols));
    try std.testing.expectError(error.UnexpectedTensor, source.cutBytes(a, t, .{ .from = 3, .to = 7 }, .rows));
    try std.testing.expectError(error.UnexpectedTensor, source.cutBytes(a, t, .{ .from = 4, .to = 4 }, .rows));
    try std.testing.expectError(error.UnexpectedTensor, source.cutBytes(a, t, try Cut.init(cols256, 8, 1), .cols));
    // the refusals return before the driver is touched
    const d: *const cuda.Driver = undefined;
    var store: Plain = .{ .t = other };
    try std.testing.expectError(error.UnsupportedQuantization, source.loadCut(a, d, &store, "p", rows6, cols256, half, .cols));
    store.t = t;
    try std.testing.expectError(error.UnexpectedTensor, source.loadCut(a, d, &store, "p", rows6, 512, half, .cols));
}

const heads_cut = @import("heads_cut.zig");
const Heads = @import("cuda_heads.zig").Heads;
const Ranges = source.Ranges;
const qmmf = cuda.qmmf;
const k64 = 64;

/// The whole checkpoint parts at 64 inputs, each read into one buffer that the next read overwrites.
const Whole = struct {
    scratch: []u8,
    dtype: @FieldType(Tensor, "dtype") = .bf16,

    const names = [_][]const u8{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj" };
    const counts = [_]usize{ 10240, 6144, 48, 48, 12288, 512, 512, 640 };

    fn part(name: []const u8) !usize {
        if (!std.mem.startsWith(u8, name, "root.") or !std.mem.endsWith(u8, name, ".weight")) return error.MissingTensor;
        const stem = name["root.".len .. name.len - ".weight".len];
        for (names, 0..) |n, i| if (std.mem.eql(u8, n, stem)) return i;
        return error.MissingTensor;
    }

    fn value(p: usize, i: usize) u16 {
        const x: u32 = @truncate((i + p * 0x1000003) *% 0x9E3779B1);
        return @truncate(x >> 16);
    }

    pub fn tensor(self: *Whole, name: []const u8) !Tensor {
        const p = try part(name);
        const n = counts[p];
        const len = switch (self.dtype) {
            .f8_e4m3 => n * k64,
            .u32 => n * (k64 / 8) * 4,
            else => n * k64 * 2,
        };
        @memset(self.scratch, 0xee);
        for (0..n * k64) |i| std.mem.writeInt(u16, self.scratch[i * 2 ..][0..2], value(p, i), .little);
        return .{ .dtype = self.dtype, .rank = 2, .shape = .{ n, if (self.dtype == .u32) k64 / 8 else k64, 1, 1, 1 }, .bytes = self.scratch[0..len] };
    }
};

fn wholeStore(a: std.mem.Allocator) !Whole {
    return .{ .scratch = try a.alloc(u8, 12288 * k64 * 2) };
}

test "the head split keeps each stitched part's ranged rows in order, and two ranks cover every row" {
    const a = std.testing.allocator;
    var fake = try wholeStore(a);
    defer a.free(fake.scratch);
    for ([_]bool{ true, false }) |linear| {
        const parts = if (linear) heads_cut.delta_parts else heads_cut.attention_parts;
        const first: usize = if (linear) 0 else 4;
        var seen: [4][12288]u8 = @splat(@splat(0));
        for (0..2) |r| {
            const s = if (linear) try heads_cut.delta(.half, r) else try heads_cut.attention(.half, r);
            try std.testing.expectEqual(@as(usize, if (linear) 8240 else 7296), s.n);
            const bytes = try source.stitchCut(a, &fake, "root", &parts, &s.ranges, s.n, k64);
            defer a.free(bytes);
            try std.testing.expectEqual(s.n * k64 * 2, bytes.len);
            var pos: usize = 0;
            for (parts, s.ranges, 0..) |name, kept, i| {
                var name_buf: [64]u8 = undefined;
                const whole = try a.dupe(u8, (try fake.tensor(try std.fmt.bufPrint(&name_buf, "root{s}.weight", .{name}))).bytes);
                defer a.free(whole);
                try std.testing.expectEqual(Whole.counts[first + i] * k64 * 2, whole.len);
                for (kept.items()) |c| {
                    const len = c.width() * k64 * 2;
                    try std.testing.expectEqualSlices(u8, whole[c.from * k64 * 2 ..][0..len], bytes[pos..][0..len]);
                    for (c.from..c.to) |row| seen[i][row] += 1;
                    pos += len;
                }
            }
            try std.testing.expectEqual(bytes.len, pos);
        }
        for (0..4) |i| for (seen[i][0..Whole.counts[first + i]]) |count| {
            try std.testing.expectEqual(@as(u8, if (!linear and i == 3) 2 else 1), count);
        };
    }
}

fn expectRanges(want: []const [2]usize, got: Ranges) !void {
    try std.testing.expectEqual(want.len, got.len);
    for (want, got.items()) |w, c| {
        try std.testing.expectEqual(w[0], c.from);
        try std.testing.expectEqual(w[1], c.to);
    }
}

test "each rank's ranges are the design's: q, k, v, z, b, a, the attention heads, the indexer whole" {
    for (0..2) |r| {
        const d = try heads_cut.delta(.half, r);
        const qkv = [_][2]usize{ .{ 1024 * r, 1024 * r + 1024 }, .{ 2048 + 1024 * r, 3072 + 1024 * r }, .{ 4096 + 3072 * r, 7168 + 3072 * r } };
        try expectRanges(&qkv, d.ranges[0]);
        try expectRanges(&.{.{ 3072 * r, 3072 * r + 3072 }}, d.ranges[1]);
        for (d.ranges[2..]) |kept| try expectRanges(&.{.{ 24 * r, 24 * r + 24 }}, kept);
        try expectRanges(&qkv, try heads_cut.convChannels(.half, r));
        const t = try heads_cut.attention(.half, r);
        try expectRanges(&.{.{ 6144 * r, 6144 * r + 6144 }}, t.ranges[0]);
        for (t.ranges[1..3]) |kept| try expectRanges(&.{.{ 256 * r, 256 * r + 256 }}, kept);
        try expectRanges(&.{.{ 0, 640 }}, t.ranges[3]);
    }
    const whole = try heads_cut.delta(.full, 0);
    try std.testing.expectEqual(@as(usize, 16480), whole.n);
    try expectRanges(&.{.{ 0, 6144 }}, whole.ranges[1]);
    try std.testing.expectEqual(@as(usize, 13952), (try heads_cut.attention(.full, 0)).n);
    try std.testing.expectError(error.UnevenHeadSplit, heads_cut.delta(.full, 1));
    try std.testing.expectError(error.UnevenHeadSplit, heads_cut.attention(.half, 2));
    try std.testing.expectError(error.UnevenHeadSplit, heads_cut.delta(.{ .nk = 8, .nv = 24, .q = 24, .kv = 2 }, 0));
    try std.testing.expectError(error.UnevenHeadSplit, heads_cut.valueHeads(.{ .nk = 0, .nv = 24, .q = 12, .kv = 1 }, 0));
    try std.testing.expectError(error.UnevenHeadSplit, heads_cut.valueHeads(.{ .nk = 8, .nv = 0x80000018, .q = 12, .kv = 1 }, 0));
}

test "a cut projection pins the whole stitched shape's K slices, so each kept row plans as the whole one does" {
    for ([_]bool{ true, false }) |linear| {
        const whole_n: usize = if (linear) 16480 else 13952;
        for (0..2) |r| {
            const s = if (linear) try heads_cut.delta(.half, r) else try heads_cut.attention(.half, r);
            try std.testing.expectEqual(qmmf.splitK(whole_n, 2560), s.sk);
            try std.testing.expect(qmmf.splitK(s.n, 2560) != s.sk);
            const w = try qmmf.pinned(cuda.modelopt.bf16Weight(0, s.n, 2560), s.sk);
            for ([_]bool{ true, false }) |cl| for ([_]usize{ 1, 16, 64, 1536 }) |m| {
                try std.testing.expectEqual(qmmf.plan(m, whole_n, 2560, cl), qmmf.planFor(m, w, cl));
            };
        }
    }
}

test "conv1d keeps the rank's q, k and v channels, and A_log and dt_bias its 24 value heads" {
    const a = std.testing.allocator;
    const whole = try a.alloc(u8, 10240 * 4 * 2);
    defer a.free(whole);
    for (0..10240 * 4) |i| std.mem.writeInt(u16, whole[i * 2 ..][0..2], Whole.value(9, i), .little);
    const t: Tensor = .{ .dtype = .bf16, .rank = 3, .shape = .{ 10240, 1, 4, 1, 1 }, .bytes = whole };
    var values: [48]f32 = undefined;
    for (&values, 0..) |*v, i| v.* = @floatFromInt(i);
    for (0..2) |r| {
        const local = try heads_cut.convRows(a, t, .half, r);
        defer a.free(local);
        try std.testing.expectEqual(@as(usize, 5120 * 4 * 2), local.len);
        for (0..5120) |row| {
            const channel = if (row < 1024) 1024 * r + row else if (row < 2048) 2048 + 1024 * r + row - 1024 else 4096 + 3072 * r + row - 2048;
            try std.testing.expectEqualSlices(u8, whole[channel * 8 ..][0..8], local[row * 8 ..][0..8]);
        }
        const kept = try heads_cut.valueSlice(&values, .half, r);
        try std.testing.expectEqual(@as(usize, 24), kept.len);
        for (kept, 0..) |v, i| try std.testing.expectEqual(@as(f32, @floatFromInt(24 * r + i)), v);
    }
    var short = t;
    short.bytes = whole[0 .. whole.len - 2];
    try std.testing.expectError(error.UnexpectedTensor, heads_cut.convRows(a, short, .half, 0));
    var other = t;
    other.dtype = .f32;
    try std.testing.expectError(error.UnexpectedTensor, heads_cut.convRows(a, other, .half, 0));
    try std.testing.expectError(error.UnexpectedTensor, heads_cut.valueSlice(values[0..24], .half, 0));
}

test "stitched cuts refuse MXFP8 and affine parts, other widths and a wrong local row count" {
    const a = std.testing.allocator;
    var fake = try wholeStore(a);
    defer a.free(fake.scratch);
    const s = try heads_cut.delta(.half, 0);
    // the refusals return before the driver is touched
    const d: *const cuda.Driver = undefined;
    for ([_]@FieldType(Tensor, "dtype"){ .f8_e4m3, .u32 }) |dtype| {
        fake.dtype = dtype;
        try std.testing.expectError(error.UnsupportedQuantization, source.loadStitchedCut(a, d, &fake, "root", &heads_cut.delta_parts, &s.ranges, s.n, k64, s.sk));
    }
    fake.dtype = .bf16;
    try std.testing.expectError(error.UnexpectedTensor, source.loadStitchedCut(a, d, &fake, "root", &heads_cut.delta_parts, &s.ranges, s.n, 128, s.sk));
    try std.testing.expectError(error.UnexpectedTensor, source.loadStitchedCut(a, d, &fake, "root", &heads_cut.delta_parts, &s.ranges, s.n + 1, k64, s.sk));
    try std.testing.expectError(error.UnexpectedTensor, source.loadStitchedCut(a, d, &fake, "root", heads_cut.delta_parts[0..3], &s.ranges, s.n, k64, s.sk));
    try std.testing.expectError(error.BadSplit, source.loadStitchedCut(a, d, &fake, "root", &heads_cut.delta_parts, &s.ranges, s.n, k64, 3));
    var past = s.ranges;
    past[2] = Ranges.of(&.{.{ .from = 40, .to = 64 }});
    try std.testing.expectError(error.UnexpectedTensor, source.stitchCut(a, &fake, "root", &heads_cut.delta_parts, &past, s.n, k64));
}
