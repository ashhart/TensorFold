//! MXFP8 stitching through a store that moves on every read, like mapped shards; BF16 cuts by rows or input columns.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Cut = @import("cuda_modelopt_experts.zig").Cut;
const source = @import("cuda_projection_load.zig");
const projection = @import("cuda_projection.zig");
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

const block_k = 256;
const block_kb = block_k / 128;
/// The fixture's BF16 tail: more rows than wt.tail_n, so the loader's width guard can be told from a short read.
const tail_rows = 768;

/// e4m3 codes whose binary64 values are exact: 0x38 1.0, 0x3c 1.5, 0x40 2.0, 0x34 0.75, 0xb8 -1.0.
const code_values = [_]u8{ 0x38, 0x3c, 0x40, 0x34, 0xb8 };

/// A block-FP8 part of the fixture: e4m3 codes [rows, block_k] and one fp32 scale a 128 x 128 block.
const Part = struct { rows: usize, codes: []u8, inv: []f32 };

fn fixturePart(a: std.mem.Allocator, rows: usize, seed: usize) !Part {
    const codes = try a.alloc(u8, rows * block_k);
    errdefer a.free(codes);
    for (0..rows) |r| for (0..block_k) |c| {
        codes[r * block_k + c] = code_values[(r + c + seed) % code_values.len];
    };
    const inv = try a.alloc(f32, (rows / 128) * block_kb);
    for (0..rows / 128) |i| for (0..block_kb) |j| {
        inv[i * block_kb + j] = @floatFromInt(i * 10 + j + 1 + seed * 3);
    };
    return .{ .rows = rows, .codes = codes, .inv = inv };
}

fn freePart(a: std.mem.Allocator, p: *Part) void {
    a.free(p.codes);
    a.free(p.inv);
}

const Bad = enum { none, weight_dtype, rows, cols, scale_shape, scale_nan, tail_dtype, tail_cols, tail_short };

/// One stitched projection's store: two block-FP8 parts, ".q" and ".k", and the BF16 indexer tail, with faults.
const Blocky = struct {
    k: usize,
    parts: [2]Part,
    tail: []u8,
    bad: Bad = .none,

    /// The part a name selects, and whether it is the scale tensor beside that part's weight.
    fn select(name: []const u8) ?struct { i: usize, scale: bool } {
        const i: usize = if (std.mem.startsWith(u8, name, "m.q.")) 0 else if (std.mem.startsWith(u8, name, "m.k.")) 1 else return null;
        if (std.mem.endsWith(u8, name, ".weight_scale_inv")) return .{ .i = i, .scale = true };
        if (std.mem.endsWith(u8, name, ".weight")) return .{ .i = i, .scale = false };
        return null;
    }

    /// Only a block-FP8 projection carries a scale_inv beside its weight: what selects this loader's arm.
    pub fn has(self: *Blocky, name: []const u8) bool {
        _ = self;
        const s = select(name) orelse return false;
        return s.scale;
    }

    /// The checkpoint-reader adapter the shared block-FP8 reader takes (fp8.read -> get).
    pub fn get(self: *Blocky, name: []const u8) !Tensor {
        return self.tensor(name);
    }

    /// A reading borrow's pages would go here; this store keeps its bytes in memory.
    pub fn release(self: *Blocky) void {
        _ = self;
    }

    pub fn tensor(self: *Blocky, name: []const u8) !Tensor {
        if (std.mem.startsWith(u8, name, "m.idx.weight")) {
            const cols: usize = if (self.bad == .tail_cols) block_k / 2 else block_k;
            const want = self.tail.len / 2;
            const bytes = if (self.bad == .tail_short) self.tail[0 .. want - 1] else self.tail;
            const dtype: core.safetensors.DType = if (self.bad == .tail_dtype) .f16 else .bf16;
            return .{ .dtype = dtype, .rank = 2, .shape = .{ self.tail.len / (block_k * 2), cols, 1, 1, 1 }, .bytes = bytes };
        }
        const s = select(name) orelse return error.MissingTensor;
        const p = self.parts[s.i];
        if (!s.scale) {
            const rows = if (self.bad == .rows and s.i == 0) 200 else p.rows;
            const cols = if (self.bad == .cols and s.i == 0) 128 else self.k;
            const dtype: core.safetensors.DType = if (self.bad == .weight_dtype and s.i == 0) .bf16 else .f8_e4m3;
            return .{ .dtype = dtype, .rank = 2, .shape = .{ rows, cols, 1, 1, 1 }, .bytes = p.codes[0 .. rows * cols] };
        }
        // a wrong block count, whole entries so the fake's own slice stays inside its buffer
        const cols: usize = if (self.bad == .scale_shape and s.i == 0) 1 else block_kb;
        const rows = p.rows / 128;
        return .{ .dtype = .f32, .rank = 2, .shape = .{ rows, cols, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(p.inv[0 .. rows * cols]) };
    }
};

fn blocky(a: std.mem.Allocator) !Blocky {
    var out: Blocky = .{ .k = block_k, .parts = undefined, .tail = &.{} };
    out.parts[0] = try fixturePart(a, 256, 0);
    errdefer freePart(a, &out.parts[0]);
    out.parts[1] = try fixturePart(a, 128, 1);
    errdefer freePart(a, &out.parts[1]);
    out.tail = try a.alloc(u8, tail_rows * block_k * 2);
    for (0..out.tail.len / 2) |i| std.mem.writeInt(u16, out.tail[i * 2 ..][0..2], @truncate(i * 37 + 3), .little);
    return out;
}

fn freeBlocky(a: std.mem.Allocator, b: *Blocky) void {
    for (&b.parts) |*p| freePart(a, p);
    a.free(b.tail);
}

test "block FP8: a projection's FP8 parts and its BF16 tail are the stored bytes the binary64 reference reads" {
    const a = std.testing.allocator;
    var fake = try blocky(a);
    defer freeBlocky(a, &fake);
    try std.testing.expectEqual(@as(usize, 2), source.block8Parts(&fake, "m", &.{ ".q", ".k", ".idx" }));
    try std.testing.expectEqual(@as(usize, 0), source.block8Parts(&fake, "m", &.{ ".idx", ".q" }));
    var head = try source.readBlock8(a, &fake, "m", &.{ ".q", ".k" }, null);
    defer head.deinit(a);
    try std.testing.expectEqual(@as(usize, 384), head.n);
    try std.testing.expectEqual(block_k, head.k);
    try std.testing.expectEqualSlices(u8, fake.parts[0].codes, head.codes[0 .. 256 * block_k]);
    try std.testing.expectEqualSlices(u8, fake.parts[1].codes, head.codes[256 * block_k ..]);
    try std.testing.expectEqualSlices(f32, fake.parts[0].inv, head.sinv[0..4]);
    try std.testing.expectEqualSlices(f32, fake.parts[1].inv, head.sinv[4..]);
    const tail = try source.readTail(a, &fake, "m", &.{".idx"}, block_k, tail_rows);
    defer a.free(tail);
    try std.testing.expectEqualSlices(u8, fake.tail, tail);
    // the served tail is whole 128-row blocks, or the plane builder above could not splice it
    try std.testing.expectEqual(@as(usize, 640), @import("cuda_prompt_wt.zig").tail_n);
    try std.testing.expectEqual(@as(usize, 0), @import("cuda_prompt_wt.zig").tail_n % 128);
    try std.testing.expect(@import("cuda_prompt_wt.zig").tail_n <= @import("cuda_prompt_wt.zig").proj_n);

    // the lane's planes: the packed codes and per-64-input scale tiles of those stored rows, then the tail's zeros
    const n = 384 + tail_rows;
    const rows = try a.alloc(u8, n * block_k);
    defer a.free(rows);
    @memcpy(rows[0..head.codes.len], head.codes);
    @memset(rows[head.codes.len..], 0);
    const inv = try a.alloc(f32, (n / 128) * block_kb);
    defer a.free(inv);
    @memcpy(inv[0..head.sinv.len], head.sinv);
    @memset(inv[head.sinv.len..], 1.0);
    const want_codes = try a.alloc(u8, cuda.fp8.padded(n) * block_k);
    defer a.free(want_codes);
    cuda.fp8.packCodes(want_codes, rows, n, block_k);
    const want_scales = try a.alloc(f32, cuda.fp8.padded(n) * (block_k / cuda.fp8.group));
    defer a.free(want_scales);
    cuda.fp8.packScales(want_scales, inv, n, block_k);
    var planes = try projection.fp8gPlanes(a, head.codes, head.sinv, head.n, block_k, tail_rows);
    defer planes.deinit(a);
    try std.testing.expectEqualSlices(u8, want_codes, planes.codes);
    try std.testing.expectEqualSlices(f32, want_scales, planes.scales);

    // the reference of the checkpoint's own tensors, code x its block scale, exact in binary64
    const ref = try a.alloc(f64, 256 * block_k);
    defer a.free(ref);
    try core.quant.fp8.reference(.{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 256, block_k, 1, 1, 1 }, .bytes = fake.parts[0].codes },
        .scale_inv = .{ .dtype = .f32, .rank = 2, .shape = .{ 2, block_kb, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(fake.parts[0].inv) },
    }, ref);
    try std.testing.expectEqual(@as(f64, 1.0), ref[0]); // 1.0 x 1
    try std.testing.expectEqual(@as(f64, 1.5), ref[1]); // 1.5 x 1
    try std.testing.expectEqual(@as(f64, 1.5), ref[128]); // 0.75 x 2
    try std.testing.expectEqual(@as(f64, 8.25), ref[128 * block_k]); // 0.75 x 11
    try std.testing.expectEqual(@as(f64, 18.0), ref[128 * block_k + 128]); // 1.5 x 12

    // a tail row on the lane is a zero code times a scale of one: its binary64 contribution is exactly +0
    const zero_codes = try a.alloc(u8, 128 * block_k);
    defer a.free(zero_codes);
    @memset(zero_codes, 0);
    const one_scale = try a.alloc(f32, block_kb);
    defer a.free(one_scale);
    @memset(one_scale, 1.0);
    const tail_ref = try a.alloc(f64, zero_codes.len);
    defer a.free(tail_ref);
    try core.quant.fp8.reference(.{
        .weight = .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ 128, block_k, 1, 1, 1 }, .bytes = zero_codes },
        .scale_inv = .{ .dtype = .f32, .rank = 2, .shape = .{ 1, block_kb, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(one_scale) },
    }, tail_ref);
    for (tail_ref) |v| try std.testing.expectEqual(@as(u64, 0), @as(u64, @bitCast(v)));
    for (inv[head.sinv.len..]) |s| try std.testing.expectEqual(@as(f32, 1.0), s);
}

/// A store holding one MXFP8 projection (e4m3 weight, u8 group-32 weight_scale) or its affine form (u32 words).
const MicroStore = struct {
    dtype: core.safetensors.DType = .f8_e4m3,
    scratch: [64 * 256]u8 = undefined,

    /// Neither form carries a scale_inv a 128 x 128 block covers: the block-FP8 arm is not this projection's.
    pub fn has(self: *MicroStore, name: []const u8) bool {
        _ = self;
        _ = name;
        return false;
    }

    pub fn tensor(self: *MicroStore, name: []const u8) !Tensor {
        @memset(&self.scratch, 0x5a);
        if (std.mem.endsWith(u8, name, ".weight_scale")) return .{ .dtype = .u8, .rank = 2, .shape = .{ 64, 8, 1, 1, 1 }, .bytes = self.scratch[0 .. 64 * 8] };
        if (!std.mem.endsWith(u8, name, ".weight")) return error.MissingTensor;
        const affine = self.dtype == .u32;
        const cols: usize = if (affine) 32 else 256;
        const words: usize = if (affine) 4 else 1;
        const len = 64 * cols * words;
        return .{ .dtype = self.dtype, .rank = 2, .shape = .{ 64, cols, 1, 1, 1 }, .bytes = self.scratch[0..len] };
    }
};

test "block FP8: MXFP8 and affine projections are not block FP8 and take the reader they took before" {
    const a = std.testing.allocator;
    var fake: MicroStore = .{};
    try std.testing.expect(!source.isBlock8(&fake, "m.q"));
    var value = try source.readMicro(a, &fake, "m.q");
    defer value.deinit(a);
    try std.testing.expectEqual(@as(usize, 64), value.n);
    try std.testing.expectEqual(@as(usize, 256), value.k);
    try std.testing.expectEqual(@as(usize, 64 * 256), value.codes.len);
    try std.testing.expectEqual(@as(usize, 64 * 8), value.scales.len);
    fake.dtype = .u32;
    try std.testing.expect(!source.isBlock8(&fake, "m.q"));
    try std.testing.expectError(error.UnexpectedTensor, source.readMicro(a, &fake, "m.q"));
}

test "block FP8: the loader refuses the shapes and the planes it cannot splice, before any device call" {
    const a = std.testing.allocator;
    var fake = try blocky(a);
    defer freeBlocky(a, &fake);
    var why: core.modelopt.Why = .{};
    for ([_]Bad{ .weight_dtype, .rows, .cols, .scale_shape }) |bad| {
        fake.bad = bad;
        try std.testing.expectError(error.UnexpectedTensor, source.readBlock8(a, &fake, "m", &.{ ".q", ".k" }, &why));
    }
    fake.bad = .scale_nan;
    fake.parts[0].inv[0] = std.math.nan(f32);
    try std.testing.expectError(error.UnsupportedQuantization, source.readBlock8(a, &fake, "m", &.{ ".q", ".k" }, &why));
    fake.bad = .none;
    fake.parts[0].inv[0] = 1.0;
    {
        const d: *cuda.Driver = undefined;
        for ([_]usize{ 0, 383, 641 }) |n| try std.testing.expectError(error.UnexpectedTensor, source.loadBlock8(a, d, &fake, "m", &.{ ".q", ".k" }, n, block_k));
        try std.testing.expectError(error.UnexpectedTensor, source.loadBlock8(a, d, &fake, "m", &.{ ".q", ".k" }, 640, block_k / 2));
        try std.testing.expectError(error.UnexpectedTensor, source.loadBlock8(a, d, &fake, "m", &.{".idx"}, 256, block_k));
        // a tail the scratch cannot hold is refused before it is read, though this store could serve it whole
        try std.testing.expectError(error.UnexpectedTensor, source.loadBlock8(a, d, &fake, "m", &.{ ".q", ".k", ".idx" }, 384 + tail_rows, block_k));
    }
    for ([_]Bad{ .tail_dtype, .tail_cols, .tail_short }) |bad| {
        fake.bad = bad;
        try std.testing.expectError(error.UnexpectedTensor, source.readTail(a, &fake, "m", &.{".idx"}, block_k, 256));
    }
    fake.bad = .none;
    try std.testing.expectError(error.UnexpectedTensor, source.readTail(a, &fake, "m", &.{".idx"}, block_k, 512));
    // the plane builder's own guards: whole 128-row blocks, a k of scale blocks, and lengths that agree
    try std.testing.expectError(error.UnexpectedTensor, projection.fp8gPlanes(a, fake.parts[0].codes, fake.parts[0].inv, 0, block_k, 0));
    try std.testing.expectError(error.UnexpectedTensor, projection.fp8gPlanes(a, fake.parts[0].codes, fake.parts[0].inv, 256, 192, 0));
    try std.testing.expectError(error.UnexpectedTensor, projection.fp8gPlanes(a, fake.parts[0].codes[1..], fake.parts[0].inv, 256, block_k, 0));
    try std.testing.expectError(error.UnexpectedTensor, projection.fp8gPlanes(a, fake.parts[0].codes, fake.parts[0].inv[1..], 256, block_k, 0));
    try std.testing.expectError(error.UnexpectedTensor, projection.fp8gPlanes(a, fake.parts[0].codes, fake.parts[0].inv, 256, block_k, 64));
    // 192 rows cannot lead a 128-row tail: the last head block would carry the tail's rows
    try std.testing.expectError(error.UnexpectedTensor, projection.fp8gPlanes(a, fake.parts[0].codes[0 .. 192 * block_k], fake.parts[0].inv[0..2], 192, block_k, 128));
}

fn blockTrial(a: std.mem.Allocator) !void {
    var fake = try blocky(a);
    defer freeBlocky(a, &fake);
    var head = try source.readBlock8(a, &fake, "m", &.{ ".q", ".k" }, null);
    defer head.deinit(a);
    const tail = try source.readTail(a, &fake, "m", &.{".idx"}, block_k, tail_rows);
    a.free(tail);
    var planes = try projection.fp8gPlanes(a, head.codes, head.sinv, head.n, block_k, tail_rows);
    planes.deinit(a);
}

test "block FP8: every partial allocation on the way to the planes is freed" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, blockTrial, .{});
}
