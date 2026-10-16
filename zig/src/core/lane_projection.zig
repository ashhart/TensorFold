//! Affine lane projections for 1-16 bf16 rows on Metal. No family or model dependency.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("frags.zig");

pub const weight_layout = "u32[ceil(N/32)][K/group][32][group*bits/32], low-bit-first codes, columns past N zero";
pub const reg_layout = "u32[N/32][K/64][2][32 lanes][4] 4-bit codes, regBlock order";
pub const metadata_layout = "bf16[K/group][N][scale,bias]";
pub const Output = enum { bf16, f32 };

pub const Format = struct {
    bits: u8,
    group: usize,

    pub fn validate(f: Format) !void {
        if (f.bits < 2 or f.bits > 8 or f.bits == 7) return error.UnsupportedBits;
        if (f.group != 32 and f.group != 64 and f.group != 128) return error.UnsupportedGroup;
    }
};

/// SK, PF and group cuts are fixed per pipeline, never chosen by row count; empty ranges mean all N/32 tiles.
pub const Layout = struct {
    n: usize,
    k: usize,
    format: Format,
    sk: usize = 1,
    pf: usize = 1,
    ranges: []const [2]usize = &.{},
    groups: ?[2]usize = null,
    output: Output = .bf16,
    /// Prepared sums reuse owned scratch; callers serialize all encodes using this projection until GPU completion.
    precompute_sums: bool = false,
    cooperative: bool = false,
    /// 4-bit g64 codes in reg_layout, unpacked into tensor-op registers; every word equals the cooperative kernel's.
    reg: bool = false,
    /// Elements between output rows (default N): a member of a fused stack writes its columns of the fused rows.
    y_stride: ?usize = null,

    pub fn validate(l: Layout) !void {
        try l.format.validate();
        if (l.cooperative and l.format.bits != 4) return error.UnsupportedCooperativeFormat;
        if (l.reg and (l.cooperative or l.format.bits != 4 or l.format.group != 64 or !l.precompute_sums)) return error.UnsupportedRegFormat;
        if ((l.cooperative or l.reg) and l.n % 32 != 0) return error.InvalidShape;
        if (l.y_stride) |s| if (s < l.n or s > std.math.maxInt(i32) / 16) return error.InvalidOutputStride;
        if (l.n == 0 or l.n % 16 != 0 or l.k == 0 or l.k % l.format.group != 0 or
            l.n > std.math.maxInt(i32) / 16 or l.k > std.math.maxInt(i32) / 16) return error.InvalidShape;
        if (l.sk == 0 or l.sk > 32 or (l.pf != 1 and l.pf != 2)) return error.InvalidSchedule;
        const g = l.groupCut();
        if (g[1] == 0 or g[0] >= l.k / l.format.group or g[1] > l.k / l.format.group - g[0]) return error.InvalidGroupCut;
        if (l.groups != null and l.output != .f32) return error.PartialOutputNeedsFloat;
        // Conservative declared storage, including the single-SK unused reduction array.
        if (l.threadgroupBytes() > 32768) return error.ThreadgroupMemoryLimit;
        var end: usize = 0;
        for (l.ranges) |r| {
            if (r[1] == 0 or r[0] < end or r[0] >= l.tileCount() or r[1] > l.tileCount() - r[0]) return error.InvalidTileRange;
            end = r[0] + r[1];
        }
        _ = try std.math.mul(usize, l.n, l.k);
    }

    pub fn groupCut(l: Layout) [2]usize {
        return l.groups orelse .{ 0, l.k / l.format.group };
    }

    /// 32-column tiles over N; a last tile of 16 columns is padded with zero codes and never stored.
    pub fn tileCount(l: Layout) usize {
        return (l.n + 31) / 32;
    }

    pub fn tiles(l: Layout) usize {
        if (l.ranges.len == 0) return l.tileCount();
        var count: usize = 0;
        for (l.ranges) |r| count += r[1];
        return count;
    }

    pub fn threadgroupBytes(l: Layout) usize {
        return l.sk * 32 * l.format.group + @max(l.sk -| 1, 1) * 16 * 32 * @sizeOf(f32);
    }

    pub fn weightWords(l: Layout) usize {
        return l.tileCount() * 32 * (l.k / 32) * l.format.bits;
    }

    /// MLX's packed W for this layout: N rows of K*bits/32 words.
    pub fn sourceWords(l: Layout) usize {
        return l.n * (l.k / 32) * l.format.bits;
    }

    pub fn metadataElements(l: Layout) usize {
        return l.n * (l.k / l.format.group) * 2;
    }

    pub fn packedBytes(l: Layout) usize {
        return l.weightWords() * 4 + l.metadataElements() * 2;
    }
};

pub fn source(a: std.mem.Allocator, l: Layout) ![]u8 {
    try l.validate();
    const g = l.groupCut();
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(a);
    try text.print(a, "#define TF_N {d}\n#define TF_K {d}\n#define TF_BITS {d}\n#define TF_GROUP {d}\n#define TF_SK {d}\n#define TF_G0 {d}\n#define TF_GN {d}\n#define TF_PF {d}\n#define TF_OUT {s}\n", .{
        l.n, l.k, l.format.bits, l.format.group, l.sk, g[0], g[1], l.pf, if (l.output == .f32) "float" else "bfloat",
    });
    try text.appendSlice(a, "inline int tf_tile(int j) {\n");
    var at: usize = 0;
    for (l.ranges) |r| {
        try text.print(a, "  if (j < {d}) return {d} + j - {d};\n", .{ at + r[1], r[0], at });
        at += r[1];
    }
    try text.appendSlice(a, if (l.ranges.len == 0) "  return j;\n}\n" else "  return 0;\n}\n");
    try text.print(a, "#define TF_PRECOMPUTE_SUMS {d}\n", .{@intFromBool(l.precompute_sums)});
    if (l.reg) try text.appendSlice(a, "#define TF_REG 1\n");
    if (l.y_stride) |s| try text.print(a, "#define TF_YS {d}\n", .{s});
    try text.appendSlice(a, ks.core_lane_projection);
    return text.toOwnedSlice(a);
}

/// Exact byte copies of MLX affine W, S and B into the kernel's layouts (no dequantization); storage disjoint.
pub fn pack(l: Layout, w: []const u32, scales: []const u16, biases: []const u16, out_w: []u32, out_sb: []u16) !void {
    try l.validate();
    const ng = l.k / l.format.group;
    const wp = l.format.group * l.format.bits / 32;
    const row_words = l.k / 32 * l.format.bits;
    if (w.len != l.sourceWords() or scales.len != l.n * ng or biases.len != l.n * ng or
        out_w.len != l.weightWords() or out_sb.len != l.metadataElements()) return error.InvalidStorage;
    if (overlap(std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(out_w)) or
        overlap(std.mem.sliceAsBytes(w), std.mem.sliceAsBytes(out_sb)) or
        overlap(std.mem.sliceAsBytes(scales), std.mem.sliceAsBytes(out_w)) or
        overlap(std.mem.sliceAsBytes(biases), std.mem.sliceAsBytes(out_w)) or
        overlap(std.mem.sliceAsBytes(scales), std.mem.sliceAsBytes(out_sb)) or
        overlap(std.mem.sliceAsBytes(biases), std.mem.sliceAsBytes(out_sb)) or
        overlap(std.mem.sliceAsBytes(out_w), std.mem.sliceAsBytes(out_sb))) return error.OverlappingStorage;
    for (0..l.tileCount()) |tile| for (0..ng) |g| for (0..32) |col| {
        const n = tile * 32 + col;
        const dst = ((tile * ng + g) * 32 + col) * wp;
        if (n >= l.n) {
            @memset(out_w[dst..][0..wp], 0);
            continue;
        }
        @memcpy(out_w[dst..][0..wp], w[n * row_words + g * wp ..][0..wp]);
        const sb = (g * l.n + n) * 2;
        out_sb[sb] = scales[n * ng + g];
        out_sb[sb + 1] = biases[n * ng + g];
    };
}

/// reg_layout for one 32-column x 64-code block, in the tensor op's right-input register order (lane_order.metal).
pub fn regBlock(tiled: *const [256]u32, out: *[256]u32) void {
    for (out, 0..) |*o, t| {
        const lane = (t >> 2) & 31;
        const w = 4 * (t >> 7) + (t & 3);
        const at = (((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1)) * 8 + ((lane >> 3) & 1) + 4 * (w & 1);
        const shift: u5 = @intCast(16 * (lane & 1));
        const a = (tiled[at] >> shift) & 0xffff;
        const c = (tiled[at + 2] >> shift) & 0xffff;
        o.* = (a & 0xf) | ((a >> 4) & 0xf0) | ((c & 0xf) << 8) | ((c << 4) & 0xf000) | (((a >> 4) & 0xf) << 16) | (((a >> 8) & 0xf0) << 16) | (((c >> 4) & 0xf) << 24) | (((c >> 8) & 0xf0) << 24);
    }
}

pub const Order = struct { w: Ref, n: usize, k: usize };

/// The reg_layout rewrite on the GPU: one library and queue for any number of runs.
pub const RegOrder = struct {
    pipe: mtl.Pipeline,
    queue: mtl.Queue,

    pub fn init(device: mtl.Device) !RegOrder {
        const lib = try mtl.Library.fromSource(device, ks.core_lane_order, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "tf_lane_order", false);
        errdefer pipe.deinit();
        return .{ .pipe = pipe, .queue = try device.queue() };
    }

    pub fn deinit(o: RegOrder) void {
        o.pipe.deinit();
        o.queue.deinit();
    }

    /// Rewrites tile-order 4-bit g64 words in place into reg_layout and waits; once per weight, before any reg encode.
    pub fn run(o: RegOrder, jobs: []const Order) !void {
        for (jobs) |j| {
            if (j.n == 0 or j.n % 32 != 0 or j.k == 0 or j.k % 64 != 0) return error.InvalidShape;
            try checkRef(j.w, j.n * j.k / 2, 4);
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = o.queue.commandBuffer();
        const e = cb.compute(.concurrent);
        e.setPipeline(o.pipe);
        for (jobs) |j| {
            e.setBuffer(j.w.buf, j.w.off, 0);
            e.dispatchGroups(mtl.Size.of(j.n / 32 * (j.k / 64), 1, 1), mtl.Size.of(256, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
    }
};

pub fn regOrder(device: mtl.Device, jobs: []const Order) !void {
    const o = try RegOrder.init(device);
    defer o.deinit();
    try o.run(jobs);
}

fn overlap(a: []const u8, b: []const u8) bool {
    const aa = @intFromPtr(a.ptr);
    const bb = @intFromPtr(b.ptr);
    return if (aa <= bb) bb - aa < a.len else aa - bb < b.len;
}

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };

pub const Projection = struct {
    pipe: mtl.Pipeline,
    n: usize,
    k: usize,
    group_count: usize,
    weight_bytes: usize,
    metadata_bytes: usize,
    output_bytes: usize,
    y_stride: usize,
    tile_count: usize,
    sk: usize,
    sums_pipe: ?mtl.Pipeline = null,
    sums_buffer: ?mtl.Buffer = null,
    threads_per_slice: usize = 32,
    reg: bool = false,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, l: Layout) !Projection {
        try l.validate();
        if (!device.tensorUnits() and (l.format.bits != 6 or l.format.group != 32 or l.cooperative or l.reg or l.precompute_sums)) return error.TensorUnitsRequired;
        const text = try source(a, l);
        defer a.free(text);
        const adapted = try frags.source(device, a, text);
        defer a.free(adapted);
        const lib = try mtl.Library.fromSource(device, adapted, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const p = try mtl.Pipeline.init(device, lib, if (l.reg) "tf_lane_reg" else if (l.cooperative) "tf_lane_pair" else "tf_lane", false);
        errdefer p.deinit();
        if (p.simdWidth() != 32 or p.maxThreads() < l.sk * (if (l.cooperative) @as(usize, 64) else 32)) return error.UnsupportedPipelineShape;
        var sums_pipe: ?mtl.Pipeline = null;
        var sums_buffer: ?mtl.Buffer = null;
        if (l.precompute_sums) {
            sums_pipe = try mtl.Pipeline.init(device, lib, "tf_lane_sums", false);
            errdefer sums_pipe.?.deinit();
            sums_buffer = try device.buffer(@as(usize, if (l.reg) reg_rows else 16) * (l.k / l.format.group) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        }
        return .{ .pipe = p, .n = l.n, .k = l.k, .group_count = l.k / l.format.group, .weight_bytes = l.weightWords() * 4, .metadata_bytes = l.metadataElements() * 2, .output_bytes = if (l.output == .f32) 4 else 2, .y_stride = l.y_stride orelse l.n, .tile_count = l.tiles(), .sk = l.sk, .sums_pipe = sums_pipe, .sums_buffer = sums_buffer, .threads_per_slice = if (l.cooperative) 64 else 32, .reg = l.reg };
    }

    pub fn deinit(p: Projection) void {
        p.pipe.deinit();
        if (p.sums_pipe) |pipe| pipe.deinit();
        if (p.sums_buffer) |buffer| buffer.deinit();
    }

    /// Encodes one window; the caller owns buffers through GPU completion and orders later accesses.
    pub fn encode(p: Projection, e: mtl.ComputeEncoder, x: Ref, w: Ref, sb: Ref, y: Ref, rows: u32) !void {
        return p.encodeImpl(e, x, w, sb, null, y, rows);
    }

    /// The reg kernel on group sums its input's producer wrote (tf_lane_sums' order): no sums dispatch, no barrier.
    pub fn encodeSums(p: Projection, e: mtl.ComputeEncoder, x: Ref, w: Ref, sb: Ref, sums: Sums, y: Ref, rows: u32) !void {
        if (!p.reg) return error.SumsNeedReg;
        if (rows == 0 or rows > reg_rows or sums.stride < rows) return error.InvalidRows;
        try checkRef(sums.ref, ((p.group_count - 1) * @as(usize, sums.stride) + rows) * 4, 4);
        return p.encodeImpl(e, x, w, sb, sums, y, rows);
    }

    fn encodeImpl(p: Projection, e: mtl.ComputeEncoder, x: Ref, w: Ref, sb: Ref, given: ?Sums, y: Ref, rows: u32) !void {
        if (rows == 0 or rows > @as(u32, if (p.reg) reg_rows else 16)) return error.InvalidRows;
        const blocks = (rows + 15) / 16;
        const stride: u32 = if (given) |g| g.stride else 16 * blocks;
        try checkRef(x, rows * p.k * 2, 2);
        try checkRef(w, p.weight_bytes, if (p.reg) 16 else 8);
        try checkRef(sb, p.metadata_bytes, 16);
        try checkRef(y, ((rows - 1) * p.y_stride + p.n) * p.output_bytes, p.output_bytes);
        if (given == null) if (p.sums_pipe) |pipe| {
            e.setPipeline(pipe);
            e.setBuffer(x.buf, x.off, 0);
            e.setValue(@as(i32, @intCast(rows)), 1);
            e.setBuffer(p.sums_buffer.?, 0, 2);
            e.setValue(@as(i32, @intCast(stride)), 3);
            e.dispatchThreads(.{ .width = stride * p.group_count }, .{ .width = 256 });
            e.barrier();
        };
        e.setPipeline(p.pipe);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(w.buf, w.off, 1);
        e.setBuffer(sb.buf, sb.off, 2);
        e.setValue([2]i32{ @intCast(rows), @intCast(stride) }, 3);
        e.setBuffer(y.buf, y.off, 4);
        if (given) |g| e.setBuffer(g.ref.buf, g.ref.off, 5) else if (p.sums_buffer) |buffer| e.setBuffer(buffer, 0, 5);
        e.dispatchGroups(mtl.Size.of(p.tile_count, if (p.reg) blocks else 1, 1), mtl.Size.of(p.threads_per_slice * p.sk, 1, 1));
        if (given == null and p.sums_buffer != null) e.barrier();
    }
};

/// Rows one reg encode takes, as 16-row blocks on the grid's y (a prompt chunk's rows in one weight read).
pub const reg_rows = 128;

/// Group sums of a projection's input: XS[group * stride + row], fp32, each group's values added in order from zero.
pub const Sums = struct { ref: Ref, stride: u32 };

fn checkRef(r: Ref, need: usize, alignment: usize) !void {
    if (r.off % alignment != 0) return error.MisalignedBuffer;
    const len = r.buf.length();
    if (r.off > len or need > len - r.off) return error.BufferTooSmall;
}

test "layouts reject unsupported formats, schedules, cuts and overlapping tile reads" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const good = Layout{ .n = 64, .k = 256, .format = .{ .bits = 4, .group = 64 } };
    const s = try source(arena.allocator(), good);
    try std.testing.expect(std.mem.indexOf(u8, s, "#define TF_BITS 4\n#define TF_GROUP 64\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, s[0 .. s.len - ks.core_lane_projection.len], "TF_YS") == null);
    var bad = good;
    for ([_]u8{ 2, 3, 5, 6, 8 }) |bits| {
        bad.format.bits = bits;
        try bad.validate();
    }
    for ([_]u8{ 0, 1, 7, 9 }) |bits| {
        bad.format.bits = bits;
        try std.testing.expectError(error.UnsupportedBits, bad.validate());
    }
    bad = good;
    bad.y_stride = 96;
    try std.testing.expect(std.mem.indexOf(u8, try source(arena.allocator(), bad), "#define TF_YS 96\n") != null);
    bad.y_stride = 63;
    try std.testing.expectError(error.InvalidOutputStride, bad.validate());
    bad = good;
    bad.n = 48;
    bad.ranges = &.{.{ 1, 1 }};
    try bad.validate();
    try std.testing.expectEqual(2, bad.tileCount());
    bad.cooperative = true;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad.cooperative = false;
    bad.reg = true;
    bad.precompute_sums = true;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad = good;
    bad.ranges = &.{.{ 2, 1 }};
    try std.testing.expectError(error.InvalidTileRange, bad.validate());
    bad = good;
    bad.format.group = 16;
    try std.testing.expectError(error.UnsupportedGroup, bad.validate());
    bad = good;
    bad.n = 33;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad = good;
    bad.k = 96;
    try std.testing.expectError(error.InvalidShape, bad.validate());
    bad = good;
    bad.sk = 0;
    try std.testing.expectError(error.InvalidSchedule, bad.validate());
    bad = good;
    bad.sk = 32;
    try std.testing.expectError(error.ThreadgroupMemoryLimit, bad.validate());
    bad = good;
    bad.pf = 3;
    try std.testing.expectError(error.InvalidSchedule, bad.validate());
    bad = good;
    bad.groups = .{ 3, 2 };
    try std.testing.expectError(error.InvalidGroupCut, bad.validate());
    bad = good;
    bad.groups = .{ 0, 2 };
    try std.testing.expectError(error.PartialOutputNeedsFloat, bad.validate());
    bad = good;
    bad.ranges = &.{ .{ 0, 1 }, .{ 0, 1 } };
    try std.testing.expectError(error.InvalidTileRange, bad.validate());
}

test "every width, group and 16-row member packs MLX codes, metadata and zero padding without changing a bit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var prng: std.Random.DefaultPrng = .init(0x1a9e);
    for ([_]u8{ 2, 3, 4, 5, 6, 8 }) |bits| for ([_]usize{ 32, 64, 128 }) |group| for ([_]usize{ 64, 48 }) |rows| {
        const l = Layout{ .n = rows, .k = 2 * group, .format = .{ .bits = bits, .group = group } };
        const codes = try a.alloc(u8, l.n * l.k);
        const w = try a.alloc(u32, l.sourceWords());
        @memset(w, 0);
        for (codes, 0..) |*v, i| {
            v.* = @intCast(prng.random().intRangeAtMost(u16, 0, (@as(u16, 1) << @intCast(bits)) - 1));
            // MLX's packing written bit by bit: row n is one low-bit-first stream of K codes.
            for (0..bits) |bit| w[(i * bits + bit) / 32] |= @as(u32, (v.* >> @intCast(bit)) & 1) << @intCast((i * bits + bit) % 32);
        }
        const s = try a.alloc(u16, l.n * 2);
        const b = try a.alloc(u16, l.n * 2);
        for (s, b, 0..) |*sv, *bv, i| {
            sv.* = @intCast(i);
            bv.* = @intCast(65535 - i);
        }
        const pw = try a.alloc(u32, l.weightWords());
        @memset(pw, 0xffff_ffff);
        const sb = try a.alloc(u16, l.metadataElements());
        try pack(l, w, s, b, pw, sb);
        const wp = group * bits / 32;
        for (0..l.tileCount() * 32) |n| for (0..l.k) |k| {
            const at = ((((n / 32) * 2 + k / group) * 32 + n % 32) * wp) * 32 + (k % group) * bits;
            var code: u32 = 0;
            for (0..bits) |bit| code |= ((pw[(at + bit) / 32] >> @intCast((at + bit) % 32)) & 1) << @intCast(bit);
            try std.testing.expectEqual(@as(u32, if (n < l.n) codes[n * l.k + k] else 0), code);
        };
        for (0..l.n) |n| for (0..2) |g| {
            try std.testing.expectEqual(s[n * 2 + g], sb[(g * l.n + n) * 2]);
            try std.testing.expectEqual(b[n * 2 + g], sb[(g * l.n + n) * 2 + 1]);
        };
        try std.testing.expectError(error.InvalidStorage, pack(l, w[1..], s, b, pw, sb));
        try std.testing.expectError(error.InvalidStorage, pack(l, w, s, b, pw[1..], sb));
        try std.testing.expectError(error.OverlappingStorage, pack(l, pw[0..w.len], s, b, pw, sb));
    };
}

test "reg order puts every code in the tensor op's right-input register for its lane" {
    var tiled: [256]u32 = undefined;
    for (&tiled, 0..) |*v, i| v.* = @as(u32, @intCast(i)) *% 0x9e3779b9 +% 0x7f4a7c15;
    var out: [256]u32 = undefined;
    regBlock(&tiled, &out);
    for (0..32) |lane| for (0..8) |w| for (0..8) |j| {
        // lane l's register 16 kc + 4 nj + e: K 4(l&1) + 8(l>>3&1) + 16 kc + e, column (l>>1&3) + 4(l>>4&1) + 8 nj
        const q = j & 3;
        const k = 4 * (lane & 1) + 8 * ((lane >> 3) & 1) + 16 * (2 * (w & 1) + (q >> 1)) + 2 * (q & 1) + (j >> 2);
        const col = ((lane >> 1) & 3) + 4 * ((lane >> 4) & 1) + 8 * (w >> 1);
        const want = (tiled[col * 8 + k / 8] >> @intCast(4 * (k % 8))) & 15;
        const got = (out[(w >> 2) * 128 + lane * 4 + (w & 3)] >> @intCast(4 * j)) & 15;
        try std.testing.expectEqual(want, got);
    };
    var bad = Layout{ .n = 64, .k = 256, .format = .{ .bits = 4, .group = 64 }, .reg = true };
    try std.testing.expectError(error.UnsupportedRegFormat, bad.validate());
    bad.precompute_sums = true;
    try bad.validate();
    bad.format.group = 128;
    try std.testing.expectError(error.UnsupportedRegFormat, bad.validate());
}
