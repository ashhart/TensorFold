//! The block-FP8 splicer at INT4-AutoRound's DeltaNet in-projection: 16,384 FP8 rows and a 96-row BF16 tail.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const source = @import("cuda_projection_load.zig");
const projection = @import("cuda_projection.zig");
const parts = @import("heads_cut.zig").delta_parts;
const Tensor = core.safetensors.Tensor;

/// The checkpoint's k of 2,560 scaled down to two 128-input blocks.
const k: usize = 256;
/// in_proj_qkv, in_proj_z, in_proj_b and in_proj_a rows, as the checkpoint stores them.
const rows = [_]usize{ 10240, 6144, 48, 48 };
const nh = rows[0] + rows[1];
const nt = rows[2] + rows[3];
const n = nh + nt;

/// 0x38 1.0, 0x3c 1.5, 0x40 2.0, 0x34 0.75, 0xb8 -1.0: e4m3 codes with exact binary64 values.
const code_values = [_]u8{ 0x38, 0x3c, 0x40, 0x34, 0xb8 };

/// One block-FP8 part of the fixture: e4m3 codes [rows, k] and one fp32 scale a 128 x 128 block.
const Part = struct { rows: usize, codes: []u8, inv: []f32 };

fn partOf(a: std.mem.Allocator, count: usize, seed: usize) !Part {
    const codes = try a.alloc(u8, count * k);
    errdefer a.free(codes);
    for (0..count) |r| for (0..k) |c| {
        codes[r * k + c] = code_values[(r + c + seed) % code_values.len];
    };
    const inv = try a.alloc(f32, (count / 128) * (k / 128));
    for (inv, 0..) |*s, i| s.* = @floatFromInt(i + 1 + seed);
    return .{ .rows = count, .codes = codes, .inv = inv };
}

fn freePart(a: std.mem.Allocator, p: *Part) void {
    a.free(p.codes);
    a.free(p.inv);
}

/// The checkpoint's four in-projection parts under one prefix: two block FP8 and two BF16 of 48 rows.
const Mixer = struct {
    block: [2]Part,
    tail: [2][]u8,

    /// The part a name selects, and whether it is the scale tensor beside that part's weight.
    fn slot(name: []const u8) ?struct { i: usize, scale: bool } {
        for (parts, 0..) |p, i| {
            var buf: [64]u8 = undefined;
            const stem = std.fmt.bufPrint(&buf, "m{s}", .{p}) catch return null;
            if (!std.mem.startsWith(u8, name, stem)) continue;
            if (std.mem.eql(u8, name[stem.len..], ".weight")) return .{ .i = i, .scale = false };
            if (std.mem.eql(u8, name[stem.len..], ".weight_scale_inv")) return .{ .i = i, .scale = true };
        }
        return null;
    }

    /// Only the block-FP8 parts carry a scale_inv beside their weight: what selects this loader's arm.
    pub fn has(self: *Mixer, name: []const u8) bool {
        _ = self;
        const s = slot(name) orelse return false;
        return s.scale and s.i < 2;
    }

    /// The checkpoint-reader adapter the shared block-FP8 reader takes (fp8.read -> get).
    pub fn get(self: *Mixer, name: []const u8) !Tensor {
        return self.tensor(name);
    }

    /// A reading borrow's pages would go here; this store keeps its bytes in memory.
    pub fn release(self: *Mixer) void {
        _ = self;
    }

    pub fn tensor(self: *Mixer, name: []const u8) !Tensor {
        const s = slot(name) orelse return error.MissingTensor;
        if (s.i >= 2) {
            if (s.scale) return error.MissingTensor;
            const t = self.tail[s.i - 2];
            return .{ .dtype = .bf16, .rank = 2, .shape = .{ t.len / (k * 2), k, 1, 1, 1 }, .bytes = t };
        }
        const p = self.block[s.i];
        if (s.scale) return .{ .dtype = .f32, .rank = 2, .shape = .{ p.rows / 128, k / 128, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(p.inv) };
        return .{ .dtype = .f8_e4m3, .rank = 2, .shape = .{ p.rows, k, 1, 1, 1 }, .bytes = p.codes };
    }

    pub fn deinit(self: *Mixer, a: std.mem.Allocator) void {
        for (&self.block) |*p| freePart(a, p);
        for (&self.tail) |t| a.free(t);
    }
};

fn mixer(a: std.mem.Allocator) !Mixer {
    var out: Mixer = .{ .block = undefined, .tail = undefined };
    out.block[0] = try partOf(a, rows[0], 0);
    errdefer freePart(a, &out.block[0]);
    out.block[1] = try partOf(a, rows[1], 1);
    errdefer freePart(a, &out.block[1]);
    for (&out.tail, rows[2..]) |*t, count| {
        t.* = try a.alloc(u8, count * k * 2);
        errdefer a.free(t.*);
    }
    for (&out.tail, rows[2..], 0..) |t, count, i| for (0..count * k) |v| {
        std.mem.writeInt(u16, t[v * 2 ..][0..2], @truncate(v * 37 + 3 + i), .little);
    };
    return out;
}

test "block FP8: a 96-row BF16 tail after 16,384 FP8 rows is spliced, not refused" {
    const a = std.testing.allocator;
    var m = try mixer(a);
    defer m.deinit(a);
    try std.testing.expectEqual(@as(usize, 2), source.block8Parts(&m, "m", &parts));
    var head = try source.readBlock8(a, &m, "m", parts[0..2], null);
    defer head.deinit(a);
    try std.testing.expectEqual(nh, head.n);
    try std.testing.expectEqual(k, head.k);
    try std.testing.expectEqualSlices(u8, m.block[0].codes, head.codes[0 .. rows[0] * k]);
    try std.testing.expectEqualSlices(u8, m.block[1].codes, head.codes[rows[0] * k ..]);
    try std.testing.expectEqualSlices(f32, m.block[0].inv, head.sinv[0 .. (rows[0] / 128) * (k / 128)]);
    try std.testing.expectEqualSlices(f32, m.block[1].inv, head.sinv[(rows[0] / 128) * (k / 128) ..]);
    const tail = try source.readTail(a, &m, "m", parts[2..], k, nt);
    defer a.free(tail);
    try std.testing.expectEqual(nt * k * 2, tail.len);
    try std.testing.expectEqualSlices(u8, m.tail[0], tail[0 .. rows[2] * k * 2]);
    try std.testing.expectEqualSlices(u8, m.tail[1], tail[rows[2] * k * 2 ..]);
    // the load's own plane builder at the checkpoint's rows: accepted, and past the head zero codes and scale one
    var planes = try projection.fp8gPlanes(a, head.codes, head.sinv, head.n, k, nt);
    defer planes.deinit(a);
    try std.testing.expectEqual(cuda.fp8.padded(n) * k, planes.codes.len);
    try std.testing.expectEqual(cuda.fp8.padded(n) * (k / cuda.fp8.group), planes.scales.len);
    for (planes.codes[nh * k ..]) |b| try std.testing.expectEqual(@as(u8, 0), b);
    for (planes.scales[(nh / 64) * (k / cuda.fp8.group) * 64 ..]) |s| try std.testing.expectEqual(@as(f32, 1.0), s);
    // no FP8 row moves: the head alone builds the planes those rows held
    var alone = try projection.fp8gPlanes(a, head.codes, head.sinv, head.n, k, 0);
    defer alone.deinit(a);
    try std.testing.expectEqualSlices(u8, alone.codes, planes.codes[0..alone.codes.len]);
    try std.testing.expectEqualSlices(f32, alone.scales, planes.scales[0..alone.scales.len]);
    // the per-part expectation: the packers over the stored head bytes, then the tail's zero rows
    const all = try a.alloc(u8, n * k);
    defer a.free(all);
    @memcpy(all[0 .. nh * k], head.codes);
    @memset(all[nh * k ..], 0);
    const inv = try a.alloc(f32, ((n + 127) / 128) * (k / 128));
    defer a.free(inv);
    @memcpy(inv[0..head.sinv.len], head.sinv);
    @memset(inv[head.sinv.len..], 1.0);
    const want = try a.alloc(u8, cuda.fp8.padded(n) * k);
    defer a.free(want);
    cuda.fp8.packCodes(want, all, n, k);
    try std.testing.expectEqualSlices(u8, want, planes.codes);
    const want_s = try a.alloc(f32, cuda.fp8.padded(n) * (k / cuda.fp8.group));
    defer a.free(want_s);
    cuda.fp8.packScales(want_s, inv, n, k);
    try std.testing.expectEqualSlices(f32, want_s, planes.scales);
    // the lane's weight serves the projection's own rows; the 129th block's 32 rows past them are packing only
    const w = cuda.fp8.weight(0, 0, n, k);
    try std.testing.expectEqual(@as(u32, 16480), w.n);
    try std.testing.expectEqual(@as(u32, 16512), w.npad);
}
