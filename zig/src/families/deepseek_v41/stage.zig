//! A rank's EXL3 bytes read into host images the device buffers copy as is: group parts and ragged routed experts.

const std = @import("std");
const exl3 = @import("core").exl3_layout;
const plan = @import("plan.zig");
const Pack = @import("pack.zig").Pack;
const Io = std.Io;

/// One group's part: trellis int16 [kt, nt, words], suh fp16 [K part], svh fp16 [N part].
pub const GroupImage = struct {
    trellis: []u8,
    suh: []u8,
    svh: []u8,

    pub fn deinit(g: *GroupImage, gpa: std.mem.Allocator) void {
        gpa.free(g.trellis);
        gpa.free(g.suh);
        gpa.free(g.svh);
        g.* = undefined;
    }
};

pub fn group(gpa: std.mem.Allocator, io: Io, pack: *const Pack, g: plan.GroupPlan) !GroupImage {
    var out: GroupImage = .{ .trellis = &.{}, .suh = &.{}, .svh = &.{} };
    errdefer out.deinit(gpa);
    var buf: [256]u8 = undefined;
    out.trellis = try gpa.alloc(u8, @intCast(g.part.trellis.bytes()));
    try pack.read(io, try pack.need(try std.fmt.bufPrint(&buf, "{s}.trellis", .{g.prefix})), g.part.trellis, out.trellis);
    out.suh = try gpa.alloc(u8, @intCast(g.part.suh.bytes()));
    try pack.read(io, try pack.need(try std.fmt.bufPrint(&buf, "{s}.suh", .{g.prefix})), g.part.suh, out.suh);
    out.svh = try gpa.alloc(u8, @intCast(g.part.svh.bytes()));
    try pack.read(io, try pack.need(try std.fmt.bufPrint(&buf, "{s}.svh", .{g.prefix})), g.part.svh, out.svh);
    return out;
}

/// A routed projection: experts' trellises back to back, suh [E, K part] and svh [E, N part] stacked, K2 per expert.
pub const ExpertsImage = struct {
    trellis: []align(16) u8,
    suh: []u8,
    svh: []u8,
    /// the kernels' int32 K2 table
    k2: []i32,
    /// byte offset of each expert's trellis in `trellis` (the int64 address table is the device base plus these)
    offsets: []u64,

    pub fn deinit(e: *ExpertsImage, gpa: std.mem.Allocator) void {
        gpa.free(e.trellis);
        gpa.free(e.suh);
        gpa.free(e.svh);
        gpa.free(e.k2);
        gpa.free(e.offsets);
        e.* = undefined;
    }
};

pub fn experts(gpa: std.mem.Allocator, io: Io, pack: *const Pack, x: *const plan.ExpertsPlan) !ExpertsImage {
    const n = x.layout.count();
    const kpart = 2 * @as(u64, x.layout.k_tiles) * 16;
    const npart = 2 * @as(u64, x.layout.n_tiles) * 16;
    var out: ExpertsImage = .{ .trellis = &.{}, .suh = &.{}, .svh = &.{}, .k2 = &.{}, .offsets = &.{} };
    errdefer out.deinit(gpa);
    out.trellis = try gpa.alignedAlloc(u8, .@"16", @intCast(x.layout.trellisBytes()));
    out.suh = try gpa.alloc(u8, @intCast(n * kpart));
    out.svh = try gpa.alloc(u8, @intCast(n * npart));
    out.k2 = try gpa.alloc(i32, n);
    out.offsets = try gpa.alloc(u64, n);
    var buf: [256]u8 = undefined;
    for (x.parts, x.prefixes, 0..) |p, pre, e| {
        const at = x.layout.byteOffset(e);
        out.offsets[e] = at;
        out.k2[e] = @intCast(p.group.k2());
        const t = out.trellis[@intCast(at)..][0..@intCast(p.trellis.bytes())];
        try pack.read(io, try pack.need(try std.fmt.bufPrint(&buf, "{s}.trellis", .{pre})), p.trellis, t);
        try pack.read(io, try pack.need(try std.fmt.bufPrint(&buf, "{s}.suh", .{pre})), p.suh, out.suh[@intCast(e * kpart)..][0..@intCast(kpart)]);
        try pack.read(io, try pack.need(try std.fmt.bufPrint(&buf, "{s}.svh", .{pre})), p.svh, out.svh[@intCast(e * npart)..][0..@intCast(npart)]);
    }
    return out;
}

const testing = std.testing;
const image = @import("pack.zig").image;

test "staged images: a column part's runs, ragged experts back to back, stacked scales" {
    const a = testing.allocator;
    const io = testing.io;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    // three experts of a gate projection (K 32, N 256): expert 1 at 3 bits, the others at 2
    const words = [_]usize{ 32, 48, 32 };
    var specs: [12]@import("pack.zig").Spec = undefined;
    var nb: [12][48]u8 = undefined;
    const shapes = [3][3]usize{ .{ 2, 16, words[0] }, .{ 2, 16, words[1] }, .{ 2, 16, words[2] } };
    for (0..3) |e| {
        const base = e * 4;
        specs[base] = .{ .name = try std.fmt.bufPrint(&nb[base], "layers.0.ffn.experts.{d}.w1.trellis", .{e}), .dtype = "I16", .shape = &shapes[e] };
        specs[base + 1] = .{ .name = try std.fmt.bufPrint(&nb[base + 1], "layers.0.ffn.experts.{d}.w1.suh", .{e}), .dtype = "F16", .shape = &.{32} };
        specs[base + 2] = .{ .name = try std.fmt.bufPrint(&nb[base + 2], "layers.0.ffn.experts.{d}.w1.svh", .{e}), .dtype = "F16", .shape = &.{256} };
        specs[base + 3] = .{ .name = try std.fmt.bufPrint(&nb[base + 3], "layers.0.ffn.experts.{d}.w1.mul1", .{e}), .dtype = "I32", .shape = &.{} };
    }
    const file = try image(a, &specs);
    defer a.free(file);
    // every byte of expert e's tensors: trellis tile (kt, nt) = 64 e + 16 kt + nt; svh element j's low byte = j
    const hl = std.mem.readInt(u64, file[0..8], .little);
    var pos: usize = 8 + hl;
    for (0..3) |e| {
        const tile = 2 * words[e];
        for (0..2) |kt| for (0..16) |nt| @memset(file[pos + (kt * 16 + nt) * tile ..][0..tile], @intCast(64 * e + 16 * kt + nt));
        pos += 2 * 16 * tile;
        @memset(file[pos..][0..64], @intCast(200 + e));
        pos += 64;
        for (0..256) |j| file[pos + 2 * j] = @intCast(j);
        pos += 512 + 4;
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "x.safetensors", .data = file });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    defer a.free(dir);
    var pack = try Pack.open(a, io, dir);
    defer pack.deinit();
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var parts: [3]exl3.Part = undefined;
    var prefixes: [3][]const u8 = undefined;
    for (0..3) |e| {
        var buf: [128]u8 = undefined;
        prefixes[e] = try std.fmt.bufPrint(&nb[e], "layers.0.ffn.experts.{d}.w1", .{e});
        parts[e] = try (try pack.group(prefixes[e], &buf)).part(.col, 1, 2);
    }
    const x: plan.ExpertsPlan = .{ .proj = .w1, .parts = &parts, .prefixes = &prefixes, .layout = try exl3.Layout.of(arena.allocator(), &parts) };
    var img = try experts(a, io, &pack, &x);
    defer img.deinit(a);
    try testing.expectEqualSlices(i32, &.{ 4, 6, 4 }, img.k2);
    try testing.expectEqualSlices(u64, &.{ 0, 2 * 8 * 64, 2 * 8 * 64 + 2 * 8 * 96 }, img.offsets);
    // rank 1's N tiles are 8..15 of each K-tile row
    for (0..3) |e| for (0..2) |kt| for (0..8) |j| {
        const tile = 2 * words[e];
        try testing.expectEqual(@as(u8, @intCast(64 * e + 16 * kt + 8 + j)), img.trellis[@intCast(img.offsets[e] + (kt * 8 + j) * tile)]);
    };
    // suh whole (a column split keeps K), svh this rank's 128 columns
    try testing.expectEqual(@as(u8, 201), img.suh[64]);
    try testing.expectEqual(@as(u8, 128), img.svh[0]);
    try testing.expectEqual(@as(u8, 128), img.svh[256 + 0]);
    try testing.expectEqual(@as(usize, 3 * 256), img.svh.len);
    var buf: [128]u8 = undefined;
    const gp: plan.GroupPlan = .{ .proj = .wq_a, .prefix = prefixes[1], .split = .row, .full = try pack.group(prefixes[1], &buf), .part = try (try pack.group(prefixes[1], &buf)).part(.whole, 0, 2), .marker = null };
    var gi = try group(a, io, &pack, gp);
    defer gi.deinit(a);
    try testing.expectEqual(@as(u8, 64 + 16 + 15), gi.trellis[(16 + 15) * 96]);
}
