//! Flash Next's dense lane projections on core/lane_projection: the recorded sums, so rows keep their bits.
const std = @import("std");
const mtl = @import("metal");
const core = @import("../../core/lane_projection.zig");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const Lane = replay.Lane;

/// A projection's layout: n by k in sk K slices, its tiles and input-group cut, and groups read ahead.
pub const Layout = struct { n: usize, k: usize, sk: usize, ranges: []const [2]usize = &.{}, groups: ?[2]usize = null, pf: usize };

pub const kernel_name = "tf_lane";

pub fn coreLayout(l: Layout) core.Layout {
    return .{ .n = l.n, .k = l.k, .sk = l.sk, .pf = l.pf, .format = .{ .bits = 6, .group = 32 }, .ranges = l.ranges, .groups = l.groups, .output = if (l.groups != null) .f32 else .bf16 };
}

pub fn pipeline(r: *Run, l: Layout) !mtl.Pipeline {
    const layout = coreLayout(l);
    try layout.validate();
    var h = std.hash.Wyhash.init(0x1a9e);
    for ([_]usize{ l.n, l.k, l.sk, l.pf }) |v| h.update(std.mem.asBytes(&v));
    h.update(std.mem.sliceAsBytes(l.ranges));
    if (l.groups) |g| h.update(std.mem.asBytes(&g));
    const key = h.final();
    if (r.lane_pipes.get(key)) |p| return p;
    const projection = try core.Projection.init(r.arena, r.device, layout);
    errdefer projection.deinit();
    try r.lane_pipes.put(r.arena, key, projection.pipe);
    return projection.pipe;
}

/// The core source with Flash Next's existing tiled code words, bf16 scale/bias pairs and recorded schedule.
pub fn source(a: std.mem.Allocator, l: Layout) ![]u8 {
    return core.source(a, coreLayout(l));
}

/// The tiles a layout runs.
pub fn tiles(l: Layout) usize {
    if (l.ranges.len == 0) return l.n / 32;
    var n: usize = 0;
    for (l.ranges) |c| n += c[1];
    return n;
}

/// The recorded lane kernel's shape for `role`: its source's N, K and SK.
pub fn recorded(r: *Run, role: []const u8) !struct { n: usize, k: usize, sk: usize } {
    var key: [96]u8 = undefined;
    const s = r.roles.get(try std.fmt.bufPrint(&key, "{s}|1", .{role})) orelse return error.NoSite;
    if (r.lane_shape.get(s.v)) |v| return .{ .n = v[0], .k = v[1], .sk = v[2] };
    const text = try Run.variantText(r.arena, s.v);
    var v: [3]usize = undefined;
    for ([_][]const u8{ "constexpr int N = ", "constexpr int K = ", "constexpr int SK = " }, 0..) |name, i| {
        const at = (std.mem.indexOf(u8, text, name) orelse return error.LanePatch) + name.len;
        const end = at + (std.mem.indexOfScalar(u8, text[at..], ';') orelse return error.LanePatch);
        v[i] = try std.fmt.parseInt(usize, text[at..end], 10);
    }
    try r.lane_shape.put(r.arena, s.v, v);
    return .{ .n = v[0], .k = v[1], .sk = v[2] };
}

/// y = x W over the layout's tiles, for the rows `mdims` holds.
pub fn project(r: *Run, l: Layout, x: Buf, w: Lane, mdims: Buf, y: Buf) !void {
    const pipe = try pipeline(r, l);
    r.enc.setPipeline(pipe);
    for ([_]Buf{ x, w.wq, w.sbt, mdims, y }, 0..) |b, j| r.enc.setBuffer(b.b, b.off, j);
    r.enc.dispatchThreads(mtl.Size.of(tiles(l) * 32 * l.sk, 1, 1), mtl.Size.of(32 * l.sk, 1, 1));
    if (!r.serial) r.enc.barrier();
}

/// A target lane projection on the core kernel at the recorded kernel's shape and K slices.
pub fn lane(r: *Run, role: []const u8, x: Buf, w: Lane, mdims: Buf, y: Buf) !void {
    if (r.skip & Run.class(role) != 0) return;
    const s = try recorded(r, role);
    try project(r, .{ .n = s.n, .k = s.k, .sk = s.sk, .pf = r.lane_pf }, x, w, mdims, y);
}

test "Flash Next adapters preserve the core format, cuts and tile map" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const l = Layout{ .n = 96, .k = 224, .sk = 4, .pf = 2, .ranges = &.{.{ 1, 1 }}, .groups = .{ 1, 3 } };
    const mapped = coreLayout(l);
    try mapped.validate();
    try std.testing.expectEqual(@as(u8, 6), mapped.format.bits);
    try std.testing.expectEqual(@as(usize, 32), mapped.format.group);
    try std.testing.expectEqual(core.Output.f32, mapped.output);
    try std.testing.expectEqualSlices([2]usize, l.ranges, mapped.ranges);
    try std.testing.expectEqual(l.groups, mapped.groups);
    const adapted = try source(arena.allocator(), l);
    const direct = try core.source(arena.allocator(), mapped);
    try std.testing.expectEqualSlices(u8, direct, adapted);
    try std.testing.expectEqual(@as(usize, 1), tiles(l));
    const full = coreLayout(.{ .n = 64, .k = 224, .sk = 2, .pf = 1 });
    try std.testing.expectEqual(core.Output.bf16, full.output);
}
