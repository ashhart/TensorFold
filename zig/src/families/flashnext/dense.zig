//! Flash Next's dense lane projections on fz_lane (kernels/metal/decode/fn_lane.metal): the recorded lane_qmm's sums,
//! so each row keeps its bits at every width, with each simdgroup's next group read during this one. A pipeline a layout.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const replay = @import("replay.zig");
const Run = replay.Run;
const Buf = replay.Buf;
const Lane = replay.Lane;

/// A projection's layout: n outputs over k inputs in sk K slices, its tiles ({first, count}; all when empty), its
/// input-group cut ({first, count}: fp32 partials out), and how many groups ahead a simdgroup reads.
pub const Layout = struct { n: usize, k: usize, sk: usize, ranges: []const [2]usize = &.{}, groups: ?[2]usize = null, pf: usize };

pub fn pipeline(r: *Run, l: Layout) !mtl.Pipeline {
    var h = std.hash.Wyhash.init(0x1a9e);
    for ([_]usize{ l.n, l.k, l.sk, l.pf }) |v| h.update(std.mem.asBytes(&v));
    h.update(std.mem.sliceAsBytes(l.ranges));
    if (l.groups) |g| h.update(std.mem.asBytes(&g));
    const key = h.final();
    if (r.lane_pipes.get(key)) |p| return p;
    const lib = try mtl.Library.fromSource(r.device, try source(r.arena, l), mtl.CompileOptions.mlx());
    const pipe = try mtl.Pipeline.init(r.device, lib, "fz_lane", false);
    try r.lane_pipes.put(r.arena, key, pipe);
    return pipe;
}

/// fz_lane's source for a layout: its constants and tile map, then kernels/metal/decode/fn_lane.metal.
pub fn source(a: std.mem.Allocator, l: Layout) ![]u8 {
    const g = l.groups orelse [2]usize{ 0, l.k / 32 };
    var text: std.ArrayList(u8) = .empty;
    try text.print(a, "#define FZ_N {d}\n#define FZ_K {d}\n#define FZ_SK {d}\n#define FZ_G0 {d}\n#define FZ_GN {d}\n#define FZ_PF {d}\n#define FZ_OUT {s}\n", .{ l.n, l.k, l.sk, g[0], g[1], l.pf, if (l.groups != null) "float" else "bfloat" });
    try text.appendSlice(a, "inline int fz_tile(int j) {\n");
    var at: usize = 0;
    for (l.ranges) |c| {
        try text.print(a, "  if (j < {d}) return {d} + j - {d};\n", .{ at + c[1], c[0], at });
        at += c[1];
    }
    try text.appendSlice(a, if (l.ranges.len == 0) "  return j;\n}\n" else "  return 0;\n}\n");
    try text.appendSlice(a, ks.flashnext_lane);
    return text.items;
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
    const f = try mtl.MappedFile.open(try std.fmt.allocPrintSentinel(r.arena, "{s}", .{s.v.file}, 0));
    var v: [3]usize = undefined;
    for ([_][]const u8{ "constexpr int N = ", "constexpr int K = ", "constexpr int SK = " }, 0..) |name, i| {
        const text = f.bytes[0..f.size];
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

/// A target lane projection on fz_lane at the recorded kernel's shape and K slices (the recorded launch's inputs).
pub fn lane(r: *Run, role: []const u8, x: Buf, w: Lane, mdims: Buf, y: Buf) !void {
    if (r.skip & Run.class(role) != 0) return;
    const s = try recorded(r, role);
    try project(r, .{ .n = s.n, .k = s.k, .sk = s.sk, .pf = r.lane_pf }, x, w, mdims, y);
}
