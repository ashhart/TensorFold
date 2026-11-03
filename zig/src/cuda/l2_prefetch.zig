//! Read-only decode-weight prefetch, sharing GLM's fork/join and exact DeepSeek launch shapes.
const std = @import("std");
const cuda = @import("root.zig");
pub const Site = enum(u2) { n, f, o };
pub const Span = struct { at: u32 = 0, rows: u32 = 0, bytes: u64 = 0 };
pub const Knobs = struct { mb: u32 = 12, ctas: u32 = 2, pace_gbps: u32 = 150 };

pub fn take(sizes: []const u64, budget: u64, out: []u64) void {
    var left = budget;
    for (sizes, out) |s, *n| {
        n.* = if (left < 16) 0 else @min(left, s) / 16 * 16;
        left -= n.*;
    }
}

/// DeepSeek V4.1's launch: 32 KiB pieces, segments over 128 threads a CTA, paced as one piece per `chunk / rate` ns.
const chunk: u32 = 32 << 10;

/// The prefetch's own stream, events, kernels and table; the main stream forks at each site and joins once a window.
pub const Runtime = struct {
    gpa: std.mem.Allocator,
    o: Knobs,
    side: cuda.Stream,
    fork_ev: cuda.Event,
    join_ev: cuda.Event,
    mods: [2]cuda.Module,
    segments: cuda.Function,
    paced: cuda.Function,
    rows: cuda.DeviceBuffer,
    spans: []Span,

    /// Takes ownership of spans only on success. The uploaded table contains read-only weight ranges.
    pub fn initRows(gpa: std.mem.Allocator, d: *const cuda.Driver, o: Knobs, host: []const [2]i64, spans: []Span) !Runtime {
        return initImages(gpa, d, o, host, spans, .{ cuda.kernels.l2pf, cuda.kernels.l2pace });
    }

    pub fn initImages(gpa: std.mem.Allocator, d: *const cuda.Driver, o: Knobs, host: []const [2]i64, spans: []Span, images: [2][]const u8) !Runtime {
        var rows = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(host));
        errdefer rows.free();
        var mods: [2]cuda.Module = .{ try cuda.Module.load(d, images[0]), undefined };
        errdefer mods[0].unload();
        mods[1] = try cuda.Module.load(d, images[1]);
        errdefer mods[1].unload();
        var side = try cuda.Stream.init(d, true);
        errdefer side.deinit();
        var fork_ev = try cuda.Event.init(d, false);
        errdefer fork_ev.deinit();
        var join_ev = try cuda.Event.init(d, false);
        errdefer join_ev.deinit();
        const segments = try mods[0].function("_ZN6tfl2pf15segments_kernelILb1EEEvPKlij");
        const paced = try mods[1].function("_ZN12dsv41_l2pace12paced_kernelEPKlijmm");
        return .{ .gpa = gpa, .o = o, .side = side, .fork_ev = fork_ev, .join_ev = join_ev, .mods = mods, .segments = segments, .paced = paced, .rows = rows, .spans = spans };
    }

    /// Before the weights free: the side stream drains first (Stream.deinit synchronizes).
    pub fn deinit(p: *Runtime) void {
        p.side.deinit();
        p.fork_ev.deinit();
        p.join_ev.deinit();
        p.rows.free();
        for (&p.mods) |*m| m.unload();
        p.gpa.free(p.spans);
        p.* = undefined;
    }

    /// Site `site` of layer `j` read into L2 on the side stream from this point of `main` on.
    pub fn fork(p: *const Runtime, main: cuda.Stream, j: usize, site: Site) !void {
        if (j * 3 >= p.spans.len) return;
        const sp = p.spans[j * 3 + @backingInt(site)];
        if (sp.rows == 0) return;
        try p.fork_ev.record(main);
        try p.side.wait(p.fork_ev);
        var a: cuda.Args = .{};
        a.add(p.rows.ptr + @as(u64, sp.at) * 16);
        a.add(@as(c_int, @intCast(sp.rows)));
        a.add(chunk);
        if (site != .o and p.o.pace_gbps > 0) {
            a.add(@as(u64, @max(1, (chunk + p.o.pace_gbps / 2) / p.o.pace_gbps))); // GB/s is bytes a ns
            a.add(@as(u64, 0));
            return cuda.launch.launch(p.paced, .{ .grid = .{ .x = p.o.ctas }, .block = .{ .x = 32 } }, p.side, &a);
        }
        const pieces = std.math.divCeil(u64, sp.bytes, chunk) catch unreachable;
        const grid: u32 = @intCast(std.math.clamp(std.math.divCeil(u64, pieces, 128) catch unreachable, 1, p.o.ctas));
        try cuda.launch.launch(p.segments, .{ .grid = .{ .x = grid }, .block = .{ .x = 128 } }, p.side, &a);
    }

    /// A window's end: `main` waits for every site forked on it, so nothing is in flight past the window.
    pub fn join(p: *const Runtime, main: cuda.Stream) !void {
        try p.join_ev.record(p.side);
        try main.wait(p.join_ev);
    }
};
