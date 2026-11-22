//! Prefix keep graphs refreshed with each call's Seg and Path values.
const cuda = @import("cuda");

pub const Capture = struct {
    exec: ?cuda.graph.Exec = null,

    pub fn deinit(c: *Capture) void {
        if (c.exec) |*exec| exec.deinit();
        c.exec = null;
    }

    /// The driver copies the new by-value arguments into each captured template before update.
    pub fn run(c: *Capture, e: anytype, sg: anytype, p: anytype) !void {
        try cuda.graph.beginCapture(e.stream, .thread_local);
        launches(e, sg, p, true) catch |err| {
            var broken = cuda.graph.endCapture(e.stream) catch return err;
            broken.deinit();
            return err;
        };
        var captured = try cuda.graph.endCapture(e.stream);
        defer captured.deinit();
        if (c.exec) |exec| if (try exec.update(captured) == .success) {
            try exec.launchOn(e.stream);
            return;
        };
        var fresh = try captured.instantiate();
        errdefer fresh.deinit();
        try fresh.launchOn(e.stream);
        if (c.exec) |*old| old.deinit();
        c.exec = fresh;
    }
};

pub fn launches(e: anytype, sg: anytype, p: anytype, apply: bool) !void {
    const g = e.g;
    if (apply) try e.run(e.keeps.apply, .{ 1, @import("cuda_geometry.zig").dk / 4, g.v_heads * g.deltas() }, .{ 32, 4, 1 }, .{ e.geo, sg, p });
    try e.run(e.keeps.conv, .{ g.deltas(), (g.convBytes() / 2 + 255) / 256, 1 }, .{ 256, 1, 1 }, .{ e.geo, sg, @as(c_int, @intCast(p.rows[@as(usize, @intCast(p.n)) - 1])) });
}
