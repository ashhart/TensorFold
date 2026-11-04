//! Connect GLM's explicit prompt-exchange peers before weights consume the registration budget.
const cuda = @import("cuda");

pub const bytes = 4 << 20;

pub fn warm(d: *const cuda.Driver, g: *const cuda.tp_group.Group, ring: bool) !void {
    if (g.world <= 1) return;
    var buf = try cuda.DeviceBuffer.alloc(d, 2 * bytes * g.world);
    defer buf.free();
    var stream = try cuda.Stream.init(d, false);
    defer stream.deinit();
    try post(g, stream, buf.ptr, ring);
    try stream.synchronize();
}

/// The existing 4 MiB send/receive, restricted to neighbours only when ring exchange was selected.
pub fn post(g: *const cuda.tp_group.Group, stream: cuda.Stream, buffer: u64, ring: bool) !void {
    if (g.world == 0 or g.rank >= g.world) return error.BadWorld;
    if (g.world == 1) return;
    const plan: cuda.relay_plan.Relay = .{ .rank = g.rank, .world = g.world, .n = 0 };
    try g.lib.check(g.lib.api.ncclGroupStart(), "ncclGroupStart");
    const posted: anyerror!void = posting: {
        for (0..g.world) |peer| {
            if (peer == g.rank or (ring and peer != plan.next() and peer != plan.prev())) continue;
            g.send(stream, buffer + peer * bytes, bytes, peer) catch |err| break :posting err;
            g.recv(stream, buffer + (g.world + peer) * bytes, bytes, peer) catch |err| break :posting err;
        }
        break :posting {};
    };
    const ended = g.lib.check(g.lib.api.ncclGroupEnd(), "ncclGroupEnd");
    try posted;
    try ended;
}
