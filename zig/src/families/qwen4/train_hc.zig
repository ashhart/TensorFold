//! Hyper-connection adjoints preserve the identity stream and use the native pass's rounded node inputs.
const core = @import("core");
pub fn branch(t: anytype, l: usize, scope: []const u8, output: []const u8) !void {
    var p: core.living_rows.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(t.e.c.hidden);
    p.u[2] = @intCast(t.e.c.streams);
    try t.ops().row(.hc_branch, p, &.{ .{ 0, t.buf(.grad) }, .{ 1, try t.tape.indexed(l, output, t.rows) }, .{ 2, try node(t, l, scope, "injection") }, .{ 8, t.buf(.branch) }, .{ 9, t.buf(.injection) } });
}
fn node(t: anytype, layer: ?usize, scope: []const u8, suffix: []const u8) !@TypeOf(t.buf(.dummy)) {
    var name: [192]u8 = undefined;
    return t.tape.get(try @import("node_trace.zig").label(&name, layer, scope, suffix), t.rows);
}
pub fn back(t: anytype, layer: ?usize, scope: []const u8, w: anytype, raw: anytype, add: bool) !void {
    const c = &t.e.c;
    const o = t.ops();
    var p: core.living_rows.Params = .{};
    p.u[0] = @intCast(t.rows);
    p.u[1] = @intCast(c.hidden);
    p.u[2] = @intCast(c.streams);
    try o.row(.hc_mix, p, &.{ .{ 0, try node(t, layer, scope, "up") }, .{ 1, try node(t, layer, scope, "norm") }, .{ 2, t.buf(.mixed) }, .{ 8, t.buf(.norm) }, .{ 9, t.buf(.hc_up) } });
    try o.matrix(w.up, t.buf(.hc_up), t.buf(.hc_low), t.rows, 0, 0, false);
    p.u[1] = @intCast(c.lowrank);
    try o.row(.hc_act, p, &.{ .{ 0, try node(t, layer, scope, "down") }, .{ 1, t.buf(.hc_low) }, .{ 8, t.buf(.hc_down) } });
    try o.matrix(w.down, t.buf(.hc_down), t.buf(.norm), t.rows, 0, 0, true);
    if (w.inject) |inject| {
        p.u[1] = @intCast(c.streams);
        p.u[3] = 1;
        try o.row(.hc_act, p, &.{ .{ 0, try node(t, layer, scope, "raw_inject") }, .{ 1, t.buf(.injection) }, .{ 8, t.buf(.hc_inject) } });
        try o.matrix(inject, t.buf(.hc_inject), t.buf(.norm), t.rows, 0, 0, true);
    }
    p.u[1] = @intCast(c.hidden);
    p.u[3] = @intFromBool(add);
    p.f[0] = c.eps;
    try o.row(.rms, p, &.{ .{ 0, raw }, .{ 1, o.raw(w.norm.buffer) }, .{ 2, t.buf(.norm) }, .{ 8, t.buf(.grad) } });
}
