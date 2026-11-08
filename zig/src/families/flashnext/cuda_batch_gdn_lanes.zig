//! A shared pass's DeltaNet steps over its lanes in one launch each (fn_gdn_lanes_*; TF_FLASHNEXT_GDN_LANES=0: off).
const std = @import("std");
const cuda = @import("cuda");
const gdn = @import("cuda_gdn.zig");
const parts = @import("cuda_batch_state.zig");
const LayerKeep = parts.LayerKeep;
const Layer = @import("cuda_prompt_load.zig").Layer;

/// A shared pass's lanes for the one-launch DeltaNet steps (fn_lanes::Lane / Lanes in flashnext_batch.cu, by value).
pub const GdnLane = extern struct { proj: u64 = 0, conv: u64 = 0, state: u64 = 0, state_out: u64 = 0, log: u64 = 0, ctl: u64 = 0, keep_proj: u64 = 0, keep_hist: u64 = 0, row0: c_int = 0, rows: c_int = 0 };
pub const gdn_lanes_max = 16;
pub const GdnLanes = extern struct { l: [gdn_lanes_max]GdnLane = @splat(.{}), n: c_int = 0 };

comptime {
    std.debug.assert(@sizeOf(GdnLane) == 72 and @sizeOf(GdnLanes) == 1160);
}

fn cint(v: usize) c_int {
    return @intCast(v);
}

/// TF_FLASHNEXT_GDN_LANES=0: a shared pass's DeltaNet steps a lane at a time (default: one launch a step).
pub fn on() bool {
    // the startup check refused every malformed value already, so a plain default stands here
    return @import("sys").environment.read(bool, "TF_FLASHNEXT_GDN_LANES", true) catch true;
}

/// A shared pass's DeltaNet steps, one launch each; false: the caller loops (off, one lane, many, TP, half the heads).
pub fn once(self: anytype, lw: *const Layer, layer: usize, li: usize) !bool {
    if (!on() or self.lanes.len < 2 or self.lanes.len > gdn_lanes_max or self.m.collective != null or !lw.heads.eql(.full)) return false;
    const s = self.s;
    var t: GdnLanes = .{ .n = @intCast(self.lanes.len) };
    var row0: usize = 0;
    var widest: usize = 0;
    var keeps = false;
    for (self.lanes, t.l[0..self.lanes.len]) |l, *x| {
        if (l.rows >= 64) return false;
        const lk: LayerKeep = if (l.keep) |kp| kp.layers[layer] else .{};
        if (lk.log != 0 and lk.state_out == 0) return error.UnexpectedTensor;
        const proj = s.proj + row0 * gdn.proj_width * 2;
        for ([_]u64{ proj, l.seq.conv[li].ptr, lk.proj, lk.hist }) |p| if (p % 16 != 0) return false;
        x.* = .{ .proj = proj, .conv = l.seq.conv[li].ptr, .state = l.seq.state[li].ptr, .state_out = lk.state_out, .log = lk.log, .ctl = lk.log_ctl, .keep_proj = lk.proj, .keep_hist = lk.hist, .row0 = @intCast(row0), .rows = @intCast(l.rows) };
        keeps = keeps or lk.proj != 0 or lk.hist != 0;
        widest = @max(widest, l.rows);
        row0 += l.rows;
    }
    const n: u32 = @intCast(self.lanes.len);
    if (keeps) {
        var a: cuda.Args = .{};
        a.add(t);
        try cuda.launch.launch(self.k.lanes_keep, .{ .grid = .{ .x = 64, .y = n }, .block = .{ .x = 256 } }, self.stream, &a);
    }
    var f: cuda.Args = .{};
    f.add(lw.cw.ptr);
    f.add(lw.a_log.ptr);
    f.add(lw.dt.ptr);
    f.add(s.gq);
    f.add(s.gk);
    f.add(s.gv);
    f.add(s.ggb);
    f.add(cint(parts.late_rows));
    f.add(t);
    try cuda.launch.launch(self.k.lanes_front, .{ .grid = .{ .x = @intCast(widest), .y = gdn.nk + gdn.nv, .z = n }, .block = .{ .x = 256 } }, self.stream, &f);
    var c: cuda.Args = .{};
    c.add(s.gq);
    c.add(s.gk);
    c.add(s.gv);
    c.add(s.ggb);
    c.add(s.ys);
    c.add(cint(parts.late_rows));
    c.add(cint(parts.late_store));
    c.add(t);
    try cuda.launch.launch(self.k.lanes_chain, .{ .grid = .{ .x = gdn.nv, .y = gdn.dv / 16, .z = n }, .block = .{ .x = 128 } }, self.stream, &c);
    var w: cuda.Args = .{};
    w.add(t);
    try cuda.launch.launch(self.k.lanes_shift, .{ .grid = .{ .x = @intCast((gdn.conv_dim + 255) / 256), .y = n }, .block = .{ .x = 256 } }, self.stream, &w);
    return true;
}

test {
    _ = @import("cuda_batch_gdn_lanes_test.zig");
}
