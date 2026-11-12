//! The draft depth rule's cost table measured on Metal: dry verify windows and one DSpark pass on the loaded model.
const std = @import("std");
const host = @import("dsv41");
const target_ = @import("target.zig");
const pass_ = @import("pass.zig");
const pipelines = @import("pipelines.zig");

const calib_lanes = host.calib_lanes;
const calib_env = host.calib_env;
const cm = host.calib_measure;
const Allocator = std.mem.Allocator;

/// Committed rows a timed pass ingests first (a round's keep is a few rows).
const ingest_rows = 3;

/// What a measurement runs on: the served target and pass and the plain numbers of the model.
pub const Subject = struct {
    t: *target_.Target,
    ad: ?*pass_.Pass,
    slots: u32,
    limit: u32,
    vocab: u32,
    shape: host.dspark.Shape,
};

const Rig = struct {
    s: Subject,
    cap: u64 = 0,
    detached: bool = false,

    fn of(ctx: *anyopaque) *Rig {
        return @ptrCast(@alignCast(ctx));
    }

    fn mark(_: *anyopaque) anyerror!u64 {
        var ts: std.c.timespec = undefined;
        _ = std.c.clock_gettime(.MONOTONIC, &ts);
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }

    fn gather(_: *anyopaque, ours: []const i64, all: []i64) anyerror!void {
        @memcpy(all, ours);
    }

    fn pending(_: *anyopaque, _: u32) bool {
        return false; // a Metal prompt leaves no verify tail
    }

    fn drop(ctx: *anyopaque, slot: u32) anyerror!void {
        return of(ctx).s.t.dropWindow(slot);
    }

    /// The prompt cache is out for the run and back after it: a calibration text is neither resumed nor saved.
    fn detach(ctx: *anyopaque) void {
        const r = of(ctx);
        if (r.detached) _ = r.s.t.setCacheCap(r.cap) else r.cap = r.s.t.setCacheCap(0);
        r.detached = !r.detached;
    }

    /// The last rows of the slot's prompt into the drafter's rings, as a round's commit does before its pass.
    fn ingest(ctx: *anyopaque, slot: u32) anyerror!void {
        const r = of(ctx);
        const ad = r.s.ad orelse return;
        const taps = r.s.t.target().taps(slot);
        const n = @min(taps.rows, ingest_rows);
        if (n == 0) return;
        var rows: [ingest_rows]u32 = undefined;
        for (rows[0..n], 0..) |*x, i| x.* = taps.rows - n + @as(u32, @intCast(i));
        try ad.pass().ingest(slot, r.s.t.length(slot) - n, taps, rows[0..n]);
    }
};

/// The table measured on `s`, with the lanes' boot lines; a model it cannot time gets the defaults and a warning.
pub fn run(gpa: Allocator, io: std.Io, dir: []const u8, k: calib_env.Knobs, want: calib_env.Shape, s: Subject) !calib_lanes.Boot {
    var rig: Rig = .{ .s = s };
    const stats = s.t.stats;
    defer {
        s.t.stats = stats;
        s.t.wipe();
        if (s.ad) |p| for (0..s.slots) |i| p.pass().reset(@intCast(i));
    }
    const mode = std.mem.trim(u8, k.mode orelse "", " \t\r\n");
    const dev: calib_lanes.Device = .{
        .ctx = &rig,
        .target = s.t.target(),
        .pass = if (s.ad) |p| p.pass() else null,
        .slots = s.slots,
        .limit = s.limit,
        .rows = @min(cm.max_rows, pipelines.max_rows),
        .world = 1,
        .vocab = s.vocab,
        .shape = s.shape,
        .mark = Rig.mark,
        .gather = Rig.gather,
        .has_pending = Rig.pending,
        .drop_window = Rig.drop,
        .detach = Rig.detach,
        .ingest = if (s.ad != null) Rig.ingest else null,
        .wide = &cm.launch_wide,
        .base = cm.launch_bound,
        .store = std.ascii.eqlIgnoreCase(mode, "zig"),
    };
    return calib_lanes.boot(gpa, io, dev, dir, k, want) catch |e| {
        std.log.warn("dsv41 metal: the draft costs could not be measured ({t}); the default table stands in", .{e});
        return .{ .costs = try host.costs.defaults(gpa, host.costs.max_rows), .path = null };
    };
}
