//! The draft depth rule's cost table measured on Metal: dry verify windows and one DSpark pass on the loaded model.
const std = @import("std");
const host = @import("dsv41");
const target_ = @import("target.zig");
const pass_ = @import("pass.zig");
const pipelines = @import("pipelines.zig");

const calib_lanes = host.calib_lanes;
const calib_env = host.calib_env;
const cm = host.calib_measure;
const environment = @import("sys").environment;
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
    /// what the measurement did, filled when it ends (null: not wanted)
    report: ?*Report = null,
};

/// The work a measurement did on the served model: prompts and rows run, windows launched, state cleared after.
pub const Report = struct {
    prompts: u64 = 0,
    prompt_rows: u64 = 0,
    windows: u64 = 0,
    wiped: target_.Target.Wiped = .{},
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

/// The settings the measurement reads: a malformed one stops the start by name, as CUDA's `measure` mode does.
const settings = [_][:0]const u8{
    "TF_DSV41_CALIB",         "TF_DSV41_CALIB_SHAPE", "TF_DSV41_CALIB_STAT",      "TF_DSV41_CALIB_REPS",      "TF_DSV41_CALIB_WARM",
    "TF_DSV41_CALIB_RECHECK", "TF_DSV41_CALIB_CYCLE", "TF_DSV41_CALIB_DEEP_REPS", "TF_DSV41_CALIB_BUDGET_MS",
};

/// What a measurement borrows from the served model: taken at `enter`, given back at `leave`.
const Held = struct {
    s: Subject,
    stats: target_.Stats,

    fn enter(s: Subject) Held {
        return .{ .s = s, .stats = s.t.stats };
    }

    fn leave(h: Held) void {
        const now = h.s.t.stats;
        const wiped = h.s.t.wipe();
        if (h.s.report) |r| r.* = .{ .prompts = now.prompts - h.stats.prompts, .prompt_rows = now.prompt_rows - h.stats.prompt_rows, .windows = now.windows - h.stats.windows, .wiped = wiped };
        h.s.t.stats = h.stats;
        if (h.s.ad) |p| for (0..h.s.slots) |i| p.pass().reset(@intCast(i));
    }
};

/// The table measured on `s`; a malformed setting is refused, a model it cannot time gets the defaults and a warning.
pub fn run(gpa: Allocator, io: std.Io, dir: []const u8, k: calib_env.Knobs, want: calib_env.Shape, s: Subject) !calib_lanes.Boot {
    for (settings) |name| try environment.check(name, calib_env.env(name));
    _ = try cm.methodFrom(&calib_env.env, cm.launch_bound);
    var rig: Rig = .{ .s = s };
    const held = Held.enter(s);
    defer held.leave();
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
