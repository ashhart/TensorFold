//! TF_DSV41_BRANCHES / TF_DSV41_MHC_DEFER: a decode window's attention sublayer and mHC coefs on side streams.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");
const fwd = @import("forward.zig");

const log = std.log.scoped(.dsv41);

pub const Prio = enum { main, side, equal };

pub const Settings = struct {
    on: bool = false,
    rows: i64 = 64,
    prio: Prio = .main,
};

fn get(name: [:0]const u8) ?[]const u8 {
    const v = std.mem.trim(u8, @import("sys").environment.Process.get(.{}, name) orelse return null, " \t");
    return if (v.len == 0) null else v;
}

pub fn settings() !Settings {
    return .{
        .on = try @import("sys").environment.read(bool, "TF_DSV41_BRANCHES", false),
        .rows = try @import("sys").environment.read(i64, "TF_DSV41_BRANCHES_ROWS", 64),
        .prio = @fromBackingInt(@intCast(try @import("sys").environment.read(u8, "TF_DSV41_BRANCHES_PRIO", 0))),
    };
}

/// The side streams and their events (the runner's `br` / `dfr` point here).
pub const Branches = struct {
    side: ?run.Runner.Side = null,
    deferred: ?run.Runner.Side = null,

    pub fn deinit(b: *Branches) void {
        for ([_]*?run.Runner.Side{ &b.side, &b.deferred }) |x| if (x.*) |*s| {
            s.stream.deinit();
            s.fork.deinit();
            s.join.deinit();
            for (&s.marks) |*m| if (m.*) |*ev| ev.deinit();
        };
    }
};

fn sideOf(d: *const cuda.Driver, prio: cuda.Stream.Priority) !run.Runner.Side {
    return .{ .stream = try cuda.Stream.initPriority(d, prio), .fork = try cuda.Event.init(d, false), .join = try cuda.Event.init(d, false) };
}

/// The branches' side with its marks (TF_DSV41_PF_OVERLAP_SITE's per-piece events).
fn sideWithMarks(d: *const cuda.Driver, prio: cuda.Stream.Priority) !run.Runner.Side {
    var s = try sideOf(d, prio);
    for (&s.marks) |*m| m.* = try cuda.Event.init(d, false);
    return s;
}

/// TF_DSV41_MHC_DEFER : on unless unset / 0.
pub fn mhcDefer() !bool {
    return @import("sys").environment.read(bool, "TF_DSV41_MHC_DEFER", false);
}

/// TF_DSV41_MHC_DEFER_AT: `exchange` holds each deferred coef launch to the next exchange; unset / 0 / off: after site.
pub fn deferAt(v: ?[]const u8) !bool {
    return @import("sys").environment.value(bool, "TF_DSV41_MHC_DEFER_AT", v, false);
}

/// TF_DSV41_COEF_LATE (ours, default off; needs MHC_DEFER): block.Options.coef_late.
pub fn coefLate() !bool {
    return @import("sys").environment.read(bool, "TF_DSV41_COEF_LATE", false);
}

/// TF_DSV41_MHC_DEFER_AT_ROWS: hold only in programs of at most this many rows (1-64); unset / 0: every program.
pub fn deferAtRows(v: ?[]const u8) !i64 {
    return @import("sys").environment.value(i64, "TF_DSV41_MHC_DEFER_AT_ROWS", v, 0);
}

/// The forward's side streams released (Forward.deinit through forward_hooks.Branches).
fn closeFn(f: *fwd.Forward) void {
    const b: *Branches = @ptrCast(@alignCast(f.branches.?));
    f.runner.br = null;
    f.runner.dfr = null;
    b.deinit();
    f.gpa.destroy(b);
}

pub const hooks_table: @import("forward_hooks.zig").Branches = .{ .close = closeFn };

/// With TF_DSV41_BRANCHES and / or TF_DSV41_MHC_DEFER on: emitter options, side streams, runner; null when both off.
pub fn install(gpa: std.mem.Allocator, f: *fwd.Forward, rank: u32) !?*Branches {
    const s = try settings();
    const dfr = (try mhcDefer()) and f.opts.r1;
    if (!s.on and !dfr) return null;
    const r = f.runner;
    const b = try gpa.create(Branches);
    errdefer gpa.destroy(b);
    b.* = .{};
    errdefer b.deinit();
    if (s.on) {
        b.side = try sideWithMarks(r.d, if (s.prio == .side) .highest else .default);
        f.opts.branches = true;
        f.opts.branch_rows = s.rows;
        r.br = &b.side.?;
    }
    if (dfr) {
        b.deferred = try sideOf(r.d, .default); // mhc_cuda._side: torch's default priority
        f.opts.mhc_defer = true;
        f.opts.mhc_defer_hold = try deferAt(get("TF_DSV41_MHC_DEFER_AT"));
        f.opts.coef_late = (try coefLate());
        f.opts.mhc_defer_hold_rows = try deferAtRows(get("TF_DSV41_MHC_DEFER_AT_ROWS"));
        r.dfr = &b.deferred.?;
    }
    if (rank == 0) {
        if (s.on and s.prio == .main) if (!quiet) log.warn("branches: TF_DSV41_BRANCHES_PRIO=main raises the main stream, the model's own here: run as equal", .{});
        if (s.on) log.info("branches: the indexer / compressor on a side stream (priority {s}), windows <= {d} rows", .{ if (s.prio == .side) "highest" else "default", s.rows });
        if (dfr) log.info("mhc defer: R1's coefficient launches on their own stream, issued {s}, joined by the next mHC call", .{if (!f.opts.mhc_defer_hold) "after their site" else if (f.opts.mhc_defer_hold_rows > 0) "at the next exchange in windows of at most TF_DSV41_MHC_DEFER_AT_ROWS rows" else "at the next exchange (TF_DSV41_MHC_DEFER_AT)"});
        if (dfr and (try coefLate())) log.info("coef late: each deferred coefficient launch issued just before its sublayer's exchange", .{});
    }
    return b;
}

// -- the plan's ordering on a window program ----------------------------------------------------------------------

/// Roles both streams touch inside one fork region of `cs`, weights left out; also checks the fork / join marks.
pub fn shared(a: std.mem.Allocator, cs: []const calls.Call) ![]const []const u8 {
    var main_t: std.StringHashMapUnmanaged(void) = .empty;
    var side_t: std.StringHashMapUnmanaged(u32) = .empty; // role -> the last side call touching it (1, 2, ...)
    var out: std.StringArrayHashMapUnmanaged(void) = .empty;
    var marks: [run.Runner.max_marks]?u32 = @splat(null);
    var seq: u32 = 0;
    var open = false;
    for (cs) |c| {
        if (c.join) {
            if (!open) return error.JoinWithoutFork;
            open = false;
            side_t.clearRetainingCapacity();
            marks = @splat(null);
        }
        if (c.wait > 0) {
            if (!open or c.side or c.wait > marks.len) return error.WaitWithoutMark;
            const at = marks[c.wait - 1] orelse return error.WaitWithoutMark;
            var done: std.ArrayList([]const u8) = .empty;
            var it = side_t.iterator();
            while (it.next()) |kv| if (kv.value_ptr.* <= at) try done.append(a, kv.key_ptr.*);
            for (done.items) |k| _ = side_t.remove(k);
        }
        if (c.fork) {
            open = true;
            main_t.clearRetainingCapacity();
        }
        if (c.side and !open) return error.SideOutsideFork;
        if (c.side) {
            seq += 1;
            for (c.args) |x| try roles(a, x.arg, &side_t, seq, &main_t, &out);
            if (c.mark > 0) {
                if (c.mark > marks.len) return error.WaitWithoutMark;
                marks[c.mark - 1] = seq;
            }
        } else for (c.args) |x| try roles(a, x.arg, &main_t, {}, &side_t, &out);
    }
    if (open) return error.OpenFork;
    return out.keys();
}

/// Roles the deferred coefficient launches share with main before their join (MHC_DEFER); checks the marks' structure.
pub fn sharedDeferred(a: std.mem.Allocator, cs: []const calls.Call) ![]const []const u8 {
    var main_t: std.StringHashMapUnmanaged(void) = .empty;
    var side_t: std.StringHashMapUnmanaged(void) = .empty;
    var out: std.StringArrayHashMapUnmanaged(void) = .empty;
    var open = false;
    for (cs) |c| {
        if (c.defer_join) {
            if (!open) return error.JoinWithoutFork;
            open = false;
            side_t.clearRetainingCapacity();
        }
        if (c.defer_side) {
            if (open) return error.DeferredTwice; // mhc: one pending coefficient set at a time
            open = true;
            main_t.clearRetainingCapacity();
        }
        const mine = if (c.defer_side) &side_t else &main_t;
        const other = if (c.defer_side) &main_t else &side_t;
        for (c.args) |x| try roles(a, x.arg, mine, {}, other, &out);
    }
    if (open) return error.OpenFork;
    return out.keys();
}

/// `x`'s buffer roles into `mine` (with `val`), and into `out` when `other` holds them.
fn roles(a: std.mem.Allocator, x: calls.Arg, mine: anytype, val: anytype, other: anytype, out: *std.StringArrayHashMapUnmanaged(void)) !void {
    switch (x) {
        .t, .opaque_table => |t| switch (t.role) {
            .buf => |b| {
                try mine.put(a, b, val);
                if (other.contains(b)) try out.put(a, b, {});
            },
            else => {},
        },
        .list => |items| for (items) |y| try roles(a, y, mine, val, other, out),
        else => {},
    }
}
