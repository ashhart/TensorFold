//! `ling-init-faults DIR [--mtp] [--every N]`: the backend opened on the fake driver, each call and allocation failing.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");

const fake = ling.fake_driver;
const Dbg = std.heap.DebugAllocator(.{ .safety = true }); // leak checks in release builds too

/// Opened (the engine, its sequence, the lane backend), with `keep` a state kept and dropped, and closed again.
fn once(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, mtp: bool, keep: bool) !void {
    const l = try open(gpa, io, ctx, dir, mtp);
    defer l.deinit(l.ctx);
    if (keep) l.target.dropState(try l.target.keepState());
}

fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, mtp: bool) !ling.native.Loaded {
    return ling.native.open(gpa, io, ctx, dir, null, .{ .context = 256, .drafts = mtp });
}

/// What one failed open left: the error, host leaks, the fake driver's live, bad releases, frees with work pending.
const Left = struct { err: ?anyerror, leaked: bool, live: usize, bad: usize, early: usize, stuck: usize, pending: bool };

fn failed(io: std.Io, ctx: *const cuda.Context, dir: []const u8, mtp: bool, keep: bool, call: ?u64, alloc: ?usize) Left {
    fake.reset(call, cuda.abi.error_not_ready);
    var dbg: Dbg = .init;
    var fa = std.testing.FailingAllocator.init(dbg.allocator(), .{ .fail_index = alloc orelse std.math.maxInt(usize) });
    const err: ?anyerror = if (once(fa.allocator(), io, ctx, dir, mtp, keep)) |_| null else |e| e;
    const live = fake.live();
    const bad = fake.badReleases();
    const leaked = dbg.deinit() == .leak;
    const out: Left = .{ .err = err, .leaked = leaked, .live = live, .bad = bad, .early = fake.freedEarly(), .stuck = fake.failedDrains(), .pending = fake.failedPending() };
    _ = fake.sweep(); // what this point left held does not count against the next
    return out;
}

const Sticky = struct { tried: usize = 0, bad: usize = 0, held: usize = 0, pending: usize = 0 };

/// Call k and every one after it failing (the open alone): no free with work pending, leaks only past a failed drain.
fn sticky(io: std.Io, ctx: *const cuda.Context, dir: []const u8, mtp: bool, calls: u64, every: usize) Sticky {
    fake.mode(true, true);
    defer fake.mode(true, false);
    var out: Sticky = .{};
    var k: u64 = 1;
    while (k <= calls) : (k += every) {
        const l = failed(io, ctx, dir, mtp, false, k, null);
        out.tried += 1;
        out.held += @intFromBool(l.live != 0);
        out.pending += @intFromBool(l.pending);
        if (l.err == null or l.leaked or l.bad != 0 or l.early != 0 or (l.live != 0 and l.stuck == 0)) {
            out.bad += 1;
            if (out.bad <= 8) std.debug.print("  driver calls {d}.. failing: {?t}, host leak {}, {d} held ({d} failed drains), {d} bad releases, {d} freed while pending\n", .{ k, l.err, l.leaked, l.live, l.stuck, l.bad, l.early });
        }
    }
    return out;
}

pub fn run(gpu: check.Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var mtp = false;
    var every: usize = 1;
    var i: usize = 1;
    while (i < rest.len) : (i += 1) {
        if (std.mem.eql(u8, rest[i], "--mtp")) mtp = true;
        if (std.mem.eql(u8, rest[i], "--every") and i + 1 < rest.len) {
            every = try std.fmt.parseInt(usize, rest[i + 1], 10);
            i += 1;
        }
    }
    const d = fake.driver();
    const ctx = fake.context(&d);
    fake.mode(true, false); // async copies and launches pending until a drain: no free may come before one, ever
    defer fake.mode(false, false);
    // a clean open and kept state first: their calls and host allocations are the failure points
    fake.reset(null, cuda.abi.success);
    var dbg: Dbg = .init;
    var counter = std.testing.FailingAllocator.init(dbg.allocator(), .{});
    const clean = try open(counter.allocator(), gpu.io, &ctx, dir, mtp);
    const opened = fake.count(); // the open's own calls: close's are not failure points
    clean.target.dropState(try clean.target.keepState());
    const calls = fake.count();
    const allocs = counter.allocations;
    clean.deinit(clean.ctx);
    try check.expect(dbg.deinit() == .ok and fake.live() == 0 and fake.badReleases() == 0 and fake.freedEarly() == 0, "a clean open and close leaves nothing", .{});
    var bad: usize = 0;
    var tried: usize = 0;
    var pending: usize = 0;
    var k: u64 = 1;
    while (k <= calls) : (k += every) {
        const l = failed(gpu.io, &ctx, dir, mtp, true, k, null);
        tried += 1;
        pending += @intFromBool(l.pending);
        if (l.err == null or !fake.fired() or l.leaked or l.live != 0 or l.bad != 0 or l.early != 0) {
            bad += 1;
            if (bad <= 8) std.debug.print("  driver call {d} failing: {?t}, host leak {}, {d} held, {d} bad releases, {d} freed while pending\n", .{ k, l.err, l.leaked, l.live, l.bad, l.early });
        }
    }
    var a: usize = 0;
    while (a < allocs) : (a += every) {
        const l = failed(gpu.io, &ctx, dir, mtp, true, null, a);
        tried += 1;
        if (l.err == null or l.leaked or l.live != 0 or l.bad != 0 or l.early != 0) {
            bad += 1;
            if (bad <= 16) std.debug.print("  host allocation {d} failing: {?t}, host leak {}, {d} held, {d} bad releases\n", .{ a, l.err, l.leaked, l.live, l.bad });
        }
    }
    const st = sticky(gpu.io, &ctx, dir, mtp, opened, every);
    bad += st.bad;
    tried += st.tried;
    fake.reset(null, cuda.abi.success);
    std.debug.print("RESULT ling-init-faults{s}: {d} driver calls ({d} the open's) and {d} host allocations in a clean open and kept state; {d} failure points tried ({d} with a transfer pending), {d} left something behind; failing from a call on: {d} points ({d} with a transfer pending), {d} leaking after a failed drain\n", .{ if (mtp) " --mtp" else "", calls, opened, allocs, tried, pending + st.pending, bad, st.tried, st.pending, st.held });
    try check.expect(bad == 0, "every failing driver call and host allocation in the open is cleaned up, nothing freed with work pending", .{});
    check.pass("ling-init-faults{s}: {d} failure points, each refused with nothing freed before a drain, held, leaked or freed twice", .{ if (mtp) " --mtp" else "", tried });
}
