//! `ling-init-faults DIR [--mtp] [--every N]`: the backend opened on the fake driver, each call and allocation failing.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const ling = @import("ling_cuda");

const fake = ling.fake_driver;
const Dbg = std.heap.DebugAllocator(.{ .safety = true }); // leak checks in release builds too

/// Opened (the engine, its sequence, the lane backend) and closed again.
fn once(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, mtp: bool) !void {
    const l = try open(gpa, io, ctx, dir, mtp);
    l.deinit(l.ctx);
}

fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, mtp: bool) !ling.native.Loaded {
    return ling.native.open(gpa, io, ctx, dir, null, .{ .context = 256, .drafts = mtp });
}

/// What one failed open left: the error, host leaks, the fake driver's live and bad releases.
const Left = struct { err: ?anyerror, leaked: bool, live: usize, bad: usize };

fn failed(io: std.Io, ctx: *const cuda.Context, dir: []const u8, mtp: bool, call: ?u64, alloc: ?usize) Left {
    fake.reset(call, cuda.abi.error_not_ready);
    var dbg: Dbg = .init;
    var fa = std.testing.FailingAllocator.init(dbg.allocator(), .{ .fail_index = alloc orelse std.math.maxInt(usize) });
    const err: ?anyerror = if (once(fa.allocator(), io, ctx, dir, mtp)) |_| null else |e| e;
    const live = fake.live();
    const bad = fake.badReleases();
    const leaked = dbg.deinit() == .leak;
    return .{ .err = err, .leaked = leaked, .live = live, .bad = bad };
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
    // a clean open first: its calls and host allocations are the failure points
    fake.reset(null, cuda.abi.success);
    var dbg: Dbg = .init;
    var counter = std.testing.FailingAllocator.init(dbg.allocator(), .{});
    const clean = try open(counter.allocator(), gpu.io, &ctx, dir, mtp);
    const calls = fake.count(); // the open's own calls and allocations: close's are not failure points
    const allocs = counter.allocations;
    clean.deinit(clean.ctx);
    try check.expect(dbg.deinit() == .ok and fake.live() == 0 and fake.badReleases() == 0, "a clean open and close leaves nothing", .{});
    var bad: usize = 0;
    var tried: usize = 0;
    var k: u64 = 1;
    while (k <= calls) : (k += every) {
        const l = failed(gpu.io, &ctx, dir, mtp, k, null);
        tried += 1;
        if (l.err == null or !fake.fired() or l.leaked or l.live != 0 or l.bad != 0) {
            bad += 1;
            if (bad <= 8) std.debug.print("  driver call {d} failing: {?t}, host leak {}, {d} held, {d} bad releases\n", .{ k, l.err, l.leaked, l.live, l.bad });
        }
    }
    var a: usize = 0;
    while (a < allocs) : (a += every) {
        const l = failed(gpu.io, &ctx, dir, mtp, null, a);
        tried += 1;
        if (l.err == null or l.leaked or l.live != 0 or l.bad != 0) {
            bad += 1;
            if (bad <= 16) std.debug.print("  host allocation {d} failing: {?t}, host leak {}, {d} held, {d} bad releases\n", .{ a, l.err, l.leaked, l.live, l.bad });
        }
    }
    fake.reset(null, cuda.abi.success);
    std.debug.print("RESULT ling-init-faults{s}: {d} driver calls and {d} host allocations in a clean open; {d} failure points tried, {d} left something behind\n", .{ if (mtp) " --mtp" else "", calls, allocs, tried, bad });
    try check.expect(bad == 0, "every failing driver call and host allocation in the open is cleaned up", .{});
    check.pass("ling-init-faults{s}: {d} failure points, each open refused with nothing left held, leaked or freed twice", .{ if (mtp) " --mtp" else "", tried });
}
