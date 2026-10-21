const std = @import("std");
const policy = @import("exl3_graph_budget.zig");

pub const Charge = policy.Charge;
pub const Class = policy.Class;
pub const Policy = struct { reservation_bytes: u64, headroom_bytes: u64, auxiliary_bytes: u64 = 0, charge: Charge, prefill_charge: ?Charge = null, chain_charge: ?Charge = null };
const mib: u64 = 1024 * 1024;

pub fn defaultPolicy() Policy {
    return workloadPolicy(8, 48) catch unreachable;
}

pub fn workloadPolicy(streams: u64, layers: u64) !Policy {
    if (streams == 0 or layers == 0) return error.InvalidGraphPolicy;
    const decode = try std.math.mul(u64, try std.math.mul(u64, streams, 2), 32 * mib);
    const prefill = try std.math.mul(u64, layers, mib);
    const chain = try std.math.mul(u64, try std.math.mul(u64, streams, 2), mib);
    const auxiliary = try std.math.add(u64, prefill, chain);
    const retained = try std.math.add(u64, decode, auxiliary);
    return .{ .reservation_bytes = try std.math.add(u64, retained, 192 * mib), .headroom_bytes = 128 * mib, .auxiliary_bytes = try std.math.add(u64, auxiliary, 32 * mib), .charge = .{ .exec = 16 * mib, .template = 16 * mib, .transient = 32 * mib }, .prefill_charge = .{ .exec = mib, .template = 0, .transient = 32 * mib }, .chain_charge = .{ .exec = mib, .template = 0, .transient = 32 * mib } };
}

pub fn policyFromEnv() !Policy {
    const stream_text = std.mem.span(std.c.getenv("TF_FLASHNEXT_EXL3_STREAMS") orelse "1");
    var settings = try workloadPolicy(try std.fmt.parseInt(u64, stream_text, 10), 48);
    settings.reservation_bytes = try envBytes("TF_FLASHNEXT_EXL3_GRAPH_RESERVATION_MIB", settings.reservation_bytes);
    settings.headroom_bytes = try envBytes("TF_FLASHNEXT_EXL3_GRAPH_HEADROOM_MIB", settings.headroom_bytes);
    settings.charge.exec = try envBytes("TF_FLASHNEXT_EXL3_GRAPH_EXEC_MIB", settings.charge.exec);
    settings.charge.template = try envBytes("TF_FLASHNEXT_EXL3_GRAPH_TEMPLATE_MIB", settings.charge.template);
    settings.charge.transient = try envBytes("TF_FLASHNEXT_EXL3_GRAPH_TRANSIENT_MIB", settings.charge.transient);
    return settings;
}

fn envBytes(name: [:0]const u8, fallback: u64) !u64 {
    const value = std.c.getenv(name) orelse return fallback;
    return parseMib(std.mem.span(value));
}

fn parseMib(value: []const u8) !u64 {
    if (value.len == 0) return error.InvalidGraphPolicy;
    for (value) |byte| if (byte < '0' or byte > '9') return error.InvalidGraphPolicy;
    return std.math.mul(u64, try std.fmt.parseInt(u64, value, 10), mib);
}
const Ledger = policy.Budget(256);
pub const Ticket = Ledger.Ticket;
var guard: std.atomic.Value(bool) = .init(false);
var ledger: ?Ledger = null;
var configured: ?Policy = null;
var free_memory_hook: ?*const fn () anyerror!u64 = null;

fn lock() void {
    while (guard.swap(true, .acquire)) std.atomic.spinLoopHint();
}

fn unlock() void {
    guard.store(false, .release);
}

pub fn configure(settings: Policy) !void {
    lock();
    defer unlock();
    try configureLocked(settings);
}

fn configureLocked(settings: Policy) !void {
    if (ledger) |*current| if (current.charged() != 0) return error.GraphBudgetBusy;
    var next = try Ledger.init(settings.reservation_bytes, settings.headroom_bytes);
    if (settings.auxiliary_bytes > next.limit or try settings.charge.peak() > next.limit - settings.auxiliary_bytes) return error.InvalidGraphPolicy;
    _ = try next.plan(settings.charge);
    if (settings.prefill_charge) |charge| _ = try next.plan(charge);
    if (settings.chain_charge) |charge| _ = try next.plan(charge);
    if (ledger) |*current| next.generation = current.generation;
    ledger = next;
    configured = settings;
}

pub fn startupReservation() !u64 {
    lock();
    defer unlock();
    if (configured == null) {
        try configureLocked(try policyFromEnv());
        const settings = configured.?;
        std.log.info("EXL3 graph budget auxiliary logical allowance: {d} MiB (heuristic, only the unmaterialized remainder is protected from TP admission)", .{settings.auxiliary_bytes / mib});
        std.log.info("EXL3 graph budget heuristic allowances (not measured exec bytes): reservation {d} MiB, headroom {d} MiB, exec/template/transient {d}/{d}/{d} MiB", .{ settings.reservation_bytes / mib, settings.headroom_bytes / mib, settings.charge.exec / mib, settings.charge.template / mib, settings.charge.transient / mib });
        std.log.info("EXL3 workload heuristic follows TF_FLASHNEXT_EXL3_STREAMS: two decode variants per stream, 48 prefill layer execs and two MTP chain variants per stream; local exec allowances are heuristic, not measured GPU sizes; local templates are released after instantiation", .{});
    }
    return configured.?.reservation_bytes;
}

pub fn setFreeMemoryHook(hook: ?*const fn () anyerror!u64) void {
    lock();
    defer unlock();
    free_memory_hook = hook;
}

pub fn admit(class: Class, cohort: u64, free_bytes: u64) !Ticket {
    lock();
    defer unlock();
    const available = if (free_memory_hook) |hook| try hook() else free_bytes;
    return admitLocked(class, cohort, available);
}

pub fn admitWithQuery(class: Class, cohort: u64, context: anytype, comptime query: anytype) !Ticket {
    lock();
    defer unlock();
    const available = if (free_memory_hook) |hook| try hook() else try query(context);
    return admitLocked(class, cohort, available);
}

fn admitLocked(class: Class, cohort: u64, available: u64) !Ticket {
    const settings = configured orelse return error.GraphBudgetUnconfigured;
    const current = &ledger.?;
    const charge = switch (class) {
        .decode => settings.charge,
        .prefill => settings.prefill_charge orelse settings.charge,
        .chain => settings.chain_charge orelse settings.charge,
    };
    const peak = try charge.peak();
    if (class == .decode and cohort != 0) {
        const unmaterialized = settings.auxiliary_bytes -| current.localCharged();
        const allowance = current.limit - unmaterialized;
        if (peak > allowance or current.charged() > allowance - peak) return error.GraphBudgetBusy;
    }
    const needed = try std.math.add(u64, try std.math.add(u64, current.pending(), peak), settings.headroom_bytes);
    if (available < needed) return error.GraphMemoryHeadroom;
    return current.reserve(class, cohort, charge, 0);
}

pub fn installed(ticket: Ticket) !void {
    lock();
    defer unlock();
    const current = if (ledger) |*value| value else return error.GraphBudgetUnconfigured;
    try current.installed(ticket);
}

pub fn release(ticket: Ticket) !void {
    lock();
    defer unlock();
    const current = if (ledger) |*value| value else return error.GraphBudgetUnconfigured;
    const entry = try current.getForRelease(ticket);
    if (entry) {
        try current.touch(ticket, 0, true);
        try current.beginRetire(ticket, try current.cohortOf(ticket));
    }
    try current.destroyed(ticket);
}

pub fn charged() u64 {
    lock();
    defer unlock();
    return if (ledger) |*current| current.charged() else 0;
}

test "exl3: runtime admissions charge pending overlap and reject reconfiguration" {
    try configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
    const first = try admit(.decode, 1, 100);
    defer release(first) catch unreachable;
    try std.testing.expectError(error.GraphMemoryHeadroom, admit(.chain, 2, 99));
    try std.testing.expectError(error.GraphBudgetBusy, configure(configured.?));
    try installed(first);
    try std.testing.expectEqual(@as(u64, 30), charged());
    try std.testing.expectError(error.GraphMemoryHeadroom, admit(.prefill, 2, 59));
    const second = try admit(.prefill, 2, 60);
    try std.testing.expectEqual(@as(u64, 70), charged());
    try release(second);
    try std.testing.expectError(error.StaleGraphTicket, release(second));
}

test "exl3: runtime policy defaults are bounded heuristics and MiB overrides parse strictly" {
    const settings = defaultPolicy();
    const bounded = try Ledger.init(settings.reservation_bytes, settings.headroom_bytes);
    _ = try bounded.plan(settings.charge);
    try std.testing.expectEqual(@as(u64, 768 * mib), settings.reservation_bytes);
    try std.testing.expectEqual(@as(u64, 128 * mib), settings.headroom_bytes);
    try std.testing.expectEqual(@as(u64, 7 * mib), try parseMib("7"));
    try std.testing.expectError(error.InvalidGraphPolicy, parseMib(""));
    try std.testing.expectError(error.InvalidGraphPolicy, parseMib("-1"));
    try std.testing.expectError(error.InvalidGraphPolicy, parseMib("1x"));
    try std.testing.expectError(error.Overflow, parseMib("18446744073709551615"));
}

test "exl3: startup reservation configures the environment policy without a separate configure call" {
    try std.testing.expectEqual(@as(u64, 0), charged());
    const previous_settings = configured;
    const previous_ledger = ledger;
    defer {
        configured = previous_settings;
        ledger = previous_ledger;
    }
    configured = null;
    ledger = null;
    const expected = try policyFromEnv();
    try std.testing.expectEqual(expected.reservation_bytes, try startupReservation());
    try std.testing.expectEqual(expected.reservation_bytes, try startupReservation());
}

test "exl3: installed physical allocations are not subtracted twice by the free query" {
    try configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
    const Query = struct {
        fn free(available: u64) !u64 {
            return available;
        }
    };
    const first = try admitWithQuery(.decode, 1, @as(u64, 100), Query.free);
    defer release(first) catch unreachable;
    try std.testing.expectError(error.GraphMemoryHeadroom, admitWithQuery(.chain, 0, @as(u64, 60), Query.free));
    try installed(first);
    const second = try admitWithQuery(.chain, 0, @as(u64, 60), Query.free);
    try release(second);
}

test "exl3: runtime policy reconfiguration cannot reuse a released ticket generation" {
    const settings: Policy = .{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } };
    try configure(settings);
    const old = try admit(.decode, 1, 100);
    try release(old);
    try configure(settings);
    const current = try admit(.decode, 1, 100);
    defer release(current) catch unreachable;
    try std.testing.expectError(error.StaleGraphTicket, release(old));
}

test "exl3: simultaneous runtime admissions cannot spend the same free headroom" {
    try configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
    const Worker = struct {
        fn run(result: *?Ticket) void {
            result.* = admit(.chain, 0, 100) catch null;
        }
    };
    var results: [8]?Ticket = @splat(null);
    var threads: [8]std.Thread = undefined;
    for (&threads, &results) |*thread, *result| thread.* = try std.Thread.spawn(.{}, Worker.run, .{result});
    for (threads) |thread| thread.join();
    var admitted: usize = 0;
    for (results) |result| if (result) |ticket| {
        admitted += 1;
        try release(ticket);
    };
    try std.testing.expectEqual(@as(usize, 2), admitted);
    try std.testing.expectEqual(@as(u64, 0), charged());
}

test "exl3: default TP cohorts leave a complete local admission peak" {
    const settings = defaultPolicy();
    try configure(settings);
    var tickets: [256]?Ticket = @splat(null);
    defer for (tickets) |maybe| if (maybe) |ticket| release(ticket) catch unreachable;
    var count: usize = 0;
    while (count < tickets.len) : (count += 1) {
        const ticket = admit(.decode, count + 1, 1024 * mib) catch |err| {
            try std.testing.expect(err == error.GraphBudgetBusy);
            break;
        };
        tickets[count] = ticket;
        try installed(ticket);
    }
    try std.testing.expectEqual(@as(usize, 16), count);
    for ([_]Class{ .prefill, .chain }) |class| {
        const local = try admit(class, 0, 1024 * mib);
        try release(local);
    }
}

test "exl3: bounded prefill admission retains reusable keys across 48 layer scans" {
    try configure(defaultPolicy());
    var cached: [48]?Ticket = @splat(null);
    defer for (cached) |maybe| if (maybe) |ticket| release(ticket) catch unreachable;
    var replays: usize = 0;
    var misses: usize = 0;
    var captures: usize = 0;
    for (0..3) |_| for (&cached) |*entry| {
        if (entry.* != null) {
            replays += 1;
            continue;
        }
        const ticket = admit(.prefill, 0, 1024 * mib) catch |err| switch (err) {
            error.GraphBudgetBusy, error.GraphEvictionRequired => {
                misses += 1;
                continue;
            },
            else => return err,
        };
        entry.* = ticket;
        try installed(ticket);
        captures += 1;
    };
    try std.testing.expectEqual(@as(usize, 48), captures);
    try std.testing.expectEqual(@as(usize, 96), replays);
    try std.testing.expectEqual(@as(usize, 0), misses);
}

test "exl3: workload heuristic scales with concurrency and shape with checked arithmetic" {
    try std.testing.expectEqual(@as(u64, 768 * mib), (try workloadPolicy(8, 48)).reservation_bytes);
    try std.testing.expectEqual(@as(u64, 504 * mib), (try workloadPolicy(4, 48)).reservation_bytes);
    try std.testing.expectEqual(@as(u64, 792 * mib), (try workloadPolicy(8, 72)).reservation_bytes);
    try std.testing.expectError(error.InvalidGraphPolicy, workloadPolicy(0, 48));
    try std.testing.expectError(error.InvalidGraphPolicy, workloadPolicy(8, 0));
    try std.testing.expectError(error.Overflow, workloadPolicy(std.math.maxInt(u64), 48));
    try std.testing.expectError(error.Overflow, workloadPolicy(8, std.math.maxInt(u64)));
    var invalid = defaultPolicy();
    invalid.prefill_charge = .{ .exec = std.math.maxInt(u64), .template = 0, .transient = 1 };
    try std.testing.expectError(error.Overflow, configure(invalid));
}

test "exl3: saturated TP cache retains all local workload variants without double protection" {
    try configure(defaultPolicy());
    var tickets: [80]?Ticket = @splat(null);
    defer for (tickets) |maybe| if (maybe) |ticket| release(ticket) catch unreachable;
    for (0..16) |index| {
        const ticket = try admit(.decode, index + 1, 1024 * mib);
        tickets[index] = ticket;
        try installed(ticket);
    }
    for (16..80) |index| {
        const ticket = try admit(if (index < 64) .prefill else .chain, 0, 1024 * mib);
        tickets[index] = ticket;
        try installed(ticket);
    }
    try std.testing.expectEqual(@as(u64, 576 * mib), charged());
    try std.testing.expectError(error.GraphBudgetBusy, admit(.decode, 99, 1024 * mib));
    try release(tickets[0].?);
    tickets[0] = null;
    tickets[0] = try admit(.decode, 99, 1024 * mib);
    try installed(tickets[0].?);
    try std.testing.expectEqual(@as(u64, 576 * mib), charged());
    const local = try admit(.chain, 0, 1024 * mib);
    try release(local);
}
