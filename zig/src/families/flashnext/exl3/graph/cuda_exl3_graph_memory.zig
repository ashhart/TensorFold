const std = @import("std");
const policy = @import("exl3_graph_budget.zig");

pub const Charge = policy.Charge;
pub const Class = policy.Class;
pub const Policy = struct { reservation_bytes: u64, headroom_bytes: u64, auxiliary_bytes: u64 = 0, charge: Charge, prefill_charge: ?Charge = null, chain_charge: ?Charge = null, two_chunk_reserved: bool = false };
const mib: u64 = 1024 * 1024;
/// Ledger slack above the retained graphs: a ledger bound, not device memory (graphs measured 48-62 MiB at one stream).
const ledger_slack_bytes: u64 = 192 * mib;

/// What startup keeps free of pools and lanes for the graphs: the ledger limit without its slack.
fn protectedBytes(reservation: u64) u64 {
    return reservation -| ledger_slack_bytes;
}

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
    return .{ .reservation_bytes = try std.math.add(u64, retained, ledger_slack_bytes), .headroom_bytes = 128 * mib, .auxiliary_bytes = try std.math.add(u64, auxiliary, 32 * mib), .charge = .{ .exec = 16 * mib, .template = 16 * mib, .transient = 32 * mib }, .prefill_charge = .{ .exec = mib, .template = 0, .transient = 32 * mib }, .chain_charge = .{ .exec = mib, .template = 0, .transient = 32 * mib } };
}

pub fn policyFromEnv() !Policy {
    const stream_text = std.mem.span(std.c.getenv("TF_FLASHNEXT_EXL3_STREAMS") orelse "1");
    return policyFromEnvForStreams(try std.fmt.parseInt(u64, stream_text, 10));
}

fn policyFromEnvForStreams(streams: u64) !Policy {
    const settings = try workloadPolicy(streams, 48);
    const window = @import("../prefill/exl3_prefill_window.zig");
    const two_chunk = window.enabled(if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK")) |value| std.mem.span(value) else null);
    return withWindowReservation(settings, two_chunk, null);
}

fn withWindowReservation(base: Policy, enabled: bool, limit: ?u64) !Policy {
    var settings = base;
    const charge = settings.prefill_charge orelse settings.charge;
    const extra = if (enabled) try @import("../prefill/exl3_prefill_window.zig").graphBytes(charge.exec, charge.template, charge.transient) else 0;
    const required = try std.math.add(u64, base.reservation_bytes, extra);
    settings.reservation_bytes = limit orelse required;
    settings.two_chunk_reserved = enabled and settings.reservation_bytes >= required;
    if (settings.two_chunk_reserved) settings.auxiliary_bytes = try std.math.add(u64, base.auxiliary_bytes, extra);
    return settings;
}

test "exl3: two-chunk graph reservation includes bindings peaks and declines explicit underfunding" {
    const base = try workloadPolicy(2, 48);
    try std.testing.expectEqualDeep(base, try withWindowReservation(base, false, null));
    const expanded = try withWindowReservation(base, true, null);
    try std.testing.expect(expanded.two_chunk_reserved);
    try std.testing.expectEqual(@as(u64, 288 * mib), expanded.reservation_bytes - base.reservation_bytes);
    try std.testing.expectEqual(@as(u64, 288 * mib), expanded.auxiliary_bytes - base.auxiliary_bytes);
    const short = try withWindowReservation(base, true, expanded.reservation_bytes - 1);
    try std.testing.expect(!short.two_chunk_reserved);
    try std.testing.expectEqual(base.auxiliary_bytes, short.auxiliary_bytes);
    var overflow = base;
    overflow.reservation_bytes = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, withWindowReservation(overflow, true, null));
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
    return startupReservationWithStreams(null);
}

pub fn twoChunkReservationBytes() u64 {
    lock();
    defer unlock();
    return if (configured) |settings| (if (settings.two_chunk_reserved) protectedBytes(settings.reservation_bytes) else 0) else 0;
}

pub fn declineTwoChunkReservation() !u64 {
    lock();
    defer unlock();
    var settings = configured orelse return 0;
    if (!settings.two_chunk_reserved) return protectedBytes(settings.reservation_bytes);
    const charge = settings.prefill_charge orelse settings.charge;
    const extra = try @import("../prefill/exl3_prefill_window.zig").graphBytes(charge.exec, charge.template, charge.transient);
    settings.reservation_bytes = try std.math.sub(u64, settings.reservation_bytes, extra);
    settings.auxiliary_bytes = try std.math.sub(u64, settings.auxiliary_bytes, extra);
    settings.two_chunk_reserved = false;
    try configureLocked(settings);
    return protectedBytes(settings.reservation_bytes);
}

test "exl3: declined two-chunk allocation returns the original graph reservation before capture" {
    const base = try workloadPolicy(2, 48);
    try configure(try withWindowReservation(base, true, null));
    try std.testing.expect(twoChunkReservationBytes() > protectedBytes(base.reservation_bytes));
    try std.testing.expectEqual(protectedBytes(base.reservation_bytes), try declineTwoChunkReservation());
    try std.testing.expectEqual(@as(u64, 0), twoChunkReservationBytes());
    try std.testing.expectEqual(protectedBytes(base.reservation_bytes), try declineTwoChunkReservation());
}

pub fn startupReservationForStreams(streams: u32) !u64 {
    if (streams == 0) return error.InvalidGraphPolicy;
    return startupReservationWithStreams(streams);
}

fn startupReservationWithStreams(streams: ?u32) !u64 {
    lock();
    defer unlock();
    if (configured == null) {
        const proposed = if (streams) |count| try policyFromEnvForStreams(count) else try policyFromEnv();
        try configureLocked(proposed);
        const settings = configured.?;
        std.log.info("EXL3 graph budget auxiliary logical allowance: {d} MiB (heuristic, only the unmaterialized remainder is protected from TP admission)", .{settings.auxiliary_bytes / mib});
        std.log.info("EXL3 graph budget heuristic allowances (not measured exec bytes): reservation {d} MiB, headroom {d} MiB, exec/template/transient {d}/{d}/{d} MiB", .{ settings.reservation_bytes / mib, settings.headroom_bytes / mib, settings.charge.exec / mib, settings.charge.template / mib, settings.charge.transient / mib });
        std.log.info("EXL3 workload heuristic follows requested server concurrency (or TF_FLASHNEXT_EXL3_STREAMS for standalone tests): two decode variants per stream, 48 prefill layer execs and two MTP chain variants per stream; local exec allowances are heuristic, not measured GPU sizes; local templates are released after instantiation", .{});
    }
    return protectedBytes(configured.?.reservation_bytes);
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

test "exl3: startup protects the retained graph allowance without the ledger slack" {
    const one = try workloadPolicy(1, 48);
    try std.testing.expectEqual(@as(u64, 114 * mib), protectedBytes(one.reservation_bytes));
    try std.testing.expectEqual(@as(u64, 0), protectedBytes(100));
}

test "exl3: runtime policy defaults are bounded heuristics" {
    const settings = defaultPolicy();
    const bounded = try Ledger.init(settings.reservation_bytes, settings.headroom_bytes);
    _ = try bounded.plan(settings.charge);
    try std.testing.expectEqual(@as(u64, 768 * mib), settings.reservation_bytes);
    try std.testing.expectEqual(@as(u64, 128 * mib), settings.headroom_bytes);
}

test "exl3: graph reservation scales with the requested concurrent stream count" {
    const one = try workloadPolicy(1, 48);
    const two = try workloadPolicy(2, 48);
    try std.testing.expectEqual(@as(u64, 66 * mib), two.reservation_bytes - one.reservation_bytes);
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
    try std.testing.expectEqual(protectedBytes(expected.reservation_bytes), try startupReservation());
    try std.testing.expectEqual(protectedBytes(expected.reservation_bytes), try startupReservation());
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
