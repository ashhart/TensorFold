//! `tf-hip-kernel-contract image <group>`: the entry and gate-5 records it runs, and its exit status; no HIP calls.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const families = @import("contract_entry.zig"); // the entry types only: no family code on the host

const Entry = families.Entry;
const Record = core.contract.hip_records.Record;
const Tally = core.contract.gates.Tally;

pub const Error = error{ UnknownGroup, NoEntry, MissingRecord, AmbiguousRecord, EmptySelection, OutOfMemory };

/// An image's entry and the records its gate 5 checks; `records` belongs to the allocator `resolve` was given.
pub const Selection = struct { entry: Entry, records: []Record };

/// `group_name`'s entry with each of its record names matched once (none: the group's records); `bad` names a fault.
pub fn resolve(
    a: std.mem.Allocator,
    entries: []const Entry,
    records: []const Record,
    group_name: []const u8,
    bad: *[]const u8,
) Error!Selection {
    bad.* = group_name;
    const group = std.meta.stringToEnum(hip.kernels.Group, group_name) orelse return error.UnknownGroup;
    const entry = for (entries) |e| {
        if (e.group == group) break e;
    } else return error.NoEntry;
    var out: std.ArrayList(Record) = .empty;
    errdefer out.deinit(a);
    if (entry.records.len == 0) for (records) |r| {
        if (std.mem.eql(u8, r.launch.image, group_name)) try out.append(a, r);
    };
    for (entry.records) |name| {
        bad.* = name;
        var found: ?Record = null;
        for (records) |r| if (std.mem.eql(u8, r.launch.name, name)) {
            if (found != null) return error.AmbiguousRecord;
            found = r;
        };
        try out.append(a, found orelse return error.MissingRecord);
    }
    bad.* = group_name;
    if (out.items.len == 0) return error.EmptySelection;
    return .{ .entry = entry, .records = try out.toOwnedSlice(a) };
}

/// The exit status: 1 when a kernel failed, a control was missed, gate 5 found a problem or no kernel ran, else 0.
pub fn verdict(t: Tally, gate5_problems: usize) u8 {
    return if (t.problems() > 0 or gate5_problems > 0 or t.kernels == 0) 1 else 0;
}

/// Two runs' tallies as one, field by field.
pub fn merge(x: Tally, y: Tally) Tally {
    var t = x;
    inline for (@typeInfo(Tally).@"struct".field_names) |name| @field(t, name) += @field(y, name);
    return t;
}

/// Every entry's selection resolves, so no run checks a family without its gate-5 records; `bad` names the fault.
pub fn validateAll(a: std.mem.Allocator, entries: []const Entry, records: []const Record, bad: *[]const u8) Error!void {
    for (entries) |e| a.free((try resolve(a, entries, records, @tagName(e.group), bad)).records);
}

/// Why a selection is refused, in words.
pub fn explain(err: Error) []const u8 {
    return switch (err) {
        error.OutOfMemory => "out of memory",
        error.UnknownGroup => "no kernel group has that name",
        error.NoEntry => "the group has no entry in contract_families.zig",
        error.MissingRecord => "the entry names a record that is not registered",
        error.AmbiguousRecord => "the entry names a record that is registered more than once",
        error.EmptySelection => "the entry selects no gate-5 record",
    };
}

/// One entry's problems on its own tally: failed kernels, missed controls, and one when it ran no kernel.
pub fn entryProblems(t: Tally) usize {
    return t.problems() + @intFromBool(t.kernels == 0);
}

/// The entries' tallies summed, and their problems each judged on its own entry's tally.
pub const Entries = struct { tally: Tally = .{}, problems: usize = 0 };

/// Every entry through its own gates 1-4 at `profile`, one ENTRY line each (the no-argument run).
pub fn runEntries(a: std.mem.Allocator, io: std.Io, entries: []const Entry, profile: families.Profile) !Entries {
    var out: Entries = .{};
    for (entries) |e| {
        const t = try e.run(a, io, profile);
        std.debug.print("ENTRY {t}: {d} kernels, {d} fail, {d} missed\n", .{ e.group, t.kernels, t.failed, t.missed });
        out.tally = merge(out.tally, t);
        out.problems += entryProblems(t);
    }
    return out;
}

fn noRun(_: std.mem.Allocator, _: std.Io, _: families.Profile) anyerror!Tally {
    return .{};
}

fn cleanRun(_: std.mem.Allocator, _: std.Io, p: families.Profile) anyerror!Tally {
    return if (p == .full) .{ .kernels = 2, .passed = 2, .controls = 1, .caught = 1 } else .{};
}

fn failRun(_: std.mem.Allocator, _: std.Io, p: families.Profile) anyerror!Tally {
    return if (p == .full) .{ .kernels = 3, .passed = 1, .failed = 2, .known = 1 } else .{};
}

fn missRun(_: std.mem.Allocator, _: std.Io, p: families.Profile) anyerror!Tally {
    return if (p == .full) .{ .kernels = 1, .passed = 1, .controls = 1, .missed = 1 } else .{};
}

fn rec(name: []const u8, image: []const u8) Record {
    return .{ .launch = .{ .name = name, .symbol = name, .image = image, .threads = 64 }, .usage = @splat(.{ .vgprs = 16 }) };
}

const fixture = [_]Record{ rec("a", "ops"), rec("b", "ops"), rec("fa", "prefill"), rec("twin", "act"), rec("twin", "act") };

fn pick(entries: []const Entry, group_name: []const u8, bad: *[]const u8) Error!Selection {
    return resolve(std.testing.allocator, entries, &fixture, group_name, bad);
}

test "contract select: an entry's names resolve across images, and no names take its group's records" {
    var bad: []const u8 = "";
    const across = try pick(&.{.{ .group = .ops, .records = &.{ "a", "fa" }, .run = noRun }}, "ops", &bad);
    defer std.testing.allocator.free(across.records);
    try std.testing.expectEqual(hip.kernels.Group.ops, across.entry.group);
    try std.testing.expectEqual(@as(usize, 2), across.records.len);
    try std.testing.expectEqualStrings("ops", across.records[0].launch.image);
    try std.testing.expectEqualStrings("prefill", across.records[1].launch.image);
    const own = try pick(&.{.{ .group = .ops, .run = noRun }}, "ops", &bad);
    defer std.testing.allocator.free(own.records);
    try std.testing.expectEqualStrings("a", own.records[0].launch.name);
    try std.testing.expectEqualStrings("b", own.records[1].launch.name);
    try std.testing.expectEqual(@as(usize, 2), own.records.len);
}

test "contract select: an unknown group, a group without an entry and an empty selection are errors" {
    var bad: []const u8 = "";
    const entries = [_]Entry{ .{ .group = .ops, .run = noRun }, .{ .group = .decode, .run = noRun } };
    try std.testing.expectError(error.UnknownGroup, pick(&entries, "opz", &bad));
    try std.testing.expectEqualStrings("opz", bad);
    try std.testing.expectError(error.NoEntry, pick(&entries, "mamba", &bad));
    try std.testing.expectEqualStrings("mamba", bad);
    try std.testing.expectError(error.EmptySelection, pick(&entries, "decode", &bad));
    try std.testing.expectEqualStrings("decode", bad);
}

test "contract select: every explicit name must match exactly one record" {
    var bad: []const u8 = "";
    const mixed = [_]Entry{.{ .group = .ops, .records = &.{ "a", "misspelt", "fa" }, .run = noRun }};
    try std.testing.expectError(error.MissingRecord, pick(&mixed, "ops", &bad));
    try std.testing.expectEqualStrings("misspelt", bad);
    const twin = [_]Entry{.{ .group = .act, .records = &.{ "a", "twin" }, .run = noRun }};
    try std.testing.expectError(error.AmbiguousRecord, pick(&twin, "act", &bad));
    try std.testing.expectEqualStrings("twin", bad);
}

test "contract select: the verdict fails a failed kernel, a missed control, a gate-5 problem and an empty run" {
    const clean: Tally = .{ .kernels = 2, .passed = 2, .controls = 1, .caught = 1, .known = 1 };
    try std.testing.expectEqual(@as(u8, 0), verdict(clean, 0));
    try std.testing.expectEqual(@as(u8, 1), verdict(.{ .kernels = 1, .failed = 1 }, 0));
    try std.testing.expectEqual(@as(u8, 1), verdict(.{ .kernels = 1, .passed = 1, .controls = 1, .missed = 1 }, 0));
    try std.testing.expectEqual(@as(u8, 1), verdict(clean, 3));
    try std.testing.expectEqual(@as(u8, 1), verdict(.{}, 0));
    try std.testing.expectEqual(@as(u8, 1), verdict(.{ .controls = 1, .caught = 1 }, 0));
    const missed: Tally = .{ .kernels = 1, .passed = 1, .controls = 1, .missed = 1 };
    try std.testing.expectEqual(@as(u8, 0), verdict(merge(clean, clean), 0));
    try std.testing.expectEqual(@as(u8, 1), verdict(merge(clean, missed), 0));
    try std.testing.expectEqual(@as(usize, 3), merge(clean, missed).kernels);
}

test "contract select: the no-argument run counts every entry's kernels and controls" {
    const two = [_]Entry{ .{ .group = .ops, .run = cleanRun }, .{ .group = .act, .run = missRun } };
    const run = try runEntries(std.testing.allocator, std.testing.io, &two, .full);
    const t = run.tally;
    try std.testing.expectEqual(@as(usize, 1), run.problems);
    try std.testing.expectEqual(@as(usize, 3), t.kernels);
    try std.testing.expectEqual(@as(usize, 3), t.passed);
    try std.testing.expectEqual(@as(usize, 2), t.controls);
    try std.testing.expectEqual(@as(usize, 1), t.caught);
    try std.testing.expectEqual(@as(usize, 1), t.missed);
    try std.testing.expectEqual(@as(u8, 1), verdict(t, 0));
    const none = try runEntries(std.testing.allocator, std.testing.io, &.{}, .full);
    try std.testing.expectEqual(Tally{}, none.tally);
    try std.testing.expectEqual(@as(usize, 0), none.problems);
}

test "contract select: each entry is judged on its own tally, so an empty run is a problem beside a clean one" {
    try std.testing.expectEqual(@as(usize, 0), entryProblems(try cleanRun(std.testing.allocator, std.testing.io, .full)));
    try std.testing.expectEqual(@as(usize, 2), entryProblems(try failRun(std.testing.allocator, std.testing.io, .full)));
    try std.testing.expectEqual(@as(usize, 1), entryProblems(try noRun(std.testing.allocator, std.testing.io, .full)));
    const three = [_]Entry{ .{ .group = .ops, .run = cleanRun }, .{ .group = .act, .run = failRun }, .{ .group = .mamba, .run = noRun } };
    const run = try runEntries(std.testing.allocator, std.testing.io, &three, .full);
    try std.testing.expectEqual(@as(usize, 3), run.problems); // 0 + 2 failed + 1 for the empty run
    try std.testing.expectEqual(@as(usize, 5), run.tally.kernels); // the merged tally alone would miss the empty run
    try std.testing.expectEqual(@as(usize, 2), run.tally.problems());
}

test "contract select: the full run validates every entry first; an empty registry is valid" {
    var bad: []const u8 = "";
    try validateAll(std.testing.allocator, &.{}, &fixture, &bad);
    const good = [_]Entry{ .{ .group = .ops, .records = &.{ "a", "fa" }, .run = cleanRun }, .{ .group = .prefill, .run = cleanRun } };
    try validateAll(std.testing.allocator, &good, &fixture, &bad);
    const empty = [_]Entry{ .{ .group = .ops, .run = cleanRun }, .{ .group = .decode, .run = cleanRun } };
    try std.testing.expectError(error.EmptySelection, validateAll(std.testing.allocator, &empty, &fixture, &bad));
    try std.testing.expectEqualStrings("decode", bad);
    const missing = [_]Entry{.{ .group = .ops, .records = &.{"misspelt"}, .run = cleanRun }};
    try std.testing.expectError(error.MissingRecord, validateAll(std.testing.allocator, &missing, &fixture, &bad));
    try std.testing.expectEqualStrings("misspelt", bad);
}
