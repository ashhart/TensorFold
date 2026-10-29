//! The opt-in admission flags, parsed beside the serve parser's own flag dispatch.
const std = @import("std");
const cli = @import("cli.zig");
const admission = @import("request_admission.zig");

pub const flags = [_]cli.Flag{
    .{ .name = "--max-inflight", .native = true },
    .{ .name = "--max-inflight-per-key", .native = true },
    .{ .name = "--requests-per-second", .native = true },
    .{ .name = "--request-burst", .native = true },
};

/// True when `name` is an admission flag and `value` an integer in its range; false for any other flag.
pub fn apply(a: std.mem.Allocator, limits: *admission.Limits, name: []const u8, value: []const u8, usage: *cli.Usage) !bool {
    const global = std.mem.eql(u8, name, "--max-inflight");
    const per_key = std.mem.eql(u8, name, "--max-inflight-per-key");
    const rate = std.mem.eql(u8, name, "--requests-per-second");
    const burst = std.mem.eql(u8, name, "--request-burst");
    if (!global and !per_key and !rate and !burst) return false;
    const maximum: u32 = if (global or per_key) admission.max_limit else admission.max_rate;
    const parsed = std.fmt.parseInt(u32, value, 10) catch maximum + 1;
    if (parsed > maximum) {
        usage.message = try std.fmt.allocPrint(a, "argument {s}: expected 0 through {d}: '{s}'", .{ name, maximum, value });
        return error.Usage;
    }
    if (global) limits.max_inflight = parsed else if (per_key) limits.max_inflight_per_key = parsed else if (rate) limits.requests_per_second = parsed else limits.request_burst = parsed;
    return true;
}

test "admission CLI: the flags default off, take bounded integers and are advertised" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    const defaults = try cli.parse(a, &.{"model"}, &usage);
    try std.testing.expectEqual(admission.Limits{}, defaults.admission);
    try std.testing.expect(!defaults.admission.enabled());
    const args = try cli.parse(a, &.{ "model", "--max-inflight=12", "--max-inflight-per-key", "3" }, &usage);
    try std.testing.expectEqual(admission.Limits{ .max_inflight = 12, .max_inflight_per_key = 3 }, args.admission);
    for (flags[0..2]) |flag| {
        for ([_][]const u8{ "0", "1", "128" }) |value| _ = try cli.parse(a, &.{ "model", flag.name, value }, &usage);
        for ([_][]const u8{ "-1", "129", "65536", "4294967296", "nan", "1.5", "" }) |value| {
            try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", flag.name, value }, &usage));
        }
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", flag.name }, &usage));
    }
    for (flags) |flag| {
        var present = false;
        for (cli.flags) |known| if (std.mem.eql(u8, known.name, flag.name)) {
            present = known.native;
        };
        try std.testing.expect(present);
    }
    const rate = try cli.parse(a, &.{ "model", "--requests-per-second=7", "--request-burst", "2" }, &usage);
    try std.testing.expectEqual(@as(u32, 7), rate.admission.requests_per_second);
    try std.testing.expectEqual(@as(u32, 2), rate.admission.request_burst);
    try std.testing.expect(!(try cli.parse(a, &.{ "model", "--request-burst", "5" }, &usage)).admission.enabled());
    for (flags[2..]) |flag| {
        for ([_][]const u8{ "0", "1", "128", "100000" }) |value| _ = try cli.parse(a, &.{ "model", flag.name, value }, &usage);
        for ([_][]const u8{ "-1", "100001", "4294967296", "nan", "1.5", "" }) |value| {
            try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", flag.name, value }, &usage));
        }
    }
}
