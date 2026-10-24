//! Final live-fetch counters, separate from the original request's prompt accounting.
const std = @import("std");
const Injection = @import("notes_decode.zig").Injection;

pub const Totals = struct {
    fetches: usize = 0,
    extra_prefill: usize = 0,
    unknown: usize = 0,
};

/// Sum every continuation, not only the final segment; absent cache telemetry is explicitly unknown.
pub fn of(injections: []const Injection) Totals {
    var total: Totals = .{ .fetches = injections.len };
    for (injections) |inj| {
        if (inj.cached) |cached| {
            total.extra_prefill += inj.prompt -| @as(usize, cached);
        } else total.unknown += 1;
    }
    return total;
}

/// Ordinary requests retain their exact existing done-line format.
pub fn suffix(buffer: []u8, injections: ?[]const Injection) []const u8 {
    const items = injections orelse return "";
    const total = of(items);
    return std.fmt.bufPrint(buffer, " fetches={d} extra_prefill={d} extra_prefill_unknown={d}", .{
        total.fetches, total.extra_prefill, total.unknown,
    }) catch unreachable;
}

test "live totals include both fetches and never invent missing prefill telemetry" {
    const testing = std.testing;
    const base: Injection = .{
        .ids = &.{1},
        .shares = &.{1},
        .at = 7,
        .place = .user,
        .block_tokens = 20,
        .lookup_ns = 0,
        .prompt = 100,
    };
    var injections = [_]Injection{ base, base };
    injections[0].cached = 40;
    injections[1].prompt = 150;
    injections[1].cached = 50;
    try testing.expectEqual(Totals{ .fetches = 2, .extra_prefill = 160 }, of(&injections));
    try testing.expect(of(&injections).extra_prefill != 100);
    injections[1].cached = null;
    try testing.expectEqual(Totals{ .fetches = 2, .extra_prefill = 60, .unknown = 1 }, of(&injections));
    try testing.expectEqual(Totals{}, of(&.{}));
    var buffer: [128]u8 = undefined;
    try testing.expectEqualStrings("", suffix(&buffer, null));
    try testing.expectEqualStrings(" fetches=0 extra_prefill=0 extra_prefill_unknown=0", suffix(&buffer, &.{}));
}
