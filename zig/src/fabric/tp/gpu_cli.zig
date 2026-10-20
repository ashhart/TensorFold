//! Validate the runner's command before opening a driver, bootstrap or fate channel.
const std = @import("std");

pub const usage = "usage: tf-tp-test suite | failfast RANK fatal|kill|transport [N [HOLD_S]] | roce-host | hca | kernel\n";
pub const Mode = enum { fatal, kill, transport };
pub const Failure = struct { victim: u32, mode: Mode, n: u32 = 50, hold: ?u32 = null };
pub const Command = union(enum) { suite, failfast: Failure, @"roce-host", hca, kernel };

pub fn parse(args: []const []const u8) error{InvalidArguments}!Command {
    if (args.len == 0) return error.InvalidArguments;
    const tag = std.meta.stringToEnum(std.meta.Tag(Command), args[0]) orelse return error.InvalidArguments;
    switch (tag) {
        .suite, .@"roce-host", .hca, .kernel => {
            if (args.len != 1) return error.InvalidArguments;
            return switch (tag) {
                .suite => .suite,
                .@"roce-host" => .@"roce-host",
                .hca => .hca,
                .kernel => .kernel,
                else => unreachable,
            };
        },
        .failfast => {
            if (args.len < 3 or args.len > 5) return error.InvalidArguments;
            const mode = std.meta.stringToEnum(Mode, args[2]) orelse return error.InvalidArguments;
            if (mode != .transport and args.len != 3) return error.InvalidArguments;
            return .{ .failfast = .{
                .victim = try number(args[1]),
                .mode = mode,
                .n = if (args.len > 3) try number(args[3]) else 50,
                .hold = if (args.len > 4) try number(args[4]) else null,
            } };
        },
    }
}

fn number(s: []const u8) error{InvalidArguments}!u32 {
    return std.fmt.parseInt(u32, s, 10) catch error.InvalidArguments;
}

// Also executed against the Linux entry point with an invalid environment, before any CUDA work.
pub const invalid = [_][]const []const u8{
    &.{},                                                   &.{"UNKNOWN"},                       &.{ "suite", "extra" },                    &.{ "kernel", "extra" },
    &.{ "hca", "extra" },                                   &.{ "roce-host", "extra" },          &.{"failfast"},                            &.{ "failfast", "0" },
    &.{ "failfast", "0", "typo" },                          &.{ "failfast", "bad", "fatal" },    &.{ "failfast", "-1", "kill" },            &.{ "failfast", "4294967296", "fatal" },
    &.{ "failfast", "0", "fatal", "50" },                   &.{ "failfast", "0", "kill", "50" }, &.{ "failfast", "0", "transport", "bad" }, &.{ "failfast", "0", "transport", "50", "bad" },
    &.{ "failfast", "0", "transport", "50", "0", "extra" },
};

test "invalid commands are refused before resources open" {
    for (invalid) |args| try std.testing.expectError(error.InvalidArguments, parse(args));
}

test "documented commands and failfast defaults" {
    inline for (.{ "suite", "roce-host", "hca", "kernel" }) |name| {
        try std.testing.expectEqualStrings(name, @tagName(try parse(&.{name})));
    }
    inline for (.{ Mode.fatal, Mode.kill, Mode.transport }) |mode| {
        const f = (try parse(&.{ "failfast", "2", @tagName(mode) })).failfast;
        try std.testing.expectEqualDeep(Failure{ .victim = 2, .mode = mode }, f);
    }
    const explicit = (try parse(&.{ "failfast", "63", "transport", "0", "0" })).failfast;
    try std.testing.expectEqualDeep(Failure{ .victim = 63, .mode = .transport, .n = 0, .hold = 0 }, explicit);
    const n = (try parse(&.{ "failfast", "1", "transport", "123" })).failfast;
    try std.testing.expectEqualDeep(Failure{ .victim = 1, .mode = .transport, .n = 123 }, n);
}
