//! The profile refusals: fast is not planned, fast-exact is not qualified, and an unknown spelling names stable.
const std = @import("std");
const cli = @import("cli.zig");

test "stable profile: fast is not planned, fast-exact is not yet qualified, and an unknown spelling names the served profile" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var u: cli.Usage = .{};
    const refused = [_]struct { name: []const u8, sentence: []const u8 }{
        .{ .name = "fast", .sentence = "argument --profile: fast is not planned: a speed mode that changes output tokens is not shipped, so the arithmetic class has no profile to join" },
        .{ .name = "fast-exact", .sentence = "argument --profile: fast-exact is not yet qualified: it needs solo, concurrent 1/2/4/8 and resumed runs equal to stable, and a measurement where no served cell for its family gets slower" },
        .{ .name = "banana", .sentence = "argument --profile: unknown profile 'banana': stable is the only served profile" },
    };
    for (refused) |row| {
        const spaced = [_][]const u8{ "m", "--profile", row.name };
        const equals = [_][]const u8{ "m", try std.fmt.allocPrint(a, "--profile={s}", .{row.name}) };
        for ([_][]const []const u8{ &spaced, &equals }) |argv| {
            try std.testing.expectError(error.Usage, cli.parse(a, argv, &u));
            try std.testing.expectEqualStrings(row.sentence, u.message);
        }
    }
}
