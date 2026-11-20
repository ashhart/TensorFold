//! The step fixtures are byte-stable and each is taken or refused as listed; TF_STEP_FIXTURES_OUT rewrites them.
const std = @import("std");
const fixtures = @import("fixtures.zig");
const t = std.testing;

/// The fixtures written into the directory TF_STEP_FIXTURES_OUT names, when it is set; true if they were.
fn rewrite() !bool {
    const dir = std.c.getenv("TF_STEP_FIXTURES_OUT") orelse return false;
    inline for (fixtures.all) |f| {
        var buf: [1024]u8 = undefined;
        const bytes = try f.build(&buf);
        var name: [4096]u8 = undefined;
        const path = try std.fmt.bufPrint(&name, "{s}{c}{s}", .{ std.mem.span(dir), std.fs.path.sep, f.name });
        try std.Io.Dir.cwd().writeFile(t.io, .{ .sub_path = path, .data = bytes });
    }
    return true;
}

test "step fixtures: every message's golden bytes are what the writer writes today" {
    if (try rewrite()) return;
    inline for (fixtures.all) |f| {
        var buf: [1024]u8 = undefined;
        const bytes = try f.build(&buf);
        t.expectEqualSlices(u8, @embedFile("fixtures/" ++ f.name), bytes) catch |e| {
            std.debug.print("fixture {s} changed\n", .{f.name});
            return e;
        };
    }
}

test "step fixtures: each good fixture is taken and each bad one refused by its name" {
    inline for (fixtures.all) |f| {
        const got = fixtures.judge(@embedFile("fixtures/" ++ f.name));
        if (f.want) |want| {
            t.expectError(want, got) catch |e| {
                std.debug.print("fixture {s}\n", .{f.name});
                return e;
            };
        } else got catch |e| {
            std.debug.print("fixture {s}\n", .{f.name});
            return e;
        };
    }
}

test "step fixtures: every name is unique" {
    for (fixtures.all, 0..) |a, i| for (fixtures.all[i + 1 ..]) |b| try t.expect(!std.mem.eql(u8, a.name, b.name));
}
