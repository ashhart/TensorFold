//! The golden fixtures are byte-stable and each is taken or refused as listed; TF_STAGE_FIXTURES_OUT rewrites them.
const std = @import("std");
const fixtures = @import("fixtures.zig");
const t = std.testing;

/// The fixtures written into the directory TF_STAGE_FIXTURES_OUT names, when it is set; true if they were.
fn rewrite() !bool {
    const dir = std.c.getenv("TF_STAGE_FIXTURES_OUT") orelse return false;
    inline for (fixtures.all) |f| {
        var buf: [512]u8 = undefined;
        const bytes = try f.build(&buf);
        var name: [4096]u8 = undefined;
        const path = try std.fmt.bufPrintSentinel(&name, "{s}/{s}", .{ std.mem.span(dir), f.name }, 0);
        const fd = std.c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o644));
        if (fd < 0) return error.FixtureUnwritable;
        defer _ = std.c.close(fd);
        if (std.c.write(fd, bytes.ptr, bytes.len) != @as(isize, @intCast(bytes.len))) return error.FixtureUnwritable;
    }
    return true;
}

test "stage fixtures: every message's golden bytes are what the encoders write today" {
    if (try rewrite()) return;
    inline for (fixtures.all) |f| {
        var buf: [512]u8 = undefined;
        const bytes = try f.build(&buf);
        t.expectEqualSlices(u8, @embedFile("fixtures/" ++ f.name), bytes) catch |e| {
            std.debug.print("fixture {s} changed\n", .{f.name});
            return e;
        };
    }
}

test "stage fixtures: each good fixture is taken and each bad one refused by its name" {
    inline for (fixtures.all) |f| {
        const got = fixtures.judge(f.kind, @embedFile("fixtures/" ++ f.name));
        if (f.want) |want| {
            t.expectError(want, got) catch |e| {
                std.debug.print("fixture {s}\n", .{f.name});
                return e;
            };
        } else try got;
    }
}

test "stage fixtures: every name is unique" {
    for (fixtures.all, 0..) |a, i| for (fixtures.all[i + 1 ..]) |b| try t.expect(!std.mem.eql(u8, a.name, b.name));
}
