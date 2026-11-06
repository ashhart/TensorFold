//! Named environment refusals, without coercing a missing setting or changing a caller's default.
const std = @import("std");
const sys = @import("sys.zig");
const Error = error{BadEnvironmentValue};

pub const Problem = struct {
    name: []const u8,
    value: []const u8,
    accepted: []const u8,

    pub fn message(p: Problem, buf: []u8) ![]const u8 {
        return std.fmt.bufPrint(buf, "{s}={s}: expected {s}", .{ p.name, p.value, p.accepted });
    }
};

pub const Flag = struct {
    trim: []const u8 = "",
    words: bool = false,
    no: bool = false,
    empty_off: bool = false,
};

pub const Parser = struct {
    problem: ?Problem = null,

    pub fn reject(p: *Parser, name: []const u8, value: []const u8, accepted: []const u8) Error {
        p.problem = .{ .name = name, .value = value, .accepted = accepted };
        return error.BadEnvironmentValue;
    }

    pub fn flag(p: *Parser, name: []const u8, raw: ?[]const u8, fallback: bool, mode: Flag) Error!bool {
        const value = raw orelse return fallback;
        const v = std.mem.trim(u8, value, mode.trim);
        if (std.mem.eql(u8, v, "0") or (mode.empty_off and v.len == 0)) return false;
        if (std.mem.eql(u8, v, "1")) return true;
        if (mode.words) {
            if (std.ascii.eqlIgnoreCase(v, "true") or std.ascii.eqlIgnoreCase(v, "on")) return true;
            if (std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "off")) return false;
        }
        if (mode.no) {
            if (std.ascii.eqlIgnoreCase(v, "yes")) return true;
            if (std.ascii.eqlIgnoreCase(v, "no")) return false;
        }
        return p.reject(name, value, if (mode.empty_off) (if (mode.no) "empty, 0, 1, true, false, on, off, yes or no" else if (mode.words) "empty, 0, 1, true, false, on or off" else "empty, 0 or 1") else if (mode.no) "0, 1, true, false, on, off, yes or no" else if (mode.words) "0, 1, true, false, on or off" else "0 or 1");
    }

    pub fn real(p: *Parser, name: []const u8, raw: ?[]const u8, fallback: f64, min: f64, max: f64, accepted: []const u8) Error!f64 {
        const value = raw orelse return fallback;
        const n = std.fmt.parseFloat(f64, value) catch return p.reject(name, value, accepted);
        if (!std.math.isFinite(n) or n < min or n > max) return p.reject(name, value, accepted);
        return n;
    }

    pub fn integer(p: *Parser, comptime T: type, name: []const u8, raw: ?[]const u8, fallback: T, min: T, max: T, accepted: []const u8) Error!T {
        const value = raw orelse return fallback;
        const n = std.fmt.parseInt(T, value, 10) catch return p.reject(name, value, accepted);
        if (n < min or n > max) return p.reject(name, value, accepted);
        return n;
    }

    pub fn log(p: Parser) void {
        if (@import("builtin").is_test) return;
        if (p.problem) |v| std.log.err("{s}={s}: expected {s}", .{ v.name, v.value, v.accepted });
    }
};

pub fn get(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(sys.c.getenv(name) orelse return null);
}

pub fn flag(name: []const u8, raw: ?[]const u8, fallback: bool, mode: Flag) Error!bool {
    var p: Parser = .{};
    return p.flag(name, raw, fallback, mode) catch |e| {
        p.log();
        return e;
    };
}

pub fn boolean(name: [:0]const u8, fallback: bool, mode: Flag) Error!bool {
    return flag(name, get(name), fallback, mode);
}

pub fn real(name: [:0]const u8, fallback: f64, min: f64, max: f64, accepted: []const u8) Error!f64 {
    var p: Parser = .{};
    return p.real(name, get(name), fallback, min, max, accepted) catch |e| {
        p.log();
        return e;
    };
}

pub fn integer(comptime T: type, name: [:0]const u8, fallback: T, min: T, max: T, accepted: []const u8) Error!T {
    var p: Parser = .{};
    return p.integer(T, name, get(name), fallback, min, max, accepted) catch |e| {
        p.log();
        return e;
    };
}

test "named environment failures retain the variable, exact value and accepted forms" {
    var p: Parser = .{};
    try std.testing.expectError(error.BadEnvironmentValue, p.flag("NAMED_FLAG", "bad value", false, .{}));
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("NAMED_FLAG=bad value: expected 0 or 1", try p.problem.?.message(&buf));
    try std.testing.expectEqual(@as(f64, 4), try p.real("FLOOR", null, 4, 0, 100, "finite 0..100"));
    for ([_][]const u8{ "nan", "inf", "-1", "101", "", "wat" }) |v| {
        try std.testing.expectError(error.BadEnvironmentValue, p.real("FLOOR", v, 4, 0, 100, "finite 0..100"));
        try std.testing.expectEqualStrings("FLOOR", p.problem.?.name);
        try std.testing.expectEqualStrings(v, p.problem.?.value);
    }
}
