//! The command line of `tf-hip-test`: the commands, and the old command names as aliases of them.

const std = @import("std");

pub const Command = enum { runtime, kernels, bench_launches, bench_overhead, info };

/// What an invocation asks for, whether by the new names or an old alias.
pub const Plan = struct {
    command: Command,
    filter: []const u8 = "",
    bench: bool = false,
    oracle: ?[]const u8 = null,
    reps: ?usize = null,
    n: ?usize = null,
};

pub const Bad = error{ BadArguments, UnknownCommand };

fn number(text: []const u8) Bad!usize {
    return std.fmt.parseInt(usize, text, 10) catch error.BadArguments;
}

fn value(rest: []const []const u8, i: *usize) Bad![]const u8 {
    i.* += 1;
    if (i.* >= rest.len) return error.BadArguments;
    return rest[i.*];
}

/// The old commands as filters of the new ones; their positional [reps] [filter] arguments are read.
fn oldAlias(cmd: []const u8, rest: []const []const u8) Bad!?Plan {
    const eql = std.mem.eql;
    const first: []const u8 = if (rest.len > 0) rest[0] else "";
    if (eql(u8, cmd, "info")) return .{ .command = .info };
    for ([_][]const u8{ "smoke", "graph", "cooperative", "library", "image" }) |name| {
        if (eql(u8, cmd, name)) return .{ .command = .runtime, .filter = name };
    }
    if (eql(u8, cmd, "affine")) return .{ .command = .kernels, .filter = "oracle", .oracle = if (rest.len > 0) first else return error.BadArguments };
    if (eql(u8, cmd, "gemm")) {
        if (eql(u8, first, "sweep")) return .{ .command = .kernels, .filter = "affine sweep" };
        if (eql(u8, first, "tiers")) return .{ .command = .kernels, .filter = "rows prefill" };
        if (eql(u8, first, "short")) return .{ .command = .kernels, .filter = "affine short", .bench = true };
        return .{ .command = .kernels, .filter = "affine prefill", .bench = true };
    }
    if (eql(u8, cmd, "gemv")) return .{ .command = .kernels, .filter = "affine decode", .bench = true };
    if (eql(u8, cmd, "exact")) return .{ .command = .kernels, .filter = "rows decode" };
    if (eql(u8, cmd, "gdn")) return .{ .command = .kernels, .filter = if (eql(u8, first, "bench")) "gdn bench" else "gdn", .bench = eql(u8, first, "bench") };
    if (eql(u8, cmd, "decode")) {
        const which: []const u8 = if (rest.len > 1) rest[1] else "";
        const filters = [_][]const u8{ "decode router", "decode tail", "decode group", "decode pair", "decode chain" };
        for (filters) |f| if (which.len > 0 and std.mem.endsWith(u8, f, which)) return .{ .command = .kernels, .filter = f, .bench = true };
        return .{ .command = .kernels, .filter = "decode ", .bench = true };
    }
    if (eql(u8, cmd, "launches")) return .{ .command = .bench_launches, .n = if (rest.len > 0) try number(first) else null, .reps = if (rest.len > 1) try number(rest[1]) else null };
    if (eql(u8, cmd, "overhead")) return .{ .command = .bench_overhead, .n = if (rest.len > 0) try number(first) else null, .reps = if (rest.len > 1) try number(rest[1]) else null };
    return null;
}

pub fn parse(cmd: []const u8, rest: []const []const u8) Bad!Plan {
    if (try oldAlias(cmd, rest)) |plan| {
        std.debug.print("note: `{s}` is an alias of the {t} command, with {s}\n", .{ cmd, plan.command, if (plan.filter.len > 0) plan.filter else "all groups" });
        return plan;
    }
    var plan: Plan = undefined;
    var i: usize = 0;
    if (std.mem.eql(u8, cmd, "runtime")) {
        plan = .{ .command = .runtime };
    } else if (std.mem.eql(u8, cmd, "kernels")) {
        plan = .{ .command = .kernels };
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (rest.len == 0) return error.BadArguments;
        plan = .{ .command = if (std.mem.eql(u8, rest[0], "launches")) .bench_launches else if (std.mem.eql(u8, rest[0], "overhead")) .bench_overhead else return error.BadArguments };
        if (rest.len > 1) plan.n = try number(rest[1]);
        if (rest.len > 2) plan.reps = try number(rest[2]);
        return plan;
    } else return error.UnknownCommand;
    while (i < rest.len) : (i += 1) {
        const a = rest[i];
        if (std.mem.eql(u8, a, "--filter")) {
            plan.filter = try value(rest, &i);
        } else if (std.mem.eql(u8, a, "--bench") and plan.command == .kernels) {
            plan.bench = true;
        } else if (std.mem.eql(u8, a, "--oracle") and plan.command == .kernels) {
            plan.oracle = try value(rest, &i);
        } else if (std.mem.eql(u8, a, "--reps") and plan.command == .kernels) {
            plan.reps = try number(try value(rest, &i));
        } else return error.BadArguments;
    }
    return plan;
}

test "the old names resolve to groups of the new commands" {
    const a = (try oldAlias("exact", &.{})).?;
    try std.testing.expectEqual(Command.kernels, a.command);
    try std.testing.expectEqualStrings("rows decode", a.filter);
    const g = (try oldAlias("gemm", &.{"tiers"})).?;
    try std.testing.expectEqualStrings("rows prefill", g.filter);
    try std.testing.expect(!g.bench);
    const m = (try oldAlias("gemv", &.{"3"})).?;
    try std.testing.expect(m.bench);
    const d = (try oldAlias("decode", &.{ "1", "pair" })).?;
    try std.testing.expectEqualStrings("decode pair", d.filter);
    const f = (try oldAlias("affine", &.{"dir"})).?;
    try std.testing.expectEqualStrings("dir", f.oracle.?);
    try std.testing.expectError(error.BadArguments, oldAlias("affine", &.{}));
    try std.testing.expect((try oldAlias("kernels", &.{})) == null);
}

test "the new commands take their flags" {
    const p = try parse("kernels", &.{ "--filter", "gdn", "--bench", "--oracle", "x", "--reps", "7" });
    try std.testing.expectEqual(Command.kernels, p.command);
    try std.testing.expectEqualStrings("gdn", p.filter);
    try std.testing.expect(p.bench);
    try std.testing.expectEqual(@as(?usize, 7), p.reps);
    const r = try parse("runtime", &.{ "--filter", "smoke" });
    try std.testing.expectEqual(Command.runtime, r.command);
    try std.testing.expectError(error.BadArguments, parse("runtime", &.{"--bench"}));
    const b = try parse("bench", &.{ "overhead", "500" });
    try std.testing.expectEqual(Command.bench_overhead, b.command);
    try std.testing.expectEqual(@as(?usize, 500), b.n);
    try std.testing.expectError(error.UnknownCommand, parse("nothing", &.{}));
}
