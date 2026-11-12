//! The lane host's prefill variables (lane_host_fill.zig): a value fromEnv would misread, refused at start.
const std = @import("std");
const Share = @import("lane_host_fill.zig").Share;

/// The first TENSORFOLD_PREFILL_* value fromEnv would misread, as "NAME=value: expected ..."; empty is unset.
pub fn problem(buf: []u8) ?[]const u8 {
    return problemIn(buf, getenv);
}

fn getenv(name: [:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// `problem` over `get` for the environment, each check its reader's parse (SHARE untrimmed, as fromText reads it).
pub fn problemIn(buf: []u8, get: *const fn ([:0]const u8) ?[]const u8) ?[]const u8 {
    if (get("TENSORFOLD_PREFILL_SHARE")) |t| if (t.len > 0) {
        const p = std.fmt.parseFloat(f64, t) catch std.math.nan(f64);
        if (!(p >= Share.min_share and p <= 1)) return refuse(buf, "TENSORFOLD_PREFILL_SHARE", t, "a number from 0.01 to 1");
    };
    if (get("TENSORFOLD_PREFILL_POLICY")) |t| {
        const s = std.mem.trim(u8, t, " \t");
        if (s.len > 0 and Share.policyOf(s) != .fcfs) return refuse(buf, "TENSORFOLD_PREFILL_POLICY", t, "fcfs, or unset for the default order");
    }
    inline for (.{ "TENSORFOLD_PREFILL_TOKENS", "TENSORFOLD_PREFILL_BUDGET" }) |name| if (get(name)) |t| {
        const s = std.mem.trim(u8, t, " \t");
        // budget's own reading: a leading digit, then a whole u32
        if (s.len > 0 and (s[0] < '0' or s[0] > '9' or (if (std.fmt.parseInt(u32, s, 10)) |_| false else |_| true)))
            return refuse(buf, name, t, "a whole number of rows");
    };
    return null;
}

fn refuse(buf: []u8, name: []const u8, value: []const u8, want: []const u8) []const u8 {
    return std.fmt.bufPrint(buf, "{s}={s}: expected {s}", .{ name, value, want }) catch name;
}

var test_vars: []const [2][]const u8 = &.{};

fn testGet(name: [:0]const u8) ?[]const u8 {
    for (test_vars) |kv| if (std.mem.eql(u8, kv[0], name)) return kv[1];
    return null;
}

fn check(vars: []const [2][]const u8) ?[]const u8 {
    test_vars = vars;
    const Buf = struct {
        var b: [160]u8 = undefined;
    };
    return problemIn(&Buf.b, testGet);
}

test "prefill variables: the values the readers take pass, as do unset and empty ones" {
    try std.testing.expectEqual(@as(?[]const u8, null), check(&.{}));
    for ([_][2][]const u8{
        .{ "TENSORFOLD_PREFILL_SHARE", "0.5" },   .{ "TENSORFOLD_PREFILL_SHARE", "0.01" },  .{ "TENSORFOLD_PREFILL_SHARE", "1" },
        .{ "TENSORFOLD_PREFILL_SHARE", "" },      .{ "TENSORFOLD_PREFILL_POLICY", "fcfs" }, .{ "TENSORFOLD_PREFILL_POLICY", " fcfs\t" },
        .{ "TENSORFOLD_PREFILL_POLICY", "" },     .{ "TENSORFOLD_PREFILL_TOKENS", "2048" }, .{ "TENSORFOLD_PREFILL_TOKENS", "99999" },
        .{ "TENSORFOLD_PREFILL_TOKENS", " 512" }, .{ "TENSORFOLD_PREFILL_BUDGET", "0" },    .{ "TENSORFOLD_PREFILL_BUDGET", "4294967295" },
    }) |kv| try std.testing.expectEqual(@as(?[]const u8, null), check(&.{kv}));
}

test "prefill variables: a value its reader would turn into a default is refused by name, with the value" {
    for ([_][3][]const u8{
        .{ "TENSORFOLD_PREFILL_SHARE", "5", "a number from 0.01 to 1" },          .{ "TENSORFOLD_PREFILL_SHARE", "0.005", "a number from 0.01 to 1" },
        .{ "TENSORFOLD_PREFILL_SHARE", "half", "a number from 0.01 to 1" },       .{ "TENSORFOLD_PREFILL_SHARE", "nan", "a number from 0.01 to 1" },
        .{ "TENSORFOLD_PREFILL_SHARE", " 0.5", "a number from 0.01 to 1" },       .{ "TENSORFOLD_PREFILL_POLICY", "FCFS", "fcfs" },
        .{ "TENSORFOLD_PREFILL_POLICY", "srpt", "fcfs" },                         .{ "TENSORFOLD_PREFILL_TOKENS", "2k", "a whole number of rows" },
        .{ "TENSORFOLD_PREFILL_TOKENS", "-64", "a whole number of rows" },        .{ "TENSORFOLD_PREFILL_BUDGET", "+5", "a whole number of rows" },
        .{ "TENSORFOLD_PREFILL_BUDGET", "4294967296", "a whole number of rows" },
    }) |c| {
        const got = check(&.{.{ c[0], c[1] }}) orelse return error.NotRefused;
        try std.testing.expect(std.mem.startsWith(u8, got, c[0]));
        try std.testing.expect(std.mem.indexOf(u8, got, c[1]) != null and std.mem.indexOf(u8, got, c[2]) != null);
    }
    // the readers agree: each refused value is one they take as their default
    try std.testing.expectEqual(@as(f64, 1), Share.fromText("5").p);
    try std.testing.expectEqual(@as(f64, 1), Share.fromText(" 0.5").p);
    try std.testing.expectEqual(Share.Policy.stock, Share.policyOf("FCFS"));
    try std.testing.expectEqual(Share.default_tokens, Share.tokensOf("2k"));
    try std.testing.expectEqual(@as(u64, 0), Share.budget("+5"));
}
