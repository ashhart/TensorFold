//! TF_SHARED_PRICE_LOG: off unless set, malformed text refused by name, and each line in its fixed form.
const std = @import("std");
const t = std.testing;
const a = t.allocator;
const sys = @import("sys");
const log = @import("price_log.zig");
const depth = @import("depth.zig");
const Table = @import("table.zig").Table;

const Window = struct {
    n: usize,
    pub fn rows(w: Window) usize {
        return w.n;
    }
};
const Stream = struct { depth: ?depth.State };
const Plan = struct { stream: *const Stream };

fn costs() !Table {
    var table: Table = .{};
    for (1..5) |n| try table.put(a, @intCast(n), 10 + @as(f64, @floatFromInt(n)));
    return table;
}

fn rule(table: *const Table) depth.Rule {
    return .{ .gpa = a, .prior = &.{0.6}, .most_drafts = 2, .family_costs = table, .shared_costs = table, .mtp_step_ms = 0.5, .plain_guard = true, .node_probabilities = false, .batch_rows = 4 };
}

test "shared price log: the load line lists the shared table by rows and the head steps" {
    var table = try costs();
    defer table.deinit(a);
    var r = rule(&table);
    defer r.deinit();
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try log.load(&out.writer, &r);
    try t.expectEqualStrings("shared price: load shared_ms=1:11.000,2:12.000,3:13.000,4:14.000 step_ms=0.500 shared_step_ms=0.500 draft_step_ms=0.000", out.written());
}

test "shared price log: a round line has its trial, rows, time, each stream's drafts, depth and estimates, and the price" {
    var table = try costs();
    defer table.deinit(a);
    var r = rule(&table);
    defer r.deinit();
    var rates = [_]f64{ 0.625, 0.5 };
    const streams = [_]Stream{ .{ .depth = .{ .p = &rates } }, .{ .depth = null } };
    const plans = [_]Plan{ .{ .stream = &streams[0] }, .{ .stream = &streams[1] } };
    const windows = [_]Window{ .{ .n = 2 }, .{ .n = 1 } };
    const round: log.Round = .{ .trial = false, .sampled = false, .ms = 15.25, .head_ms = 0.5, .heads = &.{0}, .depths = &.{1} };
    var out: std.Io.Writer.Allocating = .init(a);
    defer out.deinit();
    try log.line(&out.writer, &r, &plans, &windows, round);
    try t.expectEqualStrings("shared price: round streams=2 sampled=0 trial=0 rows=3 ms=15.250 head=0.500 forward=13.000 plain=- row=- drafts=1,0 depths=1,- accept=0.625/0.500,-", out.written());
    try r.observeShared(2, 2, 15, false, 0, true, false);
    try r.observeShared(2, 4, 19, false, 0, false, false);
    out.clearRetainingCapacity();
    try log.line(&out.writer, &r, &plans, &windows, .{ .trial = true, .sampled = false, .ms = 15, .head_ms = 0, .heads = &.{0}, .depths = &.{0} });
    try t.expectEqualStrings("shared price: round streams=2 sampled=0 trial=1 rows=3 ms=15.000 head=0.000 forward=13.000 plain=3.000 row=1.000 drafts=1,0 depths=0,- accept=0.625/0.500,-", out.written());
}

const Env = struct {
    raw: []const u8,
    pub fn get(v: Env, name: []const u8) ?[]const u8 {
        return if (std.mem.eql(u8, name, log.name)) v.raw else null;
    }
};

test "shared price log: off unset or 0, on at 1 and read once, other text refused by name" {
    defer _ = sys.env.unsetenv(log.name);
    var table = try costs();
    defer table.deinit(a);
    const E = struct { gpa: std.mem.Allocator, rule: depth.Rule, price_log: ?bool = null };
    for ([_]?[:0]const u8{ null, "", "0", "1", "yes", "2" }) |raw| {
        if (raw) |v| try t.expectEqual(@as(c_int, 0), sys.env.setenv(log.name, v, 1)) else _ = sys.env.unsetenv(log.name);
        var e: E = .{ .gpa = a, .rule = rule(&table) };
        defer e.rule.deinit();
        const want = raw != null and std.mem.eql(u8, raw.?, "1");
        if (raw != null and (std.mem.eql(u8, raw.?, "yes") or std.mem.eql(u8, raw.?, "2"))) {
            try t.expectError(error.InvalidEnvironment, log.enabled(&e));
            // the server's start refuses it first, with every backend's scopes off
            const why = (try sys.environment.refusal(a, Env{ .raw = raw.? }, .{})).?;
            defer a.free(why);
            try t.expect(std.mem.startsWith(u8, why, log.name ++ "=\""));
            continue;
        }
        try t.expectEqual(want, try log.enabled(&e));
        try t.expectEqual(@as(c_int, 0), sys.env.setenv(log.name, if (want) "0" else "1", 1));
        try t.expectEqual(want, try log.enabled(&e));
    }
}
