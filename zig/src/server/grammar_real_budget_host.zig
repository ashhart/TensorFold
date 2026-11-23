//! CPU regression with an explicit real tokenizer file; no model, GPU or network is used.
const std = @import("std");
const lanes = @import("lanes");
const json = @import("json");
const tool_grammar = @import("tool_grammar.zig");
const t = std.testing;
const offered =
    \\[{"type":"function","function":{"name":"weather","description":"Get a weather forecast","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city","days"],"additionalProperties":false}}},{"type":"function","function":{"name":"clock","description":"Read local time","parameters":{"type":"object","properties":{"timezone":{"type":"string"}},"required":["timezone"],"additionalProperties":false}}}]
;

test "real grammar budget: export the actual required Qwen tool grammar for both parallel settings" {
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, offered)).ok.array;
    for ([_]bool{ true, false }) |parallel| {
        const text = try tool_grammar.build(a, .xml, tools, parallel);
        const parsed = (try json.parse(a, text)).ok;
        try t.expectEqualStrings("structural_tag", parsed.strField("type").?);
        std.debug.print("REAL_BUDGET_INPUT {f}\n", .{std.json.fmt(.{ .name = if (parallel) "required_tools" else "required_tools_single", .kind = @as(u8, 5), .text = text }, .{})});
    }
}

const fixture = @import("grammar_real_fixture.zig");
const budget_request = @import("grammar_budget.zig");
const errors = @import("errors.zig");
const fixed_object = "{\"type\": \"object\"}";

test "real grammar budget: json_object stays covered with the real Qwen3.8 vocabulary" {
    const l = try fixture.Lib.open();
    defer l.close();
    const g = try l.compile(.json, "");
    defer g.free();
    try t.expectEqual(lanes.grammar.Readiness.ready, try g.budgetReadiness());
}

test "real grammar budget: fixed and identical client rules never borrow each other's policy in either order" {
    for ([_]bool{ false, true }) |fixed_first| {
        const l = try fixture.Lib.open();
        defer l.close();
        // Make alias isolation independent of host load; the server cannot call this existing test-only hook.
        (l.limit orelse return error.MissingSymbol)(l.h, 1_000_000);
        const fixed = try l.compile(.json, "ignored request text cannot alter the fixed OBJECT rule");
        defer fixed.free();
        // Exact OBJECT bytes and compile options may share xgrammar's compiled pointer: origin must remain distinct.
        const client = try l.compile(.json_schema, fixed_object);
        defer client.free();
        if (fixed_first) try t.expectEqual(lanes.grammar.Readiness.ready, try fixed.budgetReadiness());
        try t.expectEqual(lanes.grammar.Readiness.over_work, try client.budgetReadiness());
        var cx: errors.Cx = .{ .a = t.allocator };
        try t.expectError(error.Refused, budget_request.require(&cx, client));
        try t.expectEqual(@as(u16, 400), cx.status());
        try t.expect(std.mem.indexOf(u8, cx.message, "work limit") != null);
        try t.expectEqual(lanes.grammar.Readiness.ready, try fixed.budgetReadiness());
        try t.expectEqual(lanes.grammar.Readiness.over_work, try client.budgetReadiness());
        (l.limit orelse return error.MissingSymbol)(l.h, @as(i64, 1) << 28);
        // Churn both eight-entry client caches, freeing wrappers between calls; the fixed result must remain pinned.
        for (0..10) |i| {
            const text = try std.fmt.allocPrint(t.allocator, "{{\"type\":\"object\",\"properties\":{{\"p{d}\":{{\"const\":0}}}},\"required\":[\"p{d}\"],\"additionalProperties\":false}}", .{ i, i });
            defer t.allocator.free(text);
            const other = try l.compile(.json_schema, text);
            defer other.free();
            try t.expectEqual(lanes.grammar.Readiness.ready, try other.budgetReadiness());
        }
        const again = try l.compile(.json, "different ignored text");
        defer again.free();
        const began = std.Io.Clock.awake.now(t.io);
        try t.expectEqual(lanes.grammar.Readiness.ready, try again.budgetReadiness());
        const ns = began.durationTo(std.Io.Clock.awake.now(t.io)).toNanoseconds();
        std.debug.print("FIXED_JSON_CACHE after client churn: {d} ns\n", .{ns});
        try t.expect(ns < std.time.ns_per_s); // a cache lookup, not another seven-second exploration
    }
}

test "real grammar budget: original client caps still refuse the fixed-rule schema with a resource or time message" {
    const l = try fixture.Lib.open();
    defer l.close();
    const client = try l.compile(.json_schema, fixed_object);
    defer client.free();
    const cause = try client.budgetReadiness();
    try t.expect(cause == .over_work or cause == .timeout);
    var cx: errors.Cx = .{ .a = t.allocator };
    try t.expectError(error.Refused, budget_request.require(&cx, client));
    try t.expectEqual(@as(u16, 400), cx.status());
    try t.expect(std.mem.indexOf(u8, cx.message, "work limit") != null or std.mem.indexOf(u8, cx.message, "time limit") != null);
    try t.expect(std.mem.indexOf(u8, cx.message, "left-recursive") == null);
}

test "real grammar budget: client deadline remains two seconds and its real refusal names time" {
    const l = try fixture.Lib.open();
    defer l.close();
    // Existing host-test hook raises this compiler's work cap only, so its unchanged client deadline is reached.
    (l.limit orelse return error.MissingSymbol)(l.h, @as(i64, 1) << 38);
    const client = try l.compile(.json_schema, fixed_object);
    defer client.free();
    var cx: errors.Cx = .{ .a = t.allocator };
    try t.expectError(error.Refused, budget_request.require(&cx, client));
    try t.expectEqual(@as(u16, 400), cx.status());
    try t.expect(std.mem.indexOf(u8, cx.message, "time limit") != null);
    try t.expect(std.mem.indexOf(u8, cx.message, "left-recursive") == null);
}

const Fake = struct {
    var code: c_int = 1;
    fn status(_: *anyopaque) callconv(.c) c_int {
        return code;
    }
};

test "real grammar budget: actual server helper emits distinct 400s and keeps legacy status fallback" {
    var f: lanes.grammar.Library = undefined;
    f.budget_ready = null;
    f.budget_status = Fake.status;
    var g: lanes.grammar.Compiled = .{ .f = f, .h = @ptrFromInt(1) };
    for ([_][]const u8{ "rules", "", "work limit", "successor-edge", "table-memory", "time limit", "compile slot" }, 0..) |part, code| {
        Fake.code = @intCast(code);
        var cx: errors.Cx = .{ .a = t.allocator };
        if (code == 1) {
            try budget_request.require(&cx, g);
            continue;
        }
        try t.expectError(error.Refused, budget_request.require(&cx, g));
        try t.expectEqual(@as(u16, 400), cx.status());
        try t.expect(std.mem.indexOf(u8, cx.message, part) != null);
        if (code > 1) try t.expect(std.mem.indexOf(u8, cx.message, "left-recursive") == null);
    }
    Fake.code = -1;
    var cx: errors.Cx = .{ .a = t.allocator };
    try t.expectError(error.Refused, budget_request.require(&cx, g));
    try t.expect(std.mem.indexOf(u8, cx.message, "exploration failed") != null);
    g.f.budget_status = null;
    g.f.budget_ready = Fake.status;
    Fake.code = 2;
    try t.expectError(error.Refused, budget_request.require(&cx, g));
    try t.expect(std.mem.indexOf(u8, cx.message, "work limit") != null);
}
