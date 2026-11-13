//! XPU family facts through the lane-host API, with inert objects and no Level Zero calls.
const check = @import("draft_sources_check.zig").check;
const present = @import("draft_sources_check.zig").present;

comptime {
    _ = @import("xpu_dflash2_test.zig"); // its tests run under this file's "draft sources:" filter
    _ = @import("xpu_dflash2_depth_test.zig"); // the core depth sweep on the XPU adapter
    _ = @import("xpu_logrow.zig"); // the logprob row's own test
}

test "draft sources: XPU Qwen27 copies and actual DFlash2 attachment" {
    var b: @import("xpu_lanes.zig").Lanes = undefined;
    b.drafter = null;
    b.block = 8;
    b.timed = 0;
    b.slots = 4;
    try check(b.facts(), null, true, false);
    b.drafter = present(@TypeOf(b.drafter.?));
    try check(b.facts(), .{ .kind = "DFlash2", .name = "example/draft" }, true, false);
    _ = &@TypeOf(b).backend; // the lane backend's operations compile on hosts without Level Zero
}

test "draft sources: XPU Qwen27 declares logprobs for the lanes, the lone driver and the wrapped drafter" {
    const std = @import("std");
    const native = @import("xpu.zig");
    const info = native.laneInfo(2, 4096, null);
    try std.testing.expect(info.logprobs and info.structures);
    try std.testing.expect(!info.prompt_logprobs); // prompt logprobs stay refused
    var b: @import("xpu_lanes.zig").Lanes = undefined;
    try std.testing.expect(native.loneDriver(&b).logprobs and native.loneDriver(&b).sampled);
    try std.testing.expect(b.backend().vtable.first_row != null);
    var w = try native.wrapDrafter(std.testing.allocator, b.backend(), .{ .ptr = &b, .vtable = undefined });
    defer w.deinit();
    try std.testing.expect(w.backend().vtable.first_row != null); // a drafted request keeps its first token's row
}

test "draft sources: XPU Nemotron follows the checkpoint head, with no copy window when absent" {
    const n = @import("nemotron_xpu");
    var e: n.Engine = undefined;
    var b: @import("xpu_lane.zig").Xpu(@import("xpu_nemotron.zig")) = undefined;
    b.e = &e;
    b.streams = 4;
    b.costs = @splat(.{ .width = 1, .ms = 0 });
    b.mtp_ms = 0;
    e.head = null;
    try check(b.facts(), null, false, false);
    e.head = present(@TypeOf(e.head.?));
    try check(b.facts(), null, true, true);
}

test "slide admission: XPU refuses every family before device access" {
    const native = @import("xpu.zig");
    for (native.families) |f| try @import("slide_open_test.zig").check(native.open, "XPU", &.{f.model_type});
    try @import("slide_open_test.zig").check(@import("xpu_nem.zig").open, "XPU", &.{ "nemotron_h", "gemma4" });
}

test "capability audit: XPU unknown-model refusal names every listed family" {
    const std = @import("std");
    const native = @import("xpu.zig");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    try std.testing.expect(try native.open(a, std.testing.allocator, std.testing.io, .{ .dir = "does-not-exist", .model_type = "unknown-test-model" }, &problem) == null);
    try std.testing.expect(std.mem.indexOf(u8, problem, "unknown-test-model") != null);
    for (native.families) |f| try std.testing.expect(std.mem.indexOf(u8, problem, f.model_type) != null);
    if (@import("xpu_engine_options").gemma4) {
        try std.testing.expect(std.mem.indexOf(u8, problem, "Gemma 4 E4B (gemma4, MLX 4/64)") != null);
    } else {
        try std.testing.expect(std.mem.indexOf(u8, problem, "Gemma 4 E4B need DPAS") != null);
        try std.testing.expect(std.mem.indexOf(u8, problem, "(gemma4,") == null);
    }
}

test "draft sources: XPU verify gives each window row the logprob row of its own place in the block" {
    const std = @import("std");
    const lanes = @import("lanes");
    const gpa = std.testing.allocator;
    const Src = struct {
        buf: [8][10]f32 = undefined,
        pub fn rowLogits(src: *@This(), row: u32) ![]const f32 {
            @memset(&src.buf[row], 0);
            src.buf[row][row] = 9; // row n's best id is n
            return &src.buf[row];
        }
    };
    var src: Src = .{};
    var a = try lanes.Stream.init(gpa, .{ .id = "a", .prompt = &.{1}, .max_new = 4, .logprobs = 1 });
    defer a.deinit(gpa);
    var b = try lanes.Stream.init(gpa, .{ .id = "b", .prompt = &.{1}, .max_new = 4 }); // no logprobs
    defer b.deinit(gpa);
    var c = try lanes.Stream.init(gpa, .{ .id = "c", .prompt = &.{1}, .max_new = 4, .logprobs = 2 });
    defer c.deinit(gpa);
    const win = [_]lanes.backend.Window{
        .{ .stream = &a, .pending = 0, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{} },
        .{ .stream = &b, .pending = 0, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{} },
        .{ .stream = &c, .pending = 0, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{} },
    };
    var sa = [_]u32{ 0, 1 };
    var sb = [_]u32{ 2, 3, 4 };
    var sc = [_]u32{ 5, 6 };
    var ra: [2]lanes.LogprobRow = undefined;
    var rc: [2]lanes.LogprobRow = undefined;
    const out = [_]lanes.backend.Verified{
        .{ .sampled = &sa, .drafts = &.{}, .rows = &ra },
        .{ .sampled = &sb, .drafts = &.{} },
        .{ .sampled = &sc, .drafts = &.{}, .rows = &rc },
    };
    try @import("xpu_logrow.zig").windowRows(&src, &win, &out, &.{ 2, 3, 2 });
    for (ra, 0..) |r, i| try std.testing.expectEqual(@as(u32, @intCast(i)), r.ids[0]);
    for (rc, 5..) |r, i| try std.testing.expectEqual(@as(u32, @intCast(i)), r.ids[0]); // after the other windows' 5 rows
    try std.testing.expectEqual(@as(u8, 1), ra[0].count);
    try std.testing.expectEqual(@as(u8, 2), rc[1].count);
    try std.testing.expectEqual(@as(u32, 6), rc[1].token);
}
