//! XPU family facts through the lane-host API, with inert objects and no Level Zero calls.
const check = @import("draft_sources_check.zig").check;
const present = @import("draft_sources_check.zig").present;

comptime {
    _ = @import("xpu_dflash2_test.zig"); // its tests run under this file's "draft sources:" filter
    _ = @import("xpu_dflash2_depth_test.zig"); // the core depth sweep on the XPU adapter
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

test "lfm3 refusal: the XPU open refuses the ModelOpt NVFP4 export by name, before the driver or any weight" {
    const std = @import("std");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = @import("lfm3_xpu");
    // config.json alone and no Level Zero here: an open that went on would fail at the driver with another problem
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg.fixtures.nvfp4 });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    const o: @import("engine_api").Open = .{ .dir = dir, .model_type = "lfm3", .context = 4096 };
    try std.testing.expect((try @import("xpu.zig").open(a, std.testing.allocator, io, o, &problem)) == null);
    try std.testing.expectEqualStrings(cfg.config.servedProblem(error.Lfm3ModeloptNvfp4NotOnXpu).?, problem);
    // the MLX export goes on to the driver: the refusal is the scheme's alone
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg.fixtures.text });
    try std.testing.expect((try @import("xpu.zig").open(a, std.testing.allocator, io, o, &problem)) == null);
    try std.testing.expect(!std.mem.eql(u8, cfg.config.servedProblem(error.Lfm3ModeloptNvfp4NotOnXpu).?, problem));
}

test "lfm3 refusal: the XPU open refuses the block FP8 export by name, before the driver or any weight" {
    const std = @import("std");
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const cfg = @import("lfm3_xpu");
    // config.json alone and no Level Zero here: an open that went on would fail at the driver with another problem
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = cfg.fixtures.fp8 });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    var problem: []const u8 = "";
    const o: @import("engine_api").Open = .{ .dir = dir, .model_type = "lfm3", .context = 4096 };
    try std.testing.expect((try @import("xpu.zig").open(a, std.testing.allocator, io, o, &problem)) == null);
    try std.testing.expectEqualStrings(cfg.config.servedProblem(error.Lfm3Fp8BlockNotOnXpu).?, problem);
}
