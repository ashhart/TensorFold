//! XPU family facts through the lane-host API, with inert objects and no Level Zero calls.
const check = @import("draft_sources_check.zig").check;
const present = @import("draft_sources_check.zig").present;

test "draft sources: XPU Qwen27 copies and actual DFlash2 attachment" {
    var b: @import("xpu_lanes.zig").Lanes = undefined;
    b.drafter = null;
    b.block = 8;
    b.timed = 0;
    b.slots = 4;
    try check(b.facts(), null, true, false);
    b.drafter = present(@TypeOf(b.drafter.?));
    try check(b.facts(), .{ .kind = "DFlash2", .name = "example/draft" }, true, false);
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
