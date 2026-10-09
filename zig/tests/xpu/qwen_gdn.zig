//! Gated DeltaNet layer 0 on real weights, 8-step decode chain vs numpy. usage: xpu-qwen-gdn-test [CHECKPOINT_DIR]

const std = @import("std");
const tfix = @import("fx.zig");
const rt = @import("rig.zig");
const ld = @import("xpu").loader;
const cq = @import("qwen_xpu").config;
const qb = @import("qwen_xpu").blocks;
const ql = @import("qwen_xpu").load;
const c = @import("qwen_common.zig");

var fx: []const u8 = &.{};
var fy: []const u8 = &.{};
const steps = 8;
const layer = 0;

pub const panic = std.debug.FullPanic(@import("rig.zig").panicDrain);
fn loadFixtures() !void {
    fx = try tfix.load("qwen_gdn_x");
    fy = try tfix.load("qwen_gdn_y");
}

pub fn main(init: std.process.Init) !void {
    try loadFixtures();
    const dir = try c.checkpoint(init);
    var r = try rt.open();
    defer r.deinit();
    const cfg = try cq.parse(c.gpa, try ld.readFile(c.gpa, dir, "config.json"));
    var l = try ld.Loader.init(c.gpa, &r, dir);
    var ops = try qb.Ops.init(&r, &l, cfg.value, 16);
    try ql.attach(c.gpa, &ops, &l, cfg.value);
    const w = try ql.loadGdn(c.gpa, &ops, &l, layer);
    std.debug.print("gdn layer {d}: {d:.2} GB on device\n", .{ layer, @as(f64, @floatFromInt(l.total)) / 1e9 });
    const want = try c.aligned(fy);
    var worst_rel: f32 = 0;
    var worst_ulp: f32 = 0;
    for (0..steps) |t| {
        try r.upload(ops.s.x, fx[t * qb.hidden * 2 ..][0 .. qb.hidden * 2]);
        try ops.gdn(w);
        try ops.flush();
        const got = try c.fetch(&r, ops.s.x, qb.hidden);
        const s = c.stats(got, want[t * qb.hidden ..][0..qb.hidden]);
        std.debug.print("step {d}: {d} differ, worst {e:.2} of max|y|, worst {d:.2} ulp\n", .{ t, s.differ, s.rel_max, s.ulps });
        worst_rel = @max(worst_rel, s.rel_max);
        worst_ulp = @max(worst_ulp, s.ulps);
    }
    std.debug.print("gdn chain: worst {e:.2} of max|y|, worst {d:.2} ulp\n", .{ worst_rel, worst_ulp });
    if (worst_rel > 0.02 or worst_ulp > 8) return error.TooInaccurate;
}
