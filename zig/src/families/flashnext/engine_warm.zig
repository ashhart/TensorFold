//! Every kernel a checkpoint's widths need, compiled at load: a compile failure refuses the load, not a request.
const std = @import("std");
const replay = @import("replay.zig");
const variants = @import("variants.zig");
const lane_w = @import("lane_w.zig");
const Run = replay.Run;
const Model = replay.Model;

/// `name` at every recorded row count on its kernel at `consts` (none: the recorded width, compiled already).
fn role(r: *Run, name: []const u8, consts: []const variants.Const) !void {
    if (consts.len == 0) return;
    var buf: [96]u8 = undefined;
    for (1..replay.MAXR + 1) |n| if (r.roles.get(try std.fmt.bufPrint(&buf, "{s}|{d}", .{ name, n }))) |s| {
        _ = try variants.patched(r, s.v, consts);
    };
}

fn hc(r: *Run, h: replay.Hc, comptime at: []const u8) !void {
    const w = h.w orelse return;
    var c: [2][2]variants.Const = undefined;
    try role(r, "qa_hc_down@" ++ at, variants.width(&c[0], "BITS", "GS", w[0]));
    try role(r, "qa_hc_up@" ++ at, variants.width(&c[1], "BITS", "GS", w[1]));
}

fn mtpHc(r: *Run, h: replay.Hc, comptime at: []const u8) !void {
    const w = h.w orelse return;
    var c: [2][2]variants.Const = undefined;
    try role(r, "mtp:qa_hc_down@" ++ at, variants.width(&c[0], "BITS", "GS", w[0]));
    try role(r, "mtp:qa_hc_up@" ++ at, variants.width(&c[1], "BITS", "GS", w[1]));
}

fn experts(r: *Run, xw: ?variants.Experts, gu: []const u8, down: []const u8) !void {
    const w = xw orelse return;
    var cg: [4]variants.Const = undefined;
    var cd: [4]variants.Const = undefined;
    try role(r, gu, variants.four(&cg, w.gu));
    try role(r, down, variants.four(&cd, w.down));
}

/// A lane's runs at both row tilings (16 and 32 rows a tile).
fn lane(r: *Run, l: replay.Lane) !void {
    const w = l.w orelse return;
    for (w.runs) |p| for ([_]usize{ 1, 2 }) |tmr| {
        _ = try lane_w.pipeline(r.arena, r.device, &r.wide_pipes, .{ .n = p.n, .k = p.k, .bits = p.bits, .group = p.group, .tmr = tmr, .ld = w.n, .mapped = p.map != null, .tiled = p.tiled });
    };
}

/// Every decode kernel `m`'s widths launch; none on a 6/32 checkpoint.
pub fn all(r: *Run, m: *const Model) !void {
    for (&m.layers) |*L| {
        try hc(r, L.ahc, "ahc");
        try hc(r, L.mhc, "mhc");
        try experts(r, L.xw, "qa_expert_gateup@moe.gate", "qa_expert_down_y@moe.down");
        try lane(r, L.proj);
        try lane(r, L.out);
    }
    try hc(r, m.mix, "mix");
    try mtpHc(r, m.mtp.ahc, "mtp.ahc");
    try mtpHc(r, m.mtp.mhc, "mtp.mhc");
    try mtpHc(r, m.mtp.mix, "mtp.mix");
    try experts(r, m.mtp.xw, "mtp:qa_expert_gateup@mtp.moe.gate", "mtp:qa_expert_down_y@mtp.moe.down");
    for ([_]replay.Lane{ m.head, m.ple.kv, m.mtp.proj, m.mtp.out, m.mtp.fce, m.mtp.fch, m.mtp.draft }) |l| try lane(r, l);
    var c: [2]variants.Const = undefined;
    try role(r, "qa_embed_rows@embed", variants.width(&c, "BITS", "GS", m.embed_w));
    try role(r, "mtp:qa_embed_rows@embed", variants.width(&c, "BITS", "GS", m.embed_w));
    try role(r, "qa_ple_lookup@ple", variants.width(&c, "BITS", "GS", m.ple.w));
}
