//! glm.layer.dsa's float64 part: the layer's HC glue and MoE (glm_layer_ref), the DSA block's chain (glm_dsa_ref).
const std = @import("std");
const cuda = @import("cuda");
const ref = @import("glm_kda_ref.zig");
const lref = @import("glm_layer_ref.zig");
const dref = @import("glm_dsa_ref.zig");
const lay = cuda.glm_layer;

const Obj = std.json.ObjectMap;
const id = "glm.layer.dsa";

/// The whole layer in f64, stage by stage from each stage's inputs; returns the worst fraction of tolerance.
pub fn layerF64(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: lay.Config, h: *const lref.Host, dh: *const dref.Host, p: dref.Params, g: dref.Got, tag: []const u8, e: Obj) !f64 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const w: lref.Win = .{ .a = a, .io = io, .dir = dir, .e = e };
    var rp: lref.Report = .{ .a = a, .tag = tag, .id = id };
    const rows: usize = @intCast(ref.int(e, "rows"));
    const pos: usize = @intCast(ref.int(e, "pos"));
    const d: usize = c.d;
    // the attention site: hc_pre, the DSA block, hc_post
    const x0 = try w.bf(null, "x0");
    const n0 = try w.bf("s0", "normed");
    try lref.hcPreStage(&rp, a, "attn hc_pre", c, h.attn_hc, h.in_norm, x0, rows, n0, try w.f("s0", "post"), try w.f("s0", "comb"));
    const qd = try dref.chain(a, dh, p, g, tag, rows, pos);
    rp.worst = @max(rp.worst, qd);
    rp.line = try std.fmt.allocPrint(a, "{s} dsa block {d:.3};", .{ rp.line, qd });
    const x1 = try w.bf("s1", "x");
    try lref.hcPostStage(&rp, "attn hc_post", d, x0, try w.f("s0", "part"), try w.f("s0", "peer"), try w.f("s0", "post"), try w.f("s0", "comb"), rows, x1);
    // the MoE site: hc_pre, router, select, routed and shared experts, combine, hc_post
    const n1 = try w.bf("s1", "normed");
    try lref.hcPreStage(&rp, a, "ffn hc_pre", c, h.ffn_hc, h.post_norm, x1, rows, n1, try w.f("s1", "post"), try w.f("s1", "comb"));
    try lref.moeStage(&rp, a, w, c, h, n1, rows);
    try lref.hcPostStage(&rp, "ffn hc_post", d, x1, try w.f("s1", "part"), try w.f("s1", "peer"), try w.f("s1", "post"), try w.f("s1", "comb"), rows, try w.bf(null, "x"));
    std.debug.print("RESULT {s} {s} f64 per stage (fraction of tolerance):{s}\n", .{ id, tag, rp.line });
    return rp.worst;
}
