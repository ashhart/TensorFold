//! Optional captures and additions in the serving forward; an ordinary model has no learning hooks.
const mtl = @import("metal");
const Ref = @import("projection.zig").Ref;

pub const Point = enum { input, gdn, gdn_state, attention, prepared_attention, gated, post, mlp, final, logits };
pub const Hooks = struct {
    ctx: *anyopaque,
    save: *const fn (*anyopaque, mtl.ComputeEncoder, Point, usize, []const Ref, u32) anyerror!void,
    add: *const fn (*anyopaque, mtl.ComputeEncoder, usize, Ref, Ref, u32) anyerror!void,

    pub fn capture(h: Hooks, e: mtl.ComputeEncoder, point: Point, layer: usize, refs: []const Ref, rows: u32) !void {
        try h.save(h.ctx, e, point, layer, refs, rows);
    }
};
