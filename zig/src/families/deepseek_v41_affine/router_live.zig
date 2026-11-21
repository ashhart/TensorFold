//! Exact router projection and selection candidates for short verify windows.
const std = @import("std");
const mtl = @import("metal");
const c = @import("config.zig");
const st = @import("state.zig");
const wts = @import("weights.zig");
const probe = @import("router_reduction.zig");
const Buffer = mtl.Buffer;
pub const max_rows = 1 + c.draft_block;

pub const Kernels = struct {
    projections: [max_rows]mtl.Pipeline,
    selector: ?mtl.Pipeline,
    audit: ?mtl.Pipeline,

    pub fn load(device: mtl.Device, select: bool, audit: bool) !Kernels {
        const shader = @embedFile("router_reduction.metal") ++ "\n" ++
            \\[[kernel]] void tf_router_audit(
            \\  const device uint* logits [[buffer(0)]], const device uint* ids [[buffer(1)]],
            \\  const device uint* w [[buffer(2)]], const device uint* old_logits [[buffer(3)]],
            \\  const device uint* old_ids [[buffer(4)]], const device uint* old_w [[buffer(5)]],
            \\  device atomic_uint* differs [[buffer(6)]], constant uint& rows [[buffer(7)]],
            \\  uint t [[thread_position_in_grid]]) {
            \\    uint mask = logits[t] == old_logits[t] ? 0u : 1u;
            \\    if (t < rows * 6) {
            \\      if (ids[t] != old_ids[t]) mask |= 2u;
            \\      if (w[t] != old_w[t]) mask |= 4u;
            \\    }
            \\    if (mask != 0) atomic_fetch_or_explicit(differs, mask, memory_order_relaxed);
            \\}
        ;
        const lib = try mtl.Library.fromSource(device, shader, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var rows: [max_rows]mtl.Pipeline = undefined;
        var n: usize = 0;
        errdefer for (rows[0..n]) |p| p.deinit();
        for (1..max_rows + 1) |r| {
            rows[r - 1] = try probe.projection(device, lib, r, if (r == 1) 2 else 1);
            n += 1;
        }
        const selector = if (select) try mtl.Pipeline.init(device, lib, "tf_route_merge", false) else null;
        errdefer if (selector) |p| p.deinit();
        const checked = if (audit) try mtl.Pipeline.init(device, lib, "tf_router_audit", false) else null;
        return .{ .projections = rows, .selector = selector, .audit = checked };
    }

    pub fn deinit(p: Kernels) void {
        for (p.projections) |pipe| pipe.deinit();
        if (p.selector) |pipe| pipe.deinit();
        if (p.audit) |pipe| pipe.deinit();
    }

    pub fn issue(p: Kernels, e: mtl.ComputeEncoder, w: wts.Tensor, input: Buffer, output: Buffer, rows: usize) void {
        probe.encodeProjection(e, p.projections[rows - 1], w, input, output, rows, if (rows == 1) 4 else 2);
    }

    pub fn compare(p: mtl.Pipeline, e: mtl.ComputeEncoder, s: *st.Scratch, rows: usize) void {
        e.setPipeline(p);
        for ([_]Buffer{ s.router, s.route_pick, s.route_weight, s.router_ref_logits, s.router_ref_ids, s.router_ref_weights, s.router_variant_error }, 0..) |buf, i| e.setBuffer(buf, 0, i);
        e.setValue(@as(u32, @intCast(rows)), 7);
        e.dispatchThreads(mtl.Size.of(rows * c.experts, 1, 1), mtl.Size.of(256, 1, 1));
    }
};
