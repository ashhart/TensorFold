//! One preflighted training allocation carves disjoint persistent gradients and mixer workspaces.
const std = @import("std");
const cfg = @import("config.zig");
pub const Slot = enum {
    dummy,
    grad,
    branch,
    injection,
    norm,
    hc_up,
    hc_low,
    hc_down,
    hc_inject,
    mixed,
    act,
    gate,
    up,
    mixer,
    projection,
    gdn_act,
    gdn_qk,
    gdn_gy,
    gdn_parts,
    gdn_work,
    gdn_input,
    qsa_context,
    qsa_q,
    qsa_k,
    qsa_v,
    qsa_scores,
    qsa_dp,
    qsa_p,
    qsa_ds,
    pocket_coeff,
    logits,
    stats,
    targets,
    weights,
    delta,
    push,
};
fn mul(xs: []const usize) !usize {
    var n: usize = 1;
    for (xs) |x| n = try std.math.mul(usize, n, x);
    return n;
}
pub fn Plan(comptime B: type) type {
    return struct {
        const Self = @This();
        pub const Ref = @import("train_tensor.zig").Ref(B);
        offsets: std.EnumArray(Slot, usize) = .initFill(0),
        lengths: std.EnumArray(Slot, usize) = .initFill(0),
        bytes: usize = 0,
        pub fn make(c: *const cfg.Config, r: usize) !Self {
            var out: Self = .{};
            const q = c.heads * c.head_dim;
            const kv = c.kv_heads * c.head_dim;
            var problem: cfg.Problem = .{};
            const qsa = try @import("cuda_attention.zig").Geometry.of(c, &problem);
            const projection = @max(c.convWidth() + c.valueWidth() + 2 * c.value_heads, qsa.projectionWidth());
            inline for (std.enums.values(Slot)) |slot| {
                const len = switch (slot) {
                    .dummy => 16,
                    .grad, .norm, .hc_up => try mul(&.{ r, c.wide(), 4 }),
                    .branch, .mixed => try mul(&.{ r, c.hidden, 4 }),
                    .injection, .hc_inject => try mul(&.{ r, c.streams, 4 }),
                    .hc_low, .hc_down => try mul(&.{ r, c.lowrank, 4 }),
                    .act, .gate, .up => try mul(&.{ r, c.intermediate, 4 }),
                    .mixer => try mul(&.{ r, @max(q, c.valueWidth()), 4 }),
                    .projection => try mul(&.{ r, projection, 4 }),
                    .gdn_act => try mul(&.{ r, c.convWidth(), 2 }),
                    .gdn_input => try mul(&.{ r, c.convWidth(), 4 }),
                    .gdn_qk => try mul(&.{ r, 2, c.key_heads, 128, 4 }),
                    .gdn_gy => try mul(&.{ r, c.value_heads, 128, 4 }),
                    .gdn_parts => try mul(&.{ r, c.value_heads, 386, 4 }),
                    .gdn_work => try mul(&.{ c.value_heads, 128, 128, 4 }),
                    .qsa_context, .qsa_q => try mul(&.{ r, q, 4 }),
                    .qsa_k, .qsa_v => try mul(&.{ r, kv, 4 }),
                    .qsa_scores, .qsa_dp, .qsa_p, .qsa_ds => try mul(&.{ r, c.heads, r, 4 }),
                    .pocket_coeff => try mul(&.{ r, 64, 2, 4 }),
                    .logits => try mul(&.{ r, c.vocab, 4 }),
                    .stats => try mul(&.{ r, 2, 4 }),
                    .targets, .weights => try mul(&.{ r, 4 }),
                    .delta, .push => try mul(&.{ 3, c.layers, c.hidden, 4 }),
                };
                const at = (try std.math.add(usize, out.bytes, 15)) & ~@as(usize, 15);
                out.offsets.set(slot, at);
                out.lengths.set(slot, len);
                out.bytes = try std.math.add(usize, at, len);
            }
            _ = try B.allocationBytes(out.bytes);
            return out;
        }
        pub fn get(p: Self, buffer: B.Buffer, slot: Slot) Ref {
            return .{ .buffer = buffer, .offset = p.offsets.get(slot), .len = p.lengths.get(slot) };
        }
    };
}
