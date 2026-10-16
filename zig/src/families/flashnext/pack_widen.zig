//! Exact regrouping for lane packs: core/quant widen at the same bits, a group read as smaller ones.
const std = @import("std");
const quant_mlx = @import("../../core/quant/mlx.zig");
const quant_widen = @import("../../core/quant/widen.zig");
const affine = @import("affine.zig");
const Q = @import("pack_lane.zig").Q;

/// A pack linear as core/quant sees it.
pub fn host(q: Q) quant_mlx.Host {
    return .{ .words = q.w, .scales = q.s, .biases = q.b, .bits = q.spec.bits, .group = @intCast(q.spec.group) };
}

/// A core/quant linear back as a pack linear, its K from the tables.
pub fn fromHost(h: quant_mlx.Host) !Q {
    const k = h.scales.dim(1) * h.group;
    return .{ .w = h.words, .s = h.scales, .b = h.biases, .rows = h.words.dim(0), .k = k, .kw = h.words.dim(1), .spec = try affine.Spec.init(h.bits, h.group) };
}

/// `q` in groups of `group`, which divides its own: codes kept, each scale and bias repeated (`QWeights.widened`).
pub fn regroup(a: std.mem.Allocator, q: Q, group: usize) !Q {
    if (group == q.spec.group) return q;
    if (group > q.spec.group or q.spec.group % group != 0) return error.InexactRegroup;
    return fromHost(try quant_widen.widen(a, host(q), .{ .bits = q.spec.bits, .group = @intCast(group) }));
}

test {
    _ = @import("pack_widen_test.zig");
}
