//! Reserve one reusable expansion buffer from the served affine projection geometry before KV admission.
const std = @import("std");
const E = @import("cuda_engine.zig");

fn include(peak: *usize, projection: E.Proj) !void {
    if (projection != .a4) return;
    const w = projection.a4;
    peak.* = @max(peak.*, try std.math.mul(usize, w.npad, w.k));
}

pub fn reserve(k: anytype, e: *const E.Engine) !void {
    const native = if (k.fast8) |*n| n else return;
    if (native.affine_prepared == null) return;
    var bytes: usize = 0;
    for (e.layers) |layer| {
        switch (layer.mixer) {
            .delta => |m| for ([_]E.Proj{ m.qkv, m.z, m.a, m.b, m.out }) |p| try include(&bytes, p),
            .attention => |m| for ([_]E.Proj{ m.q, m.k, m.v, m.o }) |p| try include(&bytes, p),
        }
        switch (layer.mlp) {
            .dense => |m| for ([_]E.Proj{ m.gate, m.up, m.down }) |p| try include(&bytes, p),
            .moe => {},
        }
    }
    try native.reserveAffinePacked(bytes);
}
