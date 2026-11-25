//! Living-memory recurrent state changes: each layer's `S_after - S_prefix` kept whole in F32, with its conv state.
const std = @import("std");
const types = @import("types.zig");
const Allocator = std.mem.Allocator;
const ArchSpec = types.ArchSpec;
const RawState = types.RawState;
const RecLayerData = types.RecLayerData;

/// Every recurrent layer's change of state over the body, whole as computed, and the body's conv state as computed.
pub fn recurrentDeltas(gpa: Allocator, arch: ArchSpec, pref: *const RawState, body: *const RawState) ![]RecLayerData {
    const rec = try gpa.alloc(RecLayerData, arch.num_rec);
    var made: usize = 0;
    errdefer {
        for (rec[0..made]) |*r| r.deinit(gpa);
        gpa.free(rec);
    }
    for (0..arch.num_rec) |r| {
        const d = try gpa.alloc(f32, body.rec_state[r].len);
        errdefer gpa.free(d);
        for (pref.rec_state[r], body.rec_state[r], d) |p, b, *o| o.* = b - p;
        rec[r] = .{ .delta = d, .conv = try gpa.dupe(u16, body.rec_conv[r]) };
        made += 1;
    }
    return rec;
}

/// `dst += w * src`, one fused multiply-add per element in order: a stored change at its composition weight.
pub fn addScaled(src: []const f32, w: f32, dst: []f32) void {
    std.debug.assert(src.len == dst.len);
    for (src, dst) |s, *d| d.* = @mulAdd(f32, w, s, d.*);
}
