//! Read the bound stream's MTP probabilities before the common lane policy sizes its verify window.
const std = @import("std");
const lanes = @import("lanes");
const mtp = @import("cuda_mtp.zig");
pub fn probabilities(b: anytype, s: *lanes.Stream, out: []f64) !bool {
    return read(b, s, out, Gpu);
}
const Gpu = struct {
    pub fn bind(b: anytype, seq: anytype) void {
        b.e.bind(seq);
    }
    pub fn download(b: anytype, raw: []f32) !void {
        try b.e.ops().download(std.mem.sliceAsBytes(raw), b.head.?.probs);
        try b.e.stream.synchronize();
    }
};

/// The production read boundary, with binding and completed copying supplied for device-free owner checks.
pub fn read(b: anytype, s: *lanes.Stream, out: []f64, comptime Ops: type) !bool {
    if (b.mtp_confidence == null or b.head == null) return false;
    if (out.len > mtp.max_chain) return error.BadDraftCount;
    if (out.len == 0) return true;
    const l = b.lanes.getPtr(s) orelse return error.UnknownStream;
    Ops.bind(b, l.seq);
    var raw: [mtp.max_chain]f32 = undefined;
    try Ops.download(b, raw[0..out.len]);
    try lanes.mtp_policy.cumulative(raw[0..out.len], out);
    return true;
}
