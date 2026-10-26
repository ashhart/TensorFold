//! Numeric host instruments feed the same capture ledger; these callbacks never open a device.
const std = @import("std");
const tap = @import("../../src/families/qwen4/node_trace.zig");
pub fn note(g: anytype, layer: usize, scope: []const u8, node: []const u8, values: anytype, kind: tap.Kind, row: usize) !void {
    const sink = g.trace orelse return;
    if (values.len == 0) return;
    const bytes = std.mem.sliceAsBytes(values);
    const storage = try g.a.alignedAlloc(u8, .@"16", bytes.len);
    @memcpy(storage, bytes);
    var key: [192]u8 = undefined;
    try sink.emit(sink.ctx, try tap.label(&key, layer, scope, node), storage, 0, 1, values.len, kind, row);
}
