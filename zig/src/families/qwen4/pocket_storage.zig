//! Every backend writes the same quantized rows through the shared pending-save transaction.
const std = @import("std");
const living = @import("living");
const core = @import("core");
const schema = @import("weights.zig");
const profile = @import("learning_profile.zig");

pub fn save(a: std.mem.Allocator, io: std.Io, dir: []const u8, ns: schema.Namespace, wrapped: bool, lay: living.Layout, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
    inline for (std.enums.values(schema.Namespace)) |stored| inline for (.{ false, true }) |scope| {
        if (ns == stored and wrapped == scope) return living.shards.Io(core).writePending(a, io, dir, profile.storedFamily(stored, scope), lay, layers, neurons, &.{ gate, up }, down);
    };
    unreachable;
}
