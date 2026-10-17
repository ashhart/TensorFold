//! Flash Next (qwen4_exp) as Living Weights reads it: its family, the layout its config gives, its shard writer.
const std = @import("std");
const living = @import("living");

pub const family = living.families.qwen4_exp;

/// Every layer carries the pocket, all but the last learn; `width` the config's, with `pocket` neurons written earlier.
pub fn layout(hidden: usize, layers: usize, width: usize, pocket: usize) !living.Layout {
    if (layers > living.family.max_layers or pocket >= width) return error.BadConfig;
    var every: living.Layers = .empty;
    for (0..layers) |l| every.set(l);
    return .fromExperts(hidden, width - pocket, layers, every);
}

/// The layout of a config.json's text: hidden_size, num_hidden_layers, the shared expert's width and the pocket's.
pub fn layoutOf(text: std.json.ObjectMap) !living.Layout {
    const k = family.keys;
    const int = living.families.intField;
    return layout(int(text, k.hidden) orelse return error.BadConfig, int(text, k.layers) orelse return error.BadConfig, int(text, k.width) orelse return error.BadConfig, int(text, k.pocket) orelse 0);
}

/// The writer over `core` (its shard_edit and safetensors).
pub fn Shards(comptime core: type) type {
    const W = living.shards.Io(core);
    return struct {
        /// Layers' neurons [layers, neurons, D] into `dir`'s shards, widened on the first write; returned as written.
        pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, lay: living.Layout, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
            return W.write(gpa, io, dir, family, lay, layers, neurons, &.{ gate, up }, down);
        }

        pub fn writePending(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, lay: living.Layout, layers: []const usize, neurons: usize, gate: []f32, up: []f32, down: []f32) !usize {
            return W.writePending(gpa, io, dir, family, lay, layers, neurons, &.{ gate, up }, down);
        }
    };
}
