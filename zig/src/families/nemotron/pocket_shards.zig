//! The pocket in Nemotron's shards: living/shards.zig over its family and config (up_proj rows, down_proj columns).
const std = @import("std");
const cfg = @import("config.zig");
const living = @import("living");
const pocket_layout = @import("pocket_layout.zig");

/// The writer over `core`, the backend's own (its shard_edit and safetensors).
pub fn Io(comptime core: type) type {
    const Shards = living.shards.Io(core);
    return struct {
        /// The learned neurons of MoE layers `layers` into `dir`'s shards (widened first if new), as 4-bit; the bytes.
        pub fn write(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, c: cfg.Config, layers: []const usize, neurons: usize, up: []f32, down: []f32) !usize {
            return Shards.write(gpa, io, dir, pocket_layout.family, pocket_layout.layout(c), layers, neurons, &.{up}, down);
        }
    };
}
