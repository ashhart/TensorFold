//! Key-parallel attention for decode windows and short prompt tails (development candidate).
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

/// Key parts a row splits into at most, and the fewest keys a part holds.
pub const max_parts = 64;
pub const min_keys = 128;

pub const Pipelines = struct {
    partial: mtl.Pipeline,
    merge: mtl.Pipeline,
    segments: mtl.Pipeline,
    segments_merge: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        const source = try std.mem.concat(std.heap.c_allocator, u8, &.{ sources.lfm3_attention, "\n", sources.lfm3_attention_split });
        defer std.heap.c_allocator.free(source);
        const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
        defer library.deinit();
        const partial = try mtl.Pipeline.init(device, library, "lfm3_attn_split", false);
        errdefer partial.deinit();
        const merge = try mtl.Pipeline.init(device, library, "lfm3_attn_split_merge", false);
        errdefer merge.deinit();
        const segments = try mtl.Pipeline.init(device, library, "lfm3_attn_split_segments", false);
        errdefer segments.deinit();
        const segments_merge = try mtl.Pipeline.init(device, library, "lfm3_attn_split_segments_merge", false);
        errdefer segments_merge.deinit();
        if (partial.simdWidth() != 32 or partial.maxThreads() < 128 or segments.maxThreads() < 128) return error.SplitAttentionReflection;
        return .{ .partial = partial, .merge = merge, .segments = segments, .segments_merge = segments_merge };
    }

    pub fn deinit(self: Pipelines) void {
        inline for (.{ "partial", "merge", "segments", "segments_merge" }) |name| @field(self, name).deinit();
    }
};

/// Part count for the longest row of a launch: `count` keys visible, `window` keys kept (0: all).
pub fn parts(count: usize, window: usize) usize {
    const n = if (window == 0) count else @min(count, window);
    return @min(max_parts, (n + min_keys - 1) / min_keys);
}
