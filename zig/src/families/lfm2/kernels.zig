//! LFM2's pipelines: the family's kernels, and core/tile_qmm's prompt tiles and their decode twin.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const tile = @import("../../core/tile_qmm.zig");

const names = [_][:0]const u8{
    "lfm2_embed_b4",
    "lfm2_embed_b6",
    "lfm2_embed_b8",
    "lfm2_norm",
    "lfm2_conv",
    "lfm2_qk",
    "lfm2_attn_mma",
    "lfm2_attn_mma_r2",
    "lfm2_attn_merge",
    "lfm2_route_logits",
    "lfm2_route_select",
    "lfm2_swiglu",
    "lfm2_combine_norm",
    "lfm2_sort",
    "lfm2_combine_sorted",
};

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    tiles: tile.Pipelines,
    /// Whether the twin equals the tiles at every shape (checked at load); otherwise decode rows take the tiles too.
    twin: bool = false,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return self.pipelines[i];
        @compileError("no LFM2 kernel " ++ key);
    }

    /// The embedding gather for a table of `bits`-wide codes.
    pub fn embed(self: *const Kernels, bits: u8) mtl.Pipeline {
        return switch (bits) {
            4 => self.get("lfm2_embed_b4"),
            6 => self.get("lfm2_embed_b6"),
            else => self.get("lfm2_embed_b8"),
        };
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
        self.tiles.deinit();
    }
};

pub fn load(device: mtl.Device) !Kernels {
    var out: Kernels = undefined;
    const lib = try mtl.Library.fromSource(device, sources.lfm2, mtl.CompileOptions.mlx());
    defer lib.deinit();
    var loaded: usize = 0;
    errdefer for (out.pipelines[0..loaded]) |p| p.deinit();
    for (names, 0..) |name, i| {
        out.pipelines[i] = try mtl.Pipeline.init(device, lib, name, false);
        loaded += 1;
    }
    out.tiles = try tile.Pipelines.load(device);
    out.twin = false;
    return out;
}
