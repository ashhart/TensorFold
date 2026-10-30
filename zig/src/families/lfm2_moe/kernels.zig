//! LFM2's pipelines: the family's kernels, and core/tile_qmm's prompt tiles and their decode twin.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const tile = @import("../../core/tile_qmm.zig");
const am = @import("../../core/affine_mm.zig");
const frags = @import("../../core/frags.zig");

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
    "lfm2_argmax",
    "lfm2_conv_segs",
    "lfm2_qk_segs",
    "lfm2_attn_segs",
    "lfm2_attn_merge_segs",
    "lfm2_tap",
};

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    tiles: tile.Pipelines,
    /// Whether the twin equals the tiles at every shape (checked at load); otherwise decode rows take the tiles too.
    twin: bool = false,
    /// The tensor-unit prompt kernels at each of the tiles' widths (M5 and later; null before, every prompt row tiles).
    nax: ?[tile.bit_widths.len]am.Pipes = null,
    /// Prompt attention on the tensor units (with `nax`); null before the M5.
    attn_nax: ?mtl.Pipeline = null,

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

    /// The tensor-unit pipelines for a width the tiles take, when this GPU has them.
    pub fn naxPipes(self: *const Kernels, bits: u8) ?am.Pipes {
        const set = self.nax orelse return null;
        return set[std.mem.indexOfScalar(u8, &tile.bit_widths, bits) orelse return null];
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
        if (self.nax) |set| for (set) |pipes| for (pipes) |p| p.deinit();
        if (self.attn_nax) |p| p.deinit();
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
    errdefer out.tiles.deinit();
    out.twin = false;
    // `out` is undefined, so attn_nax's default does not apply: unset, it would hold stack garbage.
    out.attn_nax = null;
    out.nax = if (device.tensorUnits()) try naxLoad(device) else null;
    errdefer if (out.nax) |set| for (set) |pipes| for (pipes) |x| x.deinit();
    if (device.tensorUnits()) try attnNaxLoad(device, &out);
    return out;
}

/// lfm2_attn_nax compiled after lfm2.metal's text (its constants and merge) with nax.h inlined.
fn attnNaxLoad(device: mtl.Device, out: *Kernels) !void {
    const a = std.heap.c_allocator;
    const attn = try frags.source(device, a, sources.lfm2_attn_nax);
    defer a.free(attn);
    const src = try std.mem.concat(a, u8, &.{ sources.lfm2, "\n", attn });
    defer a.free(src);
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    out.attn_nax = try mtl.Pipeline.init(device, lib, "lfm2_attn_nax", false);
}

/// core/affine_mm compiled at 4, 6 and 8 bits (group 64) with mlx-lm's SwiGLU, for prompt calls on the tensor units.
fn naxLoad(device: mtl.Device) ![tile.bit_widths.len]am.Pipes {
    const a = std.heap.c_allocator;
    var set: [tile.bit_widths.len]am.Pipes = undefined;
    var made: usize = 0; // pipelines made, widths in order: a failure part way through a width releases these too
    errdefer for (0..made) |n| set[n / am.names.len][n % am.names.len].deinit();
    for (tile.bit_widths, 0..) |bits, i| {
        const src = try am.source(device, a, .{ .bits = bits, .group = 64, .glu = .plain });
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        defer lib.deinit();
        for (am.names, 0..) |name, j| {
            set[i][j] = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
    }
    return set;
}
