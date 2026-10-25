//! Cosmos3's pipelines: the family's glue (cosmos3.metal), and core/tile_qmm's prompt tiles and their decode twin.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const tile = @import("../../core/tile_qmm.zig");
const am = @import("../../core/affine_mm.zig");

const names = [_][:0]const u8{
    "cos3_embed_b8",
    "cos3_place",
    "cos3_norm",
    "cos3_qk",
    "cos3_relu2",
    "cos3_attn_128",
    "cos3_flash_128",
    "cos3_flash_72",
    "cos3_layer_norm",
    "cos3_bias",
    "cos3_add",
};

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    tiles: tile.Pipelines,
    /// Whether the twin equals the tiles at every shape (checked at load); otherwise decode rows take the tiles too.
    twin: bool = false,
    /// The tensor-unit prompt kernels at each of the tiles' widths (M5 and later; null before, every prompt row tiles).
    nax: ?[tile.bit_widths.len]am.Pipes = null,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, key)) return self.pipelines[i];
        @compileError("no Cosmos3 kernel " ++ key);
    }

    /// The tensor-unit pipelines for a width the tiles take, when this GPU has them.
    pub fn naxPipes(self: *const Kernels, bits: u8) ?am.Pipes {
        const set = self.nax orelse return null;
        return set[std.mem.indexOfScalar(u8, &tile.bit_widths, bits) orelse return null];
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |p| p.deinit();
        if (self.nax) |set| for (set) |pipes| for (pipes) |p| p.deinit();
        self.tiles.deinit();
    }
};

pub fn load(device: mtl.Device) !Kernels {
    var out: Kernels = undefined;
    const lib = try mtl.Library.fromSource(device, sources.cosmos3, mtl.CompileOptions.mlx());
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
    out.nax = if (device.tensorUnits() and std.c.getenv("TF_COSMOS3_NO_NAX") == null) try naxLoad(device) else null;
    return out;
}

/// core/affine_mm compiled at 4, 6 and 8 bits (group 64), for prompt calls on the tensor units.
fn naxLoad(device: mtl.Device) ![tile.bit_widths.len]am.Pipes {
    const a = std.heap.c_allocator;
    var set: [tile.bit_widths.len]am.Pipes = undefined;
    var made: usize = 0;
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
