//! H3's runtime library: GEMM with fragments expanded, then the model's sizes and kernel text.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const kernels = @import("h3_kernels");

pub fn available(device: mtl.Device) bool {
    return device.tensorUnits();
}

pub fn library(a: std.mem.Allocator, heads: usize, slots: usize, tiles: usize, defines: []const u8) ![]u8 {
    const source = try std.fmt.allocPrint(a, "#define H3_HEADS {d}\n#define H3_SLOTS {d}\n#define H3_TILES {d}\n{s}\n{s}", .{ heads, slots, tiles, defines, kernels.source });
    defer a.free(source);
    const gemm = try std.mem.replaceOwned(u8, a, gemmSource(), "#include \"../nax.h\"", sources.nax);
    defer a.free(gemm);
    return std.mem.concat(a, u8, &.{ gemm, source });
}

fn gemmSource() []const u8 {
    inline for (sources.prefill) |f| if (comptime std.mem.eql(u8, f.name, "gemm_nax")) return f.text;
    @compileError("no gemm_nax source");
}
