//! Unlimited-OCR-2's own CUDA kernels (uocr.cu): the Metal family's steps beside cuBLASLt's projections.
const std = @import("std");
const cuda = @import("cuda");

const names = [_][:0]const u8{
    "tf_uocr_im2col",
    "tf_uocr_add",
    "tf_uocr_layer_norm",
    "tf_uocr_layer_norm2d",
    "tf_uocr_gelu",
    "tf_uocr_quick_gelu",
    "tf_uocr_window",
    "tf_uocr_unwindow_add",
    "tf_uocr_rel_pos",
    "tf_uocr_attention",
    "tf_uocr_concat",
    "tf_uocr_clip_embed",
    "tf_uocr_embed",
    "tf_uocr_rms_norm",
    "tf_uocr_rope",
    "tf_uocr_router_norm",
    "tf_uocr_block_keys",
    "tf_uocr_route",
    "tf_uocr_attend",
    "tf_uocr_silu_mul",
    "tf_uocr_moe_gate",
    "tf_uocr_gather",
    "tf_uocr_scatter",
    "tf_uocr_moe_combine",
    "tf_uocr_matvec",
    "tf_uocr_expert_gu",
    "tf_uocr_expert_down",
    "tf_uocr_argmax",
};

pub const Kernels = struct {
    mod: cuda.Module,
    f: [names.len]cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kernels = .{ .mod = try cuda.Module.load(d, cuda.kernels.uocr), .f = undefined };
        errdefer k.mod.unload();
        for (names, &k.f) |n, *f| f.* = try k.mod.function(n);
        return k;
    }

    pub fn unload(k: *Kernels) void {
        k.mod.unload();
    }

    fn get(k: *const Kernels, comptime name: []const u8) cuda.Function {
        inline for (names, 0..) |n, i| if (comptime std.mem.eql(u8, n, "tf_uocr_" ++ name)) return k.f[i];
        @compileError("no kernel tf_uocr_" ++ name);
    }

    /// Launches kernel `name` on `s` over `grid` blocks of `threads`, its arguments in order (u64 addresses, usize
    /// as c_int, i64, f32, extern structs).
    pub fn run(k: *const Kernels, comptime name: []const u8, s: cuda.Stream, grid: [3]usize, threads: u32, args: anytype) !void {
        if (grid[0] == 0 or grid[1] == 0 or grid[2] == 0) return;
        var a: cuda.Args = .{};
        inline for (args) |v| {
            if (@TypeOf(v) == usize or @TypeOf(v) == comptime_int) a.add(@as(c_int, @intCast(v))) else a.add(v);
        }
        try cuda.launch.launch(k.get(name), .{ .grid = .{ .x = @intCast(grid[0]), .y = @intCast(grid[1]), .z = @intCast(grid[2]) }, .block = .{ .x = threads } }, s, &a);
    }
};

/// Blocks of 256 over `n` values.
pub fn blocks(n: usize) usize {
    return (n + 255) / 256;
}

test "uocr cuda: launch geometry" {
    try std.testing.expectEqual(@as(usize, 5), blocks(1280));
    try std.testing.expectEqual(@as(usize, 1), blocks(1));
}
