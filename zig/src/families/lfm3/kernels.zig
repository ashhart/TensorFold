//! LFM3's BF16 family operators and the shared MLX affine row projections.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const dual = @import("dual_projection.zig");
const row = @import("row_projection.zig");

pub const chunk_keys = 4096;
pub const max_chunks = 64;
pub const QkArgs = extern struct { len: u32, cap: u32, eps: f32, log2_theta: f32, rotary_pairs: u32 };
pub const AttentionArgs = extern struct { len: u32, cap: u32, scale: f32, window: u32, chunk_base: u32 = 0 };

pub const AttentionSegment = extern struct { keys: u64, values: u64, base: u32, rows: u32, args: AttentionArgs, pad: u32 = 0 };

pub const ConvSegment = extern struct { state: u64, base: u32, rows: u32 };
comptime {
    std.debug.assert(@sizeOf(ConvSegment) == 16 and @alignOf(ConvSegment) == 8);
    std.debug.assert(@offsetOf(ConvSegment, "state") == 0 and @offsetOf(ConvSegment, "base") == 8);
    std.debug.assert(@offsetOf(ConvSegment, "rows") == 12);
}

pub const names = [_][:0]const u8{
    "lfm3_embed_b4",      "lfm3_embed_b6",            "lfm3_embed_b8", "lfm3_norm", "lfm3_post_add", "lfm3_add",    "lfm3_post_norm",
    "lfm3_attn_segments", "lfm3_attn_segments_merge", "lfm3_conv",     "lfm3_qk",   "lfm3_gate",     "lfm3_swiglu", "lfm3_attn_partial",
    "lfm3_attn_merge",    "lfm3_conv_segments",
};

pub const Kernels = struct {
    pipelines: [names.len]mtl.Pipeline,
    rows: row.Pipelines,
    dual: dual.Pipelines,

    pub fn get(self: *const Kernels, comptime key: []const u8) mtl.Pipeline {
        inline for (names, 0..) |name, i| if (comptime std.mem.eql(u8, name, key)) return self.pipelines[i];
        @compileError("no LFM3 kernel " ++ key);
    }

    pub fn embed(self: *const Kernels, bits: u8) mtl.Pipeline {
        return switch (bits) {
            4 => self.get("lfm3_embed_b4"),
            6 => self.get("lfm3_embed_b6"),
            8 => self.get("lfm3_embed_b8"),
            else => unreachable,
        };
    }

    pub fn deinit(self: *Kernels) void {
        for (self.pipelines) |pipeline| pipeline.deinit();
        self.rows.deinit();
        self.dual.deinit();
    }
};

pub fn load(device: mtl.Device) !Kernels {
    const gpa = std.heap.c_allocator;
    const source = try std.mem.concat(gpa, u8, &.{ if (@import("lfm3_options").risk_fast) "#define L3_RISK_FAST 1\n" else "", sources.lfm3, "\n", sources.lfm3_attention });
    defer gpa.free(source);
    const library = try mtl.Library.fromSource(device, source, mtl.CompileOptions.mlx());
    defer library.deinit();
    var out: Kernels = undefined;
    var loaded: usize = 0;
    errdefer for (out.pipelines[0..loaded]) |pipeline| pipeline.deinit();
    for (names, 0..) |name, i| {
        out.pipelines[i] = try mtl.Pipeline.init(device, library, name, false);
        loaded += 1;
    }
    out.rows = try row.Pipelines.load(device);
    errdefer out.rows.deinit();
    out.dual = try dual.Pipelines.load(device);
    return out;
}
