//! Unlimited-OCR-2's Metal kernels under the contract: the tower's and decoder's rows, the data moves, attention, the
//! router and MoE gate, and the one-token projections. Each group is a file of its own (metal_uocr_*.zig).
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const sources = @import("kernel_sources");

const c = contract.spec;
const g = contract.gates;
const groups = .{ @import("metal_uocr_rows.zig"), @import("metal_uocr_moves.zig"), @import("metal_uocr_attention.zig"), @import("metal_uocr_route.zig"), @import("metal_uocr_matvec.zig") };
const fields = .{ "rows", "moves", "attention", "route", "matvec" };

/// A launch's two texts: this GPU's and the pre-M5 one (the family's sources include no nax.h, so they are equal).
pub const Texts = struct { here: []const u8, pre_m5: []const u8 };

/// The family's two libraries: the tower's own steps (uv_*) and the decoder's (ud_*).
pub const Library = enum { vision, decoder };

pub fn source(which: Library) []const u8 {
    return switch (which) {
        .vision => sources.uocr,
        .decoder => sources.uocr_decoder,
    };
}

/// One of the family's kernels, compiled as families/uocr/kernels.zig compiles it (MLX's options).
pub fn pipeline(device: mtl.Device, which: Library, name: []const u8) !mtl.Pipeline {
    const lib = try mtl.Library.fromSource(device, source(which), mtl.CompileOptions.mlx());
    defer lib.deinit();
    return mtl.Pipeline.init(device, lib, name, false);
}

/// The served widths (the release's config): the decoder's residual, heads and head width, the router's width, the
/// experts and their widths, the dense layer's; the tower's SAM and CLIP widths.
pub const hidden = 1280;
pub const heads = 10;
pub const head_dim = 128;
pub const router_dim = 128;
pub const experts = 64;
pub const topk = 6;
pub const expert_ffn = 896;
pub const shared_ffn = 2 * expert_ffn;
pub const dense_ffn = 6848;
pub const vocab = 129280;
pub const sam_width = 768;
pub const clip_width = 1024;

pub fn noSums(_: c.Shape) usize {
    return 0;
}

/// A value as fp32 then bf16 rounds it (to nearest even, twice): an fp32 step on exact inputs, then the bf16 point.
pub fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

pub const planted = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    rows: groups[0].Suite = undefined,
    moves: groups[1].Suite = undefined,
    attention: groups[2].Suite = undefined,
    route: groups[3].Suite = undefined,
    matvec: groups[4].Suite = undefined,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        inline for (fields) |f| {
            @field(s, f) = .{ .a = s.a, .device = s.device, .queue = s.queue, .kernels = s.kernels };
            try @field(s, f).collect(only);
        }
    }
    pub fn deinit(s: *Suite) void {
        inline for (fields) |f| @field(s, f).deinit();
    }
};

/// Gate 5's texts for a `uocr ` registration: the group whose launches it names compiles it.
pub fn limitsText(a: std.mem.Allocator, device: mtl.Device, name: []const u8) !Texts {
    inline for (groups) |grp| if (try grp.limitsText(a, device, name)) |t| return t;
    return error.UnknownLaunch;
}
