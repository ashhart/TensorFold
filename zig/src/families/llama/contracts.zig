//! The family's Metal kernels under the kernel contract: their sources as the loader compiles them, and the roll call.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const model = @import("model.zig");
const frags = @import("../../core/frags.zig");

/// A served geometry the attention compiles for (the head size is the family's 128).
pub const Geometry = struct { name: []const u8, heads: u32, kv_heads: u32, hidden: usize, intermediate: usize, vocab: usize };
/// Nanbeige4.3-33B, the family's first model, and Nanbeige4.1-3B, its drafter.
pub const nanbeige43: Geometry = .{ .name = "33B", .heads = 56, .kv_heads = 8, .hidden = 7168, .intermediate = 22016, .vocab = 166912 };
pub const nanbeige41: Geometry = .{ .name = "3B", .heads = 20, .kv_heads = 4, .hidden = 2560, .intermediate = 10496, .vocab = 166144 };
pub const geometries = [_]Geometry{ nanbeige43, nanbeige41 };

pub const glue_names = model.glue_names;
pub const gemm_names = [_][]const u8{ "ll_gemm_00", "ll_gemm_01", "ll_gemm_10", "ll_gemm_11" };
pub const attention_name = "ll_prompt_attention";
/// Every pipeline the family runs: the glue, the four GEMM forms and the prompt attention.
pub const pipelines = glue_names.len + gemm_names.len + 1;

/// The glue's text as model.zig builds it.
pub fn glueSource(a: std.mem.Allocator) ![]u8 {
    return std.fmt.allocPrint(a, "#define LL_DH {d}\n{s}", .{ cfg.head_dim, ks.llama_glue });
}

/// The glue library as model.zig compiles it: precise math functions.
pub fn glueLibrary(device: mtl.Device, a: std.mem.Allocator) !mtl.Library {
    const src = try glueSource(a);
    defer a.free(src);
    return mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
}

/// The prompt attention's text for a geometry, as model.zig builds it (nax.h inlined for this GPU).
pub fn attentionSource(device: mtl.Device, a: std.mem.Allocator, g: Geometry) ![]u8 {
    const body = try frags.source(device, a, ks.llama_attention_nax);
    defer a.free(body);
    return std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define SCALE {e}f\n{s}", .{ cfg.head_dim, g.heads, g.heads / g.kv_heads, g.kv_heads, cfg.headScale(), body });
}

/// The GEMM forms' text as model.zig builds it.
pub fn gemmSource(device: mtl.Device, a: std.mem.Allocator) ![]u8 {
    return frags.source(device, a, ks.llama_gemm);
}

/// Every pipeline the family runs appears among the registrations' functions.
pub fn requireComplete(functions: []const []const u8) !void {
    inline for (.{ glue_names, gemm_names, [_][]const u8{attention_name} }) |list| for (list) |name| {
        var seen = false;
        for (functions) |f| if (std.mem.eql(u8, f, name)) {
            seen = true;
        };
        if (!seen) return error.IncompleteLlamaContract;
    };
}

test "llama contracts: the roll call wants every pipeline" {
    var all: [pipelines][]const u8 = undefined;
    var i: usize = 0;
    inline for (.{ glue_names, gemm_names, [_][]const u8{attention_name} }) |list| for (list) |name| {
        all[i] = name;
        i += 1;
    };
    try requireComplete(&all);
    try std.testing.expectError(error.IncompleteLlamaContract, requireComplete(all[1..]));
}
