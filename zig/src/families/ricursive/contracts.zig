//! The family's Metal kernels under the kernel contract: their sources as model.zig compiles them, and the roll call.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const cfg = @import("config.zig");
const model = @import("model.zig");
const st = @import("state.zig");
const frags = @import("../../core/frags.zig");

/// Ripple-65B, the family's one qualified checkpoint.
pub const ripple = cfg.qualified[0];
/// The engine's sliding ring for a stream past it (state.ring at any long capacity).
pub const sliding_ring = std.mem.alignForward(usize, ripple.window - 1 + st.prompt_rows, st.ring_chunk);

pub const glue_names = model.glue_names;
pub const llama_names = model.llama_names;
pub const gemm_names = [_][]const u8{ "ll_gemm_00", "ll_gemm_01", "ll_gemm_10", "ll_gemm_11" };
pub const attention_name = "rc_prompt_attention";
pub const sample_name = "tf_sample_full";
pub const fp4_names = model.fp4_names;
pub const fp4_gemm_names = [_][]const u8{ "rc_fp4_gemm_00", "rc_fp4_gemm_01", "rc_fp4_gemm_10", "rc_fp4_gemm_11" };
/// Every pipeline model.zig builds: glue, llama's row kernels and GEMMs, attention, the draw and the NVFP4 MLP's.
pub const pipelines = glue_names.len + llama_names.len + gemm_names.len + 2 + fp4_names.len + fp4_gemm_names.len;

/// The glue's text as model.zig builds it.
pub fn glueSource(a: std.mem.Allocator) ![]u8 {
    return model.glueSource(a, "#define RC_DH {d}\n{s}", ks.ricursive_glue);
}

/// The glue library as model.zig compiles it: precise math functions.
pub fn glueLibrary(device: mtl.Device, a: std.mem.Allocator) !mtl.Library {
    const src = try glueSource(a);
    defer a.free(src);
    return mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
}

/// The prompt attention's header as model.zig writes it for Ripple.
pub fn attentionHeader(a: std.mem.Allocator) ![]u8 {
    const fmt = "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define KVH {d}\n#define SCALE {e}f\n";
    return std.fmt.allocPrint(a, fmt, .{ cfg.head_dim, ripple.heads, ripple.heads / ripple.kv_heads, ripple.kv_heads, cfg.headScale() });
}

/// The prompt attention's text as model.zig builds it (nax.h inlined for this GPU).
pub fn attentionSource(device: mtl.Device, a: std.mem.Allocator) ![]u8 {
    const head = try attentionHeader(a);
    defer a.free(head);
    const body = try frags.source(device, a, ks.ricursive_attention_nax);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ head, body });
}

/// The NVFP4 row kernels' library as model.zig compiles it: precise math functions.
pub fn fp4Library(device: mtl.Device, src: []const u8) !mtl.Library {
    return mtl.Library.fromSource(device, src, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
}

/// The NVFP4 GEMM's text: nax.h inlined for this GPU (model.zig's build), or as a GPU without tensor units takes it.
pub fn fp4GemmSource(device: mtl.Device, a: std.mem.Allocator, units: bool) ![]u8 {
    if (units) return frags.source(device, a, ks.ricursive_fp4_gemm);
    return frags.precision.source(a, false, ks.ricursive_fp4_gemm, ks.nax, false);
}

/// Every pipeline the family runs appears among the registrations' functions.
pub fn requireComplete(functions: []const []const u8) !void {
    inline for (.{ glue_names, llama_names, gemm_names, [_][]const u8{ attention_name, sample_name }, fp4_names, fp4_gemm_names }) |list| for (list) |name| {
        var seen = false;
        for (functions) |f| if (std.mem.eql(u8, f, name)) {
            seen = true;
        };
        if (!seen) return error.IncompleteRicursiveContract;
    };
}

test "ricursive contracts: the roll call wants every pipeline" {
    var all: [pipelines][]const u8 = undefined;
    var i: usize = 0;
    inline for (.{ glue_names, llama_names, gemm_names, [_][]const u8{ attention_name, sample_name }, fp4_names, fp4_gemm_names }) |list| for (list) |name| {
        all[i] = name;
        i += 1;
    };
    try requireComplete(&all);
    try std.testing.expectError(error.IncompleteRicursiveContract, requireComplete(all[1..]));
    try std.testing.expectEqual(@as(usize, 3072), sliding_ring);
}
