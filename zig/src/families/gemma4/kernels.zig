//! Gemma 4 dense pipelines: the glue compiled once for sliding layers and once for global ones, MLX's compile options.
const std = @import("std");
const mtl = @import("metal");
const glue = @import("kernel_sources").gemma4_glue;
const prompt_attention = @import("kernel_sources").gemma4_prompt_attention;
const frags = @import("../../core/frags.zig");
const cfg = @import("config.zig");

/// One layer kind's pipelines; the kind-free ones (embed, norms, tails, geglu, softcap) are the same in both.
pub const Kind = struct {
    embed: mtl.Pipeline,
    rms: mtl.Pipeline,
    qkv: mtl.Pipeline,
    partial: mtl.Pipeline,
    merge: mtl.Pipeline,
    attn_tail: mtl.Pipeline,
    dense_tail: mtl.Pipeline,
    geglu: mtl.Pipeline,
    softcap: mtl.Pipeline,
    freqs: mtl.Pipeline,
    kv_write: mtl.Pipeline,
    /// gemma_prompt_attention: a prompt chunk's rows over the cache, on the tensor units.
    prompt_attn: mtl.Pipeline,
    /// gilby's attend shape by head dim: keys a chunk, simdgroups a query head, keys scored before an update.
    chunk: usize,
    split: usize,
    block: usize,
    heads_per_kv: usize,

    fn init(a: std.mem.Allocator, device: mtl.Device, global: bool) !Kind {
        const dh: usize = if (global) cfg.global_dim else cfg.sliding_dim;
        const nk: usize = if (global) cfg.global_kv_heads else cfg.sliding_kv_heads;
        const shape: [3]usize = if (global) .{ 64, 1, 4 } else .{ 128, 4, 4 };
        const width = cfg.query_heads * dh + nk * dh * @as(usize, if (global) 1 else 2);
        const text = try std.fmt.allocPrint(a,
            \\#define D {d}
            \\#define T 256
            \\#define DH {d}
            \\#define NQ {d}
            \\#define NK {d}
            \\#define VK {d}
            \\#define W {d}
            \\#define HK {d}
            \\#define G {d}
            \\#define CK {d}
            \\#define S {d}
            \\#define BLK {d}
            \\#define SCALE 1.0f
            \\#define GELU_C0 0.7978845608028654f
            \\#define GELU_C1 0.044715f
            \\{s}
        , .{ cfg.hidden, dh, cfg.query_heads, nk, @intFromBool(global), width, nk, cfg.query_heads / nk, shape[0], shape[1], shape[2], glue });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var k: Kind = undefined;
        inline for (.{ "embed", "rms", "qkv", "partial", "merge", "attn_tail", "dense_tail", "geglu", "softcap", "freqs", "kv_write" }, .{ "gemma_embed", "gemma_rms_looped", "gemma_qkv_prep", "gemma_attention_partial", "gemma_attention_merge", "gemma_attn_tail", "gemma_dense_tail", "gemma_geglu", "gemma_softcap", "gemma_rope_freqs", "gemma_kv_write" }) |field, name|
            @field(k, field) = try mtl.Pipeline.init(device, lib, name, false);
        {
            const body = try frags.source(device, a, prompt_attention);
            defer a.free(body);
            const src = try std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define SL {d}\n#define WINDOW {d}\n#define RING {d}\n{s}", .{ dh, cfg.query_heads, cfg.query_heads / nk, dh / 128, @as(usize, if (global) 0 else cfg.window), @as(usize, if (global) 0 else @import("state.zig").ring), body });
            defer a.free(src);
            const plib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
            defer plib.deinit();
            k.prompt_attn = try mtl.Pipeline.init(device, plib, "gemma_prompt_attention", false);
        }
        k.chunk = shape[0];
        k.split = shape[1];
        k.block = shape[2];
        k.heads_per_kv = cfg.query_heads / nk;
        return k;
    }

    fn deinit(k: Kind) void {
        inline for (.{ "embed", "rms", "qkv", "partial", "merge", "attn_tail", "dense_tail", "geglu", "softcap", "freqs", "kv_write", "prompt_attn" }) |field| @field(k, field).deinit();
    }
};

pub const Kernels = struct {
    sliding: Kind,
    global: Kind,

    pub fn init(a: std.mem.Allocator, device: mtl.Device) !Kernels {
        const s = try Kind.init(a, device, false);
        errdefer s.deinit();
        return .{ .sliding = s, .global = try Kind.init(a, device, true) };
    }

    pub fn kind(k: *const Kernels, global: bool) *const Kind {
        return if (global) &k.global else &k.sliding;
    }

    pub fn deinit(k: Kernels) void {
        k.sliding.deinit();
        k.global.deinit();
    }
};

/// RoPE inverse frequencies as gilby takes them from mlx_lm: sliding ones in f64; global ones from MLX's GPU pow.
pub fn inverseFrequencies(k: *const Kernels, queue: mtl.Queue, device: mtl.Device, global: bool, out: []f32) !void {
    if (!global) return cfg.inverseFrequencies(false, out);
    const b = try device.buffer(cfg.global_rotated * 4, mtl.ResourceOptions.shared);
    defer b.deinit();
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(k.global.freqs);
    e.setBuffer(b, 0, 0);
    e.setValue(@as(f32, @floatCast(cfg.global_theta)), 1);
    e.setValue(@as(u32, cfg.global_dim), 2);
    e.dispatchThreads(.{ .width = cfg.global_rotated }, .{ .width = cfg.global_rotated });
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
    @memset(out, 0);
    for (b.slice(f32, cfg.global_rotated), out[0..cfg.global_rotated]) |f, *o| o.* = @floatCast(1.0 / @as(f64, f));
}
