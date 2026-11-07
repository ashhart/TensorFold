//! Gemma 4 dense pipelines: the glue compiled once for sliding layers and once for global ones, MLX's compile options.
const std = @import("std");
const mtl = @import("metal");
const glue = @import("kernel_sources").gemma4_glue;
const prompt_attention = @import("kernel_sources").gemma4_prompt_attention;
const ple_source = @import("kernel_sources").gemma4_ple;
const dense_source = @import("kernel_sources").gemma4_dense;
/// Simdgroups the per-layer gate splits its K over (dense.metal).
pub const gate_ksplit = 5;
/// Rows a gate threadgroup takes; the grid's second axis counts the rows' blocks.
pub const gate_rows = 32;
const frags = @import("../../core/frags.zig");
const cfg = @import("config.zig");

/// The exact glue specialization used by the target and assistant constructors.
pub fn glueSource(a: std.mem.Allocator, geo: cfg.Geometry, global: bool, q_only: bool, embed_bits: u8) ![]u8 {
    if (std.mem.indexOfScalar(u8, &cfg.widths.bit_widths, embed_bits) == null) return error.UnsupportedGemmaQuantization;
    const dh: usize = if (global) cfg.global_dim else cfg.sliding_dim;
    const nk: usize = if (global) geo.global_kv_heads else geo.sliding_kv_heads;
    const shape: [3]usize = if (global) .{ 64, 1, 4 } else .{ 128, 4, 4 };
    const vk = global and geo.keys_are_values; // the 31B's and 12B's global values are their keys; the E4B's have a v_proj
    const width = geo.query_heads * dh + (if (q_only) 0 else nk * dh * @as(usize, if (vk) 1 else 2));
    return std.fmt.allocPrint(a,
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
        \\#define EB {d}
        \\{s}
    , .{ geo.hidden, dh, geo.query_heads, nk, @intFromBool(vk), width, nk, geo.query_heads / nk, shape[0], shape[1], shape[2], embed_bits, glue });
}

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
    draft_store: mtl.Pipeline,
    draft_add: mtl.Pipeline,
    draft_swiglu: mtl.Pipeline,
    /// gemma_assist_input: the MTP assistant's rows, the pending token's embedding beside a hidden row.
    assist_input: mtl.Pipeline,
    /// gemma_prompt_attention: a prompt chunk's rows over the cache, on the tensor units.
    prompt_attn: mtl.Pipeline,
    /// gemma_decode_attention: decode rows' partials on the tensor units, a key head's query heads as one tile.
    decode_attn: mtl.Pipeline,
    /// gilby's attend shape by head dim: keys a chunk, simdgroups a query head, keys scored before an update.
    chunk: usize,
    split: usize,
    block: usize,
    heads_per_kv: usize,

    /// `q_only`: gemma_qkv_prep reads q alone (the MTP assistant's layers); `embed_bits`: the embedding table's width.
    pub fn init(a: std.mem.Allocator, device: mtl.Device, geo: cfg.Geometry, global: bool, q_only: bool, embed_bits: u8) !Kind {
        const dh: usize = if (global) cfg.global_dim else cfg.sliding_dim;
        const nk: usize = if (global) geo.global_kv_heads else geo.sliding_kv_heads;
        const shape: [3]usize = if (global) .{ 64, 1, 4 } else .{ 128, 4, 4 };
        const text = try glueSource(a, geo, global, q_only, embed_bits);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var k: Kind = undefined;
        inline for (.{ "embed", "rms", "qkv", "partial", "merge", "attn_tail", "dense_tail", "geglu", "softcap", "freqs", "kv_write", "draft_store", "draft_add", "draft_swiglu", "assist_input" }, .{ "gemma_embed", "gemma_rms_looped", "gemma_qkv_prep", "gemma_attention_partial", "gemma_attention_merge", "gemma_attn_tail", "gemma_dense_tail", "gemma_geglu", "gemma_softcap", "gemma_rope_freqs", "gemma_kv_write", "gemma_draft_store", "gemma_draft_add", "gemma_draft_swiglu", "gemma_assist_input" }) |field, name|
            @field(k, field) = try mtl.Pipeline.init(device, lib, name, false);
        {
            const body = try frags.source(device, a, prompt_attention);
            defer a.free(body);
            const src = try std.fmt.allocPrint(a, "#define DH {d}\n#define NQ {d}\n#define G {d}\n#define SL {d}\n#define WINDOW {d}\n#define RING {d}\n{s}", .{ dh, geo.query_heads, geo.query_heads / nk, dh / 128, @as(usize, if (global) 0 else geo.sliding_window), @as(usize, if (global) 0 else @import("state.zig").ringOf(geo)), body });
            defer a.free(src);
            const plib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
            defer plib.deinit();
            k.prompt_attn = try mtl.Pipeline.init(device, plib, "gemma_prompt_attention", false);
            k.decode_attn = try mtl.Pipeline.init(device, plib, "gemma_decode_attention", false);
        }
        k.chunk = shape[0];
        k.split = shape[1];
        k.block = shape[2];
        k.heads_per_kv = geo.query_heads / nk;
        return k;
    }

    pub fn deinit(k: Kind) void {
        inline for (.{ "embed", "rms", "qkv", "partial", "merge", "attn_tail", "dense_tail", "geglu", "softcap", "freqs", "kv_write", "draft_store", "draft_add", "draft_swiglu", "assist_input", "prompt_attn", "decode_attn" }) |field| @field(k, field).deinit();
    }
};

/// The E-series' per-layer input pipelines (gemma4/ple.metal, and the bf16 linears of gemma4/dense.metal).
pub const Ple = struct {
    embed: mtl.Pipeline,
    combine: mtl.Pipeline,
    dense: mtl.Pipeline,
    /// The per-layer gate, gelu(gate) x the per-layer input as its epilogue.
    gate: mtl.Pipeline,

    /// ple.metal with the geometry's constants (the contract compiles the same text).
    pub fn source(a: std.mem.Allocator, geo: cfg.Geometry) ![]u8 {
        return std.fmt.allocPrint(a,
            \\#define D {d}
            \\#define P {d}
            \\#define L {d}
            \\#define T {d}
            \\#define GELU_C0 0.7978845608028654f
            \\#define GELU_C1 0.044715f
            \\{s}
        , .{ geo.hidden, geo.layer_input, geo.layers, geo.layer_input, ple_source });
    }

    /// dense.metal: the bf16 per-layer linears and the gate with its gelu epilogue.
    pub const dense_text = "#define GATE_KS " ++ std.fmt.comptimePrint("{d}", .{gate_ksplit}) ++ "\n#define GATE_ROWS " ++ std.fmt.comptimePrint("{d}", .{gate_rows}) ++ "\n" ++ dense_source;

    pub fn init(a: std.mem.Allocator, device: mtl.Device, geo: cfg.Geometry) !Ple {
        const text = try source(a, geo);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var k: Ple = undefined;
        inline for (.{ "embed", "combine" }, .{ "gemma_ple_embed", "gemma_ple_combine" }) |field, name|
            @field(k, field) = try mtl.Pipeline.init(device, lib, name, false);
        const dl = try mtl.Library.fromSource(device, dense_text, mtl.CompileOptions.mlx());
        defer dl.deinit();
        k.dense = try mtl.Pipeline.init(device, dl, "gemma_dense4", false); // the bf16 per-layer linears
        k.gate = try mtl.Pipeline.init(device, dl, "gemma_dense4_gate", false);
        return k;
    }

    pub fn deinit(k: Ple) void {
        inline for (.{ "embed", "combine", "dense", "gate" }) |field| @field(k, field).deinit();
    }
};

pub const Kernels = struct {
    sliding: Kind,
    global: Kind,
    /// The E-series' layers that read an earlier layer's cache: q alone, a sliding and a global kind.
    shared: ?[2]Kind = null,
    ple: ?Ple = null,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, geo: cfg.Geometry, embed_bits: u8) !Kernels {
        const s = try Kind.init(a, device, geo, false, false, embed_bits);
        errdefer s.deinit();
        const g = try Kind.init(a, device, geo, true, false, embed_bits);
        errdefer g.deinit();
        if (geo.shared_layers == 0 and geo.layer_input == 0) return .{ .sliding = s, .global = g };
        const qs = try Kind.init(a, device, geo, false, true, embed_bits);
        errdefer qs.deinit();
        const qg = try Kind.init(a, device, geo, true, true, embed_bits);
        errdefer qg.deinit();
        return .{ .sliding = s, .global = g, .shared = .{ qs, qg }, .ple = try Ple.init(a, device, geo) };
    }

    pub fn kind(k: *const Kernels, global: bool) *const Kind {
        return if (global) &k.global else &k.sliding;
    }

    /// Layer `i`'s pipelines in `geo`: the q-only ones on a layer that reads another's cache.
    pub fn of(k: *const Kernels, geo: cfg.Geometry, i: usize) *const Kind {
        if (k.shared) |*q| if (i >= geo.cacheLayers()) return &q[@intFromBool(cfg.global(i))];
        return k.kind(cfg.global(i));
    }

    pub fn deinit(k: Kernels) void {
        k.sliding.deinit();
        k.global.deinit();
        if (k.shared) |q| for (q) |x| x.deinit();
        if (k.ple) |p| p.deinit();
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
