//! Config parsing, Metal pipeline compilation, and TP-sharded safetensors loader for incoai/GLM-5.3-DFlash2.
const std = @import("std");
const Io = std.Io;
const mtl = @import("metal");
const sources = @import("kernel_sources");
const st = @import("../../core/safetensors.zig");
const cfg = @import("config.zig");
const gpu = @import("gpu.zig");
const ane_mod = @import("dflash2_ane.zig");

pub const max_layers: u32 = 6;
pub const block_size: u32 = 8;
pub const draft_tokens: u32 = block_size - 1;
pub const num_taps: u32 = 6;

pub const Config = struct {
    layers: u32 = 6,
    hidden: u32 = 6144,
    ffn: u32 = 12288,
    heads: u32 = 64,
    kv_heads: u32 = 8,
    head_dim: u32 = 128,
    vocab: u32 = 154880,
    window: u32 = 2048,
    block: u32 = 8,
    mask: u32 = 154856,
    rank: u32 = 256,
    top_k: u32 = 16,
    group: u32 = 16,
    taps: [num_taps]u32 = .{ 5, 19, 33, 47, 61, 75 },
    eps: f32 = 1e-5,
    theta: f32 = 1_000_000.0,

    pub fn ring(c: *const Config) u32 {
        return std.mem.alignForward(u32, c.window + c.block, 128);
    }

    pub fn parse(gpa: std.mem.Allocator, text: []const u8) !Config {
        var doc = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
        defer doc.deinit();
        const o = doc.value.object;
        const dc = (o.get("dflash_config") orelse return error.BadDFlash2Config).object;
        const get = struct {
            fn u(m: std.json.ObjectMap, k: []const u8) !u32 {
                const v = m.get(k) orelse return error.BadDFlash2Config;
                return if (v == .integer and v.integer >= 0) @intCast(v.integer) else error.BadDFlash2Config;
            }
            fn fl(m: std.json.ObjectMap, k: []const u8) !f32 {
                const v = m.get(k) orelse return error.BadDFlash2Config;
                return switch (v) {
                    .integer => |i| @floatFromInt(i),
                    .float => |x| @floatCast(x),
                    else => error.BadDFlash2Config,
                };
            }
        };
        var c: Config = .{
            .layers = try get.u(o, "num_hidden_layers"),
            .hidden = try get.u(o, "hidden_size"),
            .ffn = try get.u(o, "intermediate_size"),
            .heads = try get.u(o, "num_attention_heads"),
            .kv_heads = try get.u(o, "num_key_value_heads"),
            .head_dim = try get.u(o, "head_dim"),
            .vocab = try get.u(o, "vocab_size"),
            .window = try get.u(o, "sliding_window"),
            .block = try get.u(dc, "block_size"),
            .mask = try get.u(dc, "mask_token_id"),
            .rank = try get.u(dc, "selector_rank"),
            .top_k = try get.u(dc, "selector_top_k"),
            .group = try get.u(dc, "conv_group_size"),
            .eps = try get.fl(o, "rms_norm_eps"),
        };
        if (o.get("rope_parameters")) |rp| if (rp == .object) {
            c.theta = get.fl(rp.object, "rope_theta") catch c.theta;
        };
        const taps = dc.get("target_layer_ids") orelse return error.BadDFlash2Config;
        if (taps != .array or taps.array.items.len != num_taps) return error.UnsupportedDFlash2;
        for (taps.array.items, &c.taps) |x, *y| y.* = @intCast(x.integer);
        if (c.layers > max_layers or c.block != block_size or c.head_dim != 128 or c.rank != 256 or c.top_k != 16) return error.UnsupportedDFlash2;
        return c;
    }
};

pub const Q8 = struct {
    w: mtl.Buffer,
    s: mtl.Buffer,

    pub fn deinit(q: Q8) void {
        q.w.deinit();
        q.s.deinit();
    }

    pub fn bytes(q: Q8) u64 {
        return q.w.length() + q.s.length();
    }
};

const QuantArgs = extern struct { N: u32, K: u32, p0: u32 = 0, p1: u32 = 0 };

pub const LayerW = struct {
    in_norm: mtl.Buffer,
    post_norm: mtl.Buffer,
    q_norm: mtl.Buffer,
    k_norm: mtl.Buffer,
    q_w: Q8,
    k_w: mtl.Buffer,
    v_w: mtl.Buffer,
    k_q8: Q8,
    v_q8: Q8,
    o_w: Q8,
    gate_w: Q8,
    up_w: Q8,
    down_w: Q8,
    aconv_base: mtl.Buffer,
    aconv_proj: Q8,
    mconv_base: mtl.Buffer,
    mconv_proj: Q8,

    pub fn deinit(l: LayerW) void {
        inline for (.{ l.in_norm, l.post_norm, l.q_norm, l.k_norm, l.k_w, l.v_w, l.aconv_base, l.mconv_base }) |b| b.deinit();
        inline for (.{ l.q_w, l.k_q8, l.v_q8, l.o_w, l.gate_w, l.up_w, l.down_w, l.aconv_proj, l.mconv_proj }) |q| q.deinit();
    }
};

pub const SlotCache = struct {
    k: [max_layers]mtl.Buffer,
    v: [max_layers]mtl.Buffer,
    from: u32 = 0,
    end: u32 = 0,
    use_mtp: bool = false,
    df2_rounds: u32 = 0,
    df2_toks: u32 = 0,
    last6: [6]u32 = @splat(0),
    hist_n: u32 = 0,
    hist_pos0: u32 = 0,
    hist_toks: [129]u32 = @splat(0),
};

pub const Pipes = struct {
    tap: mtl.Pipeline,
    gemv: mtl.Pipeline,
    q8_gemv: mtl.Pipeline,
    q8_gemv_q1: mtl.Pipeline,
    q8_gemv_q2_xh: mtl.Pipeline,
    q8_gu_silu: mtl.Pipeline,
    quant_q8: mtl.Pipeline,
    gemm: mtl.Pipeline,
    norm: mtl.Pipeline,
    conv: mtl.Pipeline,
    conv_add_rms: mtl.Pipeline,
    prep: mtl.Pipeline,
    attend: mtl.Pipeline,
    merge: mtl.Pipeline,
    accept: mtl.Pipeline,
    tree_accept: mtl.Pipeline,
    lm_q4_topk: mtl.Pipeline,
    lm_q8_topk: mtl.Pipeline,
    lm_topk: mtl.Pipeline,
    topk_red: mtl.Pipeline,
    walk: mtl.Pipeline,
    compact_taps: mtl.Pipeline,
    compact_taps_indirect: mtl.Pipeline,
    ane_pack: mtl.Pipeline,
    ane_unpack: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipes {
        const opts: mtl.CompileOptions = .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise };
        const lib = try mtl.Library.fromSource(device, sources.glm_dsa_dflash2, opts);
        defer lib.deinit();
        const hlib = try mtl.Library.fromSource(device, sources.glm_dsa_dflash2_head, opts);
        defer hlib.deinit();
        return .{
            .tap = try mtl.Pipeline.init(device, lib, "df2_tap_capture", false),
            .gemv = try mtl.Pipeline.init(device, lib, "df2_bf16_gemv_n", false),
            .q8_gemv = try mtl.Pipeline.init(device, lib, "df2_q8_gemv_n", false),
            .q8_gemv_q1 = try mtl.Pipeline.init(device, lib, "df2_q8_gemv_q1", false),
            .q8_gemv_q2_xh = try mtl.Pipeline.init(device, lib, "df2_q8_gemv_q2_xh", false),
            .q8_gu_silu = try mtl.Pipeline.init(device, lib, "df2_q8_gu_silu_n", false),
            .quant_q8 = try mtl.Pipeline.init(device, lib, "df2_quant_q8_g64", false),
            .gemm = try mtl.Pipeline.init(device, lib, "df2_bf16_gemm", false),
            .norm = try mtl.Pipeline.init(device, lib, "df2_rms_norm", false),
            .conv = try mtl.Pipeline.init(device, lib, "df2_conv", false),
            .conv_add_rms = try mtl.Pipeline.init(device, lib, "df2_conv_add_rms", false),
            .prep = try mtl.Pipeline.init(device, lib, "df2_prep_qkv", false),
            .attend = try mtl.Pipeline.init(device, lib, "df2_attend_split", false),
            .merge = try mtl.Pipeline.init(device, lib, "df2_attend_merge", false),
            .accept = try mtl.Pipeline.init(device, hlib, "df2_accept_setup", false),
            .tree_accept = try mtl.Pipeline.init(device, hlib, "df2_tree_accept", false),
            .lm_q4_topk = try mtl.Pipeline.init(device, hlib, "df2_lm_q4_mma_7", false),
            .lm_q8_topk = try mtl.Pipeline.init(device, hlib, "df2_lm_q8_mma_7", false),
            .lm_topk = try mtl.Pipeline.init(device, hlib, "df2_lm_bf16_topk16", false),
            .topk_red = try mtl.Pipeline.init(device, hlib, "df2_topk16_reduce", false),
            .walk = try mtl.Pipeline.init(device, hlib, "df2_merge_walk", false),
            .compact_taps = try mtl.Pipeline.init(device, hlib, "df2_compact_taps", false),
            .compact_taps_indirect = try mtl.Pipeline.init(device, hlib, "df2_compact_taps_indirect", false),
            .ane_pack = try mtl.Pipeline.init(device, hlib, "df2_ane_pack_in", false),
            .ane_unpack = try mtl.Pipeline.init(device, hlib, "df2_ane_unpack_out", false),
        };
    }

    pub fn deinit(p: Pipes) void {
        inline for (.{ p.tap, p.gemv, p.q8_gemv, p.q8_gemv_q1, p.q8_gemv_q2_xh, p.q8_gu_silu, p.quant_q8, p.gemm, p.norm, p.conv, p.conv_add_rms, p.prep, p.attend, p.merge, p.accept, p.tree_accept, p.lm_q4_topk, p.lm_q8_topk, p.lm_topk, p.topk_red, p.walk, p.compact_taps, p.compact_taps_indirect, p.ane_pack, p.ane_unpack }) |x| x.deinit();
    }
};

fn readExact(io: Io, file: Io.File, dst: []u8, offset: u64) !void {
    var got: usize = 0;
    while (got < dst.len) {
        const n = try file.readPositionalAll(io, dst[got..], offset + got);
        if (n == 0) return error.UnexpectedEof;
        got += n;
    }
}

pub fn loadFull(device: mtl.Device, io: Io, file: Io.File, meta: *const st.Metadata, name: []const u8) !mtl.Buffer {
    const e = meta.names.get(name) orelse return error.MissingDFlash2Tensor;
    const nbytes: usize = @intCast(e.end - e.begin);
    const buf = try device.buffer(nbytes, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    try readExact(io, file, buf.slice(u8, nbytes), meta.data + e.begin);
    return buf;
}

pub fn loadRowSlice(device: mtl.Device, io: Io, file: Io.File, meta: *const st.Metadata, name: []const u8, r0: usize, rn: usize, cols: usize) !mtl.Buffer {
    const e = meta.names.get(name) orelse return error.MissingDFlash2Tensor;
    const nbytes = rn * cols * 2;
    const buf = try device.buffer(nbytes, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    try readExact(io, file, buf.slice(u8, nbytes), meta.data + e.begin + r0 * cols * 2);
    return buf;
}

pub fn loadColSlice(gpa: std.mem.Allocator, device: mtl.Device, io: Io, file: Io.File, meta: *const st.Metadata, name: []const u8, rows: usize, c0: usize, cn: usize, full_cols: usize) !mtl.Buffer {
    const e = meta.names.get(name) orelse return error.MissingDFlash2Tensor;
    const nbytes = rows * cn * 2;
    const buf = try device.buffer(nbytes, mtl.ResourceOptions.shared);
    errdefer buf.deinit();
    const dst = buf.slice(u8, nbytes);
    if (cn == full_cols) {
        try readExact(io, file, dst, meta.data + e.begin);
        return buf;
    }
    const slab_rows: usize = @min(rows, 256);
    const tmp = try gpa.alloc(u8, slab_rows * full_cols * 2);
    defer gpa.free(tmp);
    var r0: usize = 0;
    while (r0 < rows) {
        const nr = @min(slab_rows, rows - r0);
        try readExact(io, file, tmp[0 .. nr * full_cols * 2], meta.data + e.begin + r0 * full_cols * 2);
        for (0..nr) |dr| {
            @memcpy(dst[(r0 + dr) * cn * 2 ..][0 .. cn * 2], tmp[(dr * full_cols + c0) * 2 ..][0 .. cn * 2]);
        }
        r0 += nr;
    }
    return buf;
}

/// Discover `{model_dir}-DFlash2` or `TF_GLM_DFLASH2`, returning null if disabled or absent.
pub fn resolveDir(gpa: std.mem.Allocator, io: Io, mc: *const cfg.Config, model_dir: []const u8) !?struct { dir: []u8, c: Config } {
    if (mc.pp > 1) return null;
    const env = @import("sys").environment;
    const env_path = (env.path("TF_GLM_DFLASH2", env.rawText("TF_GLM_DFLASH2")) catch null) orelse "";
    if (std.mem.eql(u8, env_path, "0") or std.mem.eql(u8, env_path, "false") or std.mem.eql(u8, env_path, "off")) return null;
    const dir = if (env_path.len > 0) try gpa.dupe(u8, env_path) else try std.fmt.allocPrint(gpa, "{s}-DFlash2", .{std.mem.trimEnd(u8, model_dir, "/")});
    errdefer gpa.free(dir);
    const cfg_path = try std.fmt.allocPrint(gpa, "{s}/config.json", .{dir});
    defer gpa.free(cfg_path);
    const cfg_text = Io.Dir.cwd().readFileAlloc(io, cfg_path, gpa, .limited(1 << 20)) catch {
        gpa.free(dir);
        return null;
    };
    defer gpa.free(cfg_text);
    const c = try Config.parse(gpa, cfg_text);
    if (c.hidden != mc.hidden or c.vocab != mc.vocab or c.heads % mc.tp != 0 or c.kv_heads % mc.tp != 0 or
        c.ffn % mc.tp != 0 or (num_taps * c.hidden) % mc.tp != 0 or c.hidden % 512 != 0 or
        ((c.heads / mc.tp) * 128) % 512 != 0 or (c.ffn / mc.tp) % 512 != 0 or
        ((num_taps * c.hidden) / mc.tp) % 512 != 0) return error.IncompatibleDFlash2;
    return .{ .dir = dir, .c = c };
}

fn encodeQ8(device: mtl.Device, enc: mtl.ComputeEncoder, pipe: mtl.Pipeline, src: mtl.Buffer, N: u32, K: u32) !Q8 {
    const w = try device.buffer(@as(usize, N) * K, mtl.ResourceOptions.private);
    errdefer w.deinit();
    const s = try device.buffer(@as(usize, N) * (K / 64) * 2, mtl.ResourceOptions.private);
    errdefer s.deinit();
    enc.setPipeline(pipe);
    enc.setBuffer(src, 0, 0);
    enc.setValue(QuantArgs{ .N = N, .K = K }, 1);
    enc.setBuffer(w, 0, 2);
    enc.setBuffer(s, 0, 3);
    enc.dispatchGroups(.{ .width = K / 64, .height = N }, .{ .width = 32 });
    return .{ .w = w, .s = s };
}

pub fn loadLayers(gpa: std.mem.Allocator, device: mtl.Device, io: Io, file: Io.File, meta: *const st.Metadata, c: Config, tp: u32, tp_rank: u32, quant_pipe: mtl.Pipeline, fc_w: mtl.Buffer, k_loc: u32, out: *[max_layers]LayerW) !Q8 {
    const q_loc = (c.heads / tp) * 128;
    const kv_dim_loc = (c.kv_heads / tp) * 128;
    const ffn_loc = c.ffn / tp;
    const dyn_w = 4 * (c.hidden / c.group);
    var nb: [128]u8 = undefined;
    const nm = struct {
        fn f(buf: []u8, i: usize, suf: []const u8) ![]const u8 {
            return std.fmt.bufPrint(buf, "layers.{d}.{s}", .{ i, suf });
        }
    }.f;
    const queue = try device.queue();
    defer queue.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const enc = cb.compute(.concurrent);
    var tmps: [max_layers * 7]mtl.Buffer = undefined;
    var n_tmp: usize = 0;
    defer for (tmps[0..n_tmp]) |b| b.deinit();
    const fc_q8 = try encodeQ8(device, enc, quant_pipe, fc_w, c.hidden, k_loc);
    errdefer fc_q8.deinit();
    for (0..c.layers) |i| {
        const tq = try loadRowSlice(device, io, file, meta, try nm(&nb, i, "self_attn.q_proj.weight"), tp_rank * q_loc, q_loc, c.hidden);
        tmps[n_tmp] = tq;
        n_tmp += 1;
        const to = try loadColSlice(gpa, device, io, file, meta, try nm(&nb, i, "self_attn.o_proj.weight"), c.hidden, tp_rank * q_loc, q_loc, c.heads * 128);
        tmps[n_tmp] = to;
        n_tmp += 1;
        const tg = try loadRowSlice(device, io, file, meta, try nm(&nb, i, "mlp.gate_proj.weight"), tp_rank * ffn_loc, ffn_loc, c.hidden);
        tmps[n_tmp] = tg;
        n_tmp += 1;
        const tu = try loadRowSlice(device, io, file, meta, try nm(&nb, i, "mlp.up_proj.weight"), tp_rank * ffn_loc, ffn_loc, c.hidden);
        tmps[n_tmp] = tu;
        n_tmp += 1;
        const td = try loadColSlice(gpa, device, io, file, meta, try nm(&nb, i, "mlp.down_proj.weight"), c.hidden, tp_rank * ffn_loc, ffn_loc, c.ffn);
        tmps[n_tmp] = td;
        n_tmp += 1;
        const ta = try loadFull(device, io, file, meta, try nm(&nb, i, "attention_conv.kernel_projection.weight"));
        tmps[n_tmp] = ta;
        n_tmp += 1;
        const tm = try loadFull(device, io, file, meta, try nm(&nb, i, "mlp_conv.kernel_projection.weight"));
        tmps[n_tmp] = tm;
        n_tmp += 1;
        const kw = try loadRowSlice(device, io, file, meta, try nm(&nb, i, "self_attn.k_proj.weight"), tp_rank * kv_dim_loc, kv_dim_loc, c.hidden);
        const vw = try loadRowSlice(device, io, file, meta, try nm(&nb, i, "self_attn.v_proj.weight"), tp_rank * kv_dim_loc, kv_dim_loc, c.hidden);
        out[i] = .{
            .in_norm = try loadFull(device, io, file, meta, try nm(&nb, i, "input_layernorm.weight")),
            .post_norm = try loadFull(device, io, file, meta, try nm(&nb, i, "post_attention_layernorm.weight")),
            .q_norm = try loadFull(device, io, file, meta, try nm(&nb, i, "self_attn.q_norm.weight")),
            .k_norm = try loadFull(device, io, file, meta, try nm(&nb, i, "self_attn.k_norm.weight")),
            .q_w = try encodeQ8(device, enc, quant_pipe, tq, q_loc, c.hidden),
            .k_w = kw,
            .v_w = vw,
            .k_q8 = try encodeQ8(device, enc, quant_pipe, kw, kv_dim_loc, c.hidden),
            .v_q8 = try encodeQ8(device, enc, quant_pipe, vw, kv_dim_loc, c.hidden),
            .o_w = try encodeQ8(device, enc, quant_pipe, to, c.hidden, q_loc),
            .gate_w = try encodeQ8(device, enc, quant_pipe, tg, ffn_loc, c.hidden),
            .up_w = try encodeQ8(device, enc, quant_pipe, tu, ffn_loc, c.hidden),
            .down_w = try encodeQ8(device, enc, quant_pipe, td, c.hidden, ffn_loc),
            .aconv_base = try loadFull(device, io, file, meta, try nm(&nb, i, "attention_conv.base_kernel")),
            .aconv_proj = try encodeQ8(device, enc, quant_pipe, ta, dyn_w, c.hidden),
            .mconv_base = try loadFull(device, io, file, meta, try nm(&nb, i, "mlp_conv.base_kernel")),
            .mconv_proj = try encodeQ8(device, enc, quant_pipe, tm, dyn_w, c.hidden),
        };
    }
    enc.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("glm_dsa DFlash2 Q8 quantize failed: {s}", .{msg});
        return error.GpuFailure;
    }
    return fc_q8;
}

pub fn initAne(gpa: std.mem.Allocator, device: mtl.Device, fc_w: mtl.Buffer, hidden: u32, k_loc: u32) ?*ane_mod.AneFc {
    const use_ane = @import("sys").environment.read(bool, "TF_GLM_DFLASH2_ANE", true) catch true;
    if (!use_ane or !mtl.ane.available()) return null;
    const fc_slice = fc_w.slice(u16, @as(usize, hidden) * k_loc);
    return ane_mod.AneFc.init(gpa, device, fc_slice, hidden, k_loc) catch |err| blk: {
        std.log.warn("glm_dsa DFlash2 ANE init skipped: {s}", .{@errorName(err)});
        break :blk null;
    };
}
