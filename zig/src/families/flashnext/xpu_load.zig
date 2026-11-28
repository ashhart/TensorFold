//! Qwen3.8 Flash Next device weight structures and TP={1,2,4} sharded safetensors loader (Xe2 16-col DPAS tile layout).
const std = @import("std");
const core = @import("core");
const ple = @import("xpu_ple.zig");
const host_pool = @import("host_pool.zig");
const proposal_table = @import("proposal_table.zig");
const st = @import("xpu_store.zig");

pub const layers_n = st.layers_n;
pub const linear_n = st.linear_n;
pub const attn_n = st.attn_n;
pub const dims = st.dims;
pub const streams = st.streams;
pub const wide = st.wide;
pub const low = st.low;
pub const eps = st.eps;
pub const vocab = st.vocab;
pub const experts_n = st.experts_n;
pub const topk = st.topk;
pub const slots = st.slots;
pub const moe_w = st.moe_w;
pub const q_heads = st.q_heads;
pub const kv_heads = st.kv_heads;
pub const head_dim = st.head_dim;
pub const index_heads = st.index_heads;
pub const index_dim = st.index_dim;
pub const half = st.half;
pub const index_ratio = st.index_ratio;
pub const index_top = st.index_top;
pub const index_idw = st.index_idw;
pub const max_attn_slices = st.max_attn_slices;
pub const ple_layer = st.ple_layer;
pub const ple_tail_n = st.ple_tail_n;
pub const gdn_taps = st.gdn_taps;
pub const max_tp = st.max_tp;
pub const max_verify_rows = st.max_verify_rows;
pub const chunk_rows = st.chunk_rows;
pub const default_solo_rows = st.default_solo_rows;
pub const max_chunk_rows = st.max_chunk_rows;
pub const attn_tile_rows = st.attn_tile_rows;
pub const linearLayer = st.linearLayer;
pub const linearIndex = st.linearIndex;
pub const attnIndex = st.attnIndex;
pub const promote = st.promote;
pub const TpShape = st.TpShape;
pub const Kernels = st.Kernels;
pub const Dev = st.Dev;
pub const off = st.off;
pub const detectDeviceId = st.detectDeviceId;
pub const Store = st.Store;
pub const asF32 = st.asF32;
const Tensor = st.Tensor;

pub const Q4 = struct {
    w: ?*anyopaque = null,
    s: ?*anyopaque = null,
    b: ?*anyopaque = null,
    n: usize = 0,
    k: usize = 0,
};

pub const HcW = struct {
    scale: *anyopaque,
    down: Q4,
    up: Q4,
    n_down: usize,
    has_inj: i32,
};

pub const EmbedW = struct {
    w: *anyopaque,
    s: ?*anyopaque = null,
    b: ?*anyopaque = null,
    dense: bool = false,
};

pub const LayerW = struct {
    linear: bool,
    with_ple: bool,
    attn_hc: HcW,
    mlp_hc: HcW,
    proj: Q4,
    out: Q4,
    router: *anyopaque, // [33, 160, 8, 16] i32 packed bf16 pairs for DPAS
    cw: ?*anyopaque = null, // [gdn_conv, 4] bf16
    a_log: ?*anyopaque = null, // [nv] f32
    dt: ?*anyopaque = null, // [nv] f32
    norm: ?*anyopaque = null, // [128] bf16
    q_scale: ?*anyopaque = null, // [256] f32
    k_scale: ?*anyopaque = null, // [256] f32
    i_scale: ?*anyopaque = null, // [128] f32
    ik_scale: ?*anyopaque = null, // [128] f32
    ple_key: Q4 = .{},
    ple_val: Q4 = .{},
    ple_nk: ?*anyopaque = null, // [10240] f32
    ple_nq: ?*anyopaque = null, // [10240] f32
    ple_nc: ?*anyopaque = null, // [10240] f32
    ple_cw: ?*anyopaque = null, // [10240, 4] bf16
    routed_gate: Q4,
    routed_up: Q4,
    routed_down: Q4,
    shared_gate: Q4,
    shared_up: Q4,
    shared_down: Q4,
};

pub const MtpW = struct {
    norm_e: *anyopaque,
    norm_h: *anyopaque,
    fc_e: Q4,
    fc_h: Q4,
    layer: LayerW,
    mixer: HcW,
    draft: Q4,
    draft_lo: usize,
    draft_len: usize,
};

pub const RankWeights = struct {
    embed: EmbedW,
    inv_freq: *anyopaque,
    layers: [layers_n]LayerW,
    mixer: HcW,
    head: Q4,
    mtp: ?MtpW = null,
};

const Span = struct { from: usize, len: usize };

fn holdTriple(store: *Store, buf: []u8, prefix: []const u8) !struct { w: Tensor, s: Tensor, b: Tensor } {
    const w = try store.hold(try std.fmt.bufPrint(buf, "{s}.weight", .{prefix}));
    const s = try store.hold(try std.fmt.bufPrint(buf, "{s}.scales", .{prefix}));
    const b = try store.hold(try std.fmt.bufPrint(buf, "{s}.biases", .{prefix}));
    return .{ .w = w, .s = s, .b = b };
}

pub const PackJob = struct {
    w_in: []const u8,
    s_in: []const u8,
    b_in: []const u8,
    w_out: []u32,
    s_out: []u16,
    b_out: []u16,
    n: usize,
    groups: usize,
    nb_total: usize,
    per: usize,

    pub fn packRange(j: *const PackJob, nb_lo: usize, nb_hi: usize) void {
        const groups = j.groups;
        const words = groups * 4;
        for (nb_lo..nb_hi) |nb| for (0..groups) |g| {
            const tg = nb * groups + g;
            const wp = j.w_out[tg * 64 ..][0..64];
            const sp = j.s_out[tg * 16 ..][0..16];
            const bp = j.b_out[tg * 16 ..][0..16];
            for (0..16) |lane| {
                const col = nb * 16 + lane;
                if (col < j.n) {
                    const wbase = (col * words + g * 4) * 4;
                    wp[0 + lane] = std.mem.readInt(u32, j.w_in[wbase + 0 ..][0..4], .little);
                    wp[16 + lane] = std.mem.readInt(u32, j.w_in[wbase + 4 ..][0..4], .little);
                    wp[32 + lane] = std.mem.readInt(u32, j.w_in[wbase + 8 ..][0..4], .little);
                    wp[48 + lane] = std.mem.readInt(u32, j.w_in[wbase + 12 ..][0..4], .little);
                    const sb_off = (col * groups + g) * 2;
                    sp[lane] = std.mem.readInt(u16, j.s_in[sb_off..][0..2], .little);
                    bp[lane] = std.mem.readInt(u16, j.b_in[sb_off..][0..2], .little);
                } else {
                    wp[0 + lane] = 0;
                    wp[16 + lane] = 0;
                    wp[32 + lane] = 0;
                    wp[48 + lane] = 0;
                    sp[lane] = 0;
                    bp[lane] = 0;
                }
            }
        };
    }

    fn part(ctx: *const anyopaque, idx: usize) anyerror!void {
        const j: *const PackJob = @ptrCast(@alignCast(ctx));
        const lo = @min(j.nb_total, idx * j.per);
        j.packRange(lo, @min(j.nb_total, lo + j.per));
    }
};

/// Pack row-major `[n, k]` Q4 weights into `[ceil(n/16), k/32, 4, 16]` coalesced Xe2 DPAS tile layout and upload.
fn packAndUploadQ4(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, w_raw: []const u8, s_raw: []const u8, b_raw: []const u8, n: usize, k: usize) !Q4 {
    if (k == 0 or k % 32 != 0) return error.UnexpectedTensor;
    const groups = k / 32;
    const nb_total = (n + 15) / 16;
    const w_out = try gpa.alloc(u32, nb_total * groups * 64);
    defer gpa.free(w_out);
    const s_out = try gpa.alloc(u16, nb_total * groups * 16);
    defer gpa.free(s_out);
    const b_out = try gpa.alloc(u16, nb_total * groups * 16);
    defer gpa.free(b_out);
    const n_parts = if (pool != null and nb_total >= 128) @min(@as(usize, 32), nb_total / 32) else 1;
    const job: PackJob = .{ .w_in = w_raw, .s_in = s_raw, .b_in = b_raw, .w_out = w_out, .s_out = s_out, .b_out = b_out, .n = n, .groups = groups, .nb_total = nb_total, .per = (nb_total + n_parts - 1) / n_parts };
    if (n_parts > 1) try pool.?.run(n_parts, .{ .ctx = &job, .run = PackJob.part }) else job.packRange(0, nb_total);
    return .{ .w = try d.upload(std.mem.sliceAsBytes(w_out)), .s = try d.upload(std.mem.sliceAsBytes(s_out)), .b = try d.upload(std.mem.sliceAsBytes(b_out)), .n = n, .k = k };
}

fn uploadFullQ4(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, prefix: []const u8, n: usize, k: usize) !Q4 {
    var buf: [256]u8 = undefined;
    const t = try holdTriple(store, &buf, prefix);
    if (!t.w.is(.u32, &.{ n, k / 8 }) or !t.s.is(.bf16, &.{ n, k / 32 }) or !t.b.is(.bf16, &.{ n, k / 32 })) return error.UnexpectedTensor;
    return packAndUploadQ4(gpa, pool, d, t.w.bytes, t.s.bytes, t.b.bytes, n, k);
}

fn uploadRowSliceQ4(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, prefix: []const u8, full_n: usize, k: usize, from: usize, len: usize) !Q4 {
    var buf: [256]u8 = undefined;
    const t = try holdTriple(store, &buf, prefix);
    if (!t.w.is(.u32, &.{ full_n, k / 8 }) or !t.s.is(.bf16, &.{ full_n, k / 32 }) or !t.b.is(.bf16, &.{ full_n, k / 32 }) or from + len > full_n) return error.UnexpectedTensor;
    const wb = (k / 8) * 4;
    const gb = (k / 32) * 2;
    return packAndUploadQ4(gpa, pool, d, t.w.bytes[from * wb ..][0 .. len * wb], t.s.bytes[from * gb ..][0 .. len * gb], t.b.bytes[from * gb ..][0 .. len * gb], len, k);
}

fn uploadColSliceQ4(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, prefix: []const u8, n: usize, full_k: usize, col_from: usize, col_len: usize) !Q4 {
    if (col_from % 32 != 0 or col_len % 32 != 0 or col_from + col_len > full_k) return error.UnexpectedTensor;
    var buf: [256]u8 = undefined;
    const t = try holdTriple(store, &buf, prefix);
    if (!t.w.is(.u32, &.{ n, full_k / 8 }) or !t.s.is(.bf16, &.{ n, full_k / 32 }) or !t.b.is(.bf16, &.{ n, full_k / 32 })) return error.UnexpectedTensor;
    const full_wb = (full_k / 8) * 4;
    const full_gb = (full_k / 32) * 2;
    const part_wb = (col_len / 8) * 4;
    const part_gb = (col_len / 32) * 2;
    const woff = (col_from / 8) * 4;
    const goff = (col_from / 32) * 2;

    const w_host = try gpa.alloc(u8, n * part_wb);
    defer gpa.free(w_host);
    const s_host = try gpa.alloc(u8, n * part_gb);
    defer gpa.free(s_host);
    const b_host = try gpa.alloc(u8, n * part_gb);
    defer gpa.free(b_host);
    for (0..n) |r| {
        @memcpy(w_host[r * part_wb ..][0..part_wb], t.w.bytes[r * full_wb + woff ..][0..part_wb]);
        @memcpy(s_host[r * part_gb ..][0..part_gb], t.s.bytes[r * full_gb + goff ..][0..part_gb]);
        @memcpy(b_host[r * part_gb ..][0..part_gb], t.b.bytes[r * full_gb + goff ..][0..part_gb]);
    }
    return packAndUploadQ4(gpa, pool, d, w_host, s_host, b_host, n, col_len);
}

const PartSlice = struct { suffix: []const u8, full_n: usize, spans: []const Span };

fn uploadStitchedQ4(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, prefix: []const u8, k: usize, parts: []const PartSlice) !Q4 {
    var total_n: usize = 0;
    for (parts) |p| for (p.spans) |sp| {
        total_n += sp.len;
    };
    const wb = (k / 8) * 4;
    const gb = (k / 32) * 2;
    const w_host = try gpa.alloc(u8, total_n * wb);
    defer gpa.free(w_host);
    const s_host = try gpa.alloc(u8, total_n * gb);
    defer gpa.free(s_host);
    const b_host = try gpa.alloc(u8, total_n * gb);
    defer gpa.free(b_host);

    var row_at: usize = 0;
    var buf: [256]u8 = undefined;
    var pbuf: [256]u8 = undefined;
    for (parts) |p| {
        const full_name = try std.fmt.bufPrint(&pbuf, "{s}{s}", .{ prefix, p.suffix });
        const t = try holdTriple(store, &buf, full_name);
        if (!t.w.is(.u32, &.{ p.full_n, k / 8 }) or !t.s.is(.bf16, &.{ p.full_n, k / 32 }) or !t.b.is(.bf16, &.{ p.full_n, k / 32 })) return error.UnexpectedTensor;
        for (p.spans) |sp| {
            if (sp.from + sp.len > p.full_n) return error.UnexpectedTensor;
            @memcpy(w_host[row_at * wb ..][0 .. sp.len * wb], t.w.bytes[sp.from * wb ..][0 .. sp.len * wb]);
            @memcpy(s_host[row_at * gb ..][0 .. sp.len * gb], t.s.bytes[sp.from * gb ..][0 .. sp.len * gb]);
            @memcpy(b_host[row_at * gb ..][0 .. sp.len * gb], t.b.bytes[sp.from * gb ..][0 .. sp.len * gb]);
            row_at += sp.len;
        }
    }
    if (row_at != total_n) return error.UnexpectedTensor;
    return packAndUploadQ4(gpa, pool, d, w_host, s_host, b_host, total_n, k);
}

fn uploadRoutedSliceQ4(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, prefix: []const u8, n: usize, k: usize, first_e: usize, local_e: usize) !Q4 {
    var buf: [256]u8 = undefined;
    const t = try holdTriple(store, &buf, prefix);
    if (!t.w.is(.u32, &.{ experts_n, n, k / 8 }) or !t.s.is(.bf16, &.{ experts_n, n, k / 32 }) or !t.b.is(.bf16, &.{ experts_n, n, k / 32 })) return error.UnexpectedTensor;
    const e_wb = n * (k / 8) * 4;
    const e_gb = n * (k / 32) * 2;
    return packAndUploadQ4(gpa, pool, d, t.w.bytes[first_e * e_wb ..][0 .. local_e * e_wb], t.s.bytes[first_e * e_gb ..][0 .. local_e * e_gb], t.b.bytes[first_e * e_gb ..][0 .. local_e * e_gb], local_e * n, k);
}

fn loadHcW(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, prefix: []const u8, inject: bool) !HcW {
    var buf: [256]u8 = undefined;
    const gamma = try store.hold(try std.fmt.bufPrint(&buf, "{s}.hc_norm.weight", .{prefix}));
    const scale_host = try store.normScale(gpa, gamma);
    defer gpa.free(scale_host);
    if (scale_host.len != wide) return error.UnexpectedTensor;
    const scale_dev = try d.upload(std.mem.sliceAsBytes(scale_host));

    const down_q = if (inject)
        try uploadStitchedQ4(gpa, pool, d, store, prefix, wide, &.{
            .{ .suffix = ".input_mix_weight_down", .full_n = low, .spans = &.{.{ .from = 0, .len = low }} },
            .{ .suffix = ".block_inject_weight", .full_n = streams, .spans = &.{.{ .from = 0, .len = streams }} },
        })
    else
        try uploadFullQ4(gpa, pool, d, store, try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_down", .{prefix}), low, wide);

    const up_q = try uploadFullQ4(gpa, pool, d, store, try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up", .{prefix}), wide, low);
    return .{ .scale = scale_dev, .down = down_q, .up = up_q, .n_down = if (inject) low + streams else low, .has_inj = @intFromBool(inject) };
}

fn packRouterTiles(gpa: std.mem.Allocator, d: *Dev, router_host: []const u16) !*anyopaque {
    const packed_r = try gpa.alloc(u32, 33 * 160 * 8 * 16);
    defer gpa.free(packed_r);
    for (0..33) |nb| {
        for (0..160) |kb| {
            const base = (nb * 160 + kb) * 128;
            for (0..8) |p| {
                for (0..16) |lane| {
                    const col = nb * 16 + lane;
                    if (col < experts_n + 1) {
                        const r_off = col * dims + kb * 16 + p * 2;
                        const lo: u32 = router_host[r_off];
                        const hi: u32 = router_host[r_off + 1];
                        packed_r[base + p * 16 + lane] = lo | (hi << 16);
                    } else {
                        packed_r[base + p * 16 + lane] = 0;
                    }
                }
            }
        }
    }
    return d.upload(std.mem.sliceAsBytes(packed_r));
}

fn loadLayerRank(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, root: []const u8, linear: bool, with_ple: bool, shp: TpShape, rk: usize) !LayerW {
    var buf: [256]u8 = undefined;
    var pbuf: [256]u8 = undefined;

    const attn_hc = try loadHcW(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.attn_hyper_connection", .{root}), true);
    const mlp_hc = try loadHcW(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.mlp_hyper_connection", .{root}), true);

    var mlp_buf: [160]u8 = undefined;
    const mlp_root = try std.fmt.bufPrint(&mlp_buf, "{s}.mlp", .{root});

    const router_host = try gpa.alloc(u16, (experts_n + 1) * dims);
    defer gpa.free(router_host);
    const gate_t = try store.hold(try std.fmt.bufPrint(&buf, "{s}.gate.weight", .{mlp_root}));
    if (!gate_t.is(.bf16, &.{ experts_n, dims })) return error.UnexpectedTensor;
    @memcpy(std.mem.sliceAsBytes(router_host[0 .. experts_n * dims]), gate_t.bytes);

    const sg_w_name = try std.fmt.bufPrint(&buf, "{s}.shared_expert_gate.weight", .{mlp_root});
    const sg_w_t = try store.hold(sg_w_name);
    if (sg_w_t.dtype == .bf16) {
        if (!sg_w_t.is(.bf16, &.{ 1, dims })) return error.UnexpectedTensor;
        @memcpy(std.mem.sliceAsBytes(router_host[experts_n * dims ..]), sg_w_t.bytes);
    } else {
        const sg_prefix = try std.fmt.bufPrint(&pbuf, "{s}.shared_expert_gate", .{mlp_root});
        const sg = try holdTriple(store, &buf, sg_prefix);
        try ple.dequantRow(sg.w.bytes, sg.s.bytes, sg.b.bytes, dims, 0, 1.0, router_host[experts_n * dims ..][0..dims]);
    }
    const router_dev = try packRouterTiles(gpa, d, router_host);

    var lw: LayerW = .{
        .linear = linear,
        .with_ple = with_ple,
        .attn_hc = attn_hc,
        .mlp_hc = mlp_hc,
        .proj = .{},
        .out = .{},
        .router = router_dev,
        .routed_gate = .{},
        .routed_up = .{},
        .routed_down = .{},
        .shared_gate = .{},
        .shared_up = .{},
        .shared_down = .{},
    };

    if (linear) {
        var lp_buf: [160]u8 = undefined;
        const lp = try std.fmt.bufPrint(&lp_buf, "{s}.linear_attn", .{root});
        const qk = shp.nk * 128;
        const v = shp.nv * 128;
        const full_qk = 16 * 128;
        const qkv_spans = [_]Span{
            .{ .from = rk * qk, .len = qk },
            .{ .from = full_qk + rk * qk, .len = qk },
            .{ .from = 2 * full_qk + rk * v, .len = v },
        };
        const z_spans = [_]Span{.{ .from = rk * v, .len = v }};
        const ba_spans = [_]Span{.{ .from = rk * shp.nv, .len = shp.nv }};
        lw.proj = try uploadStitchedQ4(gpa, pool, d, store, lp, dims, &.{
            .{ .suffix = ".in_proj_qkv", .full_n = 10240, .spans = &qkv_spans },
            .{ .suffix = ".in_proj_z", .full_n = 6144, .spans = &z_spans },
            .{ .suffix = ".in_proj_b", .full_n = 48, .spans = &ba_spans },
            .{ .suffix = ".in_proj_a", .full_n = 48, .spans = &ba_spans },
        });
        lw.out = try uploadColSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.out_proj", .{lp}), dims, 6144, rk * shp.gdn_vdim, shp.gdn_vdim);

        const conv_t = try store.hold(try std.fmt.bufPrint(&buf, "{s}.conv1d.weight", .{lp}));
        if (conv_t.dtype != .bf16 or conv_t.dim(0) != 10240 or conv_t.bytes.len != 10240 * gdn_taps * 2) return error.UnexpectedTensor;
        const row_b = gdn_taps * 2;
        const cw_host = try gpa.alloc(u8, shp.gdn_conv * row_b);
        defer gpa.free(cw_host);
        var cat_at: usize = 0;
        for (qkv_spans) |sp| {
            @memcpy(cw_host[cat_at..][0 .. sp.len * row_b], conv_t.bytes[sp.from * row_b ..][0 .. sp.len * row_b]);
            cat_at += sp.len * row_b;
        }
        lw.cw = try d.upload(cw_host);

        const a_all = try asF32(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.A_log", .{lp})));
        defer gpa.free(a_all);
        const dt_all = try asF32(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.dt_bias", .{lp})));
        defer gpa.free(dt_all);
        if (a_all.len != 48 or dt_all.len != 48) return error.UnexpectedTensor;
        lw.a_log = try d.upload(std.mem.sliceAsBytes(a_all[rk * shp.nv ..][0..shp.nv]));
        lw.dt = try d.upload(std.mem.sliceAsBytes(dt_all[rk * shp.nv ..][0..shp.nv]));

        const nw_t = try store.hold(try std.fmt.bufPrint(&buf, "{s}.norm.weight", .{lp}));
        if (!nw_t.is(.bf16, &.{128})) return error.UnexpectedTensor;
        lw.norm = try d.upload(nw_t.bytes);
    } else {
        var ap_buf: [160]u8 = undefined;
        const ap = try std.fmt.bufPrint(&ap_buf, "{s}.self_attn", .{root});
        const q_rows = shp.nq * 512;
        const kv_rows = shp.nkv * 256;
        const kv_idx = if (shp.tp == 1) 0 else (rk * 2) / shp.tp;
        const q_spans = [_]Span{.{ .from = rk * q_rows, .len = q_rows }};
        const kv_spans = [_]Span{.{ .from = kv_idx * 256, .len = kv_rows }};
        const idx_spans = [_]Span{.{ .from = 0, .len = 640 }};
        lw.proj = try uploadStitchedQ4(gpa, pool, d, store, ap, dims, &.{
            .{ .suffix = ".q_proj", .full_n = 12288, .spans = &q_spans },
            .{ .suffix = ".k_proj", .full_n = 512, .spans = &kv_spans },
            .{ .suffix = ".v_proj", .full_n = 512, .spans = &kv_spans },
            .{ .suffix = ".indexer.index_qk_proj", .full_n = 640, .spans = &idx_spans },
        });
        lw.out = try uploadColSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.o_proj", .{ap}), dims, 6144, rk * shp.attn_odim, shp.attn_odim);

        const qn = try store.normScale(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.q_norm.weight", .{ap})));
        defer gpa.free(qn);
        const kn = try store.normScale(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.k_norm.weight", .{ap})));
        defer gpa.free(kn);
        const iqn = try store.normScale(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.indexer.q_layernorm.weight", .{ap})));
        defer gpa.free(iqn);
        const ikn = try store.normScale(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.indexer.k_layernorm.weight", .{ap})));
        defer gpa.free(ikn);
        if (qn.len != head_dim or kn.len != head_dim or iqn.len != index_dim or ikn.len != index_dim) return error.UnexpectedTensor;
        lw.q_scale = try d.upload(std.mem.sliceAsBytes(qn));
        lw.k_scale = try d.upload(std.mem.sliceAsBytes(kn));
        lw.i_scale = try d.upload(std.mem.sliceAsBytes(iqn));
        lw.ik_scale = try d.upload(std.mem.sliceAsBytes(ikn));
    }

    if (with_ple) {
        var pp_buf: [160]u8 = undefined;
        const pp = try std.fmt.bufPrint(&pp_buf, "{s}.ple", .{root});
        lw.ple_key = try uploadFullQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.key_proj", .{pp}), wide, dims);
        lw.ple_val = try uploadFullQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.value_proj", .{pp}), dims, dims);
        const nk = try store.normScale(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.norm_key.weight", .{pp})));
        defer gpa.free(nk);
        const nq = try store.normScale(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.norm_query.weight", .{pp})));
        defer gpa.free(nq);
        const nc = try store.normScale(gpa, try store.hold(try std.fmt.bufPrint(&buf, "{s}.norm_conv.weight", .{pp})));
        defer gpa.free(nc);
        if (nk.len != wide or nq.len != wide or nc.len != wide) return error.UnexpectedTensor;
        lw.ple_nk = try d.upload(std.mem.sliceAsBytes(nk));
        lw.ple_nq = try d.upload(std.mem.sliceAsBytes(nq));
        lw.ple_nc = try d.upload(std.mem.sliceAsBytes(nc));
        const pconv = try store.hold(try std.fmt.bufPrint(&buf, "{s}.conv1d.weight", .{pp}));
        if (pconv.dtype != .bf16 or pconv.dim(0) != wide or pconv.bytes.len != wide * 4 * 2) return error.UnexpectedTensor;
        lw.ple_cw = try d.upload(pconv.bytes);
    }

    const first_e = rk * shp.local_e;
    lw.routed_gate = try uploadRoutedSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.switch_mlp.gate_proj", .{mlp_root}), moe_w, dims, first_e, shp.local_e);
    lw.routed_up = try uploadRoutedSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.switch_mlp.up_proj", .{mlp_root}), moe_w, dims, first_e, shp.local_e);
    lw.routed_down = try uploadRoutedSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.switch_mlp.down_proj", .{mlp_root}), dims, moe_w, first_e, shp.local_e);

    lw.shared_gate = try uploadRowSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.shared_expert.gate_proj", .{mlp_root}), moe_w, dims, rk * shp.sh_w, shp.sh_w);
    lw.shared_up = try uploadRowSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.shared_expert.up_proj", .{mlp_root}), moe_w, dims, rk * shp.sh_w, shp.sh_w);
    lw.shared_down = try uploadColSliceQ4(gpa, pool, d, store, try std.fmt.bufPrint(&pbuf, "{s}.shared_expert.down_proj", .{mlp_root}), dims, moe_w, rk * shp.sh_w, shp.sh_w);
    return lw;
}

pub fn draftIds(gpa: std.mem.Allocator) ![]u32 {
    return core.proposal_vocab.legacy.ids(gpa, proposal_table.text, .{ .vocab = vocab, .skip_past_vocab = true, .delimiters = " \t\r\n," });
}

fn uploadDraftSliceQ4(gpa: std.mem.Allocator, pool: ?*host_pool.Pool, d: *Dev, store: *Store, ids: []const u32) !Q4 {
    var buf: [256]u8 = undefined;
    const t = try holdTriple(store, &buf, "language_model.lm_head");
    if (!t.w.is(.u32, &.{ vocab, dims / 8 }) or !t.s.is(.bf16, &.{ vocab, dims / 32 }) or !t.b.is(.bf16, &.{ vocab, dims / 32 })) return error.UnexpectedTensor;
    const wb = (dims / 8) * 4;
    const gb = (dims / 32) * 2;
    const w_host = try gpa.alloc(u8, ids.len * wb);
    defer gpa.free(w_host);
    const s_host = try gpa.alloc(u8, ids.len * gb);
    defer gpa.free(s_host);
    const b_host = try gpa.alloc(u8, ids.len * gb);
    defer gpa.free(b_host);
    for (ids, 0..) |tok, i| {
        const row: usize = @intCast(tok);
        @memcpy(w_host[i * wb ..][0..wb], t.w.bytes[row * wb ..][0..wb]);
        @memcpy(s_host[i * gb ..][0..gb], t.s.bytes[row * gb ..][0..gb]);
        @memcpy(b_host[i * gb ..][0..gb], t.b.bytes[row * gb ..][0..gb]);
    }
    return packAndUploadQ4(gpa, pool, d, w_host, s_host, b_host, ids.len, dims);
}

/// Load all weights across `devs[0..tp]` in layer order into coalesced 16-column Xe2 DPAS tile layout.
pub fn loadAllRanks(gpa: std.mem.Allocator, io: std.Io, pool: ?*host_pool.Pool, dir: []const u8, devs: []Dev, shp: TpShape, want_mtp: bool, draft_ids: []const u32, out: []RankWeights) !void {
    var store = try Store.open(gpa, io, dir);
    defer store.deinit();
    try store.detectNorms();

    var inv: [half]f32 = undefined;
    for (0..half) |i| inv[i] = @floatCast(1.0 / std.math.pow(f64, 10_000_000.0, @as(f64, @floatFromInt(2 * i)) / 64.0));

    {
        var buf: [256]u8 = undefined;
        const ew = try store.hold("language_model.model.embed_tokens.weight");
        if (ew.dtype == .bf16) {
            if (!ew.is(.bf16, &.{ vocab, dims })) return error.UnexpectedTensor;
            for (devs, 0..) |*d, rk| {
                out[rk].embed = .{ .w = try d.upload(ew.bytes), .dense = true };
                out[rk].inv_freq = try d.upload(std.mem.sliceAsBytes(&inv));
            }
        } else {
            const et = try holdTriple(&store, &buf, "language_model.model.embed_tokens");
            if (!et.w.is(.u32, &.{ vocab, dims / 8 }) or !et.s.is(.bf16, &.{ vocab, dims / 32 }) or !et.b.is(.bf16, &.{ vocab, dims / 32 })) return error.UnexpectedTensor;
            for (devs, 0..) |*d, rk| {
                out[rk].embed = .{ .w = try d.upload(et.w.bytes), .s = try d.upload(et.s.bytes), .b = try d.upload(et.b.bytes), .dense = false };
                out[rk].inv_freq = try d.upload(std.mem.sliceAsBytes(&inv));
            }
        }
        store.release();
    }

    for (0..layers_n) |l| {
        var root_buf: [96]u8 = undefined;
        const root = try std.fmt.bufPrint(&root_buf, "language_model.model.layers.{d}", .{l});
        for (devs, 0..) |*d, rk| {
            out[rk].layers[l] = try loadLayerRank(gpa, pool, d, &store, root, linearLayer(l), l == ple_layer, shp, rk);
        }
        store.release();
    }

    for (devs, 0..) |*d, rk| {
        out[rk].mixer = try loadHcW(gpa, pool, d, &store, "language_model.model.hyper_connection_mixer", false);
        out[rk].head = try uploadRowSliceQ4(gpa, pool, d, &store, "language_model.lm_head", vocab, dims, rk * shp.vocab_per, shp.vocab_per);
        out[rk].mtp = null;
    }
    store.release();

    if (want_mtp and store.has("language_model.mtp.pre_fc_norm_embedding.weight")) {
        const ne = try store.normScale(gpa, try store.hold("language_model.mtp.pre_fc_norm_embedding.weight"));
        defer gpa.free(ne);
        const nh = try store.normScale(gpa, try store.hold("language_model.mtp.pre_fc_norm_hidden.weight"));
        defer gpa.free(nh);
        if (ne.len != dims or nh.len != wide) return error.UnexpectedTensor;

        for (devs, 0..) |*d, rk| {
            const sl = shp.draftSlice(rk, draft_ids.len);
            out[rk].mtp = .{
                .norm_e = try d.upload(std.mem.sliceAsBytes(ne)),
                .norm_h = try d.upload(std.mem.sliceAsBytes(nh)),
                .fc_e = try uploadFullQ4(gpa, pool, d, &store, "language_model.mtp.fc_embedding", dims, dims),
                .fc_h = try uploadFullQ4(gpa, pool, d, &store, "language_model.mtp.fc_hidden", dims, dims),
                .layer = try loadLayerRank(gpa, pool, d, &store, "language_model.mtp.layers.0", false, false, shp, rk),
                .mixer = try loadHcW(gpa, pool, d, &store, "language_model.mtp.hyper_connection_mixer", false),
                .draft = try uploadDraftSliceQ4(gpa, pool, d, &store, draft_ids[sl.lo .. sl.lo + sl.len]),
                .draft_lo = sl.lo,
                .draft_len = sl.len,
            };
        }
        store.release();
    }

    for (devs) |*d| d.dropStaging();
}
