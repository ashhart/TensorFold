//! Qwen3.8 Flash Next per-rank sequence state, snapshots, and scratch allocation on Intel Xe2.
const std = @import("std");
const ze = @import("ze");
const ld = @import("xpu_load.zig");
const fwd = @import("xpu_forward.zig");

const Runner = fwd.Runner;
const linear_n = ld.linear_n;
const attn_n = ld.attn_n;
const dims = ld.dims;
const streams = ld.streams;
const wide = ld.wide;
const low = ld.low;
const vocab = ld.vocab;
const slots = ld.slots;
const moe_w = ld.moe_w;
const head_dim = ld.head_dim;
const index_heads = ld.index_heads;
const index_dim = ld.index_dim;
const index_ratio = ld.index_ratio;
const index_idw = ld.index_idw;
const max_attn_slices = ld.max_attn_slices;
const ple_tail_n = ld.ple_tail_n;
const gdn_taps = ld.gdn_taps;
const max_tp = ld.max_tp;
const max_verify_rows = ld.max_verify_rows;
const chunk_rows = ld.chunk_rows;
const Dev = ld.Dev;
const TpShape = ld.TpShape;

pub const RankSeq = struct {
    conv: [linear_n]*anyopaque = undefined, // [3, gdn_conv] bf16
    state: [linear_n]*anyopaque = undefined, // [nv, 128, 128] f32
    saved_proj: [linear_n]*anyopaque = undefined, // [max_verify_rows, gdn_pw] bf16 (per-stream rollback)
    tmp_state: [linear_n]*anyopaque = undefined, // [nv, 128, 128] f32 (per-stream rollback)
    kc: [attn_n]*anyopaque = undefined, // [cap, nkv, 256] bf16
    vc: [attn_n]*anyopaque = undefined, // [cap, nkv, 256] bf16
    ikc: [attn_n]*anyopaque = undefined, // [cap, 128] bf16
    pooled: [attn_n]*anyopaque = undefined, // [(cap + 3) / 4, 128] bf16
    ple_tail: *anyopaque = undefined, // [9, 10240] bf16
    saved_ple_nrow: *anyopaque = undefined, // [max_verify_rows, 10240] bf16 (per-stream rollback)
    // MTP caches & saved hidden stream buffers
    mtp_kc: ?*anyopaque = null,
    mtp_vc: ?*anyopaque = null,
    mtp_ikc: ?*anyopaque = null,
    mtp_pooled: ?*anyopaque = null,
    last_streams: *anyopaque = undefined, // [10240] bf16 (prompt's final row)
    verify_streams: *anyopaque = undefined, // [max_verify_rows, 10240] bf16 (kept rows for MTP)
    mtp_out: *anyopaque = undefined, // [10240] bf16 (MTP chain step output)
};

pub const Seq = struct {
    slot: u32 = 0,
    cap: usize,
    pos: u32 = 0,
    history: [2]i64,
    ranks: [max_tp]RankSeq = undefined,
    // Pending verify-window rollback bookkeeping
    win_open: bool = false,
    win_pos0: u32 = 0,
    win_rows: u32 = 0,
    win_history: [2]i64 = .{ 0, 0 },
    win_ids: [max_verify_rows]u32 = @splat(0),
};

/// Prompt-cache snapshot at prompt position `at`.
pub const Snap = struct {
    at: u32,
    history: [2]i64,
    conv: [max_tp][linear_n]*anyopaque = undefined,
    state: [max_tp][linear_n]*anyopaque = undefined,
    kc: [max_tp][attn_n]*anyopaque = undefined,
    vc: [max_tp][attn_n]*anyopaque = undefined,
    ikc: [max_tp][attn_n]*anyopaque = undefined,
    pooled: [max_tp][attn_n]*anyopaque = undefined,
    ple_tail: [max_tp]*anyopaque = undefined,
    mtp_kc: [max_tp]?*anyopaque = @splat(null),
    mtp_vc: [max_tp]?*anyopaque = @splat(null),
    mtp_ikc: [max_tp]?*anyopaque = @splat(null),
    mtp_pooled: [max_tp]?*anyopaque = @splat(null),
    last_streams: [max_tp]*anyopaque = undefined,
};

/// Per-rank activation scratch sized for up to `chunk_rows` (64) rows.
pub const RankActs = struct {
    ids: *anyopaque, // [chunk_rows] i32
    dummy: *anyopaque, // 4096 zero bytes
    h: *anyopaque, // [chunk_rows, 10240] bf16
    prompt_streams: *anyopaque, // [chunk_rows, 10240] bf16 (saved post-L47 streams for MTP prefill absorb)
    norm: *anyopaque, // [chunk_rows, 10240] bf16
    act: *anyopaque, // [chunk_rows, 320] bf16
    inj: *anyopaque, // [chunk_rows, 4] bf16
    mixed: *anyopaque, // [chunk_rows, 2560] bf16
    branch: *anyopaque, // [chunk_rows, 2560] f32
    mtp_branch: *anyopaque, // [chunk_rows, 2560] bf16
    proj: *anyopaque, // [chunk_rows, max(gdn_pw, attn_pw)] bf16
    // PLE scratch
    ple_emb: *anyopaque, // [chunk_rows, 2560] bf16
    ple_keys: *anyopaque, // [chunk_rows, 10240] bf16
    ple_vals: *anyopaque, // [chunk_rows, 2560] bf16
    ple_gated: *anyopaque, // [chunk_rows, 10240] bf16
    ple_pss: *anyopaque, // [chunk_rows, 4] f32
    ple_nrow: *anyopaque, // [chunk_rows, 10240] bf16
    // GDN scratch + verify window saved projections & temp end-states
    gq: *anyopaque, // [chunk_rows, nk, 128] f32
    gk: *anyopaque, // [chunk_rows, nk, 128] f32
    gv: *anyopaque, // [chunk_rows, nv, 128] f32
    ggb: *anyopaque, // [chunk_rows, nv, 2] f32
    ys: *anyopaque, // [chunk_rows, nv, 128] f32
    gout: *anyopaque, // [chunk_rows, nv * 128] bf16
    // QSA Attention scratch
    q: *anyopaque, // [chunk_rows, nq, 256] bf16
    iq: *anyopaque, // [chunk_rows, 4, 128] bf16
    qsa_sc: *anyopaque, // [chunk_rows, nb_max] f32
    qsa_ids: *anyopaque, // [chunk_rows, 2052] i32
    qsa_nkr: *anyopaque, // [chunk_rows] i32
    qsa_spr: *anyopaque, // [chunk_rows] i32
    pm: *anyopaque, // [chunk_rows, nq, 65] f32
    pl: *anyopaque, // [chunk_rows, nq, 65] f32
    po: *anyopaque, // [chunk_rows, nq, 65, 256] f32
    ao: *anyopaque, // [chunk_rows, nq, 256] bf16
    agated: *anyopaque, // [chunk_rows, nq * 256] bf16
    // MoE scratch
    router_l: *anyopaque, // [chunk_rows, 513] f32
    picks: *anyopaque, // [chunk_rows, 11] i32
    wts: *anyopaque, // [chunk_rows, 11] f32
    active: *anyopaque, // [chunk_rows * 10] i32
    exp_grp: *anyopaque, // [chunk_rows * 10] i32
    n_act: *anyopaque, // [4] i32
    row_list: *anyopaque, // [chunk_rows * 11] i32
    moe_act: *anyopaque, // [chunk_rows * 10, 640] bf16
    sh_act: *anyopaque, // [chunk_rows, sh_w] bf16
    // Logits & Draft Head scratch
    shard_logits: *anyopaque, // [max_verify_rows, vocab_per] bf16
    shard_dlogits: ?*anyopaque = null, // [1, draft_len] bf16
    // Rank 0 full-vocab / draft-vocab buffers
    logits_bf16: ?*anyopaque = null, // [max_verify_rows, vocab] bf16
    dlogits_bf16: ?*anyopaque = null, // [1, draft_ids.len] bf16
    part_v: ?*anyopaque = null, // [max_verify_rows, 256] f32
    part_i: ?*anyopaque = null, // [max_verify_rows, 256] i32
    part_s: ?*anyopaque = null, // [max_verify_rows, 256] f32
    out_tok: ?*anyopaque = null, // [max_verify_rows] i32
    out_best: ?*anyopaque = null, // [max_verify_rows, 4] f32
    top64_v: ?*anyopaque = null, // [256 * 64] f32
    top64_i: ?*anyopaque = null, // [256 * 64] i32
    row_f32: ?*anyopaque = null, // [vocab] f32
};

pub fn allocHostSlice(comptime T: type, d0: *Dev, gpus: []ze.Gpu, count: usize) ![]T {
    const bytes = @max(count * @sizeOf(T), 64);
    const ptr = try d0.g.hostAlloc(bytes);
    for (gpus) |*g| try g.makeResident(ptr, bytes);
    const slice = @as([*]T, @ptrCast(@alignCast(ptr)))[0..count];
    @memset(slice, 0);
    return slice;
}

pub fn allocRankActs(d: *Dev, shp: TpShape, rk: usize, nb_max: usize, draft_total: usize, has_mtp: bool) !RankActs {
    const max_pw = @max(shp.gdn_pw, shp.attn_pw);
    var a: RankActs = .{
        .ids = try d.allocZero(chunk_rows * 4),
        .dummy = try d.allocZero(4096),
        .h = try d.allocZero(chunk_rows * wide * 2),
        .prompt_streams = try d.allocZero(chunk_rows * wide * 2),
        .norm = try d.allocZero(chunk_rows * wide * 2),
        .act = try d.allocZero(chunk_rows * low * 2),
        .inj = try d.allocZero(chunk_rows * streams * 2),
        .mixed = try d.allocZero(chunk_rows * dims * 2),
        .branch = try d.allocZero(chunk_rows * dims * 4),
        .mtp_branch = try d.allocZero(chunk_rows * dims * 2),
        .proj = try d.allocZero(chunk_rows * max_pw * 2),
        .ple_emb = try d.allocZero(chunk_rows * dims * 2),
        .ple_keys = try d.allocZero(chunk_rows * wide * 2),
        .ple_vals = try d.allocZero(chunk_rows * dims * 2),
        .ple_gated = try d.allocZero(chunk_rows * wide * 2),
        .ple_pss = try d.allocZero(chunk_rows * streams * 4),
        .ple_nrow = try d.allocZero(chunk_rows * wide * 2),
        .gq = try d.allocZero(chunk_rows * shp.nk * 128 * 4),
        .gk = try d.allocZero(chunk_rows * shp.nk * 128 * 4),
        .gv = try d.allocZero(chunk_rows * shp.nv * 128 * 4),
        .ggb = try d.allocZero(chunk_rows * shp.nv * 2 * 4),
        .ys = try d.allocZero(chunk_rows * shp.nv * 128 * 4),
        .gout = try d.allocZero(chunk_rows * shp.gdn_vdim * 2),
        .q = try d.allocZero(chunk_rows * shp.nq * head_dim * 2),
        .iq = try d.allocZero(chunk_rows * index_heads * index_dim * 2),
        .qsa_sc = try d.allocZero(chunk_rows * nb_max * 4),
        .qsa_ids = try d.allocZero(chunk_rows * index_idw * 4),
        .qsa_nkr = try d.allocZero(chunk_rows * 4),
        .qsa_spr = try d.allocZero(chunk_rows * 4),
        .pm = try d.allocZero(chunk_rows * shp.nq * max_attn_slices * 4),
        .pl = try d.allocZero(chunk_rows * shp.nq * max_attn_slices * 4),
        .po = try d.allocZero(chunk_rows * shp.nq * max_attn_slices * head_dim * 4),
        .ao = try d.allocZero(chunk_rows * shp.nq * head_dim * 2),
        .agated = try d.allocZero(chunk_rows * shp.attn_odim * 2),
        .router_l = try d.allocZero(chunk_rows * 513 * 4),
        .picks = try d.allocZero(chunk_rows * slots * 4),
        .wts = try d.allocZero(chunk_rows * slots * 4),
        .active = try d.allocZero(chunk_rows * 10 * 4),
        .exp_grp = try d.allocZero(chunk_rows * 10 * 4),
        .n_act = try d.allocZero(16),
        .row_list = try d.allocZero(chunk_rows * slots * 4),
        .moe_act = try d.allocZero(chunk_rows * 10 * moe_w * 2),
        .sh_act = try d.allocZero(chunk_rows * shp.sh_w * 2),
        .shard_logits = try d.allocZero(max_verify_rows * shp.vocab_per * 2),
    };
    if (has_mtp) {
        const sl = shp.draftSlice(rk, draft_total);
        a.shard_dlogits = try d.allocZero(@max(sl.len * 2, 64));
    }
    if (rk == 0) {
        a.logits_bf16 = if (shp.tp == 1) a.shard_logits else try d.allocZero(max_verify_rows * vocab * 2);
        if (has_mtp) {
            a.dlogits_bf16 = if (shp.tp == 1) a.shard_dlogits else try d.allocZero(draft_total * 2);
        }
        a.part_v = try d.allocZero(max_verify_rows * 256 * 4);
        a.part_i = try d.allocZero(max_verify_rows * 256 * 4);
        a.part_s = try d.allocZero(max_verify_rows * 256 * 4);
        a.out_tok = try d.allocZero(max_verify_rows * 4);
        a.out_best = try d.allocZero(max_verify_rows * 4 * 4);
        a.top64_v = try d.allocZero(256 * 64 * 4);
        a.top64_i = try d.allocZero(256 * 64 * 4);
        a.row_f32 = try d.allocZero(vocab * 4);
    }
    return a;
}

pub fn seqBytesPerRank(r: *const Runner, cap: usize) usize {
    const conv_b = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b = r.shp.nv * 128 * 128 * 4;
    const sproj_b = max_verify_rows * r.shp.gdn_pw * 2;
    const kv_b = cap * r.shp.nkv * head_dim * 2;
    const ik_b = cap * index_dim * 2;
    const pl_b = ((cap + index_ratio - 1) / index_ratio) * index_dim * 2;
    const attn_total = attn_n + @as(usize, @intFromBool(r.has_mtp));
    return linear_n * (conv_b + 2 * st_b + sproj_b) + attn_total * (2 * kv_b + ik_b + pl_b) + (ple_tail_n + 2 * max_verify_rows + 2) * wide * 2;
}

pub fn initSeqValue(r: *Runner, cap: usize) !Seq {
    var s: Seq = .{
        .cap = cap,
        .pos = 0,
        .history = .{ r.eos, r.eos },
    };
    const conv_b = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b = r.shp.nv * 128 * 128 * 4;
    const sproj_b = max_verify_rows * r.shp.gdn_pw * 2;
    const kv_b = cap * r.shp.nkv * head_dim * 2;
    const ik_b = cap * index_dim * 2;
    const pl_b = @max(((cap + index_ratio - 1) / index_ratio) * index_dim * 2, 64);
    for (0..r.tp) |rk| {
        const d = &r.devs[rk];
        var rs: RankSeq = .{};
        for (0..linear_n) |li| {
            rs.conv[li] = try d.allocZero(conv_b);
            rs.state[li] = try d.allocZero(st_b);
            rs.saved_proj[li] = try d.allocZero(sproj_b);
            rs.tmp_state[li] = try d.allocZero(st_b);
        }
        for (0..attn_n) |ai| {
            rs.kc[ai] = try d.allocZero(kv_b);
            rs.vc[ai] = try d.allocZero(kv_b);
            rs.ikc[ai] = try d.allocZero(ik_b);
            rs.pooled[ai] = try d.allocZero(pl_b);
        }
        rs.ple_tail = try d.allocZero(ple_tail_n * wide * 2);
        rs.saved_ple_nrow = try d.allocZero(max_verify_rows * wide * 2);
        if (r.has_mtp) {
            rs.mtp_kc = try d.allocZero(kv_b);
            rs.mtp_vc = try d.allocZero(kv_b);
            rs.mtp_ikc = try d.allocZero(ik_b);
            rs.mtp_pooled = try d.allocZero(pl_b);
        }
        rs.last_streams = try d.allocZero(wide * 2);
        rs.verify_streams = try d.allocZero(max_verify_rows * wide * 2);
        rs.mtp_out = try d.allocZero(wide * 2);
        s.ranks[rk] = rs;
        r.gpus[rk] = d.g;
    }
    return s;
}

pub fn newSeq(r: *Runner) !*Seq {
    const s = try r.gpa.create(Seq);
    errdefer r.gpa.destroy(s);
    s.* = try r.initSeqValue(r.max_len);
    return s;
}

pub fn freeSeq(r: *Runner, s: *Seq) void {
    if (s == &r.own) return;
    if (r.cur == s) r.cur = &r.own;
    r.syncAll() catch {};
    const conv_b = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b = r.shp.nv * 128 * 128 * 4;
    const sproj_b = max_verify_rows * r.shp.gdn_pw * 2;
    const kv_b = s.cap * r.shp.nkv * head_dim * 2;
    const ik_b = s.cap * index_dim * 2;
    const pl_b = @max(((s.cap + index_ratio - 1) / index_ratio) * index_dim * 2, 64);
    for (0..r.tp) |rk| {
        const d = &r.devs[rk];
        const rs = &s.ranks[rk];
        for (0..linear_n) |li| {
            d.freeBuf(rs.conv[li], conv_b);
            d.freeBuf(rs.state[li], st_b);
            d.freeBuf(rs.saved_proj[li], sproj_b);
            d.freeBuf(rs.tmp_state[li], st_b);
        }
        for (0..attn_n) |ai| {
            d.freeBuf(rs.kc[ai], kv_b);
            d.freeBuf(rs.vc[ai], kv_b);
            d.freeBuf(rs.ikc[ai], ik_b);
            d.freeBuf(rs.pooled[ai], pl_b);
        }
        d.freeBuf(rs.ple_tail, ple_tail_n * wide * 2);
        d.freeBuf(rs.saved_ple_nrow, max_verify_rows * wide * 2);
        if (rs.mtp_kc) |p| d.freeBuf(p, kv_b);
        if (rs.mtp_vc) |p| d.freeBuf(p, kv_b);
        if (rs.mtp_ikc) |p| d.freeBuf(p, ik_b);
        if (rs.mtp_pooled) |p| d.freeBuf(p, pl_b);
        d.freeBuf(rs.last_streams, wide * 2);
        d.freeBuf(rs.verify_streams, max_verify_rows * wide * 2);
        d.freeBuf(rs.mtp_out, wide * 2);
        r.gpus[rk] = d.g;
    }
    r.gpa.destroy(s);
}

pub fn bind(r: *Runner, s: *Seq) void {
    r.cur = s;
}

pub fn resetSeq(r: *Runner, s: *Seq) !void {
    try r.syncAll();
    const conv_b = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b = r.shp.nv * 128 * 128 * 4;
    for (0..r.tp) |rk| {
        const d = &r.devs[rk];
        const rs = &s.ranks[rk];
        for (0..linear_n) |li| {
            try d.zeroBuf(rs.conv[li], conv_b);
            try d.zeroBuf(rs.state[li], st_b);
        }
        try d.zeroBuf(rs.ple_tail, ple_tail_n * wide * 2);
    }
    s.pos = 0;
    s.history = .{ r.eos, r.eos };
    s.win_open = false;
    s.win_rows = 0;
}

// Prompt Cache Snapshots

pub fn snapBytes(r: *const Runner, at: u32) u64 {
    const conv_b: u64 = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b: u64 = r.shp.nv * 128 * 128 * 4;
    const kv_b: u64 = @as(u64, at) * r.shp.nkv * head_dim * 2;
    const ik_b: u64 = @as(u64, at) * index_dim * 2;
    const pl_b: u64 = @as(u64, (at + 3) / 4) * index_dim * 2;
    const attn_total: u64 = attn_n + @as(u64, @intFromBool(r.has_mtp));
    const per_rank = linear_n * (conv_b + st_b) + attn_total * (2 * kv_b + ik_b + pl_b) + (ple_tail_n + 1) * wide * 2;
    return per_rank * @as(u64, r.tp);
}

pub fn snapSave(r: *Runner, at: u32) !*Snap {
    const s = r.cur;
    if (s.pos != at or s.win_open) return error.SnapshotOutOfStep;
    try r.syncAll();
    const sn = try r.gpa.create(Snap);
    sn.* = .{ .at = at, .history = s.history };
    const conv_b = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b = r.shp.nv * 128 * 128 * 4;
    const kv_b = @as(usize, at) * r.shp.nkv * head_dim * 2;
    const ik_b = @as(usize, at) * index_dim * 2;
    const pl_b = @as(usize, (at + 3) / 4) * index_dim * 2;
    for (0..r.tp) |rk| {
        const d = &r.devs[rk];
        const rs = &s.ranks[rk];
        for (0..linear_n) |li| {
            sn.conv[rk][li] = try d.allocBuf(conv_b);
            sn.state[rk][li] = try d.allocBuf(st_b);
            try d.copy(sn.conv[rk][li], rs.conv[li], conv_b);
            try d.copy(sn.state[rk][li], rs.state[li], st_b);
        }
        for (0..attn_n) |ai| {
            sn.kc[rk][ai] = try d.allocBuf(@max(kv_b, 64));
            sn.vc[rk][ai] = try d.allocBuf(@max(kv_b, 64));
            sn.ikc[rk][ai] = try d.allocBuf(@max(ik_b, 64));
            sn.pooled[rk][ai] = try d.allocBuf(@max(pl_b, 64));
            if (kv_b > 0) try d.copy(sn.kc[rk][ai], rs.kc[ai], kv_b);
            if (kv_b > 0) try d.copy(sn.vc[rk][ai], rs.vc[ai], kv_b);
            if (ik_b > 0) try d.copy(sn.ikc[rk][ai], rs.ikc[ai], ik_b);
            if (pl_b > 0) try d.copy(sn.pooled[rk][ai], rs.pooled[ai], pl_b);
        }
        sn.ple_tail[rk] = try d.allocBuf(ple_tail_n * wide * 2);
        try d.copy(sn.ple_tail[rk], rs.ple_tail, ple_tail_n * wide * 2);
        sn.last_streams[rk] = try d.allocBuf(wide * 2);
        try d.copy(sn.last_streams[rk], rs.last_streams, wide * 2);
        if (r.has_mtp) {
            sn.mtp_kc[rk] = try d.allocBuf(@max(kv_b, 64));
            sn.mtp_vc[rk] = try d.allocBuf(@max(kv_b, 64));
            sn.mtp_ikc[rk] = try d.allocBuf(@max(ik_b, 64));
            sn.mtp_pooled[rk] = try d.allocBuf(@max(pl_b, 64));
            if (kv_b > 0) try d.copy(sn.mtp_kc[rk].?, rs.mtp_kc.?, kv_b);
            if (kv_b > 0) try d.copy(sn.mtp_vc[rk].?, rs.mtp_vc.?, kv_b);
            if (ik_b > 0) try d.copy(sn.mtp_ikc[rk].?, rs.mtp_ikc.?, ik_b);
            if (pl_b > 0) try d.copy(sn.mtp_pooled[rk].?, rs.mtp_pooled.?, pl_b);
        }
        r.gpus[rk] = d.g;
    }
    try r.syncAll();
    return sn;
}

pub fn snapRestore(r: *Runner, sn: *const Snap) !void {
    const s = r.cur;
    try r.syncAll();
    const conv_b = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b = r.shp.nv * 128 * 128 * 4;
    const kv_b = @as(usize, sn.at) * r.shp.nkv * head_dim * 2;
    const ik_b = @as(usize, sn.at) * index_dim * 2;
    const pl_b = @as(usize, (sn.at + 3) / 4) * index_dim * 2;
    for (0..r.tp) |rk| {
        const d = &r.devs[rk];
        const rs = &s.ranks[rk];
        for (0..linear_n) |li| {
            try d.copy(rs.conv[li], sn.conv[rk][li], conv_b);
            try d.copy(rs.state[rk * 0 + li], sn.state[rk][li], st_b);
        }
        for (0..attn_n) |ai| {
            if (kv_b > 0) try d.copy(rs.kc[ai], sn.kc[rk][ai], kv_b);
            if (kv_b > 0) try d.copy(rs.vc[ai], sn.vc[rk][ai], kv_b);
            if (ik_b > 0) try d.copy(rs.ikc[ai], sn.ikc[rk][ai], ik_b);
            if (pl_b > 0) try d.copy(rs.pooled[ai], sn.pooled[rk][ai], pl_b);
        }
        try d.copy(rs.ple_tail, sn.ple_tail[rk], ple_tail_n * wide * 2);
        try d.copy(rs.last_streams, sn.last_streams[rk], wide * 2);
        if (r.has_mtp) {
            if (kv_b > 0) try d.copy(rs.mtp_kc.?, sn.mtp_kc[rk].?, kv_b);
            if (kv_b > 0) try d.copy(rs.mtp_vc.?, sn.mtp_vc[rk].?, kv_b);
            if (ik_b > 0) try d.copy(rs.mtp_ikc.?, sn.mtp_ikc[rk].?, ik_b);
            if (pl_b > 0) try d.copy(rs.mtp_pooled.?, sn.mtp_pooled[rk].?, pl_b);
        }
    }
    try r.syncAll();
    s.pos = sn.at;
    s.history = sn.history;
    s.win_open = false;
    s.win_rows = 0;
}

pub fn snapDrop(r: *Runner, sn: *Snap) void {
    r.syncAll() catch {};
    const conv_b = (gdn_taps - 1) * r.shp.gdn_conv * 2;
    const st_b = r.shp.nv * 128 * 128 * 4;
    const kv_b = @max(@as(usize, sn.at) * r.shp.nkv * head_dim * 2, 64);
    const ik_b = @max(@as(usize, sn.at) * index_dim * 2, 64);
    const pl_b = @max(@as(usize, (sn.at + 3) / 4) * index_dim * 2, 64);
    for (0..r.tp) |rk| {
        const d = &r.devs[rk];
        for (0..linear_n) |li| {
            d.freeBuf(sn.conv[rk][li], conv_b);
            d.freeBuf(sn.state[rk][li], st_b);
        }
        for (0..attn_n) |ai| {
            d.freeBuf(sn.kc[rk][ai], kv_b);
            d.freeBuf(sn.vc[rk][ai], kv_b);
            d.freeBuf(sn.ikc[rk][ai], ik_b);
            d.freeBuf(sn.pooled[rk][ai], pl_b);
        }
        d.freeBuf(sn.ple_tail[rk], ple_tail_n * wide * 2);
        d.freeBuf(sn.last_streams[rk], wide * 2);
        if (sn.mtp_kc[rk]) |p| d.freeBuf(p, kv_b);
        if (sn.mtp_vc[rk]) |p| d.freeBuf(p, kv_b);
        if (sn.mtp_ikc[rk]) |p| d.freeBuf(p, ik_b);
        if (sn.mtp_pooled[rk]) |p| d.freeBuf(p, pl_b);
        r.gpus[rk] = d.g;
    }
    r.gpa.destroy(sn);
}
