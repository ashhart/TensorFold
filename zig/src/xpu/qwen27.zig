//! Qwen3.8-27B forward on Intel Xe2 over Level Zero (MLX affine 4-bit g64 / GPTQ 4-bit g128), TP={1,2,4}.
const std = @import("std");
pub const ze = @import("ze");
const kb = @import("xpu_qwen27_kernels");
const dev = @import("dev.zig");
const device_id = @import("device_id");
const fwd = @import("forward.zig");
const loader = @import("load.zig");
const multi = @import("runner_multi.zig");
const tree = @import("runner_tree.zig");
const slot_states = @import("slot_states.zig");
const shape = @import("qwen27_config.zig");
pub const tp_mod = @import("xpu_tp");
pub const limits = @import("limits.zig");
const Prof = dev.Prof;
const a8 = fwd.a8;
const forwardRunner = fwd.forwardRunner;
const kvCap = fwd.kvCap;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn close(fd: c_int) c_int;
extern "c" fn munmap(addr: *anyopaque, len: usize) c_int;
pub const MAX_SLOTS = loader.MAX_SLOTS;
pub const freeAfterQueue = slot_states.freeAfterQueue;
pub const Tensor = dev.Tensor;
pub const Checkpoint = dev.Checkpoint;
pub const Dev = dev.Dev;
pub const copy = dev.copy;
pub const off = dev.off;
pub const profReport = dev.profReport;
pub const Proj = loader.Proj;
pub const Model = loader.Model;
pub const Acts = fwd.Acts;
pub const cand = @import("cand.zig");
pub const embedRows = fwd.embedRows;
pub const gemm = fwd.gemm;
pub const Want = fwd.Want;
pub const no_q8 = fwd.no_q8;
pub const Mode = fwd.Mode;

pub const alloc = std.heap.c_allocator;

pub const D = shape.hidden;
pub const FF = shape.intermediate;
pub const LAYERS = shape.layers;
pub const VOCAB = shape.vocab;
pub const MASK_WORDS = (VOCAB + 31) / 32;
pub var GS: u32 = 64;
pub var TP: u32 = 1;
pub const GDN_IN = 16640;
pub const ATT_IN = 14336;
pub var CHUNK: u32 = 1024;

pub const TpShape = struct {
    ff: u32,
    gk: u32,
    gv: u32,
    nqk: u32,
    vd: u32,
    nconv: u32,
    ga: u32,
    gb: u32,
    gdn_in: u32,
    aq: u32,
    akv: u32,
    od: u32,
    att_in: u32,
    st_q: usize,
    st_k: usize,
    st_v: usize,
    st_g: usize,
    st_b: usize,
    st_p: usize,
    stash: usize,
};

pub fn tpShapeFor(tp: u32) TpShape {
    const ff = FF / tp;
    const gk = 16 / tp;
    const gv = 48 / tp;
    const nqk = gk * 128;
    const vd = gv * 128;
    const nconv = 2 * nqk + vd;
    const ga = nconv + vd;
    const gb = ga + gv;
    const gdn_in = (gb + gv + 255) / 256 * 256;
    const aq = 24 / tp;
    const akv = 4 / tp;
    const od = aq * 256;
    const att_in = aq * 512 + 2 * akv * 256;
    const st_q: usize = 0;
    const st_k: usize = st_q + 16 * @as(usize, nqk) * 2;
    const st_v: usize = st_k + 16 * @as(usize, nqk) * 2;
    const st_g: usize = st_v + 16 * @as(usize, vd) * 2;
    const st_b: usize = st_g + 16 * @as(usize, gv) * 4;
    const st_p: usize = st_b + 16 * @as(usize, gv) * 4;
    const stash: usize = st_p + 16 * @as(usize, gdn_in) * 2;
    return .{
        .ff = ff,
        .gk = gk,
        .gv = gv,
        .nqk = nqk,
        .vd = vd,
        .nconv = nconv,
        .ga = ga,
        .gb = gb,
        .gdn_in = gdn_in,
        .aq = aq,
        .akv = akv,
        .od = od,
        .att_in = att_in,
        .st_q = st_q,
        .st_k = st_k,
        .st_v = st_v,
        .st_g = st_g,
        .st_b = st_b,
        .st_p = st_p,
        .stash = stash,
    };
}

pub fn tpShape() TpShape {
    return tpShapeFor(TP);
}

pub fn maxRows() u32 {
    return CHUNK + 128;
}
pub fn chunkEnd(at: usize, len: usize) usize {
    return if (len - at < CHUNK + 128) len else at + CHUNK;
}
pub var MAX_CTX: u32 = 4096;
pub const SPLIT = 32;
pub const EPS: f32 = 1e-6;

pub const ST_Q = 0;
pub const ST_K = ST_Q + 16 * 16 * 128 * 2;
pub const ST_V = ST_K + 16 * 16 * 128 * 2;
pub const ST_G = ST_V + 16 * 48 * 128 * 2;
pub const ST_B = ST_G + 16 * 48 * 4;
pub const ST_P = ST_B + 16 * 48 * 4;
pub const STASH = ST_P + 16 * GDN_IN * 2;
pub const TAP_LAYERS = [5]usize{ 5, 19, 33, 47, 61 };

pub const isAttention = shape.isAttention;

pub var dump_dir: ?[]const u8 = null;
pub var verbose = false;
pub var dump_h: ?c_int = null;
pub var prompt_tail_on = true;
pub var a8_group: u32 = 0;
pub var kv8: i32 = 0;
pub var kv_i8 = false;
pub var rot_on: i32 = 0;
pub var prefill_int4 = false;
pub var int4_sites: u32 = 8;
pub var prefill_int8 = false;
pub var a8_sites: u32 = 15;
pub var a8_rot: i32 = 0;
pub var a8_levels: i32 = 127;
pub var a8_sites4: u32 = 15;
pub var prof: Prof = .{};

const Options = struct {
    dump_dir: ?[]const u8,
    dump_h: ?c_int,
    a8_group: u32,
    kv8: i32,
    kv_i8: bool,
    prefill_int4: bool,
    int4_sites: u32,
    prefill_int8: bool,
    a8_sites: u32,
    a8_rot: i32,
    a8_levels: i32,
    a8_sites4: u32,
    chunk: u32,
    tp: u32,
};
var defaults: ?Options = null;
var live = false;

fn options() Options {
    return .{ .dump_dir = dump_dir, .dump_h = dump_h, .a8_group = a8_group, .kv8 = kv8, .kv_i8 = kv_i8, .prefill_int4 = prefill_int4, .int4_sites = int4_sites, .prefill_int8 = prefill_int8, .a8_sites = a8_sites, .a8_rot = a8_rot, .a8_levels = a8_levels, .a8_sites4 = a8_sites4, .chunk = CHUNK, .tp = TP };
}

fn restore(o: Options) void {
    dump_dir = o.dump_dir;
    dump_h = o.dump_h;
    a8_group = o.a8_group;
    kv8 = o.kv8;
    kv_i8 = o.kv_i8;
    prefill_int4 = o.prefill_int4;
    int4_sites = o.int4_sites;
    prefill_int8 = o.prefill_int8;
    a8_sites = o.a8_sites;
    a8_rot = o.a8_rot;
    a8_levels = o.a8_levels;
    a8_sites4 = o.a8_sites4;
    CHUNK = o.chunk;
    TP = o.tp;
}

extern "c" fn clock_gettime(clk: c_int, ts: *[2]i64) c_int;
pub fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

fn initDevKernels(d: *Dev, tp: u32) !void {
    d.list = try d.g.immediate(0, 0);
    const glue_bin: []const u8 = switch (tp) {
        1 => &kb.glue,
        2 => &kb.glue_tp2,
        4 => &kb.glue_tp4,
        else => return error.InvalidTpDegree,
    };
    const dec_bin: []const u8 = switch (tp) {
        1 => &kb.decode,
        2 => &kb.decode_tp2,
        4 => &kb.decode_tp4,
        else => return error.InvalidTpDegree,
    };
    const mg = try d.g.module(glue_bin);
    const md = try d.g.module(dec_bin);
    const mm = try d.g.module(&kb.gemm);
    const mgd = try d.g.module(&kb.gdn);
    const mw = try d.g.module(&kb.ngen_wide);
    const mn = try d.g.module(&kb.ngen_narrow);
    const mm128 = try d.g.module(&kb.gemm128);
    const mw128 = try d.g.module(&kb.ngen_wide128);
    const mn128 = try d.g.module(&kb.ngen_narrow128);
    d.k = .{
        .embed = try d.kernel(mg, "embed", 256),
        .add_rmsnorm = try d.kernel(mg, "add_rmsnorm", 128),
        .add_rmsnorm_dec = try d.kernel(mg, "add_rmsnorm_dec", 512),
        .gdn_pre = try d.kernel(mg, "gdn_pre", 256),
        .gdn_pre_rows = try d.kernel(mg, "gdn_pre_rows", 256),
        .gated_norm = try d.kernel(mg, "gated_norm", 256),
        .attn_prep = try d.kernel(mg, "attn_prep", 256),
        .attn_gate = try d.kernel(mg, "attn_gate", 256),
        .swiglu = try d.kernel(mg, "swiglu", 256),
        .argmax = try d.kernel(mg, "argmax", 256),
        .argmax_part = try d.kernel(mg, "argmax_part", 256),
        .argmax_part_masked = try d.kernel(mg, "argmax_part_masked", 256),
        .topk_part = try d.kernel(mg, "topk_part", 256),
        .topk_merge = try d.kernel(mg, "topk_merge", 256),
        .tap = try d.kernel(mg, "tap", 256),
        .conv_commit = try d.kernel(mg, "conv_commit", 256),
        .repack = try d.kernel(mg, "repack", 16),
        .gdn_step = try d.kernel(md, "gdn_step", 16),
        .attn_decode = try d.kernel(md, "attn_decode", 16),
        .attn_merge = try d.kernel(md, "attn_merge", 16),
        .kv_move = try d.kernel(md, "kv_move", 16),
        .exl3_gemv = try d.kernel(md, "exl3_gemv", 16),
        .rot_heads = try d.kernel(mg, "rot_heads", 16),
        .attn_prompt = try d.kernel(md, "attn_prompt", 128),
        .attn_prompt8 = try d.kernel(md, "attn_prompt8", 128),
        .embed_dense = try d.kernel(mg, "embed_dense", 256),
        .dense_rows = try d.kernel(mg, "dense_rows", 256),
        .repack_gptq = try d.kernel(mg, "repack_gptq", 16),
        .quantize_mlx = try d.kernel(mg, "quantize_mlx", 16),
        .dec_wide_g128 = try d.kernel(mm128, "w4k1_decode", 64),
        .dec_wide64_g128 = try d.kernel(mm128, "w4k1_d64", 64),
        .dec_narrow_g128 = try d.kernel(mm128, "w4k_dsplit", 64),
        .ngen_wide_g128 = try d.kernel(mw128, "w4g_dpb", 256),
        .ngen_narrow_g128 = try d.kernel(mn128, "w4g_s4dpb", 256),
        .ngen_swiglu = try d.kernel(try d.g.module(&kb.ngen_swiglu), "w4g_dpb_swiglu", 256),
        .ngen_swiglu_g128 = try d.kernel(try d.g.module(&kb.ngen_swiglu128), "w4g_dpb_swiglu", 256),
        .rowsums_ff = try d.kernel(mg, "rowsums_ff", 128),
        .fakequant_rows = try d.kernel(mg, "fakequant_rows", 256),
        .a8 = try d.kernel(try d.g.module(&kb.ngen_a8), "w4a8", 256),
        .a8_swiglu = try d.kernel(try d.g.module(&kb.ngen_a8_swiglu), "w4a8_swiglu", 256),
        .quant_a8 = try d.kernel(mg, "quant_a8", 128),
        .quant_a4 = try d.kernel(mg, "quant_a4", 128),
        .a4 = try d.kernel(try d.g.module(&kb.ngen_a4), "w4a4", 256),
        .dec_wide = try d.kernel(mm, "w4k1_decode", 64),
        .dec_wide64 = try d.kernel(mm, "w4k1_d64", 64),
        .dec_narrow = try d.kernel(mm, "w4k_dsplit", 64),
        .gdn_prep = try d.kernel(mgd, "gdn_prep", 64),
        .gdn_state = try d.kernel(mgd, "gdn_state", 256),
        .gdn_out = try d.kernel(mgd, "gdn_out", 128),
        .ngen_wide = try d.kernel(mw, "w4g_dpb", 256),
        .ngen_narrow = try d.kernel(mn, "w4g_s4dpb", 256),
    };
    for ([_]ze.H{ d.k.attn_prep, d.k.attn_decode, d.k.gdn_pre, d.k.gdn_step }) |k| try d.g.indirect(k);
}

pub const Runner = struct {
    tp: u32 = 1,
    d: Dev,
    model: Model,
    acts: Acts,
    peers_d: [3]Dev = undefined,
    peers_model: [3]Model = undefined,
    peers_acts: [3]Acts = undefined,
    comm: tp_mod.TpComm = undefined,
    vocab_off: [4]u32 = .{ 0, 0, 0, 0 },
    vocab_len: [4]u32 = .{ VOCAB, 0, 0, 0 },
    logits_ev: [4]ze.H = @splat(null),
    pos: u32 = 0,
    last: u32 = 0,
    context: u32,
    ckpt: Checkpoint = .{},
    host_logits: []f32 = &.{},
    block: [16]u32 = @splat(0),
    tok_host: [16]i32 = @splat(0), // the last forward's greedy token of every row, as the card wrote them
    cand: cand.State = .{}, // the card top-k of the last step or verify, and how many candidates the next one keeps
    block_rows: u32 = 0,
    tree_on: bool = false,
    tree_host: [16 * 17]i32 = @splat(0),
    path_host: [16]i32 = @splat(0),
    slot: u32 = 0,
    mt_hosts: [4][LAYERS * 16 * 8]u64 = @splat(@splat(0)),
    pos_host: [16]i32 = @splat(0),
    win_host: [32]i32 = @splat(0),
    ids_host: [16]u32 = @splat(0),
    multi_slots: [16]u32 = @splat(0),
    multi_count: u32 = 0,
    multi_tree: bool = false,
    mpath_host: [16 * 2 * 16]i32 = @splat(0),
    seqs: [MAX_SLOTS]struct { pos: u32 = 0, last: u32 = 0 } = @splat(.{}),
    taps_on: bool = false,
    on_taps: ?struct { ctx: *anyopaque, f: *const fn (*anyopaque, *anyopaque, u32) anyerror!void } = null,

    pub fn devAt(r: *Runner, rank: usize) *Dev {
        return if (rank == 0) &r.d else &r.peers_d[rank - 1];
    }

    pub fn modelAt(r: *Runner, rank: usize) *Model {
        return if (rank == 0) &r.model else &r.peers_model[rank - 1];
    }

    pub fn actsAt(r: *Runner, rank: usize) *Acts {
        return if (rank == 0) &r.acts else &r.peers_acts[rank - 1];
    }

    pub fn settleAll(r: *Runner) void {
        for (0..r.tp) |rk| r.devAt(rk).settle();
    }

    pub fn syncAll(r: *Runner) !void {
        if (r.tp == 1) return r.d.sync();
        var gpus: [4]ze.Gpu = undefined;
        var lists: [4]ze.H = undefined;
        for (0..r.tp) |rk| {
            const d = r.devAt(rk);
            gpus[rk] = d.g;
            lists[rk] = d.list;
        }
        try r.comm.syncAll(gpus[0..r.tp], lists[0..r.tp]);
        for (0..r.tp) |rk| {
            try ze.check(r.d.g.api.zeEventHostReset(r.logits_ev[rk]));
        }
        if (prof.on) try r.d.sync();
    }

    pub fn init(dir: []const u8, context: u32) !*Runner {
        return initTp(dir, context, 0);
    }

    /// Loads `dir` onto `tp_arg` (1, 2 or 4; 0 reads `XPU_TP` else 1) Arc Pro B70 GPUs.
    pub fn initTp(dir: []const u8, context: u32, tp_arg: u32) !*Runner {
        if (live) return error.RunnerOpen;
        if (defaults) |o| restore(o) else defaults = options();
        live = true;
        errdefer live = false;
        const sys = @import("sys");
        const tp_deg: u32 = if (tp_arg != 0) tp_arg else try sys.environment.read(u32, "XPU_TP", 1);
        if (tp_deg != 1 and tp_deg != 2 and tp_deg != 4) return error.InvalidTpDegree;
        TP = tp_deg;

        if (std.c.getenv("XPU_DUMP")) |v| dump_dir = std.mem.span(v);
        if (std.c.getenv("XPU_PREFILL_INT8") != null) prefill_int8 = true;
        if (std.c.getenv("XPU_KV_INT8") != null) kv_i8 = true;
        if (std.c.getenv("XPU_PREFILL_INT4") != null) prefill_int4 = true;
        if (std.c.getenv("XPU_PREFILL_INT4_SITES")) |v| int4_sites = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_KV8")) |v| kv8 = try std.fmt.parseInt(i32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8_ROT") != null) a8_rot = 1;
        if (std.c.getenv("XPU_A8_SITES4")) |v| a8_sites4 = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        const bits = try sys.environment.read(u32, "XPU_A8_BITS", if (a8_levels == 7) 4 else 8);
        a8_levels = if (bits == 4) 7 else 127;
        if (std.c.getenv("XPU_A8_SITES")) |v| a8_sites = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8")) |v| a8_group = if (std.mem.eql(u8, std.mem.span(v), "token")) std.math.maxInt(u32) else try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (try sys.environment.path("XPU_DUMP_H", sys.getenv("XPU_DUMP_H"))) |p| {
            const pz = try std.fmt.allocPrintSentinel(alloc, "{s}", .{p}, 0);
            defer alloc.free(pz);
            dump_h = open(pz.ptr, 0o2101, @as(c_uint, 0o644));
        }
        if (std.c.getenv("XPU_CHUNK")) |v| CHUNK = try limits.chunk(std.mem.span(v));
        MAX_CTX = try limits.context(context);
        try shape.admit(alloc, dir);
        {
            const marker = try std.fmt.allocPrintSentinel(alloc, "{s}/hadamard128", .{dir}, 0);
            defer alloc.free(marker);
            const fd = open(marker.ptr, 0);
            rot_on = if (fd >= 0) 1 else 0;
            if (fd >= 0) _ = close(fd);
        }
        if (prefill_int4 and (rot_on == 0 or !prefill_int8)) return error.Int4NeedsRotatedWeightsAndInt8;
        var ckpt: Checkpoint = .{};
        errdefer ckpt.deinit();
        {
            const index_path = try std.fmt.allocPrintSentinel(alloc, "{s}/model.safetensors.index.json", .{dir}, 0);
            defer alloc.free(index_path);
            const fd = open(index_path.ptr, 0);
            if (fd < 0) return error.FileNotFound;
            defer _ = close(fd);
            const end = lseek(fd, 0, 2);
            if (end <= 0) return error.BadIndex;
            const size: usize = @intCast(end);
            const base = mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed;
            if (@intFromPtr(base) == std.math.maxInt(usize)) return error.MmapFailed;
            defer _ = munmap(base, size);
            const parsed = std.json.parseFromSlice(std.json.Value, alloc, @as([*]const u8, @ptrCast(base))[0..size], .{}) catch return error.BadIndex;
            defer parsed.deinit();
            const map = if (parsed.value == .object) parsed.value.object.get("weight_map") else null;
            if (map == null or map.? != .object) return error.BadIndex;
            var seen: std.ArrayList([]const u8) = .empty;
            defer seen.deinit(alloc);
            var it = map.?.object.iterator();
            while (it.next()) |e| {
                if (e.value_ptr.* != .string) return error.BadIndex;
                const f = e.value_ptr.string;
                for (seen.items) |s| {
                    if (std.mem.eql(u8, s, f)) break;
                } else {
                    try seen.append(alloc, f);
                    const shard = try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, f }, 0);
                    defer alloc.free(shard);
                    try ckpt.addShard(shard);
                }
            }
        }

        var open_gpus: [4]ze.Gpu = undefined;
        try device_id.openPeers(open_gpus[0..tp_deg]);
        errdefer ze.Gpu.closePeers(open_gpus[0..tp_deg]);

        const r = try alloc.create(Runner);
        errdefer alloc.destroy(r);
        r.* = .{
            .tp = tp_deg,
            .d = .{ .g = open_gpus[0], .list = undefined, .k = undefined },
            .model = undefined,
            .acts = undefined,
            .context = context,
            .ckpt = ckpt,
        };
        for (1..tp_deg) |rk| {
            r.peers_d[rk - 1] = .{ .g = open_gpus[rk], .list = undefined, .k = undefined };
        }

        // Partition VOCAB across TP ranks in multiples of 256 (970 tiles total: 970, 485+485, or 243+243+242+242)
        const tiles_total: u32 = VOCAB / 256;
        var off_cur: u32 = 0;
        for (0..tp_deg) |rk| {
            const base_t = tiles_total / tp_deg;
            const extra_t: u32 = if (rk < (tiles_total % tp_deg)) 1 else 0;
            const vlen = (base_t + extra_t) * 256;
            r.vocab_off[rk] = off_cur;
            r.vocab_len[rk] = vlen;
            off_cur += vlen;
        }

        for (0..tp_deg) |rk| try initDevKernels(r.devAt(rk), tp_deg);

        if (std.c.getenv("XPU_PROFILE") != null) {
            prof.on = true;
            prof.events = try r.d.g.events(4096);
            inline for (@typeInfo(@TypeOf(r.d.k)).@"struct".field_names, 0..) |name, i| prof.names[i] = .{ .k = @field(r.d.k, name), .name = name };
        }
        const prof_on = prof.on;
        prof.on = false;

        const sh = tpShape();
        for (0..tp_deg) |rk| {
            const d = r.devAt(rk);
            r.modelAt(rk).* = try loader.loadRank(d, &ckpt, @intCast(rk), r.vocab_off[rk], r.vocab_len[rk]);
            r.actsAt(rk).* = try Acts.initRank(d, @intCast(rk), r.vocab_len[rk]);
            try d.fill(r.actsAt(rk).zeros, @as(u16, 0), maxRows() * @as(usize, sh.gv) * 128 * 2);
            try d.fill(r.actsAt(rk).ones, @as(f32, 1.0), maxRows() * @as(usize, sh.gv) * 4);
        }
        prof.on = prof_on;

        if (tp_deg > 1) {
            var gpus_slice: [4]ze.Gpu = undefined;
            for (0..tp_deg) |rk| gpus_slice[rk] = r.devAt(rk).g;
            r.comm = try tp_mod.TpComm.init(gpus_slice[0..tp_deg], &kb.tp, @as(usize, maxRows()) * D * 2);
            // Copy newly appended pools/events back onto rank 0's Gpu ownership list
            r.d.g = gpus_slice[0];
            for (1..tp_deg) |rk| r.peers_d[rk - 1].g = gpus_slice[rk];

            // Make each rank's shard_logits resident on rank 0 for the final P2P logit gather
            for (0..tp_deg) |rk| {
                try r.d.g.makeResident(r.actsAt(rk).shard_logits, 16 * @as(usize, r.vocab_len[rk]) * 2);
            }
            for (0..tp_deg) |rk| gpus_slice[rk] = r.devAt(rk).g;
            try ze.Gpu.sharedEvents(gpus_slice[0..tp_deg], r.logits_ev[0..tp_deg]);
            r.d.g = gpus_slice[0];
        }

        r.store();
        r.host_logits = try alloc.alloc(f32, VOCAB);
        return r;
    }

    pub fn deinit(r: *Runner) void {
        r.settleAll();
        for (1..r.tp) |rk| if (r.peers_d[rk - 1].staging.len > 0) r.peers_d[rk - 1].g.free(r.peers_d[rk - 1].staging.ptr);
        if (r.d.staging.len > 0) r.d.g.free(r.d.staging.ptr);
        var gpus: [4]ze.Gpu = undefined;
        for (0..r.tp) |rk| gpus[rk] = r.devAt(rk).g;
        ze.Gpu.closePeers(gpus[0..r.tp]);
        prof.on = false;
        live = false;
        r.ckpt.deinit();
        if (r.host_logits.len > 0) alloc.free(r.host_logits);
        alloc.destroy(r);
    }

    pub const select = tree.select;
    pub const store = tree.store;
    pub const Win = multi.Win;
    pub const verifyMulti = multi.verifyMulti;
    pub const keepMulti = multi.keepMulti;
    pub const keepPaths = multi.keepPaths;
    pub const stepMulti = multi.stepMulti;

    pub const Snap = slot_states.Snap;
    pub const slotBytes = slot_states.slotBytes;
    pub const snapBytes = slot_states.snapBytes;
    pub const saveSlot = slot_states.saveSlot;
    pub const restoreSlot = slot_states.restoreSlot;
    pub const dropSnap = slot_states.dropSnap;
    pub const unallocatedSlots = slot_states.unallocatedSlots;
    pub const ensure = slot_states.ensure;

    pub const reset = tree.reset;
    pub const step = tree.step;
    pub const verify = tree.verify;

    pub fn prompt(r: *Runner, ids: []const u32, last: bool) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const end = chunkEnd(at, ids.len);
            const want = last and end == ids.len;
            r.cand.rows = 0;
            if (r.pos + (end - at) > r.context) return error.ContextFull;
            if (try forwardRunner(r, @ptrCast(ids[at..end]), r.pos, .{ .prompt = true, .want = if (want) .last else .none, .taps = r.taps_on })) |t| r.last = @intCast(t);
            if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, @intCast(end - at));
            r.pos += @intCast(end - at);
            at = end;
        }
    }

    pub const commit = tree.commit;
    pub const verifyTree = tree.verifyTree;
    pub const commitPath = tree.commitPath;
    pub const argmaxMasked = tree.argmaxMasked;
    pub const candidates = cand.candidates;
    pub const candMode = cand.mode;
    pub const rowLogits = tree.rowLogits;
    pub const logits = tree.logits;

    pub fn deviceName(r: *Runner) []const u8 {
        return std.mem.sliceTo(&r.d.g.props.name, 0);
    }
};

pub const drafter = @import("drafter.zig");

test "every open starts from the process's options, and a second runner is refused while one is open" {
    const before = options();
    defer {
        restore(before);
        defaults = null;
        live = false;
    }
    defaults = before;
    prefill_int8 = true;
    CHUNK = 64;
    kv_i8 = true;
    TP = 4;
    restore(defaults.?);
    try std.testing.expect(!prefill_int8 and !kv_i8);
    try std.testing.expectEqual(before.chunk, CHUNK);
    try std.testing.expectEqual(before.tp, TP);
    live = true;
    try std.testing.expectError(error.RunnerOpen, Runner.init("/nonexistent", 4096));
}

test {
    _ = @import("prompt_tail.zig");
    _ = @import("runner_drain_test.zig");
}
