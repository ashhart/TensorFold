//! Qwen3.8-27B forward on Intel Xe2 over Level Zero (MLX affine 4-bit g64), chunked prompt, serial decode.
const std = @import("std");
pub const ze = @import("ze.zig");
const kb = @import("xpu_qwen27_kernels");
const dev = @import("dev.zig");
const fwd = @import("forward.zig");
const loader = @import("load.zig");
const multi = @import("runner_multi.zig");
const slot_states = @import("slot_states.zig");
const Prof = dev.Prof;
const a8 = fwd.a8;
const forward = fwd.forward;
const kvCap = fwd.kvCap;
const load = loader.load;
extern "c" fn lseek(fd: c_int, off: i64, whence: c_int) i64;
extern "c" fn mmap(addr: ?*anyopaque, len: usize, prot: c_int, flags: c_int, fd: c_int, off: i64) ?*anyopaque;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
pub const MAX_SLOTS = loader.MAX_SLOTS;
pub const Tensor = dev.Tensor;
pub const Checkpoint = dev.Checkpoint;
pub const Dev = dev.Dev;
pub const copy = dev.copy;
pub const off = dev.off;
pub const profReport = dev.profReport;
pub const Proj = loader.Proj;
pub const Model = loader.Model;
pub const Acts = fwd.Acts;
pub const embedRows = fwd.embedRows;
pub const gemm = fwd.gemm;
pub const Want = fwd.Want;
pub const no_q8 = fwd.no_q8;
pub const Mode = fwd.Mode;

pub const alloc = std.heap.c_allocator;

// Qwen3.8-27B shape (checked against config.json at load).
pub const D = 5120;
pub const FF = 17408;
pub const LAYERS = 64;
pub const VOCAB = 248320;
pub const MASK_WORDS = (VOCAB + 31) / 32; // a grammar row's bits (structured output), one a token
/// The layers' quantization group (64: MLX affine, 128: GPTQ), set at load; the head and the drafter keep 64.
pub var GS: u32 = 64;
pub const GDN_IN = 16640; // qkv 10240 | z 6144 | a 48 | b 48 = 16480, padded to the GEMM tile (256)
pub const ATT_IN = 14336; // (q 256 | gate 256) × 24 | k 1024 | v 1024
pub var CHUNK: u32 = 1024; // prompt rows per forward (XPU_CHUNK, a multiple of 64)
pub var MAX_CTX: u32 = 4096; // KV cache rows (XPU_CTX)
pub const SPLIT = 32; // keys per attention split (decode.cl)
pub const EPS: f32 = 1e-6;

// Layout of a GDN layer's stash (bytes): q, k, v, g, β, then the qkv rows of P.
pub const ST_Q = 0;
pub const ST_K = ST_Q + 16 * 16 * 128 * 2;
pub const ST_V = ST_K + 16 * 16 * 128 * 2;
pub const ST_G = ST_V + 16 * 48 * 128 * 2;
pub const ST_B = ST_G + 16 * 48 * 4;
pub const ST_P = ST_B + 16 * 48 * 4;
pub const STASH = ST_P + 16 * GDN_IN * 2;
/// Target layers whose hidden states feed the DFlash2 drafter (its config's target_layer_ids).
pub const TAP_LAYERS = [5]usize{ 5, 19, 33, 47, 61 };

pub fn isAttention(i: usize) bool {
    return (i + 1) % 4 == 0;
}

/// XPU_DUMP=DIR: the first forward writes x (fp32 [T, D]) after each of the first four layers, for comparison.
pub var dump_dir: ?[]const u8 = null;
pub var verbose = false;
pub var dump_h: ?c_int = null;
/// W4A8 study (XPU_A8): prompt GEMM inputs quantized to int8 and back in place; 128 per group, "token" per row.
pub var a8_group: u32 = 0;
/// K/V cache study (XPU_KV8): 1 rotates q, k, v by Hadamard, 2 quantizes k, v to int8 per head, 3 both.
pub var kv8: i32 = 0;
/// The prompt attention: our DPAS kernel (attn_prompt), or the port's Triton one with XPU_ATTN_TRITON=1.
pub var attn_own = true;
/// Int8 K/V cache (XPU_KV_INT8=1): 256 int8 per position and head, then float scales after MAX_CTX positions.
pub var kv_i8 = false;
/// Rotated weights (`hadamard128` in the checkpoint): every GEMM input is rotated by blocks of 128 where it is made.
pub var rot_on: i32 = 0;
/// W4A4 on down (and mixer out projections with XPU_PREFILL_INT4_SITES=10): int4 inputs per row group, DPAS u4 × s4.
pub var prefill_int4 = false;
pub var int4_sites: u32 = 8; // XPU_PREFILL_INT4_SITES: 8 is down only, 2 adds the mixers' out projections
/// W4A8 prompt GEMMs (XPU_PREFILL_INT8, local option, GPTQ g128 only): int8 activations by (row, group of 128).
pub var prefill_int8 = false;
pub var a8_sites: u32 = 15; // XPU_A8_SITES: 1 mixer in, 2 mixer out, 4 gate|up, 8 down
pub var a8_rot: i32 = 0; // XPU_A8_ROT=1: Hadamard by blocks of 128 before the quantization (weights rotated too)
pub var a8_levels: i32 = 127; // XPU_A8_BITS=4: int4 activations (7 levels a side)
pub var a8_sites4: u32 = 15; // XPU_A8_SITES4: the sites in int4 when XPU_A8_BITS=4 (the others in int8)
pub var prof: Prof = .{};

extern "c" fn clock_gettime(clk: c_int, ts: *[2]i64) c_int;
pub fn now() f64 {
    var ts: [2]i64 = undefined;
    _ = clock_gettime(1, &ts);
    return @as(f64, @floatFromInt(ts[0])) + @as(f64, @floatFromInt(ts[1])) * 1e-9;
}

/// The whole model on the device and one sequence's state: what a serial host drives.
pub const Runner = struct {
    d: Dev,
    model: Model,
    acts: Acts,
    pos: u32 = 0,
    last: u32 = 0, // greedy next token of the last forward
    context: u32,
    ckpt: Checkpoint = .{},
    host_logits: []f32 = &.{},
    block: [16]u32 = @splat(0), // the last verified block's greedy tokens
    block_rows: u32 = 0,
    tree_on: bool = false, // the last verified block was a tree (commitPath)
    tree_host: [16 * 17]i32 = @splat(0),
    path_host: [16]i32 = @splat(0),
    slot: u32 = 0, // the selected sequence (select)
    mt_host: [LAYERS * 16 * 8]u64 = @splat(0), // verifyMulti's per-row state addresses, by layer
    pos_host: [16]i32 = @splat(0),
    win_host: [32]i32 = @splat(0), // and each window's first row and width (keepMulti)
    ids_host: [16]u32 = @splat(0),
    multi_slots: [16]u32 = @splat(0), // the last verifyMulti's slots, by window
    multi_count: u32 = 0,
    multi_tree: bool = false, // the last verifyMulti had tree windows: every window's states are replayed at the keep
    mpath_host: [16 * 2 * 16]i32 = @splat(0),
    seqs: [MAX_SLOTS]struct { pos: u32 = 0, last: u32 = 0 } = @splat(.{}),
    /// Target hidden states for a drafter: when on, every forward fills acts.taps and calls `on_taps` with its rows.
    taps_on: bool = false,
    on_taps: ?struct { ctx: *anyopaque, f: *const fn (*anyopaque, *anyopaque, u32) anyerror!void } = null,

    /// Loads `dir` (MLX affine 4/64) onto the first B70 with room for `context` positions.
    pub fn init(dir: []const u8, context: u32) !*Runner {
        if (std.c.getenv("XPU_DUMP")) |v| dump_dir = std.mem.span(v);
        if (std.c.getenv("XPU_PREFILL_INT8") != null) prefill_int8 = true;
        if (std.c.getenv("XPU_ATTN_TRITON") != null) attn_own = false;
        if (std.c.getenv("XPU_KV_INT8") != null) kv_i8 = true;
        if (std.c.getenv("XPU_PREFILL_INT4") != null) prefill_int4 = true;
        if (std.c.getenv("XPU_PREFILL_INT4_SITES")) |v| int4_sites = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_KV8")) |v| kv8 = try std.fmt.parseInt(i32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8_ROT") != null) a8_rot = 1;
        if (std.c.getenv("XPU_A8_SITES4")) |v| a8_sites4 = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8_BITS")) |v| a8_levels = if (std.mem.eql(u8, std.mem.span(v), "4")) 7 else 127;
        if (std.c.getenv("XPU_A8_SITES")) |v| a8_sites = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_A8")) |v| a8_group = if (std.mem.eql(u8, std.mem.span(v), "token")) std.math.maxInt(u32) else try std.fmt.parseInt(u32, std.mem.span(v), 10);
        if (std.c.getenv("XPU_DUMP_H")) |v| dump_h = open(v, 0o2101, @as(c_uint, 0o644)); // append
        if (std.c.getenv("XPU_CHUNK")) |v| CHUNK = try std.fmt.parseInt(u32, std.mem.span(v), 10);
        MAX_CTX = context;
        {
            const marker = try std.fmt.allocPrintSentinel(alloc, "{s}/hadamard128", .{dir}, 0);
            const fd = open(marker.ptr, 0);
            rot_on = if (fd >= 0) 1 else 0;
        }
        if (prefill_int4 and (rot_on == 0 or !prefill_int8)) return error.Int4NeedsRotatedWeightsAndInt8;
        var ckpt: Checkpoint = .{};
        {
            const index_path = try std.fmt.allocPrintSentinel(alloc, "{s}/model.safetensors.index.json", .{dir}, 0);
            const fd = open(index_path.ptr, 0);
            if (fd < 0) return error.FileNotFound;
            const size: usize = @intCast(lseek(fd, 0, 2));
            const text: [*]const u8 = @ptrCast(mmap(null, size, 1, 2, fd, 0) orelse return error.MmapFailed);
            const parsed = try std.json.parseFromSlice(std.json.Value, alloc, text[0..size], .{});
            var seen: std.ArrayList([]const u8) = .empty;
            var it = parsed.value.object.get("weight_map").?.object.iterator();
            while (it.next()) |e| {
                const f = e.value_ptr.string;
                for (seen.items) |s| {
                    if (std.mem.eql(u8, s, f)) break;
                } else {
                    try seen.append(alloc, f);
                    try ckpt.addShard(try std.fmt.allocPrintSentinel(alloc, "{s}/{s}", .{ dir, f }, 0));
                }
            }
        }
        const r = try alloc.create(Runner);
        r.* = .{ .d = .{ .g = try ze.Gpu.open(0xe223), .list = undefined, .k = undefined }, .model = undefined, .acts = undefined, .context = context, .ckpt = ckpt };
        const d = &r.d;
        d.list = try d.g.immediate(0, 0);
        const mg = try d.g.module(&kb.glue);
        const md = try d.g.module(&kb.decode);
        const mm = try d.g.module(&kb.gemm);
        const mgd = try d.g.module(&kb.gdn);
        const mw = try d.g.module(&kb.ngen_wide);
        const mn = try d.g.module(&kb.ngen_narrow);
        const mm128 = try d.g.module(&kb.gemm128);
        const mw128 = try d.g.module(&kb.ngen_wide128);
        const mn128 = try d.g.module(&kb.ngen_narrow128);
        const mpa = try d.g.module(&kb.prompt_attention);
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
            .prompt_attn = try d.kernel(mpa, "_prompt", 512),
        };
        // the shared rounds' kernels reach each slot's states through the per-row table (acts.mt)
        for ([_]ze.H{ d.k.attn_prep, d.k.attn_decode, d.k.gdn_pre, d.k.gdn_step }) |k| try d.g.indirect(k);
        if (std.c.getenv("XPU_PROFILE") != null) {
            prof.on = true;
            prof.events = try d.g.events(4096);
            inline for (@typeInfo(@TypeOf(d.k)).@"struct".field_names, 0..) |name, i| prof.names[i] = .{ .k = @field(d.k, name), .name = name };
        }
        const prof_on = prof.on;
        prof.on = false; // not during the load
        r.model = try load(d, &ckpt);
        prof.on = prof_on;
        r.acts = try Acts.init(d);
        r.store(); // slot 0: the states the load allocated
        try d.fill(r.acts.zeros, @as(u16, 0), CHUNK * 48 * 128 * 2);
        try d.fill(r.acts.ones, @as(f32, 1.0), CHUNK * 48 * 4);
        r.host_logits = try alloc.alloc(f32, VOCAB);
        return r;
    }

    /// Makes `slot` the one every call works on (states allocated on first use); other slots keep their state.
    pub fn select(r: *Runner, slot: u32) !void {
        if (slot >= MAX_SLOTS) return error.NoSuchSlot;
        if (slot == r.slot) return;
        try r.ensure(slot);
        r.store();
        r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
        for (&r.model.layers) |*l| {
            if (l.attn) {
                l.kc = l.slot_kc[slot];
                l.vc = l.slot_vc[slot];
            } else {
                l.rec = l.slot_rec[slot];
                l.rec_save = l.slot_save[slot];
                l.conv = l.slot_conv[slot];
                l.conv_cur = l.slot_cur[slot];
            }
        }
        r.pos = r.seqs[slot].pos;
        r.last = r.seqs[slot].last;
        r.slot = slot;
        r.block_rows = 0;
        r.tree_on = false;
    }

    pub const Win = multi.Win;
    pub const verifyMulti = multi.verifyMulti;
    pub const keepMulti = multi.keepMulti;
    pub const keepPaths = multi.keepPaths;
    pub const stepMulti = multi.stepMulti;

    /// The selected slot's state pointers back into its slot arrays (a verify's commit may have swapped them).
    pub fn store(r: *Runner) void {
        for (&r.model.layers) |*l| {
            if (l.attn) {
                l.slot_kc[r.slot] = l.kc;
                l.slot_vc[r.slot] = l.vc;
            } else {
                l.slot_rec[r.slot] = l.rec;
                l.slot_save[r.slot] = l.rec_save;
                l.slot_conv[r.slot] = l.conv;
                l.slot_cur[r.slot] = l.conv_cur;
            }
        }
        r.seqs[r.slot] = .{ .pos = r.pos, .last = r.last };
    }

    pub const Snap = slot_states.Snap;
    pub const slotBytes = slot_states.slotBytes;
    pub const snapBytes = slot_states.snapBytes;
    pub const saveSlot = slot_states.saveSlot;
    pub const restoreSlot = slot_states.restoreSlot;
    pub const dropSnap = slot_states.dropSnap;
    pub const unallocatedSlots = slot_states.unallocatedSlots;
    pub const ensure = slot_states.ensure;

    /// A fresh sequence: GDN and convolution states to zero (the KV cache is overwritten position by position).
    pub fn reset(r: *Runner) !void {
        for (&r.model.layers) |*l| if (!l.attn) {
            try r.d.fill(l.rec.?, @as(u32, 0), 48 * 128 * 128 * 4);
            for (l.conv) |cv| try r.d.fill(cv.?, @as(u16, 0), 3 * 10240 * 2);
            l.conv_cur = 0;
        };
        r.pos = 0;
    }

    /// Feeds prompt tokens in chunks; `last`: the final chunk, whose last row's logits are kept.
    pub fn prompt(r: *Runner, ids: []const u32, last: bool) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + CHUNK);
            const want = last and end == ids.len;
            if (r.pos + (end - at) > r.context) return error.ContextFull;
            if (try forward(&r.d, &r.model, &r.acts, @ptrCast(ids[at..end]), r.pos, .{ .want = if (want) .last else .none, .taps = r.taps_on })) |t| r.last = @intCast(t);
            if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, @intCast(end - at));
            r.pos += @intCast(end - at);
            at = end;
        }
    }

    /// One decoded token in; the next one's logits kept.
    pub fn step(r: *Runner, token: u32) !void {
        if (r.pos + 1 > r.context) return error.ContextFull;
        const id: i32 = @intCast(token);
        r.last = @intCast((try forward(&r.d, &r.model, &r.acts, &.{id}, r.pos, .{ .taps = r.taps_on })).?);
        r.pos += 1;
        if (r.taps_on) if (r.on_taps) |cb| try cb.f(cb.ctx, r.acts.taps, 1);
    }

    /// Verifies a block: `ids` (≤ 16 rows) run from `pos`, each row's greedy token returned; state moves till `commit`.
    pub fn verify(r: *Runner, ids: []const u32) ![]const u32 {
        if (ids.len == 0 or ids.len > 16) return error.BadBlock;
        if (r.pos + ids.len > r.context) return error.ContextFull;
        _ = try forward(&r.d, &r.model, &r.acts, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on });
        var toks: [16]i32 = undefined;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, &toks, r.acts.tok, ids.len * 4, null, 0, null));
        try r.d.sync();
        for (r.block[0..ids.len], toks[0..ids.len]) |*o, t| o.* = @intCast(t);
        r.block_rows = @intCast(ids.len);
        return r.block[0..ids.len];
    }

    /// Keeps the first `keep` rows of the last verified block: GDN and convolution states replay those rows from entry.
    pub fn commit(r: *Runner, keep: u32) !void {
        if (keep == 0 or keep > r.block_rows or r.tree_on) return error.BadCommit;
        const nul: ?*anyopaque = null;
        for (&r.model.layers) |*l| if (!l.attn) {
            if (keep == r.block_rows) { // every row kept: the block's final state is the state
                std.mem.swap(?*anyopaque, &l.rec, &l.rec_save);
                continue;
            }
            const st = l.stash.?;
            try r.d.launch(r.d.k.gdn_step, .{ .x = 16, .y = 48 }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.rec.?, l.rec.?, r.acts.y, @as(i32, @intCast(keep)), nul, nul, nul, nul });
            // the block's forward flipped conv_cur: conv[cur] is the state after all rows, conv[1 - cur] the entry
            try r.d.launch(r.d.k.conv_commit, .{ .x = 40 }, .{ l.conv[1 - l.conv_cur].?, off(st, ST_P), l.conv[l.conv_cur].?, @as(i32, @intCast(keep)), nul });
        };
        r.pos += keep;
        r.last = r.block[keep - 1];
        r.block_rows = 0; // no wait: the next commands queue behind these on the in-order list
    }

    /// Verifies a tree: row 0 is the pending token, `parents[t]` the row below row t; returns each row's greedy token.
    pub fn verifyTree(r: *Runner, ids: []const u32, parents: []const i32) ![]const u32 {
        if (ids.len == 0 or ids.len > 16 or parents.len != ids.len) return error.BadBlock;
        var depth_max: u32 = 0;
        for (0..ids.len) |t| {
            const row = r.tree_host[t * 17 ..][0..17];
            if (t == 0) {
                row[0] = 0;
                row[1] = 0;
                continue;
            }
            const p: usize = @intCast(parents[t]);
            if (p >= t) return error.BadBlock;
            const pr = r.tree_host[p * 17 ..][0..17];
            row[0] = pr[0] + 1;
            @memcpy(row[1 .. 1 + @as(usize, @intCast(pr[0])) + 1], pr[1 .. 1 + @as(usize, @intCast(pr[0])) + 1]);
            row[1 + @as(usize, @intCast(row[0]))] = @intCast(t);
            depth_max = @max(depth_max, @as(u32, @intCast(row[0])));
        }
        for (1..ids.len) |t| { // LOAD: the parent is not the row before ; SAVE: a child is not the row after
            const p: usize = @intCast(parents[t]);
            if (p + 1 != t) {
                r.tree_host[t * 17] |= 512;
                r.tree_host[p * 17] |= 256;
            }
        }
        if (r.pos + depth_max + 1 > r.context or r.pos + ids.len > r.context) return error.ContextFull;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, r.acts.tree, &r.tree_host, ids.len * 17 * 4, null, 0, null));
        _ = try forward(&r.d, &r.model, &r.acts, @ptrCast(ids), r.pos, .{ .want = .rows, .stash = true, .taps = r.taps_on, .tree = true });
        var toks: [16]i32 = undefined;
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, &toks, r.acts.tok, ids.len * 4, null, 0, null));
        try r.d.sync();
        for (r.block[0..ids.len], toks[0..ids.len]) |*o, t| o.* = @intCast(t);
        r.block_rows = @intCast(ids.len);
        r.tree_on = true;
        return r.block[0..ids.len];
    }

    /// Keeps the root path `path` of the last verified tree: states replayed along it, K/V moved, taps gathered.
    pub fn commitPath(r: *Runner, path: []const u32) !void {
        if (!r.tree_on or path.len == 0 or path.len > r.block_rows or path[0] != 0) return error.BadCommit;
        var straight = true;
        for (path, 0..) |row, dd| {
            r.path_host[dd] = @intCast(row);
            straight = straight and row == dd;
        }
        const keep: i32 = @intCast(path.len);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, r.acts.path, &r.path_host, path.len * 4, null, 0, null));
        const nul: ?*anyopaque = null;
        for (&r.model.layers) |*l| {
            if (l.attn) {
                if (!straight) try r.d.launch(r.d.k.kv_move, .{ .x = 4 }, .{ l.kc.?, l.vc.?, @as(i32, @intCast(r.pos)), r.acts.path, keep, kvCap() });
                continue;
            }
            const st = l.stash.?;
            try r.d.launch(r.d.k.gdn_step, .{ .x = 16, .y = 48 }, .{ off(st, ST_Q), off(st, ST_K), off(st, ST_V), off(st, ST_G), off(st, ST_B), l.rec.?, l.rec.?, r.acts.y, keep, r.acts.path, nul, nul, nul });
            try r.d.launch(r.d.k.conv_commit, .{ .x = 40 }, .{ l.conv[1 - l.conv_cur].?, off(st, ST_P), l.conv[l.conv_cur].?, keep, r.acts.path });
        }
        if (r.taps_on and !straight) for (path, 0..) |row, dd| if (row != dd)
            try copy(&r.d, off(r.acts.taps, dd * 5 * D * 2), off(r.acts.taps, row * 5 * D * 2), 5 * D * 2);
        r.pos += @intCast(path.len);
        r.last = r.block[path[path.len - 1]];
        r.block_rows = 0;
        r.tree_on = false;
        try r.d.sync(); // path_host is read by the queued copy
    }

    /// Structured output: each row's greedy token among those its grammar `bits` allow, computed on the device.
    pub fn argmaxMasked(r: *Runner, rows: []const u32, bits: []const u32, words: usize, out: []u32) !void {
        const n = rows.len;
        if (n == 0 or n > 16 or words > MASK_WORDS or bits.len != n * words or out.len < n) return error.BadMasks;
        const d = &r.d;
        var rh: [16]i32 = undefined;
        for (rows, 0..) |x, j| rh[j] = @intCast(x);
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mrows, &rh, n * 4, null, 0, null));
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, r.acts.mbits, bits.ptr, bits.len * 4, null, 0, null));
        try d.launch(d.k.argmax_part_masked, .{ .x = 256, .y = @intCast(n) }, .{ r.acts.logits, @as(i32, VOCAB), r.acts.mbits, @as(i32, @intCast(words)), r.acts.mrows, r.acts.pv, r.acts.pi });
        try d.launch(d.k.argmax, .{ .x = @intCast(n) }, .{ r.acts.pv, r.acts.pi, r.acts.mtok });
        var toks: [16]i32 = undefined;
        try ze.check(d.g.api.zeCommandListAppendMemoryCopy(d.list, &toks, r.acts.mtok, n * 4, null, 0, null));
        try d.sync();
        for (out[0..n], toks[0..n]) |*o, t| o.* = @intCast(t);
    }

    /// The logits of row `row` of the last verified block (or of the last forward's row, row 0), fp32 on the host.
    pub fn rowLogits(r: *Runner, row: u32) ![]const f32 {
        const raw = try alloc.alloc(f16, VOCAB);
        defer alloc.free(raw);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, off(r.acts.logits, row * VOCAB * 2), VOCAB * 2, null, 0, null));
        try r.d.sync();
        for (r.host_logits, raw) |*o, v| o.* = v;
        return r.host_logits;
    }

    /// The kept row's logits, fp32 on the host (for sampling).
    pub fn logits(r: *Runner) ![]const f32 {
        const raw = try alloc.alloc(f16, VOCAB);
        defer alloc.free(raw);
        try ze.check(r.d.g.api.zeCommandListAppendMemoryCopy(r.d.list, raw.ptr, r.acts.logits, VOCAB * 2, null, 0, null));
        try r.d.sync();
        for (r.host_logits, raw) |*o, v| o.* = v;
        return r.host_logits;
    }

    /// The device's name, as the server reports the chip.
    pub fn deviceName(r: *Runner) []const u8 {
        return std.mem.sliceTo(&r.d.g.props.name, 0);
    }
};

pub const drafter = @import("drafter.zig");
