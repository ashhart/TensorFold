//! The DFlash2 drafter's load on a rank: its EXL3 pack's weights (split like the target) and its device state.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const load_ = @import("load.zig");
const dc = @import("dflash2_config.zig");
const ml = @import("pack.zig");
const dflash2 = @import("dflash2.zig");
const Drafter = dflash2.Drafter;
const DLayer = dflash2.DLayer;
const Lin = dflash2.Lin;
const Stream = cuda.Stream;
const MAX_STREAMS = dflash2.MAX_STREAMS;

fn ci(x: usize) c_int {
    return @intCast(x);
}

/// The drafter pack's weights on this rank: fc, norms, selector and codebooks, the layers, the mask row.
pub fn loadWeights(dr: *Drafter, io: std.Io, dir: []const u8) !void {
    const gpa = dr.gpa;
    const ar = &dr.arena;
    const full = dr.full;
    const rank = dr.m.w.rank;
    const world = dr.m.w.world;
    var ck = try core.Checkpoint.openModel(gpa, io, dir);
    defer ck.close();
    dr.fc = try ml.linear(gpa, ar, try ml.cut(gpa, try ml.readMat(gpa, &ck, "fc"), dc.fcCut(full, rank, world)));
    if (dr.fc.k != full.ntaps * full.hidden) return error.FcWidth;
    dr.fc_norm = try f32Up(gpa, ar, try ck.get("hidden_norm.weight"));
    dr.norm = try f32Up(gpa, ar, try ck.get("norm.weight"));
    dr.sel = try lin(gpa, ar, &ck, "candidate_selector.hidden_projection", .{ .prefix = "candidate_selector.hidden_projection" });
    if (dr.sel.n() != full.sel_rank) return error.SelectorShape;
    const pc = try codebook(&ck, "candidate_selector.predecessor_codebook");
    const sc = try codebook(&ck, "candidate_selector.successor_codebook");
    for ([_]core.safetensors.Tensor{ pc, sc }) |t| if (t.rank != 2 or t.shape[0] != full.vocab or t.shape[1] != full.sel_rank or t.dtype != pc.dtype or (t.dtype != .bf16 and t.dtype != .f16)) return error.CodebookShape;
    dr.cb_bf16 = pc.dtype == .bf16;
    dr.pcb = try ar.upload(pc.bytes);
    dr.scb = try ar.upload(sc.bytes);
    dr.layers = try gpa.alloc(DLayer, full.layers);
    errdefer gpa.free(dr.layers);
    var nb: [160]u8 = undefined;
    var pb: [160]u8 = undefined;
    for (dr.layers, 0..) |*L, i| {
        L.in_norm = try f32Up(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, "layers.{d}.input_layernorm.weight", .{i})));
        L.post_norm = try f32Up(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, "layers.{d}.post_attention_layernorm.weight", .{i})));
        L.q_norm = try f32Up(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, "layers.{d}.self_attn.q_norm.weight", .{i})));
        L.k_norm = try f32Up(gpa, ar, try ck.get(try std.fmt.bufPrint(&nb, "layers.{d}.self_attn.k_norm.weight", .{i})));
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }, 0..) |f, j| {
            const prefix = try std.fmt.bufPrint(&pb, "layers.{d}.{s}", .{ i, dc.mat_names[j] });
            const cut = dc.layerCut(full, @fromBackingInt(@intCast(j)), prefix, rank, world);
            @field(L, f) = try ml.linear(gpa, ar, try ml.cut(gpa, try ml.readMat(gpa, &ck, prefix), cut));
        }
        inline for (.{ "aconv", "mconv" }, .{ "attention_conv", "mlp_conv" }, 7..) |f, cname, j| {
            const prefix = try std.fmt.bufPrint(&pb, "layers.{d}.{s}", .{ i, dc.mat_names[j] });
            @field(L, f).proj = try lin(gpa, ar, &ck, prefix, .{ .prefix = prefix });
            if (@field(L, f).proj.n() != 2 * full.conv_k * (full.hidden / full.conv_g)) return error.ConvShape;
            const bt = try ck.get(try std.fmt.bufPrint(&nb, "layers.{d}.{s}.base_kernel", .{ i, cname }));
            if (bt.numel() != 2 * full.conv_k * full.hidden) return error.ConvShape;
            @field(L, f).base = try f32Up(gpa, ar, bt);
        }
    }
    // the mask rows: a shipped mask_embedding, else the target's (raw) embedding row of mask_token_id
    if (ck.get("mask_embedding")) |t| {
        dr.mask_emb = try f32Up(gpa, ar, t);
    } else |_| {
        const D = full.hidden;
        const row = try gpa.alloc(u16, D);
        defer gpa.free(row);
        @memcpy(std.mem.sliceAsBytes(row), dr.m.w.embed_host[full.mask_id * D * 2 ..][0 .. D * 2]);
        const f = try gpa.alloc(f32, D);
        defer gpa.free(f);
        for (row, f) |x, *o| o.* = @bitCast(@as(u32, x) << 16);
        dr.mask_emb = try ar.upload(std.mem.sliceAsBytes(f));
    }
}

/// The context caches, RoPE tables and draft-pass scratch on this rank.
pub fn allocState(dr: *Drafter, s: Stream) !void {
    const gpa = dr.gpa;
    const ar = &dr.arena;
    const m = dr.m;
    const c = dr.c;
    const D = c.hidden;
    const L = dr.layers.len;
    dr.kc = try gpa.alloc(u64, L);
    errdefer gpa.free(dr.kc);
    dr.vc = try gpa.alloc(u64, L);
    errdefer gpa.free(dr.vc);
    for (0..L) |i| {
        dr.kc[i] = try ar.zeros(m.slots * dr.ring * c.kv_heads * 128 * 2);
        dr.vc[i] = try ar.zeros(m.slots * dr.ring * c.kv_heads * 128 * 2);
    }
    const tlen = m.context + dr.block;
    dr.cos = try ar.alloc(tlen * 64 * 4);
    dr.sin = try ar.alloc(tlen * 64 * 4);
    {
        const f = try m.glue.function("muse_rope_tables");
        var a: cuda.Args = .{};
        a.add(dr.cos);
        a.add(dr.sin);
        a.add(ci(tlen));
        a.add(c.theta);
        a.add(ci(128));
        try cuda.launch.launch(f, .{ .grid = .{ .x = 256, .y = 1, .z = 1 }, .block = .{ .x = 256, .y = 1, .z = 1 } }, s, &a);
    }
    const R = m.rows;
    const SB = dr.streams * dr.block;
    const RR = @max(R, SB);
    const S1 = dr.streams * (dr.block - 1);
    const W = m.w.world;
    const HQ = c.heads * 128;
    const HK = c.kv_heads * 128;
    const vr = m.w.head.n;
    var max_z: usize = @max(Drafter.zBytes(dr.fc, R), Drafter.zBytes(m.w.head, S1));
    for (dr.layers) |ly| {
        inline for (.{ "q", "k", "v", "o", "gate", "up", "down" }) |f| max_z = @max(max_z, Drafter.zBytes(@field(ly, f), RR));
        for ([_]Lin{ ly.aconv.proj, ly.mconv.proj }) |p| if (p == .q) {
            max_z = @max(max_z, Drafter.zBytes(p.q, SB));
        };
    }
    if (dr.sel == .q) max_z = @max(max_z, Drafter.zBytes(dr.sel.q, S1));
    var b = &dr.b;
    b.tapbuf = try ar.zeros(R * dr.taps.len * D * 2);
    b.ctx16 = try ar.zeros(R * dr.fc.n * 2);
    b.ctxg = if (W > 1) try ar.zeros(W * R * dr.fc.n * 2) else 0;
    b.ctx = try ar.zeros(R * D * 2);
    b.kb = try ar.zeros(RR * HK * 2);
    b.vb = try ar.zeros(RR * HK * 2);
    b.xh = try ar.zeros(RR * @max(dr.taps.len * D, @max(c.inter, HQ)) * 2);
    b.z = try ar.zeros(@max(max_z, 4));
    b.x32 = try ar.zeros(SB * D * 4);
    b.h = try ar.zeros(SB * D * 2);
    b.c16 = try ar.zeros(SB * D * 2);
    b.dyn = try ar.zeros(SB * dr.layers[0].aconv.proj.n() * 2);
    b.q16 = try ar.zeros(SB * HQ * 2);
    b.q32 = try ar.zeros(SB * HQ * 4);
    b.a = try ar.zeros(SB * HQ * 2);
    b.part = try ar.zeros(SB * D * 4);
    b.recv = if (W > 1) try ar.zeros(W * SB * D * 4) else 0;
    b.gb = try ar.zeros(SB * c.inter * 2);
    b.ub = try ar.zeros(SB * c.inter * 2);
    b.mm = try ar.zeros(SB * c.inter * 2);
    b.tk = try ar.zeros(S1 * D * 2);
    b.head = try ar.zeros(S1 * vr * 4);
    b.top = try ar.zeros(S1 * 32 * 4);
    b.topg = if (W > 1) try ar.zeros(W * S1 * 32 * 4) else 0;
    b.sg = try ar.zeros(S1 * c.sel_rank * 2);
    b.am = try ar.zeros(S1 * 4);
    b.in_tok = try ar.zeros(dr.streams * 8);
    b.in_pos0 = try ar.zeros(dr.streams * 4);
    b.in_slot = try ar.zeros(dr.streams * 4);
    b.rpos = try ar.zeros(SB * 4);
    b.rslot = try ar.zeros(SB * 4);
    b.anc = try ar.zeros(dr.streams * D * 2);
    {
        var io_: [MAX_STREAMS]i64 = undefined;
        for (&io_, 0..) |*x, j| x.* = @intCast(j);
        b.anc_idx = try ar.upload(std.mem.sliceAsBytes(io_[0..dr.streams]));
    }
    dr.h_anc = try dr.gpa.alloc(u16, dr.streams * D);
}

pub fn near(a: f64, b: f64) bool {
    return @abs(a - b) <= 1e-6 * @max(@abs(a), @abs(b));
}

pub fn readFile(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
}

fn f32Up(gpa: std.mem.Allocator, ar: *load_.Arena, t: core.safetensors.Tensor) !u64 {
    const f = try ml.asF32(gpa, t);
    defer gpa.free(f);
    return ar.upload(std.mem.sliceAsBytes(f));
}

/// A replicated matrix: EXL3 when the pack quantised it, else its 16-bit rows (`{prefix}.weight`, [N, K]).
fn lin(gpa: std.mem.Allocator, ar: *load_.Arena, ck: *core.Checkpoint, prefix: []const u8, cut: @import("config.zig").Cut) !Lin {
    if (ml.readMat(gpa, ck, prefix)) |mat| {
        return .{ .q = try ml.linear(gpa, ar, try ml.cut(gpa, mat, cut)) };
    } else |_| {}
    var nb: [192]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{prefix}));
    if (t.rank != 2 or (t.dtype != .bf16 and t.dtype != .f16)) return error.BadDenseMatrix;
    return .{ .dense = .{ .w = try ar.upload(t.bytes), .n = t.shape[0], .k = t.shape[1], .bf16 = t.dtype == .bf16 } };
}

/// The selector codebooks: bare keys, or with `.weight` (older trainers), 16-bit.
fn codebook(ck: *core.Checkpoint, key: []const u8) !core.safetensors.Tensor {
    if (ck.get(key)) |t| return t else |_| {}
    var nb: [128]u8 = undefined;
    return ck.get(try std.fmt.bufPrint(&nb, "{s}.weight", .{key}));
}
