//! Nemotron-H's three layer kinds over a forward's rows: Mamba (per-stream states), MoE, attention (per-stream KV).
const std = @import("std");
const mtl = @import("metal");
const wts = @import("weights.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const kern = @import("kernels.zig");
const tree = @import("tree.zig");

const Buffer = mtl.Buffer;
const Enc = fwd.Enc;
const Forward = fwd.Forward;
const ints8 = fwd.ints8;

/// Mamba tables: each virtual row (a segment's replayed rows, then its rows) by segment, place, parent and ancestors.
pub const Tables = struct {
    va: [2 * st.max_rows][4]i32 = undefined, // segment, place, state slot to store (-1), parent place (-1: slot state)
    vb: [2 * st.max_rows][4]i32 = undefined, // ancestors 3, 2, 1 back (negative: the conv state), replayed input row
    vc: [2 * st.max_rows]i32 = @splat(-1), // tree-state slot for a branch parent (-1: none)
    segi: [8 * st.max_rows]i32 = @splat(0),
    virtual: usize = 0,
    segs: usize = 0,
    saves: usize = 0,

    pub fn of(segs: []const fwd.Seg) Tables {
        var t = Tables{ .segs = segs.len };
        var real: usize = 0;
        for (segs, 0..) |g, s| {
            const c = g.cache;
            const k: i32 = @intCast(c.replay);
            const n = c.replay + g.rows;
            const save = c.rid >= 0 and g.rows <= st.replay_rows;
            const v0 = t.virtual;
            t.segi[8 * s ..][0..8].* = .{ @intCast(real), k, @intCast(c.slot), if (save) c.rid else 0, c.parity, @intFromBool(save), @intCast(v0), 0 };
            for (0..n) |l| {
                const loc: i32 = @intCast(l);
                const par = parentOf(g, k, loc);
                const anc2 = parentOf(g, k, par);
                const store: i32 = switch (g.store) {
                    .full => if (l == n - 1) g.slot else -1,
                    .lag => if (loc == k - 1) g.slot else -1,
                    .all => if (loc >= k) g.slots[l - c.replay] else -1,
                };
                t.va[v0 + l] = .{ @intCast(s), loc, store, par };
                t.vb[v0 + l] = .{ parentOf(g, k, anc2), anc2, par, if (loc < k) c.replay_map[l] else 0 };
                // a parent whose child is not its next row keeps its state for the scan to reload
                if (par >= 0 and par != loc - 1 and t.vc[v0 + @as(usize, @intCast(par))] < 0) {
                    std.debug.assert(t.saves < st.tree_saves);
                    t.vc[v0 + @as(usize, @intCast(par))] = @intCast(t.saves);
                    t.saves += 1;
                }
            }
            t.virtual += n;
            real += g.rows;
        }
        return t;
    }

    /// A place's parent: the previous place in a chain or replay; a tree row's parent (its root after the replayed rows).
    fn parentOf(g: fwd.Seg, k: i32, loc: i32) i32 {
        if (loc < k) return loc - 1;
        const p = g.parents orelse return loc - 1;
        const q = p[@intCast(loc - k)];
        return if (q >= 0) k + q else k - 1;
    }

    fn bytes(table: anytype, n: usize) []const u8 {
        return std.mem.sliceAsBytes(table[0..@max(n, 8)]);
    }
};

/// A GPU-side round's Mamba tables and dims (one segment), its routed rows, at byte offsets in one buffer.
pub const RoundArgs = struct { buffer: Buffer, va: usize, vb: usize, vc: usize, segi: usize, dims: usize, rows: usize, virtual: usize };

pub fn mamba(f: Forward, e: *Enc, m: wts.Mamba, index: usize, t: *const Tables, rows: usize) void {
    const c = f.c;
    const s = f.s;
    f.coop(e, "in", m.in_proj, s.x, 0, s.xs, s.proj, rows);
    const dims = [2]i32{ @intCast(t.virtual), st.replay_rows };
    const ra = f.round;

    e.pipe(f.k.get("tf_tree_conv"));
    e.buf(s.proj, 0, 0);
    e.buf(f.pool.conv[index], 0, 1);
    e.buf(m.conv_w, 0, 2);
    e.buf(m.conv_b, 0, 3);
    if (ra) |r| {
        for ([_]usize{ r.va, r.vb, r.segi, r.dims }, 4..) |off, i| e.buf(r.buffer, off, i);
    } else {
        e.e.setBytes(Tables.bytes(&t.va, t.virtual), 4);
        e.e.setBytes(Tables.bytes(&t.vb, t.virtual), 5);
        e.e.setBytes(Tables.bytes(&t.segi, 8 * t.segs), 6);
        e.bytes(dims, 7);
    }
    e.buf(f.pool.raw[index], 0, 8);
    e.buf(s.xbc, 0, 9);
    e.buf(f.pool.conv[index], 0, 10);
    e.run(.{ c.convDim(), if (ra) |r| r.virtual else t.virtual, 1 }, .{ 256, 1, 1 });

    e.pipe(f.k.get("tf_tree_scan"));
    e.buf(s.proj, 0, 0);
    e.buf(s.xbc, 0, 1);
    e.buf(f.pool.ssm[index], 0, 2);
    e.buf(m.a_log, 0, 3);
    e.buf(m.d_skip, 0, 4);
    e.buf(m.dt_bias, 0, 5);
    e.bytes([2]f32{ 0.0, std.math.inf(f32) }, 6);
    if (ra) |r| {
        for ([_]usize{ r.dims, r.va, r.vb, r.vc, r.segi }, 7..) |off, i| e.buf(r.buffer, off, i);
    } else {
        e.bytes(dims, 7);
        e.e.setBytes(Tables.bytes(&t.va, t.virtual), 8);
        e.e.setBytes(Tables.bytes(&t.vb, t.virtual), 9);
        e.e.setBytes(Tables.bytes(&t.vc, t.virtual), 10);
        e.e.setBytes(Tables.bytes(&t.segi, 8 * t.segs), 11);
    }
    e.buf(f.pool.dtraw[index], 0, 12);
    e.buf(s.y, 0, 13);
    e.buf(f.pool.ssm[index], 0, 14);
    e.buf(f.pool.tree, 0, 15);
    e.run(.{ 32, c.mamba_head_dim, c.mamba_heads }, .{ 32, 8, 1 });

    const group = c.inner() / c.groups;
    e.pipe(groupNorm(f, rows));
    e.buf(s.y, 0, 0);
    e.buf(m.norm.buffer, m.norm.offset, 1);
    e.bytes(c.eps, 2);
    e.bytes(Forward.mdims(rows), 3);
    e.buf(s.yn, 0, 4);
    e.buf(s.ys, 0, 5);
    e.run(.{ group / 4 * c.groups, Forward.mp(rows), 1 }, .{ group / 4, 1, 1 });

    f.coop(e, "out", m.out_proj, s.yn, 0, s.ys, s.delta, rows);
}

fn groupNorm(f: Forward, rows: usize) mtl.Pipeline {
    return switch (Forward.mp(rows)) {
        16 => f.k.get("group_norm_xs_16"),
        32 => f.k.get("group_norm_xs_32"),
        48 => f.k.get("group_norm_xs_48"),
        64 => f.k.get("group_norm_xs_64"),
        80 => f.k.get("group_norm_xs_80"),
        96 => f.k.get("group_norm_xs_96"),
        112 => f.k.get("group_norm_xs_112"),
        else => f.k.get("group_norm_xs_128"),
    };
}

pub fn moe(f: Forward, e: *Enc, m: wts.Moe, next: anytype, rows: usize, eps: f32) void {
    const c = f.c;
    const s = f.s;
    const pairs = rows * c.top_k;

    e.pipe(f.k.get("router"));
    e.buf(s.x, 0, 0);
    e.buf(m.gate.buffer, m.gate.offset, 1);
    e.bytes(@as(i32, @intCast(rows)), 2);
    e.buf(s.logits_e, 0, 3);
    e.run(.{ 256, c.experts, (rows + 15) / 16 }, .{ 256, 1, 1 });

    // the shared expert's up + relu2 (+ its down input sums) beside the router: both read only x and xs
    e.alongside();
    upRelu2(f, e, m.shared_up, rows);

    const scaling: f32 = c.routed_scaling;
    var groups: usize = pairs;
    if (rows == 1) {
        // a group a pair, the route kernel's ids as UIDS (tables padded to 8 elements)
        e.pipe(f.k.get("route"));
        e.buf(s.logits_e, 0, 0);
        e.buf(m.gate_bias.buffer, m.gate_bias.offset, 1);
        e.bytes(scaling, 2);
        e.buf(s.idx, 0, 3);
        e.buf(s.wt, 0, 4);
        e.run(.{ 32 * rows, 1, 1 }, .{ 32, 1, 1 });
    } else {
        e.pipe(f.k.get("route_group"));
        e.buf(s.logits_e, 0, 0);
        e.buf(m.gate_bias.buffer, m.gate_bias.offset, 1);
        e.bytes(scaling, 2);
        if (f.round) |r| e.buf(r.buffer, r.rows, 3) else e.bytes(@as(i32, @intCast(rows)), 3);
        e.buf(s.idx, 0, 4);
        e.buf(s.wt, 0, 5);
        e.buf(s.uids, 0, 6);
        e.buf(s.start, 0, 7);
        e.buf(s.count, 0, 8);
        e.buf(s.members, 0, 9);
        e.buf(s.ucount, 0, 10);
        e.run(.{ 512, 1, 1 }, .{ 512, 1, 1 });
        groups = @min(pairs, c.experts);
    }
    if (s.route_log) |log| {
        f.copyIds(e, s.idx, 0, log, s.route_at * pairs * 4, pairs);
        s.route_at += 1;
    }

    // the shared expert's down projection first, the routed experts' up projection beside it
    f.coop(e, "down", m.shared_down, s.sh_act, 0, s.sh_xs, s.sh, rows);
    inline for (.{ "expert_up", "expert_down" }, 0..) |key, j| {
        if (j == 0) e.alongside();
        const fc = if (j == 0) m.fc1 else m.fc2;
        const many = rows > 1 and f.members > 0;
        const geo: [2]usize = if (f.geometry > 0 and !many) kern.geometries[f.geometry - 1] else .{ 4, 2 };
        e.pipe(if (many) expertRows(f, j, f.members) else if (f.geometry > 0) expertGeo(f, j, f.geometry - 1) else f.k.get(key));
        e.buf(if (j == 0) s.x else s.act, 0, 0);
        if (rows == 1) {
            e.buf(s.idx, 0, 1);
            e.bytes(ints8(&.{ 0, 1, 2, 3, 4, 5 }), 2);
            e.bytes(ints8(&.{ 1, 1, 1, 1, 1, 1 }), 3);
            e.bytes(ints8(&.{ 0, 1, 2, 3, 4, 5 }), 4);
            e.bytes(@as(i32, @intCast(pairs)), 5);
        } else {
            e.buf(s.uids, 0, 1);
            e.buf(s.start, 0, 2);
            e.buf(s.count, 0, 3);
            e.buf(s.members, 0, 4);
            e.buf(s.ucount, 0, 5);
        }
        e.buf(fc[0].buffer, fc[0].offset, 6);
        e.buf(fc[1].buffer, fc[1].offset, 7);
        e.buf(fc[2].buffer, fc[2].offset, 8);
        e.buf(if (j == 0) s.act else s.ey, 0, 9);
        const n = if (j == 0) c.expert_width else c.hidden;
        e.run(.{ 32 * geo[1], n / (geo[0] * geo[1]), groups }, .{ 32 * geo[1], 1, 1 });
    }

    // combine (routed in route order) + shared + residual, then the next input norm and its sums
    e.pipe(f.k.get("add_norm_moe_xs"));
    e.buf(s.h, 0, 0);
    e.buf(s.ey, 0, 1);
    e.buf(s.wt, 0, 2);
    e.buf(s.sh, 0, 3);
    e.buf(next.buffer, next.offset, 4);
    e.bytes(eps, 5);
    e.bytes(Forward.mdims(rows), 6);
    e.buf(s.h, 0, 7);
    e.buf(s.x, 0, 8);
    e.buf(s.xs, 0, 9);
    e.run(.{ 896 * Forward.mp(rows), 1, 1 }, .{ 896, 1, 1 });
}

/// The routed-expert kernel (j 0: up, 1: down) taking `mb` member rows a pass.
fn expertRows(f: Forward, j: usize, mb: usize) mtl.Pipeline {
    if (mb >= 4) return if (j == 0) f.k.get("tf_xup_rows4") else f.k.get("tf_xdown_rows4");
    return if (j == 0) f.k.get("tf_xup_rows2") else f.k.get("tf_xdown_rows2");
}

/// A routed-expert kernel (j 0: up, 1: down) at geometry `g` (kernels.geometries).
fn expertGeo(f: Forward, j: usize, g: usize) mtl.Pipeline {
    inline for (kern.geometries, 0..) |geo, i| {
        if (i == g) return if (j == 0) f.k.get(std.fmt.comptimePrint("tf_xup_{d}_{d}", .{ geo[0], geo[1] })) else f.k.get(std.fmt.comptimePrint("tf_xdown_{d}_{d}", .{ geo[0], geo[1] }));
    }
    unreachable;
}

/// The shared expert's up projection + relu2 + the down projection's input sums (lane_fused.up_relu2).
fn upRelu2(f: Forward, e: *Enc, lin: wts.Linear, rows: usize) void {
    const s = f.s;
    if (f.rowCall(e, lin, s.x, 0, s.sh_act, 0, rows, true)) return;
    const sk = fwd.splitK(lin.n, lin.k);
    const mpr = Forward.mp(rows);
    const block = if (mpr <= 32) mpr else 32;
    const tmr = block / 16;
    const edge = mpr % block != 0;
    e.pipe(if (tmr == 1) f.k.get("up_relu2_1_0") else if (!edge) f.k.get("up_relu2_2_0") else f.k.get("up_relu2_2_1"));
    e.buf(s.x, 0, 0);
    e.buf(s.xs, 0, 1);
    e.buf(lin.w, 0, 2);
    e.buf(lin.sbt, 0, 3);
    e.bytes(Forward.mdims(rows), 4);
    e.buf(s.sh_act, 0, 5);
    e.buf(s.sh_xs, 0, 6);
    e.run(.{ lin.n / 64 * 64 * sk, (mpr + block - 1) / block, 1 }, .{ 64 * sk, 1, 1 });
}

/// A segment's cache: `len` rows before it, its rows written at `write` (default `len`), a tree's depth and tables row.
pub const Kvs = struct { kv: st.Kv, len: usize, rows: usize, tree: bool = false, deepest: usize = 0, write: ?usize = null, tables: usize = 0, gpu: ?GpuArgs = null, gpu_tree: ?tree.GpuTree = null, layer: usize = 0 };

/// A round's kv write (4 u32) and attention dims (5 i32) written on the GPU, and the chunks to dispatch at most.
pub const GpuArgs = struct { buffer: Buffer, kv: usize, attn: usize, chunks: usize };

/// q/k/v for all rows, each segment's new rows into its cache, its rows attending to its cache, then o_proj.
pub fn attention(f: Forward, e: *Enc, a: wts.Attention, segs: []const Kvs, rows: usize) void {
    f.coop(e, "qkv", a.qkv, f.s.x, 0, f.s.xs, f.s.qkv, rows);
    var r0: usize = 0;
    for (segs) |sg| {
        if (sg.gpu) |ga| kvWriteAt(f, e, sg.kv, ga.buffer, ga.kv, r0, sg.rows) else kvWrite(f, e, sg.kv, sg.write orelse sg.len, r0, sg.rows);
        r0 += sg.rows;
    }
    attend(f, e, a, segs, rows);
}

/// attention() after its q/k/v rows are in scratch.qkv and the new keys and values in the caches.
pub fn attend(f: Forward, e: *Enc, a: wts.Attention, segs: []const Kvs, rows: usize) void {
    const c = f.c;
    const s = f.s;
    const g = c.heads / c.kv_heads;
    const hd = c.head_dim;
    const qkv_n = c.qkvDim();
    // at one row in one segment, the stacked output's q columns are Qp and the output is o_proj's row
    const lone = segs.len == 1 and rows == 1 and !segs[0].tree;
    var r0: usize = 0;
    for (segs) |sg| {
        const t = sg.rows;
        var qp = s.qkv;
        var qp_off: usize = 0;
        if (!lone) {
            e.pipe(f.k.get("tf_attn_q"));
            e.buf(s.qkv, r0 * qkv_n * 2, 0);
            e.buf(s.qp, 0, 1);
            e.bytes([2]u32{ @intCast(qkv_n), @intCast(g) }, 2);
            e.run(.{ hd, g * t, c.kv_heads }, .{ hd, 1, 1 });
            qp = s.qp;
            qp_off = 0;
        }
        if (sg.tree) {
            if (sg.gpu_tree) |gt| {
                if (gt.dual) |d| {
                    // the window's chain path or its tree path: the GPU zeroes the other's threadgroup counts
                    var chain_gate = d.gate(sg.layer, 0);
                    e.gate = &chain_gate;
                    chainAttend(f, e, sg, qp, qp_off);
                    var tree_gate = d.gate(sg.layer, 1);
                    e.gate = &tree_gate;
                    tree.attentionGpu(f, e, sg.kv, t, partial(f, @min(t, 16)), gt);
                    e.gate = null;
                } else tree.attentionGpu(f, e, sg.kv, t, partial(f, @min(t, 16)), gt);
            } else tree.attention(f, e, sg.kv, sg.len, t, sg.deepest, partial(f, @min(t, 16)), sg.tables);
            e.pipe(f.k.get("tf_attn_out"));
            e.buf(s.att, 0, 0);
            e.buf(s.ax, r0 * c.heads * hd * 2, 1);
            e.run(.{ hd, c.heads, t }, .{ hd, 1, 1 });
            r0 += t;
            continue;
        }
        const keys = sg.len + t;
        const chunks = (keys + 511) / 512;
        const sga = g * t / 16;
        const tiles: usize = @min(sga, 16);
        const dims = [5]i32{ @intCast(keys), @intCast(chunks), @intCast(t), 1, @intCast(sga) };
        const grid_chunks = if (sg.gpu) |ga| ga.chunks else chunks;
        const strides = [4]i64{ @intCast(c.kv_heads * sg.kv.capacity * hd), @intCast(sg.kv.capacity * hd), @intCast(hd), 1 };
        e.pipe(partial(f, tiles));
        e.buf(qp, qp_off, 0);
        e.buf(sg.kv.k, 0, 1);
        e.bytes(strides, 2);
        e.buf(sg.kv.v, 0, 3);
        e.bytes(strides, 4);
        e.bytes(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))))), 5);
        if (sg.gpu) |ga| e.buf(ga.buffer, ga.attn, 6) else e.bytes(dims, 6);
        e.buf(s.po, 0, 7);
        e.buf(s.pm, 0, 8);
        e.buf(s.pl, 0, 9);
        e.run(.{ c.kv_heads * 32 * tiles, grid_chunks, (sga + tiles - 1) / tiles }, .{ 32 * tiles, 1, 1 });

        e.pipe(f.k.get("attn_merge"));
        e.buf(s.po, 0, 0);
        e.buf(s.pm, 0, 1);
        e.buf(s.pl, 0, 2);
        if (sg.gpu) |ga| e.buf(ga.buffer, ga.attn, 3) else e.bytes(dims, 3);
        e.buf(s.att, 0, 4);
        e.run(.{ c.kv_heads * 32, g * t, 1 }, .{ 32, 1, 1 });
        if (!lone) {
            e.pipe(f.k.get("tf_attn_out"));
            e.buf(s.att, 0, 0);
            e.buf(s.ax, r0 * c.heads * hd * 2, 1);
            e.run(.{ hd, c.heads, t }, .{ hd, 1, 1 });
        }
        r0 += t;
    }
    const o_in = if (lone) s.att else s.ax;
    f.xsum(e, "xsum_4096", c.heads * hd, o_in, 0, s.axs, rows);
    f.coop(e, "out", a.o_proj, o_in, 0, s.axs, s.delta, rows);
}

/// A GPU round's chain attention over its window (dims written on the GPU), into scratch.att.
fn chainAttend(f: Forward, e: *Enc, sg: Kvs, qp: Buffer, qp_off: usize) void {
    const c = f.c;
    const s = f.s;
    const g = c.heads / c.kv_heads;
    const hd = c.head_dim;
    const t = sg.rows;
    const ga = sg.gpu.?;
    const sga = g * t / 16;
    const tiles: usize = @min(sga, 16);
    const strides = [4]i64{ @intCast(c.kv_heads * sg.kv.capacity * hd), @intCast(sg.kv.capacity * hd), @intCast(hd), 1 };
    e.pipe(partial(f, tiles));
    e.buf(qp, qp_off, 0);
    e.buf(sg.kv.k, 0, 1);
    e.bytes(strides, 2);
    e.buf(sg.kv.v, 0, 3);
    e.bytes(strides, 4);
    e.bytes(@as(f32, @floatCast(1.0 / @sqrt(@as(f64, @floatFromInt(hd))))), 5);
    e.buf(ga.buffer, ga.attn, 6);
    e.buf(s.po, 0, 7);
    e.buf(s.pm, 0, 8);
    e.buf(s.pl, 0, 9);
    e.run(.{ c.kv_heads * 32 * tiles, ga.chunks, (sga + tiles - 1) / tiles }, .{ 32 * tiles, 1, 1 });
    e.pipe(f.k.get("attn_merge"));
    e.buf(s.po, 0, 0);
    e.buf(s.pm, 0, 1);
    e.buf(s.pl, 0, 2);
    e.buf(ga.buffer, ga.attn, 3);
    e.buf(s.att, 0, 4);
    e.run(.{ c.kv_heads * 32, g * t, 1 }, .{ 32, 1, 1 });
}

/// kvWrite with its (QKV, KOFF, capacity, start) read from a GPU buffer at `at`.
pub fn kvWriteAt(f: Forward, e: *Enc, kv: st.Kv, args: Buffer, at: usize, row0: usize, rows: usize) void {
    const c = f.c;
    e.pipe(f.k.get("tf_kv_write"));
    e.buf(f.s.qkv, row0 * c.qkvDim() * 2, 0);
    e.buf(kv.k, 0, 1);
    e.buf(kv.v, 0, 2);
    e.buf(args, at, 3);
    e.run(.{ c.head_dim, c.kv_heads, rows }, .{ c.head_dim, 1, 1 });
}

/// The stacked q/k/v rows from `row0` (their keys and values) into a cache after its `len` rows.
pub fn kvWrite(f: Forward, e: *Enc, kv: st.Kv, len: usize, row0: usize, rows: usize) void {
    const c = f.c;
    e.pipe(f.k.get("tf_kv_write"));
    e.buf(f.s.qkv, row0 * c.qkvDim() * 2, 0);
    e.buf(kv.k, 0, 1);
    e.buf(kv.v, 0, 2);
    e.bytes([4]u32{ @intCast(c.qkvDim()), @intCast(c.heads * c.head_dim), @intCast(kv.capacity), @intCast(len) }, 3);
    e.run(.{ c.head_dim, c.kv_heads, rows }, .{ c.head_dim, 1, 1 });
}

fn partial(f: Forward, tiles: usize) mtl.Pipeline {
    const keys = [_][]const u8{ "attn_partial_1", "attn_partial_2", "attn_partial_3", "attn_partial_4", "attn_partial_5", "attn_partial_6", "attn_partial_7", "attn_partial_8", "attn_partial_9", "attn_partial_10", "attn_partial_11", "attn_partial_12", "attn_partial_13", "attn_partial_14", "attn_partial_15", "attn_partial_16" };
    inline for (keys, 1..) |key, n| if (n == tiles) return f.k.get(key);
    unreachable;
}
