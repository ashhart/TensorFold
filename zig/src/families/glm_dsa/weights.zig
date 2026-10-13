//! One GLM-5.3 layer's EXLR8 weights for this Mac, packed: its heads' attention, its dense rows or expert halves.
const std = @import("std");
const ex = @import("../../core/exlr8.zig");
const cfg = @import("config.zig");
const st = ex.safetensors;
const dense = ex.dense;
const pack = ex.pack;
const Proj = ex.experts.Proj;

/// x's projections in one dispatch: q_a [0, 2048), kv_a [2048, 2624), then the indexer's key and head weights.
pub const proj_width = 2784;
pub const kv_col = 2048;
pub const wk_col = 2624;
pub const wp_col = 2752;

pub const Attn = struct {
    x_jobs: pack.Range,
    q_b: pack.Linear,
    wq_b: ?pack.Rows,
    /// kv_b's rows for this Mac's heads (FP8 128x128 blocks): W_uk read by the absorb kernel, W_uv as jobs.
    kvb_w: u64,
    kvb_s: u64,
    uv_jobs: pack.Range,
    o: pack.Linear,
    q_norm: u64,
    kv_norm: u64,
    k_norm: u64,
    k_bias: u64,
};

/// The most 128-channel halves one Mac holds of an expert (all 16 on one Mac).
pub const max_halves = 16;

/// An expert's halves on this Mac: gate and up per local half, down per local half; tables by offset.
pub const Expert = struct {
    gate: [max_halves]pack.Half,
    up: [max_halves]pack.Half,
    down: [max_halves]pack.Half,
    /// The local halves' down channel scales back to back (the down inputs' rotation reads them in one run).
    down_suh: u32,
    /// Tile-run offsets: 48 input blocks a gate or up half; 8 local halves a down output block, 48 blocks.
    gate_base: [max_halves]u32,
    up_base: [max_halves]u32,
    down_base: u32,
};

pub const Moe = struct {
    router: pack.Rows,
    bias: u64,
    signs: u32,
    /// Experts 0-255, then the shared expert.
    experts: []Expert,
};

pub const Mlp = union(enum) {
    dense: struct { gate: pack.Linear, up: pack.Linear, down: pack.Linear },
    moe: Moe,
};

pub const Layer = struct {
    number: u32,
    full: bool,
    in_norm: u64,
    post_norm: u64,
    attn: Attn,
    mlp: Mlp,
    /// The MTP layer's input norms and eh_proj (null for backbone layers).
    mtp: ?struct { enorm: u64, hnorm: u64, eh: pack.Linear, head_norm: u64 } = null,
};

fn plain(p: *pack.Pack, f: *const st.File, name: []const u8) !u64 {
    const t = f.get(name) orelse return error.MissingGlmTensor;
    return p.vector(t.bytes);
}

/// Layer `n`'s weights from the release folder `root` into `p`, sliced for c's rank, experts at the manifest's widths.
pub fn load(gpa: std.mem.Allocator, io: std.Io, root: []const u8, c: *const cfg.Config, manifest: *const ex.manifest.Manifest, n: u32, p: *pack.Pack, experts: ?*ex.experts.Layer) !Layer {
    const path = try std.fmt.allocPrint(gpa, "{s}/dense/exlr8-dense-{d:0>3}.safetensors", .{ root, n });
    defer gpa.free(path);
    var f = try st.File.open(gpa, io, path);
    defer f.close(io);
    const full = c.full.isSet(n);
    const h0 = c.head0();
    const nh = c.localHeads();
    const qh = c.qHead();
    var l: Layer = .{ .number = n, .full = full, .in_norm = try plain(p, &f, "input_layernorm.weight"), .post_norm = try plain(p, &f, "post_attention_layernorm.weight"), .attn = undefined, .mlp = undefined };
    const x_lo: u32 = @intCast(p.dense_jobs.items.len);
    const cols = [_]struct { name: []const u8, col: u32 }{ .{ .name = "self_attn.q_a_proj", .col = 0 }, .{ .name = "self_attn.kv_a_proj_with_mqa", .col = kv_col }, .{ .name = "self_attn.indexer.wk", .col = wk_col }, .{ .name = "self_attn.indexer.weights_proj", .col = wp_col } };
    for (cols[0..if (full) 4 else 2]) |m| {
        const mat = try dense.Matrix.get(&f, m.name);
        const r = (try p.linear(mat, .{ 0, mat.record.out }, .{ 0, mat.record.in })).rows;
        p.dense_jobs.items[r.job].y_col = m.col;
    }
    const q_b = try dense.Matrix.get(&f, "self_attn.q_b_proj");
    const kvb = try dense.Matrix.get(&f, "self_attn.kv_b_proj");
    const kv_rows = (try p.linear(kvb, .{ 448 * h0, 448 * (h0 + nh) }, .{ 0, c.kv_lora })).rows;
    const kvj = p.dense_jobs.items[kv_rows.job];
    _ = p.dense_jobs.pop();
    const uv_lo: u32 = @intCast(p.dense_jobs.items.len);
    for (0..nh) |h| {
        var j = kvj;
        j.out0 = @intCast(448 * h + c.nope);
        j.outs = c.v_dim;
        j.x_col = @intCast(h * c.kv_lora);
        j.y_col = @intCast(h * c.v_dim);
        try p.dense_jobs.append(p.gpa, j);
    }
    // the W_uv run ends here: the projections below append jobs of their own
    const uv_hi: u32 = @intCast(p.dense_jobs.items.len);
    l.attn = .{
        .x_jobs = .{ .lo = x_lo, .hi = x_lo + @as(u32, if (full) 4 else 2) },
        .q_b = try p.linear(q_b, .{ qh * h0, qh * (h0 + nh) }, .{ 0, c.q_lora }),
        .wq_b = if (full) (try p.linear(try dense.Matrix.get(&f, "self_attn.indexer.wq_b"), .{ 0, c.i_heads * c.i_dim }, .{ 0, c.q_lora })).rows else null,
        .kvb_w = kvj.w,
        .kvb_s = kvj.s,
        .uv_jobs = .{ .lo = uv_lo, .hi = uv_hi },
        .o = try p.linear(try dense.Matrix.get(&f, "self_attn.o_proj"), .{ 0, c.hidden }, .{ c.v_dim * h0, c.v_dim * (h0 + nh) }),
        .q_norm = try plain(p, &f, "self_attn.q_a_layernorm.weight"),
        .kv_norm = try plain(p, &f, "self_attn.kv_a_layernorm.weight"),
        .k_norm = if (full) try plain(p, &f, "self_attn.indexer.k_norm.weight") else 0,
        .k_bias = if (full) try plain(p, &f, "self_attn.indexer.k_norm.bias") else 0,
    };
    if (!c.isMoe(n)) {
        const rows: [2]usize = .{ c.rank * c.localDense(), (c.rank + 1) * c.localDense() };
        l.mlp = .{ .dense = .{
            .gate = try p.linear(try dense.Matrix.get(&f, "mlp.gate_proj"), rows, .{ 0, c.hidden }),
            .up = try p.linear(try dense.Matrix.get(&f, "mlp.up_proj"), rows, .{ 0, c.hidden }),
            .down = try p.linear(try dense.Matrix.get(&f, "mlp.down_proj"), .{ 0, c.hidden }, rows),
        } };
    } else {
        l.mlp = .{ .moe = try loadMoe(gpa, c, manifest, n, p, &f, experts orelse return error.MissingGlmExperts) };
    }
    if (n == c.layers) l.mtp = .{
        .enorm = try plain(p, &f, "enorm.weight"),
        .hnorm = try plain(p, &f, "hnorm.weight"),
        .eh = try p.linear(try dense.Matrix.get(&f, "eh_proj"), .{ 0, c.hidden }, .{ 0, 2 * c.hidden }),
        .head_norm = try plain(p, &f, "shared_head.norm.weight"),
    };
    return l;
}

fn loadMoe(gpa: std.mem.Allocator, c: *const cfg.Config, manifest: *const ex.manifest.Manifest, n: u32, p: *pack.Pack, f: *const st.File, x: *ex.experts.Layer) !Moe {
    const router = (try p.linear(try dense.Matrix.get(f, "mlp.gate"), .{ 0, c.experts }, .{ 0, c.hidden })).rows;
    const bias = try plain(p, f, "mlp.gate.e_score_correction_bias");
    const signs = try p.layerSigns(x);
    const all = try gpa.alloc(Expert, c.experts + 1);
    for (all, 0..) |*e, ei| {
        const id: u16 = @intCast(ei);
        for ([_]Proj{ .gate, .up, .down }) |proj| for (0..c.localHalves()) |hl| {
            const h = c.half0() + hl;
            const k = manifest.width(n, id, switch (proj) {
                .gate => .gate,
                .up => .up,
                .down => .down,
            }, h) orelse return error.BadExlr8Manifest;
            const half = try x.half(id, proj, h, k);
            const got = try p.half(half);
            switch (proj) {
                .gate => e.gate[hl] = got,
                .up => e.up[hl] = got,
                .down => e.down[hl] = got,
            }
        };
        e.down_suh = @intCast(p.halves.items.len);
        for (0..c.localHalves()) |hl| {
            var chan: [128]f16 = undefined;
            @memcpy(&chan, p.halves.items[e.down[hl].chan..][0..128]);
            try p.halves.appendSlice(p.gpa, &chan);
        }
        for (0..c.localHalves()) |hl| {
            e.gate_base[hl] = try runs(p, e.gate[hl], 48, 1);
            e.up_base[hl] = try runs(p, e.up[hl], 48, 1);
        }
        e.down_base = @intCast(p.bases.items.len);
        for (0..48) |ob| for (e.down[0..c.localHalves()]) |d| try p.bases.append(p.gpa, d.at + ob * 64 * ex.codec.tileBytes(d.k));
    }
    return .{ .router = router, .bias = bias, .signs = signs, .experts = all };
}

/// `n` tile runs of a half, `step` runs apart (64 tiles a run): their offsets, appended to the pack's bases.
fn runs(p: *pack.Pack, h: pack.Half, n: usize, step: usize) !u32 {
    const at: u32 = @intCast(p.bases.items.len);
    for (0..n) |b| try p.bases.append(p.gpa, h.at + b * step * 64 * ex.codec.tileBytes(h.k));
    return at;
}
