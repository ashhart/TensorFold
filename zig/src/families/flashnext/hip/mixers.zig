//! Flash Next's two mixers on a HIP rank: the Gated DeltaNet and the sparse attention with its indexer.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");

const Ops = hip.ops.Ops;
const Tensor = hip.ops.Tensor;
const take = forward.take;
const mixers_attend = @import("mixers_attend.zig");
const attend = mixers_attend.attend;

/// normalize_qk's constants: q and k scaled by rsqrt(sum + 1e-6) (q also by dk^-0.5) as an RMS norm with a weight.
pub fn qkConstants(key_dim: usize) struct { q: f32, k: f32, eps: f32 } {
    const inv = std.math.pow(f64, @floatFromInt(key_dim), -0.5);
    return .{ .q = @floatCast(inv * inv), .k = @floatCast(inv), .eps = @floatCast(1e-6 * inv * inv) };
}

/// The Gated DeltaNet over every segment: products and norms on all rows, conv and recurrence a stream at a time.
pub fn linear(
    o: Ops,
    m: *const model.Model,
    l: model.Linear,
    segs: []const forward.Seg,
    index: usize,
    x: Tensor,
    rows: usize,
    window: bool,
) !Tensor {
    const d = m.dims;
    const ch = d.convWidth();
    const kw = d.key_heads * d.key_dim;
    const vw = d.value_heads * d.value_dim;
    var outs: [4]Tensor = undefined;
    if (!try o.projectGroup(x, &.{ l.qkv, l.z, l.a, l.b }, rows, &outs)) {
        for (&outs, [_]model.Projection{ l.qkv, l.z, l.a, l.b }) |*t, p| t.* = try o.project(x, p, rows, false);
    }
    const qkv, const z, const a, const b = outs;
    const q = try o.arena.of(f32, rows * kw);
    const k = try o.arena.of(f32, rows * kw);
    const v = try o.arena.of(f32, rows * vw);
    const gate = try o.arena.of(f32, rows * d.value_heads);
    const beta = try o.arena.of(f32, rows * d.value_heads);
    if (window) {
        // a window's conv, split, norms and gates in one launch a stream; k, v, gate and beta into its window inputs
        var at_row: usize = 0;
        for (segs) |g| {
            const n = g.tokens.len;
            const nh = n * d.value_heads;
            const in = g.c.inputs(index);
            try o.l.tf_conv_split(.{
                .x = forward.at(qkv, at_row * ch).ptr,
                .kind = @backingInt(m.act),
                .weight = l.conv,
                .state = g.c.linearAt(index).conv,
                .states = g.c.snapshot(index, 0),
                .from = g.c.start(),
                .qn = q + at_row * kw * 4,
                .kn = in.k,
                .v = in.v,
                .channels = @intCast(ch),
                .kernel = @intCast(d.conv),
                .rows = @intCast(n),
                .kw = @intCast(kw),
                .vw = @intCast(vw),
                .qw = m.qk.q,
                .kw_w = m.qk.k,
                .eps = m.qk.eps,
                .norm = 128,
                .ga = forward.at(a, at_row * d.value_heads).ptr,
                .gb = forward.at(b, at_row * d.value_heads).ptr,
                .a_log = l.a_log,
                .dt_bias = l.dt_bias,
                .gate = in.gate,
                .beta = in.beta,
                .gcount = @intCast(nh),
                .heads = @intCast(d.value_heads),
            }, o.stream);
            at_row += n;
        }
        return recur(o, m, l, segs, index, q, k, v, gate, beta, z, rows, true);
    }
    if (o.fused() and d.key_dim == 128 and d.conv <= 8) {
        // a stream's conv, split, q and k norms and gates in one launch (tf_gdn_prompt_prep: the chain's values)
        var at_row: usize = 0;
        for (segs) |g| {
            const n = g.tokens.len;
            const st = g.c.linearAt(index);
            const conv_bytes = (d.conv - 1) * ch * 4;
            const next = try o.arena.take(conv_bytes);
            try o.l.tf_gdn_prompt_prep(
                @ptrFromInt(forward.at(qkv, at_row * ch).ptr),
                @backingInt(m.act),
                @ptrFromInt(l.conv),
                @ptrFromInt(st.conv),
                @ptrFromInt(next),
                @ptrFromInt(q + at_row * kw * 4),
                @ptrFromInt(k + at_row * kw * 4),
                @ptrFromInt(v + at_row * vw * 4),
                @ptrFromInt(m.qk.q),
                @ptrFromInt(m.qk.k),
                m.qk.eps,
                @intCast(n),
                @intCast(kw),
                @intCast(vw),
                @intCast(d.conv),
                @ptrFromInt(forward.at(a, at_row * d.value_heads).ptr),
                @ptrFromInt(forward.at(b, at_row * d.value_heads).ptr),
                @ptrFromInt(l.a_log),
                @ptrFromInt(l.dt_bias),
                @ptrFromInt(gate + at_row * d.value_heads * 4),
                @ptrFromInt(beta + at_row * d.value_heads * 4),
                @intCast(d.value_heads),
                o.stream,
            );
            if (o.stream != hip.counting) try hip.runtime.check(o.l.r.api.hipMemcpyDtoDAsync(@ptrFromInt(st.conv), @ptrFromInt(next), conv_bytes, o.stream));
            at_row += n;
        }
        return recur(o, m, l, segs, index, q, k, v, gate, beta, z, rows, false);
    }
    const mixed = try o.arena.of(f32, rows * ch);
    var off: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        const st = g.c.linearAt(index);
        const out = mixed + off * ch * 4;
        const conv_bytes = (d.conv - 1) * ch * 4;
        const next = try o.arena.take(conv_bytes);
        try o.convPrefill(forward.at(qkv, off * ch), l.conv, st.conv, out, next, n, ch, d.conv);
        if (o.stream != hip.counting) try hip.runtime.check(o.l.r.api.hipMemcpyDtoDAsync(
            @ptrFromInt(st.conv),
            @ptrFromInt(next),
            conv_bytes,
            o.stream,
        ));
        off += n;
    }
    const mixed_t: Tensor = .{ .ptr = mixed, .kind = .f32 };
    const qc = try o.arena.of(f32, rows * kw);
    const kc = try o.arena.of(f32, rows * kw);
    try o.copyCols(mixed_t, ch, 0, qc, rows, kw);
    try o.copyCols(mixed_t, ch, kw, kc, rows, kw);
    try o.copyCols(mixed_t, ch, 2 * kw, v, rows, vw);
    try o.rms(
        .{ .ptr = qc, .kind = .f32 },
        m.qk.q,
        .{ .ptr = q, .kind = .f32 },
        rows * d.key_heads,
        d.key_dim,
        m.qk.eps,
    );
    try o.rms(
        .{ .ptr = kc, .kind = .f32 },
        m.qk.k,
        .{ .ptr = k, .kind = .f32 },
        rows * d.key_heads,
        d.key_dim,
        m.qk.eps,
    );
    try o.gdnGatePrefill(a, b, l.a_log, l.dt_bias, gate, beta, rows, d.value_heads);
    return recur(o, m, l, segs, index, q, k, v, gate, beta, z, rows, false);
}

/// The delta rule a stream at a time (a window's rows each with its state kept aside), the gated norm, the out product.
fn recur(
    o: Ops,
    m: *const model.Model,
    l: model.Linear,
    segs: []const forward.Seg,
    index: usize,
    q: u64,
    k: u64,
    v: u64,
    gate: u64,
    beta: u64,
    z: Tensor,
    rows: usize,
    window: bool,
) !Tensor {
    const d = m.dims;
    const kw = d.key_heads * d.key_dim;
    const vw = d.value_heads * d.value_dim;
    const y = try o.arena.of(f32, rows * vw);
    var off: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        // a window's k, v, gate and beta are in the stream's window inputs (linear's conv and split wrote them there)
        const in: cache.WindowInputs = if (window) g.c.inputs(index) else .{
            .k = k + off * kw * 4,
            .v = v + off * vw * 4,
            .gate = gate + off * d.value_heads * 4,
            .beta = beta + off * d.value_heads * 4,
        };
        if (window and n > 1) {
            // the window holds the state (its keep runs it); a one-row window takes the plain scan, which stores it
            try o.gatedDeltaHold(q + off * kw * 4, in.k, in.v, in.gate, in.beta, g.c.linearAt(index).state, y + off * vw * 4, n, d.key_heads, d.value_heads, d.key_dim, d.value_dim);
            off += n;
            continue;
        }
        try o.gatedDelta(
            q + off * kw * 4,
            in.k,
            in.v,
            in.gate,
            in.beta,
            g.c.linearAt(index).state,
            y + off * vw * 4,
            n,
            d.key_heads,
            d.value_heads,
            d.key_dim,
            d.value_dim,
            null,
            0,
        );
        off += n;
    }
    const out = try take(o, m.act, rows * vw);
    try o.gnormOut(y, l.norm, z, out, rows * d.value_heads, d.value_dim, d.eps, d.gate);
    return o.project(out, l.out, rows, l.out.partial);
}

/// A segment's keys, values, index keys and pools into its sparse caches: `sparse`'s eager writes (not tf_fn_qsa_prep).
fn writeKeys(
    o: Ops,
    m: *const model.Model,
    d: model.Dims,
    s: model.Sparse,
    g: forward.Seg,
    layer: usize,
    keys: Tensor,
    values: Tensor,
    idx: Tensor,
    off: usize,
    at_pos: u64,
) !void {
    const hd = d.head_dim;
    const di = d.index_dim;
    const width = (d.index_heads + 1) * di;
    const n = g.tokens.len;
    const kv = g.c.sparse(layer);
    const cap = g.c.layout.cap;
    const pos = g.pos;
    try o.qkRope(
        forward.at(keys, off * d.kv_heads * hd),
        d.kv_heads * hd,
        hd,
        s.k_norm,
        d.eps,
        n,
        d.kv_heads,
        hd,
        d.rotary,
        d.theta,
        at_pos,
        null,
        kv.k,
        cap,
    );
    try o.kvWrite(forward.at(values, off * d.kv_heads * hd), kv.v, n, d.kv_heads, hd, cap, pos);
    try o.copyCols(
        forward.at(idx, off * width),
        width,
        d.index_heads * di,
        kv.raw + pos * di * m.act.size(),
        n,
        di,
    );
    const blocks = (pos + n) / d.index_ratio;
    const pooled = g.c.pooled;
    if (blocks > pooled) try o.l.fn_idx_pool(
        @ptrFromInt(kv.raw),
        @ptrFromInt(kv.pooled + pooled * di * m.act.size()),
        @ptrFromInt(s.ik_norm),
        @backingInt(m.act),
        @intCast(pooled),
        @intCast(blocks - pooled),
        @intCast(di),
        @intCast(d.index_ratio),
        @intCast(d.rotary),
        d.theta,
        d.eps,
        o.stream,
    );
}

/// A prompt's rows into a sparse layer's caches only (sparse()'s bytes, no attention): the MTP head's prompt rows.
pub fn sparseKeys(o: Ops, m: *const model.Model, d: model.Dims, s: model.Sparse, g: forward.Seg, layer: usize, x: Tensor, dev_pos: u64) !void {
    const rows = g.tokens.len;
    var outs: [4]Tensor = undefined;
    if (!try o.projectGroup(x, &.{ s.q, s.k, s.v, s.index }, rows, &outs)) {
        for (&outs, [_]model.Projection{ s.q, s.k, s.v, s.index }) |*t, p| t.* = try o.project(x, p, rows, false);
    }
    try writeKeys(o, m, d, s, g, layer, outs[1], outs[2], outs[3], 0, dev_pos);
}

/// The rows' ends and complete blocks on the device, and the blocks a window's scores walk (see forward.Staged).
pub const Rows = struct { ends: u64, complete: u64, bound: usize };

/// The sparse attention over every segment's rows; with `staged`, a window's launches read positions from the device.
pub fn sparse(
    o: Ops,
    m: *const model.Model,
    d: model.Dims,
    s: model.Sparse,
    segs: []const forward.Seg,
    layer: usize,
    x: Tensor,
    rows: usize,
    dev_pos: u64,
    host: *forward.Host,
    staged: ?Rows,
) !Tensor {
    const hd = d.head_dim;
    const di = d.index_dim;
    const width = (d.index_heads + 1) * di;
    const kind = @backingInt(m.act);
    var outs: [4]Tensor = undefined;
    if (!try o.projectGroup(x, &.{ s.q, s.k, s.v, s.index }, rows, &outs)) {
        for (&outs, [_]model.Projection{ s.q, s.k, s.v, s.index }) |*t, p| t.* = try o.project(x, p, rows, false);
    }
    const qg, const keys, const values, const idx = outs;
    const q32 = try o.arena.of(f32, rows * d.heads * hd);
    const iq = try o.arena.of(f32, rows * d.index_heads * di);
    const graphed = o.window and staged != null;
    const prep = prepWrites(graphed, rows, segs.len, segs[0].c.pooled, segs[0].pos, d.index_ratio, m.act);
    // at device positions the first segment's prep launch also takes q and the index q (tf_fn_qsa_prep)
    if (!prep) try o.qkRope(
        qg,
        d.heads * 2 * hd,
        2 * hd,
        s.q_norm,
        d.eps,
        rows,
        d.heads,
        hd,
        d.rotary,
        d.theta,
        dev_pos,
        q32,
        null,
        0,
    );
    const att = try o.arena.of(f32, rows * d.heads * hd);
    if (!prep) try o.qkRope(idx, width, di, s.iq_norm, d.eps, rows, d.index_heads, di, d.rotary, d.theta, dev_pos, iq, null, 0);
    var off: usize = 0;
    for (segs) |g| {
        const n = g.tokens.len;
        const kv = g.c.sparse(layer);
        const cap = g.c.layout.cap;
        const pos = g.pos;
        const at_pos = dev_pos + off * 4;
        if (prep) {
            // q and the index q (the first segment's launch), k, v, the raw index key and the pools in one launch
            const first = off == 0;
            const rope = hip.launches.flashnext.QsaRope;
            try o.l.fn_qsa_prep(.{
                .q = if (first) rope{ .src = qg.ptr, .s_row = @intCast(d.heads * 2 * hd), .s_head = @intCast(2 * hd), .heads = @intCast(d.heads), .width = @intCast(hd), .weight = s.q_norm, .wide = q32 } else .{},
                .iq = if (first) rope{ .src = idx.ptr, .s_row = @intCast(width), .s_head = @intCast(di), .heads = @intCast(d.index_heads), .width = @intCast(di), .weight = s.iq_norm, .wide = iq } else .{},
                .k = .{ .src = forward.at(keys, off * d.kv_heads * hd).ptr, .s_row = @intCast(d.kv_heads * hd), .s_head = @intCast(hd), .heads = @intCast(d.kv_heads), .width = @intCast(hd), .weight = s.k_norm, .cache = kv.k },
                .v = forward.at(values, off * d.kv_heads * hd).ptr,
                .v_cache = kv.v,
                .v_width = @intCast(hd),
                .raw = forward.at(idx, off * width).ptr,
                .raw_stride = @intCast(width),
                .raw_offset = @intCast(d.index_heads * di),
                .raw_cache = kv.raw,
                .raw_width = @intCast(di),
                .pooled = kv.pooled,
                .pool_scale = s.ik_norm,
                .ratio = @intCast(d.index_ratio),
                .pos = dev_pos,
                .seg_pos = at_pos,
                .rows = @intCast(rows),
                .n = @intCast(n),
                .total = @intCast(cap),
                .kind = kind,
                .eps = d.eps,
                .rotary = @intCast(d.rotary),
                .theta = d.theta,
            }, o.stream);
        } else try writeKeys(o, m, d, s, g, layer, keys, values, idx, off, at_pos);
        const r: ?Rows = if (staged) |st| .{
            .ends = st.ends + off * 4,
            .complete = st.complete + off * 4,
            .bound = st.bound,
        } else null;
        try attend(
            o,
            m,
            d,
            kv,
            cap,
            q32 + off * d.heads * hd * 4,
            iq + off * d.index_heads * di * 4,
            att + off * d.heads * hd * 4,
            n,
            pos,
            host,
            r,
            graphed,
        );
        off += n;
    }
    const gated = try take(o, m.act, rows * d.heads * hd);
    try o.attnGate(att, qg, gated, rows, d.heads, hd, true);
    return o.project(gated, s.o, rows, s.o.partial);
}

/// Rows from which a call's attention takes the prompt kernel: below, the tile runs in parts for a wider grid.
const prompt_rows = 16;

/// Whether tf_fn_qsa_prep takes the writes: a graphed window, or one 16-bit segment < prompt_rows with pools current.
pub fn prepWrites(graphed: bool, rows: usize, segs: usize, pooled: usize, pos: usize, ratio: usize, act: model.Kind) bool {
    if (graphed) return true;
    if (act == .f32 or rows == 0 or rows >= prompt_rows or segs != 1) return false;
    return pooled == pos / ratio;
}

test "QSA writes: a graphed window always takes the prep launch; an eager call only when its pools are all its rows'" {
    // graphed: always (its caches are pooled through every window's prep)
    try std.testing.expect(prepWrites(true, 8, 1, 0, 4000, 4, .bf16));
    // the MTP head's step: one row, its cache pooled to its first row (pos 4001: 1,000 blocks)
    try std.testing.expect(prepWrites(false, 1, 1, 1000, 4001, 4, .bf16));
    try std.testing.expect(prepWrites(false, 1, 1, 1000, 4003, 4, .bf16));
    // a cache behind its row's blocks (writeKeys pools 999 too), a prompt's rows, two segments, fp32 rows: eager
    try std.testing.expect(!prepWrites(false, 1, 1, 999, 4001, 4, .bf16));
    try std.testing.expect(!prepWrites(false, 16, 1, 1000, 4001, 4, .bf16));
    try std.testing.expect(!prepWrites(false, 2, 2, 1000, 4001, 4, .bf16));
    try std.testing.expect(!prepWrites(false, 1, 1, 1000, 4001, 4, .f32));
}
