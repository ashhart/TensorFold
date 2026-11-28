//! A paired prompt step's pass (TF_GLM_PROMPT_PAIR): two chunks site by site, each one's exchanges under the other's.
const trace = @import("cuda_prefill_trace.zig");
const glue = @import("cuda_glue.zig");
const latent = @import("cuda_latent.zig");
const fw = @import("cuda_forward.zig");
const pieces = @import("cuda_forward_pieces.zig");
const moe = @import("cuda_moe.zig");
const experts = @import("cuda_experts.zig");
const Forward = fw.Forward;
const Pieces = fw.Pieces;
const Site = fw.Forward.Site;
const room = fw.Forward.room;
const share = fw.Forward.share;

/// Two consecutive prompt chunks of one sequence (`b` a view of `a`'s at its end), each on `pieced`'s kernels and bits.
pub fn pairBody(a: *const Forward, b: *const Forward, ra: usize, rb: usize, final: bool) !void {
    const two = [_]*const Forward{ a, b };
    const rows = [_]usize{ ra, rb };
    var cuts: [2][fw.max_pieces][2]usize = undefined;
    var n: [2]usize = undefined;
    for (two, rows, 0..) |f, R, i| {
        if (f.pieces == null or f.prof != null or f.taps != null) return error.BadPair;
        trace.part(if (i == 0) 0 else ra, R);
        trace.context(null, null, .chunk);
        const embed_stamp = trace.span(f.stream, .embed);
        const c = &f.ranks[0].w.config;
        for (f.ranks) |r| {
            if (r.segs.len > 0 or !r.b.prefill or R == 0 or R > r.b.rows) return error.BadPair;
            try glue.embedB16(f.set(), f.stream, r.b.ids, r.w.embed, r.b.x, R, c.hidden, c.hc);
        }
        if (f.images) |im| {
            const image_stamp = trace.span(f.stream, .image_splice);
            defer image_stamp.end();
            for (f.ranks) |r| try im.v.splice(f.stream, im.slot, r.st.pos, r.st.pos + R, r.b.x, c.hc);
        }
        embed_stamp.end();
        n[i] = fw.ranges(R, f.pieces.?.count, &cuts[i]);
        if (n[i] == 1 and !f.splits(R)) return error.BadPair; // run would take this chunk whole
        trace.context(0, null, .mixer);
        for (f.ranks) |r| try f.pre(r, r.w.layers[0].attn_hc, r.w.layers[0].in_norm, R);
    }
    const layers = a.ranks[0].w.layers;
    var li: usize = 0;
    var di: usize = 0;
    for (layers, 0..) |l, j| {
        // b's mixer follows a's on the main stream, so a's caches, KDA state and conv rows of this layer are in
        for (two, rows, 0..) |f, R, i| {
            trace.part(if (i == 0) 0 else ra, R);
            trace.context(j, null, .mixer);
            if (j > 0) try awaitShared(f, f.pieces.?, n[i]);
            for (f.ranks) |r| try f.mixer(r, r.w.layers[j], li, di, R, @min(latent.chunksFor(r.st.pos + R), r.b.nch), null);
            try exchangeGlue(f, f.pieces.?, j, .mixer, cuts[i][0..n[i]]);
        }
        const joint = a.pair_experts and switch (l.ffn) {
            .dense => false,
            .moe => |m| m.exl3 == null and m.tfq3 == null and m.routed.format == .nvfp4,
        };
        if (joint) switch (l.ffn) {
            .moe => |m| try pairedMoe(a, b, m, j, ra, rb, cuts[0][0..n[0]], cuts[1][0..n[1]]),
            .dense => unreachable,
        } else for (two, rows, 0..) |f, R, i| {
            trace.part(if (i == 0) 0 else ra, R);
            trace.context(j, null, .ffn);
            try awaitShared(f, f.pieces.?, n[i]);
            for (f.ranks) |r| try f.ffn(r, r.w.layers[j], R, null);
            try exchangeGlue(f, f.pieces.?, j, .ffn, cuts[i][0..n[i]]);
        }
        switch (l.mixer) {
            .kda => li += 1,
            .mla => di += 1,
        }
    }
    const heads = [2]bool{ false, final };
    for (two, rows, 0..) |f, R, i| {
        trace.part(if (i == 0) 0 else ra, R);
        trace.context(null, null, .chunk);
        try awaitShared(f, f.pieces.?, n[i]);
        try f.finish(R, heads[i]);
    }
}

/// Both chunks' NVFP4 routed experts on one plan, with shared prefixes and combines on their original pieces.
fn pairedMoe(a: *const Forward, b: *const Forward, m: moe.Layer, j: usize, ra: usize, rb: usize, cuts_a: []const [2]usize, cuts_b: []const [2]usize) !void {
    trace.part(0, ra);
    trace.context(j, null, .ffn);
    const slots = m.top_k + 1;
    const split = ra * slots;
    const prepare_a = trace.ffnPart(a.stream, j, 0, ra);
    try awaitShared(a, a.pieces.?, cuts_a.len);
    for (a.ranks) |r| {
        const k = a.moeKernels();
        const d = r.w.config.hidden;
        try moe.prepare(k, a.stream, m, r.b.moe, r.b.normed, d, ra, r.b.moe.pick, null); // prompt rows never take the direct path
        for (cuts_a, 0..) |piece, q| {
            trace.context(j, q, .ffn);
            const shared_stamp = trace.span(a.stream, .shared_expert);
            try moe.shared(k, a.stream, m, r.b.moe, r.b.normed, d, piece[0], piece[1], true);
            shared_stamp.end();
        }
    }
    prepare_a.end();
    trace.part(ra, rb);
    trace.context(j, null, .ffn);
    try awaitShared(b, b.pieces.?, cuts_b.len);
    for (a.ranks, b.ranks) |ar, br| {
        trace.part(ra, rb);
        trace.context(j, null, .ffn);
        const prepare_b = trace.ffnPart(a.stream, j, ra, rb);
        const k = a.moeKernels();
        const d = ar.w.config.hidden;
        try moe.prepare(k, a.stream, m, br.b.moe, br.b.normed, d, rb, ar.b.moe.pick + split * 4, null);
        for (cuts_b, 0..) |piece, q| {
            trace.context(j, q, .ffn);
            const shared_stamp = trace.span(a.stream, .shared_expert);
            try moe.shared(k, a.stream, m, br.b.moe, br.b.normed, d, piece[0], piece[1], true);
            shared_stamp.end();
        }
        prepare_b.end();
        const joint_stamp = trace.jointFfn(a.stream, j, ra, rb);
        const plan: experts.Plan = .{ .route = ar.b.moe.plan, .slots = slots, .experts = m.count + 1, .prompt = true };
        const expert_stamp = trace.span(a.stream, .experts);
        try experts.runPair(k.experts, a.stream, m.routed, plan, ar.b.moe.pick, ar.b.normed, br.b.normed, d, ra, rb, ar.b.moe.eact, br.b.moe.eact, ar.b.moe.ey, br.b.moe.ey, @intCast(m.count));
        expert_stamp.end();
        joint_stamp.end();
    }
    const exchange_a = trace.ffnPart(a.stream, j, 0, ra);
    try exchangeCombined(a, a.pieces.?, j, m, cuts_a);
    exchange_a.end();
    const exchange_b = trace.ffnPart(b.stream, j, ra, rb);
    try exchangeCombined(b, b.pieces.?, j, m, cuts_b);
    exchange_b.end();
}

/// A prepared MoE site's combine, then the same exchange, glue and sharing as exchangeGlue.
fn exchangeCombined(f: *const Forward, p: *const Pieces, j: usize, m: moe.Layer, cut: []const [2]usize) !void {
    const D = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    const last = j + 1 == f.ranks[0].w.layers.len;
    for (cut, 0..) |piece, q| {
        trace.context(j, q, .ffn);
        const lo = piece[0];
        const n = piece[1] - lo;
        const split = f.splits(n);
        const output_stamp = trace.span(f.stream, .output_projection);
        for (f.ranks) |r| try moe.combine(f.moeKernels(), f.stream, m, r.b.moe, lo, piece[1], true, if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
        output_stamp.end();
        try p.filled[q].record(f.stream);
        const filled_stamp = trace.span(p.side, .wait_filled);
        try p.side.wait(p.filled[q]);
        filled_stamp.end();
        const partial_stamp = trace.span(p.side, .partials);
        try pieces.partials(f, p, lo, n, split);
        partial_stamp.end();
        var side = f.*;
        side.stream = p.side;
        if (split) {
            for (f.ranks) |r| {
                const mine = share(n, world, r.w.share.rank);
                try side.glueRows(r, j, .ffn, lo + mine[0], lo + mine[1], room(r, lo), r.b.part + (lo + mine[0]) * D * 4); // split sends peers only (#4510)
            }
            const glued_stamp = trace.span(p.side, .glued);
            try pieces.shareGlued(f, p, lo, n, last);
            glued_stamp.end();
        } else for (f.ranks) |r| try side.glueRows(r, j, .ffn, lo, piece[1], room(r, lo), 0);
        try p.shared[q].record(p.side);
    }
}

/// A paired site: each piece filled on the main stream, then on the side stream its exchange, glue and (split) sharing.
fn exchangeGlue(f: *const Forward, p: *const Pieces, j: usize, site: Site, cut: []const [2]usize) !void {
    const D = f.ranks[0].w.config.hidden;
    const world = f.ranks[0].w.share.world;
    const last = site == .ffn and j + 1 == f.ranks[0].w.layers.len;
    var g = f.*; // the glue's launches go on the side stream
    g.stream = p.side;
    for (cut, 0..) |piece, q| {
        trace.context(j, q, if (site == .ffn) .ffn else .mixer);
        const lo = piece[0];
        const n = piece[1] - lo;
        const split = f.splits(n);
        for (f.ranks) |r| try f.fillRows(r, r.w.layers[j], site, lo, piece[1], if (split) r.b.part + lo * D * 4 else room(r, lo) + r.w.share.rank * n * D * 4);
        try p.filled[q].record(f.stream);
        const filled_stamp = trace.span(p.side, .wait_filled);
        try p.side.wait(p.filled[q]);
        filled_stamp.end();
        const partial_stamp = trace.span(p.side, .partials);
        try pieces.partials(f, p, lo, n, split);
        partial_stamp.end();
        if (split) {
            for (f.ranks) |r| {
                const mine = share(n, world, r.w.share.rank);
                try g.glueRows(r, j, site, lo + mine[0], lo + mine[1], room(r, lo), r.b.part + (lo + mine[0]) * D * 4);
            }
            const glued_stamp = trace.span(p.side, .glued);
            try pieces.shareGlued(f, p, lo, n, last);
            glued_stamp.end();
        } else for (f.ranks) |r| try g.glueRows(r, j, site, lo, piece[1], room(r, lo), 0);
        try p.shared[q].record(p.side);
    }
}

/// Before a paired chunk's next site: its last site's pieces glued and shared on the side stream.
fn awaitShared(f: *const Forward, p: *const Pieces, count: usize) !void {
    const stamp = trace.span(f.stream, .wait_shared);
    defer stamp.end();
    for (p.shared[0..count]) |e| try f.stream.wait(e);
}
