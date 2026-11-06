//! One forward of rows through the decoder's 12 layers on CUDA: a prompt pass (experts grouped on the host) or a step.
const std = @import("std");
const cuda = @import("cuda");
const cfg = @import("config.zig");
const ks = @import("cuda_kernels.zig");
const decoder_mod = @import("cuda_decoder.zig");
const Decoder = decoder_mod.Decoder;
const Part = decoder_mod.Scratch.Part;

const H = cfg.hidden;
const D = cfg.router_dim;
const E = cfg.expert_ffn;

const Pass = struct {
    d: *Decoder,
    s0: usize,
    r: usize,
    prompt: bool,
    /// The rows' rotary values and candidate rows: the prompt's parts, or a step's packed row.
    cos: u64,
    sin: u64,
    qmask: u64,

    fn s(p: *const Pass, part: Part) u64 {
        return p.d.at(part);
    }

    fn run(p: *const Pass, comptime name: []const u8, grid: [3]usize, threads: u32, args: anytype) !void {
        try p.d.k.run(name, p.d.s, grid, threads, args);
    }

    /// y [r, n] = x [r, k] w^T: cuBLASLt, one algorithm an (n, k), for a prompt's rows; the one-row product for a step.
    fn mm(p: *const Pass, x: u64, w: u64, y: u64, n: usize, k: usize) !void {
        if (p.prompt) return p.d.gemms.run(p.d.s, x, w, y, p.r, n, k);
        try p.matvec(x, w, y, n, k);
    }

    fn matvec(p: *const Pass, x: u64, w: u64, y: u64, n: usize, k: usize) !void {
        try p.run("matvec", .{ (n + 31) / 32, 1, 1 }, 256, .{ x, w, y, n, k });
    }

    /// With `add`: x = bf16(x + a); then y = the norm of x under `w`.
    fn norm(p: *const Pass, add: bool, a: Part, w: u64) !void {
        try p.run("norm", .{ p.r, 1, 1 }, 256, .{ p.s(.x), p.s(a), w, p.s(.y), @as(c_int, @intFromBool(add)), cfg.eps });
    }

    fn silu(p: *const Pass, g: u64, u: u64, y: u64, rows: usize, width: usize, stride: usize) !void {
        try p.run("silu_mul", .{ rows, ks.blocks(width), 1 }, 256, .{ g, u, y, width, stride, stride });
    }

    fn attention(p: *const Pass, l: usize) !void {
        const d = p.d;
        const layer = d.w.layers[l];
        const c = d.caches[l];
        const pos0: c_int = @intCast(p.s0);
        try p.mm(p.s(.y), layer.q.dev, p.s(.proj), 3 * H, H);
        try p.mm(p.s(.y), layer.router.q.dev, p.s(.rproj), 2 * D, H);
        const r = layer.router;
        try p.run("qk", .{ p.r, 1, 1 }, cfg.router_dim, .{ p.s(.proj), p.s(.rproj), p.cos, p.sin, r.q_norm.dev, r.k_norm.dev, p.s(.q), c.k, c.v, p.s(.rq), c.rk, pos0, cfg.eps });
        const c0 = p.s0 / cfg.block;
        const c1 = (p.s0 + p.r) / cfg.block;
        if (c1 > c0) try p.run("block_keys", .{ c1 - c0, 1, 1 }, 32, .{ c.rk, r.head_cls.dev, c.kb, c0 });
        try p.run("route", .{ p.r, 1, 1 }, 256, .{ p.s(.rq), c.kb, d.masks.?.ptr, p.qmask, p.s(.sel), p.s(.bias), pos0, d.words });
        try p.run("attend", .{ cfg.heads, p.r, 1 }, ks.attend_threads, .{ p.s(.q), c.k, c.v, p.s(.sel), p.s(.bias), p.s(.att), pos0, @as(c_int, @intFromBool(p.prompt)) });
        try p.mm(p.s(.att), layer.o.dev, p.s(.o), H, H);
    }

    /// The layer's MLP rows from y into .mlp: the dense FFN, or the shared expert plus the 6 routed ones.
    fn ffn(p: *const Pass, l: usize) !void {
        const d = p.d;
        const layer = d.w.layers[l];
        const width: usize = if (l == 0) cfg.dense_ffn else cfg.shared_ffn;
        const f = p.s(.ffn);
        try p.mm(p.s(.y), layer.ffn.gate.dev, f, 2 * width, H);
        try p.silu(f, f + width * 2, p.s(.act), p.r, width, 2 * width);
        try p.mm(p.s(.act), layer.ffn.down.dev, p.s(if (l == 0) .mlp else .shared), H, width);
        if (l == 0) return;
        try p.run("moe_gate", .{ p.r, 1, 1 }, 1024, .{ p.s(.y), layer.gate.?.dev, p.s(.eids), p.s(.wts) });
        if (p.prompt) return p.experts(l);
        const table = d.experts.ptr + l * cfg.experts * 3 * 8;
        try p.run("expert_gu", .{ cfg.top_k * 2 * E / 32, 1, 1 }, 256, .{ p.s(.y), table, p.s(.eids), p.s(.ge), E });
        try p.silu(p.s(.ge), p.s(.ge) + E * 2, p.s(.ae), cfg.top_k, E, 2 * E);
        try p.run("expert_down", .{ cfg.top_k * H / 32, 1, 1 }, 256, .{ p.s(.ae), table, p.s(.eids), p.s(.de), E });
        try p.run("combine", .{ 1, ks.blocks(H), 1 }, 256, .{ p.s(.de), p.s(.ident), p.s(.wts), p.s(.shared), p.s(.mlp) });
    }

    /// A prompt's routed experts: the gate's picks read back, rows grouped by expert, one product an expert.
    fn experts(p: *const Pass, l: usize) !void {
        const d = p.d;
        const n = p.r * cfg.top_k;
        const ids = d.host.eids[0..n];
        try d.get(.eids, std.mem.sliceAsBytes(ids));
        var start: [cfg.experts + 1]usize = @splat(0);
        for (ids) |e| {
            if (e >= cfg.experts) return error.UocrExpertId;
            start[e + 1] += 1;
        }
        for (1..start.len) |e| start[e] += start[e - 1];
        var fill = start;
        const gidx = d.host.gidx[0..n];
        const slot = d.host.slot[0..n];
        for (ids, 0..) |e, i| {
            gidx[fill[e]] = @intCast(i / cfg.top_k);
            slot[i] = @intCast(fill[e]);
            fill[e] += 1;
        }
        try d.put(.gidx, std.mem.sliceAsBytes(gidx));
        try d.put(.slot, std.mem.sliceAsBytes(slot));
        try p.run("gather", .{ n, ks.blocks(H), 1 }, 256, .{ p.s(.y), p.s(.gidx), p.s(.xe) });
        const layer = d.w.layers[l];
        for (layer.experts, 0..) |ex, e| {
            const m = start[e + 1] - start[e];
            if (m == 0) continue;
            const at = start[e];
            try d.gemms.run(d.s, p.s(.xe) + at * H * 2, ex.gate.dev, p.s(.ge) + at * E * 2, m, E, H);
            try d.gemms.run(d.s, p.s(.xe) + at * H * 2, ex.up.dev, p.s(.ue) + at * E * 2, m, E, H);
        }
        try p.silu(p.s(.ge), p.s(.ue), p.s(.ae), n, E, E);
        for (layer.experts, 0..) |ex, e| {
            const m = start[e + 1] - start[e];
            if (m == 0) continue;
            try d.gemms.run(d.s, p.s(.ae) + start[e] * E * 2, ex.down.dev, p.s(.de) + start[e] * H * 2, m, H, E);
        }
        try p.run("combine", .{ p.r, ks.blocks(H), 1 }, 256, .{ p.s(.de), p.s(.slot), p.s(.wts), p.s(.shared), p.s(.mlp) });
    }
};

/// Rows s0 .. s0 + r into the caches and the last row's logits: a prompt pass eagerly and drained, a step as a graph.
pub fn forward(d: *Decoder, s0: usize, r: usize, prompt: bool) !void {
    d.last = r;
    if (prompt) {
        try launches(d, s0, r, true);
        return d.s.synchronize();
    }
    // a step's launches differ only in their positions, and in block_keys when the step fills a block
    const slot = @intFromBool((s0 + 1) / cfg.block > s0 / cfg.block);
    try cuda.graph.beginCapture(d.s, .thread_local);
    launches(d, s0, 1, false) catch |e| {
        if (cuda.graph.endCapture(d.s)) |g| {
            var gg = g;
            gg.deinit();
        } else |_| {}
        return e;
    };
    var g = try cuda.graph.endCapture(d.s);
    defer g.deinit();
    if (d.graphs[slot]) |*e| {
        if (try e.update(g) != .success) {
            e.deinit();
            d.graphs[slot] = null;
        }
    }
    if (d.graphs[slot] == null) d.graphs[slot] = try g.instantiate();
    try d.graphs[slot].?.launchOn(d.s);
}

/// The forward's launches, no host copies (a step's are captured).
fn launches(d: *Decoder, s0: usize, r: usize, prompt: bool) !void {
    const step = d.at(.step);
    const p: Pass = if (prompt)
        .{ .d = d, .s0 = s0, .r = r, .prompt = true, .cos = d.at(.cos), .sin = d.at(.sin), .qmask = d.at(.qmask) }
    else
        .{ .d = d, .s0 = s0, .r = r, .prompt = false, .cos = step, .sin = step + @offsetOf(decoder_mod.StepRow, "sin"), .qmask = step + @offsetOf(decoder_mod.StepRow, "qmask") };
    for (0..cfg.layers) |l| {
        const layer = d.w.layers[l];
        // the previous layer's MLP rows join the residual first
        try p.norm(l > 0, .mlp, layer.in_norm.dev);
        try p.attention(l);
        try p.norm(true, .o, layer.post_norm.dev);
        try p.ffn(l);
    }
    try p.norm(true, .mlp, d.w.norm.dev);
    try p.matvec(p.s(.y) + (r - 1) * H * 2, d.w.head.dev, p.s(.logits), cfg.vocab, H);
}
