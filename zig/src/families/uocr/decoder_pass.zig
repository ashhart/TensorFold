//! One forward of rows (a prompt chunk or a decode step) through the 12 layers; a prompt's experts group on the CPU.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ks = @import("kernels.zig");
const decoder_mod = @import("decoder.zig");
const Ref = ks.Ref;
const Decoder = decoder_mod.Decoder;
const Part = decoder_mod.Scratch.Part;

const H = cfg.hidden;
const E = cfg.expert_ffn;

/// The width of a layer's joint projection: q, k and v (1280 each), then the router's q and k (128 each).
pub const proj_width = 3 * H + 2 * cfg.router_dim;

const UdMv = extern struct { n: u32, k: u32 };
const UdNorm = extern struct { add: u32, eps: f32 };
const UdQk = extern struct { pos0: u32, eps: f32 };
const UdRoute = extern struct { pos0: u32, words: u32 };
const UdAttend = extern struct { pos0: u32, bf16_bias: u32 };
const UdAct = extern struct { width: u32, gstride: u32, ustride: u32 };
const UdExp = extern struct { width: u32 };

/// A prompt's per-layer rows kept for checks: [attention, MLP, layer output][layer][row][1280] bf16.
pub const Trace = struct {
    buf: mtl.Buffer,
    n: usize,

    pub fn init(device: mtl.Device, n: usize) !Trace {
        return .{ .buf = try device.buffer(3 * cfg.layers * n * H * 2, mtl.ResourceOptions.shared), .n = n };
    }

    pub fn deinit(t: *Trace) void {
        t.buf.deinit();
    }

    fn at(t: Trace, part: usize, l: usize, row: usize) Ref {
        return .{ .buf = t.buf, .off = ((part * cfg.layers + l) * t.n + row) * H * 2 };
    }

    pub fn copyOut(t: Trace, layers: ?[][]u16, parts: ?[2][][]u16, n: usize) void {
        const all: [*]const u16 = @ptrCast(@alignCast(t.buf.contents()));
        for (0..cfg.layers) |l| {
            if (layers) |ls| @memcpy(ls[l][0 .. n * H], all[(2 * cfg.layers + l) * t.n * H ..][0 .. n * H]);
            if (parts) |ps| for (0..2) |p| @memcpy(ps[p][l][0 .. n * H], all[(p * cfg.layers + l) * t.n * H ..][0 .. n * H]);
        }
    }
};

const Pass = struct {
    d: *Decoder,
    s0: usize,
    r: usize,
    prompt: bool,
    tr: ?*Trace,
    cb: mtl.CommandBuffer = undefined,
    enc: mtl.ComputeEncoder = undefined,

    fn open(p: *Pass) void {
        p.cb = p.d.queue.commandBuffer();
        p.enc = p.cb.compute(.concurrent);
        for (p.d.held[0..p.d.nheld]) |b| p.enc.useResource(b, mtl.ResourceUsage.read);
    }

    /// Run what is encoded and wait for it.
    fn flush(p: *Pass) !void {
        p.enc.end();
        p.cb.commit();
        p.cb.wait();
        if (p.cb.failure()) |msg| {
            std.log.err("uocr decoder: {s}", .{msg});
            return error.UocrGpu;
        }
    }

    fn s(p: *Pass, part: Part) Ref {
        return p.d.scratch.at(part);
    }

    fn bind(p: *Pass, comptime name: []const u8, refs: []const Ref, params: anytype) void {
        ks.Kernels.bind(p.enc, p.d.k.get(name), refs, params);
    }

    fn groups(p: *Pass, n: usize, threads: usize) void {
        p.enc.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(threads, 1, 1));
        p.enc.barrier();
    }

    fn grid(p: *Pass, x: usize, y: usize) void {
        p.enc.dispatchThreads(mtl.Size.of(x, y, 1), mtl.Size.of(256, 1, 1));
        p.enc.barrier();
    }

    /// y [r, n] = x [r, k] w^T: the tiled product for a prompt's rows, the one-row product for a step.
    fn mm(p: *Pass, x: Ref, w: Ref, y: Ref, n: usize, k: usize) void {
        if (p.prompt) {
            p.d.k.linear(p.enc, x, w, y, p.r, n, k);
            p.enc.barrier();
        } else {
            p.bind("ud_matvec", &.{ x, w, y }, UdMv{ .n = @intCast(n), .k = @intCast(k) });
            p.groups((n + 31) / 32, 256);
        }
    }

    fn matvec(p: *Pass, x: Ref, w: Ref, y: Ref, n: usize, k: usize) void {
        p.bind("ud_matvec", &.{ x, w, y }, UdMv{ .n = @intCast(n), .k = @intCast(k) });
        p.groups((n + 31) / 32, 256);
    }

    /// With `add`: x = bf16(x + a); then y = the norm of x under `w`.
    fn norm(p: *Pass, add: bool, a: Part, w: Ref) void {
        p.bind("ud_norm", &.{ p.s(.x), p.s(a), w, p.s(.y) }, UdNorm{ .add = @intFromBool(add), .eps = cfg.eps });
        p.groups(p.r, 256);
    }

    fn silu(p: *Pass, g: Ref, u: Ref, y: Ref, rows: usize, width: usize, stride: usize) void {
        p.bind("ud_silu_mul", &.{ g, u, y }, UdAct{ .width = @intCast(width), .gstride = @intCast(stride), .ustride = @intCast(stride) });
        p.grid(width, rows);
    }

    /// The prompt's rows `part` into the trace at `which` (0 attention, 1 MLP, 2 layer output).
    fn keep(p: *Pass, part: Part, which: usize, l: usize) void {
        const t = p.tr orelse return;
        p.bind("ud_gather", &.{ p.s(part), p.s(.ident), t.at(which, l, p.s0) }, {});
        p.grid(H, p.r);
    }

    fn attention(p: *Pass, l: usize) void {
        const d = p.d;
        const layer = d.w.layers[l];
        const c = d.caches[l];
        const pos0: u32 = @intCast(p.s0);
        p.mm(p.s(.y), d.attn_cat[l], p.s(.proj), proj_width, H);
        const cache = struct {
            fn ref(b: mtl.Buffer) Ref {
                return .{ .buf = b };
            }
        }.ref;
        p.bind("ud_qk", &.{ p.s(.proj), p.s(.cos), p.s(.sin), layer.router.q_norm, layer.router.k_norm, p.s(.q), cache(c.k), cache(c.v), p.s(.rq), cache(c.rk) }, UdQk{ .pos0 = pos0, .eps = cfg.eps });
        p.groups(p.r, cfg.router_dim);
        const c0 = p.s0 / cfg.block;
        const c1 = (p.s0 + p.r) / cfg.block;
        if (c1 > c0) {
            p.bind("ud_block_keys", &.{ cache(c.rk), layer.router.head_cls, cache(c.kb) }, @as(u32, @intCast(c0)));
            p.groups(c1 - c0, 32);
        }
        const masks: Ref = .{ .buf = d.masks.? };
        p.bind("ud_route", &.{ p.s(.rq), cache(c.kb), masks, p.s(.qmask), p.s(.sel), p.s(.bias) }, UdRoute{ .pos0 = pos0, .words = @intCast(d.words) });
        p.groups(p.r, 256);
        p.bind("ud_attend", &.{ p.s(.q), cache(c.k), cache(c.v), p.s(.sel), p.s(.bias), p.s(.att) }, UdAttend{ .pos0 = pos0, .bf16_bias = @intFromBool(p.prompt) });
        p.enc.dispatchGroups(mtl.Size.of(cfg.heads, p.r, 1), mtl.Size.of(256, 1, 1));
        p.enc.barrier();
        p.mm(p.s(.att), layer.o, p.s(.o), H, H);
    }

    /// The layer's MLP rows from y into .mlp: the dense FFN, or the shared expert plus the 6 routed ones.
    fn ffn(p: *Pass, l: usize) !void {
        const d = p.d;
        const layer = d.w.layers[l];
        const width: usize = if (l == 0) cfg.dense_ffn else cfg.shared_ffn;
        const f = p.s(.ffn);
        p.mm(p.s(.y), d.ffn_cat[l], f, 2 * width, H);
        p.silu(f, f.at(width * 2), p.s(.act), p.r, width, 2 * width);
        p.mm(p.s(.act), layer.ffn.down, p.s(if (l == 0) .mlp else .shared), H, width);
        if (l == 0) return;
        p.bind("ud_moe_gate", &.{ p.s(.y), layer.gate.?, p.s(.ids), p.s(.wts) }, {});
        p.groups(p.r, 1024);
        if (p.prompt) return p.experts(l);
        const table: Ref = .{ .buf = d.experts, .off = l * cfg.experts * 3 * 8 };
        p.bind("ud_expert_gu", &.{ p.s(.y), table, p.s(.ids), p.s(.ge) }, UdExp{ .width = E });
        p.groups(cfg.top_k * 2 * E / 32, 256);
        p.silu(p.s(.ge), p.s(.ge).at(E * 2), p.s(.ae), cfg.top_k, E, 2 * E);
        p.bind("ud_expert_down", &.{ p.s(.ae), table, p.s(.ids), p.s(.de) }, UdExp{ .width = E });
        p.groups(cfg.top_k * H / 32, 256);
        p.bind("ud_combine", &.{ p.s(.de), p.s(.ident), p.s(.wts), p.s(.shared), p.s(.mlp) }, {});
        p.grid(H, 1);
    }

    /// A prompt's routed experts: the gate's picks read back, rows grouped by expert, one product an expert.
    fn experts(p: *Pass, l: usize) !void {
        const d = p.d;
        try p.flush();
        p.open();
        const n = p.r * cfg.top_k;
        const ids = d.scratch.host(u32, .ids, n);
        var start: [cfg.experts + 1]usize = @splat(0);
        for (ids) |e| start[e + 1] += 1;
        for (1..start.len) |e| start[e] += start[e - 1];
        var fill = start;
        const gidx = d.scratch.host(u32, .gidx, n);
        const slot = d.scratch.host(u32, .slot, n);
        for (ids, 0..) |e, i| {
            gidx[fill[e]] = @intCast(i / cfg.top_k);
            slot[i] = @intCast(fill[e]);
            fill[e] += 1;
        }
        p.bind("ud_gather", &.{ p.s(.y), p.s(.gidx), p.s(.xe) }, {});
        p.grid(H, n);
        const layer = d.w.layers[l];
        for (layer.experts, 0..) |ex, e| {
            const m = start[e + 1] - start[e];
            if (m == 0) continue;
            const at = start[e];
            d.k.linear(p.enc, p.s(.xe).at(at * H * 2), ex.gate, p.s(.ge).at(at * E * 2), m, E, H);
            d.k.linear(p.enc, p.s(.xe).at(at * H * 2), ex.up, p.s(.ue).at(at * E * 2), m, E, H);
        }
        p.enc.barrier();
        p.silu(p.s(.ge), p.s(.ue), p.s(.ae), n, E, E);
        for (layer.experts, 0..) |ex, e| {
            const m = start[e + 1] - start[e];
            if (m == 0) continue;
            d.k.linear(p.enc, p.s(.ae).at(start[e] * E * 2), ex.down, p.s(.de).at(start[e] * H * 2), m, H, E);
        }
        p.enc.barrier();
        p.bind("ud_combine", &.{ p.s(.de), p.s(.slot), p.s(.wts), p.s(.shared), p.s(.mlp) }, {});
        p.grid(H, p.r);
    }
};

/// Rows s0 .. s0 + r through every layer (inputs in the scratch): the caches take their keys, the last row its logits.
pub fn forward(d: *Decoder, s0: usize, r: usize, prompt: bool, tr: ?*Trace) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var p: Pass = .{ .d = d, .s0 = s0, .r = r, .prompt = prompt, .tr = tr };
    p.open();
    for (0..cfg.layers) |l| {
        const layer = d.w.layers[l];
        // the previous layer's MLP rows join the residual first
        p.norm(l > 0, .mlp, layer.in_norm);
        if (l > 0) p.keep(.x, 2, l - 1);
        p.attention(l);
        p.keep(.o, 0, l);
        p.norm(true, .o, layer.post_norm);
        try p.ffn(l);
        p.keep(.mlp, 1, l);
    }
    p.norm(true, .mlp, d.w.norm);
    p.keep(.x, 2, cfg.layers - 1);
    p.matvec(p.s(.y).at((r - 1) * H * 2), d.w.head, p.s(.logits), cfg.vocab, H);
    try p.flush();
}
