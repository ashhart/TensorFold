//! Unlimited-OCR-2's decoder on Metal: the prompt in chunks of rows, then a token a step, over exact bf16 caches.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const ks = @import("kernels.zig");
const wts = @import("weights.zig");
const prompt_mod = @import("prompt.zig");
const pass_mod = @import("decoder_pass.zig");
const Ref = ks.Ref;

const H = cfg.hidden;
const D = cfg.router_dim;

/// A layer's caches: bf16 keys, values [cap, 1280] and router keys [cap, 128]; f32 block summaries [cap / 32, 128].
pub const Cache = struct { k: mtl.Buffer, v: mtl.Buffer, rk: mtl.Buffer, kb: mtl.Buffer };

/// A forward's working rows, each part 256-aligned in one buffer.
pub const Scratch = struct {
    pub const Part = enum { x, y, proj, q, att, o, ffn, act, mlp, shared, ids, wts, xe, ge, ue, ae, de, gidx, slot, ident, rq, sel, bias, cos, sin, qmask, logits };
    buf: mtl.Buffer,
    offs: [std.enums.values(Part).len]usize,

    fn bytes(part: Part, r: usize) usize {
        const e = cfg.expert_ffn;
        return switch (part) {
            .x, .y, .q, .att, .o, .mlp, .shared => r * H * 2,
            .proj => r * pass_mod.proj_width * 2,
            .ffn => r * 2 * cfg.dense_ffn * 2,
            .act => r * cfg.dense_ffn * 2,
            .ids, .gidx, .slot, .ident => r * cfg.top_k * 4,
            .wts => r * cfg.top_k * 4,
            .xe, .de => r * cfg.top_k * H * 2,
            .ge, .ue, .ae => r * cfg.top_k * 2 * e * 2,
            .rq => r * D * 2,
            .sel, .bias => r * cfg.read_blocks * 4,
            .cos, .sin => r * cfg.rope_pairs * 2,
            .qmask => r * 4,
            .logits => cfg.vocab * 2,
        };
    }

    fn make(device: mtl.Device, r: usize) !Scratch {
        var s: Scratch = undefined;
        var total: usize = 0;
        for (std.enums.values(Part), 0..) |p, i| {
            s.offs[i] = total;
            total += std.mem.alignForward(usize, bytes(p, r), 256);
        }
        s.buf = try device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        return s;
    }

    pub fn at(s: Scratch, p: Part) Ref {
        return .{ .buf = s.buf, .off = s.offs[@backingInt(p)] };
    }

    pub fn host(s: Scratch, comptime T: type, p: Part, n: usize) []T {
        return @as([*]T, @ptrCast(@alignCast(s.at(p).addr())))[0..n];
    }
};

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    k: *const ks.Kernels,
    w: *const wts.Weights,
    /// Positions the caches hold, and positions filled so far.
    cap: usize = 0,
    len: usize = 0,
    /// Prompt rows a prefill pass takes at once.
    rows: usize = 0,
    /// For checks: each layer's output rows during a prefill, and each layer's attention and MLP outputs.
    trace: ?[][]u16 = null,
    trace_parts: ?[2][][]u16 = null,
    caches: [cfg.layers]Cache = undefined,
    /// The projections a pass reads as one: [q | k | v | router q | router k] and [gate | up] a layer.
    cat: mtl.Buffer = undefined,
    attn_cat: [cfg.layers]Ref = undefined,
    ffn_cat: [cfg.layers]Ref = undefined,
    /// The routed experts' GPU addresses, [layer][expert] of (gate, up, down), and the buffers they point into.
    experts: mtl.Buffer = undefined,
    held: [8]mtl.Buffer = undefined,
    nheld: usize = 0,
    scratch: Scratch = undefined,
    /// The prompt's candidate rows (prompt.zig), grown as prompts need.
    masks: ?mtl.Buffer = null,
    mask_bytes: usize = 0,
    words: usize = 0,

    /// A decoder for up to `cap` positions, prefilling `rows` prompt rows a pass.
    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, k: *const ks.Kernels, w: *const wts.Weights, cap: usize, rows: usize) !Decoder {
        if (cap == 0 or cap > cfg.max_context or rows == 0) return error.UocrDecoderSize;
        var d: Decoder = .{ .gpa = gpa, .device = device, .queue = queue, .k = k, .w = w, .cap = cap, .rows = rows };
        const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
        const blocks = std.math.divCeil(usize, cap, cfg.block) catch unreachable;
        var made: usize = 0;
        errdefer for (d.caches[0..made]) |c| inline for (.{ "k", "v", "rk", "kb" }) |f| @field(c, f).deinit();
        for (&d.caches) |*c| {
            c.k = try device.buffer(cap * H * 2, opts);
            errdefer c.k.deinit();
            c.v = try device.buffer(cap * H * 2, opts);
            errdefer c.v.deinit();
            c.rk = try device.buffer(cap * D * 2, opts);
            errdefer c.rk.deinit();
            c.kb = try device.buffer(blocks * D * 4, opts);
            made += 1;
        }
        try d.concat();
        errdefer d.cat.deinit();
        d.experts = try expertTable(device, w);
        errdefer d.experts.deinit();
        for (w.layers[1..]) |layer| for (layer.experts) |e| for ([_]Ref{ e.gate, e.up, e.down }) |r| {
            for (d.held[0..d.nheld]) |b| {
                if (b.id == r.buf.id) break;
            } else {
                if (d.nheld == d.held.len) return error.UocrExpertBuffers;
                d.held[d.nheld] = r.buf;
                d.nheld += 1;
            }
        };
        d.scratch = try Scratch.make(device, @max(rows, 1));
        errdefer d.scratch.buf.deinit();
        for (d.scratch.host(u32, .ident, rows * cfg.top_k), 0..) |*v, i| v.* = @intCast(i);
        return d;
    }

    pub fn deinit(d: *Decoder) void {
        for (d.caches) |c| inline for (.{ "k", "v", "rk", "kb" }) |f| @field(c, f).deinit();
        d.cat.deinit();
        d.experts.deinit();
        d.scratch.buf.deinit();
        if (d.masks) |m| m.deinit();
        d.* = undefined;
    }

    /// Each layer's projections copied side by side, so one product makes all of a step's.
    fn concat(d: *Decoder) !void {
        const w = d.w;
        var total: usize = 0;
        for (0..cfg.layers) |l| total += pass_mod.proj_width * H * 2 + ffnRows(l) * H * 2;
        d.cat = try d.device.buffer(total, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        var at: usize = 0;
        for (w.layers, 0..) |layer, l| {
            d.attn_cat[l] = .{ .buf = d.cat, .off = at };
            const r = layer.router;
            for ([_]struct { Ref, usize }{ .{ layer.q, H }, .{ layer.k, H }, .{ layer.v, H }, .{ r.q, D }, .{ r.k, D } }) |part| {
                @memcpy((d.cat.contents() + at)[0 .. part[1] * H * 2], part[0].addr()[0 .. part[1] * H * 2]);
                at += part[1] * H * 2;
            }
            d.ffn_cat[l] = .{ .buf = d.cat, .off = at };
            const half = ffnRows(l) / 2;
            for ([_]Ref{ layer.ffn.gate, layer.ffn.up }) |part| {
                @memcpy((d.cat.contents() + at)[0 .. half * H * 2], part.addr()[0 .. half * H * 2]);
                at += half * H * 2;
            }
        }
    }

    /// Prefill a prompt: `image_rows` (bf16 [image tokens, 1280]) at its image tokens, the token embeddings elsewhere.
    pub fn prefill(d: *Decoder, p: prompt_mod.Prompt, image_rows: []const u16) !void {
        const n = p.len();
        if (n == 0 or n > d.cap) return error.UocrPromptTooLong;
        var images: usize = 0;
        for (p.ids) |id| images += @intFromBool(id == cfg.image_token);
        if (image_rows.len != images * H) return error.UocrImageRows;
        try d.upload(p);
        var tr: ?pass_mod.Trace = null;
        if (d.trace != null or d.trace_parts != null) tr = try pass_mod.Trace.init(d.device, n);
        defer if (tr) |*t| t.deinit();
        d.len = 0;
        var img: usize = 0;
        var s0: usize = 0;
        const embed: [*]const u16 = @ptrCast(@alignCast(d.w.embed.addr()));
        while (s0 < n) {
            const r = @min(d.rows, n - s0);
            const x = d.scratch.host(u16, .x, r * H);
            for (0..r) |i| {
                const id = p.ids[s0 + i];
                if (id == cfg.image_token) {
                    @memcpy(x[i * H ..][0..H], image_rows[img * H ..][0..H]);
                    img += 1;
                } else @memcpy(x[i * H ..][0..H], embed[@as(usize, id) * H ..][0..H]);
                d.coords(i, .{ .tok = p.tok[s0 + i], .page = p.page[s0 + i], .mask = p.mask[s0 + i] });
            }
            try pass_mod.forward(d, s0, r, true, if (tr) |*t| t else null);
            s0 += r;
            d.len = s0;
        }
        if (tr) |t| t.copyOut(d.trace, d.trace_parts, n);
    }

    /// One decode step for token `id` at the next position, with its coordinates from prompt.Step.
    pub fn step(d: *Decoder, id: u32, c: prompt_mod.Coord) !void {
        if (d.len >= d.cap) return error.UocrContextFull;
        if (id >= cfg.vocab) return error.UocrBadToken;
        const embed: [*]const u16 = @ptrCast(@alignCast(d.w.embed.addr()));
        @memcpy(d.scratch.host(u16, .x, H), embed[@as(usize, id) * H ..][0..H]);
        d.coords(0, c);
        try pass_mod.forward(d, d.len, 1, false, null);
        d.len += 1;
    }

    /// Row i's rotary values and candidate row.
    fn coords(d: *Decoder, i: usize, c: prompt_mod.Coord) void {
        const cos = d.scratch.host(u16, .cos, (i + 1) * cfg.rope_pairs);
        const sin = d.scratch.host(u16, .sin, (i + 1) * cfg.rope_pairs);
        cfg.rope(c.tok, c.page, cos[i * cfg.rope_pairs ..][0..cfg.rope_pairs], sin[i * cfg.rope_pairs ..][0..cfg.rope_pairs]);
        d.scratch.host(i32, .qmask, i + 1)[i] = c.mask;
    }

    /// The prompt's candidate rows into the masks buffer.
    fn upload(d: *Decoder, p: prompt_mod.Prompt) !void {
        const need = @max(p.masks.len * 4, 16);
        if (d.masks == null or d.mask_bytes < need) {
            if (d.masks) |m| m.deinit();
            d.masks = null;
            d.masks = try d.device.buffer(need, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            d.mask_bytes = need;
        }
        @memcpy(d.masks.?.contents()[0 .. p.masks.len * 4], std.mem.sliceAsBytes(p.masks));
        d.words = p.words;
    }

    /// The last row's logits, bf16 [vocab].
    pub fn logits(d: *Decoder) ![]const u16 {
        if (d.len == 0) return error.UocrNothingDecoded;
        return d.scratch.host(u16, .logits, cfg.vocab);
    }
};

/// Rows of a layer's gate | up: the dense layer's 2 x 6,848, a routed layer's shared expert's 2 x 1,792.
pub fn ffnRows(l: usize) usize {
    return 2 * @as(usize, if (l == 0) cfg.dense_ffn else cfg.shared_ffn);
}

/// The routed experts' (gate, up, down) GPU addresses, [layer][expert], as decoder.metal's UdExpert reads them.
fn expertTable(device: mtl.Device, w: *const wts.Weights) !mtl.Buffer {
    const buf = try device.buffer(cfg.layers * cfg.experts * 3 * 8, mtl.ResourceOptions.shared);
    const t: [*]u64 = @ptrCast(@alignCast(buf.contents()));
    @memset(t[0 .. cfg.layers * cfg.experts * 3], 0);
    for (w.layers[1..], 1..) |layer, l| for (layer.experts, 0..) |e, j| {
        for ([_]Ref{ e.gate, e.up, e.down }, 0..) |r, i| t[(l * cfg.experts + j) * 3 + i] = r.buf.gpuAddress() + r.off;
    };
    return buf;
}
