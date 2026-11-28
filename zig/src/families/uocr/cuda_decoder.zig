//! Unlimited-OCR-2's decoder on CUDA: prompt passes then a token a step, caches in bf16 as the Metal decoder's.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const wts = @import("cuda_weights.zig");
const ks = @import("cuda_kernels.zig");
const prompt_mod = @import("prompt.zig");
const pass_mod = @import("cuda_decoder_pass.zig");
const gemm_mod = @import("cuda_gemm.zig");

const H = cfg.hidden;
const D = cfg.router_dim;
const E = cfg.expert_ffn;

/// A layer's caches: keys and values [cap, 1280], router keys [cap, 128] bf16; block summaries [cap / 32, 128] f32.
pub const Cache = struct { k: u64, v: u64, rk: u64, kb: u64 };

/// A forward's working rows, each part 256-aligned in one device buffer.
pub const Scratch = struct {
    pub const Part = enum { ids, img, dest, x, y, proj, rproj, q, att, o, ffn, act, mlp, shared, eids, wts, xe, ge, ue, ae, de, gidx, slot, ident, rq, sel, bias, cos, sin, qmask, logits, pick, step };
    buf: cuda.DeviceBuffer,
    offs: [std.enums.values(Part).len]usize,

    fn bytes(part: Part, r: usize) usize {
        return switch (part) {
            .ids, .dest, .qmask => r * 4,
            .img, .x, .y, .q, .att, .o, .mlp, .shared => r * H * 2,
            .proj => r * 3 * H * 2,
            .rproj => r * 2 * D * 2,
            .ffn => r * 2 * cfg.dense_ffn * 2,
            .act => r * cfg.dense_ffn * 2,
            .eids, .gidx, .slot, .ident, .wts => r * cfg.top_k * 4,
            .xe, .de => r * cfg.top_k * H * 2,
            .ge => r * cfg.top_k * 2 * E * 2,
            .ue, .ae => r * cfg.top_k * E * 2,
            .rq => r * D * 2,
            .sel, .bias => r * cfg.read_blocks * 4,
            .cos, .sin => r * cfg.rope_pairs * 2,
            .logits => cfg.vocab * 2,
            .pick => 4,
            .step => @sizeOf(StepRow),
        };
    }

    fn make(d: *const cuda.Driver, r: usize) !Scratch {
        var s: Scratch = undefined;
        var total: usize = 0;
        for (std.enums.values(Part), 0..) |p, i| {
            s.offs[i] = total;
            total += std.mem.alignForward(usize, bytes(p, r), 256);
        }
        s.buf = try cuda.DeviceBuffer.alloc(d, total);
        return s;
    }

    pub fn at(s: Scratch, p: Part) u64 {
        return s.buf.ptr + s.offs[@backingInt(p)];
    }
};

/// What a pass writes from the host each time: token ids, rotary rows, candidate rows and the experts' groups.
pub const Host = struct {
    ids: []u32,
    dest: []u32,
    cos: []u16,
    sin: []u16,
    qmask: []i32,
    eids: []u32,
    gidx: []u32,
    slot: []u32,

    fn init(a: std.mem.Allocator, r: usize) !Host {
        var h: Host = .{ .ids = &.{}, .dest = &.{}, .cos = &.{}, .sin = &.{}, .qmask = &.{}, .eids = &.{}, .gidx = &.{}, .slot = &.{} };
        errdefer h.deinit(a);
        h.ids = try a.alloc(u32, r);
        h.dest = try a.alloc(u32, r);
        h.qmask = try a.alloc(i32, r);
        h.cos = try a.alloc(u16, r * cfg.rope_pairs);
        h.sin = try a.alloc(u16, r * cfg.rope_pairs);
        h.eids = try a.alloc(u32, r * cfg.top_k);
        h.gidx = try a.alloc(u32, r * cfg.top_k);
        h.slot = try a.alloc(u32, r * cfg.top_k);
        return h;
    }

    fn deinit(h: *Host, a: std.mem.Allocator) void {
        a.free(h.ids);
        a.free(h.dest);
        a.free(h.qmask);
        a.free(h.cos);
        a.free(h.sin);
        a.free(h.eids);
        a.free(h.gidx);
        a.free(h.slot);
    }
};

/// A decode step's coordinates as one upload, in pinned memory: rotary values and candidate row; the pick comes back.
pub const StepRow = extern struct {
    cos: [cfg.rope_pairs]u16,
    sin: [cfg.rope_pairs]u16,
    qmask: i32,
    pick: u32,

    /// The bytes a step uploads: everything before the pick.
    pub const upload = @offsetOf(StepRow, "pick");
};

pub const Decoder = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    s: cuda.Stream,
    k: *const ks.Kernels,
    gemms: *gemm_mod.Gemms,
    w: *const wts.Weights,
    /// Positions the caches hold, and positions filled so far.
    cap: usize = 0,
    len: usize = 0,
    /// Prompt rows a prefill pass takes at once, and the rows of the last forward.
    rows: usize = 0,
    last: usize = 0,
    cache: cuda.DeviceBuffer = undefined,
    caches: [cfg.layers]Cache = undefined,
    /// The routed experts' addresses, [layer][expert] of (gate, up, down), as tf_uocr_expert_gu reads them.
    experts: cuda.DeviceBuffer = undefined,
    scratch: Scratch = undefined,
    host: Host = undefined,
    /// The prompt's candidate rows (prompt.zig), grown as prompts need.
    masks: ?cuda.DeviceBuffer = null,
    words: usize = 0,
    /// The pinned step row, and the event after its last upload (the row is rewritten only once that copy is done).
    pinned: cuda.HostBuffer = undefined,
    sent: cuda.Event = undefined,
    /// A step's captured forward: without and with block_keys (a step that fills a block), updated in place each step.
    graphs: [2]?cuda.graph.Exec = .{ null, null },

    /// A decoder for up to `cap` positions, prefilling `rows` prompt rows a pass.
    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, s: cuda.Stream, k: *const ks.Kernels, gemms: *gemm_mod.Gemms, w: *const wts.Weights, cap: usize, rows: usize) !Decoder {
        if (cap == 0 or cap > cfg.max_context or rows == 0) return error.UocrDecoderSize;
        try joined(w);
        var dec: Decoder = .{ .gpa = gpa, .d = d, .s = s, .k = k, .gemms = gemms, .w = w, .cap = cap, .rows = rows };
        dec.cache = try cuda.DeviceBuffer.alloc(d, cacheBytes(cap));
        errdefer dec.cache.free();
        var base = dec.cache.ptr;
        for (&dec.caches) |*c| {
            const p = layerParts(cap);
            c.* = .{ .k = base, .v = base + p[0], .rk = base + p[0] + p[1], .kb = base + p[0] + p[1] + p[2] };
            base += p[0] + p[1] + p[2] + p[3];
        }
        dec.experts = try expertTable(d, w);
        errdefer dec.experts.free();
        dec.scratch = try Scratch.make(d, rows);
        errdefer dec.scratch.buf.free();
        dec.host = try Host.init(gpa, rows);
        errdefer dec.host.deinit(gpa);
        for (dec.host.slot, 0..) |*v, i| v.* = @intCast(i);
        dec.pinned = try cuda.HostBuffer.alloc(d, @sizeOf(StepRow));
        errdefer dec.pinned.free();
        dec.sent = try cuda.Event.init(d, false);
        errdefer dec.sent.deinit();
        try dec.sent.record(s);
        try dec.put(.ident, std.mem.sliceAsBytes(dec.host.slot));
        try s.synchronize();
        return dec;
    }

    pub fn deinit(dec: *Decoder) void {
        dec.s.synchronize() catch {};
        if (dec.masks) |*m| m.free();
        for (&dec.graphs) |*g| if (g.*) |*e| e.deinit();
        dec.sent.deinit();
        dec.pinned.free();
        dec.host.deinit(dec.gpa);
        dec.scratch.buf.free();
        dec.experts.free();
        dec.cache.free();
        dec.* = undefined;
    }

    /// The cache bytes for `cap` positions.
    pub fn cacheBytes(cap: usize) usize {
        const p = layerParts(cap);
        return cfg.layers * (p[0] + p[1] + p[2] + p[3]);
    }

    /// Prefill a prompt: `image_rows` (bf16 [image tokens, 1280]) at its image tokens, the token embeddings elsewhere.
    pub fn prefill(dec: *Decoder, p: prompt_mod.Prompt, image_rows: []const u16) !void {
        return dec.prefillFor(p, image_rows, null);
    }

    /// prefill for a served stream: error.Cancelled before the upload or a chunk once its request is cancelled.
    pub fn prefillFor(dec: *Decoder, p: prompt_mod.Prompt, image_rows: []const u16, stream: ?*const lanes.Stream) !void {
        const n = p.len();
        if (n == 0 or n > dec.cap) return error.UocrPromptTooLong;
        var images: usize = 0;
        for (p.ids) |id| images += @intFromBool(id == cfg.image_token);
        if (image_rows.len != images * H) return error.UocrImageRows;
        for (p.ids) |id| if (id >= cfg.vocab) return error.UocrBadToken;
        if (stream) |st| if (st.isCancelled()) return error.Cancelled;
        try dec.upload(p);
        dec.len = 0;
        var img: usize = 0;
        var s0: usize = 0;
        while (s0 < n) {
            if (stream) |st| if (st.isCancelled()) return error.Cancelled; // dec.len stays at the last whole chunk
            const r = @min(dec.rows, n - s0);
            var m: usize = 0;
            for (0..r) |i| {
                const id = p.ids[s0 + i];
                dec.host.ids[i] = id;
                if (id == cfg.image_token) {
                    dec.host.dest[m] = @intCast(i);
                    m += 1;
                }
                dec.coords(i, .{ .tok = p.tok[s0 + i], .page = p.page[s0 + i], .mask = p.mask[s0 + i] });
            }
            try dec.put(.ids, std.mem.sliceAsBytes(dec.host.ids[0..r]));
            try dec.k.run("embed", dec.s, .{ r, ks.blocks(H), 1 }, 256, .{ dec.at(.ids), dec.w.embed.dev, dec.at(.x), @as(u32, cfg.image_token) });
            if (m > 0) {
                try dec.put(.dest, std.mem.sliceAsBytes(dec.host.dest[0..m]));
                try dec.put(.img, std.mem.sliceAsBytes(image_rows[img * H ..][0 .. m * H]));
                try dec.k.run("scatter", dec.s, .{ m, ks.blocks(H), 1 }, 256, .{ dec.at(.img), dec.at(.dest), dec.at(.x) });
                img += m;
            }
            try dec.putCoords(r);
            try pass_mod.forward(dec, s0, r, true);
            s0 += r;
            dec.len = s0;
        }
    }

    /// One decode step for token `id` at the next position, with its coordinates from prompt.Step.
    pub fn step(dec: *Decoder, id: u32, c: prompt_mod.Coord) !void {
        if (dec.len >= dec.cap) return error.UocrContextFull;
        if (id >= cfg.vocab) return error.UocrBadToken;
        if (dec.masks == null) return error.UocrNoPrompt;
        // the token's embedding row copied as is, then the step's coordinates in one upload from pinned memory
        try dec.check(dec.d.api.cuMemcpyDtoDAsync_v2(dec.at(.x), dec.w.embed.dev + @as(u64, id) * H * 2, H * 2, dec.s.handle));
        try dec.sent.synchronize();
        const row = dec.pinnedRow();
        cfg.rope(c.tok, c.page, &row.cos, &row.sin);
        row.qmask = c.mask;
        try dec.check(dec.d.api.cuMemcpyHtoDAsync_v2(dec.at(.step), row, StepRow.upload, dec.s.handle));
        try dec.sent.record(dec.s);
        try pass_mod.forward(dec, dec.len, 1, false);
        dec.len += 1;
    }

    /// bf16 logits of the last forward's last row, copied to `out` ([vocab]).
    pub fn logits(dec: *Decoder, out: []u16) !void {
        if (dec.len == 0) return error.UocrNothingDecoded;
        if (out.len != cfg.vocab) return error.UocrLogitsSize;
        try dec.get(.logits, std.mem.sliceAsBytes(out));
    }

    /// The greedy pick over the logits on the device (the first maximum; EOS skipped when `bar_eos`).
    pub fn argmax(dec: *Decoder, bar_eos: bool) !u32 {
        if (dec.len == 0) return error.UocrNothingDecoded;
        try dec.k.run("argmax", dec.s, .{ 1, 1, 1 }, 1024, .{ dec.at(.logits), cfg.vocab, @as(c_int, @intFromBool(bar_eos)), dec.at(.pick) });
        const row = dec.pinnedRow();
        try dec.check(dec.d.api.cuMemcpyDtoHAsync_v2(&row.pick, dec.at(.pick), 4, dec.s.handle));
        try dec.s.synchronize();
        return row.pick;
    }

    /// The last forward's rows after the final norm (bf16 [rows, 1280]), for checks.
    pub fn hidden(dec: *Decoder, out: []u16) !void {
        if (dec.last == 0 or out.len != dec.last * H) return error.UocrHiddenSize;
        try dec.get(.y, std.mem.sliceAsBytes(out));
    }

    pub fn at(dec: *const Decoder, p: Scratch.Part) u64 {
        return dec.scratch.at(p);
    }

    fn pinnedRow(dec: *Decoder) *StepRow {
        return @ptrCast(@alignCast(dec.pinned.bytes.ptr));
    }

    fn check(dec: *Decoder, r: anytype) !void {
        try dec.d.check(r, "uocr decoder copy");
    }

    /// Host bytes into a scratch part, in stream order.
    pub fn put(dec: *Decoder, p: Scratch.Part, bytes: []const u8) !void {
        const b: cuda.DeviceBuffer = .{ .d = dec.d, .ptr = dec.at(p), .len = bytes.len, .borrowed = true };
        try b.uploadAsync(0, bytes, dec.s.handle);
    }

    /// A scratch part's first bytes once the stream has drained.
    pub fn get(dec: *Decoder, p: Scratch.Part, out: []u8) !void {
        try dec.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = dec.d, .ptr = dec.at(p), .len = out.len, .borrowed = true };
        try b.download(0, out);
    }

    /// Row i's rotary values and candidate row, on the host.
    fn coords(dec: *Decoder, i: usize, c: prompt_mod.Coord) void {
        const pr = cfg.rope_pairs;
        cfg.rope(c.tok, c.page, dec.host.cos[i * pr ..][0..pr], dec.host.sin[i * pr ..][0..pr]);
        dec.host.qmask[i] = c.mask;
    }

    fn putCoords(dec: *Decoder, r: usize) !void {
        try dec.put(.cos, std.mem.sliceAsBytes(dec.host.cos[0 .. r * cfg.rope_pairs]));
        try dec.put(.sin, std.mem.sliceAsBytes(dec.host.sin[0 .. r * cfg.rope_pairs]));
        try dec.put(.qmask, std.mem.sliceAsBytes(dec.host.qmask[0..r]));
    }

    /// The prompt's candidate rows onto the device.
    fn upload(dec: *Decoder, p: prompt_mod.Prompt) !void {
        const need = @max(p.masks.len * 4, 16);
        try dec.s.synchronize();
        if (dec.masks == null or dec.masks.?.len < need) {
            if (dec.masks) |*m| m.free();
            dec.masks = null;
            dec.masks = try cuda.DeviceBuffer.alloc(dec.d, need);
        }
        try dec.masks.?.uploadAsync(0, std.mem.sliceAsBytes(p.masks), dec.s.handle);
        dec.words = p.words;
    }
};

/// A layer's cache parts' bytes, each 256-aligned: keys, values, router keys, block summaries.
fn layerParts(cap: usize) [4]usize {
    const blocks = std.math.divCeil(usize, cap, cfg.block) catch unreachable;
    const al = struct {
        fn f(n: usize) usize {
            return std.mem.alignForward(usize, n, 256);
        }
    }.f;
    return .{ al(cap * H * 2), al(cap * H * 2), al(cap * D * 2), al(blocks * D * 4) };
}

/// A layer's q | k | v, router q | k and FFN gate | up rows sit back to back as loaded: one product a group.
fn joined(w: *const wts.Weights) !void {
    for (w.layers, 0..) |layer, l| {
        const f: usize = if (l == 0) cfg.dense_ffn else cfg.shared_ffn;
        const pairs = [_][3]u64{
            .{ layer.q.dev, layer.k.dev, H * H * 2 },
            .{ layer.k.dev, layer.v.dev, H * H * 2 },
            .{ layer.router.q.dev, layer.router.k.dev, D * H * 2 },
            .{ layer.ffn.gate.dev, layer.ffn.up.dev, f * H * 2 },
        };
        for (pairs) |p| if (p[0] + p[2] != p[1]) return error.UocrWeightLayout;
    }
}

/// The routed experts' (gate, up, down) device addresses, [layer][expert] (layer 0's left zero).
fn expertTable(d: *const cuda.Driver, w: *const wts.Weights) !cuda.DeviceBuffer {
    var t: [cfg.layers * cfg.experts * 3]u64 = @splat(0);
    for (w.layers[1..], 1..) |layer, l| for (layer.experts, 0..) |e, j| {
        for ([_]wts.T{ e.gate, e.up, e.down }, 0..) |r, i| t[(l * cfg.experts + j) * 3 + i] = r.dev;
    };
    return cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(&t));
}

test "uocr cuda: cache bytes a position" {
    // keys and values 2 x 2,560 bytes, router keys 256 and a block summary's 512 / 32: 5,392 a layer
    try std.testing.expectEqual(@as(usize, 12 * 5392 * 4096), Decoder.cacheBytes(4096));
}
