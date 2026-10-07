//! One prompt, drafts off: each token is one fused decode row, layers in order, then the argmax.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const qmm = @import("cuda_qmm.zig");
const gdn = @import("cuda_gdn.zig");
const ple = @import("cuda_ple.zig");
const weights = @import("cuda_prompt_wt.zig");
const step = @import("cuda_prompt_step.zig");
const load = @import("cuda_prompt_load.zig");

const layers_n = weights.layers_n;
const linear_n = weights.linear_n;
const attn_n = weights.attn_n;
const dims = weights.dims;
const streams = weights.streams;
const wide = weights.wide;
const vocab = weights.vocab;
const experts_n = weights.experts_n;
const slots = weights.slots;
const moe_w = weights.moe_w;
const head_dim = weights.head_dim;
const index_dim = weights.index_dim;
const proj_n = weights.proj_n;
const out_k = weights.out_k;
const ple_layer = weights.ple_layer;
const inj_stride = weights.inj_stride;
const wts_stride = weights.wts_stride;
const ple_tail_n = weights.ple_tail_n;
const state_n = weights.state_n;
const kv_heads = weights.kv_heads;
const Store = weights.Store;
const Face = weights.Face;
const Q4 = weights.Q4;
const HcW = weights.HcW;
const linearLayer = weights.linearLayer;
const linearIndex = weights.linearIndex;
const attnIndex = weights.attnIndex;
const promote = weights.promote;
const asF32 = weights.asF32;
const uploadQ4 = weights.uploadQ4;
const join3 = weights.join3;
const loadHc = weights.loadHc;
const stitch = weights.stitch;
const Scratch = step.Scratch;
const writeback = step.writeback;
const readout = step.readout;
const gdnStep = step.gdnStep;
const attnStep = step.attnStep;
const moeStep = step.moeStep;
const pleStep = step.pleStep;
const Layer = load.Layer;
const loadLayer = load.loadLayer;

pub fn argmax(raw: []const u8) u32 {
    var best = promote(std.mem.readInt(u16, raw[0..2], .little));
    var at: usize = 0;
    var i: usize = 1;
    while (i < vocab) : (i += 1) {
        const v = promote(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
        if (v > best) {
            best = v;
            at = i;
        }
    }
    return @intCast(at);
}

const Pass = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: *cuda.Stream,
    store: *Store,
    model_dir: []const u8,
    s: *Scratch,
    table: embed.Table,
    rows: usize,
    mode: []u8,
    history: *[2]i64,
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    mixer: HcW,
    head_q: Q4,
    logits: []u8,
    plan_fn: cuda.Function,
    up_fn: cuda.Function,
    down_fn: cuda.Function,
};

fn embedRow(comptime Tri: type, p: *Pass, tri: Tri, token: u32, h: u64) !void {
    const id: i32 = @intCast(token);
    try p.s.ids.upload(0, std.mem.asBytes(&id));
    try tri.embed(p.s.ids.ptr, p.table.w.ptr, p.table.s.ptr, p.table.b.ptr, h, dims, streams, 1);
}

fn runLayer(comptime Tri: type, p: *Pass, tri: Tri, lw: *Layer, layer: usize, ids: []const u32, pos0: usize) !void {
    for (ids, 0..) |token, t| {
        const h = p.s.h.ptr + @as(u64, t) * wide * 2;
        const inj = p.s.inj.ptr + @as(u64, t) * inj_stride;
        const y = p.s.y.ptr + @as(u64, t) * slots * dims * 4;
        const wt = p.s.wts.ptr + @as(u64, t) * wts_stride;
        if (layer == ple_layer and p.mode[t] == 2) {
            try writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        }
        if (layer == ple_layer) try pleStep(Tri, p.gpa, p.io, p.driver, p.stream, tri, p.s, p.model_dir, p.history, token, h, lw.key, lw.val, lw.nk.ptr, lw.nq.ptr, lw.nc.ptr, lw.ple_cw.ptr, &p.tail);
        if (p.mode[t] == 2) {
            try writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        } else try writeback(tri, p.s, h, 0, inj, y, wt);
        try readout(tri, p.s, lw.attn_hc, h, inj);
        if (lw.linear) {
            const li = linearIndex(layer);
            try gdnStep(p.gpa, p.driver, p.stream, p.s, lw.proj, lw.out, p.conv[li], lw.cw.ptr, lw.a_log.ptr, lw.dt.ptr, lw.norm.ptr, &p.state[li]);
        } else {
            const ai = attnIndex(layer);
            try attnStep(Tri, p.driver, p.stream, tri, p.s, lw.proj, lw.out, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, p.kc[ai].ptr, p.vc[ai].ptr, p.ikc[ai].ptr, @intCast(pos0 + t));
        }
        try writeback(tri, p.s, h, 4, inj, y, wt);
        try readout(tri, p.s, lw.mlp_hc, h, inj);
        try moeStep(Tri, tri, p.s, p.stream, p.plan_fn, p.up_fn, p.down_fn, lw.router.ptr, lw.experts_up.ptr, lw.experts_down.ptr, y, wt);
        p.mode[t] = 2;
    }
}

fn zeros(gpa: std.mem.Allocator, driver: *cuda.Driver, n: usize, len: usize) ![]cuda.DeviceBuffer {
    const out = try gpa.alloc(cuda.DeviceBuffer, n);
    var i: usize = 0;
    errdefer {
        for (out[0..i]) |*b| b.free();
        gpa.free(out);
    }
    while (i < n) : (i += 1) {
        out[i] = try cuda.DeviceBuffer.alloc(driver, len);
        try out[i].fill8(0, null);
    }
    return out;
}

fn head(comptime Tri: type, p: *Pass, tri: Tri, row: usize) !void {
    if (p.mode[row] != 2) return error.UnexpectedTensor;
    const h = p.s.h.ptr + @as(u64, row) * wide * 2;
    const inj = p.s.inj.ptr + @as(u64, row) * inj_stride;
    const y = p.s.y.ptr + @as(u64, row) * slots * dims * 4;
    const wt = p.s.wts.ptr + @as(u64, row) * wts_stride;
    try writeback(tri, p.s, h, 2, inj, y, wt);
    p.mode[row] = 0;
    try readout(tri, p.s, p.mixer, h, inj);
    try qmm.matmul(p.driver, p.stream.*, p.s.mixed.ptr, p.s.xsm.ptr, p.head_q.w.ptr, p.head_q.s.ptr, p.head_q.b.ptr, p.s.logits.ptr, 1, vocab, dims);
    try p.stream.synchronize();
    try p.s.logits.download(0, p.logits);
}

fn forward(comptime Tri: type, p: *Pass, tri: Tri, layers: []Layer, ids: []const u32, pos0: usize) !void {
    @memset(p.mode, 0);
    for (ids, 0..) |token, t| try embedRow(Tri, p, tri, token, p.s.h.ptr + @as(u64, t) * wide * 2);
    for (layers, 0..) |*lw, layer| try runLayer(Tri, p, tri, lw, layer, ids, pos0);
    try head(Tri, p, tri, ids.len - 1);
}

pub const vocabulary: usize = vocab;

/// One loaded checkpoint: the embedding table, the mixer, the head, and a scratch row the forwards share.
pub const Engine = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: *cuda.Stream,
    store: Store,
    model_dir: []u8,
    table: embed.Table,
    mixer: HcW,
    head_q: Q4,
    scratch: Scratch,
    scratch_rows: usize,
    mode: []u8,
    eos: i64,
    experts: cuda.Module,
    plan_fn: cuda.Function,
    up_fn: cuda.Function,
    down_fn: cuda.Function,
    layers: [layers_n]Layer,
    n_layers: usize,

    pub fn init(gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, model_dir: []const u8) !*Engine {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        e.gpa = gpa;
        e.io = io;
        e.driver = driver;
        e.stream = stream;
        e.n_layers = 0;
        errdefer for (e.layers[0..e.n_layers]) |*l| l.free(gpa);
        e.model_dir = try gpa.dupe(u8, model_dir);
        errdefer gpa.free(e.model_dir);
        e.store = try Store.open(gpa, io, e.model_dir);
        errdefer e.store.deinit();
        const weight = try e.store.tensor(embed.weight_name);
        const scales = try e.store.tensor(embed.scales_name);
        const biases = try e.store.tensor(embed.biases_name);
        e.table = try embed.Table.upload(driver, weight, scales, biases);
        errdefer e.table.deinit();
        e.eos = try ple.eosOf(gpa, io, e.model_dir);
        e.scratch = try Scratch.alloc(driver, 1);
        errdefer e.scratch.free();
        try e.scratch.dummy.fill8(0, null);
        e.scratch_rows = 1;
        e.mode = try gpa.alloc(u8, 1);
        errdefer gpa.free(e.mode);
        e.mixer = try loadHc(gpa, driver, &e.store, "language_model.model.hyper_connection_mixer", false);
        errdefer e.mixer.free();
        var buf: [80]u8 = undefined;
        const head_w = try join3(gpa, &e.store, &buf, "language_model.lm_head", "");
        defer gpa.free(head_w.w);
        defer gpa.free(head_w.s);
        defer gpa.free(head_w.b);
        if (head_w.w.len != vocab * (dims / 8) * 4) return error.UnexpectedTensor;
        e.head_q = try uploadQ4(gpa, driver, head_w.w, head_w.s, head_w.b, vocab, dims / 8);
        errdefer e.head_q.free();
        e.experts = try cuda.Module.load(driver, cuda.kernels.experts);
        errdefer e.experts.unload();
        e.plan_fn = try e.experts.function(weights.plan_symbol);
        e.up_fn = try e.experts.function(weights.up_symbol);
        e.down_fn = try e.experts.function(weights.down_symbol);
        var pack_mod = try cuda.Module.load(driver, cuda.kernels.experts_pack);
        defer pack_mod.unload();
        const pack_fn = try pack_mod.function(weights.pack_symbol);
        for (0..layers_n) |i| {
            e.layers[i] = try loadLayer(gpa, driver, stream, &e.store, pack_fn, i);
            e.n_layers += 1;
        }
        e.store.release();
        e.store.closeFile();
        return e;
    }

    pub fn deinit(e: *Engine) void {
        e.experts.unload();
        for (e.layers[0..e.n_layers]) |*l| l.free(e.gpa);
        e.head_q.free();
        e.mixer.free();
        e.gpa.free(e.mode);
        e.scratch.free();
        e.table.deinit();
        e.store.deinit();
        e.gpa.free(e.model_dir);
        e.gpa.destroy(e);
    }

    fn grow(e: *Engine, rows: usize) !void {
        if (rows <= e.scratch_rows) return;
        var next = try Scratch.alloc(e.driver, rows);
        errdefer next.free();
        try next.dummy.fill8(0, null);
        const mode = try e.gpa.alloc(u8, rows);
        e.scratch.free();
        e.gpa.free(e.mode);
        e.scratch = next;
        e.mode = mode;
        e.scratch_rows = rows;
    }

    /// A fresh cache. `cap` is how many tokens it can hold, prompt plus generated.
    pub fn newSeq(e: *Engine, cap: usize) !Seq {
        if (cap == 0) return error.UnexpectedTensor;
        const logits = try e.gpa.alloc(u8, vocab * 2);
        errdefer e.gpa.free(logits);
        const conv = try zeros(e.gpa, e.driver, linear_n, (gdn.taps - 1) * gdn.conv_dim * 2);
        errdefer freeOwned(e.gpa, conv);
        const state = try zeros(e.gpa, e.driver, linear_n, state_n * 4);
        errdefer freeOwned(e.gpa, state);
        const kc = try zeros(e.gpa, e.driver, attn_n, cap * kv_heads * head_dim * 2);
        errdefer freeOwned(e.gpa, kc);
        const vc = try zeros(e.gpa, e.driver, attn_n, cap * kv_heads * head_dim * 2);
        errdefer freeOwned(e.gpa, vc);
        const ikc = try zeros(e.gpa, e.driver, attn_n, cap * index_dim * 2);
        errdefer freeOwned(e.gpa, ikc);
        var tail = try cuda.DeviceBuffer.alloc(e.driver, ple_tail_n * wide * 2);
        errdefer tail.free();
        try tail.fill8(0, null);
        return .{
            .conv = conv,
            .state = state,
            .kc = kc,
            .vc = vc,
            .ikc = ikc,
            .tail = tail,
            .history = .{ e.eos, e.eos },
            .fed = 0,
            .logits = logits,
            .cap = cap,
        };
    }

    /// Run `ids` on from `seq.fed` and leave the next token's scores in `seq.logits`.
    pub fn predict(e: *Engine, comptime Tri: type, tri: Tri, seq: *Seq, ids: []const u32) !void {
        if (ids.len == 0 or seq.fed + ids.len > seq.cap) return error.UnexpectedTensor;
        try e.grow(ids.len);
        var pass = Pass{
            .gpa = e.gpa,
            .io = e.io,
            .driver = e.driver,
            .stream = e.stream,
            .store = &e.store,
            .model_dir = e.model_dir,
            .s = &e.scratch,
            .table = e.table,
            .rows = ids.len,
            .mode = e.mode,
            .history = &seq.history,
            .conv = seq.conv,
            .state = seq.state,
            .kc = seq.kc,
            .vc = seq.vc,
            .ikc = seq.ikc,
            .tail = seq.tail,
            .mixer = e.mixer,
            .head_q = e.head_q,
            .logits = seq.logits,
            .plan_fn = e.plan_fn,
            .up_fn = e.up_fn,
            .down_fn = e.down_fn,
        };
        try forward(Tri, &pass, tri, &e.layers, ids, seq.fed);
        seq.fed += ids.len;
    }
};

/// One stream's caches. `fed` tokens have already gone through the model.
pub const Seq = struct {
    conv: []cuda.DeviceBuffer,
    state: []cuda.DeviceBuffer,
    kc: []cuda.DeviceBuffer,
    vc: []cuda.DeviceBuffer,
    ikc: []cuda.DeviceBuffer,
    tail: cuda.DeviceBuffer,
    history: [2]i64,
    fed: usize,
    logits: []u8,
    cap: usize,

    pub fn deinit(s: *Seq, gpa: std.mem.Allocator) void {
        freeBufs(s.conv);
        freeBufs(s.state);
        freeBufs(s.kc);
        freeBufs(s.vc);
        freeBufs(s.ikc);
        gpa.free(s.conv);
        gpa.free(s.state);
        gpa.free(s.kc);
        gpa.free(s.vc);
        gpa.free(s.ikc);
        if (s.tail.ptr != 0) s.tail.free();
        gpa.free(s.logits);
    }
};

fn freeBufs(bs: []cuda.DeviceBuffer) void {
    for (bs) |*b| b.free();
}

fn freeOwned(gpa: std.mem.Allocator, bs: []cuda.DeviceBuffer) void {
    freeBufs(bs);
    gpa.free(bs);
}

/// `out` receives `new_tokens` greedy ids. The first is the prompt's argmax. The rest are fed back one at a time.
pub fn run(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, model_dir: []const u8, tokens: []const u32, new_tokens: usize, out: []u32) !void {
    if (tokens.len == 0 or new_tokens == 0 or out.len < new_tokens or tokens.len + new_tokens > 256) return error.UnexpectedTensor;
    var e = try Engine.init(gpa, io, driver, stream, model_dir);
    defer e.deinit();
    var seq = try e.newSeq(tokens.len + new_tokens);
    defer seq.deinit(gpa);
    try e.predict(Tri, tri, &seq, tokens);
    out[0] = argmax(seq.logits);
    std.debug.print("token {d}\n", .{out[0]});
    var tok = out[0];
    for (1..new_tokens) |i| {
        var one = [_]u32{tok};
        try e.predict(Tri, tri, &seq, &one);
        tok = argmax(seq.logits);
        out[i] = tok;
        std.debug.print("token {d}\n", .{tok});
    }
}

test "linear layers skip every fourth index and attention starts at 3" {
    try std.testing.expect(linearLayer(0));
    try std.testing.expect(!linearLayer(3));
    try std.testing.expect(!linearLayer(47));
    try std.testing.expectEqual(@as(usize, 0), linearIndex(0));
    try std.testing.expectEqual(@as(usize, 2), linearIndex(2));
    try std.testing.expectEqual(@as(usize, 35), linearIndex(46));
    try std.testing.expectEqual(@as(usize, 0), attnIndex(3));
    try std.testing.expectEqual(@as(usize, 11), attnIndex(47));
}
