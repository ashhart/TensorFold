//! The row-by-row reference: each prompt row one fused decode step, the batched pass's check in flashnext-prompt.

const std = @import("std");
const cuda = @import("cuda");
const embed = @import("cuda_embed.zig");
const qmm = @import("cuda_qmm.zig");
const weights = @import("cuda_prompt_wt.zig");
const step = @import("cuda_prompt_step.zig");
const load = @import("cuda_prompt_load.zig");
const prompt = @import("cuda_prompt.zig");

const dims = weights.dims;
const streams = weights.streams;
const wide = weights.wide;
const vocab = weights.vocab;
const slots = weights.slots;
const ple_layer = weights.ple_layer;
const inj_stride = weights.inj_stride;
const wts_stride = weights.wts_stride;
const Q4 = weights.Q4;
const HcW = weights.HcW;
const Scratch = step.Scratch;
const Layer = load.Layer;
const promote = weights.promote;

/// The row path's own scratch and expert kernels, made for a check and dropped after it.
pub const Rows = struct {
    gpa: std.mem.Allocator,
    driver: *cuda.Driver,
    scratch: Scratch,
    scratch_rows: usize,
    mode: []u8,
    experts: cuda.Module,
    plan_fn: cuda.Function,
    up_fn: cuda.Function,
    down_fn: cuda.Function,

    pub fn init(gpa: std.mem.Allocator, driver: *cuda.Driver, stream: *cuda.Stream) !Rows {
        var r: Rows = .{ .gpa = gpa, .driver = driver, .scratch = try Scratch.alloc(driver, 1), .scratch_rows = 1, .mode = undefined, .experts = undefined, .plan_fn = undefined, .up_fn = undefined, .down_fn = undefined };
        errdefer r.scratch.free();
        try r.scratch.dummy.fill8(0, stream.handle);
        r.mode = try gpa.alloc(u8, 1);
        errdefer gpa.free(r.mode);
        r.experts = try cuda.Module.load(driver, cuda.kernels.experts);
        errdefer r.experts.unload();
        r.plan_fn = try r.experts.function(weights.plan_symbol);
        r.up_fn = try r.experts.function(weights.up_symbol);
        r.down_fn = try r.experts.function(weights.down_symbol);
        return r;
    }

    pub fn deinit(r: *Rows) void {
        r.experts.unload();
        r.gpa.free(r.mode);
        r.scratch.free();
    }

    fn grow(r: *Rows, stream: cuda.Stream, rows: usize) !void {
        if (rows <= r.scratch_rows) return;
        var next = try Scratch.alloc(r.driver, rows);
        errdefer next.free();
        try next.dummy.fill8(0, stream.handle);
        const mode = try r.gpa.alloc(u8, rows);
        r.scratch.free();
        r.gpa.free(r.mode);
        r.scratch = next;
        r.mode = mode;
        r.scratch_rows = rows;
    }

    /// Every row of `ids` from seq.fed one fused decode step; the last row's scores land in seq.logits.
    pub fn predict(r: *Rows, e: *prompt.Engine, comptime Tri: type, tri: Tri, seq: *prompt.Seq, ids: []const u32) !void {
        if (ids.len == 0 or seq.fed + ids.len > seq.cap) return error.UnexpectedTensor;
        // dense attention only: the reference stops where the indexer would start choosing
        if (seq.fed + ids.len > weights.dense_nch * 512) return error.UnexpectedTensor;
        try r.grow(e.stream.*, ids.len);
        var pass = Pass{ .gpa = e.gpa, .io = e.io, .driver = e.driver, .stream = e.stream, .model_dir = e.model_dir, .s = &r.scratch, .table = e.table, .mode = r.mode, .history = &seq.history, .conv = seq.conv, .state = seq.state, .kc = seq.kc, .vc = seq.vc, .ikc = seq.ikc, .tail = seq.tail, .mixer = e.mixer, .head_q = e.head_q, .logits = seq.logits, .plan_fn = r.plan_fn, .up_fn = r.up_fn, .down_fn = r.down_fn };
        @memset(pass.mode, 0);
        for (ids, 0..) |token, t| try embedRow(Tri, &pass, tri, token, pass.s.h.ptr + @as(u64, t) * wide * 2);
        for (e.layers[0..e.n_layers], 0..) |*lw, layer| try runLayer(Tri, &pass, tri, lw, layer, ids, seq.fed);
        try head(Tri, &pass, tri, ids.len - 1);
        seq.fed += ids.len;
    }
};

const Pass = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    driver: *cuda.Driver,
    stream: *cuda.Stream,
    model_dir: []const u8,
    s: *Scratch,
    table: embed.Table,
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
    // on the compute stream: a synchronous copy can land before an earlier row's queued _embed reads its id
    try p.s.ids.uploadAsync(0, std.mem.asBytes(&id), p.stream.handle);
    try tri.embed(p.s.ids.ptr, p.table.w.ptr, p.table.s.ptr, p.table.b.ptr, h, dims, streams, 1);
}

fn runLayer(comptime Tri: type, p: *Pass, tri: Tri, lw: *Layer, layer: usize, ids: []const u32, pos0: usize) !void {
    for (ids, 0..) |token, t| {
        const h = p.s.h.ptr + @as(u64, t) * wide * 2;
        const inj = p.s.inj.ptr + @as(u64, t) * inj_stride;
        const y = p.s.y.ptr + @as(u64, t) * slots * dims * 4;
        const wt = p.s.wts.ptr + @as(u64, t) * wts_stride;
        if (layer == ple_layer and p.mode[t] == 2) {
            try step.writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        }
        if (layer == ple_layer) try step.pleStep(Tri, p.gpa, p.io, p.driver, p.stream, tri, p.s, p.model_dir, p.history, token, h, lw.key, lw.val, lw.nk.ptr, lw.nq.ptr, lw.nc.ptr, lw.ple_cw.ptr, &p.tail);
        if (p.mode[t] == 2) {
            try step.writeback(tri, p.s, h, 2, inj, y, wt);
            p.mode[t] = 0;
        } else try step.writeback(tri, p.s, h, 0, inj, y, wt);
        try step.readout(tri, p.s, lw.attn_hc, h, inj);
        if (lw.linear) {
            const li = weights.linearIndex(layer);
            try step.gdnStep(p.gpa, p.driver, p.stream, p.s, lw.proj, lw.out, p.conv[li], lw.cw.ptr, lw.a_log.ptr, lw.dt.ptr, lw.norm.ptr, &p.state[li]);
        } else {
            const ai = weights.attnIndex(layer);
            try step.attnStep(Tri, p.driver, p.stream, tri, p.s, lw.proj, lw.out, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, p.kc[ai].ptr, p.vc[ai].ptr, p.ikc[ai].ptr, @intCast(pos0 + t));
        }
        try step.writeback(tri, p.s, h, 4, inj, y, wt);
        try step.readout(tri, p.s, lw.mlp_hc, h, inj);
        try step.moeStep(Tri, tri, p.s, p.stream, p.plan_fn, p.up_fn, p.down_fn, lw.router.ptr, lw.experts_up.ptr, lw.experts_down.ptr, y, wt);
        p.mode[t] = 2;
    }
}

fn head(comptime Tri: type, p: *Pass, tri: Tri, row: usize) !void {
    if (p.mode[row] != 2) return error.UnexpectedTensor;
    const h = p.s.h.ptr + @as(u64, row) * wide * 2;
    const inj = p.s.inj.ptr + @as(u64, row) * inj_stride;
    const y = p.s.y.ptr + @as(u64, row) * slots * dims * 4;
    const wt = p.s.wts.ptr + @as(u64, row) * wts_stride;
    try step.writeback(tri, p.s, h, 2, inj, y, wt);
    p.mode[row] = 0;
    try step.readout(tri, p.s, p.mixer, h, inj);
    try qmm.matmul(p.driver, p.stream.*, p.s.mixed.ptr, p.s.xsm.ptr, p.head_q.w.ptr, p.head_q.s.ptr, p.head_q.b.ptr, p.s.logits.ptr, 1, vocab, dims);
    try p.stream.synchronize();
    try p.s.logits.download(0, p.logits);
}

/// The five best scores of one row, best first, as "id:score" pairs on stderr (the capture's top5 field).
fn printTop(raw: []const u8) void {
    var ids: [5]usize = @splat(0);
    var vals: [5]f32 = @splat(-std.math.inf(f32));
    for (0..vocab) |i| {
        const v = promote(std.mem.readInt(u16, raw[2 * i ..][0..2], .little));
        if (v <= vals[4]) continue;
        var at: usize = 4;
        while (at > 0 and v > vals[at - 1]) : (at -= 1) {
            vals[at] = vals[at - 1];
            ids[at] = ids[at - 1];
        }
        vals[at] = v;
        ids[at] = i;
    }
    std.debug.print("top5", .{});
    for (ids, vals) |id, v| std.debug.print(" {d}:{d:.4}", .{ id, v });
    std.debug.print("\n", .{});
}

fn ms(ns: i96) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// `out` gets `new_tokens` greedy ids: the prompt's argmax, then one fed back at a time (TF_FLASHNEXT_PROMPT=rows).
pub fn run(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, model_dir: []const u8, tokens: []const u32, new_tokens: usize, out: []u32) !void {
    if (tokens.len == 0 or new_tokens == 0 or out.len < new_tokens or tokens.len + new_tokens > prompt.max_tokens) return error.UnexpectedTensor;
    var e = try prompt.Engine.init(gpa, io, driver, stream, model_dir);
    defer e.deinit();
    const by_rows = if (std.c.getenv("TF_FLASHNEXT_PROMPT")) |v| std.mem.eql(u8, std.mem.span(v), "rows") else false;
    var rows: ?Rows = if (by_rows) try Rows.init(gpa, driver, stream) else null;
    defer if (rows) |*r| r.deinit();
    var seq = try e.newSeq(tokens.len + new_tokens);
    defer seq.deinit(gpa);
    const loaded = cuda.usage(true);
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    if (rows) |*r| try r.predict(e, Tri, tri, &seq, tokens) else try e.predict(Tri, tri, &seq, tokens);
    const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
    out[0] = prompt.argmax(seq.logits);
    printTop(seq.logits);
    // TF_FLASHNEXT_LOGITS=path: the prompt's last row of scores (bf16), to compare the two paths byte for byte
    if (std.c.getenv("TF_FLASHNEXT_LOGITS")) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = std.mem.span(path), .data = seq.logits });
    std.debug.print("token {d}\n", .{out[0]});
    var tok = out[0];
    for (1..new_tokens) |i| {
        var one = [_]u32{tok};
        if (rows) |*r| try r.predict(e, Tri, tri, &seq, &one) else try e.predict(Tri, tri, &seq, &one);
        tok = prompt.argmax(seq.logits);
        if (i < 4) printTop(seq.logits);
        out[i] = tok;
        std.debug.print("token {d}\n", .{tok});
    }
    const t2 = std.Io.Clock.awake.now(io).toNanoseconds();
    const per = if (new_tokens > 1) ms(t2 - t1) / @as(f64, @floatFromInt(new_tokens - 1)) else 0;
    std.debug.print("timing prompt {d} rows {d:.1} ms, decode {d} tokens {d:.2} ms/token\n", .{ tokens.len, ms(t1 - t0), new_tokens - 1, per });
    const used = cuda.usage(false);
    std.debug.print("memory: {d:.2} GiB on the device after the load and the sequence, peak {d:.2} GiB during the prompt and decode (+{d:.2} GiB)\n", .{ gib(loaded.device), gib(used.peak), gib(used.peak - loaded.device) });
}
