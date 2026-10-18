//! The 27B trainer's input-weight fits on the device and its greedy scoring through serving's own rounds (train.zig).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const living = @import("living");
const c = @import("cuda_learn_shape.zig");

const hooks = @import("cuda_learn_hooks.zig");
const to = @import("cuda_train_ops.zig");
const engine = @import("cuda_engine.zig");

const load = @import("cuda_train_load.zig");

const gt = cuda.gdn_train;
const train_mode = living.train_mode;
const lesson_route = living.lesson_route;
const Count = living.pocket_clock.Count;
const train = @import("cuda_train.zig");
const max_rows = train.max_rows;
const keepLow = train.keepLow;
const Seq = train.Seq;
const pack_rows = train.pack_rows;
const max_seqs = train.max_seqs;

/// Input reads consume only layer taps; loss and target passes still keep the complete vocabulary head.
pub fn keepFor(mode: train_mode.Mode) engine.Keep {
    return if (mode == .keys) .hidden else .logits;
}

/// Pocket neurons' input weights fitted on the device: each job's Adam steps.
pub fn fitDevice(t: anytype, jobs: []const living.pocket_fit.DeviceFit) !void {
    if (jobs.len == 0) return;
    const pf = living.pocket_fit;
    const d = jobs[0].d;
    const m = jobs[0].kinds.len;
    for (jobs) |j| if (j.d != d or j.kinds.len != m or j.steps != jobs[0].steps or j.below != jobs[0].below or j.rows.len < m * j.slots * d) return error.BadFit;
    const tree = if (@import("sys").c.getenv("TF_SLIDE_FIT_REDUCE")) |v| std.mem.eql(u8, std.mem.span(v), "tree") else false;
    const gib: usize = if (@import("sys").c.getenv("TF_SLIDE_FIT_GIB")) |v| std.fmt.parseInt(usize, std.mem.span(v), 10) catch 4 else 4;
    const per = m * d * 2;
    const L = @max(1, @min(jobs.len, (gib << 30) / per));
    const drv = t.e.d;
    const sh = t.e.stream.handle;
    var bufs: [8]cuda.DeviceBuffer = undefined;
    const sizes = [_]usize{ L * per, L * m, L * d * 4, L * d * 4, L * d * 4, L * m * 4, L * 4 * 4, L * 4 };
    var made: usize = 0;
    defer for (bufs[0..made]) |*b| b.free();
    for (sizes, 0..) |n, i| {
        bufs[i] = try cuda.DeviceBuffer.alloc(drv, @max(n, 256));
        made += 1;
    }
    errdefer @import("qwen_vision").completion.drain(t.e.d, t.e.stream);
    const x = bufs[0].ptr;
    const kinds = bufs[1].ptr;
    const uu = bufs[2].ptr;
    const mm = bufs[3].ptr;
    const vv = bufs[4].ptr;
    const cc = bufs[5].ptr;
    const wrong = bufs[6].ptr;
    const invs = bufs[7].ptr;
    const o = t.ops();
    var g0: usize = 0;
    while (g0 < jobs.len) : (g0 += L) {
        const group = jobs[g0..@min(jobs.len, g0 + L)];
        const n = group.len;
        try t.e.stream.synchronize();
        var inv_host: [64]f32 = undefined;
        if (n > inv_host.len) return error.BadFit;
        for (group, 0..) |j, i| {
            const cp: cuda.abi.Memcpy2D = .{ .src_type = 1, .src_host = @ptrCast(j.rows.ptr + j.slot * d), .src = 0, .src_pitch = j.slots * d * 2, .dst = x + i * per, .dst_pitch = d * 2, .width = d * 2, .height = m };
            try drv.check(drv.api.cuMemcpy2DAsync_v2(&cp, sh), "cuMemcpy2DAsync");
            try bufs[1].uploadAsync(i * m, std.mem.sliceAsBytes(j.kinds), sh);
            try bufs[2].uploadAsync(i * d * 4, std.mem.sliceAsBytes(j.u), sh);
            inv_host[i] = j.inv;
        }
        try bufs[7].uploadAsync(0, std.mem.sliceAsBytes(inv_host[0..n]), sh);
        try drv.check(drv.api.cuMemsetD8Async(mm, 0, n * d * 4, sh), "cuMemsetD8Async");
        try drv.check(drv.api.cuMemsetD8Async(vv, 0, n * d * 4, sh), "cuMemsetD8Async");
        try t.e.stream.synchronize(); // the host rows and lists go on being read only until here
        for (1..group[0].steps + 1) |s| {
            try drv.check(drv.api.cuMemsetD8Async(wrong, 0, n * 16, sh), "cuMemsetD8Async");
            try o.fitDots(tree, x, m, d, uu, kinds, invs, group[0].below, cc, wrong, n);
            const cr = pf.corrections(s);
            try o.fitStep(x, m, d, kinds, cc, wrong, invs, uu, mm, vv, pf.fit_rate, cr[0], cr[1], n);
        }
        try t.e.stream.synchronize();
        for (group, 0..) |j, i| try bufs[2].download(i * d * 4, std.mem.sliceAsBytes(j.u));
    }
}

/// Whether each of `seqs`' answers is its prompt's greedy reply under the model as it is now.
pub fn greedy(t: anytype, seqs: []const Seq, same: []bool) !void {
    if (seqs.len > max_seqs or same.len < seqs.len) return error.BadPass;
    const e = t.e;
    for (seqs, 0..) |sq, si| {
        if (sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len > max_rows + 1) return error.BadExample;
        try t.states[si].reset(e);
        same[si] = true;
        for (0..sq.ids.len - 1) |row| {
            try e.forward(&.{.{ .cache = &t.states[si], .rows = 1 }}, sq.ids[row..][0..1], .{ .draws = &.{.{ .rule = null, .position = row + 1 }} });
            if (row + 1 >= sq.start) same[si] = same[si] and e.tokens[0] == sq.ids[row + 1];
        }
    }
}

/// Full fresh conversation taps; the optional prefix optimization is deliberately unused.
pub fn tapRows(t: anytype, ids: []const u32, context: usize, rows: []const u32, layers: []const usize, out: []u16) !void {
    const e = t.e;
    const D = t.g.hidden;
    if (ids.len == 0 or context > ids.len or out.len != rows.len * layers.len * D) return error.BadTap;
    for (rows, 0..) |row, i| if (row >= ids.len or (i > 0 and row <= rows[i - 1])) return error.BadTap;
    for (layers) |l| if (l >= t.g.layers) return error.BadTap;
    var cache = try engine.Cache.init(t.gpa, e, ids.len + 1);
    defer {
        @import("qwen_vision").completion.drain(e.d, e.stream);
        cache.deinit(t.gpa);
    }
    const prior = e.learning;
    defer e.learning = prior;
    var keep = t.keep;
    keep.layers = t.light;
    keep.final_h = 0;
    keep.logits = 0;
    keep.at = 0;
    e.learning = &keep;
    var ri: usize = 0;
    for (ids, 0..) |id, row| {
        try e.forward(&.{.{ .cache = &cache, .rows = 1 }}, &.{id}, keepFor(.keys));
        if (ri == rows.len or row != rows[ri]) continue;
        for (layers, 0..) |l, j| @memcpy(out[(ri * layers.len + j) * D ..][0..D], t.normed(l, 0));
        ri += 1;
    }
}

/// The keep lists and the backward's scratch.
pub fn carve(t: anytype) !void {
    const R = pack_rows;
    const g = t.g;
    const low = keepLow(g.layers);
    t.full = try t.gpa.alloc(hooks.LayerKeep, g.layers);
    t.light = try t.gpa.alloc(hooks.LayerKeep, g.layers);
    const proj = g.convDim() + g.vInner() + 2 * g.linear_v_heads;
    for (0..g.layers) |l| {
        const y = t.ys.dev + l * R * g.hidden * 2;
        t.light[l] = .{ .y = y };
        var k: hooks.LayerKeep = .{ .y = y };
        if (l >= low) {
            k.h_in = try t.device(R * g.hidden * 2);
            k.h_mid = try t.device(R * g.hidden * 2);
            k.gate = try t.device(R * g.intermediate * 2);
            k.up = try t.device(R * g.intermediate * 2);
            if (c.linear(l)) {
                k.proj = try t.device(R * proj * 2);
            } else {
                k.pq = try t.device(R * 2 * g.qInner() * 2);
                k.pk = try t.device(R * g.kvInner() * 2);
                k.pv = try t.device(R * g.kvInner() * 2);
                k.gated = try t.device(R * g.qInner() * 2);
            }
        }
        t.full[l] = k;
    }
    t.logits = try t.device(R * c.vocab * 2);
    t.last = try t.device(R * g.hidden * 2);
    t.last_xs = try t.device(R * g.hidden / 64 * 4);
    t.final_h = try t.device(R * g.hidden * 2);
    t.grad = try t.device(R * g.hidden * 4);
    t.dy = try t.device(R * g.hidden * 4);
    t.dlast = try t.device(R * g.hidden * 4);
    t.parts = try t.device((R + 512) * g.hidden * 4);
    t.ms = .{ .dact = try t.device(R * g.intermediate * 4), .dgate = try t.device(R * g.intermediate * 4), .dup = try t.device(R * g.intermediate * 4) };
    const M = max_rows;
    const qw = g.qInner();
    const kw = g.kvInner();
    // dg, dpq, dpk and dv over every row of a pass: the projections run once for all its sequences
    t.as = .{
        .qkv = try t.device(M * (qw + 2 * kw) * 2),
        .dg = try t.device(R * qw * 4),
        .dout = try t.device(M * qw * 4),
        .dpq = try t.device(R * 2 * qw * 4),
        .dq = try t.device(M * qw * 4),
        .dk = try t.device(M * kw * 4),
        .dv = try t.device(R * kw * 4),
        .dpk = try t.device(R * kw * 4),
        .probs = try t.device(g.query_heads * M * M * 4),
        .dscores = try t.device(g.query_heads * M * M * 4),
    };
    t.gw = gt.Wide.carve(try t.device(gt.Wide.bytes(R, g.linear_k_heads, g.linear_v_heads, g.hidden)), R, g.linear_k_heads, g.linear_v_heads);
    t.gs = gt.Scratch.carve(try t.device(gt.Scratch.bytes(M, g.linear_k_heads, g.linear_v_heads, g.hidden)), M, g.linear_k_heads, g.linear_v_heads, g.hidden);
    t.keep = .{ .t = t.ops(), .d = t.e.d, .g = g, .layers = t.full, .final_h = t.final_h, .last = t.last, .last_xs = t.last_xs, .logits = t.logits };
}
