//! The expert stage on this GPU against host references: picks, weights, sort, sum byte-equal; experts to float64.
const std = @import("std");
const mtl = @import("metal");
const core = @import("core");
const tile = core.tile_qmm;
const route = core.moe_route;
const Config = @import("config.zig").Config;
const moe = @import("moe.zig");
const mw = core.qwen35_experts;

const hidden = 256;
const experts = 8;
const topk = 2;
const inter = 64;
const capacity = 16;
const ws = topk + 1;

fn config() Config {
    return .{ .hidden = hidden, .intermediate = 0, .layers = 1, .vocab = 1, .heads = 1, .kv_heads = 1, .head_dim = 1, .k_heads = 1, .v_heads = 1, .dk = 1, .dv = 1, .conv_kernel = 1, .eps = 1e-6, .rope_dims = 2, .rope_theta = 1, .experts = experts, .topk = topk, .moe_intermediate = inter, .shared_intermediate = inter };
}

const cols = (experts + 1 + 3) / 4 * 4;

fn wait(cb: mtl.CommandBuffer) !void {
    cb.commit();
    cb.wait();
    if (cb.failure()) |why| {
        std.log.err("moe test: {s}", .{why});
        return error.GpuFailure;
    }
}

/// precise::exp of each value on this GPU: the host rule takes the kernel's exp, the rest is IEEE f32.
fn gpuExp(device: mtl.Device, queue: mtl.Queue, xs: []f32) !void {
    const src =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\[[kernel]] void probe(device float* X [[buffer(0)]], uint i [[thread_position_in_grid]]) { X[i] = precise::exp(X[i]); }
    ;
    const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const pipe = try mtl.Pipeline.init(device, lib, "probe", false);
    defer pipe.deinit();
    const b = try device.buffer(xs.len * 4, mtl.ResourceOptions.shared);
    defer b.deinit();
    @memcpy(b.slice(f32, xs.len), xs);
    const cb = queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(pipe);
    e.setBuffer(b, 0, 0);
    e.dispatchThreads(mtl.Size.of(xs.len, 1, 1), mtl.Size.of(32, 1, 1));
    e.end();
    try wait(cb);
    @memcpy(xs, b.slice(f32, xs.len));
}

/// The select rule: top-k by logit (ties to the lower id), bf16 softmax in pick order, bf16(sigmoid(bf16(gate))).
fn hostSelect(device: mtl.Device, queue: mtl.Queue, logits: []const f32, rows: usize, picks: []i32, wts: []f32) !void {
    var args: [capacity * ws]f32 = undefined;
    for (0..rows) |r| {
        const row = logits[r * cols ..][0..cols];
        var taken: [experts]bool = @splat(false);
        var top: f32 = 0;
        for (0..topk) |k| {
            var best: usize = experts;
            for (0..experts) |id| if (!taken[id] and (best == experts or row[id] > row[best])) {
                best = id;
            };
            taken[best] = true;
            if (k == 0) top = row[best];
            picks[r * topk + k] = @intCast(best);
            args[r * ws + k] = row[best] - top;
        }
        args[r * ws + topk] = @abs(mw.float(mw.bf16(row[experts])));
    }
    try gpuExp(device, queue, args[0 .. rows * ws]);
    for (0..rows) |r| {
        const ex = args[r * ws ..][0..ws];
        var total: f32 = 0;
        for (ex[0..topk]) |v| total += v;
        for (0..topk) |k| wts[r * ws + k] = mw.float(mw.bf16(ex[k] / total));
        const g = mw.float(mw.bf16(logits[r * cols + experts]));
        const y = 1.0 / (1.0 + ex[topk]);
        wts[r * ws + topk] = mw.float(mw.bf16(if (g < 0) y else 1.0 - y));
    }
}

const Rig = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    stage: *moe.Stage,
    x: mw.Experts,
    mats: [6][3]mtl.Buffer,
    input: mtl.Buffer,
    out: mtl.Buffer,
    /// f64 weights: router [cols][hidden]; gate, up [e][inter][hidden]; down [e][hidden][inter]; shared last.
    router: []f64,
    gate: []f64,
    up: []f64,
    down: []f64,

    fn deinit(r: *Rig, a: std.mem.Allocator) void {
        r.stage.deinit(a);
        r.x.router.deinit();
        for (r.mats) |set| for (set) |b| b.deinit();
        r.input.deinit();
        r.out.deinit();
        for ([_][]f64{ r.router, r.gate, r.up, r.down }) |s| a.free(s);
        r.queue.deinit();
        r.device.deinit();
    }
};

/// `count` random 4-bit g64 matrices [n, k] into `bufs`, their dequantized weights into `f`.
fn matrices(device: mtl.Device, rng: std.Random, count: usize, n: usize, k: usize, f: []f64, bufs: *[3]mtl.Buffer) !tile.Matrix {
    const words = count * n * k / 8;
    const groups = count * n * k / 64;
    for (bufs, [_]usize{ words * 4, groups * 2, groups * 2 }) |*b, bytes| b.* = try device.buffer(bytes, mtl.ResourceOptions.shared);
    const w = bufs[0].slice(u32, words);
    const s = bufs[1].slice(u16, groups);
    const b = bufs[2].slice(u16, groups);
    for (w) |*v| v.* = rng.int(u32);
    for (s, b) |*sv, *bv| {
        sv.* = mw.bf16(0.01 + 0.02 * rng.float(f32));
        bv.* = mw.bf16(-0.15 + 0.05 * rng.float(f32));
    }
    for (0..count * n) |row| for (0..k) |j| {
        const q = (w[row * k / 8 + j / 8] >> @intCast(4 * (j % 8))) & 15;
        const g = row * (k / 64) + j / 64;
        f[row * k + j] = @as(f64, mw.float(s[g])) * @as(f64, @floatFromInt(q)) + @as(f64, mw.float(b[g]));
    };
    return .{ .w = bufs[0], .scales = bufs[1], .biases = bufs[2], .n = n, .k = k, .bits = 4 };
}

fn rig(a: std.mem.Allocator) !Rig {
    var r: Rig = undefined;
    r.device = mtl.Device.init() catch return error.SkipZigTest;
    r.queue = try r.device.queue();
    var prng = std.Random.DefaultPrng.init(0x35a3);
    const rng = prng.random();
    r.router = try a.alloc(f64, cols * hidden);
    const ei = experts * inter * hidden;
    for ([_]*[]f64{ &r.gate, &r.up, &r.down }) |p| p.* = try a.alloc(f64, ei + inter * hidden);
    r.x.gate = try matrices(r.device, rng, experts, inter, hidden, r.gate[0..ei], &r.mats[0]);
    r.x.up = try matrices(r.device, rng, experts, inter, hidden, r.up[0..ei], &r.mats[1]);
    r.x.down = try matrices(r.device, rng, experts, hidden, inter, r.down[0..ei], &r.mats[2]);
    r.x.shared_gate = try matrices(r.device, rng, 1, inter, hidden, r.gate[ei..], &r.mats[3]);
    r.x.shared_up = try matrices(r.device, rng, 1, inter, hidden, r.up[ei..], &r.mats[4]);
    r.x.shared_down = try matrices(r.device, rng, 1, hidden, inter, r.down[ei..], &r.mats[5]);
    const router = try a.alloc(u16, cols * hidden);
    defer a.free(router);
    @memset(router, 0);
    for (router[0 .. (experts + 1) * hidden]) |*v| v.* = mw.bf16(0.2 * (rng.float(f32) * 2 - 1));
    for (router, r.router) |v, *f| f.* = mw.float(v);
    r.x.router = try r.device.buffer(cols * hidden * 2, mtl.ResourceOptions.shared);
    route.pack(router, r.x.router.slice(u16, cols * hidden), cols, hidden);
    r.input = try r.device.buffer(capacity * hidden * 2, mtl.ResourceOptions.shared);
    r.out = try r.device.buffer(capacity * hidden * 2, mtl.ResourceOptions.shared);
    for (r.input.slice(u16, capacity * hidden)) |*v| v.* = mw.bf16(rng.float(f32) * 2 - 1);
    r.stage = try moe.Stage.init(a, r.device, config(), capacity, r.x);
    return r;
}

fn dot(w: []const f64, x: anytype) f64 {
    var acc: f64 = 0;
    for (w, 0..) |v, i| acc += v * x[i];
    return acc;
}

/// One stage call: logits to f64; picks, weights, sort, sum to the host on the GPU's inputs; output to f64.
fn check(r: *Rig, rows: u32, prompt: bool) !void {
    const s = r.stage;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    try s.encode(e, r.x, .{ .buffer = r.input }, .{ .buffer = r.out }, rows, prompt);
    e.end();
    try wait(cb);
    const slots = rows * topk;
    const x = r.input.slice(u16, rows * hidden);
    var xf: [capacity * hidden]f64 = undefined;
    for (x, 0..) |v, i| xf[i] = mw.float(v);
    const logits = s.get(.logits).slice(f32, rows * cols);
    var logit_worst: f64 = 0;
    for (0..rows) |row| for (0..experts + 1) |col| {
        const want = dot(r.router[col * hidden ..][0..hidden], xf[row * hidden ..]);
        try std.testing.expectApproxEqAbs(want, @as(f64, logits[row * cols + col]), 1e-4 * (1 + @abs(want)));
        logit_worst = @max(logit_worst, @abs(want - logits[row * cols + col]) / (1 + @abs(want)));
    };
    var picks: [capacity * topk]i32 = undefined;
    var wts: [capacity * ws]f32 = undefined;
    try hostSelect(r.device, r.queue, logits, rows, &picks, &wts);
    try std.testing.expectEqualSlices(i32, picks[0..slots], s.get(.picks).slice(i32, slots));
    try std.testing.expectEqualSlices(u32, @ptrCast(wts[0 .. rows * ws]), s.get(.wts).slice(u32, rows * ws));
    const twin = !prompt and s.twin and slots <= tile.window_slots;
    var pos: [capacity * topk]u32 = undefined;
    for (&pos, 0..) |*p, i| p.* = @intCast(i);
    if (!twin) {
        // a stable counting sort by expert
        var at: u32 = 0;
        const starts = s.get(.starts).slice(u32, experts + 1);
        for (0..experts) |id| {
            try std.testing.expectEqual(at, starts[id]);
            for (0..slots) |j| if (picks[j] == @as(i32, @intCast(id))) {
                pos[j] = at;
                try std.testing.expectEqual(@as(u32, @intCast(j / topk)), s.get(.rows).slice(u32, slots)[at]);
                at += 1;
            };
        }
        try std.testing.expectEqual(@as(u32, slots), starts[experts]);
        try std.testing.expectEqualSlices(u32, pos[0..slots], s.get(.pos).slice(u32, slots));
        try std.testing.expectEqualSlices(u32, &.{ 0, rows }, s.get(.one).slice(u32, 2));
    }
    const y = s.get(.y).slice(u16, slots * hidden);
    const ys = s.get(.shared_y).slice(u16, rows * hidden);
    const out = r.out.slice(u16, rows * hidden);
    var worst: f64 = 0;
    var scale: f64 = 0;
    for (0..rows) |row| {
        var want: [hidden]f64 = @splat(0);
        for (0..ws) |k| {
            const id: usize = if (k == topk) experts else @intCast(picks[row * topk + k]);
            var act: [inter]f64 = undefined;
            for (&act, 0..) |*v, j| {
                const g = dot(r.gate[(id * inter + j) * hidden ..][0..hidden], xf[row * hidden ..]);
                v.* = g / (1 + @exp(-g)) * dot(r.up[(id * inter + j) * hidden ..][0..hidden], xf[row * hidden ..]);
            }
            for (&want, 0..) |*v, d| v.* += wts[row * ws + k] * dot(r.down[(id * hidden + d) * inter ..][0..inter], &act);
        }
        for (0..hidden) |d| {
            var acc: f32 = 0;
            for (0..ws) |k| {
                const v = if (k == topk) ys[row * hidden + d] else y[pos[row * topk + k] * hidden + d];
                acc = @mulAdd(f32, wts[row * ws + k], mw.float(v), acc);
            }
            try std.testing.expectEqual(mw.bf16(acc), out[row * hidden + d]);
            worst = @max(worst, @abs(want[d] - mw.float(out[row * hidden + d])));
            scale = @max(scale, @abs(want[d]));
        }
    }
    try std.testing.expect(scale > 0.05 and worst <= 0.03 * scale);
    const path = if (prompt) "prompt tiles" else if (twin) "decode twin" else "by-expert window";
    std.debug.print("qwen27 moe {s} rows={d}: router logits worst {e:.2} of 1+|f64|; output worst column {e:.2} at scale {d:.3}\n", .{ path, rows, logit_worst, worst, scale });
}

test "the expert stage: prompt tiles, decode twin and a by-expert window against the host rules and float64" {
    const a = std.testing.allocator;
    var r = try rig(a);
    defer r.deinit(a);
    try check(&r, 5, true);
    try check(&r, 1, false);
    try check(&r, 3, false);
    try check(&r, capacity, false);
    try check(&r, capacity, true);
}

fn run(r: *Rig, rows: u32, prompt: bool) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    try r.stage.encode(e, r.x, .{ .buffer = r.input }, .{ .buffer = r.out }, rows, prompt);
    e.end();
    try wait(cb);
}

test "every sequence of up to three stage calls: each row's bytes are its one-row call's, whatever ran before" {
    const a = std.testing.allocator;
    var r = try rig(a);
    defer r.deinit(a);
    const in = r.input.slice(u16, capacity * hidden);
    const out = r.out.slice(u16, capacity * hidden);
    var pool: [capacity * hidden]u16 = undefined;
    @memcpy(&pool, in);
    var want: [capacity * hidden]u16 = undefined;
    for (0..capacity) |i| {
        @memcpy(in[0..hidden], pool[i * hidden ..][0..hidden]);
        try run(&r, 1, false);
        @memcpy(want[i * hidden ..][0..hidden], out[0..hidden]);
    }
    // prompt tiles, the decode twin and the by-expert window, each at a few widths
    const Event = struct { rows: u32, prompt: bool };
    const events = [_]Event{ .{ .rows = 1, .prompt = true }, .{ .rows = 5, .prompt = true }, .{ .rows = capacity, .prompt = true }, .{ .rows = 1, .prompt = false }, .{ .rows = 3, .prompt = false }, .{ .rows = capacity, .prompt = false } };
    var sequences: usize = 0;
    for (1..4) |len| for (0..std.math.pow(usize, events.len, len)) |code| {
        var at: usize = 0;
        var c = code;
        for (0..len) |_| {
            const ev = events[c % events.len];
            c /= events.len;
            for (0..ev.rows) |j| @memcpy(in[j * hidden ..][0..hidden], pool[(at + j) % capacity * hidden ..][0..hidden]);
            @memset(out, 0xffff);
            try run(&r, ev.rows, ev.prompt);
            for (0..ev.rows) |j| try std.testing.expectEqualSlices(u16, want[(at + j) % capacity * hidden ..][0..hidden], out[j * hidden ..][0..hidden]);
            at += ev.rows;
        }
        sequences += 1;
    };
    try std.testing.expectEqual(@as(usize, 6 + 36 + 216), sequences);
}

test "select: equal logits go to the lower id and the weights are the host rule's bits" {
    const a = std.testing.allocator;
    var r = try rig(a);
    defer r.deinit(a);
    const s = r.stage;
    const rows = 4;
    const logits = s.get(.logits).slice(f32, rows * cols);
    const cases = [rows][experts + 1]f32{
        .{ 1, 1, 1, 1, 1, 1, 1, 1, 0 },
        .{ -3, 2, 2, -1, 2, 0.5, 0.5, -8, -2.5 },
        .{ 0.25, 7, -7, 7, 3, 3, 3, 3, 9 },
        .{ -0.0, 0.0, -1e-30, 1e-30, 4, 4, -4, 0, -0.001 },
    };
    for (cases, 0..) |row, i| @memcpy(logits[i * cols ..][0 .. experts + 1], &row);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = r.queue.commandBuffer();
    const e = cb.compute(.serial);
    e.setPipeline(s.pipe(.select));
    e.setBuffer(s.get(.logits), 0, 0);
    e.setBuffer(s.get(.picks), 0, 1);
    e.setBuffer(s.get(.wts), 0, 2);
    e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
    e.end();
    try wait(cb);
    var picks: [rows * topk]i32 = undefined;
    var wts: [rows * ws]f32 = undefined;
    try hostSelect(r.device, r.queue, logits, rows, &picks, &wts);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 1, 2, 1, 3, 4, 5 }, &picks);
    try std.testing.expectEqualSlices(i32, &picks, s.get(.picks).slice(i32, rows * topk));
    try std.testing.expectEqualSlices(u32, @ptrCast(wts[0..]), s.get(.wts).slice(u32, rows * ws));
}
