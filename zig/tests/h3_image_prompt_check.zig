//! tf-h3-image-prompt selftest's checks: each prompt kernel alone against the float64 reference, and the gates.
const std = @import("std");
const mtl = @import("metal");
const hp = @import("h3_prompt");
const bf16_round = @import("bf16_round");

const rules = hp.rules;
const ref = hp.reference;
const Ref = hp.tensors.Ref;

/// The selftest's gates, fixed before any run.
pub const gates = struct {
    /// Each kernel alone against float64: this share bit-equal, no value further than this from it by its row's RMS.
    pub const alone_equal: f64 = 0.999;
    pub const alone_worst: f64 = 1.0 / 32.0;
    /// Whole runs against float64: 1 - cosine and relative error at most twice a float32-sum host run's, cosine 0.999.
    pub const noise_factor: f64 = 2;
    pub const cosine_floor: f64 = 0.999;
};

const shared = @import("h3_image_match.zig");
pub const Match = shared.Match;
pub const match = shared.match;
pub const report = shared.report;
pub const bf16s = shared.bf16s;

/// A whole run `m` beside the float32-sum host run's `noise`, both against float64.
pub fn rowsPass(m: Match, noise: Match) bool {
    return shared.within(m, noise, gates.noise_factor, gates.cosine_floor);
}

/// Shared buffers filled with bf16 draws, their values kept as fp32 for the reference.
const Bench = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    rand: std.Random,
    made: std.ArrayList(mtl.Buffer) = .empty,

    const Data = struct { ref: Ref, vals: []f32 };

    fn data(b: *Bench, n: usize, center: f32, spread: f32) !Data {
        const buf = try b.device.buffer(n * 2, mtl.ResourceOptions.shared);
        try b.made.append(b.a, buf);
        const vals = try b.a.alloc(f32, n);
        for (vals, buf.slice(u16, n)) |*v, *h| {
            h.* = bf16_round.bits(center + spread * (2 * b.rand.float(f32) - 1));
            v.* = bf16_round.value(h.*);
        }
        return .{ .ref = .{ .buffer = buf }, .vals = vals };
    }

    fn floats(b: *Bench, vals: []const f32) !Ref {
        const buf = try b.device.buffer(vals.len * 4, mtl.ResourceOptions.shared);
        try b.made.append(b.a, buf);
        @memcpy(buf.slice(f32, vals.len), vals);
        return .{ .buffer = buf };
    }

    fn deinit(b: *Bench) void {
        for (b.made.items) |buf| buf.deinit();
    }
};

fn bits(r: Ref, n: usize) []const u16 {
    return @as([*]const u16, @ptrCast(@alignCast(r.buffer.contents() + r.offset)))[0..n];
}

fn submit(runner: *hp.Runner, comptime encode: anytype, args: anytype) !void {
    const cb = runner.queue.commandBuffer();
    const e = cb.compute(.serial);
    encode(runner, e, args);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.debug.print("a kernel failed on the GPU: {s}\n", .{msg});
        return error.GpuFailed;
    }
}

fn gate(label: []const u8, ours: []const u16, theirs: []const f32, width: usize) bool {
    const m = match(ours, theirs, width);
    const pass = m.equal >= gates.alone_equal and m.worst <= gates.alone_worst;
    report(label, m, pass);
    return pass;
}

/// Each kernel alone on the same bf16 draws as the reference: products, norms, heads, attention, gates, residual add.
pub fn kernelsAlone(runner: *hp.Runner, a: std.mem.Allocator, rand: std.Random) !bool {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    var b = Bench{ .a = a, .device = runner.device, .rand = rand };
    defer b.deinit();
    const T = 70;
    const D = rules.head_dim;
    var pass = true;

    // the product on ragged rows with a bias
    const x = try b.data(T * 288, 0, 1);
    const w = try b.data(320 * 288, 0, 1);
    const bias = try b.data(320, 0, 1);
    const y = try b.data(T * 320, 0, 0);
    try submit(runner, struct {
        fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
            r.k.gemm(e, p[0], p[1], p[2], p[3], 70, 320, 288);
        }
    }.f, .{ x.ref, w.ref, bias.ref, y.ref });
    ref.gemm(f64, x.vals, w.vals, bias.vals, y.vals, T, 320, 288);
    pass = gate("product alone, 70 x 320 x 288 with a bias", bits(y.ref, y.vals.len), y.vals, 320) and pass;

    // both norms' roundings
    inline for (.{ .{ hp.kernels.Mode.qwen, ref.Mode.qwen, "RMSNorm as the text model rounds" }, .{ hp.kernels.Mode.h3, ref.Mode.h3, "RMSNorm as H3's blocks round" } }) |m| {
        const nx = try b.data(T * 256, 0, 2);
        const nw = try b.data(256, 1, 0.2);
        const ny = try b.data(T * 256, 0, 0);
        try submit(runner, struct {
            fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
                r.k.norm(e, p[0], p[1], p[2], 70, 256, 1e-6, m[0]);
            }
        }.f, .{ nx.ref, nw.ref, ny.ref });
        ref.norm(f64, nx.vals, nw.vals, ny.vals, 256, 1e-6, m[1]);
        pass = gate(m[2], bits(ny.ref, ny.vals.len), ny.vals, 256) and pass;
    }

    // the text model's heads: 4 query and 2 KV heads normed and turned, then causal attention over them
    const q = try b.data(T * 4 * D, 0, 2);
    const k = try b.data(T * 2 * D, 0, 2);
    const v = try b.data(T * 2 * D, 0, 2);
    const wq = try b.data(D, 1, 0.2);
    const wk = try b.data(D, 1, 0.2);
    const pos = try a.alloc([3]i32, T);
    for (pos, 0..) |*p, i| p.* = @splat(@intCast(3 * i));
    const cs = try a.alloc(f32, T * 64);
    const sn = try a.alloc(f32, T * 64);
    rules.rotary(cs, sn, pos, 1e4, .{ 24, 20, 20 });
    const rot = [2]Ref{ try b.floats(cs), try b.floats(sn) };
    const o = try b.data(T * 4 * D, 0, 0);
    try submit(runner, struct {
        fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
            r.k.heads(e, p[0], .{ .count = 4, .row = 4 * 128, .head = 128 }, p[1], .{ .count = 2, .row = 2 * 128, .head = 128 }, p[2], p[3], p[4], 70, 1e-6, .qwen);
        }
    }.f, .{ q.ref, k.ref, wq.ref, wk.ref, rot });
    ref.heads(f64, q.vals, 4, 4 * D, D, wq.vals, .{ cs, sn }, T, 1e-6, .qwen);
    ref.heads(f64, k.vals, 2, 2 * D, D, wk.vals, .{ cs, sn }, T, 1e-6, .qwen);
    pass = gate("query heads normed and turned", bits(q.ref, q.vals.len), q.vals, D) and pass;
    pass = gate("key heads normed and turned", bits(k.ref, k.vals.len), k.vals, D) and pass;
    @memcpy(@constCast(bits(q.ref, q.vals.len)), try bf16s(a, q.vals));
    @memcpy(@constCast(bits(k.ref, k.vals.len)), try bf16s(a, k.vals));
    try submit(runner, struct {
        fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
            r.k.attention(e, p[0], .{ .count = 4, .row = 4 * 128, .head = 128 }, p[1], p[2], .{ .count = 2, .row = 2 * 128, .head = 128 }, p[3], 70, true);
        }
    }.f, .{ q.ref, k.ref, v.ref, o.ref });
    try ref.attention(f64, a, q.vals, .{ .count = 4, .row = 4 * D, .head = D }, k.vals, v.vals, .{ .count = 2, .row = 2 * D, .head = D }, o.vals, T, true);
    pass = gate("causal attention, 4 query heads over 2 KV heads", bits(o.ref, o.vals.len), o.vals, D) and pass;

    // the refiner's fused rows: 2 heads of [q | k | v], q and k normed in place, then every row to every row
    const qkv = try b.data(T * 2 * 3 * D, 0, 2);
    const fo = try b.data(T * 2 * D, 0, 0);
    const fused: hp.kernels.Heads = .{ .count = 2, .row = 6 * D, .head = 3 * D };
    const host: ref.Heads = .{ .count = 2, .row = 6 * D, .head = 3 * D };
    try submit(runner, struct {
        fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
            r.k.heads(e, p[0], p[1], shifted(p[0], 1), p[1], p[2], p[3], null, 70, 1e-5, .h3);
        }
    }.f, .{ qkv.ref, fused, wq.ref, wk.ref });
    ref.heads(f64, qkv.vals, 2, 6 * D, 3 * D, wq.vals, null, T, 1e-5, .h3);
    ref.heads(f64, qkv.vals[D..], 2, 6 * D, 3 * D, wk.vals, null, T, 1e-5, .h3);
    pass = gate("fused q and k heads normed as H3's blocks", bits(qkv.ref, qkv.vals.len), qkv.vals, 3 * D) and pass;
    @memcpy(@constCast(bits(qkv.ref, qkv.vals.len)), try bf16s(a, qkv.vals));
    try submit(runner, struct {
        fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
            r.k.attention(e, p[0], p[1], shifted(p[0], 1), shifted(p[0], 2), p[1], p[2], 70, false);
        }
    }.f, .{ qkv.ref, fused, fo.ref });
    try ref.attention(f64, a, qkv.vals, host, qkv.vals[D..], qkv.vals[2 * D ..], host, fo.vals, T, false);
    pass = gate("attention over every row, fused heads", bits(fo.ref, fo.vals.len), fo.vals, D) and pass;

    // both gated units' roundings, then the residual add
    inline for (.{ .{ hp.kernels.Mode.qwen, ref.Mode.qwen, "SiLU gate as the text model rounds" }, .{ hp.kernels.Mode.h3, ref.Mode.h3, "SiLU gate as H3's blocks round" } }) |m| {
        const wide = try b.data(T * 1024, 0, 4);
        const act = try b.data(T * 512, 0, 0);
        try submit(runner, struct {
            fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
                r.k.glu(e, p[0], .{ .buffer = p[0].buffer, .offset = 512 * 2 }, p[1], 70, 512, 1024, 1024, m[0]);
            }
        }.f, .{ wide.ref, act.ref });
        ref.glu(f64, wide.vals, wide.vals[512..], act.vals, T, 512, 1024, 1024, m[1]);
        pass = gate(m[2], bits(act.ref, act.vals.len), act.vals, 512) and pass;
    }
    const rx = try b.data(T * 256, 0, 2);
    const ry = try b.data(T * 256, 0, 2);
    try submit(runner, struct {
        fn f(r: *hp.Runner, e: mtl.ComputeEncoder, p: anytype) void {
            r.k.add(e, p[0], p[1], 70 * 256);
        }
    }.f, .{ rx.ref, ry.ref });
    ref.add(rx.vals, ry.vals);
    pass = gate("residual add", bits(rx.ref, rx.vals.len), rx.vals, 256) and pass;
    return pass;
}

/// `s` moved on by `n` heads of 128 bf16 values.
fn shifted(s: Ref, n: usize) Ref {
    return .{ .buffer = s.buffer, .offset = s.offset + n * 256 };
}
