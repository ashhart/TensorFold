//! Block-bidirectional prompt attention (the 9B's GQA 4) against a float64 CPU reference over every key before `end`.
const std = @import("std");
const mtl = @import("metal");
const q = @import("qwen27");
const attn = q.attention;
const planning = q.attention_plan;

const capacity: u32 = 1024;
const max_rows: u32 = 128;
const heads: u32 = 16;
const kv_heads: u32 = 4;
const dim: u32 = 256;
const group = heads / kv_heads;
/// Worst |GPU - reference| allowed on outputs of size about 1: half probabilities and bf16 outputs.
const tolerance: f64 = 0.02;
/// The other mask's reference must miss by more than this, or the case cannot tell the two masks apart.
const apart: f64 = 0.05;

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s >> 12;
        r.s ^= r.s << 25;
        r.s ^= r.s >> 27;
        return r.s *% 0x2545f4914f6cdd1d;
    }
    /// A bf16 in [-2, 2): the upper half of an f32.
    fn bf16(r: *Rng) u16 {
        const unit: f32 = @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
        return @intCast(@as(u32, @bitCast((unit - 0.5) * 4.0)) >> 16);
    }
};

fn float(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}

fn filled(device: mtl.Device, rng: *Rng, bytes: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    for (b.slice(u16, bytes / 2)) |*w| w.* = rng.bf16();
    return b;
}

/// Rows `start..start+rows` attend keys [0, end) when `bidirectional`, else each its own causal count.
const Case = struct { start: u32, rows: u32, end: u32 };

/// Worst |out - reference| of `out` (rows x heads x dim bf16) against a float64 softmax over the cache's keys.
fn worst(out: []const u16, keys: []const u16, values: []const u16, query: []const u16, c: Case, bidirectional: bool) f64 {
    var max: f64 = 0;
    var scores: [capacity]f64 = undefined;
    for (0..c.rows) |r| for (0..heads) |h| {
        const kh = h / group;
        const n: usize = if (bidirectional) c.end else c.start + r + 1;
        const qrow = query[(r * heads + h) * dim ..][0..dim];
        var peak: f64 = -std.math.inf(f64);
        for (0..n) |key| {
            const krow = keys[(kh * capacity + key) * dim ..][0..dim];
            var dot: f64 = 0;
            for (qrow, krow) |x, y| dot += float(x) * float(y);
            scores[key] = dot / 16.0;
            peak = @max(peak, scores[key]);
        }
        var z: f64 = 0;
        for (scores[0..n]) |*s| {
            s.* = @exp(s.* - peak);
            z += s.*;
        }
        for (0..dim) |d| {
            var acc: f64 = 0;
            for (0..n) |key| acc += scores[key] * float(values[(kh * capacity + key) * dim + d]);
            max = @max(max, @abs(acc / z - float(out[(r * heads + h) * dim + d])));
        }
    };
    return max;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    const units = device.tensorUnits();
    const text = try q.simd_attention.grouped(a, units, true, group, kv_heads);
    const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer library.deinit();
    var op = try attn.Attention.init(device, library);
    defer op.deinit();
    if (op.prompt == null) {
        std.debug.print("attention_block: no prompt attention kernel on {s}; bidirectional blocks are refused here\n", .{std.mem.span(device.name())});
        return error.NoPromptKernel;
    }
    const geometry = planning.Geometry{ .heads = heads, .kv_heads = kv_heads, .shared_prefix = units };
    const limits = attn.Limits{ .rows = max_rows, .streams = 1, .keys = capacity, .shared_prefix = units, .geometry = geometry };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();
    var binding = try attn.Binding.init(device, limits);
    defer binding.deinit();
    var rng = Rng{ .s = 0x9e3779b97f4a7c15 };
    const cache_bytes = @as(usize, capacity) * kv_heads * dim * 2;
    const keys = try filled(device, &rng, cache_bytes);
    const values = try filled(device, &rng, cache_bytes);
    const query = try filled(device, &rng, max_rows * heads * dim * 2);
    const key = try filled(device, &rng, max_rows * kv_heads * dim * 2);
    const value = try filled(device, &rng, max_rows * kv_heads * dim * 2);
    const out = try device.buffer(max_rows * heads * dim * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    const out_old = try device.buffer(max_rows * heads * dim * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    defer for ([_]mtl.Buffer{ keys, values, query, key, value, out, out_old }) |b| b.deinit();
    const caches = [_]attn.Cache{.{ .keys = .{ .buffer = keys }, .values = .{ .buffer = values }, .capacity = capacity }};
    // one-pass blocks, blocks across a 512-key chunk, and layer-major chunks whose block runs past their own rows
    const cases = [_]Case{ .{ .start = 0, .rows = 1, .end = 1 }, .{ .start = 0, .rows = 7, .end = 7 }, .{ .start = 0, .rows = 128, .end = 128 }, .{ .start = 37, .rows = 50, .end = 87 }, .{ .start = 490, .rows = 40, .end = 530 }, .{ .start = 100, .rows = 40, .end = 300 }, .{ .start = 640, .rows = 128, .end = 1000 }, .{ .start = 896, .rows = 128, .end = 1024 } };
    var failed = false;
    for (cases) |c| {
        var parents: [max_rows]i32 = undefined;
        for (parents[0..c.rows], 0..) |*p, r| p.* = @as(i32, @intCast(r)) - 1;
        var plan = try attn.Plan.init(a, &.{.{ .parents = parents[0..c.rows], .start = c.start, .capacity = capacity }}, geometry);
        defer plan.deinit();
        const n = @as(usize, c.rows) * heads * dim;
        var errs: [2]f64 = undefined;
        var crossed: [2]f64 = undefined;
        for ([_]bool{ true, false }, 0..) |bidirectional, i| {
            try scratch.upload(plan);
            try binding.upload(plan, &caches);
            @memset(out.contents()[0 .. n * 2], 0);
            const cb = queue.commandBuffer();
            const e = cb.compute(.concurrent);
            try op.encodeTo(e, &scratch, &binding, plan, .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out }, &caches, null, true, false, if (bidirectional) c.end else 0);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure() != null) return error.GpuFailed;
            const o = out.slice(u16, n);
            errs[i] = worst(o, keys.slice(u16, cache_bytes / 2), values.slice(u16, cache_bytes / 2), query.slice(u16, n), c, bidirectional);
            crossed[i] = worst(o, keys.slice(u16, cache_bytes / 2), values.slice(u16, cache_bytes / 2), query.slice(u16, n), c, !bidirectional);
            if (!bidirectional) {
                // the causal path through `encode` is the same bytes as `encodeTo` with end 0
                const cb2 = queue.commandBuffer();
                const e2 = cb2.compute(.concurrent);
                try op.encode(e2, &scratch, &binding, plan, .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out_old }, &caches, null, true, false);
                e2.end();
                cb2.commit();
                cb2.wait();
                if (cb2.failure() != null) return error.GpuFailed;
                if (!std.mem.eql(u8, out.contents()[0 .. n * 2], out_old.contents()[0 .. n * 2])) {
                    std.debug.print("FAIL causal encode and encodeTo(end 0) differ at start {d} rows {d}\n", .{ c.start, c.rows });
                    failed = true;
                }
            }
        }
        // a case whose block is its whole key range has both masks equal only for its last row: one row alone
        const discriminates = c.rows > 1 or c.end > c.start + c.rows;
        const ok = errs[0] <= tolerance and errs[1] <= tolerance and (!discriminates or (crossed[0] > apart and crossed[1] > apart));
        if (!ok) failed = true;
        std.debug.print("{s} start {d} rows {d} end {d}: bidirectional worst {d:.5} (vs causal reference {d:.4}), causal worst {d:.5} (vs bidirectional reference {d:.4})\n", .{ if (ok) "ok  " else "FAIL", c.start, c.rows, c.end, errs[0], crossed[0], errs[1], crossed[1] });
    }
    if (failed) return error.BidirectionalAttention;
    std.debug.print("attention_block: {d} cases on {s} ({s} prompt kernel) within {d} of float64\n", .{ cases.len, std.mem.span(device.name()), if (units) "tensor-unit" else "simdgroup", tolerance });
}
