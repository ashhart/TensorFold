//! Attention rows against a float64 softmax of the same bf16 inputs; --prompt: 128-row chunks on both paths.
const std = @import("std");
const mtl = @import("metal");
const q = @import("qwen27");
const attn = q.attention;

const heads = 24;
const kv_heads = 4;
const dim = 256;
const group = heads / kv_heads;
const capacity: u32 = 12288;
const most_rows = 128;
/// Where a row's key chunk changes in 1.0.13 (128 keys below it, 512 from it); rows from it compare across builds.
const split_at: u32 = 8192;

const Rng = struct {
    s: u64,
    fn next(r: *Rng) u64 {
        r.s ^= r.s >> 12;
        r.s ^= r.s << 25;
        r.s ^= r.s >> 27;
        return r.s *% 0x2545f4914f6cdd1d;
    }
    /// A bf16 in [-width, width): the upper half of an f32.
    fn bf16(r: *Rng, width: f32) u16 {
        const unit: f32 = @as(f32, @floatFromInt(r.next() >> 40)) / @as(f32, 1 << 24);
        return @intCast(@as(u32, @bitCast((unit - 0.5) * 2.0 * width)) >> 16);
    }
};

fn value(x: u16) f64 {
    return @as(f32, @bitCast(@as(u32, x) << 16));
}

/// Chains of `rows` rows from `start`; one-row windows are a decode step, the longer ones drafted windows.
const Case = struct { start: u32, rows: u32 };
const cases = [_]Case{
    .{ .start = 0, .rows = 1 },     .{ .start = 1, .rows = 1 },     .{ .start = 63, .rows = 1 },    .{ .start = 127, .rows = 1 },
    .{ .start = 128, .rows = 1 },   .{ .start = 511, .rows = 1 },   .{ .start = 512, .rows = 1 },   .{ .start = 1000, .rows = 1 },
    .{ .start = 2047, .rows = 1 },  .{ .start = 4095, .rows = 1 },  .{ .start = 6000, .rows = 1 },  .{ .start = 8063, .rows = 1 },
    .{ .start = 8190, .rows = 1 },  .{ .start = 8191, .rows = 1 },  .{ .start = 100, .rows = 8 },   .{ .start = 600, .rows = 16 },
    .{ .start = 4000, .rows = 8 },  .{ .start = 8176, .rows = 16 }, .{ .start = 8184, .rows = 16 }, .{ .start = 8192, .rows = 1 },
    .{ .start = 8193, .rows = 1 },  .{ .start = 8703, .rows = 1 },  .{ .start = 8704, .rows = 1 },  .{ .start = 10000, .rows = 1 },
    .{ .start = 12000, .rows = 1 }, .{ .start = 9000, .rows = 16 },
};
/// Prompt chunks: from the first key, ending inside a 32-key step, across 8192, and ending at the cache's last key.
const prompt_cases = [_]Case{ .{ .start = 0, .rows = 128 }, .{ .start = 1000, .rows = 128 }, .{ .start = 4000, .rows = 128 }, .{ .start = 8100, .rows = 128 }, .{ .start = 11000, .rows = 128 }, .{ .start = 12160, .rows = 128 } };

/// Every compared value on one side of `split_at`: how many, the largest and summed errors, the output's bytes.
const Tally = struct {
    n: usize = 0,
    max: f64 = 0,
    sum: f64 = 0,
    bad: usize = 0, // a NaN or an infinity the kernel wrote
    bytes: std.crypto.hash.sha2.Sha256 = .init(.{}),

    fn add(t: *Tally, got: u16, want: f64) void {
        const x = value(got);
        if (!std.math.isFinite(x)) {
            t.bad += 1;
            return;
        }
        const e = @abs(x - want);
        t.n += 1;
        t.max = @max(t.max, e);
        t.sum += e;
    }
    fn print(t: *Tally, label: []const u8) void {
        var digest: [32]u8 = undefined;
        t.bytes.final(&digest);
        std.debug.print("  {s}: {d} values, max abs error {e:.4}, mean abs error {e:.4}, {d} not finite, output sha256 {s}\n", .{ label, t.n, t.max, t.sum / @as(f64, @floatFromInt(@max(t.n, 1))), t.bad, std.fmt.bytesToHex(digest[0..8].*, .lower) });
    }
};

/// One row's float64 attention for query head `h` over keys 0..position (its own key last), as the kernels define it.
fn reference(keys: []const u16, values: []const u16, query: []const u16, position: u32, h: usize, scores: []f64, out: *[dim]f64) void {
    const g = h / group;
    const qv = query[h * dim ..][0..dim];
    var top = -std.math.inf(f64);
    for (scores[0 .. position + 1], 0..) |*s, j| {
        var dot: f64 = 0;
        for (qv, keys[(g * capacity + j) * dim ..][0..dim]) |x, y| dot += value(x) * value(y);
        s.* = dot * 0.0625;
        top = @max(top, s.*);
    }
    var total: f64 = 0;
    @memset(out, 0);
    for (scores[0 .. position + 1], 0..) |s, j| {
        const w = @exp(s - top);
        total += w;
        for (out, values[(g * capacity + j) * dim ..][0..dim]) |*o, v| o.* += w * value(v);
    }
    for (out) |*o| o.* /= total;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    const units = device.tensorUnits();
    const args = try init.minimal.args.toSlice(a);
    const prompt = args.len > 1 and std.mem.eql(u8, args[1], "--prompt");
    // the engine's own build of the kernels: before M5 the tail kernel's results read in the simdgroup layout
    if (!units) try q.simd_attention.check(device, queue);
    const library = try mtl.Library.fromSource(device, try q.simd_attention.library(a, units, true), mtl.CompileOptions.mlx());
    defer library.deinit();
    var op = try attn.Attention.init(device, library);
    defer op.deinit();
    const limits = attn.Limits{ .rows = most_rows, .streams = 1, .keys = capacity, .shared_prefix = units };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();
    var binding = try attn.Binding.init(device, limits);
    defer binding.deinit();
    const options = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const cache_bytes = @as(usize, capacity) * kv_heads * dim * 2;
    const keys = try device.buffer(cache_bytes, options);
    const values = try device.buffer(cache_bytes, options);
    const query = try device.buffer(most_rows * heads * dim * 2, options);
    const key = try device.buffer(most_rows * kv_heads * dim * 2, options);
    const val = try device.buffer(most_rows * kv_heads * dim * 2, options);
    const out = try device.buffer(most_rows * heads * dim * 2, options);
    defer for ([_]mtl.Buffer{ keys, values, query, key, val, out }) |b| b.deinit();
    const caches = [1]attn.Cache{.{ .keys = .{ .buffer = keys }, .values = .{ .buffer = values }, .capacity = capacity }};
    const scores = try a.alloc(f64, capacity);
    const wants = try a.alloc([dim]f64, most_rows * heads); // a case's reference, kept for its second pass
    var parents: [most_rows]i32 = undefined;
    for (&parents, 0..) |*p, r| p.* = @as(i32, @intCast(r)) - 1;
    std.debug.print("attention_precision on {s}, tensor units {s}: {s} rows against float64 softmax of the same bf16 rows\n", .{ std.mem.span(device.name()), if (units) "yes" else "no", if (prompt) "prompt" else "decode" });
    // queries of width 1 give mild softmaxes, width 4 peaked ones whose small probabilities test the chunks' rounding
    for ([_]f32{ 1, 4 }) |width| {
        var rng = Rng{ .s = 0x9e3779b97f4a7c15 };
        for (keys.slice(u16, cache_bytes / 2), values.slice(u16, cache_bytes / 2)) |*k, *v| {
            k.* = rng.bf16(2);
            v.* = rng.bf16(2);
        }
        var below: Tally = .{};
        var from: Tally = .{};
        var kernel: Tally = .{};
        var tree: Tally = .{};
        for (if (prompt) &prompt_cases else &cases) |c| for (0..@as(usize, if (prompt) 2 else 1)) |pass| {
            if (pass == 0) {
                for (query.slice(u16, c.rows * heads * dim)) |*x| x.* = rng.bf16(width);
            }
            // the window's own key and value are the cache's rows at its positions, so appending them changes nothing
            for (0..c.rows) |r| for (0..kv_heads) |g| {
                const at = (g * capacity + c.start + r) * dim;
                @memcpy(key.slice(u16, most_rows * kv_heads * dim)[(r * kv_heads + g) * dim ..][0..dim], keys.slice(u16, cache_bytes / 2)[at..][0..dim]);
                @memcpy(val.slice(u16, most_rows * kv_heads * dim)[(r * kv_heads + g) * dim ..][0..dim], values.slice(u16, cache_bytes / 2)[at..][0..dim]);
            };
            @memset(out.slice(u16, most_rows * heads * dim), 0xffff);
            var plan = try attn.Plan.init(a, &.{.{ .parents = parents[0..c.rows], .start = c.start, .capacity = capacity }}, .{ .shared_prefix = units });
            defer plan.deinit();
            try scratch.upload(plan);
            try binding.upload(plan, &caches);
            const cb = queue.commandBuffer();
            const e = cb.compute(.concurrent);
            try op.encode(e, &scratch, &binding, plan, .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = val }, .{ .buffer = out }, &caches, null, prompt and pass == 0);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.debug.print("FAIL the GPU at start {d}, {d} rows: {s}\n", .{ c.start, c.rows, msg });
                return error.GpuFailed;
            }
            const got = out.slice(u16, c.rows * heads * dim);
            for (0..c.rows) |r| {
                const position = c.start + @as(u32, @intCast(r));
                const t = if (prompt) (if (pass == 0) &kernel else &tree) else if (position < split_at) &below else &from;
                t.bytes.update(std.mem.sliceAsBytes(got[r * heads * dim ..][0 .. heads * dim]));
                for (0..heads) |h| {
                    const want = &wants[r * heads + h];
                    if (pass == 0) reference(keys.slice(u16, cache_bytes / 2), values.slice(u16, cache_bytes / 2), query.slice(u16, c.rows * heads * dim)[r * heads * dim ..], position, h, scores, want);
                    for (got[(r * heads + h) * dim ..][0..dim], want) |x, y| t.add(x, y);
                }
            }
        };
        std.debug.print("queries in [-{d}, {d}):\n", .{ width, width });
        if (prompt) {
            kernel.print("prompt kernel");
            tree.print("decode path");
        } else {
            below.print("rows below 8192");
            from.print("rows from 8192");
        }
        if (below.bad + from.bad + kernel.bad + tree.bad > 0) return error.NotFinite;
    }
}
