//! Chains, trees and mixed streams through tree-round attention: a fused chain equals the tail kernel byte for byte.
const std = @import("std");
const mtl = @import("metal");
const source = @import("qwen_runtime_sources");
const q = @import("qwen27");
const attn = q.attention;
const planning = q.attention_plan;
const Ref = q.projection.Ref;

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
        const f: f32 = (unit - 0.5) * 4.0;
        return @intCast(@as(u32, @bitCast(f)) >> 16);
    }
    fn range(r: *Rng, n: u32) u32 {
        return @intCast((r.next() >> 33) % n);
    }
};

const Case = struct { streams: u32, rows: [4]u32, starts: [4]u32, tree: bool = false };
const capacity: u32 = 4096;
const max_rows: u32 = 40;

fn buffer(device: mtl.Device, rng: *Rng, bytes: usize, fill: bool) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    const words = b.slice(u16, bytes / 2);
    if (fill) {
        for (words) |*w| w.* = rng.bf16();
    } else @memset(words, 0);
    return b;
}

fn run(queue: mtl.Queue, op: *const attn.Attention, scratch: *attn.Scratch, binding: *attn.Binding, plan: attn.Plan, caches: []const attn.Cache, query: Ref, key: Ref, value: Ref, out: Ref, gate: ?Ref) !void {
    try scratch.upload(plan);
    try binding.upload(plan, caches);
    const cb = queue.commandBuffer();
    const e = cb.compute(.concurrent);
    try op.encode(e, scratch, binding, plan, query, key, value, out, caches, gate, false);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.GpuFailed;
}

/// A chain's attention over 16 layers of one command buffer, each with its own cache, timed per layer at a context.
fn bench(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, op: *const attn.Attention, rng: *Rng, context: u32, rows: u32) !void {
    const layers = 16;
    const cap = context + 64;
    const limits = attn.Limits{ .rows = rows, .streams = 1, .keys = cap };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();
    var bindings: [layers]attn.Binding = undefined;
    var caches: [layers]attn.Cache = undefined;
    var bufs: [2 * layers]mtl.Buffer = undefined;
    for (0..layers) |i| {
        const bytes = @as(usize, cap) * 4 * 256 * 2;
        bufs[2 * i] = try buffer(device, rng, bytes, true);
        bufs[2 * i + 1] = try buffer(device, rng, bytes, true);
        caches[i] = .{ .keys = .{ .buffer = bufs[2 * i] }, .values = .{ .buffer = bufs[2 * i + 1] }, .capacity = cap };
        bindings[i] = try attn.Binding.init(device, limits);
    }
    defer for (&bindings) |*b| b.deinit();
    defer for (bufs) |b| b.deinit();
    const query = try buffer(device, rng, rows * 6144 * 2, true);
    const key = try buffer(device, rng, rows * 1024 * 2, true);
    const value = try buffer(device, rng, rows * 1024 * 2, true);
    const raw_gate = try buffer(device, rng, rows * 12288 * 2, true);
    const out = try buffer(device, rng, rows * 6144 * 2, false);
    defer for ([_]mtl.Buffer{ query, key, value, raw_gate, out }) |b| b.deinit();
    var parents: [max_rows]i32 = undefined;
    for (parents[0..rows], 0..) |*p, r| p.* = @as(i32, @intCast(r)) - 1;
    var plan = try attn.Plan.init(a, &.{.{ .parents = parents[0..rows], .start = context, .capacity = cap }}, .{});
    defer plan.deinit();
    try scratch.upload(plan);
    for (0..layers) |i| try bindings[i].upload(plan, caches[i .. i + 1]);
    var times: [9]f64 = undefined;
    for (&times) |*t| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.concurrent);
        for (0..layers) |i| try op.encode(e, &scratch, &bindings[i], plan, .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out }, caches[i .. i + 1], .{ .buffer = raw_gate }, false);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.GpuFailed;
        t.* = cb.gpuSeconds() * 1e6 / layers;
    }
    std.mem.sort(f64, &times, {}, std.sort.asc(f64));
    std.debug.print("attention bench: rows {d} context {d}: {d:.1} us per layer (median of 9; min {d:.1}, max {d:.1})\n", .{ rows, context, times[4], times[0], times[8] });
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    if (!device.tensorUnits()) {
        std.debug.print("attention_fused: no tensor units on {s}, nothing to check\n", .{std.mem.span(device.name())});
        return;
    }
    const queue = try device.queue();
    const args = try init.minimal.args.toSlice(a);
    const text = try a.dupe(u8, source.glue ++ source.attention_mpp ++ source.attention_io ++ source.attention_prompt);
    const library = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
    defer library.deinit();
    var op = try attn.Attention.init(device, library, true);
    defer op.deinit();
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--bench")) {
        var brng = Rng{ .s = 0x1234567 };
        for ([_]u32{ 1, 4, 16, 32 }) |rows| for ([_]u32{ 2048, 16384 }) |context| try bench(a, device, queue, &op, &brng, context, rows);
        return;
    }
    const limits = attn.Limits{ .rows = max_rows, .streams = 4, .keys = capacity };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();
    var binding = try attn.Binding.init(device, limits);
    defer binding.deinit();
    // Chips without tensor units plan with shared_prefix off: no prefix kernel, every key through the tail kernel.
    const plain_limits = attn.Limits{ .rows = max_rows, .streams = 4, .keys = capacity, .shared_prefix = false };
    var plain_scratch = try attn.Scratch.init(device, plain_limits);
    defer plain_scratch.deinit();
    var plain_binding = try attn.Binding.init(device, plain_limits);
    defer plain_binding.deinit();
    var rng = Rng{ .s = 0x9e3779b97f4a7c15 };
    var cache_buffers: [8]mtl.Buffer = undefined;
    var caches: [4]attn.Cache = undefined;
    for (&caches, 0..) |*cache, i| {
        const bytes = @as(usize, capacity) * 4 * 256 * 2;
        cache_buffers[2 * i] = try buffer(device, &rng, bytes, true);
        cache_buffers[2 * i + 1] = try buffer(device, &rng, bytes, true);
        cache.* = .{ .keys = .{ .buffer = cache_buffers[2 * i] }, .values = .{ .buffer = cache_buffers[2 * i + 1] }, .capacity = capacity };
    }
    defer for (cache_buffers) |b| b.deinit();
    const query = try buffer(device, &rng, max_rows * 6144 * 2, true);
    const key = try buffer(device, &rng, max_rows * 1024 * 2, true);
    const value = try buffer(device, &rng, max_rows * 1024 * 2, true);
    const raw_gate = try buffer(device, &rng, max_rows * 12288 * 2, true);
    const out_chain = try buffer(device, &rng, max_rows * 6144 * 2, false);
    const out_tail = try buffer(device, &rng, max_rows * 6144 * 2, false);
    const out_plain = try buffer(device, &rng, max_rows * 6144 * 2, false);
    defer for ([_]mtl.Buffer{ query, key, value, raw_gate, out_chain, out_tail, out_plain }) |b| b.deinit();

    const edges = [_]u32{ 0, 1, 62, 63, 64, 65, 127, 128, 191, 255, 256, 447, 448, 449, 510, 511, 512, 513, 575, 576, 1000, 1023, 1024, 1087, 2047, 2048, 3000, 4000 };
    var cases: std.ArrayList(Case) = .empty;
    for (edges) |start| for ([_]u32{ 1, 2, 3, 5, 6, 7, 16, 17, 40 }) |rows| {
        if (start + rows > capacity) continue;
        try cases.append(a, .{ .streams = 1, .rows = .{ rows, 0, 0, 0 }, .starts = .{ start, 0, 0, 0 } });
    };
    for (0..200) |_| {
        var c = Case{ .streams = 1 + rng.range(4), .rows = @splat(0), .starts = @splat(0) };
        var left: u32 = max_rows;
        for (0..c.streams) |s| {
            c.rows[s] = @min(1 + rng.range(12), left - (c.streams - @as(u32, @intCast(s)) - 1));
            left -= c.rows[s];
            c.starts[s] = if (rng.range(3) == 0) edges[rng.range(@intCast(edges.len))] else rng.range(capacity - c.rows[s] + 1);
            if (c.starts[s] + c.rows[s] > capacity) c.starts[s] = capacity - c.rows[s];
        }
        try cases.append(a, c);
    }
    for (0..100) |_| {
        var c = Case{ .streams = 1 + rng.range(3), .rows = @splat(0), .starts = @splat(0), .tree = true };
        for (0..c.streams) |s| {
            c.rows[s] = 2 + rng.range(12);
            c.starts[s] = if (rng.range(3) == 0) edges[rng.range(@intCast(edges.len))] else rng.range(capacity - c.rows[s] + 1);
            if (c.starts[s] + c.rows[s] > capacity) c.starts[s] = capacity - c.rows[s];
        }
        try cases.append(a, c);
    }
    var checked: usize = 0;
    var chains: usize = 0;
    var bytes_checked: usize = 0;
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    var plain_digest = std.crypto.hash.sha2.Sha256.init(.{});
    for (cases.items, 0..) |case, n| {
        var parents: [4][max_rows]i32 = undefined;
        var windows: [4]planning.Window = undefined;
        var total: u32 = 0;
        for (0..case.streams) |s| {
            for (0..case.rows[s]) |r| parents[s][r] = if (case.tree and r > 0 and rng.range(3) != 0) @intCast(rng.range(@intCast(r))) else @as(i32, @intCast(r)) - 1;
            if (case.tree and s == 0) parents[s][1] = 0;
            windows[s] = .{ .parents = parents[s][0..case.rows[s]], .start = case.starts[s], .capacity = capacity };
            total += case.rows[s];
        }
        var plan = try attn.Plan.init(a, windows[0..case.streams], .{});
        defer plan.deinit();
        const gate: ?Ref = if (n % 2 == 0) .{ .buffer = raw_gate } else null;
        try run(queue, &op, &scratch, &binding, plan, caches[0..case.streams], .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out_chain }, gate);
        const len = @as(usize, total) * 6144 * 2;
        if (std.mem.allEqual(u8, out_chain.contents()[0..len], 0)) return error.NothingWritten;
        digest.update(out_chain.contents()[0..len]);
        if (plan.fused) {
            var tailed = try attn.Plan.init(a, windows[0..case.streams], .{ .fuse_chain = false });
            defer tailed.deinit();
            if (tailed.fused) return error.NotFused;
            try run(queue, &op, &scratch, &binding, tailed, caches[0..case.streams], .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out_tail }, gate);
            if (!std.mem.eql(u8, out_chain.contents()[0..len], out_tail.contents()[0..len])) {
                std.debug.print("FAIL case {d}: streams {d} rows {any} starts {any}\n", .{ n, case.streams, case.rows, case.starts });
                for (out_chain.slice(u16, len / 2), out_tail.slice(u16, len / 2), 0..) |x, y, i| if (x != y) {
                    std.debug.print("first difference at element {d} (row {d}, head {d}): fused {x} tail {x}\n", .{ i, i / 6144, i % 6144 / 256, x, y });
                    break;
                };
                return error.FusedChainBits;
            }
            chains += 1;
        }
        var plain = try attn.Plan.init(a, windows[0..case.streams], .{ .shared_prefix = false });
        defer plain.deinit();
        try run(queue, &op, &plain_scratch, &plain_binding, plain, caches[0..case.streams], .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out_plain }, gate);
        if (std.mem.allEqual(u8, out_plain.contents()[0..len], 0)) return error.NothingWritten;
        plain_digest.update(out_plain.contents()[0..len]);
        checked += 1;
        bytes_checked += len;
    }
    var sum: [32]u8 = undefined;
    var plain_sum: [32]u8 = undefined;
    digest.final(&sum);
    plain_digest.final(&plain_sum);
    std.debug.print("attention_fused: {d} plans, {d} chains byte-identical through the fused prefix kernel and the tail kernel, {d} bytes, sha256 {s}\n", .{ checked, chains, bytes_checked, std.fmt.bytesToHex(sum, .lower) });
    std.debug.print("attention_fused: the same {d} plans without a shared prefix, sha256 {s}\n", .{ checked, std.fmt.bytesToHex(plain_sum, .lower) });
}
