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

const Case = struct { streams: u32, rows: [8]u32, starts: [8]u32, tree: bool = false };
const capacity: u32 = 4096;
/// Rows of a window and of a round in the first plans, whose inputs stay the first rows of the wider buffers.
const max_rows: u32 = 40;
/// Rows of a round of 8 streams with 16-row windows, the most a served host's slots send together.
const round_rows: u32 = 128;

fn buffer(device: mtl.Device, rng: *Rng, bytes: usize, fill: bool) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    const words = b.slice(u16, bytes / 2);
    if (fill) {
        for (words) |*w| w.* = rng.bf16();
    } else @memset(words, 0);
    return b;
}

/// A bf16 buffer whose first `head` bytes come from `rng` and the rest from `tail`, so growing it keeps the old inputs.
fn wideBuffer(device: mtl.Device, rng: *Rng, tail: *Rng, bytes: usize, head: usize) !mtl.Buffer {
    const b = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    const words = b.slice(u16, bytes / 2);
    for (words[0 .. head / 2]) |*w| w.* = rng.bf16();
    for (words[head / 2 ..]) |*w| w.* = tail.bf16();
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
fn bench(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, op: *const attn.Attention, rng: *Rng, context: u32, rows: u32, tree: bool) !void {
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
    for (parents[0..rows], 0..) |*p, r| p.* = if (tree and r > 1) @intCast(r / 2) else @as(i32, @intCast(r)) - 1; // a tree: row r under r/2
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
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(out.contents()[0 .. rows * 6144 * 2], &digest, .{});
    std.debug.print("attention bench: rows {d} context {d} {s}: {d:.1} us per layer (median of 9; min {d:.1}, max {d:.1}) out {x}\n", .{ rows, context, if (tree) "tree" else "chain", times[4], times[0], times[8], digest[0..6] });
}

/// What every plan runs on: scratch and bindings with and without a shared prefix, caches, inputs and outputs.
const Rig = struct {
    a: std.mem.Allocator,
    queue: mtl.Queue,
    op: *const attn.Attention,
    scratch: *attn.Scratch,
    binding: *attn.Binding,
    plain_scratch: *attn.Scratch,
    plain_binding: *attn.Binding,
    caches: []const attn.Cache,
    query: mtl.Buffer,
    key: mtl.Buffer,
    value: mtl.Buffer,
    raw_gate: mtl.Buffer,
    out_chain: mtl.Buffer,
    out_tail: mtl.Buffer,
    out_plain: mtl.Buffer,
    rng: *Rng,
};

const Totals = struct { plans: usize = 0, chains: usize = 0, bytes: usize = 0, sum: [32]u8 = undefined, plain_sum: [32]u8 = undefined };

/// An output that nothing wrote is all zeros; `rows` wants every row of 6144 bf16 written, not only some of them.
fn written(out: mtl.Buffer, len: usize, rows: bool) !void {
    const row_bytes = 6144 * 2;
    if (!rows) {
        if (std.mem.allEqual(u8, out.contents()[0..len], 0)) return error.NothingWritten;
        return;
    }
    for (0..len / row_bytes) |row| if (std.mem.allEqual(u8, out.contents()[row * row_bytes ..][0..row_bytes], 0)) return error.RowNotWritten;
}

/// Runs each case with a shared prefix, as a chain also through the tail kernel, and without a shared prefix.
fn check(r: *Rig, cases: []const Case, fresh: bool) !Totals {
    var t = Totals{};
    var digest = std.crypto.hash.sha2.Sha256.init(.{});
    var plain_digest = std.crypto.hash.sha2.Sha256.init(.{});
    for (cases, 0..) |case, n| {
        var parents: [8][max_rows]i32 = undefined;
        var windows: [8]planning.Window = undefined;
        var total: u32 = 0;
        for (0..case.streams) |s| {
            for (0..case.rows[s]) |row| parents[s][row] = if (case.tree and row > 0 and r.rng.range(3) != 0) @intCast(r.rng.range(@intCast(row))) else @as(i32, @intCast(row)) - 1;
            if (case.tree and s == 0) parents[s][1] = 0;
            windows[s] = .{ .parents = parents[s][0..case.rows[s]], .start = case.starts[s], .capacity = capacity };
            total += case.rows[s];
        }
        const len = @as(usize, total) * 6144 * 2;
        const caches = r.caches[0..case.streams];
        var plan = try attn.Plan.init(r.a, windows[0..case.streams], .{});
        defer plan.deinit();
        const gate: ?Ref = if (n % 2 == 0) .{ .buffer = r.raw_gate } else null;
        if (fresh) @memset(r.out_chain.contents()[0..len], 0);
        try run(r.queue, r.op, r.scratch, r.binding, plan, caches, .{ .buffer = r.query }, .{ .buffer = r.key }, .{ .buffer = r.value }, .{ .buffer = r.out_chain }, gate);
        try written(r.out_chain, len, fresh);
        digest.update(r.out_chain.contents()[0..len]);
        if (plan.fused) {
            var tailed = try attn.Plan.init(r.a, windows[0..case.streams], .{ .fuse_chain = false });
            defer tailed.deinit();
            if (tailed.fused) return error.NotFused;
            if (fresh) @memset(r.out_tail.contents()[0..len], 0);
            try run(r.queue, r.op, r.scratch, r.binding, tailed, caches, .{ .buffer = r.query }, .{ .buffer = r.key }, .{ .buffer = r.value }, .{ .buffer = r.out_tail }, gate);
            if (!std.mem.eql(u8, r.out_chain.contents()[0..len], r.out_tail.contents()[0..len])) {
                std.debug.print("FAIL case {d}: streams {d} rows {any} starts {any}\n", .{ n, case.streams, case.rows, case.starts });
                for (r.out_chain.slice(u16, len / 2), r.out_tail.slice(u16, len / 2), 0..) |x, y, i| if (x != y) {
                    std.debug.print("first difference at element {d} (row {d}, head {d}): fused {x} tail {x}\n", .{ i, i / 6144, i % 6144 / 256, x, y });
                    break;
                };
                return error.FusedChainBits;
            }
            t.chains += 1;
        }
        var plain = try attn.Plan.init(r.a, windows[0..case.streams], .{ .shared_prefix = false });
        defer plain.deinit();
        if (fresh) @memset(r.out_plain.contents()[0..len], 0);
        try run(r.queue, r.op, r.plain_scratch, r.plain_binding, plain, caches, .{ .buffer = r.query }, .{ .buffer = r.key }, .{ .buffer = r.value }, .{ .buffer = r.out_plain }, gate);
        try written(r.out_plain, len, fresh);
        plain_digest.update(r.out_plain.contents()[0..len]);
        t.plans += 1;
        t.bytes += len;
    }
    digest.final(&t.sum);
    plain_digest.final(&t.plain_sum);
    return t;
}

/// Chains of 5 to 8 streams with windows of up to 16 rows, the rounds a host's slots send together with --parallel.
fn wideCases(a: std.mem.Allocator, rng: *Rng, edges: []const u32) !std.ArrayList(Case) {
    var cases: std.ArrayList(Case) = .empty;
    // the same rows in every stream, all at 0, all at one tile or chunk edge, and each at an edge of its own
    for ([_]u32{ 5, 6, 7, 8 }) |streams| for ([_]u32{ 1, 2, 3, 5, 8, 15, 16 }) |rows| for (0..3) |pattern| {
        var c = Case{ .streams = streams, .rows = @splat(0), .starts = @splat(0) };
        for (0..streams) |s| {
            c.rows[s] = rows;
            const edge: u32 = switch (pattern) {
                0 => 0,
                1 => edges[(rows + streams) % edges.len],
                else => edges[(rows + streams + 5 * s) % edges.len],
            };
            c.starts[s] = @min(edge, capacity - rows);
        }
        try cases.append(a, c);
    };
    // windows of unequal rows, each across a tile or chunk edge
    const mixed = [8]u32{ 16, 1, 9, 2, 16, 3, 15, 7 };
    const crossings = [_]u32{ 64, 512, 576, 1024, 2048, 3584 };
    for ([_]u32{ 5, 6, 7, 8 }) |streams| for (0..mixed.len) |shift| {
        var c = Case{ .streams = streams, .rows = @splat(0), .starts = @splat(0) };
        for (0..streams) |s| {
            c.rows[s] = mixed[(s + shift) % mixed.len];
            c.starts[s] = crossings[(shift + 2 * s + streams) % crossings.len] - c.rows[s] / 2;
        }
        try cases.append(a, c);
    };
    // the fullest round, 8 windows of 16 rows, at the ends of the caches and across chunk edges
    for (0..8) |pattern| {
        var c = Case{ .streams = 8, .rows = @splat(16), .starts = @splat(0) };
        for (0..8) |s| {
            const edge: u32 = switch (pattern) {
                0 => 0,
                1 => capacity,
                2 => 512 * @as(u32, @intCast(s)),
                3 => 512 * @as(u32, @intCast(s)) + 504,
                4 => 64 * @as(u32, @intCast(s)) + 56,
                5 => 512 * (7 - @as(u32, @intCast(s))),
                6 => rng.range(capacity),
                else => edges[3 * s % edges.len],
            };
            c.starts[s] = @min(edge, capacity - 16);
        }
        try cases.append(a, c);
    }
    for (0..150) |_| {
        var c = Case{ .streams = 5 + rng.range(4), .rows = @splat(0), .starts = @splat(0) };
        for (0..c.streams) |s| {
            c.rows[s] = 1 + rng.range(16);
            const start = if (rng.range(3) == 0) edges[rng.range(@intCast(edges.len))] else rng.range(capacity - c.rows[s] + 1);
            c.starts[s] = @min(start, capacity - c.rows[s]);
        }
        try cases.append(a, c);
    }
    return cases;
}

/// Caches past 8,192, where a row's key chunk widens from 128 to 512 keys, and past 8,704, the next 512-key edge.
const boundary_capacity: u32 = 8960;
const boundary_targets = [_]u32{ 8063, 8064, 8190, 8191, 8192, 8193, 8194, 8703, 8704, 8705 };

/// A window's row inputs by stream and cache position alone: key and value are the cache's own rows there.
fn boundaryInputs(windows: []const planning.Window, ids: []const u64, caches: []const attn.Cache, query: mtl.Buffer, key: mtl.Buffer, value: mtl.Buffer, gate: mtl.Buffer) void {
    var node: usize = 0;
    for (windows, ids, caches) |w, id, cache| for (0..w.parents.len) |row| {
        const at: usize = w.start + row;
        var rng = Rng{ .s = ((id + 1) *% 0x9e3779b97f4a7c15) ^ (@as(u64, at) *% 0xbf58476d1ce4e5b9) | 1 };
        for (query.slice(u16, round_rows * 6144)[node * 6144 ..][0..6144]) |*x| x.* = rng.bf16();
        for (gate.slice(u16, round_rows * 12288)[node * 12288 ..][0..12288]) |*x| x.* = rng.bf16();
        for (0..4) |hk| {
            const from = (hk * cache.capacity + at) * 256;
            @memcpy(key.slice(u16, round_rows * 1024)[(node * 4 + hk) * 256 ..][0..256], cache.keys.buffer.slice(u16, cache.capacity * 1024)[from..][0..256]);
            @memcpy(value.slice(u16, round_rows * 1024)[(node * 4 + hk) * 256 ..][0..256], cache.values.buffer.slice(u16, cache.capacity * 1024)[from..][0..256]);
        }
        node += 1;
    };
}

/// Rows around 8,064, 8,192 and 8,704 equal their one-row steps byte for byte in chains, trees, beside neighbors.
fn boundary(r: *Rig, device: mtl.Device) !void {
    var rng = Rng{ .s = 0x5851f42d4c957f2d };
    var cache_buffers: [4]mtl.Buffer = undefined;
    var caches: [2]attn.Cache = undefined;
    for (&caches, 0..) |*cache, i| {
        const bytes = @as(usize, boundary_capacity) * 4 * 256 * 2;
        cache_buffers[2 * i] = try buffer(device, &rng, bytes, true);
        cache_buffers[2 * i + 1] = try buffer(device, &rng, bytes, true);
        cache.* = .{ .keys = .{ .buffer = cache_buffers[2 * i] }, .values = .{ .buffer = cache_buffers[2 * i + 1] }, .capacity = boundary_capacity };
    }
    defer for (cache_buffers) |b| b.deinit();
    const query = try buffer(device, &rng, round_rows * 6144 * 2, false);
    const key = try buffer(device, &rng, round_rows * 1024 * 2, false);
    const value = try buffer(device, &rng, round_rows * 1024 * 2, false);
    const gate = try buffer(device, &rng, round_rows * 12288 * 2, false);
    const out = try buffer(device, &rng, round_rows * 6144 * 2, false);
    defer for ([_]mtl.Buffer{ query, key, value, gate, out }) |b| b.deinit();
    const row_bytes = 6144 * 2;
    const modes = [_]planning.Geometry{ .{}, .{ .fuse_chain = false }, .{ .shared_prefix = false } };
    // each target's one-row step, with and without a shared prefix
    var steps: [2][boundary_targets.len][row_bytes]u8 = undefined;
    for ([_]usize{ 0, 2 }, 0..) |mode, m| for (boundary_targets, 0..) |position, t| {
        const windows = [_]planning.Window{.{ .parents = &.{-1}, .start = position, .capacity = boundary_capacity }};
        var plan = try attn.Plan.init(r.a, &windows, modes[mode]);
        defer plan.deinit();
        boundaryInputs(&windows, &.{0}, caches[0..1], query, key, value, gate);
        @memset(out.contents()[0..row_bytes], 0);
        try run(r.queue, r.op, if (mode == 2) r.plain_scratch else r.scratch, if (mode == 2) r.plain_binding else r.binding, plan, caches[0..1], .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out }, .{ .buffer = gate });
        try written(out, row_bytes, true);
        @memcpy(&steps[m][t], out.contents()[0..row_bytes]);
    };
    const Shape = struct { start: u32, parents: []const i32 };
    var long: [round_rows]i32 = undefined;
    for (&long, 0..) |*p, i| p.* = @as(i32, @intCast(i)) - 1;
    // chains, and trees whose first rows are a chain, so a target at depth d is row d at start + d
    const shapes = [_]Shape{
        .{ .start = 8191, .parents = &.{ -1, 0 } },
        .{ .start = 8192, .parents = &.{ -1, 0 } },
        .{ .start = 8190, .parents = &.{ -1, 0, 1, 2, 3 } },
        .{ .start = 8189, .parents = &.{ -1, 0, 1 } },
        .{ .start = 8060, .parents = &.{ -1, 0, 1, 2, 3, 4, 5 } },
        .{ .start = 8190, .parents = &.{ -1, 0, 1, 2, 3, 0, 1, 2, 3, 4 } },
        .{ .start = 8191, .parents = &.{ -1, 0, 1, 0, 0, 1 } },
        .{ .start = 8700, .parents = &.{ -1, 0, 1, 2, 3, 4 } },
        .{ .start = 8701, .parents = &.{ -1, 0, 1, 2, 3, 2, 1 } },
        .{ .start = 8066, .parents = long[0..128] },
        .{ .start = 8100, .parents = long[0..100] },
        .{ .start = 8640, .parents = long[0..80] },
    };
    // a neighbor in the other cache, before or after: short row, chains across and below 8192, a tree, past 8704
    const Neighbor = struct { start: u32, parents: []const i32, before: bool };
    const neighbors = [_]?Neighbor{ null, .{ .start = 150, .parents = &.{-1}, .before = false }, .{ .start = 8193, .parents = &.{ -1, 0, 1 }, .before = true }, .{ .start = 8000, .parents = long[0..16], .before = false }, .{ .start = 8190, .parents = &.{ -1, 0, 0, 1 }, .before = true }, .{ .start = 8900, .parents = &.{-1}, .before = true } };
    var plans: usize = 0;
    var compared: usize = 0;
    for (modes, 0..) |geometry, mode| for (shapes) |shape| for (neighbors) |neighbor| {
        var windows: [2]planning.Window = undefined;
        var ids: [2]u64 = undefined;
        var count: usize = 0;
        var own: usize = 0;
        for ([_]bool{ true, false }) |before| {
            if (!before) {
                own = count;
                windows[count] = .{ .parents = shape.parents, .start = shape.start, .capacity = boundary_capacity };
                ids[count] = 0;
                count += 1;
            }
            if (neighbor) |nb| if (nb.before == before) {
                windows[count] = .{ .parents = nb.parents, .start = nb.start, .capacity = boundary_capacity };
                ids[count] = 1;
                count += 1;
            };
        }
        var rows: usize = 0;
        for (windows[0..count]) |w| rows += w.parents.len;
        if (rows > round_rows) continue;
        const stream_caches: [2]attn.Cache = if (own == 0) caches else .{ caches[1], caches[0] };
        var plan = try attn.Plan.init(r.a, windows[0..count], geometry);
        defer plan.deinit();
        boundaryInputs(windows[0..count], ids[0..count], stream_caches[0..count], query, key, value, gate);
        @memset(out.contents()[0 .. rows * row_bytes], 0);
        try run(r.queue, r.op, if (mode == 2) r.plain_scratch else r.scratch, if (mode == 2) r.plain_binding else r.binding, plan, stream_caches[0..count], .{ .buffer = query }, .{ .buffer = key }, .{ .buffer = value }, .{ .buffer = out }, .{ .buffer = gate });
        try written(out, rows * row_bytes, true);
        plans += 1;
        const first: usize = @intCast(plan.meta[planning.global_words + own * planning.stream_words + 6]);
        for (shape.parents, 0..) |parent, row| {
            if (parent != @as(i32, @intCast(row)) - 1) break;
            const position = shape.start + @as(u32, @intCast(row));
            const t = std.mem.indexOfScalar(u32, &boundary_targets, position) orelse continue;
            const got = out.contents()[(first + row) * row_bytes ..][0..row_bytes];
            if (!std.mem.eql(u8, got, &steps[if (mode == 2) 1 else 0][t])) {
                std.debug.print("FAIL boundary: position {d}, mode {d}, window from {d} of {d} rows, neighbor {any}\n", .{ position, mode, shape.start, shape.parents.len, neighbor });
                return error.BoundaryBits;
            }
            compared += 1;
        }
    };
    var same: usize = 0;
    for (steps[0], steps[1]) |a, b| same += @intFromBool(std.mem.eql(u8, &a, &b));
    std.debug.print("attention_fused: {d} rows at {any} in {d} plans across 8192 (windows, trees, the tail kernel, neighbors, no shared prefix) byte-identical to their one-row steps; {d} of {d} steps equal with and without a shared prefix\n", .{ compared, boundary_targets, plans, same, boundary_targets.len });
    if (compared != 642 or plans != 201) return error.BoundaryCoverage;
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
    var op = try attn.Attention.init(device, library);
    defer op.deinit();
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--bench")) {
        var brng = Rng{ .s = 0x1234567 };
        if (args.len >= 4) { // --bench CONTEXTS ROWS [tree]: comma-separated contexts
            var it = std.mem.tokenizeScalar(u8, args[2], ',');
            while (it.next()) |c| try bench(a, device, queue, &op, &brng, try std.fmt.parseInt(u32, c, 10), try std.fmt.parseInt(u32, args[3], 10), args.len >= 5);
            return;
        }
        for ([_]u32{ 1, 4, 16, 32 }) |rows| for ([_]u32{ 2048, 16384 }) |context| try bench(a, device, queue, &op, &brng, context, rows, false);
        return;
    }
    const limits = attn.Limits{ .rows = round_rows, .streams = 8, .keys = capacity };
    var scratch = try attn.Scratch.init(device, limits);
    defer scratch.deinit();
    var binding = try attn.Binding.init(device, limits);
    defer binding.deinit();
    // Chips without tensor units plan with shared_prefix off: no prefix kernel, every key through the tail kernel.
    const plain_limits = attn.Limits{ .rows = round_rows, .streams = 8, .keys = capacity, .shared_prefix = false };
    var plain_scratch = try attn.Scratch.init(device, plain_limits);
    defer plain_scratch.deinit();
    var plain_binding = try attn.Binding.init(device, plain_limits);
    defer plain_binding.deinit();
    // `rng` fills 4 caches and the first rows of each input, `wide_rng` the rest, so the first plans keep their bytes
    var rng = Rng{ .s = 0x9e3779b97f4a7c15 };
    var wide_rng = Rng{ .s = 0xd1b54a32d192ed03 };
    var cache_buffers: [16]mtl.Buffer = undefined;
    var caches: [8]attn.Cache = undefined;
    for (&caches, 0..) |*cache, i| {
        const bytes = @as(usize, capacity) * 4 * 256 * 2;
        const from = if (i < 4) &rng else &wide_rng;
        cache_buffers[2 * i] = try buffer(device, from, bytes, true);
        cache_buffers[2 * i + 1] = try buffer(device, from, bytes, true);
        cache.* = .{ .keys = .{ .buffer = cache_buffers[2 * i] }, .values = .{ .buffer = cache_buffers[2 * i + 1] }, .capacity = capacity };
    }
    defer for (cache_buffers) |b| b.deinit();
    const query = try wideBuffer(device, &rng, &wide_rng, round_rows * 6144 * 2, max_rows * 6144 * 2);
    const key = try wideBuffer(device, &rng, &wide_rng, round_rows * 1024 * 2, max_rows * 1024 * 2);
    const value = try wideBuffer(device, &rng, &wide_rng, round_rows * 1024 * 2, max_rows * 1024 * 2);
    const raw_gate = try wideBuffer(device, &rng, &wide_rng, round_rows * 12288 * 2, max_rows * 12288 * 2);
    const out_chain = try buffer(device, &rng, round_rows * 6144 * 2, false);
    const out_tail = try buffer(device, &rng, round_rows * 6144 * 2, false);
    const out_plain = try buffer(device, &rng, round_rows * 6144 * 2, false);
    defer for ([_]mtl.Buffer{ query, key, value, raw_gate, out_chain, out_tail, out_plain }) |b| b.deinit();

    const edges = [_]u32{ 0, 1, 62, 63, 64, 65, 127, 128, 191, 255, 256, 447, 448, 449, 510, 511, 512, 513, 575, 576, 1000, 1023, 1024, 1087, 2047, 2048, 3000, 4000 };
    var cases: std.ArrayList(Case) = .empty;
    for (edges) |start| for ([_]u32{ 1, 2, 3, 5, 6, 7, 16, 17, 40 }) |rows| {
        if (start + rows > capacity) continue;
        try cases.append(a, .{ .streams = 1, .rows = .{ rows, 0, 0, 0, 0, 0, 0, 0 }, .starts = .{ start, 0, 0, 0, 0, 0, 0, 0 } });
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
    const wide = try wideCases(a, &wide_rng, &edges);
    var rig = Rig{ .a = a, .queue = queue, .op = &op, .scratch = &scratch, .binding = &binding, .plain_scratch = &plain_scratch, .plain_binding = &plain_binding, .caches = &caches, .query = query, .key = key, .value = value, .raw_gate = raw_gate, .out_chain = out_chain, .out_tail = out_tail, .out_plain = out_plain, .rng = &rng };
    const old = try check(&rig, cases.items, false);
    std.debug.print("attention_fused: {d} plans, {d} chains byte-identical through the fused prefix kernel and the tail kernel, {d} bytes, sha256 {s}\n", .{ old.plans, old.chains, old.bytes, std.fmt.bytesToHex(old.sum, .lower) });
    std.debug.print("attention_fused: the same {d} plans without a shared prefix, sha256 {s}\n", .{ old.plans, std.fmt.bytesToHex(old.plain_sum, .lower) });
    const wide_totals = try check(&rig, wide.items, true);
    std.debug.print("attention_fused: {d} plans of 5 to 8 streams, {d} chains byte-identical through the fused prefix kernel and the tail kernel, {d} bytes, sha256 {s}\n", .{ wide_totals.plans, wide_totals.chains, wide_totals.bytes, std.fmt.bytesToHex(wide_totals.sum, .lower) });
    std.debug.print("attention_fused: the same {d} plans of 5 to 8 streams without a shared prefix, sha256 {s}\n", .{ wide_totals.plans, std.fmt.bytesToHex(wide_totals.plain_sum, .lower) });
    try boundary(&rig, device);
}
