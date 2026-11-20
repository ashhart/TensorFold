//! Exact CPU forward seams on a small synthetic F32 source: whole, chunked and one-row prompts, threads and reset.
const std = @import("std");
const config = @import("config.zig");
const op = @import("cpu_rows.zig");
const g = @import("core").gguf;
const Pool = @import("cpu_gguf_pool.zig").Pool;
const Forward = @import("cpu_forward.zig").Forward(Synthetic);

/// Deterministic F32 tensors keyed by name; KDA rates are positive, as the GGUF stores them.
const Synthetic = struct {
    arena: std.heap.ArenaAllocator,
    made: std.StringHashMapUnmanaged([]f32) = .empty,
    hidden: usize,
    pool: ?*Pool = null,

    fn deinit(s: *Synthetic) void {
        if (s.pool) |p| p.destroy();
        s.arena.deinit();
    }

    /// Workers that dispatch even these small matrices, so every projection and expert row crosses threads.
    fn workers(s: *Synthetic, threads: usize) !*Pool {
        if (s.pool) |p| {
            if (p.threads == threads) return p;
            p.destroy();
            s.pool = null;
        }
        const p = try Pool.create(std.testing.allocator, std.testing.io, threads);
        p.min_weights = 0;
        s.pool = p;
        return p;
    }

    fn matrix(s: *Synthetic, name: []const u8, width: usize, n: usize) !g.Tensor {
        const w = std.mem.sliceAsBytes(try s.tensor(name, n * width));
        return .{ .name = name, .kind = .f32, .rank = 2, .shape = .{ width, n, 1, 1 }, .offset = 0, .len = w.len, .bytes = w };
    }

    fn tensor(s: *Synthetic, name: []const u8, n: usize) ![]f32 {
        if (s.made.get(name)) |v| {
            if (v.len != n) return error.SyntheticShape;
            return v;
        }
        const a = s.arena.allocator();
        const v = try a.alloc(f32, n);
        var rng = std.Random.DefaultPrng.init(std.hash.Wyhash.hash(7, name));
        const r = rng.random();
        const norm = std.mem.endsWith(u8, name, "norm.weight");
        const rate = std.mem.endsWith(u8, name, "A_log");
        for (v) |*x| {
            const u = r.float(f32) * 2 - 1;
            x.* = if (norm) 1 + u / 8 else if (rate) 0.5 + (u + 1) / 2 else u / 8;
        }
        try s.made.put(a, try a.dupe(u8, name), v);
        return v;
    }

    pub fn values(s: *Synthetic, a: std.mem.Allocator, name: []const u8, shape: []const usize) ![]f32 {
        var n: usize = 1;
        for (shape) |d| n *= d;
        return a.dupe(f32, try s.tensor(name, n));
    }

    pub fn project(s: *Synthetic, name: []const u8, x: []const f32, y: []f32, rows: usize, width: usize, outputs: usize, rounded: bool, threads: usize) !void {
        const t = try s.matrix(name, width, outputs);
        if (threads > 1) return (try s.workers(threads)).project(t, x, y, rows, rounded);
        const w = try s.tensor(name, outputs * width);
        const dense = op.Tensor{ .dtype = .f32, .rank = 2, .shape = .{ outputs, width, 1, 1, 1 }, .bytes = std.mem.sliceAsBytes(w) };
        try op.projectOn(dense, x, y, rows, width, outputs, rounded, 1);
    }

    pub fn projectExperts(s: *Synthetic, prefix: []const u8, ids: []const u32, x: []const f32, out: []f32, hidden: usize, inter: usize, limit: f32, threads: usize) !void {
        var list: [16]@import("cpu_gguf_pool.zig").Expert = undefined;
        var buf: [128]u8 = undefined;
        for (ids, 0..) |id, j| list[j] = .{
            .gate = try s.matrix(try s.arena.allocator().dupe(u8, try std.fmt.bufPrint(&buf, "{s}experts.{d}.gate_proj.weight", .{ prefix, id })), hidden, inter),
            .up = try s.matrix(try s.arena.allocator().dupe(u8, try std.fmt.bufPrint(&buf, "{s}experts.{d}.up_proj.weight", .{ prefix, id })), hidden, inter),
            .down = try s.matrix(try s.arena.allocator().dupe(u8, try std.fmt.bufPrint(&buf, "{s}experts.{d}.down_proj.weight", .{ prefix, id })), inter, hidden),
            .out = out[j * hidden ..].ptr,
        };
        try (try s.workers(threads)).experts(list[0..ids.len], x, hidden, inter, limit);
    }

    pub fn embed(s: *Synthetic, ids: []const u32, out: []f32) !void {
        const table = try s.tensor("model.word_embeddings.weight", tiny().vocab * s.hidden);
        for (ids, 0..) |id, row| @memcpy(out[row * s.hidden ..][0..s.hidden], table[id * s.hidden ..][0..s.hidden]);
    }
};

fn tiny() config.Config {
    var c = std.mem.zeroes(config.Config);
    c.hidden = 64;
    c.layers = 4;
    c.vocab = 48;
    c.eps = 0.000001;
    c.rope_theta = 6000000;
    c.rope_dim = 64;
    c.rope_style = .gptj;
    c.group = 4;
    c.dense_layers = 1;
    c.dense_inter = 32;
    c.kda_heads = 1;
    c.kda_dim = 128;
    c.conv = 4;
    c.lower_bound = -5;
    c.mla_heads = 2;
    c.q_lora = 32;
    c.kv_lora = 512;
    c.nope = 128;
    c.v_dim = 128;
    c.head_gate = true;
    c.experts = 8;
    c.topk = 2;
    c.n_group = 4;
    c.topk_group = 2;
    c.routed_scale = 2.5;
    c.norm_topk = true;
    c.expert_bias = true;
    c.moe_inter = 16;
    c.shared_inter = 16;
    c.eos_n = 1;
    return c;
}

fn last(r: @import("cpu_forward.zig").Result, vocab: usize) []const u8 {
    return std.mem.sliceAsBytes(r.logits[r.logits.len - vocab ..]);
}

test "whole, chunked and one-row prompts give the same bytes at 1, 2, 3, 4 and 8 threads, and reset replays them" {
    const a = std.testing.allocator;
    var source = Synthetic{ .arena = std.heap.ArenaAllocator.init(a), .hidden = 64 };
    defer source.deinit();
    const c = tiny();
    const ids = [_]u32{ 3, 17, 5, 40, 9, 22 };
    var whole = try Forward.Session.init(a, &source, c, 16, 1);
    defer whole.deinit();
    var want = try whole.run(a, &ids);
    defer want.deinit();
    for (want.picks) |p| try std.testing.expect(p < c.experts);
    var stateless = try Forward.run(a, &source, &c, &ids, 1, null, null);
    defer stateless.deinit();
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want.logits), std.mem.sliceAsBytes(stateless.logits));
    for ([_]usize{ 1, 2, 3, 4, 8 }) |threads| for ([_]usize{ 1, 2, 4 }) |chunk| {
        var s = try Forward.Session.init(a, &source, c, 16, threads);
        defer s.deinit();
        for (0..2) |_| {
            var at: usize = 0;
            while (at < ids.len) {
                const end = @min(ids.len, at + chunk);
                var got = try s.run(a, ids[at..end]);
                defer got.deinit();
                at = end;
                if (at == ids.len) try std.testing.expectEqualSlices(u8, last(want, c.vocab), last(got, c.vocab));
            }
            s.reset();
        }
    };
}

test "a session refuses tokens past the vocabulary and past its capacity without advancing" {
    const a = std.testing.allocator;
    var source = Synthetic{ .arena = std.heap.ArenaAllocator.init(a), .hidden = 64 };
    defer source.deinit();
    var s = try Forward.Session.init(a, &source, tiny(), 2, 1);
    defer s.deinit();
    try std.testing.expectError(error.InvalidToken, s.run(a, &.{48}));
    try std.testing.expectError(error.CpuContextCapacity, s.run(a, &.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(u32, 0), s.state.position);
    var r = try s.run(a, &.{ 1, 2 });
    r.deinit();
    try std.testing.expectEqual(@as(u32, 2), s.state.position);
}
