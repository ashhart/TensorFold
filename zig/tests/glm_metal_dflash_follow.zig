//! A DFlash2 pass whose block follows a window accepted on the GPU, over the synthetic drafter.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const syn = @import("glm_metal_dflash_synthetic.zig");
const check = @import("glm_metal_dflash_ahead.zig");
const config = syn.config;
const Context = glm.metal_dflash_model.Context;
const Model = glm.metal_dflash_model.Model;
const ahead = glm.metal_dflash_ahead;
const same = glm.metal_dflash_same;
const accept = glm.metal_dflash_accept;
const op = tf.qwen27.dflash.operators;

/// The accept kernel over a round's windows, encoded ahead of the pass where a window's verify would be.
const Window = struct {
    k: *const accept.Accept,
    tokens: mtl.Buffer,
    picks: mtl.Buffer,
    offsets: mtl.Buffer,
    rows: mtl.Buffer,
    out: mtl.Buffer,
    streams: u32,

    fn call(ctx: *anyopaque, e: mtl.ComputeEncoder) !void {
        const w: *Window = @ptrCast(@alignCast(ctx));
        try w.k.encode(e, .{ .buf = w.tokens }, .{ .buf = w.picks }, .{ .buf = w.offsets }, .{ .buf = w.rows }, .{ .buf = w.out }, w.streams);
        e.barrier();
    }
};

fn sameLattice(x: op.Lattice, y: op.Lattice) !void {
    try std.testing.expectEqualSlices(u32, x.candidates, y.candidates);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(x.unary), std.mem.sliceAsBytes(y.unary));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(x.projected), std.mem.sliceAsBytes(y.projected));
}

/// Chain tokens two trees agree on before they first differ, and the chain's length.
pub const Agreement = struct { equal: u64 = 0, chain: u64 = 0 };

/// Rounds of `n` streams accepted on the GPU: lattices equal the host's plan; chains are compared with today's pass.
fn testRounds(a: std.mem.Allocator, h: *syn.Harness, n: usize, block: u32, q4: bool, agree: *Agreement) !void {
    const m = try syn.create(a, h, q4);
    defer m.deinit();
    const kernel = try glm.metal_dflash_follow.Follow.init(a, h.device, null);
    defer kernel.deinit();
    const acc = try accept.Accept.init(a, h.device, null);
    defer acc.deinit();
    var taps: [3]mtl.Buffer = undefined;
    for (taps[0..n], 0..) |*t, k| t.* = try check.seeded(h.device, 300, 900 + k);
    defer for (taps[0..n]) |t| t.deinit();
    const lens = [3]u32{ 30, 61, 9 };
    const rows = [3]u32{ 8, 5, 8 };
    var gpu: [3]*Context = undefined;
    var host: [3]*Context = undefined;
    var today: [3]*Context = undefined;
    for (0..n) |k| {
        gpu[k] = try check.primed(a, m, taps[k], lens[k]);
        host[k] = try check.primed(a, m, taps[k], lens[k]);
        today[k] = try check.primed(a, m, taps[k], lens[k]);
    }
    defer for (0..n) |k| {
        gpu[k].deinit(a);
        host[k].deinit(a);
        today[k].deinit(a);
    };
    const opts = mtl.ResourceOptions.shared;
    var w: Window = .{ .k = &acc, .tokens = try h.device.buffer(64 * 4, opts), .picks = try h.device.buffer(64 * 4, opts), .offsets = try h.device.buffer(32, opts), .rows = try h.device.buffer(32, opts), .out = try h.device.buffer(64, opts), .streams = @intCast(n) };
    defer for ([_]mtl.Buffer{ w.tokens, w.picks, w.offsets, w.rows, w.out }) |x| x.deinit();
    var random: std.Random.DefaultPrng = .init(4242 + n * 17 + block);
    const rng = random.random();
    const nodes = block - 1;
    for (0..24) |round| {
        var gpu_aheads: [3]ahead.Ahead = undefined;
        var host_aheads: [3]ahead.Ahead = undefined;
        var words: [6]u32 = undefined;
        for (0..n) |k| {
            const first = gpu[k].session.context.end;
            const at = k * 16;
            w.offsets.slice(u32, 8)[k] = @intCast(at);
            w.rows.slice(u32, 8)[k] = rows[k];
            const keep = rng.uintLessThan(u32, rows[k]);
            const tokens = w.tokens.slice(u32, 64)[at..][0..rows[k]];
            const picks = w.picks.slice(u32, 64)[at..][0..rows[k]];
            for (tokens, picks) |*t, *p| {
                t.* = rng.uintLessThan(u32, config.vocab);
                p.* = rng.uintLessThan(u32, config.vocab);
            }
            for (0..keep) |j| picks[j] = tokens[j + 1];
            if (keep + 1 < rows[k] and picks[keep] == tokens[keep + 1]) picks[keep] = (tokens[keep + 1] + 1) % config.vocab;
            const want = accept.accept(tokens, picks);
            words[2 * k] = want[0];
            words[2 * k + 1] = want[1];
            for ([_]*Context{ gpu[k], host[k] }) |x| {
                x.committed_end = first;
                try syn.feedStream(m, x, taps[k], @intCast(first), rows[k]);
            }
            gpu_aheads[k] = .{ .context = gpu[k], .first = first, .rows = rows[k] };
            host_aheads[k] = .{ .context = host[k], .first = first, .rows = rows[k] };
        }
        var from_gpu: [3]op.Lattice = undefined;
        // the accept kernel and the pass in one buffer the test commits, as a window's verify and its drafter pass
        const cb = h.queue.commandBuffer();
        const enc = cb.compute(.serial);
        try Window.call(&w, enc);
        const epoch = try same.beginOn(m, cb, enc);
        const p1 = same.encode(m, &kernel, epoch, gpu_aheads[0..n], block, .{ .gpu = .{ .buf = w.out } }) catch |err| {
            same.release(m);
            return err;
        };
        enc.end();
        cb.commit();
        cb.wait();
        try same.landed(m);
        try same.lattices(m, &p1, from_gpu[0..n]);
        same.release(m);
        defer for (from_gpu[0..n]) |*l| l.deinit();
        try std.testing.expectEqualSlices(u32, words[0 .. 2 * n], w.out.slice(u32, 2 * n));
        var from_host: [3]op.Lattice = undefined;
        const p2 = try same.run(m, &kernel, host_aheads[0..n], block, .{ .host = words[0 .. 2 * n] });
        try same.lattices(m, &p2, from_host[0..n]);
        same.release(m);
        defer for (from_host[0..n]) |*l| l.deinit();
        for (0..n) |k| {
            try sameLattice(from_gpu[k], from_host[k]);
            const first = gpu_aheads[k].first;
            ahead.settle(config, gpu[k], first, words[2 * k]);
            ahead.settle(config, host[k], first, words[2 * k]);
            // today's pass: the kept rows absorbed at the stream's own context, its block at that context's keys
            try syn.feedStream(m, today[k], taps[k], @intCast(first), words[2 * k] + 1);
            try m.swap(today[k]);
            var want = m.proposeOptions(words[2 * k + 1], .{ .nodes = nodes, .children = 1 }) catch |err| {
                m.swap(today[k]) catch unreachable;
                return err;
            };
            try m.swap(today[k]);
            defer want.deinit();
            var got = try same.tree(m, &p1, k, from_gpu[k], words[2 * k], words[2 * k + 1], .{ .nodes = nodes, .children = 1 });
            defer got.deinit();
            try check.sameCaches(gpu[k], today[k]);
            var x: [7]u32 = undefined;
            var y: [7]u32 = undefined;
            var cx: [7]f64 = undefined;
            var cy: [7]f64 = undefined;
            const nx = glm.metal_dflash_rounds.flatten(&want, &x, &cx);
            const ny = glm.metal_dflash_rounds.flatten(&got, &y, &cy);
            var e: u32 = 0;
            while (e < @min(nx, ny) and x[e] == y[e]) e += 1;
            agree.equal += e;
            agree.chain += nx;
        }
        _ = round;
    }
}

pub fn run(a: std.mem.Allocator, h: *syn.Harness) !void {
    var agree: Agreement = .{};
    for ([_]usize{ 1, 3 }) |n| for ([_]u32{ 8, 16 }) |block| for ([_]bool{ true, false }) |q4| {
        try testRounds(a, h, n, block, q4, &agree);
    };
    std.debug.print("glm dflash2 follow: chains agree with today's pass on {d} of {d} drafts\n", .{ agree.equal, agree.chain });
}
