//! tf-glm-dflash-speed --overlap: the round trip a same-buffer drafter pass would save, measured on synthetic weights.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glm = tf.glm;
const speed = @import("glm_metal_dflash_speed.zig");
const streams = @import("glm_metal_dflash_speed_streams.zig");
const Model = glm.metal_dflash_model.Model;
const Context = glm.metal_dflash_model.Context;
const stock = glm.metal_dflash_config.stock;

/// A target window's stand-in, committed: `mb` MiB copied among the dummy buffers on the target's queue.
fn window(queue: mtl.Queue, dummies: []const mtl.Buffer, mb: u32, round: usize) mtl.CommandBuffer {
    const cb = queue.commandBuffer();
    copies(cb, dummies, mb, round);
    cb.commit();
    return cb;
}

/// The window stand-in's copies encoded into `cb`, not committed.
fn copies(cb: mtl.CommandBuffer, dummies: []const mtl.Buffer, mb: u32, round: usize) void {
    const e = cb.blit();
    var left: usize = @as(usize, mb) << 20;
    var i = round;
    while (left > 0) : (i += 1) {
        const from = dummies[i % dummies.len];
        const to = dummies[(i + 1) % dummies.len];
        const n = @min(left, @min(from.length(), to.length()));
        e.copy(from, 0, to, 0, n);
        left -= n;
    }
    e.end();
}

/// N streams' rounds after a window: host waits on it before the pass (today), or the pass is committed behind it.
pub fn run(a: std.mem.Allocator, h: *speed.Harness, q4: bool, args: speed.Args) !void {
    if (h.dummies.items.len < 2 or args.window_mb == 0) return error.Usage; // needs --resident-gib and --window-mb
    const n = @max(1, args.streams);
    const m = try speed.create(a, h, q4);
    defer m.deinit();
    var ctx: [8]*Context = undefined;
    var feeds: [8]streams.Feed = undefined;
    for (0..n) |k| {
        ctx[k] = try Context.init(a, m);
        try ctx[k].reset(stock, 0);
        feeds[k] = .{ .source = try speed.taps(h.device, args.context + args.accepted * (2 * args.rounds + 8), 300 + k) };
        try streams.feed(m, ctx[k], &feeds[k], args.context);
    }
    defer for (0..n) |k| {
        ctx[k].deinit(a);
        feeds[k].source.deinit();
    };
    var inline_state = try Same.init(a, h, m, n, args);
    defer inline_state.deinit(a);
    var walls: [3][256]f64 = undefined;
    var alone: [256]f64 = undefined;
    const rounds = @min(args.rounds, 256);
    for (0..rounds + 2) |r| {
        const w0 = speed.now();
        window(h.queue, h.dummies.items, args.window_mb, r).wait();
        const w1 = speed.now();
        for (0..2) |mode| {
            for (0..n) |k| try streams.feed(m, ctx[k], &feeds[k], args.accepted);
            const t0 = speed.now();
            const cb = window(h.queue, h.dummies.items, args.window_mb, r);
            if (mode == 0) cb.wait(); // today: the host sees the window end before it commits the pass
            try pass(m, ctx[0..n], r);
            cb.wait();
            if (r >= 2) walls[mode][r - 2] = speed.ms(t0, speed.now());
        }
        const t2 = speed.now();
        try inline_state.round(h, m, args, r);
        if (r >= 2) walls[2][r - 2] = speed.ms(t2, speed.now());
        if (r >= 2) alone[r - 2] = speed.ms(w0, w1);
    }
    const serial = speed.median(walls[0][0..rounds]);
    const behind = speed.median(walls[1][0..rounds]);
    std.debug.print("\n== {s} drafter, {d} stream(s), window {d} MiB copied ({d:.3} ms alone) ==\n", .{ if (q4) "4-bit" else "bf16", n, args.window_mb, speed.median(alone[0..rounds]) });
    std.debug.print("  window then pass, host between: {d:.3} ms; pass committed behind the window: {d:.3} ms; saved {d:.3} ms a round\n", .{ serial, behind, serial - behind });
    const one = speed.median(walls[2][0..rounds]);
    std.debug.print("  window, acceptance and pass in one buffer, one wait (same-buffer): {d:.3} ms; saved {d:.3} ms a round\n", .{ one, serial - one });
}

/// The same-buffer rounds: each window's 8 rows absorbed ahead, accepted on the GPU, the block behind them.
const Same = struct {
    ctx: [8]*Context = undefined,
    feeds: [8]streams.Feed = undefined,
    n: usize,
    kernel: glm.metal_dflash_follow.Follow,
    acc: glm.metal_dflash_accept.Accept,
    tokens: mtl.Buffer,
    picks: mtl.Buffer,
    table: mtl.Buffer,
    words: mtl.Buffer,
    const rows = 8;

    fn init(a: std.mem.Allocator, h: *speed.Harness, m: *Model, n: usize, args: speed.Args) !Same {
        const opts = mtl.ResourceOptions.shared;
        var x: Same = .{ .n = n, .kernel = try glm.metal_dflash_follow.Follow.init(a, h.device, null), .acc = try glm.metal_dflash_accept.Accept.init(a, h.device, null), .tokens = try h.device.buffer(128 * 4, opts), .picks = try h.device.buffer(128 * 4, opts), .table = try h.device.buffer(64, opts), .words = try h.device.buffer(64, opts) };
        const kept = @min(@max(args.accepted, 1), rows) - 1; // window keeps first row plus accepted drafts
        for (0..n) |k| {
            x.ctx[k] = try Context.init(a, m);
            try x.ctx[k].reset(stock, 0);
            x.feeds[k] = .{ .source = try speed.taps(h.device, args.context + rows * (args.rounds + 8), 600 + k) };
            try streams.feed(m, x.ctx[k], &x.feeds[k], args.context);
            try m.swap(x.ctx[k]);
            defer m.swap(x.ctx[k]) catch unreachable;
            try m.flush();
            x.table.slice(u32, 16)[k] = @intCast(k * rows);
            x.table.slice(u32, 16)[8 + k] = rows;
            const t = x.tokens.slice(u32, 128)[k * rows ..][0..rows];
            const p = x.picks.slice(u32, 128)[k * rows ..][0..rows];
            for (t, p, 0..) |*ti, *pi, j| {
                ti.* = @intCast((j * 7919 + k * 31) % 150000);
                pi.* = (ti.* + 1) % 150000;
            }
            for (0..kept) |j| p[j] = t[j + 1];
        }
        return x;
    }

    fn deinit(x: *Same, a: std.mem.Allocator) void {
        for (0..x.n) |k| {
            x.ctx[k].deinit(a);
            x.feeds[k].source.deinit();
        }
        x.kernel.deinit();
        x.acc.deinit();
        for ([_]mtl.Buffer{ x.tokens, x.picks, x.table, x.words }) |b| b.deinit();
    }

    /// One round: the window's taps fed, then window, acceptance and pass in one buffer, one wait, the trees drawn.
    fn round(x: *Same, h: *speed.Harness, m: *Model, args: speed.Args, r: usize) !void {
        const same = glm.metal_dflash_same;
        const ahead = glm.metal_dflash_ahead;
        var aheads: [8]ahead.Ahead = undefined;
        for (0..x.n) |k| {
            const first = x.ctx[k].session.context.end;
            x.ctx[k].committed_end = first;
            try streams.feed(m, x.ctx[k], &x.feeds[k], rows);
            aheads[k] = .{ .context = x.ctx[k], .first = first, .rows = rows };
        }
        const cb = h.queue.commandBuffer();
        copies(cb, h.dummies.items, args.window_mb, r);
        const e = cb.compute(.serial);
        try x.acc.encode(e, .{ .buf = x.tokens }, .{ .buf = x.picks }, .{ .buf = x.table }, .{ .buf = x.table, .off = 32 }, .{ .buf = x.words }, @intCast(x.n));
        e.barrier();
        const epoch = try same.beginOn(m, cb, e);
        defer same.release(m);
        const p = try same.encode(m, &x.kernel, epoch, aheads[0..x.n], m.blockFor(7), .{ .gpu = .{ .buf = x.words } });
        e.end();
        cb.commit();
        cb.wait();
        try same.landed(m);
        var lattices: [8]tf.qwen27.dflash.operators.Lattice = undefined;
        try same.lattices(m, &p, lattices[0..x.n]);
        defer for (lattices[0..x.n]) |*l| l.deinit();
        const words = x.words.slice(u32, 16);
        for (0..x.n) |k| {
            var t = try same.tree(m, &p, k, lattices[k], words[2 * k], words[2 * k + 1], .{ .nodes = 7, .children = 1 });
            t.deinit();
            ahead.settle(stock, x.ctx[k], aheads[k].first, words[2 * k]);
        }
    }
};

fn pass(m: *Model, ctx: []const *Context, r: usize) !void {
    if (ctx.len == 1) {
        try m.swap(ctx[0]);
        var t = m.proposeOptions(@intCast((r * 7919) % 150000), .{ .nodes = 7 }) catch |err| {
            m.swap(ctx[0]) catch {};
            return err;
        };
        t.deinit();
        return m.swap(ctx[0]);
    }
    var entries: [8]Model.Proposal = undefined;
    for (ctx, 0..) |x, k| entries[k] = .{ .context = x, .pending = @intCast((r * 7919 + k * 31) % 150000), .options = .{ .nodes = 7 } };
    var many: [8]tf.qwen27.dflash.selector.Tree = undefined;
    try m.proposeBatch(entries[0..ctx.len], many[0..ctx.len]);
    for (many[0..ctx.len]) |*t| t.deinit();
}
