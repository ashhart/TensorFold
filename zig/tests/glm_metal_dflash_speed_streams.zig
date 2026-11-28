//! tf-glm-dflash-speed --streams N: N streams' rounds as N single passes and as one batched pass, and their trees.
const std = @import("std");
const tf = @import("tensorfold");
const glm = tf.glm;
const speed = @import("glm_metal_dflash_speed.zig");
const Model = glm.metal_dflash_model.Model;
const Context = glm.metal_dflash_model.Context;
const stock = glm.metal_dflash_config.stock;

/// Each stream's context length as a share of --context: streams of different lengths share a pass.
const shares = [_]f64{ 1.0, 0.55, 0.3, 0.8, 0.45, 0.7, 0.2, 0.9 };

/// One stream's committed taps: its own seeded rows, `fed` of them absorbed so far.
pub const Feed = struct { source: @import("metal").Buffer, fed: u32 = 0 };

/// `rows` more of `f`'s taps into `x`, 128 a call, the model holding `x` meanwhile.
pub fn feed(m: *Model, x: *Context, f: *Feed, rows: u32) !void {
    const stride = @as(usize, stock.tapWidth()) * 2;
    try m.swap(x);
    defer m.swap(x) catch unreachable;
    var left = rows;
    while (left > 0) {
        const n = @min(left, 128);
        try m.absorb(.{ .buf = f.source, .off = f.fed * stride }, n);
        f.fed += n;
        left -= n;
    }
}

fn same(a: tf.qwen27.dflash.selector.Tree, b: tf.qwen27.dflash.selector.Tree) bool {
    return std.mem.eql(u32, a.tokens, b.tokens) and std.mem.eql(i32, a.parents, b.parents) and std.mem.eql(u8, std.mem.sliceAsBytes(a.scores), std.mem.sliceAsBytes(b.scores));
}

/// Two copies of N streams fed the same taps: copy A drafts one stream a pass, copy B all in one batched pass.
pub fn run(a: std.mem.Allocator, h: *speed.Harness, q4: bool, args: speed.Args) !void {
    const n = args.streams;
    const m = try speed.create(a, h, q4);
    defer m.deinit();
    var ctx: [2][8]*Context = undefined;
    var feeds: [2][8]Feed = undefined;
    const extra = args.accepted * (args.rounds + 14);
    for (0..2) |set| for (0..n) |k| {
        const len: u32 = @max(16, @as(u32, @intFromFloat(@as(f64, @floatFromInt(args.context)) * shares[k])));
        ctx[set][k] = try Context.init(a, m);
        try ctx[set][k].reset(stock, 0);
        feeds[set][k] = .{ .source = try speed.taps(h.device, len + extra, 200 + k) };
        try feed(m, ctx[set][k], &feeds[set][k], len);
        try m.swap(ctx[set][k]);
        try m.flush();
        try m.swap(ctx[set][k]);
    };
    std.debug.print("\n== {s} drafter, {d} streams (contexts", .{ if (q4) "4-bit" else "bf16", n });
    for (0..n) |k| std.debug.print(" {d}", .{feeds[0][k].fed});
    std.debug.print("), {d} new taps a stream a round, head {d}-bit ==\n", .{ args.accepted, args.head_bits });
    var singles: [256]f64 = undefined;
    var batched: [256]f64 = undefined;
    var equal: usize = 0;
    var total: usize = 0;
    const rounds = @min(args.rounds, 256);
    for (0..rounds + 2) |r| {
        for (0..2) |set| for (0..n) |k| try feed(m, ctx[set][k], &feeds[set][k], args.accepted);
        var one: [8]tf.qwen27.dflash.selector.Tree = undefined;
        const t0 = speed.now();
        for (0..n) |k| {
            try m.swap(ctx[0][k]);
            one[k] = try m.proposeOptions(@intCast((r * 7919 + k * 31) % 150000), .{ .nodes = 15 });
            try m.swap(ctx[0][k]);
        }
        const t1 = speed.now();
        var many: [8]tf.qwen27.dflash.selector.Tree = undefined;
        if (n > 1) {
            var entries: [8]Model.Proposal = undefined;
            for (0..n) |k| entries[k] = .{ .context = ctx[1][k], .pending = @intCast((r * 7919 + k * 31) % 150000), .options = .{ .nodes = 15 } };
            try m.proposeBatch(entries[0..n], many[0..n]);
        }
        const t2 = speed.now();
        for (0..n) |k| {
            if (n > 1) {
                total += 1;
                if (same(one[k], many[k])) equal += 1;
                many[k].deinit();
            }
            one[k].deinit();
        }
        if (r < 2) continue;
        singles[r - 2] = speed.ms(t0, t1);
        batched[r - 2] = speed.ms(t1, t2);
    }
    std.debug.print("  {d} single passes: {d:.3} ms a round\n", .{ n, speed.median(singles[0..rounds]) });
    if (n > 1) {
        std.debug.print("  one batched pass:  {d:.3} ms a round\n", .{speed.median(batched[0..rounds])});
        std.debug.print("  trees equal to the stream's single pass: {d} of {d}\n", .{ equal, total });
    }
    if (n > 1) {
        // the batched rounds again, every vtable call its own command buffer: GPU ms by operator and shape
        var split = speed.Split{ .inner = m.backend.ops(), .b = &m.backend };
        for (0..8) |r| {
            for (0..n) |k| try feed(m, ctx[1][k], &feeds[1][k], args.accepted);
            var entries: [8]Model.Proposal = undefined;
            for (0..n) |k| entries[k] = .{ .context = ctx[1][k], .pending = @intCast((r * 13 + k) % 150000), .options = .{ .nodes = 15 } };
            var many: [8]tf.qwen27.dflash.selector.Tree = undefined;
            try m.proposeBatchOps(split.ops(), entries[0..n], many[0..n]);
            for (many[0..n]) |*t| t.deinit();
        }
        split.report();
    }
    for (0..2) |set| for (0..n) |k| {
        ctx[set][k].deinit(a);
        feeds[set][k].source.deinit();
    };
}
