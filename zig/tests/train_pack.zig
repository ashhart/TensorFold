//! Training passes holding several sequences against one a pass, and from kept prefixes against whole, on Metal.
const std = @import("std");
const tf = @import("tensorfold");
const mtl = @import("metal");

const nm = tf.nemotron;
const train = nm.train;
const Trainer = train.Trainer;
const compare = @import("cuda/pack_compare.zig").For(train);
const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// A pocket's neurons at a layer, the last `open` of them the open fact's (scaled by each sequence).
const neurons = 6;
const open = 2;

pub fn main(init: std.process.Init) !void {
    const gpa = std.heap.c_allocator;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print("usage: tf-train-pack MODEL_DIR EXAMPLES [COUNT]   (EXAMPLES: a pocket dump's fact-N-examples.txt)\n", .{});
        std.process.exit(2);
    }
    const count: usize = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 24;
    const m = try nm.Model.load(gpa, init.io, args[1], false);
    const b = try nm.backend.Metal.init(gpa, m, .{ .capacity = 4096, .chunk = 2048, .drafts = false, .streams = 2 });
    const t0 = now(init.io);
    const t = try Trainer.init(gpa, b);
    std.debug.print("RESULT trainer ready in {d:.1} s\n", .{now(init.io) - t0});
    const c = m.config;
    const d = c.hidden;
    var prng = std.Random.DefaultPrng.init(11);
    const rng = prng.random();
    // four MoE layers in the middle of the model, each with a pocket of six small random neurons
    var layers: [4]usize = undefined;
    var found: usize = 0;
    for (0..c.layers) |l| if (found < layers.len and c.kinds[l] == .moe and l >= c.layers * (2 + 2 * found) / 10) {
        layers[found] = l;
        found += 1;
    };
    for (layers[0..found]) |l| {
        const up = try gpa.alloc(f32, neurons * d);
        const down = try gpa.alloc(f32, neurons * d);
        for (up) |*v| v.* = rng.floatNorm(f32) * 5e-4;
        for (down) |*v| v.* = rng.floatNorm(f32) * 2e-2;
        try setPocket(b, l, up, down);
        t.from[l] = neurons - open;
    }
    // the checks' sequences and results live until the end
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const seqs = try compare.examples(a, init.io, args[2], count, layers[0..found], rng);
    // lane projections up to 128 rows, routed experts row by row below four pairs an expert: 85 rows
    try compare.pairs(a, t, seqs, layers[0..found], @min(nm.prefill.lane_rows, (4 * c.experts - 1) / c.top_k), true);
    try compare.check(a, init.io, t, seqs, layers[0..found]);
    try compare.prefixes(a, init.io, t, seqs, layers[0..found]);
    try profile(t, m, seqs);
}

/// One pass of as many sequences as it holds, keys then targets, each dispatch alone: where a pass's time goes.
fn profile(t: *Trainer, m: anytype, seqs: []const train.Seq) !void {
    var n: usize = 0;
    var rows: usize = 0;
    while (n < seqs.len and n < train.max_seqs and rows + seqs[n].ids.len - 1 <= t.pack) : (n += 1) rows += seqs[n].ids.len - 1;
    for ([_]train.Mode{ .keys, .target }) |mode| {
        for (seqs[0..n], 0..) |s, i| for (s.nudge.layers, 0..) |mask, k| {
            var bits = mask;
            while (bits != 0) : (bits &= bits - 1) @memset(t.deltaAt(i, k, @ctz(bits)), 0);
        };
        var prof: nm.encoder.Profiler = .{ .queue = m.queue, .kernels = &m.kernels, .prompt = &m.prefill };
        t.profile = &prof;
        var got: [train.max_seqs]train.Result = undefined;
        try t.pass(seqs[0..n], mode, .{ .decide = 0, .whole = true }, got[0..n]);
        t.profile = null;
        std.debug.print("RESULT profile {s}: {d} sequences, {d} rows\n", .{ @tagName(mode), n, rows });
        prof.report(1);
    }
}

fn now(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / 1e9;
}

/// MoE layer l's pocket as the forward reads it (slide.zig's host makes it the same way).
fn setPocket(b: *nm.backend.Metal, l: usize, up: []const f32, down: []const f32) !void {
    const w = &b.m.weights;
    const mo = switch (w.layers[l]) {
        .moe => |*x| x,
        else => return error.NotAnMoeLayer,
    };
    const scratch = 8192 + 256;
    var made: [4]mtl.Buffer = undefined;
    for (&made, [_]usize{ 64 * b.m.config.hidden * 4, 64 * b.m.config.hidden * 4, scratch * 64 * 4, scratch * 4 }) |*buf, bytes| buf.* = try b.m.device.buffer(bytes, opts);
    @memset(made[0].slice(f32, 64 * b.m.config.hidden), 0);
    @memset(made[1].slice(f32, 64 * b.m.config.hidden), 0);
    @memcpy(made[0].slice(f32, up.len), up);
    @memcpy(made[1].slice(f32, down.len), down);
    mo.pocket = .{ .up = made[0], .down = made[1], .z = made[2], .zn = made[3], .used = neurons };
}
