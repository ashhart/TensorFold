//! Metal 3 against Metal 4 verify passes on real weights, interleaved in one process: the same kernels on both queues.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const contract = tf.tree_round;
const Round = q.round_plan.Round;
const help = "tf-qwen27-metal4-ab --model DIR --tokens IDS.json [--rounds 90] [--chunk 128]";

/// Two trees from real DFlash2 rounds: a width takes their leading rows.
const trees = [2][16]i32{
    .{ -1, 0, 1, 2, 3, 4, 3, 6, 5, 8, 7, 5, 9, 10, 12, 13 },
    .{ -1, 0, 0, 2, 1, 1, 0, 0, 6, 5, 7, 0, 0, 3, 1, 11 },
};
const widths = [_]u32{ 1, 8, 11, 13, 16 };

const Picks = struct {
    ops: tf.tree_round_gpu.Ops,
    out: mtl.Buffer,
    logits: q.projection.Ref,
    rows: u32,
    vocab: u32,

    fn encode(ptr: *anyopaque, e: mtl.ComputeEncoder) anyerror!void {
        const p: *Picks = @ptrCast(@alignCast(ptr));
        try p.ops.argmax(e, .{ .buf = p.logits.buffer, .off = p.logits.offset }, .{ .buf = p.out }, .{ .rows = p.rows, .vocab = p.vocab, .stride = p.vocab });
        e.barrier();
    }
};

fn median(values: []f64) f64 {
    std.mem.sort(f64, values, {}, std.sort.asc(f64));
    return values[values.len / 2];
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var dir: ?[]const u8 = null;
    var tokens: ?[]const u8 = null;
    var rounds: usize = 90;
    var chunk: u32 = 128;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--help")) return std.debug.print("{s}\n", .{help});
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        if (std.mem.eql(u8, args[i - 1], "--model")) dir = args[i] else if (std.mem.eql(u8, args[i - 1], "--tokens")) tokens = args[i] else if (std.mem.eql(u8, args[i - 1], "--rounds")) rounds = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, args[i - 1], "--chunk")) chunk = try std.fmt.parseInt(u32, args[i], 10) else return error.UnknownFlag;
    }
    const text = try std.Io.Dir.cwd().readFileAlloc(io, tokens orelse return error.MissingTokens, a, .limited(16 << 20));
    const prompt = (try std.json.parseFromSlice([]u32, a, text, .{})).value;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.model.Model.load(gpa, io, dir orelse return error.MissingModel, 128, null);
    defer m.deinit();
    if (m.metal4 == null) return error.NoMetal4; // TF_Q27_METAL4=0, or a Mac without a Metal 4 queue
    const vocab: u32 = @intCast(m.config.vocab);
    var runners: [2]q.decode_round.Runner = undefined;
    for (&runners, 0..) |*r, k| {
        r.* = try q.decode_round.Runner.init(gpa, m, 1, 4096);
        r.metal4 = k == 1;
    }
    defer for (&runners) |*r| r.deinit();
    var pending: u32 = 0;
    for (&runners, 0..) |*r, k| {
        const s = q.session.Session{ .runner = r };
        try s.prefill(prompt, chunk);
        const first = try s.greedy();
        if (k == 1 and first != pending) return error.PromptsDiffer;
        pending = first;
    }
    var picks = Picks{
        .ops = try tf.tree_round_gpu.Ops.init(m.device),
        .out = try m.device.buffer(contract.max_rows * @sizeOf(contract.Pick), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked),
        .logits = m.frame.get(.logits),
        .rows = 0,
        .vocab = vocab,
    };
    defer picks.ops.deinit();
    defer picks.out.deinit();
    const saved = try gpa.alloc(u16, 16 * vocab);
    defer gpa.free(saved);
    var saved_picks: [16]contract.Pick = undefined;
    const walls = try a.alloc(f64, 2 * widths.len * rounds);
    const gpus = try a.alloc(f64, 2 * widths.len * rounds);
    var counts: [widths.len]usize = @splat(0);
    for (0..rounds + 4) |round| {
        const w = round % widths.len;
        const rows = widths[w];
        var ids: [16]u32 = undefined;
        ids[0] = pending;
        for (1..rows) |j| ids[j] = @intCast((j * 104729 + round * 31) % 150000 + 1000);
        const parents = trees[round / widths.len % 2][0..rows];
        // Which kernel runs first alternates, so neither always follows the other's caches.
        for ([_]usize{ round % 2, 1 - round % 2 }, 0..) |k, order| {
            const r = &runners[k];
            picks.rows = rows;
            var plan = try Round.init(gpa, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = ids[0..rows], .parents = parents }}, @intCast(m.config.conv_kernel), r.slots, vocab);
            defer plan.deinit();
            r.tail = .{ .ptr = &picks, .encode = Picks.encode };
            const began = mtl.clock.seconds();
            try r.verifyHead(&plan, .all);
            const wall = mtl.clock.seconds() - began;
            r.tail = null;
            if (round >= 4) {
                walls[(w * rounds + counts[w]) * 2 + k] = wall * 1e3;
                gpus[(w * rounds + counts[w]) * 2 + k] = r.last_gpu_seconds * 1e3;
            }
            const logits = m.frame.get(.logits).buffer.slice(u16, rows * vocab);
            const out = picks.out.slice(contract.Pick, rows);
            if (order == 0) {
                @memcpy(saved[0 .. rows * vocab], logits);
                @memcpy(saved_picks[0..rows], out);
            } else {
                if (!std.mem.eql(u16, saved[0 .. rows * vocab], logits)) return error.LogitsDiffer;
                for (saved_picks[0..rows], out) |x, y| if (x.token != y.token or x.nonfinite != y.nonfinite) return error.PicksDiffer;
            }
            for (out[0..rows], 0..) |pick, row| {
                const host = contract.rowArgmax(logits[row * vocab ..][0..vocab]);
                if (pick.token != host.token or pick.nonfinite != host.nonfinite) return error.PickAgainstHost;
            }
            if (order == 1) pending = out[0].token;
            const root = [_]u32{0};
            try r.keep(&plan, &.{&root});
        }
        if (round >= 4) counts[w] += 1;
    }
    std.debug.print("{d} rounds a side, every logit and pick equal, and every pick equal to the host's argmax\n", .{rounds + 4});
    std.debug.print("rows  Metal 3 wall (GPU)       Metal 4 wall (GPU)  wall saved\n", .{});
    for (widths, 0..) |rows, w| {
        const n = counts[w];
        if (n == 0) continue;
        const m3_wall = try a.alloc(f64, n);
        const m4_wall = try a.alloc(f64, n);
        const m3_gpu = try a.alloc(f64, n);
        const m4_gpu = try a.alloc(f64, n);
        for (0..n) |at| {
            m3_wall[at] = walls[(w * rounds + at) * 2];
            m4_wall[at] = walls[(w * rounds + at) * 2 + 1];
            m3_gpu[at] = gpus[(w * rounds + at) * 2];
            m4_gpu[at] = gpus[(w * rounds + at) * 2 + 1];
        }
        const x = median(m3_wall);
        const y = median(m4_wall);
        std.debug.print("{d:4}  {d:6.2} ({d:6.2})         {d:6.2} ({d:6.2})    {d:5.2} ms, {d:4.1}%\n", .{ rows, x, median(m3_gpu), y, median(m4_gpu), x - y, (x - y) / x * 100 });
    }
}
