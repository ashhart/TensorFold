//! tf_lane_reg against qk on real weights, interleaved in one process: load leaves plain matrices, this copies them.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const lane = tf.lane_projection;
const contract = tf.tree_round;
const Round = q.round_plan.Round;
const help = "tf-qwen27-qk-ab --model DIR --tokens IDS.json [--rounds 90] [--chunk 128]";

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

/// tf_lane_reg's copy of every plain matrix, reordered on the GPU as load would have: the buffers stay in `owned`.
fn regCopies(a: std.mem.Allocator, device: mtl.Device, w: *q.gpu_weights.Weights, owned: *std.ArrayList(mtl.Buffer)) !usize {
    var list: std.ArrayList(*q.projection.Linear) = .empty;
    defer list.deinit(a);
    for (w.layers) |*l| {
        switch (l.mixer) {
            .linear => |*g| try list.appendSlice(a, &.{ &g.qkv, &g.zba, &g.out }),
            .attention => |*t| try list.appendSlice(a, &.{ &t.q, &t.kv, &t.out }),
        }
        try list.appendSlice(a, &.{ &l.gu, &l.down });
    }
    try list.append(a, &w.head);
    var jobs: std.ArrayList(lane.Order) = .empty;
    defer jobs.deinit(a);
    for (list.items) |p| {
        if (!p.plain) continue;
        const bytes = @as(usize, p.n) * p.k / 2;
        const buf = try device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        try owned.append(a, buf);
        @memcpy(buf.contents()[0..bytes], p.words.buffer.contents()[p.words.offset..][0..bytes]);
        p.reg_copy = .{ .buffer = buf };
        try jobs.append(a, .{ .w = .{ .buf = buf }, .n = p.n, .k = p.k });
    }
    if (jobs.items.len > 0) try lane.regOrder(device, jobs.items);
    return jobs.items.len;
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
    var owned: std.ArrayList(mtl.Buffer) = .empty;
    defer {
        for (owned.items) |b| b.deinit();
        owned.deinit(gpa);
    }
    const copied = try regCopies(gpa, m.device, &m.weights, &owned);
    if (copied == 0) return error.NoPlainMatrices; // TF_Q27_QK=0, or a GPU that does not build qk
    std.debug.print("{d} plain matrices, each with a reg copy\n", .{copied});
    const vocab: u32 = @intCast(m.config.vocab);
    var runners: [2]q.decode_round.Runner = undefined;
    for (&runners) |*r| r.* = try q.decode_round.Runner.init(gpa, m, 1, 4096);
    defer for (&runners) |*r| r.deinit();
    var pending: u32 = 0;
    for (&runners, 0..) |*r, k| {
        q.quant_gpu.qk_rows = k == 1;
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
            q.quant_gpu.qk_rows = k == 1;
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
    q.quant_gpu.qk_rows = true;
    std.debug.print("{d} rounds a side, every logit and pick equal, and every pick equal to the host's argmax\n", .{rounds + 4});
    std.debug.print("rows  tf_lane_reg wall (GPU)   qk wall (GPU)      wall saved\n", .{});
    for (widths, 0..) |rows, w| {
        const n = counts[w];
        if (n == 0) continue;
        const reg_wall = try a.alloc(f64, n);
        const qk_wall = try a.alloc(f64, n);
        const reg_gpu = try a.alloc(f64, n);
        const qk_gpu = try a.alloc(f64, n);
        for (0..n) |at| {
            reg_wall[at] = walls[(w * rounds + at) * 2];
            qk_wall[at] = walls[(w * rounds + at) * 2 + 1];
            reg_gpu[at] = gpus[(w * rounds + at) * 2];
            qk_gpu[at] = gpus[(w * rounds + at) * 2 + 1];
        }
        const x = median(reg_wall);
        const y = median(qk_wall);
        std.debug.print("{d:4}  {d:6.2} ({d:6.2})         {d:6.2} ({d:6.2})    {d:5.2} ms, {d:4.1}%\n", .{ rows, x, median(reg_gpu), y, median(qk_gpu), x - y, (x - y) / x * 100 });
    }
}
