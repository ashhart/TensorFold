//! The 27B's projections timed by K split on its own shapes, the projections-only one-row floor, and shared decode
//! rounds by stream count (wall and GPU ms a round).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const Round = q.round_plan.Round;
const Linear = q.projection.Linear;
const Ref = q.projection.Ref;
const help = "tf-qwen27-bench --model DIR [--ms 30] [--rounds 24]";

/// Where a projection reads its rows and their group sums in the frame, as forward.zig and the mixers dispatch it.
const Input = enum { input, activated, mixer_out };

const Case = struct { name: []const u8, lin: Linear, from: Input };

fn source(f: *q.gpu_frame.Frame, from: Input) Ref {
    return switch (from) {
        .input => f.get(.input),
        .activated => f.get(.activated),
        .mixer_out => f.get(.mixer_out),
    };
}

fn sumsOf(f: *q.gpu_frame.Frame, from: Input, rows: u32) ?q.quant_gpu.Sums {
    return switch (from) {
        .input => q.forward.sums(f, .sums, rows),
        .activated => q.forward.sums(f, .act_sums, rows),
        .mixer_out => null,
    };
}

/// Each source row's group sums, as a norm or the MLP leaves them.
fn fillSums(m: *q.model.Model, c: Case, rows: u32) !void {
    const s = sumsOf(&m.frame, c.from, rows) orelse return;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    try m.kernels.sum(e, source(&m.frame, c.from), m.frame.get(.dims), s.ref, c.lin.k, rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.SumGpuFailure;
}

/// One projection `reps` times in one command buffer; GPU microseconds a call.
fn time(m: *q.model.Model, c: Case, lin: Linear, rows: u32, reps: usize) !f64 {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const f = &m.frame;
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..reps) |_| try m.kernels.quant(e, lin, source(f, c.from), sumsOf(f, c.from, rows), f.get(.dims), f.get(.logits), rows);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure() != null) return error.BenchGpuFailure;
    return cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(reps));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var dir: ?[]const u8 = null;
    var burst_ms: usize = 30;
    var rounds: usize = 24;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--help")) return std.debug.print("{s}\n", .{help});
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        if (std.mem.eql(u8, args[i - 1], "--model")) dir = args[i] else if (std.mem.eql(u8, args[i - 1], "--ms")) burst_ms = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, args[i - 1], "--rounds")) rounds = try std.fmt.parseInt(usize, args[i], 10) else return error.UnknownFlag;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.model.Model.load(gpa, io, dir orelse return error.MissingModel, 128);
    defer m.deinit();
    const w = m.weights;
    var gdn: ?q.gpu_weights.Gdn = null;
    var attn: ?q.gpu_weights.Attention = null;
    for (w.layers) |l| switch (l.mixer) {
        .linear => |g| gdn = gdn orelse g,
        .attention => |x| attn = attn orelse x,
    };
    const l0 = w.layers[0];
    const cases = [_]Case{
        .{ .name = "gdn_qkv", .lin = gdn.?.qkv, .from = .input },
        .{ .name = "gdn_zba", .lin = gdn.?.zba, .from = .input },
        .{ .name = "gdn_out", .lin = gdn.?.out, .from = .mixer_out },
        .{ .name = "attn_q", .lin = attn.?.q, .from = .input },
        .{ .name = "attn_kv", .lin = attn.?.kv, .from = .input },
        .{ .name = "attn_out", .lin = attn.?.out, .from = .mixer_out },
        .{ .name = "gate_up", .lin = l0.gu, .from = .input },
        .{ .name = "down", .lin = l0.down, .from = .activated },
        .{ .name = "head", .lin = w.head, .from = .input },
    };
    // plausible bf16 activations in every source the cases read
    const f = &m.frame;
    for ([_]Input{ .input, .activated, .mixer_out }) |from| {
        const r = source(f, from);
        const words: [*]u16 = @ptrCast(@alignCast(r.buffer.contents() + r.offset));
        const n = @min(32 * @as(usize, m.config.intermediate), (r.buffer.length() - r.offset) / 2);
        for (words[0..n], 0..) |*v, j| v.* = @truncate(0x3c00 + (j * 7) % 96 + (j / 1000) % 16);
    }
    const vocab = m.config.vocab;
    const reference = try gpa.alloc(u8, 32 * vocab * 2);
    defer gpa.free(reference);
    const logits = f.get(.logits);
    std.debug.print("hidden {d} layers {d}; {d} ms bursts a cell; us a call; '=' same bytes as the shipped split, '!' differs\n", .{ m.config.hidden, m.config.layers, burst_ms });
    for (cases) |c| {
        const lin = c.lin;
        const path: []const u8 = if (lin.raw != null) "row kernel" else "lane kernel";
        std.debug.print("{s} N {d} K {d}: {s}, shipped split {d}\n", .{ c.name, lin.n, lin.k, path, lin.slices });
        for ([_]u32{ 1, 8, 16, 32 }) |rows| {
            try fillSums(m, c, rows);
            // bursts of at least `burst_ms` so the GPU clock is up; the shipped split again at the end of the line
            const probe = try time(m, c, lin, rows, 20);
            const reps: usize = @max(40, @as(usize, @intFromFloat(@as(f64, @floatFromInt(burst_ms)) * 1000 / probe)));
            _ = try time(m, c, lin, rows, reps / 2);
            const base = try time(m, c, lin, rows, reps);
            const bytes = @as(usize, rows) * lin.n * 2;
            @memcpy(reference[0..bytes], logits.buffer.contents()[logits.offset..][0..bytes]);
            std.debug.print("  rows {d:>2} ({d} reps): shipped {d:>7.1}", .{ rows, reps, base });
            if (lin.raw == null) for ([_]u32{ 1, 2, 4, 8 }) |sk| {
                if (sk == lin.slices or lin.k / 64 % sk != 0) continue;
                var variant = lin;
                variant.slices = sk;
                @memset(logits.buffer.contents()[logits.offset..][0..bytes], 0);
                _ = try time(m, c, variant, rows, reps / 2);
                const t = try time(m, c, variant, rows, reps);
                const same = std.mem.eql(u8, reference[0..bytes], logits.buffer.contents()[logits.offset..][0..bytes]);
                std.debug.print("  sk{d} {d:.1}{s}", .{ sk, t, if (same) "=" else "!" });
            };
            std.debug.print("  shipped again {d:.1}\n", .{try time(m, c, lin, rows, reps)});
        }
    }

    // projections-only floor: every projection of a one-row step in one command buffer
    {
        const pool2 = mtl.objc.Pool.push();
        defer pool2.pop();
        for ([_]Input{ .input, .activated }) |from| try fillSums(m, .{ .name = "", .lin = if (from == .input) l0.gu else l0.down, .from = from }, 1);
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        var calls: usize = 0;
        for (w.layers) |l| {
            const mixer: []const Case = switch (l.mixer) {
                .linear => |g| &.{ .{ .name = "", .lin = g.qkv, .from = .input }, .{ .name = "", .lin = g.zba, .from = .input }, .{ .name = "", .lin = g.out, .from = .mixer_out } },
                .attention => |x| &.{ .{ .name = "", .lin = x.q, .from = .input }, .{ .name = "", .lin = x.kv, .from = .input }, .{ .name = "", .lin = x.out, .from = .mixer_out } },
            };
            for (mixer) |c| try m.kernels.quant(e, c.lin, source(f, c.from), sumsOf(f, c.from, 1), f.get(.dims), logits, 1);
            try m.kernels.quant(e, l.gu, f.get(.input), sumsOf(f, .input, 1), f.get(.dims), logits, 1);
            try m.kernels.quant(e, l.down, f.get(.activated), sumsOf(f, .activated, 1), f.get(.dims), logits, 1);
            calls += mixer.len + 2;
        }
        try m.kernels.quant(e, w.head, f.get(.input), sumsOf(f, .input, 1), f.get(.dims), logits, 1);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure() != null) return error.BenchGpuFailure;
        std.debug.print("projections-only one-row step: {d} calls, {d:.2} ms GPU\n", .{ calls + 1, cb.gpuSeconds() * 1e3 });
    }

    // shared decode rounds: N streams, one row each, verified and kept as the lane core does
    const slots = 8;
    var r = try q.decode_round.Runner.init(gpa, m, slots, 1024);
    defer r.deinit();
    var pending: [slots]u32 = undefined;
    for (0..slots) |k| {
        var prompt: [64]u32 = undefined;
        for (&prompt, 0..) |*t, j| t.* = @intCast((j * 7919 + k * 104729 + 13) % 150000 + 1000);
        const s = q.session.Session{ .runner = &r, .slot = @intCast(k) };
        try s.prefill(&prompt, 64);
        pending[k] = try s.greedy();
    }
    std.debug.print("shared rounds ({d} a stream count, one row a stream):\n", .{rounds});
    for ([_]usize{ 1, 2, 4, 8 }) |n| {
        var wall: f64 = 0;
        var gpu: f64 = 0;
        for (0..rounds + 2) |round| {
            var inputs: [slots]q.round_plan.Input = undefined;
            var ids: [slots][1]u32 = undefined;
            for (0..n) |k| {
                ids[k] = .{pending[k]};
                inputs[k] = .{ .slot = @intCast(k), .start = r.offsets[k], .capacity = r.capacity, .ids = &ids[k] };
            }
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            var plan = try Round.init(gpa, inputs[0..n], @intCast(m.config.conv_kernel), r.slots, @intCast(vocab));
            defer plan.deinit();
            try r.verifyHead(&plan, .all);
            const verify_gpu = r.last_gpu_seconds;
            const words = logits.buffer.slice(u16, n * vocab);
            for (0..n) |k| pending[k] = try q.session.argmax(words[k * vocab ..][0..vocab]);
            const zero = [_]u32{0};
            var paths: [slots][]const u32 = @splat(&zero);
            try r.keep(&plan, paths[0..n]);
            const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
            if (round < 2) continue; // the first rounds of a width warm its pipelines
            wall += ms;
            gpu += verify_gpu * 1e3;
        }
        const per = wall / @as(f64, @floatFromInt(rounds));
        std.debug.print("  {d} streams: {d:.2} ms a round ({d:.2} ms verify GPU), {d:.1} tok/s in all\n", .{ n, per, gpu / @as(f64, @floatFromInt(rounds)), @as(f64, @floatFromInt(n)) * 1000 / per });
    }
}
