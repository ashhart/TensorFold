//! The 27B's projections by K split, the one-row projections floor, and shared rounds by stream count (wall, GPU ms).
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const Round = q.round_plan.Round;
const Linear = q.projection.Linear;
const Ref = q.projection.Ref;
const help = "tf-qwen27-bench --model DIR [--ms 30] [--rounds 24] [--widths chain|tree|dflash|slots|shared]";

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
    var shape: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "--help")) return std.debug.print("{s}\n", .{help});
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        if (std.mem.eql(u8, args[i - 1], "--model")) dir = args[i] else if (std.mem.eql(u8, args[i - 1], "--ms")) burst_ms = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, args[i - 1], "--rounds")) rounds = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, args[i - 1], "--widths")) shape = args[i] else return error.UnknownFlag;
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.model.Model.load(gpa, io, dir orelse return error.MissingModel, 128, null);
    defer m.deinit();
    if (shape) |sh| if (std.mem.eql(u8, sh, "shared")) return shared(gpa, io, m, rounds);
    if (shape) |sh| return if (std.mem.eql(u8, sh, "slots")) slotsCheck(gpa, m) else widths(gpa, m, rounds, if (std.mem.eql(u8, sh, "tree")) .binary else if (std.mem.eql(u8, sh, "dflash")) .dflash else .chain);
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

    // each shape by K split over every layer's matrix of its kind, one row: the weights stream from memory, not a cache
    if (std.c.getenv("QWEN27_DISTINCT") != null) {
        const kinds = [_]struct { name: []const u8, from: Input }{ .{ .name = "gdn_qkv", .from = .input }, .{ .name = "gdn_zba", .from = .input }, .{ .name = "gdn_out", .from = .mixer_out }, .{ .name = "attn_q", .from = .input }, .{ .name = "attn_kv", .from = .input }, .{ .name = "attn_out", .from = .mixer_out }, .{ .name = "gate_up", .from = .input }, .{ .name = "down", .from = .activated } };
        for ([_]Input{ .input, .activated }) |from| try fillSums(m, .{ .name = "", .lin = if (from == .input) l0.gu else l0.down, .from = from }, 1);
        for (kinds) |kind| {
            var lins: std.ArrayList(Linear) = .empty;
            defer lins.deinit(gpa);
            for (w.layers) |l| {
                const pick: ?Linear = switch (l.mixer) {
                    .linear => |g| if (std.mem.eql(u8, kind.name, "gdn_qkv")) g.qkv else if (std.mem.eql(u8, kind.name, "gdn_zba")) g.zba else if (std.mem.eql(u8, kind.name, "gdn_out")) g.out else null,
                    .attention => |x| if (std.mem.eql(u8, kind.name, "attn_q")) x.q else if (std.mem.eql(u8, kind.name, "attn_kv")) x.kv else if (std.mem.eql(u8, kind.name, "attn_out")) x.out else null,
                };
                const chosen = pick orelse (if (std.mem.eql(u8, kind.name, "gate_up")) l.gu else if (std.mem.eql(u8, kind.name, "down")) l.down else null);
                if (chosen) |c| try lins.append(gpa, c);
            }
            if (lins.items.len == 0) continue;
            const first = lins.items[0];
            std.debug.print("{s} N {d} K {d} x {d} matrices, shipped split {d}:", .{ kind.name, first.n, first.k, lins.items.len, first.slices });
            for ([_]u32{ 1, 2, 4, 8 }) |sk| {
                if (first.k / 64 % sk != 0) continue;
                var best: f64 = 1e9;
                for (0..5) |_| {
                    const pool2 = mtl.objc.Pool.push();
                    defer pool2.pop();
                    const cb = m.queue.commandBuffer();
                    const e = cb.compute(.serial);
                    for (lins.items) |lin| {
                        var variant = lin;
                        variant.slices = sk;
                        try m.kernels.quant(e, variant, source(f, kind.from), sumsOf(f, kind.from, 1), f.get(.dims), logits, 1);
                    }
                    e.end();
                    cb.commit();
                    cb.wait();
                    if (cb.failure() != null) return error.BenchGpuFailure;
                    best = @min(best, cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(lins.items.len)));
                }
                const bytes = @as(f64, @floatFromInt(first.n)) * @as(f64, @floatFromInt(first.k)) * 0.5625;
                std.debug.print("  sk{d} {d:.1} us ({d:.0} GB/s)", .{ sk, best, bytes / best / 1e3 });
            }
            std.debug.print("\n", .{});
        }
    }

    // one-row rounds through the step prepare against the general prepare, commit and publication: the same state bytes
    if (std.c.getenv("QWEN27_STEPCHECK") != null) {
        var prints: [2][32]u8 = undefined;
        for ([_]bool{ true, false }, 0..) |on, k| {
            q.gdn_pool_gpu.step_rounds = on;
            var rr = try q.decode_round.Runner.init(gpa, m, 1, 1024);
            defer rr.deinit();
            const s = q.session.Session{ .runner = &rr, .slot = 0 };
            var prompt: [96]u32 = undefined;
            for (&prompt, 0..) |*t, j| t.* = @intCast((j * 7919 + 13) % 150000 + 1000);
            try s.prefill(&prompt, 64);
            for (0..24) |_| try s.step(try s.greedy());
            prints[k] = try s.fingerprint();
        }
        q.gdn_pool_gpu.step_rounds = true;
        std.debug.print("step prepare vs general prepare, 24 one-row steps: states and logits {s}\n", .{if (std.mem.eql(u8, &prints[0], &prints[1])) "byte-equal" else "DIFFER"});
        return;
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

    try shared(gpa, io, m, rounds);
}

/// Shared decode rounds: N streams, one row each, verified and kept as the lane core does (wall and verify GPU ms).
fn shared(gpa: std.mem.Allocator, io: std.Io, m: *q.model.Model, rounds: usize) !void {
    const slots = 128;
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
    for ([_]usize{ 1, 16, 32, 64, 128 }) |n| {
        var wall: f64 = 0;
        var gpu: f64 = 0;
        var parts: [4]f64 = .{ 0, 0, 0, 0 };
        for (0..rounds + 2) |round| {
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            var timing: [4]f64 = @splat(0);
            const verify_gpu = try sharedRound(gpa, io, m, &r, pending[0..n], &timing);
            const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
            if (round < 2) continue; // the first rounds of a width warm its pipelines
            wall += ms;
            gpu += verify_gpu * 1e3;
            for (&parts, timing) |*part, t| part.* += t;
        }
        const per = wall / @as(f64, @floatFromInt(rounds));
        const nr: f64 = @floatFromInt(rounds);
        std.debug.print("  {d} streams: {d:.2} ms a round ({d:.2} ms verify GPU), {d:.1} tok/s in all; plan {d:.2}, verify wall {d:.2}, argmax {d:.2}, keep {d:.2} ms\n", .{ n, per, gpu / nr, @as(f64, @floatFromInt(n)) * 1000 / per, parts[0] / nr, parts[1] / nr, parts[2] / nr, parts[3] / nr });
        if (std.c.getenv("QWEN27_PROFILE") == null or n != slots) continue;
        var trace = tf.gpu_profile.Trace{ .queue = m.queue };
        r.profile(&trace);
        trace.enabled = true;
        for (0..4) |_| _ = try sharedRound(gpa, io, m, &r, pending[0..n], null);
        r.profile(null);
        inline for (@typeInfo(tf.gpu_profile.Class).@"enum".field_names, 0..) |name, ci| if (trace.calls[ci] > 0) std.debug.print("  PROFILE {d} streams class={s} ms/round={d:.2}\n", .{ n, name, trace.seconds[ci] * 1e3 / 4 });
        for (trace.shapes[0..trace.shape_count]) |sh| std.debug.print("  SHAPE n={d} k={d} sk={d} calls/round={d} ms/round={d:.2} TFLOPS={d:.1}\n", .{ sh.n, sh.k, sh.sk, sh.calls / 4, sh.gpu_seconds * 1e3 / 4, 2.0 * @as(f64, @floatFromInt(n * sh.n * sh.k * sh.calls)) / sh.gpu_seconds / 1e12 });
    }
}

/// One round of a one-row window a slot, `pending` the slots' next tokens in and out; the verify's GPU seconds.
fn sharedRound(gpa: std.mem.Allocator, io: std.Io, m: *q.model.Model, r: *q.decode_round.Runner, pending: []u32, timing: ?*[4]f64) !f64 {
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    const most = q.attention_plan.max_streams;
    const n = pending.len;
    const vocab = m.config.vocab;
    var inputs: [most]q.round_plan.Input = undefined;
    var ids: [most][1]u32 = undefined;
    for (0..n) |k| {
        ids[k] = .{pending[k]};
        inputs[k] = .{ .slot = @intCast(k), .start = r.offsets[k], .capacity = r.capacity, .ids = &ids[k] };
    }
    var plan = try Round.init(gpa, inputs[0..n], @intCast(m.config.conv_kernel), r.slots, @intCast(vocab));
    defer plan.deinit();
    const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
    const advance = std.c.getenv("QWEN27_ADVANCE") != null;
    if (advance) try r.advance(&plan, .all) else try r.verifyHead(&plan, .all);
    const t2 = std.Io.Clock.awake.now(io).toNanoseconds();
    const gpu = r.last_gpu_seconds;
    const words = m.frame.get(.logits).buffer.slice(u16, n * vocab);
    for (0..n) |k| pending[k] = try q.session.argmax(words[k * vocab ..][0..vocab]);
    const t3 = std.Io.Clock.awake.now(io).toNanoseconds();
    const zero = [_]u32{0};
    var paths: [most][]const u32 = @splat(&zero);
    if (!advance) try r.keep(&plan, paths[0..n]);
    const t4 = std.Io.Clock.awake.now(io).toNanoseconds();
    if (timing) |ts| for (ts, [_]i128{ t1 - t0, t2 - t1, t3 - t2, t4 - t3 }) |*t, dt| {
        t.* = @as(f64, @floatFromInt(dt)) / 1e6;
    };
    return gpu;
}

/// A width sweep's window: a chain, a binary tree, or a main path with a side branch every third row.
const Shape = enum { chain, binary, dflash };
fn widths(gpa: std.mem.Allocator, m: *q.model.Model, rounds: usize, shape: Shape) !void {
    var r = try q.decode_round.Runner.init(gpa, m, 1, 4096);
    defer r.deinit();
    const vocab = m.config.vocab;
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*t, j| t.* = @intCast((j * 7919 + 13) % 150000 + 1000);
    const s = q.session.Session{ .runner = &r };
    try s.prefill(&prompt, 64);
    var pending = try s.greedy();
    var times: [64]f64 = undefined;
    const n = @min(rounds, times.len);
    for ([_]u32{ 1, 8, 16, 17, 24, 32, 48, 64, 96, 128 }) |rows| {
        var ids: [128]u32 = undefined;
        var parents: [128]i32 = undefined;
        var digest: [32]u8 = undefined;
        for (0..n + 2) |round| {
            ids[0] = pending;
            for (1..rows) |j| ids[j] = @intCast((j * 104729 + round * 31) % 150000 + 1000);
            for (parents[0..rows], 0..) |*p, j| p.* = if (j == 0) -1 else switch (shape) {
                .chain => @intCast(j - 1),
                .binary => @intCast((j - 1) / 2),
                .dflash => @intCast(if (j % 3 == 0) j - 2 else j - 1),
            };
            var plan = try Round.init(gpa, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = ids[0..rows], .parents = parents[0..rows] }}, @intCast(m.config.conv_kernel), r.slots, @intCast(vocab));
            defer plan.deinit();
            try r.verifyHead(&plan, .all);
            if (round >= 2) times[round - 2] = r.last_gpu_seconds * 1e3;
            std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(m.frame.get(.logits).buffer.slice(u16, rows * vocab)), &digest, .{});
            pending = try q.session.argmax(m.frame.get(.logits).buffer.slice(u16, vocab));
            const root = [_]u32{0};
            try r.keep(&plan, &.{&root});
        }
        std.mem.sort(f64, times[0..n], {}, std.sort.asc(f64));
        std.debug.print("{s} rows {d:>3}: verify {d:.2} ms GPU (median of {d}; min {d:.2}) logits {x}\n", .{ @tagName(shape), rows, times[n / 2], n, times[0], digest[0..6] });
        if (std.c.getenv("QWEN27_PROFILE") == null or (rows != 16 and rows != 17 and rows != 32)) continue;
        var trace = tf.gpu_profile.Trace{ .queue = m.queue };
        r.profile(&trace);
        trace.enabled = true;
        for (0..4) |round| {
            ids[0] = pending;
            for (1..rows) |j| ids[j] = @intCast((j * 104729 + round * 31) % 150000 + 1000);
            var plan = try Round.init(gpa, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = ids[0..rows], .parents = parents[0..rows] }}, @intCast(m.config.conv_kernel), r.slots, @intCast(vocab));
            defer plan.deinit();
            try r.verifyHead(&plan, .all);
            pending = try q.session.argmax(m.frame.get(.logits).buffer.slice(u16, vocab));
            const root = [_]u32{0};
            try r.keep(&plan, &.{&root});
        }
        r.profile(null);
        inline for (@typeInfo(tf.gpu_profile.Class).@"enum".field_names, 0..) |name, ci| if (trace.calls[ci] > 0) std.debug.print("  PROFILE rows {d} class={s} ms/round={d:.2}\n", .{ rows, name, trace.seconds[ci] * 1e3 / 4 });
        for (trace.shapes[0..trace.shape_count]) |sh| std.debug.print("  PROFILE rows {d} shape n={d} k={d} sk={d} ms/round={d:.2}\n", .{ rows, sh.n, sh.k, sh.sk, sh.gpu_seconds * 1e3 / 4 });
    }
}

/// Compare slot and snapshot scans at widths 2-48 after one prompt, checking every logit byte.
fn slotsCheck(gpa: std.mem.Allocator, m: *q.model.Model) !void {
    var runners: [2]q.decode_round.Runner = undefined;
    for (&runners) |*r| r.* = try q.decode_round.Runner.init(gpa, m, 1, 4096);
    defer for (&runners) |*r| r.deinit();
    runners[0].gdn.tree_mode = .snapshot;
    const vocab = m.config.vocab;
    var prompt: [64]u32 = undefined;
    for (&prompt, 0..) |*t, j| t.* = @intCast((j * 7919 + 13) % 150000 + 1000);
    var pending: u32 = 0;
    for (&runners) |*r| {
        const s = q.session.Session{ .runner = r };
        try s.prefill(&prompt, 64);
        pending = try s.greedy();
    }
    const saved = try gpa.alloc(u16, 48 * vocab);
    defer gpa.free(saved);
    var cases: usize = 0;
    for ([_]u32{ 2, 5, 8, 11, 16, 17, 24, 32, 48 }) |rows| for (0..3) |shape| {
        var ids: [48]u32 = undefined;
        var parents: [48]i32 = undefined;
        ids[0] = pending;
        for (1..rows) |j| ids[j] = @intCast((j * 104729 + rows * 7 + shape) % 150000 + 1000);
        for (parents[0..rows], 0..) |*p, j| p.* = if (j == 0) -1 else switch (shape) {
            0 => @as(i32, @intCast(j - 1)),
            1 => @intCast((j - 1) / 2),
            else => @intCast(if (j < 6) j - 1 else (j * 5) % j),
        };
        for (&runners, 0..) |*r, k| {
            var plan = try Round.init(gpa, &.{.{ .slot = 0, .start = r.offsets[0], .capacity = r.capacity, .ids = ids[0..rows], .parents = parents[0..rows] }}, @intCast(m.config.conv_kernel), r.slots, @intCast(vocab));
            defer plan.deinit();
            try r.verifyHead(&plan, .all);
            const logits = m.frame.get(.logits).buffer.slice(u16, rows * vocab);
            if (k == 0) @memcpy(saved[0 .. rows * vocab], logits) else if (!std.mem.eql(u16, saved[0 .. rows * vocab], logits)) {
                std.debug.print("slots DIFF: rows {d} shape {d}\n", .{ rows, shape });
                return error.SlotsDiffer;
            }
            if (k == 0) pending = try q.session.argmax(logits[0..vocab]);
            const root = [_]u32{0};
            try r.keep(&plan, &.{&root});
        }
        cases += 1;
    };
    std.debug.print("slots check: {d} rounds, every logit equal\n", .{cases});
}
