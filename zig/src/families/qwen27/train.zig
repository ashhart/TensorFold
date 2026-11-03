//! The shared Living Weights lesson gate over the 27B's actual Metal forward and native input-gradient backward.
const std = @import("std");
const mtl = @import("metal");
const living = @import("living");
const Model = @import("model.zig").Model;
const Config = @import("config.zig").Config;
const Runner = @import("decode_round.zig").Runner;
pub const Row = @import("train_row.zig").Row;
const to = @import("train_ops.zig");
const capture = @import("train_tape.zig");
const load = @import("train_load.zig");
const back = @import("train_back.zig");
const more = @import("train_more.zig");
const schedule = @import("train_schedule.zig");
const submit = @import("train_submit.zig");
const mode_api = living.train_mode;
pub const Turn = living.lesson_route.Turn;
pub const Mode = mode_api.Mode;
pub const Result = mode_api.Result;
pub const Seq = mode_api.Seq;
pub const Span = mode_api.Span;
pub const max_rows = 512;
pub const pack_rows = 1024;
pub const max_seqs = 8;
pub fn keepLow(layers: usize) usize {
    return layers / 5;
}

pub const Trainer = struct {
    gpa: std.mem.Allocator,
    io: ?std.Io = null,
    m: *Model,
    g: Config,
    dir: []const u8,
    k: to.Kernels = undefined,
    k_ready: bool = false,
    runner: Runner = undefined,
    runner_ready: bool = false,
    owned: std.ArrayList(mtl.Buffer) = .empty,
    follow: ?mtl.ResidencySet.Follow = null,
    w: []load.Layer = &.{},
    tape: []capture.Layer = &.{},
    head: to.Q4 = undefined,
    sc: back.Scratch = undefined,
    final_h: to.Ref = undefined,
    logits: to.Ref = undefined,
    dlogits: to.Ref = undefined,
    grad: to.Ref = undefined,
    dy: to.Ref = undefined,
    delta: to.Ref = undefined,
    push: to.Ref = undefined,
    targets: to.Ref = undefined,
    weights: to.Ref = undefined,
    stats: to.Ref = undefined,
    zero_history: to.Ref = undefined,
    spans: [max_seqs]Span = undefined,
    answers: [max_seqs]mode_api.Answer = undefined,
    seqs: [max_seqs]Seq = undefined,
    cp_first: [max_seqs]usize = @splat(0),
    nseq: usize = 0,
    rows: usize = 0,
    pack: usize = pack_rows,
    count: living.pocket_clock.Count = .{},
    prefixes: struct {
        pub const most = 0;
    } = .{},
    nudge: mode_api.Nudge = .{},
    from: [living.family.max_layers]u32 = @splat(std.math.maxInt(u32)),
    keys_top: ?usize = null,
    pocket_back: bool = true,
    mode: Mode = .keys,
    low: usize = 0,
    hook_at: usize = 0,
    local_row: usize = 0,
    slot: u32 = 0,
    reserved: u64 = 0,

    pub fn init(a: std.mem.Allocator, io: std.Io, m: *Model, dir: []const u8, follow: ?mtl.ResidencySet.Follow) !*Trainer {
        if (m.widths_tag != 0 or m.config.dk != 128 or m.config.dv != 128 or m.config.conv_kernel != 4 or m.config.head_dim != 256) return error.UnsupportedLearningShape;
        const t = try a.create(Trainer);
        t.* = .{ .gpa = a, .io = if (std.c.getenv("TF_Q27_LEARN_DIAG") != null) io else null, .m = m, .g = m.config, .dir = dir, .follow = follow, .reserved = m.learn_reserved };
        m.learn_reserved = 0; // The allocation replaces the reservation, under the lane host's exclusive learn phase.
        errdefer t.deinit();
        t.k = try to.Kernels.init(a, m.device);
        t.k_ready = true;
        t.runner = try Runner.init(a, m, max_seqs, max_rows + 1);
        t.runner_ready = true;
        t.runner.learning_rows = true;
        t.runner.learning_io = t.io;
        t.runner.setResident(follow);
        try load.weights(t, io);
        try capture.carve(t);
        const g = t.g;
        const R = pack_rows;
        t.final_h = try t.alloc(R * g.hidden * 2);
        t.logits = try t.alloc(R * g.vocab * 2);
        t.dlogits = try t.alloc(R * g.vocab * 4);
        t.grad = try t.alloc(R * g.hidden * 4);
        t.dy = try t.alloc(R * g.hidden * 4);
        const vectors = max_seqs * 3 * g.layers * g.hidden * 4;
        t.delta = try t.alloc(vectors);
        t.push = try t.alloc(vectors);
        t.targets = try t.alloc(R * 4);
        t.weights = try t.alloc(R * 4);
        t.stats = try t.alloc(R * 8);
        t.zero_history = try t.alloc(3 * g.gdnQkvDim() * 2);
        t.sc = try back.Scratch.init(t);
        var total: usize = 0;
        for (t.owned.items) |b| total += b.length();
        std.log.info("slide: the Metal trainer owns {d:.2} GiB beside its runner", .{@as(f64, @floatFromInt(total)) / (1 << 30)});
        return t;
    }

    pub fn deinit(t: *Trainer) void {
        t.drain();
        if (t.runner_ready) t.runner.deinit();
        for (t.owned.items) |buffer| {
            if (t.follow) |f| f.leave(f.ctx, buffer);
            buffer.deinit();
        }
        t.owned.deinit(t.gpa);
        t.gpa.free(t.w);
        t.gpa.free(t.tape);
        if (t.k_ready) t.k.deinit();
        t.m.learn_reserved = t.reserved;
        t.gpa.destroy(t);
    }
    pub fn drain(t: *Trainer) void {
        // Each forward/backward submission below waits before returning, including every error path.
        std.debug.assert(t.m.learning == null);
    }
    pub fn alloc(t: *Trainer, size: usize) !to.Ref {
        try t.owned.ensureUnusedCapacity(t.gpa, 1);
        const buffer = try t.m.device.buffer(@max(16, size), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(buffer.contents()[0..buffer.length()], 0);
        t.owned.appendAssumeCapacity(buffer);
        if (t.follow) |f| f.join(f.ctx, buffer);
        return .{ .buffer = buffer };
    }
    pub fn bytes(t: *Trainer, data: []const u8) !to.Ref {
        const ref = try t.alloc(data.len);
        @memcpy(ref.buffer.contents()[0..data.len], data);
        return ref;
    }
    pub fn vec(t: *const Trainer, s: usize, i: usize, l: usize) usize {
        return ((s * 3 + i) * t.g.layers + l) * t.g.hidden;
    }
    pub fn pushAt(t: *const Trainer, s: usize, i: usize, l: usize) []const f32 {
        return slice(f32, to.plus(t.push, t.vec(s, i, l) * 4), t.g.hidden);
    }
    pub fn deltaAt(t: *const Trainer, s: usize, i: usize, l: usize) []f32 {
        return slice(f32, to.plus(t.delta, t.vec(s, i, l) * 4), t.g.hidden);
    }
    pub fn normed(t: *const Trainer, l: usize, row: usize) []const u16 {
        return slice(u16, to.plus(t.tape[l].y, row * t.g.hidden * 2), t.g.hidden);
    }
    pub fn readScale(t: *const Trainer, l: usize, out: []f32) !void {
        if (out.len != t.g.hidden) return error.BadScale;
        for (out, t.m.weights.layers[l].post_norm.host(u16)) |*v, b| v.* = @bitCast(@as(u32, b) << 16);
    }
    pub fn residual(t: *Trainer, l: usize, row: usize) []const u16 {
        return more.residual(t, l, row);
    }
    pub fn packs(_: *const Trainer) bool {
        return true;
    }
    pub fn prefixSlots(_: *Trainer, n: usize) !void {
        if (n != 0) return error.TooManyPrefixes;
    }
    pub fn greedy(t: *Trainer, seqs: []const Seq, same: []bool) !void {
        return more.greedy(t, seqs, same);
    }
    pub fn tapRows(t: *Trainer, ids: []const u32, context: usize, rows: []const u32, layers: []const usize, out: []u16) !void {
        return more.tapRows(t, ids, context, rows, layers, out);
    }
    pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: Mode, turn: living.lesson_route.Turn) !Result {
        var out: [1]Result = undefined;
        try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
        return out[0];
    }

    pub fn pass(t: *Trainer, seqs: []const Seq, mode: Mode, _: living.lesson_route.Turn, out: []Result) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const began = submit.now(t.io);
        t.runner.learning_stats = .{};
        var forward_seconds: f64 = 0;
        var back_encode: f64 = 0;
        var back_wait: f64 = 0;
        var back_gpu: f64 = 0;
        var affine_work: to.Work = .{};
        if (seqs.len == 0 or seqs.len > max_seqs or out.len < seqs.len) return error.BadPass;
        switch (mode) {
            .keys, .loss, .target => {},
            .prime => {
                for (out[0..seqs.len]) |*r| r.* = .{ .loss = 0, .recalled = false };
                return;
            },
            else => return error.RoutedUnsupported,
        }
        var rows: usize = 0;
        var checkpoints: usize = 0;
        var low = t.g.layers;
        for (seqs, 0..) |sq, si| {
            if (sq.ids.len < 2 or sq.start < 1 or sq.start >= sq.ids.len or sq.ids.len - 1 > max_rows) return error.BadExample;
            if (sq.skip != 0 or !std.math.isFinite(sq.scale)) return error.BadPrefix;
            const n = sq.ids.len - 1;
            t.spans[si] = .{ .at = rows, .rows = n };
            t.cp_first[si] = checkpoints;
            checkpoints += (n + 3) / 4;
            rows += n;
            for (sq.nudge.layers, sq.nudge.rows) |mask, row| {
                if (mask == 0) continue;
                if (row >= n or (t.g.layers < 64 and mask >> @intCast(t.g.layers) != 0)) return error.BadNudge;
                if (mode == .target) low = @min(low, @ctz(mask));
            }
        }
        if (rows > t.pack) return error.PassTooLong;
        if (low < keepLow(t.g.layers)) return error.NudgeBelowKept;
        t.low = if (mode == .target and low == t.g.layers) keepLow(t.g.layers) else low;
        t.mode = mode;
        t.nseq = seqs.len;
        t.rows = rows;
        @memcpy(t.seqs[0..seqs.len], seqs);
        t.count = .{ .passes = t.count.passes + 1, .rows = t.count.rows + rows };
        @memset(slice(u32, t.targets, rows), 0);
        @memset(slice(f32, t.weights, rows), 0);
        _ = mode_api.answers(seqs, t.spans[0..seqs.len], t.answers[0..seqs.len]);
        for (seqs, t.answers[0..seqs.len], 0..) |sq, a, si| {
            for (0..a.rows) |row| {
                slice(u32, t.targets, rows)[a.first + row] = sq.ids[sq.start + row];
                slice(f32, t.weights, rows)[a.first + row] = 1 / @as(f32, @floatFromInt(a.rows));
            }
            try t.runner.reset(@intCast(si));
        }
        const previous = t.m.kernels.prompt;
        t.m.kernels.prompt = false;
        defer t.m.kernels.prompt = previous;
        std.debug.assert(t.m.learning == null);
        t.m.learning = .{ .ctx = t, .save = capture.save, .add = add };
        defer t.m.learning = null;
        const forward_start = submit.now(t.io);
        for (seqs, t.spans[0..seqs.len], 0..) |sq, sp, si| for (0..sp.rows) |row| {
            t.hook_at = sp.at + row;
            t.local_row = row;
            t.slot = @intCast(si);
            try t.forwardRow(&t.runner, @intCast(si), row, sq.ids[row..][0..1], schedule.head(mode, row, sq.start));
        };
        forward_seconds = submit.now(t.io) - forward_start;
        // Remove the hook before backward and before any caller can observe completion.
        t.m.learning = null;
        if (mode != .keys) {
            const back_start = submit.now(t.io);
            const cb = t.m.queue.commandBuffer();
            const encoder = cb.compute(.serial);
            var ended = false;
            defer if (!ended) encoder.end();
            const o = to.Ops{ .kernels = &t.k, .e = encoder, .work = if (t.io != null) &affine_work else null };
            try schedule.loss(t, o);
            if (mode == .target and low < t.g.layers) try back.run(t, o);
            encoder.end();
            ended = true;
            back_encode = submit.now(t.io) - back_start;
            const wait_start = submit.now(t.io);
            cb.commit();
            cb.wait();
            back_wait = submit.now(t.io) - wait_start;
            back_gpu = if (t.io != null) cb.gpuSeconds() else 0;
            if (@import("decode_round.zig").gpuFailed(cb, "learning backward")) return error.LearningGpuFailure;
        }
        if (t.io != null) {
            const f = t.runner.learning_stats;
            var answer_rows: usize = 0;
            for (t.answers[0..seqs.len]) |a| answer_rows += a.rows;
            std.log.info("slide qwen27: pass {d} {t}, {d} sequences, {d} rows, {d} answer rows: wall {d:.3} s; forward {d:.3} s (plan/upload {d:.3}, encode {d:.3}, commit/wait {d:.3}, GPU {d:.3}); backward/loss encode {d:.3}, commit/wait {d:.3}, GPU {d:.3}; row buffers/waits {d}/{d}, resets {d}, tail {d}; affine decodes {d}, FMAs {d}", .{ t.count.passes, mode, seqs.len, rows, answer_rows, submit.now(t.io) - began, forward_seconds, @max(0, forward_seconds - f.encode - f.wait), f.encode, f.wait, f.gpu, back_encode, back_wait, back_gpu, f.buffers, f.waits, seqs.len, @intFromBool(mode != .keys), affine_work.decodes, affine_work.fmas });
        }
        for (out[0..seqs.len], t.answers[0..seqs.len]) |*r, a| {
            r.* = .{ .loss = 0, .recalled = mode != .keys };
            if (mode != .keys) for (slice([2]f32, to.plus(t.stats, a.first * 8), a.rows)) |stat| {
                if (!std.math.isFinite(stat[0]) or !std.math.isFinite(stat[1])) return error.NonFiniteLearningLoss;
                r.loss += stat[0] / @as(f32, @floatFromInt(a.rows));
                r.recalled = r.recalled and stat[1] > 0.5;
            };
        }
    }
    pub fn forwardRow(t: *Trainer, runner: *Runner, slot: u32, at: usize, ids: []const u32, logits: bool) !void {
        if (ids.len != 1) return error.BadLearningRows;
        var row: Row = .{};
        var round = try row.view(t.gpa, ids[0], slot, @intCast(at), runner.capacity, @intCast(t.g.conv_kernel), runner.slots, @intCast(t.g.vocab));
        try runner.advance(&round, if (logits) .all else .none);
    }
};
pub fn slice(comptime T: type, ref: to.Ref, n: usize) []T {
    return @as([*]T, @ptrCast(@alignCast(ref.buffer.contents() + ref.offset)))[0..n];
}
fn add(ctx: *anyopaque, e: mtl.ComputeEncoder, l: usize, x: to.Ref, out: to.Ref, rows: u32) !void {
    const t: *Trainer = @ptrCast(@alignCast(ctx));
    const o = to.Ops{ .kernels = &t.k, .e = e };
    if (rows != 1) return error.BadLearningRows;
    const sq = t.seqs[t.slot];
    if (t.m.pockets) |p| if (p.layers[l]) |pk| if (pk.used > 0) {
        const pipeline = t.k.pipelines.get("tf_q27_pocket_scaled").?;
        e.setPipeline(pipeline);
        e.setBuffer(x.buffer, x.offset, 0);
        e.setBuffer(pk.weights, 0, 1);
        e.setBuffer(out.buffer, out.offset, 2);
        e.setValue(@Vector(4, u32){ @intCast(t.g.hidden), @intCast(pk.used), @intCast(pk.cap), t.from[l] }, 3);
        e.setValue(sq.scale, 4);
        e.dispatchGroups(.{ .width = 1 }, .{ .width = 256 });
        e.barrier();
    };
    for (sq.nudge.layers, sq.nudge.rows, 0..) |mask, row, i| {
        if (mask >> @intCast(l) & 1 != 0 and t.local_row == row) try o.nudge(out, to.plus(t.delta, t.vec(t.slot, i, l) * 4), t.g.hidden);
    }
}
