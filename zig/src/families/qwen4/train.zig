//! A bounded native learner tapes fresh causal sequences and dispatches analytic pullbacks on the serving backend.
const std = @import("std");
const core = @import("core");
const living = @import("living");
const tm = living.train_mode;
const S = @import("train_buffers.zig").Slot;
pub fn Trainer(comptime E: type, comptime K: type) type {
    const B = E.Storage;
    const Plan = @import("train_buffers.zig").Plan(B);
    const Tape = @import("train_tape.zig").Tape(B);
    const Ref = @import("train_tensor.zig").Ref(B);
    const Runtime = @import("learning_runtime.zig").Runtime(E, K);
    return struct {
        const Self = @This();
        pub const Context = *Runtime;
        pub const max_rows = @import("train_budget.zig").max_rows;
        pub const max_seqs = 1;
        a: std.mem.Allocator,
        e: *E,
        runtime: *Runtime,
        tape: Tape,
        plan: Plan,
        buffer: B.Buffer,
        host: std.heap.ArenaAllocator,
        inputs: []u16,
        delta: []f32,
        push: []f32,
        targets: []u32,
        weights: []f32,
        stats: [][2]f32,
        resid: []u16,
        wide: []u16,
        inject: []u16,
        flags: []i32,
        ids: []i32,
        retained: Runtime.Retained = undefined,
        spans: [1]tm.Span = undefined,
        seq: tm.Seq = undefined,
        rows: usize = 0,
        capture_back: bool = false,
        pack: usize,
        count: living.pocket_clock.Count = .{},
        prefixes: struct {
            pub const most = 0;
        } = .{},
        nudge: tm.Nudge = .{},
        from: [living.family.max_layers]u32 = @splat(std.math.maxInt(u32)),
        pub fn init(a: std.mem.Allocator, _: std.Io, e: *E, pockets: anytype, runtime: Context) !*Self {
            if (pockets != &runtime.pockets or e != runtime.e) return error.Qwen4ForeignTrainer;
            var problem: @import("config.zig").Problem = .{};
            try @import("learning_profile.zig").checkpoint(&e.c, &e.reader, &problem);
            const pack: usize = @min(max_rows, e.o.capacity);
            if (pack < 2) return error.Qwen4TrainingRows;
            var tape = try Tape.describe(a, e.backend, &e.c, pack);
            errdefer tape.deinitDrained();
            const plan = try Plan.make(&e.c, pack);
            const c = &e.c;
            var gp: @import("config.zig").Problem = .{};
            const geom = try @import("cuda_attention.zig").Geometry.of(c, &gp);
            const idw = try geom.ids();
            const charge = try @import("train_budget.zig").bytes(B, a, c, e.o.capacity);
            try runtime.reserveTraining(charge);
            errdefer runtime.releaseTraining();
            try tape.allocate(tape.bytes);
            const buffer = try e.backend.allocate(plan.bytes);
            errdefer e.backend.release(buffer);
            var host: std.heap.ArenaAllocator = .init(a);
            errdefer host.deinit();
            const h = host.allocator();
            const inputs = try h.alloc(u16, c.layers * pack * c.hidden);
            const delta = try h.alloc(f32, 3 * c.layers * c.hidden);
            @memset(delta, 0);
            const push = try h.alloc(f32, delta.len);
            const targets = try h.alloc(u32, pack);
            const weights = try h.alloc(f32, pack);
            const stats = try h.alloc([2]f32, pack);
            const resid = try h.alloc(u16, c.hidden);
            const wide = try h.alloc(u16, c.wide());
            const inject = try h.alloc(u16, c.streams);
            const flags = try h.alloc(i32, pack * 2);
            const ids = try h.alloc(i32, idw);
            const t = try a.create(Self);
            t.* = .{ .a = a, .e = e, .runtime = runtime, .tape = tape, .plan = plan, .buffer = buffer, .host = host, .inputs = inputs, .delta = delta, .push = push, .targets = targets, .weights = weights, .stats = stats, .resid = resid, .wide = wide, .inject = inject, .flags = flags, .ids = ids, .pack = pack };
            t.retained = .{ .ctx = t, .drop = drop };
            return t;
        }
        pub fn drain(t: *Self) void {
            t.e.backend.drain() catch {
                t.e.fail();
            };
        }
        pub fn deinit(t: *Self) void {
            t.e.backend.drain() catch {
                t.runtime.retain(&t.retained);
                return;
            };
            drop(t);
        }
        fn drop(ctx: *anyopaque) void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            t.tape.deinitDrained();
            t.e.backend.release(t.buffer);
            t.runtime.releaseTraining();
            t.host.deinit();
            t.a.destroy(t);
        }
        pub fn buf(t: *Self, slot: S) Ref {
            return t.plan.get(t.buffer, slot);
        }
        pub fn ops(t: *Self) @import("train_encode.zig").Ops(E, K) {
            return .{ .e = t.e, .k = &t.runtime.kernels, .dummy = t.buf(.dummy) };
        }
        pub fn packs(_: *Self) bool {
            return false;
        }
        pub fn prefixSlots(_: *Self, n: usize) !void {
            if (n != 0) return error.TooManyPrefixes;
        }
        pub fn normed(t: *Self, l: usize, row: usize) []const u16 {
            return t.inputs[(l * t.pack + row) * t.e.c.hidden ..][0..t.e.c.hidden];
        }
        fn vector(t: *Self, key: usize, l: usize) usize {
            return (key * t.e.c.layers + l) * t.e.c.hidden;
        }
        pub fn pushAt(t: *Self, seq: usize, key: usize, l: usize) []const f32 {
            std.debug.assert(seq == 0);
            return t.push[t.vector(key, l)..][0..t.e.c.hidden];
        }
        pub fn deltaAt(t: *Self, seq: usize, key: usize, l: usize) []f32 {
            std.debug.assert(seq == 0);
            return t.delta[t.vector(key, l)..][0..t.e.c.hidden];
        }
        fn read(t: *Self, r: Ref, bytes: []u8) !void {
            try t.runtime.kernels.read(t.e.backend, r, bytes);
        }
        fn upload(t: *Self, r: Ref, bytes: []const u8) !void {
            if (bytes.len > r.len) return error.Qwen4TrainingRange;
            try t.e.backend.write(r.buffer, r.offset, bytes);
        }
        pub fn residual(t: *Self, l: usize, row: usize) []const u16 {
            @memset(t.resid, 0);
            if (l == 0 or l > t.e.c.layers or row >= t.rows) return t.resid;
            const ref = t.tape.indexed(l - 1, "mlp_hyper_connection.residual", t.rows) catch return t.resid;
            const inj = t.tape.indexed(l - 1, "mlp_hyper_connection.injection", t.rows) catch return t.resid;
            t.read(ref.plus(row * t.wide.len * 2, t.wide.len * 2) catch return t.resid, std.mem.sliceAsBytes(t.wide)) catch {
                t.e.fail();
                return t.resid;
            };
            t.read(inj.plus(row * t.inject.len * 2, t.inject.len * 2) catch return t.resid, std.mem.sliceAsBytes(t.inject)) catch {
                t.e.fail();
                return t.resid;
            };
            var square: f32 = 0;
            for (t.inject) |v| {
                const x: f32 = @bitCast(@as(u32, v) << 16);
                square += x * x;
            }
            if (square == 0) return t.resid;
            for (t.resid, 0..) |*v, i| {
                var sum: f32 = 0;
                for (t.inject, 0..) |inj_, s| sum += @as(f32, @bitCast(@as(u32, inj_) << 16)) * @as(f32, @bitCast(@as(u32, t.wide[s * t.resid.len + i]) << 16));
                v.* = core.quant.convert.bf16(sum / square);
            }
            return t.resid;
        }
        pub fn step(t: *Self, ids: []const u32, start: usize, mode: tm.Mode, turn: living.lesson_route.Turn) !tm.Result {
            var out: [1]tm.Result = undefined;
            try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
            return out[0];
        }
        pub fn pass(t: *Self, seqs: []const tm.Seq, mode: tm.Mode, _: living.lesson_route.Turn, out: []tm.Result) !void {
            if (seqs.len != 1 or out.len < 1) return error.BadPass;
            if (mode == .prime) {
                out[0] = .{ .loss = 0, .recalled = false };
                return;
            }
            if (mode != .keys and mode != .loss and mode != .target) return error.RoutedUnsupported;
            const seq = seqs[0];
            if (seq.ids.len < 2 or seq.start < 1 or seq.start >= seq.ids.len or seq.ids.len - 1 > t.pack or seq.decide >= seq.ids.len - 1 or !std.math.isFinite(seq.scale)) return error.BadExample;
            if (seq.skip != 0) return error.BadPrefix;
            for (seq.ids) |token| if (token >= t.e.c.vocab) return error.TokenOutOfRange;
            const rows = seq.ids.len - 1;
            for (seq.nudge.layers, seq.nudge.rows) |mask, row| {
                if (t.e.c.layers < 64 and mask >> @intCast(t.e.c.layers) != 0) return error.BadExample;
                if (mask != 0 and row >= rows) return error.BadExample;
            }
            t.rows = rows;
            t.seq = seq;
            t.spans[0] = .{ .at = 0, .rows = rows };
            t.tape.reset(if (mode == .keys) .keys else if (mode == .target or t.capture_back) .back else .loss);
            try t.e.backend.drain();
            errdefer {
                t.e.gpu.discard();
                t.e.backend.drain() catch t.e.fail();
            }
            try t.e.gpu.begin();
            try t.upload(t.buf(.delta), std.mem.sliceAsBytes(t.delta));
            try t.e.gpu.submit();
            try t.e.backend.drain();
            try t.forward();
            for (0..t.e.c.layers) |l| {
                const ref = try t.tape.indexed(l, "mlp_hyper_connection.mix", rows);
                try t.read(ref, std.mem.sliceAsBytes(t.inputs[l * t.pack * t.e.c.hidden ..][0 .. rows * t.e.c.hidden]));
            }
            t.count.passes += 1;
            t.count.rows += rows;
            out[0] = .{ .loss = 0, .recalled = false };
            if (mode == .keys) return;
            if (mode == .target) try t.selections();
            @memset(t.targets, 0);
            @memset(t.weights, 0);
            const answer = seq.ids.len - seq.start;
            for (seq.start - 1..rows) |r| {
                t.targets[r] = seq.ids[r + 1];
                t.weights[r] = 1 / @as(f32, @floatFromInt(answer));
            }
            try t.e.gpu.begin();
            try t.upload(t.buf(.targets), std.mem.sliceAsBytes(t.targets));
            try t.upload(t.buf(.weights), std.mem.sliceAsBytes(t.weights));
            var p: core.living_rows.Params = .{};
            p.u[0] = @intCast(rows);
            p.u[1] = @intCast(t.e.c.vocab);
            try t.ops().row(.loss, p, &.{ .{ 0, try t.tape.get("head", rows) }, .{ 1, t.buf(.targets) }, .{ 2, t.buf(.weights) }, .{ 8, t.buf(.logits) }, .{ 9, t.buf(.stats) } });
            if (mode == .target) try @import("train_back.zig").run(t);
            try t.e.gpu.submit();
            try t.e.backend.drain();
            try t.read(t.buf(.stats), std.mem.sliceAsBytes(t.stats[0..rows]));
            var total: f64 = 0;
            var recalled = true;
            for (t.stats[seq.start - 1 .. rows]) |r| {
                if (!std.math.isFinite(r[0]) or !std.math.isFinite(r[1])) return error.NonFinite;
                total += r[0];
                recalled = recalled and r[1] == 1;
            }
            if (mode == .target) {
                try t.read(t.buf(.push), std.mem.sliceAsBytes(t.push));
                for (t.push) |value| if (!std.math.isFinite(value)) return error.NonFinite;
            }
            out[0] = .{ .loss = @floatCast(total / @as(f64, @floatFromInt(answer))), .recalled = recalled };
        }
        fn forward(t: *Self) !void {
            const e = t.e;
            const slot = try e.lease();
            defer e.release(slot) catch e.fail();
            const old_trace = e.trace;
            const old_gpu_trace = e.gpu.trace;
            const old_hook = e.learning;
            defer {
                e.trace = old_trace;
                e.gpu.trace = old_gpu_trace;
                e.learning = old_hook;
            }
            e.trace = t.tape.sink();
            e.gpu.trace = e.trace;
            e.learning = .{ .ctx = t, .begin = begin, .dense = dense };
            var first: usize = 0;
            var problem: @import("config.zig").Problem = .{};
            while (first < t.rows) {
                const n = @min(e.o.window, t.rows - first);
                t.tape.first = first;
                try e.forward(&.{.{ .state = &slot.state, .tokens = t.seq.ids[first..][0..n] }}, E.NoDraw{}, &problem);
                try e.keep(slot, n);
                try e.backend.drain();
                first += n;
            }
        }
        fn begin(ctx: *anyopaque, runs: []const @import("learning_hook.zig").Run, count: usize) !void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            try t.runtime.prepare(runs, count);
        }
        fn dense(ctx: *anyopaque, _: *E.Gpu, l: usize, input: E.Tensor, output: E.Tensor) !void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            try t.runtime.apply(l, input, output, t.seq.scale, t.from[l]);
            for (t.seq.nudge.layers, t.seq.nudge.rows, 0..) |mask, row, key| {
                if (mask & (@as(u64, 1) << @intCast(l)) == 0 or row < t.tape.first or row - t.tape.first >= output.rows) continue;
                const local = row - t.tape.first;
                var p: core.living_rows.Params = .{};
                p.u[0] = 1;
                p.u[1] = @intCast(t.e.c.hidden);
                const tensor = try output.subRows(local, 1);
                const ref: Ref = .{ .buffer = tensor.buffer, .offset = tensor.offset, .len = try tensor.bytes() };
                try t.ops().row(.nudge, p, &.{ .{ 0, ref }, .{ 1, try t.buf(.delta).plus(t.vector(key, l) * 4, t.e.c.hidden * 4) }, .{ 8, ref } });
            }
        }
        fn selections(t: *Self) !void {
            const idw = t.ids.len;
            for (0..t.e.c.layers) |l| {
                if (t.e.c.kinds[l] != .indexed_attention) continue;
                try t.read(try t.tape.indexed(l, "self_attn.selection_count", t.rows), std.mem.sliceAsBytes(t.flags[0..t.rows]));
                try t.read(try t.tape.indexed(l, "self_attn.selection_sparse", t.rows), std.mem.sliceAsBytes(t.flags[t.pack..][0..t.rows]));
                for (t.flags[0..t.rows], t.flags[t.pack..][0..t.rows], 0..) |count, sparse, row| {
                    if (sparse != 0 and sparse != 1) return error.Qwen4TrainingSelection;
                    if (count < 1 or count > idw or count > row + 1) return error.Qwen4TrainingSelection;
                    if (sparse == 0) {
                        if (count != row + 1) return error.Qwen4TrainingSelection;
                        continue;
                    }
                    const ref = try t.tape.raw(l, "self_attn.selected_ids", t.rows);
                    const n: usize = @intCast(count);
                    try t.read(try ref.plus(row * idw * 4, n * 4), std.mem.sliceAsBytes(t.ids[0..n]));
                    for (t.ids[0..n], 0..) |id, i| {
                        if (id < 0 or id > row or (i > 0 and id <= t.ids[i - 1])) return error.Qwen4TrainingSelection;
                    }
                }
            }
        }
    };
}
