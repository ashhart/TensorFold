//! A bounded native trainer records fresh causal sequences and exposes the common learner's rows and loss adjoints.
const std = @import("std");
const core = @import("core");
const living = @import("living");
const tm = living.train_mode;
const hook = @import("learning_hook.zig");
const buffers = @import("learning_buffers.zig");
pub fn Trainer(comptime E: type) type {
    const R = E.Runtime;
    const B = R.Backend;
    const Ref = R.Ref;
    const Plan = buffers.Plan(B);
    return struct {
        const Self = @This();
        pub const Context = *E;
        pub const max_rows = buffers.max_rows;
        pub const max_seqs = 1;
        a: std.mem.Allocator,
        e: *E,
        r: *R,
        shape: @import("learning_profile.zig").Shape,
        plan: Plan,
        buffer: B.Buffer,
        native: E.Train,
        reader: core.Checkpoint,
        host: std.heap.ArenaAllocator,
        inputs: []u16,
        delta: []f32,
        push: []f32,
        targets: []u32,
        weights: []f32,
        stats: [][2]f32,
        resid: []u16,
        retained: R.Retained = undefined,
        spans: [1]tm.Span = undefined,
        seq: tm.Seq = undefined,
        encoding: B.Pass = undefined,
        rows: usize = 0,
        first: usize = 0,
        written: usize = 0,
        capture_back: bool = false,
        seen: [61]u32 = @splat(0),
        pack: usize,
        count: living.pocket_clock.Count = .{},
        prefixes: struct {
            pub const most = 0;
        } = .{},
        nudge: tm.Nudge = .{},
        from: [living.family.max_layers]u32 = @splat(std.math.maxInt(u32)),
        pub fn init(a: std.mem.Allocator, io: std.Io, e: *E, pockets: anytype, context: Context) !*Self {
            if (e != context or pockets != &e.r.pockets or e.r.failed) return error.GemmaForeignTrainer;
            const plan = try Plan.make(e.r.shape, max_rows);
            const charge = try std.math.add(u64, try B.allocationBytes(plan.bytes), try std.math.add(u64, try plan.hostBytes(), try e.trainBytes(max_rows)));
            try e.r.reserve(charge);
            var transferred = false;
            errdefer if (!transferred) e.r.unreserve();
            e.r.backend.drain() catch |err| {
                e.r.failed = true;
                return err;
            };
            try e.r.kernels.ensure(e.r.backend);
            var reader = try core.Checkpoint.openModel(a, io, e.dir);
            errdefer if (!transferred) reader.close();
            const buffer = try e.r.backend.allocate(plan.bytes);
            errdefer if (!transferred) e.r.backend.release(buffer);
            var host: std.heap.ArenaAllocator = .init(a);
            errdefer if (!transferred) host.deinit();
            const h = host.allocator();
            const s = e.r.shape;
            const inputs = try h.alloc(u16, s.layers * max_rows * s.hidden);
            const delta = try h.alloc(f32, 3 * s.layers * s.hidden);
            @memset(delta, 0);
            const push = try h.alloc(f32, delta.len);
            const targets = try h.alloc(u32, max_rows);
            const weights = try h.alloc(f32, max_rows);
            const stats = try h.alloc([2]f32, max_rows);
            const resid = try h.alloc(u16, s.hidden);
            const t = try a.create(Self);
            errdefer if (!transferred) a.destroy(t);
            t.* = .{ .a = a, .e = e, .r = e.r, .shape = s, .plan = plan, .buffer = buffer, .native = undefined, .reader = reader, .host = host, .inputs = inputs, .delta = delta, .push = push, .targets = targets, .weights = weights, .stats = stats, .resid = resid, .pack = max_rows };
            t.retained = .{ .ctx = t, .drop = drop };
            t.native = try e.trainOpen(max_rows);
            e.r.backend.drain() catch |err| {
                transferred = true;
                e.r.retain(&t.retained);
                return err;
            };
            return t;
        }
        pub fn drain(t: *Self) void {
            t.r.backend.drain() catch {
                t.r.failed = true;
            };
        }
        pub fn deinit(t: *Self) void {
            if (!t.r.closing_completed) t.r.backend.drain() catch {
                t.r.retain(&t.retained);
                return;
            };
            drop(t);
        }
        fn drop(ctx: *anyopaque) void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            t.native.deinit();
            t.r.backend.release(t.buffer);
            t.reader.close();
            t.r.unreserve();
            t.host.deinit();
            t.a.destroy(t);
        }
        pub fn buf(t: *Self, slot: buffers.Slot) Ref {
            return t.plan.buf(t.buffer, slot);
        }
        pub fn vec(t: *Self, slot: buffers.Slot, width: usize) Ref {
            return t.buf(slot).plus(0, t.rows * width * 4) catch unreachable;
        }
        pub fn taped(t: *Self, layer: ?usize, node: hook.Node) !Ref {
            return t.plan.taped(t.buffer, layer, node, 0, t.rows);
        }
        pub fn ops(t: *Self) @import("learning_ops.zig").Ops(R) {
            return .{ .r = t.r, .pass = t.encoding, .dummy = t.buf(.dummy) };
        }
        pub fn write(t: *Self, v: Ref, bytes: []const u8) !void {
            if (bytes.len > v.len) return error.GemmaTrainingRange;
            try t.r.backend.write(v.buffer, v.offset, bytes);
        }
        fn read(t: *Self, v: Ref, bytes: []u8) !void {
            if (bytes.len > v.len) return error.GemmaTrainingRange;
            try t.r.backend.read(v.buffer, v.offset, bytes);
        }
        pub fn packs(_: *Self) bool {
            return false;
        }
        pub fn prefixSlots(_: *Self, n: usize) !void {
            if (n != 0) return error.TooManyPrefixes;
        }
        pub fn normed(t: *Self, l: usize, row: usize) []const u16 {
            return t.inputs[(l * t.pack + row) * t.shape.hidden ..][0..t.shape.hidden];
        }
        pub fn vector(t: *Self, key: usize, l: usize) usize {
            return (key * t.shape.layers + l) * t.shape.hidden;
        }
        pub fn pushAt(t: *Self, seq: usize, key: usize, l: usize) []const f32 {
            std.debug.assert(seq == 0);
            return t.push[t.vector(key, l)..][0..t.shape.hidden];
        }
        pub fn deltaAt(t: *Self, seq: usize, key: usize, l: usize) []f32 {
            std.debug.assert(seq == 0);
            return t.delta[t.vector(key, l)..][0..t.shape.hidden];
        }
        pub fn residual(t: *Self, l: usize, row: usize) []const u16 {
            @memset(t.resid, 0);
            if (l == 0 or l > t.shape.layers or row >= t.rows) return t.resid;
            const v = t.plan.taped(t.buffer, l - 1, .mlp_base, row, 1) catch return t.resid;
            t.read(v, std.mem.sliceAsBytes(t.resid)) catch {
                t.r.failed = true;
            };
            return t.resid;
        }
        pub fn step(t: *Self, ids: []const u32, start: usize, mode: tm.Mode, turn: living.lesson_route.Turn) !tm.Result {
            var out: [1]tm.Result = undefined;
            try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide, .nudge = t.nudge }}, mode, turn, &out);
            return out[0];
        }
        pub fn pass(t: *Self, seqs: []const tm.Seq, mode: tm.Mode, _: living.lesson_route.Turn, out: []tm.Result) !void {
            if (seqs.len != 1 or out.len != 1 or t.r.failed) return error.GemmaTrainingSequences;
            const seq = seqs[0];
            if (mode == .prime) {
                out[0] = .{ .loss = 0, .recalled = false };
                return;
            }
            if (mode != .keys and mode != .loss and mode != .target) return error.RoutedUnsupported;
            if (seq.ids.len < 2 or seq.start < 1 or seq.start >= seq.ids.len or seq.ids.len - 1 > t.pack or seq.decide >= seq.ids.len - 1 or !std.math.isFinite(seq.scale)) return error.BadExample;
            if (seq.skip != 0 or seq.slot != 0) return error.BadPrefix;
            for (seq.ids) |id| if (id >= t.shape.vocab) return error.TokenOutOfRange;
            const rows = seq.ids.len - 1;
            for (seq.nudge.layers, seq.nudge.rows) |mask, row| {
                if (mask >> @intCast(t.shape.layers) != 0 or (mask != 0 and row >= rows)) return error.BadExample;
            }
            t.spans[0] = .{ .at = 0, .rows = rows };
            t.seq = seq;
            t.rows = rows;
            t.capture_back = mode == .target;
            errdefer {
                t.r.backend.abort();
                t.r.backend.drain() catch {
                    t.r.failed = true;
                };
            }
            t.encoding = try t.r.backend.begin();
            try t.write(t.buf(.delta), std.mem.sliceAsBytes(t.delta));
            try t.r.backend.submit(t.encoding);
            try t.r.backend.drain();
            const old = t.r.observer;
            defer t.r.observer = old;
            t.r.observer = .{ .ctx = t, .begin = begin, .note = note, .dense = dense };
            t.written = 0;
            try t.native.forward(t.r.listener(), seq.ids[0..rows]);
            if (t.written != rows) return error.GemmaTrainingTape;
            for (0..t.shape.layers) |l| try t.read(try t.taped(l, .mlp_input), std.mem.sliceAsBytes(t.inputs[l * t.pack * t.shape.hidden ..][0 .. rows * t.shape.hidden]));
            t.count.passes += 1;
            t.count.rows += rows;
            out[0] = .{ .loss = 0, .recalled = false };
            if (mode == .keys) return;
            @memset(t.targets, 0);
            @memset(t.weights, 0);
            const answer = seq.ids.len - seq.start;
            for (seq.start - 1..rows) |row| {
                t.targets[row] = seq.ids[row + 1];
                t.weights[row] = 1 / @as(f32, @floatFromInt(answer));
            }
            t.encoding = try t.r.backend.begin();
            try t.write(t.buf(.targets), std.mem.sliceAsBytes(t.targets));
            try t.write(t.buf(.weights), std.mem.sliceAsBytes(t.weights));
            try t.ops().zero(t.buf(.sparse));
            var p: core.living_rows.Params = .{};
            p.u[0] = @intCast(rows);
            p.u[1] = @intCast(t.shape.vocab);
            try t.ops().row(.loss, p, &.{ .{ 0, try t.taped(null, .head) }, .{ 1, t.buf(.targets) }, .{ 2, t.buf(.weights) }, .{ 8, t.buf(.logits) }, .{ 9, t.buf(.stats) } });
            if (mode == .target) try @import("learning_back.zig").run(t);
            try t.r.backend.submit(t.encoding);
            try t.r.backend.drain();
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
        fn begin(ctx: *anyopaque, runs: []const hook.Run, rows: usize) !void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            if (runs.len != 1 or runs[0].untaught or runs[0].first != 0 or runs[0].position != t.written or rows > t.rows -| t.written) return error.GemmaTrainingTape;
            t.first = runs[0].position;
            t.written += rows;
            @memset(&t.seen, 0);
        }
        fn note(ctx: *anyopaque, encoder: B.Pass, l: ?usize, node: hook.Node, v: R.View) !void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const width = buffers.width(t.shape, l, node);
            if (width == 0 or (!t.capture_back and node != .mlp_input and node != .mlp_base and node != .head and node != .head_raw)) return;
            if (v.width != width) return error.GemmaTrainingTape;
            const at = l orelse t.shape.layers;
            const bit = @as(u32, 1) << @intCast(@backingInt(node));
            if (t.seen[at] & bit != 0) return error.GemmaTrainingTape;
            t.seen[at] |= bit;
            var p: core.living_rows.Params = .{};
            p.u[0] = @intCast(v.rows);
            p.u[1] = @intCast(width);
            p.u[2] = @intCast(v.stride);
            p.u[3] = @backingInt(v.layout);
            p.u[4] = @intCast(v.dim);
            const ops_: @import("learning_ops.zig").Ops(R) = .{ .r = t.r, .pass = encoder, .dummy = t.buf(.dummy) };
            try ops_.gemma(.copy, p, &.{ .{ 0, .{ .buffer = v.buffer, .offset = v.offset, .len = v.len } }, .{ 8, try t.plan.taped(t.buffer, l, node, t.first, v.rows) } });
            if (node == .head and t.capture_back) for (0..t.shape.layers + 1) |layer| inline for (std.enums.values(hook.Node)) |required| {
                if (buffers.width(t.shape, if (layer == t.shape.layers) null else layer, required) != 0 and t.seen[layer] & (@as(u32, 1) << @intCast(@backingInt(required))) == 0) return error.GemmaTrainingTape;
            };
        }

        fn dense(ctx: *anyopaque, encoder: B.Pass, l: usize, x: R.View, y: R.View) !void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            try t.r.apply(encoder, l, x, y, t.seq.scale, t.from[l]);
            const ops_: @import("learning_ops.zig").Ops(R) = .{ .r = t.r, .pass = encoder, .dummy = t.buf(.dummy) };
            for (t.seq.nudge.layers, t.seq.nudge.rows, 0..) |mask, row, key| {
                if (mask & (@as(u64, 1) << @intCast(l)) == 0 or row < t.first or row - t.first >= y.rows) continue;
                var p: core.living_rows.Params = .{};
                p.u[0] = 1;
                p.u[1] = @intCast(t.shape.hidden);
                const v: Ref = .{ .buffer = y.buffer, .offset = y.offset + (row - t.first) * t.shape.hidden * 2, .len = t.shape.hidden * 2 };
                try ops_.row(.nudge, p, &.{ .{ 0, v }, .{ 1, try t.buf(.delta).plus(t.vector(key, l) * 4, t.shape.hidden * 4) }, .{ 8, v } });
            }
        }
    };
}
