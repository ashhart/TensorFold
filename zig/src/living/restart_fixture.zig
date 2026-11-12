//! Two numeric facts through the real learner, with persistence supplied by the shard-test root.
const std = @import("std");
const living = @import("living.zig");
const tm = living.train_mode;
const Example = living.learner.Example;
const d = 64;
const n = 64;
const width = d * n;

pub fn Fixture(comptime Store: type) type {
    return struct {
        const Self = @This();
        pub const family = living.families.nemotron_h;
        pub const max_rows = 64;
        pub const Ctx = *Self;
        pub const Learner = living.learner.Learner(Self);
        dir: []const u8,
        up: [width]f32 = @splat(0),
        down: [width]f32 = @splat(0),
        used: usize = 0,

        pub const Trainer = struct {
            host: *Self,
            pack: usize = 64,
            spans: [8]tm.Span = undefined,
            from: [2]u32 = @splat(std.math.maxInt(u32)),
            nudge: tm.Nudge = .{},
            count: living.pocket_clock.Count = .{},
            prefixes: struct {
                pub const most = 0;
            } = .{},
            rows: [64][d]u16 = undefined,
            pushes: [8][d]f32 = undefined,
            scratch: [d]f32 = @splat(0),

            pub fn packs(_: *const Trainer) bool {
                return false;
            }
            pub fn prefixSlots(_: *Trainer, _: usize) !void {}
            pub fn normed(t: *Trainer, _: usize, row: usize) []const u16 {
                return &t.rows[row];
            }
            pub fn residual(t: *Trainer, _: usize, row: usize) []const u16 {
                return &t.rows[row];
            }
            pub fn pushAt(t: *Trainer, i: usize, _: usize, _: usize) []const f32 {
                return &t.pushes[i];
            }
            pub fn deltaAt(t: *Trainer, _: usize, _: usize, _: usize) []f32 {
                return &t.scratch;
            }

            pub fn pass(t: *Trainer, seqs: []const tm.Seq, mode: tm.Mode, _: living.lesson_route.Turn, out: []tm.Result) !void {
                var at: usize = 0;
                for (seqs, out, 0..) |q, *got, s| {
                    t.spans[s] = .{ .at = at, .rows = q.ids.len - 1 };
                    for (0..q.ids.len - 1) |r| {
                        var x: [d]f32 = @splat(1);
                        x[if (r == q.decide) q.ids[0] else 8 + r] = 8;
                        for (&t.rows[at + r], x) |*b, v| b.* = @truncate(@as(u32, @bitCast(v)) >> 16);
                    }
                    var y: [d]f32 = @splat(0);
                    const x = &t.rows[at + q.decide];
                    for (0..t.host.used) |k| {
                        const z = living.pocket_fit.dotBf16(t.host.up[k * d ..][0..d], x);
                        const scale: f32 = if (mode == .target and k >= t.from[0]) q.scale else 1;
                        for (&y, t.host.down[k * d ..][0..d]) |*v, w| v.* += scale * @max(z, 0) * @max(z, 0) * w;
                    }
                    var loss: f32 = 0;
                    var other: f32 = 0.4;
                    for (y, &t.pushes[s], 0..) |v, *g, o| {
                        const want: f32 = if (o == q.ids[0]) 1 else if (o == 0) -1 else 0;
                        g.* = v - want;
                        loss += g.* * g.* / 2;
                        if (o != q.ids[0]) other = @max(other, v);
                    }
                    got.* = .{ .loss = loss, .recalled = y[q.ids[0]] > other };
                    at += q.ids.len - 1;
                }
                t.count.passes += 1;
                t.count.rows += at;
            }
            pub fn step(t: *Trainer, ids: []const u32, start: usize, mode: tm.Mode, turn: living.lesson_route.Turn) !tm.Result {
                var out: [1]tm.Result = undefined;
                try t.pass(&.{.{ .ids = ids, .start = start, .decide = turn.decide }}, mode, turn, &out);
                return out[0];
            }
        };

        pub fn layout(_: Ctx) living.Layout {
            var has: living.Layers = .empty;
            has.set(0);
            return .fromExperts(d, 64, 2, has);
        }
        pub fn trainer(a: std.mem.Allocator, _: std.Io, h: Ctx) !*Trainer {
            const t = try a.create(Trainer);
            t.* = .{ .host = h };
            return t;
        }
        pub fn free(t: *Trainer, a: std.mem.Allocator) void {
            a.destroy(t);
        }
        pub fn seconds(_: std.Io) f64 {
            return 0;
        }
        pub fn pocket(h: Ctx, _: usize, up: []const f32, down: []const f32, used: usize) !void {
            @memcpy(&h.up, up);
            @memcpy(&h.down, down);
            h.used = used;
        }
        pub fn pocketRead(h: Ctx, _: usize, up: []f32, down: []f32) usize {
            @memcpy(up, &h.up);
            @memcpy(down, &h.down);
            return h.used;
        }
        pub fn pocketSave(h: Ctx, a: std.mem.Allocator, io: std.Io, layers: []const usize, neurons: usize, up: []f32, down: []f32) !usize {
            return Store.write(a, io, h.dir, family, layout(h), layers, neurons, &.{up}, down);
        }
        pub fn reopen(h: *Self) !void {
            try Store.read(std.testing.allocator, std.testing.io, h.dir, &h.up, &h.down);
            h.used = 0;
            for (0..n) |k| if (!std.mem.allEqual(f32, h.up[k * d ..][0..d], 0)) {
                h.used = k + 1;
            };
        }
    };
}

pub fn teach(l: anytype, fact: u32, keep: []const Example) !void {
    const ids = [_]u32{ fact, 2, 3, 4 };
    const held_ids = [_]u32{ fact, 7, 3, 4 };
    const train = [_]Example{.{ .ids = &ids, .start = 3, .decide = 2 }};
    const held = [_]Example{.{ .ids = &held_ids, .start = 3, .decide = 2 }};
    try l.begin(.{ .train = &train, .held = &held, .keep = keep });
    for (0..10000) |_| {
        const s = l.step();
        if (s.report) |r| switch (r) {
            .failed => |why| {
                std.debug.print("restart fixture failed: {s}\n", .{why});
                return error.LessonFailed;
            },
            .learned => |v| {
                if (!v.recalled) std.debug.print("fact {d}: loss {d:.6}, held {d:.6}, used {d}, outputs {any}\n", .{ fact, l.pockets.v.running, l.pockets.v.best_held, l.b.used, l.b.down[0..4] });
                try std.testing.expect(v.recalled);
            },
        };
        if (s.done) return;
    }
    return error.NoVerdict;
}

pub fn save(l: anytype) !void {
    try l.begin(.{ .save = true });
    const s = l.step();
    try std.testing.expect(s.done and s.report == null and !s.stale);
}
