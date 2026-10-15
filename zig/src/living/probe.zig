//! A measure, not a lesson: how far vectors fitted at a few layers bring a fact back (Nemotron's Metal trainer only).
const std = @import("std");
const examples = @import("lesson_examples.zig");
const lesson_route = @import("lesson_route.zig");
const subspace = @import("subspace.zig");

const Example = examples.Example;
const Acting = lesson_route.Acting;

/// The row a setup's vectors go on: the reply's first (the assistant header) or the question's subject.
pub const Row = enum { decide, subject };

const steps = 30; // Adam steps a setup is fitted for
const reach: f32 = 0.05; // each step's size against the residual's
const decay: f32 = 0.05; // the vectors' squared size against the residual's, added to the loss
const furthest: f32 = 3; // a vector's size at most, against the residual's

/// A probe over trainer T (Metal's: its residual and nudges).
pub fn Probe(comptime T: type) type {
    return struct {
        const Self = @This();
        const Setup = struct { row: Row, layers: u64 };

        gpa: std.mem.Allocator,
        answered: []const Example = &.{},
        held: []const Example = &.{},
        setups: std.ArrayList(Setup) = .empty,
        setup: usize = 0,
        round: usize = 0,
        at: usize = 0,
        judging: bool = false,
        loss: f32 = 0,
        first: f32 = 0,
        final: f32 = 0, // the answers' loss at the last fitting round
        back: usize = 0, // held-out answers that came back token for token
        size: std.ArrayList(f32) = .empty, // [layers]: the row's mean residual length after each layer
        state: std.ArrayList(f32) = .empty, // [4, layers, D]: the vectors, their Adam moments, a round's gradient

        pub fn init(gpa: std.mem.Allocator) Self {
            return .{ .gpa = gpa };
        }

        pub fn deinit(p: *Self) void {
            p.setups.deinit(p.gpa);
            p.size.deinit(p.gpa);
            p.state.deinit(p.gpa);
        }

        /// A lesson's setups: the subject's row at MoE layers 13 to 25, then in windows of three MoE layers, two apart.
        pub fn begin(p: *Self, t: *T, answered: []const Example, held: []const Example) !void {
            const c = t.config();
            p.answered = answered;
            p.held = held;
            p.setups.clearRetainingCapacity();
            var moe: [64]usize = undefined;
            var count_moe: usize = 0;
            var mid: u64 = 0;
            for (0..c.layers - 1) |l| if (c.kinds[l] == .moe) {
                moe[count_moe] = l;
                count_moe += 1;
                if (l >= 13 and l <= 25) mid |= @as(u64, 1) << @intCast(l);
            };
            try p.setups.append(p.gpa, .{ .row = .subject, .layers = mid });
            var w: usize = 0;
            while (w + 3 <= count_moe) : (w += 2) {
                var mask: u64 = 0;
                for (moe[w..][0..3]) |l| mask |= @as(u64, 1) << @intCast(l);
                try p.setups.append(p.gpa, .{ .row = .subject, .layers = mask });
            }
            p.setup = 0;
            try p.start(t);
        }

        fn start(p: *Self, t: *T) !void {
            const c = t.config();
            p.round = 0;
            p.at = 0;
            p.loss = 0;
            p.judging = false;
            p.back = 0;
            p.size.clearRetainingCapacity();
            try p.size.appendNTimes(p.gpa, 0, c.layers);
            p.state.clearRetainingCapacity();
            try p.state.appendNTimes(p.gpa, 0, 4 * c.layers * c.hidden);
        }

        fn row(s: Setup, ex: Example) ?usize {
            return switch (s.row) {
                .decide => ex.decide,
                .subject => if (ex.subject > 0 and ex.subject < ex.start) ex.subject else null,
            };
        }

        /// One pass; true once every setup has been fitted and judged.
        pub fn step(p: *Self, t: *T, acting: Acting) !bool {
            if (p.setup == p.setups.items.len) return true;
            const c = t.config();
            const d = c.hidden;
            const n = c.layers * d;
            const s = p.setups.items[p.setup];
            const vecs = p.state.items[0..n];
            const grad = p.state.items[3 * n ..][0..n];
            const list = if (p.judging) p.held else p.answered;
            while (p.at < list.len) {
                const ex = list[p.at];
                p.at += 1;
                const r = row(s, ex) orelse continue;
                t.nudge = .{ .layers = .{ s.layers, 0, 0 }, .rows = .{ r, 0, 0 } };
                @memcpy(t.delta.slice(f32, n), vecs);
                const got = try t.step(ex.ids, ex.start, .target, .{ .decide = ex.decide, .bits = acting, .whole = true });
                p.loss += got.loss;
                if (p.judging) {
                    p.back += @intFromBool(got.recalled);
                    return false;
                }
                for (0..c.layers) |l| if (s.layers >> @intCast(l) & 1 != 0) {
                    for (grad[l * d ..][0..d], t.push.slice(f32, n)[l * d ..][0..d]) |*g, x| g.* += x;
                    if (p.round == 0) p.size.items[l] += norm(t.residual(l + 1, r));
                };
                return false;
            }
            const used = count(s, list);
            const mean = p.loss / @as(f32, @floatFromInt(@max(used, 1)));
            p.loss = 0;
            p.at = 0;
            if (p.judging) {
                var len: f32 = 0;
                var layers: usize = 0;
                for (0..c.layers) |l| if (s.layers >> @intCast(l) & 1 != 0) {
                    len += @sqrt(subspace.dot(vecs[l * d ..][0..d], vecs[l * d ..][0..d])) / p.size.items[l];
                    layers += 1;
                };
                const last = 63 - @clz(s.layers);
                std.log.info("slide probe: the {t}'s row, {d} MoE layers from {d} to {d}: the answers' loss {d:.3} to {d:.3} in {d} steps, vectors {d:.2} of the residual each; held-out {d} of {d} come back, their loss {d:.3}", .{ s.row, layers, @ctz(s.layers), last, p.first, p.final, steps, len / @as(f32, @floatFromInt(@max(layers, 1))), p.back, used, mean });
                p.setup += 1;
                if (p.setup < p.setups.items.len) try p.start(t);
                return false;
            }
            if (p.round == 0) {
                p.first = mean;
                for (p.size.items) |*x| x.* /= @floatFromInt(@max(used, 1));
            }
            p.round += 1;
            p.final = mean;
            if (p.round > steps) {
                p.judging = true;
                return false;
            }
            const m = p.state.items[n..][0..n];
            const v = p.state.items[2 * n ..][0..n];
            const tt: f32 = @floatFromInt(p.round);
            for (0..c.layers) |l| if (s.layers >> @intCast(l) & 1 != 0) {
                const size = p.size.items[l];
                const lr = reach * size / @sqrt(@as(f32, @floatFromInt(d)));
                const x = vecs[l * d ..][0..d];
                for (x, m[l * d ..][0..d], v[l * d ..][0..d], grad[l * d ..][0..d]) |*xi, *mi, *vi, gi| {
                    const g = gi / @as(f32, @floatFromInt(@max(used, 1))) + 2 * decay * xi.* / (size * size);
                    mi.* = 0.9 * mi.* + 0.1 * g;
                    vi.* = 0.999 * vi.* + 0.001 * g * g;
                    xi.* -= lr * (mi.* / (1 - std.math.pow(f32, 0.9, tt))) / (@sqrt(vi.* / (1 - std.math.pow(f32, 0.999, tt))) + 1e-8);
                }
                const len = @sqrt(subspace.dot(x, x));
                if (len > furthest * size) for (x) |*xi| {
                    xi.* *= furthest * size / len;
                };
            };
            @memset(grad, 0);
            return false;
        }

        fn count(s: Setup, list: []const Example) usize {
            var k: usize = 0;
            for (list) |ex| k += @intFromBool(row(s, ex) != null);
            return k;
        }
    };
}

fn norm(b: []const u16) f32 {
    var s: f32 = 0;
    for (b) |x| {
        const f: f32 = @bitCast(@as(u32, x) << 16);
        s += f * f;
    }
    return @sqrt(s);
}
