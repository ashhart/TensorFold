//! Nemotron's 1,024-thread training scan: forward checkpoints, backward gradients and grouped partials.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

const Shape = extern struct { rows: u32, heads: u32, dh: u32, groups: u32, n: u32, inner: u32, conv: u32, proj: u32 };
const guard = 16;
const sentinel: u32 = 0x7fc00000;
const a = std.testing.allocator;

fn bf(v: f32) u16 {
    const u: u32 = @bitCast(v);
    return @intCast((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn value(v: u16) f64 {
    return @as(f32, @bitCast(@as(u32, v) << 16));
}

fn dyadic(i: usize, scale: f32) f32 {
    return @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / scale;
}

const Fixture = struct {
    q: mtl.Queue,
    b: [12]mtl.Buffer,
    counts: [12]usize,
    s: Shape,
    seeded: bool,

    fn init(d: mtl.Device, q: mtl.Queue, rows: usize, heads: usize, groups: usize, seeded: bool) !Fixture {
        const inner = heads * 64;
        const conv = inner + groups * 256;
        const state = heads * 64 * 128;
        const counts = [_]usize{ rows * conv, rows * heads, heads, heads, rows * inner, ((rows + 7) / 8 + 1) * state, rows * conv, rows * heads * 128, rows * heads * 128, rows * heads, state, rows * inner };
        var f = Fixture{ .q = q, .b = undefined, .counts = counts, .seeded = seeded, .s = .{ .rows = @intCast(rows), .heads = @intCast(heads), .dh = 64, .groups = @intCast(groups), .n = 128, .inner = @intCast(inner), .conv = @intCast(conv), .proj = 0 } };
        var made: usize = 0;
        errdefer for (f.b[0..made]) |b| b.deinit();
        for (counts, 0..) |count, i| {
            const bytes = count * @as(usize, if (i == 0 or i == 3) 2 else 4) + guard * 4;
            f.b[i] = try d.buffer(bytes, mtl.ResourceOptions.shared);
            @memset(f.b[i].slice(u32, bytes / 4), sentinel);
            made += 1;
        }
        for (f.b[0].slice(u16, counts[0]), 0..) |*v, i| v.* = bf(dyadic(i * 7 + i / conv * 11, 32));
        for (f.b[1].slice(f32, counts[1]), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 5 + 1)) / 32;
        for (f.b[2].slice(f32, heads), 0..) |*v, h| v.* = -@as(f32, @floatFromInt(h % 4 + 1)) / 4;
        for (f.b[3].slice(u16, heads), 0..) |*v, h| v.* = bf(dyadic(h * 3 + 10, 16));
        for (f.b[4].slice(f32, counts[4]), 0..) |*v, i| v.* = dyadic(i * 11 + i / inner * 3, 32);
        for (f.b[10].slice(f32, state), 0..) |*v, i| v.* = dyadic(i * 5 + i / 128 * 7, 256);
        return f;
    }

    fn deinit(f: Fixture) void {
        for (f.b) |b| b.deinit();
    }

    fn act(f: Fixture, r: usize, c: usize) f64 {
        return value(f.b[0].slice(u16, f.counts[0])[r * f.s.conv + c]);
    }

    fn floats(f: Fixture, i: usize) []f32 {
        return f.b[i].slice(f32, f.counts[i]);
    }

    fn tails(f: Fixture) !void {
        for (f.b, f.counts, 0..) |b, count, i| {
            const bytes = count * @as(usize, if (i == 0 or i == 3) 2 else 4);
            for (b.slice(u32, b.length() / 4)[bytes / 4 ..]) |v| try std.testing.expectEqual(sentinel, v);
        }
    }

    fn forward(f: Fixture, p: mtl.Pipeline) !void {
        const cb = f.q.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(p);
        for ([_]usize{ 0, 1, 2, 3, 11, 5 }, 0..) |idx, slot| e.setBuffer(f.b[idx], 0, slot);
        e.setValue(f.s, 6);
        e.setBuffer(f.b[10], 0, 7);
        e.setValue(@as(u32, @intFromBool(f.seeded)), 8);
        e.dispatchThreads(mtl.Size.of(1024 * f.s.heads, 1, 1), mtl.Size.of(1024, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
    }

    fn backward(f: Fixture, back: mtl.Pipeline, bc: mtl.Pipeline, ref: Reference) !void {
        const cb = f.q.commandBuffer();
        const e = cb.compute(.serial);
        e.setPipeline(back);
        for (f.b[0..10], 0..) |b, i| e.setBuffer(b, 0, i);
        e.setValue(f.s, 10);
        e.dispatchThreads(mtl.Size.of(1024 * f.s.heads, 1, 1), mtl.Size.of(1024, 1, 1));
        e.end();
        cb.commit();
        cb.wait();
        try std.testing.expect(cb.failure() == null);
        var worst: [4]f64 = @splat(0);
        for (0..f.s.rows) |r| {
            worst[0] = @max(worst[0], try compare(f.floats(6)[r * f.s.conv ..][0..f.s.inner], ref.dact[r * f.s.conv ..][0..f.s.inner]));
            for (f.b[6].slice(u32, f.counts[6])[r * f.s.conv + f.s.inner ..][0 .. f.s.conv - f.s.inner]) |v| {
                try std.testing.expectEqual(sentinel, v);
            }
        }
        worst[1] = try compare(f.floats(7), ref.db);
        worst[2] = try compare(f.floats(8), ref.dc);
        worst[3] = try compare(f.floats(9), ref.dt);
        try f.tails();
        const next = f.q.commandBuffer();
        const enc = next.compute(.serial);
        enc.setPipeline(bc);
        for ([_]usize{ 7, 8, 6 }, 0..) |idx, slot| enc.setBuffer(f.b[idx], 0, slot);
        enc.setValue(f.s, 3);
        enc.dispatchThreads(mtl.Size.of(128, f.s.groups, f.s.rows), mtl.Size.of(128, 1, 1));
        enc.end();
        next.commit();
        next.wait();
        try std.testing.expect(next.failure() == null);
        _ = try compare(f.floats(6), ref.dact);
        try f.tails();
        std.debug.print("SSM seeded={any} heads={d} groups={d} rows={d}: max abs dx/dB/dC/ddt {any}\n", .{ f.seeded, f.s.heads, f.s.groups, f.s.rows, worst });
    }
};

fn compare(actual: []const f32, expected: []const f64) !f64 {
    try std.testing.expectEqual(expected.len, actual.len);
    var worst: f64 = 0;
    for (actual, expected) |got, want| {
        try std.testing.expect(std.math.isFinite(got));
        const err = @abs(@as(f64, got) - want);
        worst = @max(worst, err);
        try std.testing.expect(err <= 2e-6 + 3e-5 * @abs(want));
    }
    return worst;
}

const Reference = struct {
    states: []f64,
    y: []f64,
    dact: []f64,
    db: []f64,
    dc: []f64,
    dt: []f64,

    fn init(f: Fixture) !Reference {
        const size = f.counts[10];
        var ref: Reference = undefined;
        const fields = .{ "states", "y", "dact", "db", "dc", "dt" };
        const counts = [_]usize{ (f.s.rows + 1) * size, f.counts[11], f.counts[6], f.counts[7], f.counts[8], f.counts[9] };
        var made: usize = 0;
        errdefer inline for (fields, 0..) |field, i| {
            if (i < made) a.free(@field(ref, field));
        };
        inline for (fields, counts) |field, count| {
            @field(ref, field) = try a.alloc(f64, count);
            @memset(@field(ref, field), 0);
            made += 1;
        }
        for (ref.states[0..size], f.floats(10)) |*v, initial| v.* = if (f.seeded) initial else 0;
        for (0..f.s.rows) |r| for (0..f.s.heads) |h| {
            const g = h / (f.s.heads / f.s.groups);
            const d: f64 = f.floats(1)[r * f.s.heads + h];
            const decay = @exp(d * @as(f64, f.floats(2)[h]));
            for (0..64) |p| {
                const x = f.act(r, h * 64 + p);
                var y = value(f.b[3].slice(u16, f.s.heads)[h]) * x;
                for (0..128) |n| {
                    const idx = (h * 64 + p) * 128 + n;
                    const cur = decay * ref.states[r * size + idx] + d * x * f.act(r, f.s.inner + g * 128 + n);
                    ref.states[(r + 1) * size + idx] = cur;
                    y += cur * f.act(r, f.s.inner + f.s.groups * 128 + g * 128 + n);
                }
                ref.y[r * f.s.inner + h * 64 + p] = y;
            }
        };
        const grad = try a.alloc(f64, size);
        defer a.free(grad);
        @memset(grad, 0);
        var r: usize = f.s.rows;
        while (r > 0) {
            r -= 1;
            for (0..f.s.heads) |h| {
                const g = h / (f.s.heads / f.s.groups);
                const d: f64 = f.floats(1)[r * f.s.heads + h];
                const an: f64 = f.floats(2)[h];
                const decay = @exp(d * an);
                for (0..64) |p| {
                    const x = f.act(r, h * 64 + p);
                    const dy: f64 = f.floats(4)[r * f.s.inner + h * 64 + p];
                    var dx = value(f.b[3].slice(u16, f.s.heads)[h]) * dy;
                    for (0..128) |n| {
                        const idx = (h * 64 + p) * 128 + n;
                        const at = (r * f.s.heads + h) * 128 + n;
                        const b = f.act(r, f.s.inner + g * 128 + n);
                        const c = f.act(r, f.s.inner + f.s.groups * 128 + g * 128 + n);
                        grad[idx] += dy * c;
                        ref.db[at] += d * x * grad[idx];
                        ref.dc[at] += dy * ref.states[(r + 1) * size + idx];
                        dx += d * grad[idx] * b;
                        ref.dt[r * f.s.heads + h] += grad[idx] * (x * b + an * decay * ref.states[r * size + idx]);
                        grad[idx] *= decay;
                    }
                    ref.dact[r * f.s.conv + h * 64 + p] = dx;
                }
                for (0..128) |n| {
                    ref.dact[r * f.s.conv + f.s.inner + g * 128 + n] += ref.db[(r * f.s.heads + h) * 128 + n];
                    ref.dact[r * f.s.conv + f.s.inner + f.s.groups * 128 + g * 128 + n] += ref.dc[(r * f.s.heads + h) * 128 + n];
                }
            }
        }
        return ref;
    }

    fn deinit(ref: Reference) void {
        inline for (.{ "states", "y", "dact", "db", "dc", "dt" }) |field| a.free(@field(ref, field));
    }

    fn checkpoints(ref: Reference, f: Fixture) !void {
        _ = try compare(f.floats(11), ref.y);
        const count = (f.s.rows + 7) / 8 + 1;
        for (0..f.s.heads) |h| for (0..count) |c| {
            const actual = f.floats(5)[(h * count + c) * 8192 ..][0..8192];
            if (c * 8 <= f.s.rows) {
                _ = try compare(actual, ref.states[c * 8 * f.counts[10] + h * 8192 ..][0..8192]);
            } else {
                for (actual) |v| try std.testing.expectEqual(sentinel, @as(u32, @bitCast(v)));
            }
        };
        try f.tails();
    }
};

const Perturb = struct { row: usize, column: usize, dt: bool, head: usize, shared: bool };

// Differentiate the FP64 scalar loss continuously at the dequantized inputs, without the GPU's lane reductions.
fn loss(f: Fixture, at: Perturb, delta: f64) f64 {
    const per = f.s.heads / f.s.groups;
    const first = if (at.shared) at.head / per * per else at.head;
    const last = first + @as(usize, if (at.shared) per else 1);
    var total: f64 = 0;
    for (first..last) |h| {
        var st: [8192]f64 = undefined;
        for (&st, f.floats(10)[h * 8192 ..][0..8192]) |*v, initial| v.* = if (f.seeded) initial else 0;
        const g = h / per;
        for (0..f.s.rows) |r| {
            const d = @as(f64, f.floats(1)[r * f.s.heads + h]) + if (at.dt and at.row == r and at.head == h) delta else 0;
            const decay = @exp(d * @as(f64, f.floats(2)[h]));
            for (0..64) |p| {
                const xc = h * 64 + p;
                const x = f.act(r, xc) + if (!at.dt and at.row == r and at.column == xc) delta else 0;
                var y = value(f.b[3].slice(u16, f.s.heads)[h]) * x;
                for (0..128) |n| {
                    const bc = f.s.inner + g * 128 + n;
                    const cc = f.s.inner + f.s.groups * 128 + g * 128 + n;
                    const b = f.act(r, bc) + if (!at.dt and at.row == r and at.column == bc) delta else 0;
                    const c = f.act(r, cc) + if (!at.dt and at.row == r and at.column == cc) delta else 0;
                    const idx = p * 128 + n;
                    st[idx] = decay * st[idx] + d * x * b;
                    y += st[idx] * c;
                }
                total += y * @as(f64, f.floats(4)[r * f.s.inner + xc]);
            }
        }
    }
    return total;
}

fn finiteDifferences(f: Fixture, ref: Reference) !void {
    const h = f.s.heads - 1;
    const g = f.s.groups - 1;
    for ([_]usize{ 0, @min(8, f.s.rows - 1), f.s.rows - 1 }) |r| {
        const cases = [_]Perturb{
            .{ .row = r, .column = h * 64 + 63, .dt = false, .head = h, .shared = false },
            .{ .row = r, .column = f.s.inner + g * 128 + 127, .dt = false, .head = h, .shared = true },
            .{ .row = r, .column = f.s.inner + f.s.groups * 128 + g * 128 + 11, .dt = false, .head = h, .shared = true },
            .{ .row = r, .column = 0, .dt = true, .head = h, .shared = false },
        };
        for (cases) |at| {
            const numerical = (loss(f, at, 1e-5) - loss(f, at, -1e-5)) / 2e-5;
            const expected = if (at.dt) ref.dt[r * f.s.heads + h] else ref.dact[r * f.s.conv + at.column];
            try std.testing.expect(@abs(numerical - expected) <= 1e-8 + 2e-6 * @abs(expected));
        }
    }
}

fn check(seeded: bool) !void {
    const d = mtl.Device.init() catch return error.SkipZigTest;
    defer d.deinit();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const q = try d.queue();
    defer q.deinit();
    const lib = try mtl.Library.fromSource(d, sources.train_mixers, mtl.CompileOptions.mlx());
    defer lib.deinit();
    const fwd = try mtl.Pipeline.init(d, lib, "tf_train_ssm_fwd", false);
    defer fwd.deinit();
    const back = try mtl.Pipeline.init(d, lib, "tf_train_ssm_back", false);
    defer back.deinit();
    const bc = try mtl.Pipeline.init(d, lib, "tf_train_ssm_bc", false);
    defer bc.deinit();
    std.debug.print("SSM pipeline capacity: forward={d}, backward={d}\n", .{ fwd.maxThreads(), back.maxThreads() });
    try std.testing.expect(fwd.maxThreads() >= 1024);
    try std.testing.expect(back.maxThreads() >= 1024);
    for ([_][2]usize{ .{ 2, 1 }, .{ 4, 2 }, .{ 64, 8 } }) |shape| {
        const rows: []const usize = if (shape[0] == 64) &.{ 1, 9, 17 } else &.{ 1, 7, 8, 9, 15, 16, 17, 25 };
        for (rows) |r| {
            const f = try Fixture.init(d, q, r, shape[0], shape[1], seeded);
            defer f.deinit();
            const ref = try Reference.init(f);
            defer ref.deinit();
            if ((shape[0] == 4 and (r == 9 or r == 25)) or (shape[0] == 64 and r == 17)) try finiteDifferences(f, ref);
            try f.forward(fwd);
            try ref.checkpoints(f);
            try f.backward(back, bc, ref);
        }
    }
}

test "Nemotron SSM backward: 1024 threads and FP64 gradients from zero state" {
    try check(false);
}

test "Nemotron SSM backward: seeded checkpoints, groups and finite differences" {
    try check(true);
}
