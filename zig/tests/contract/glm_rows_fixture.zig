//! Borrowed GLM inputs and logical output maps, independent of the Metal row-loop implementation.
const std = @import("std");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const cases = @import("glm_rows_cases.zig");
const reference_threads = contract.reference_threads;
fn bf(x: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(x)));
}
fn minimum(x: f64, y: f64) f64 {
    return if (std.math.isNan(x)) y else @min(x, y);
}
fn maximum(x: f64, y: f64) f64 {
    return if (std.math.isNan(x)) y else @max(x, y);
}
fn glu(gate: f64, value: f64) f64 {
    const x = bf(minimum(bf(gate), 10));
    const up = bf(minimum(maximum(bf(value), -10), 10));
    const e = bf(@exp(@abs(x)));
    const y = bf(1 / bf(1 + e));
    return bf(bf(x * (if (x < 0) y else bf(1 - y))) * up);
}
pub const Fixture = struct {
    a: std.mem.Allocator,
    v: cases.Entry,
    shape: c.Shape,
    f: Affine,
    up: ?Affine,
    heads: usize,
    ldx: usize,
    ldo: usize,
    hso: usize,
    order: []i32,
    offsets: [4]i32,
    pub fn init(a: std.mem.Allocator, v: cases.Entry, s: c.Shape, fill: g.Fill, seed: u64) !Fixture {
        const rows = v.kind == .narrow or v.kind == .wide;
        const group: usize = if (s.rows % 2 != 0) 32 else 64;
        const heads: usize = if (rows and s.rows % 2 != 0 and s.n % 4 == 0) 2 else 1;
        const weight_rows = if (rows) heads * std.mem.alignForward(usize, s.n, 4) else 4 * s.n;
        const fs: c.Shape = .{ .rows = s.rows * heads, .n = weight_rows, .k = s.k };
        const f = try Affine.init(a, fs, v.bits, group);
        errdefer f.deinit();
        try f.fill(fill, seed);
        if (v.kind == .glu and fill == .nonfinite) {
            @memset(f.x, c.toBf16(1));
            f.x[s.k / 3] = c.toBf16(std.math.nan(f32));
            const row = if (s.rows > 1) s.k else 0;
            f.x[row + s.k / 2] = c.toBf16(std.math.inf(f32));
            f.x[row + s.k - 1] = c.toBf16(-std.math.inf(f32));
        }
        var up: ?Affine = null;
        errdefer if (up) |u| u.deinit();
        if (v.kind == .glu) {
            const at = std.mem.indexOfScalar(u8, &cases.bits, v.bits).?;
            up = try Affine.init(a, fs, cases.bits[(at + 1) % cases.bits.len], if (group == 32) 64 else 32);
            try up.?.fill(fill, seed ^ 0x1130);
            if (fill == .stress and (v.mode == .up_oracle or (v.mode != .oracle and up.?.bits > f.bits))) @memcpy(f.x, up.?.x);
            @memcpy(up.?.x, f.x);
            if (fill == .stress and v.nonlinear()) {
                // Keep final SwiGLU's sigmoid normal; the unscaled gate/up observers prove the large FP32 sums.
                for ([_]Affine{ f, up.? }) |q| {
                    const scale = std.math.ldexp(@as(f32, 1), 2 - @as(i32, q.bits));
                    @memset(q.scales, c.toBf16(scale));
                    @memset(q.biases, c.toBf16(-2));
                }
            }
        }
        const order = try a.alloc(i32, s.rows);
        for (order, 0..) |*o, i| o.* = @intCast(s.rows - 1 - i);
        const first: i32 = @intCast(@min(s.rows, 3));
        const hso = s.n + if (heads > 1) @as(usize, 4) else 0;
        return .{ .a = a, .v = v, .shape = s, .f = f, .up = up, .heads = heads, .ldx = heads * s.k + if (heads > 1) @as(usize, 16) else 0, .ldo = if (rows) heads * hso + if (heads > 1) @as(usize, 8) else 0 else s.n + 8, .hso = hso, .order = order, .offsets = .{ 0, 0, first, first } };
    }
    pub fn takes(v: cases.Entry, s: c.Shape) bool {
        const group: usize = if (s.rows % 2 != 0) 32 else 64;
        return std.mem.indexOfScalar(u8, &cases.bits, v.bits) != null and s.rows > 0 and s.rows <= (if (v.kind == .narrow or v.kind == .wide) @as(usize, 17) else 129) and s.n > 0 and s.k > 0 and s.k % group == 0 and
            s.n % (if (v.kind == .narrow or v.kind == .wide) @as(usize, 2) else 4) == 0 and
            (v.kind != .glu or s.k % 64 == 0);
    }
    pub fn deinit(f: Fixture) void {
        f.f.deinit();
        if (f.up) |u| u.deinit();
        f.a.free(f.order);
    }
    pub fn count(f: Fixture) usize {
        return f.shape.rows * f.heads * f.shape.n;
    }
    pub fn outputIndex(f: Fixture, i: usize) usize {
        return (i / (f.heads * f.shape.n)) * f.ldo + (i / f.shape.n % f.heads) * f.hso + i % f.shape.n;
    }
    pub fn inputIndex(f: Fixture, row: usize) usize {
        return row / f.heads * f.ldx + row % f.heads * f.shape.k;
    }
    pub fn outputElements(f: Fixture) usize {
        return f.shape.rows * f.ldo;
    }
    pub fn inputElements(f: Fixture) usize {
        return f.shape.rows * f.ldx;
    }
    pub fn packX(f: Fixture, out: []u16) void {
        @memset(out, 0x7fc1);
        for (0..f.f.shape.rows) |row| @memcpy(out[f.inputIndex(row)..][0..f.shape.k], f.f.x[row * f.shape.k ..][0..f.shape.k]);
    }
    pub fn coordinates(f: Fixture, i: usize) struct { row: usize, col: usize } {
        const n = f.shape.n;
        if (f.v.kind == .narrow or f.v.kind == .wide) return .{ .row = i / n, .col = (i / n % f.heads) * std.mem.alignForward(usize, n, 4) + i % n };
        const pos = if (f.v.kind == .down) f.shape.rows - 1 - i / n else i / n;
        const expert: usize = if (pos < @as(usize, @intCast(f.offsets[2]))) 1 else 3;
        return .{ .row = if (f.v.kind == .glu) @as(usize, @intCast(f.order[pos])) / 2 else pos, .col = expert * n + i % n };
    }
    pub fn reference(f: Fixture, want: []f64, mag: []f64) void {
        f.referenceParallel(want, mag, reference_threads.maximum, .normal);
    }
    fn referenceParallel(f: Fixture, want: []f64, mag: []f64, requested: usize, mode: reference_threads.Mode) void {
        const jobs_count = reference_threads.count(requested, want.len);
        if (jobs_count == 0) return;
        var jobs: [reference_threads.maximum]ReferenceJob = undefined;
        for (jobs[0..jobs_count], 0..) |*job, worker| job.* = .{
            .f = f,
            .first = worker * want.len / jobs_count,
            .end = (worker + 1) * want.len / jobs_count,
            .want = want,
            .mag = mag,
        };
        reference_threads.run(ReferenceJob, jobs[0..jobs_count], mode);
    }
    pub fn referenceSerial(f: Fixture, want: []f64, mag: []f64) void {
        for (want, mag, 0..) |*w, *m, i| {
            const at = f.coordinates(i);
            const q = if (f.v.mode == .up_oracle) f.up.? else f.f;
            const dot = q.one(at.row, at.col);
            if (f.v.nonlinear()) {
                w.* = glu(dot[0], f.up.?.one(at.row, at.col)[0]);
                m.* = @abs(w.*);
            } else {
                w.* = dot[0];
                m.* = dot[1];
            }
        }
    }
    pub fn controlDiffers(f: Fixture) bool {
        for (0..f.count()) |i| {
            const at = f.coordinates(i);
            const wrong_row = if (f.v.kind == .narrow or f.v.kind == .wide) at.row % f.heads else at.row;
            const wrong_col = if (f.v.kind == .narrow or f.v.kind == .wide) at.col else at.col % f.shape.n;
            const got = f.f.one(wrong_row, wrong_col)[0];
            const want = f.f.one(at.row, at.col)[0];
            if (f.v.kind == .glu) {
                if (glu(got, f.up.?.one(wrong_row, wrong_col)[0]) != glu(want, f.up.?.one(at.row, at.col)[0])) return true;
            } else if (bf(got) != bf(want)) return true;
        }
        return false;
    }
    pub fn simulate(f: Fixture, acc: c.Accum, order: g.Order, out: []f64) !void {
        const k = f.shape.k;
        const terms = try f.a.alloc(f64, 2 * k);
        defer f.a.free(terms);
        const groups = try f.a.alloc(f64, k / 32);
        defer f.a.free(groups);
        for (out, 0..) |*o, i| {
            const at = f.coordinates(i);
            const q = if (f.v.mode == .up_oracle) f.up.? else f.f;
            const dot = linear(q, at.row, at.col, acc, order, terms, groups);
            o.* = if (f.v.nonlinear()) glu(dot, linear(f.up.?, at.row, at.col, acc, order, terms, groups)) else dot;
        }
    }
};
const ReferenceJob = struct {
    f: Fixture,
    first: usize,
    end: usize,
    want: []f64,
    mag: []f64,
    pub fn run(job: *@This()) void {
        const f = job.f;
        for (job.first..job.end) |i| {
            const at = f.coordinates(i);
            const q = if (f.v.mode == .up_oracle) f.up.? else f.f;
            const dot = q.one(at.row, at.col);
            if (f.v.nonlinear()) {
                job.want[i] = glu(dot[0], f.up.?.one(at.row, at.col)[0]);
                job.mag[i] = @abs(job.want[i]);
            } else {
                job.want[i] = dot[0];
                job.mag[i] = dot[1];
            }
        }
    }
};

fn linear(f: Affine, row: usize, col: usize, acc: c.Accum, order: g.Order, t: []f64, groups: []f64) f64 {
    const k = f.shape.k;
    const kg = k / f.group;
    const s: f64 = c.fromBf16(f.scales[col * kg]);
    const b: f64 = c.fromBf16(f.biases[col * kg]);
    for (0..k) |i| {
        t[i] = @as(f64, @floatFromInt(f.codes[col * k + i])) * c.fromBf16(f.x[row * k + i]);
        t[k + i] = c.fromBf16(f.x[row * k + i]);
    }
    const sum = contract.rows.sumIn;
    const whole = sum(acc, order, t[0..k]) * s + sum(acc, order, t[k .. 2 * k]) * b;
    for (0..kg) |gi| groups[gi] = sum(acc, .sequential, t[gi * f.group ..][0..f.group]) * s + sum(acc, .sequential, t[k + gi * f.group ..][0..f.group]) * b;
    const grouped = sum(acc, order, groups[0..kg]);
    for (0..k) |i| t[i] = (s * @as(f64, @floatFromInt(f.codes[col * k + i])) + b) * c.fromBf16(f.x[row * k + i]);
    const direct = sum(acc, order, t[0..k]);
    const want = f.one(row, col)[0];
    var best = whole;
    for ([_]f64{ grouped, direct }) |v| if (!std.math.isFinite(best) or @abs(v - want) < @abs(best - want)) {
        best = v;
    };
    return best;
}

fn checkParallel(f: Fixture) !void {
    const a = std.testing.allocator;
    const n = f.count();
    const values = try a.alloc(f64, 4 * n);
    defer a.free(values);
    const want = values[0..n];
    const mag = values[n .. 2 * n];
    const got = values[2 * n .. 3 * n];
    const got_mag = values[3 * n ..];
    f.referenceSerial(want, mag);
    for ([_]usize{ 1, 3, 8 }) |workers| {
        for ([_]reference_threads.Mode{ .normal, .partial_failure, .all_failure }) |mode| {
            @memset(got, -517);
            @memset(got_mag, -519);
            f.referenceParallel(got, got_mag, workers, mode);
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
            try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mag), std.mem.sliceAsBytes(got_mag));
        }
    }
    var failing = std.testing.FailingAllocator.init(a, .{ .fail_index = 0 });
    var borrowed = f;
    borrowed.a = failing.allocator();
    borrowed.f.a = failing.allocator();
    if (borrowed.up) |*up| up.a = failing.allocator();
    borrowed.reference(got, got_mag);
    try std.testing.expect(!failing.has_induced_failure);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(want), std.mem.sliceAsBytes(got));
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(mag), std.mem.sliceAsBytes(got_mag));
}

pub fn checkParallelReference() !void {
    const a = std.testing.allocator;
    for (cases.entries) |v| for ([_]usize{ 3, 4 }) |count| {
        for ([_]g.Fill{ .random, .stress, .nonfinite }) |fill| {
            var f = try Fixture.init(a, v, .{ .rows = count, .n = 8, .k = 128 }, fill, 517 + @as(u64, v.bits));
            defer f.deinit();
            try checkParallel(f);
            f.f.x[0] = c.toBf16(7);
            f.f.codes[0] ^= 1;
            f.f.scales[0] = c.toBf16(-0.5);
            if (f.up) |up| {
                up.x[0] = c.toBf16(7);
                up.codes[0] ^= 1;
            }
            f.order[0] = f.order[f.order.len - 1];
            f.offsets[2] = 1;
            f.offsets[3] = 1;
            try checkParallel(f);
        }
    };
}
