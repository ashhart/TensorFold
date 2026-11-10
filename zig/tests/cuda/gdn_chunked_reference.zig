//! Independent scalar and blocked FP64 delta recurrence over the actual BF16 operands and FP32 initial state.
const std = @import("std");
pub const width = 128;
pub const block = 16;
pub fn bf16(bits: u16) f64 {
    return @as(f32, @bitCast(@as(u32, bits) << 16));
}
pub fn narrow(value: f32) u16 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x7fffffff) > 0x7f800000) return @as(u16, @truncate(bits >> 16)) | 0x40;
    return @truncate((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}
pub const Input = struct {
    rows: usize,
    hk: usize,
    hv: usize,
    q: []const u16,
    k: []const u16,
    v: []const u16,
    decay: []const f32,
    beta: []const u16,
    state: []const f32,

    fn qk(self: Input, data: []const u16, row: usize, head: usize, key: usize) f64 {
        return bf16(data[(row * self.hk + head / (self.hv / self.hk)) * width + key]);
    }
    fn initial(self: Input, head: usize, value: usize, out: *[width]f64) void {
        for (out, self.state[(head * width + value) * width ..][0..width]) |*a, b| a.* = b;
    }
    pub fn finiteHead(self: Input, head: usize) bool {
        for (0..self.rows) |row| {
            if (!std.math.isFinite(self.decay[row * self.hv + head]) or !std.math.isFinite(bf16(self.beta[row * self.hv + head]))) return false;
            for (0..width) |i| if (!std.math.isFinite(self.qk(self.q, row, head, i)) or
                !std.math.isFinite(self.qk(self.k, row, head, i)) or !std.math.isFinite(bf16(self.v[(row * self.hv + head) * width + i]))) return false;
        }
        for (self.state[head * width * width ..][0 .. width * width]) |v| if (!std.math.isFinite(v)) return false;
        return true;
    }
};

pub fn serial(x: Input, head: usize, value: usize, out: []f64, state: *[width]f64) void {
    std.debug.assert(out.len == x.rows);
    x.initial(head, value, state);
    for (0..x.rows) |t| {
        const gate: f64 = x.decay[t * x.hv + head];
        var prediction: f64 = 0;
        for (state, 0..) |*s, key| {
            s.* *= gate;
            prediction += s.* * x.qk(x.k, t, head, key);
        }
        const delta = (bf16(x.v[(t * x.hv + head) * width + value]) - prediction) * bf16(x.beta[t * x.hv + head]);
        var y: f64 = 0;
        for (state, 0..) |*s, key| {
            s.* += delta * x.qk(x.k, t, head, key);
            y += s.* * x.qk(x.q, t, head, key);
        }
        out[t] = y;
    }
}

pub fn blocked(x: Input, head: usize, value: usize, out: []f64, state: *[width]f64) void {
    return blockEvaluate(false, x, head, value, out, state);
}
pub fn magnitude(x: Input, head: usize, value: usize, out: []f64, state: *[width]f64) void {
    return blockEvaluate(true, x, head, value, out, state);
}
fn operand(comptime positive: bool, value: f64) f64 {
    return if (positive) @abs(value) else value;
}
fn blockEvaluate(comptime positive: bool, x: Input, head: usize, value: usize, out: []f64, state: *[width]f64) void {
    if (!x.finiteHead(head)) return serial(x, head, value, out, state);
    x.initial(head, value, state);
    if (positive) for (state) |*s| {
        s.* = @abs(s.*);
    };
    var begin: usize = 0;
    while (begin < x.rows) : (begin += block) {
        const count = @min(block, x.rows - begin);
        var prefix: [block]f64 = @splat(1);
        var suffix: [block]f64 = @splat(1);
        var lower: [block][block]f64 = @splat(@splat(0));
        var coefficients: [block][block]f64 = @splat(@splat(0));
        var attention: [block][block]f64 = @splat(@splat(0));
        for (0..count) |i| {
            for (0..i + 1) |t| prefix[i] *= operand(positive, x.decay[(begin + t) * x.hv + head]);
            for (i + 1..count) |t| suffix[i] *= operand(positive, x.decay[(begin + t) * x.hv + head]);
            for (0..i + 1) |j| {
                var decay: f64 = 1;
                for (j + 1..i + 1) |t| decay *= operand(positive, x.decay[(begin + t) * x.hv + head]);
                var kk: f64 = 0;
                var qk: f64 = 0;
                for (0..width) |key| {
                    kk += operand(positive, x.qk(x.k, begin + i, head, key) * x.qk(x.k, begin + j, head, key));
                    qk += operand(positive, x.qk(x.q, begin + i, head, key) * x.qk(x.k, begin + j, head, key));
                }
                if (j < i) lower[i][j] = operand(positive, bf16(x.beta[(begin + i) * x.hv + head])) * decay * kk;
                attention[i][j] = decay * qk;
            }
        }
        for (0..count) |column| {
            var inverse: [block]f64 = @splat(0);
            for (column..count) |row| {
                inverse[row] = if (row == column) 1 else 0;
                for (column..row) |j| inverse[row] += (if (positive) @as(f64, 1) else -1) * lower[row][j] * inverse[j];
                coefficients[row][column] = inverse[row] * operand(positive, bf16(x.beta[(begin + column) * x.hv + head]));
            }
        }
        var rhs: [block]f64 = @splat(0);
        var query_state: [block]f64 = @splat(0);
        var delta: [block]f64 = @splat(0);
        for (0..count) |row| {
            var predicted: f64 = 0;
            for (state, 0..) |s, key| {
                predicted += s * operand(positive, x.qk(x.k, begin + row, head, key));
                query_state[row] += s * operand(positive, x.qk(x.q, begin + row, head, key));
            }
            rhs[row] = operand(positive, bf16(x.v[((begin + row) * x.hv + head) * width + value])) +
                (if (positive) @as(f64, 1) else -1) * prefix[row] * predicted;
        }
        for (0..count) |row| {
            for (0..row + 1) |j| delta[row] += coefficients[row][j] * rhs[j];
            out[begin + row] = prefix[row] * query_state[row];
            for (0..row + 1) |j| out[begin + row] += attention[row][j] * delta[j];
        }
        for (state, 0..) |*s, key| {
            s.* *= prefix[count - 1];
            for (0..count) |j| s.* += suffix[j] * delta[j] * operand(positive, x.qk(x.k, begin + j, head, key));
        }
    }
}

pub const Fixture = struct {
    a: std.mem.Allocator,
    x: Input,
    pub fn init(a: std.mem.Allocator, rows: usize, hk: usize, hv: usize) !Fixture {
        const q = try a.alloc(u16, rows * hk * width);
        errdefer a.free(q);
        const k = try a.alloc(u16, q.len);
        errdefer a.free(k);
        const v = try a.alloc(u16, rows * hv * width);
        errdefer a.free(v);
        const decay = try a.alloc(f32, rows * hv);
        errdefer a.free(decay);
        const beta = try a.alloc(u16, decay.len);
        errdefer a.free(beta);
        const state = try a.alloc(f32, hv * width * width);
        errdefer a.free(state);
        var random = std.Random.DefaultPrng.init(777 + rows + hk + hv);
        for (q) |*p| p.* = narrow((random.random().float(f32) * 2 - 1) / 128);
        for (k) |*p| p.* = narrow((random.random().float(f32) * 2 - 1) / 16);
        for (v) |*p| p.* = narrow(random.random().float(f32) * 2 - 1);
        for (decay) |*p| p.* = 0.875 + random.random().float(f32) * 0.125;
        for (beta) |*p| p.* = narrow(random.random().float(f32) * 0.5);
        for (state) |*p| p.* = (random.random().float(f32) * 2 - 1) / 8;
        return .{ .a = a, .x = .{ .rows = rows, .hk = hk, .hv = hv, .q = q, .k = k, .v = v, .decay = decay, .beta = beta, .state = state } };
    }
    pub fn deinit(f: *Fixture) void {
        f.a.free(f.x.q);
        f.a.free(f.x.k);
        f.a.free(f.x.v);
        f.a.free(f.x.decay);
        f.a.free(f.x.beta);
        f.a.free(f.x.state);
    }
};

fn agree(x: Input) !void {
    const a = std.testing.allocator;
    const direct = try a.alloc(f64, x.rows);
    defer a.free(direct);
    const chunks = try a.alloc(f64, x.rows);
    defer a.free(chunks);
    for (0..x.hv) |head| for ([_]usize{ 0, 63, 127 }) |value| {
        var s0: [width]f64 = undefined;
        var s1: [width]f64 = undefined;
        serial(x, head, value, direct, &s0);
        blocked(x, head, value, chunks, &s1);
        for (direct, chunks) |first, second| {
            if (std.math.isNan(first)) try std.testing.expect(std.math.isNan(second)) else if (std.math.isInf(first)) try std.testing.expectEqual(first, second) else try std.testing.expectApproxEqAbs(first, second, 1e-11);
        }
        for (s0, s1) |first, second| {
            if (std.math.isNan(first)) try std.testing.expect(std.math.isNan(second)) else if (std.math.isInf(first)) try std.testing.expectEqual(first, second) else try std.testing.expectApproxEqAbs(first, second, 1e-11);
        }
    };
}

test "GDN chunk reference matches scalar FP64 through tails and head mapping" {
    for ([_]usize{ 1, 15, 16, 17, 31, 32, 33, 65 }) |rows| {
        var f = try Fixture.init(std.testing.allocator, rows, 2, 6);
        defer f.deinit();
        try agree(f.x);
    }
}
test "GDN chunk reference does not divide by zero decay and preserves beta zero" {
    var f = try Fixture.init(std.testing.allocator, 33, 1, 3);
    defer f.deinit();
    @constCast(f.x.decay)[15 * 3] = 0;
    @constCast(f.x.decay)[16 * 3 + 1] = 0;
    @memset(@constCast(f.x.beta)[0..3], 0);
    try agree(f.x);
}
test "GDN chunk reference retains nonfinite scalar semantics" {
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |bad| {
        var f = try Fixture.init(std.testing.allocator, 17, 1, 3);
        defer f.deinit();
        @constCast(f.x.v)[(15 * 3 + 1) * width] = narrow(bad);
        try std.testing.expect(!f.x.finiteHead(1));
        try std.testing.expect(f.x.finiteHead(0));
        try agree(f.x);
    }
}

test "GDN chunk positive polynomial dominates its signed FP64 value" {
    var f = try Fixture.init(std.testing.allocator, 17, 1, 3);
    defer f.deinit();
    var out: [17]f64 = undefined;
    var mag: [17]f64 = undefined;
    var state: [width]f64 = undefined;
    var state_mag: [width]f64 = undefined;
    for (0..3) |head| {
        blocked(f.x, head, 63, &out, &state);
        magnitude(f.x, head, 63, &mag, &state_mag);
        for (out, mag) |v, m| try std.testing.expect(@abs(v) <= m);
        for (state, state_mag) |v, m| try std.testing.expect(@abs(v) <= m);
    }
}
