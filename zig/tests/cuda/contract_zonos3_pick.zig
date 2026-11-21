//! ZONOS3's picking kernels under the contract: the MoE router and the quantizer's nearest code, by their own rules.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const ref = @import("contract_zonos3_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const experts = 64;

/// Rows x experts: a prompt chunk's rows and the decode row.
pub fn routeSpec(fault: bool) c.Spec {
    return .{
        .name = if (fault) "control: zonos3 z3_route with its bias one expert late" else "cuda zonos3 z3_route",
        .accum = .none,
        .depth = ref.noSums,
        .out = .raw,
        .nan = .{ .stated = "a NaN score never displaces a held pick; a picked NaN or +Inf logit makes both weights NaN" },
        .served = &.{ .{ .rows = 64, .n = experts, .k = 1 }, .{ .rows = 1, .n = experts, .k = 1 }, .{ .rows = 7, .n = experts, .k = 1 } },
    };
}

/// The top two of logits - bias by strict comparison in index order, weighted by the softmax of their raw logits.
pub const Route = struct {
    f: cuda.Function,
    /// control: the device's bias is rotated by one expert
    fault: bool = false,
    rows: usize = 0,
    logits: []f32 = &.{},
    bias: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: [2]u64 = @splat(0),

    pub fn spec(self: *const Route) c.Spec {
        return routeSpec(self.fault);
    }

    pub fn control(self: *const Route) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Route, a: std.mem.Allocator) void {
        a.free(self.logits);
        a.free(self.bias);
        self.logits = &.{};
        self.bias = &.{};
    }

    pub fn prepare(self: *Route, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != experts) return false;
        self.free(a);
        self.rows = shape.rows;
        self.logits = try ref.values(a, rng, shape.rows * experts, fill, 4);
        self.bias = try ref.values(a, rng, experts, fill, 2);
        // ties: two experts of a row share a logit and a bias
        for (0..shape.rows) |r| self.logits[r * experts + 11] = self.logits[r * experts + 40];
        self.bias[11] = self.bias[40];
        if (fill == .nonfinite) {
            const at = [_][2]usize{ .{ 0, 0 }, .{ 1, 5 }, .{ 2, 7 }, .{ 3, 9 } };
            const put = [_]f32{ ref.nan, ref.nan, ref.inf, -ref.inf };
            for (at, put) |p, v| if (p[0] < shape.rows) {
                self.logits[p[0] * experts + p[1]] = v;
            };
        }
        const seen = try a.dupe(f32, self.bias);
        defer a.free(seen);
        if (self.fault) std.mem.rotate(f32, seen, 1);
        self.dev = .{ try ref.input(rig, self.logits), try ref.input(rig, seen) };
        self.out = .{ try rig.output(shape.rows * 2 * 4), try rig.output(shape.rows * 2 * 4) };
        return true;
    }

    pub fn run(self: *Route, rig: *rig_.Rig) !void {
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(self.dev[1]);
        a.add(ref.int(experts));
        a.add(self.out[0]);
        a.add(self.out[1]);
        try ref.go(self.f, rig.s, self.rows, 1, 32, 0, &a);
    }

    pub fn count(self: *const Route) usize {
        return self.rows * 16;
    }

    pub fn check(self: *const Route, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        const weights = got[self.rows * 8 ..];
        for (0..self.rows) |r| {
            const l = self.logits[r * experts ..][0..experts];
            var b: [2]i32 = .{ -1, -1 };
            var v: [2]f32 = .{ -ref.inf, -ref.inf };
            for (l, self.bias, 0..) |x, bias, e| {
                const s = x - bias;
                if (b[0] < 0 or s > v[0]) {
                    b[1] = b[0];
                    v[1] = v[0];
                    b[0] = @intCast(e);
                    v[0] = s;
                } else if (b[1] < 0 or s > v[1]) {
                    b[1] = @intCast(e);
                    v[1] = s;
                }
            }
            const pick = [2]f32{ l[@intCast(b[0])], l[@intCast(b[1])] };
            // fmaxf: a NaN gives way to the other logit
            const m: f32 = if (std.math.isNan(pick[0])) pick[1] else if (std.math.isNan(pick[1])) pick[0] else @max(pick[0], pick[1]);
            const e0 = @exp(@as(f64, pick[0] - m));
            const e1 = @exp(@as(f64, pick[1] - m));
            const want = [2]f64{ e0 / (e0 + e1), e1 / (e0 + e1) };
            for (0..2) |j| {
                const have = ref.readI32(got, r * 2 + j);
                if (have != b[j]) return try std.fmt.bufPrint(msg, "row {d} pick {d}: expert {d} where {d} is due", .{ r, j, have, b[j] });
                const w: f64 = ref.readF32(weights, r * 2 + j);
                const same = if (std.math.isNan(want[j])) std.math.isNan(w) else @abs(w - want[j]) <= 1e-6 * want[j] + 0x1p-140;
                if (!same) return try std.fmt.bufPrint(msg, "row {d} weight {d}: {e} where float64 gives {e}", .{ r, j, w, want[j] });
            }
        }
        return null;
    }
};

const codes = 4096;
const code_dim = 8;
const columns = 10;
const column = 3;

/// Rows x 8 against the codebook's 4,096 unit rows.
pub fn nearestSpec(fault: bool) c.Spec {
    return .{
        .name = if (fault) "control: zonos3 ze_nearest writing the next codebook's column" else "cuda zonos3_encoder ze_nearest",
        .accum = .none,
        .depth = ref.noSums,
        .out = .raw,
        .nan = .{ .stated = "a NaN dot is never the nearest; a row whose every dot is NaN or -Inf gives code 0" },
        .served = &.{ .{ .rows = 64, .n = code_dim, .k = codes }, .{ .rows = 1, .n = code_dim, .k = codes }, .{ .rows = 300, .n = code_dim, .k = codes } },
    };
}

/// The code whose row has the largest dot with a row's 8 values (an fma chain in index order), ties to the lower.
pub const Nearest = struct {
    f: cuda.Function,
    /// control: the device writes column 4
    fault: bool = false,
    rows: usize = 0,
    e: []f32 = &.{},
    book: []f32 = &.{},
    dev: [2]u64 = @splat(0),
    out: u64 = 0,

    pub fn spec(self: *const Nearest) c.Spec {
        return nearestSpec(self.fault);
    }

    pub fn control(self: *const Nearest) ?g.Gate {
        return if (self.fault) .shapes else null;
    }

    pub fn free(self: *Nearest, a: std.mem.Allocator) void {
        a.free(self.e);
        a.free(self.book);
        self.e = &.{};
        self.book = &.{};
    }

    pub fn prepare(self: *Nearest, a: std.mem.Allocator, rig: *rig_.Rig, shape: c.Shape, fill: g.Fill, rng: std.Random) !bool {
        if (shape.rows == 0 or shape.n != code_dim or shape.k != codes) return false;
        self.free(a);
        self.rows = shape.rows;
        self.e = try ref.values(a, rng, shape.rows * code_dim, fill, 1);
        self.book = try ref.values(a, rng, codes * code_dim, .random, 1);
        for (0..codes / 2) |r| {
            const row = self.book[r * code_dim ..][0..code_dim];
            var ss: f32 = 0;
            for (row) |v| ss += v * v;
            for (row) |*v| v.* /= @sqrt(ss);
            // every code has a twin in the upper half: each row's best dot is a tie
            @memcpy(self.book[(codes / 2 + r) * code_dim ..][0..code_dim], row);
        }
        if (fill == .nonfinite) {
            self.e[1] = ref.nan;
            if (shape.rows > 1) self.e[code_dim + 2] = ref.inf;
            if (shape.rows > 2) self.e[2 * code_dim + 5] = -ref.inf;
        }
        self.dev = .{ try ref.input(rig, self.e), try ref.input(rig, self.book) };
        self.out = try rig.output(shape.rows * columns * 4);
        return true;
    }

    pub fn run(self: *Nearest, rig: *rig_.Rig) !void {
        var a: cuda.Args = .{};
        a.add(self.dev[0]);
        a.add(self.dev[1]);
        a.add(self.out);
        a.add(ref.int(column + ref.one(self.fault)));
        try ref.go(self.f, rig.s, self.rows, 1, 256, 0, &a);
    }

    pub fn count(self: *const Nearest) usize {
        return self.rows * columns * 4;
    }

    pub fn check(self: *const Nearest, _: std.mem.Allocator, got: []const u8, msg: []u8) !?[]const u8 {
        for (0..self.rows) |r| {
            const x = self.e[r * code_dim ..][0..code_dim];
            var best: f32 = -ref.inf;
            var want: i32 = 0;
            for (0..codes) |code| {
                var d: f32 = 0;
                for (x, self.book[code * code_dim ..][0..code_dim]) |p, q| d = @mulAdd(f32, p, q, d);
                if (d > best) {
                    best = d;
                    want = @intCast(code);
                }
            }
            for (0..columns) |col| {
                const have = ref.readI32(got, r * columns + col);
                const due: i32 = if (col == column) want else -1;
                if (have != due) return try std.fmt.bufPrint(msg, "row {d} column {d}: {d} where {d} is due", .{ r, col, have, due });
            }
        }
        return null;
    }
};
