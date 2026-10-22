//! fn_hc_dense_mix8 under the contract: fn_hc_dense_mix's bits, fp32 group sums; controls: swapped inputs, bf16 sums.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const contract = core.contract;

pub const Kind = enum { mix8, control, sums, sums_control };

const tiles: [3]c.Tile = .{ .{ .size = 1 }, .{ .size = 256, .step = 32, .min = 32 }, .{} };
const served = [_]c.Shape{ .{ .rows = 1, .n = 2560, .k = 4 }, .{ .rows = 17, .n = 2560, .k = 4 }, .{ .rows = 3, .n = 96, .k = 4 } };

const rule = "as fn_hc_dense_mix gives them: NaN and infinite up rows and normed rows";

fn depth(s: c.Shape) usize {
    return s.k + 8;
}

/// A group's 32 stored values summed in order.
fn groupDepth(_: c.Shape) usize {
    return 32;
}

fn sumsKind(kind: Kind) bool {
    return kind == .sums or kind == .sums_control;
}

pub fn spec(kind: Kind) c.Spec {
    if (sumsKind(kind)) return .{
        .name = if (kind == .sums) "cuda Flash Next fn_hc_dense_mix8 group sums (of its stored bf16 values)" else "control: fn_hc_dense_mix8 summing its groups in bf16",
        .accum = .f32,
        .depth = groupDepth,
        .out = .f32,
        .replaces = "fn_hc_dense_mix",
        .nan = .{ .stated = rule },
        .tiles = tiles,
        .served = &served,
    };
    return .{
        .name = if (kind == .mix8) "cuda Flash Next fn_hc_dense_mix8 (replaces fn_hc_dense_mix)" else "control: fn_hc_dense_mix8 with its inputs swapped posing as fn_hc_dense_mix",
        .accum = .f32,
        .depth = depth,
        .ops = 4,
        .mids = 2,
        .out = .bf16,
        .replaces = "fn_hc_dense_mix",
        .nan = .{ .stated = rule },
        .tiles = tiles,
        .served = &served,
        // each value sums four bf16-rounded terms and is itself rounded to bf16: no accumulator's error shows at gate 4
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "four bf16-rounded terms a value, rounded to bf16 again" },
    };
}

pub const Mix = struct {
    rig: rig_.Rig,
    kind: Kind,
    new_fn: cuda.Function,
    old_fn: cuda.Function,
    ctl_fn: ?cuda.Function = null,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = undefined,
    up: []u16 = &.{},
    normed: []u16 = &.{},
    dup: u64 = 0,
    dnormed: u64 = 0,
    mixed: u64 = 0,
    out: u64 = 0,
    base: u64 = 0,
    fill: g.Fill = .random,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Mix) void {
        if (self.arena) |*arena| arena.deinit();
        self.rig.deinit();
    }

    const vt = g.hooks(Mix);

    pub fn kernel(self: *Mix) g.Kernel {
        const control: ?g.Gate = switch (self.kind) {
            .control => .shapes,
            .sums_control => .sums,
            else => null,
        };
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt, .control = control };
    }

    fn rowBytes(self: *const Mix) usize {
        return self.shape.rows * self.shape.n * 2;
    }

    fn sumBytes(self: *const Mix) usize {
        return self.shape.rows * (self.shape.n / 32) * 4;
    }

    fn outBytes(self: *const Mix) usize {
        return self.rowBytes() + self.sumBytes();
    }

    /// Gate 4's groups: up 16 (sigmoid 1 in bf16), each stream's normed 4096, 1, -4096 and zeros: fp32 sums 1, bf16 0.
    fn cancel(self: *Mix) void {
        @memset(self.up, c.toBf16(16));
        const n = self.shape.n;
        for (0..self.shape.rows * self.shape.k) |rs| for (0..n) |d| {
            const at = (d / 32) % 10 * 3;
            const e = d % 32;
            const v: i32 = if (e == at) 4096 else if (e == at + 1) 1 else if (e == at + 2) -4096 else 0;
            self.normed[rs * n + d] = c.toBf16(@floatFromInt(if ((d / 32) % 2 == 0) v else -v));
        };
    }

    pub fn prepare(self: *Mix, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k > 8) return false;
        self.rig.reset();
        if (self.arena) |*arena| arena.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.rig.gpu.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        self.fill = fill;
        const total = shape.rows * shape.k * shape.n;
        self.up = try a.alloc(u16, total);
        self.normed = try a.alloc(u16, total);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.up) |*v| v.* = c.toBf16(rng.float(f32) * 8 - 4);
        for (self.normed) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        if (fill == .nonfinite) {
            self.up[0] = c.toBf16(std.math.nan(f32));
            self.normed[total - 1] = c.toBf16(std.math.inf(f32));
            if (total > 40) self.normed[33] = c.toBf16(-std.math.inf(f32));
        }
        if (sumsKind(self.kind) and fill == .stress) self.cancel();
        self.dup = try self.rig.input(std.mem.sliceAsBytes(self.up));
        self.dnormed = try self.rig.input(std.mem.sliceAsBytes(self.normed));
        if (sumsKind(self.kind)) {
            self.mixed = try self.rig.scratch(self.rowBytes());
            self.out = try self.rig.output(self.sumBytes());
        } else self.out = try self.rig.output(self.outBytes());
        self.base = try self.rig.scratch(self.outBytes());
        return true;
    }

    fn launch(self: *Mix, f: cuda.Function, new: bool, swap: bool, rows: u64, sums: u64) !void {
        var args: cuda.Args = .{};
        for ([_]u64{ if (swap) self.dnormed else self.dup, if (swap) self.dup else self.dnormed, rows, sums }) |v| args.add(v);
        args.add(@as(c_int, @intCast(self.shape.n)));
        args.add(@as(c_int, @intCast(self.shape.k)));
        const groups = self.shape.n / 32;
        const cfg: cuda.launch.Config = if (new) .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast((groups + 7) / 8) }, .block = .{ .x = 256 } } else .{ .grid = .{ .x = @intCast(self.shape.rows), .y = @intCast(groups) }, .block = .{ .x = 32 } };
        try cuda.launch.launch(f, cfg, self.rig.s, &args);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Mix) !void {
        switch (self.kind) {
            .mix8, .control => try self.launch(self.new_fn, true, self.kind == .control, self.out, self.out + self.rowBytes()),
            .sums => try self.launch(self.new_fn, true, false, self.mixed, self.out),
            .sums_control => try self.launch(self.ctl_fn.?, true, false, self.mixed, self.out),
        }
    }

    pub fn replaced(self: *Mix) ![]const u8 {
        try self.launch(self.old_fn, false, false, self.base, self.base + self.rowBytes());
        if (sumsKind(self.kind)) return self.rig.bytes(self.base + self.rowBytes(), self.sumBytes());
        return self.rig.bytes(self.base, self.outBytes());
    }

    pub fn check(self: *Mix, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from fn_hc_dense_mix's", .{ at, got.len });
        for (0..self.count()) |i| if (c.classOf(c.read(spec(self.kind).out, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(self: *Mix) usize {
        return if (sumsKind(self.kind)) self.shape.rows * (self.shape.n / 32) else self.shape.rows * self.shape.n;
    }

    /// Output o's group as stored: under gate 4's fill every stream's normed value, since its sigmoid is 1.
    fn group(self: *const Mix, o: usize, t: *[32]f64) !void {
        if (self.fill != .stress) return error.GroupSumsNeedStress;
        const groups = self.shape.n / 32;
        const at = (o / groups) * self.shape.k * self.shape.n + (o % groups) * 32;
        for (t, 0..) |*v, e| v.* = c.fromBf16(self.normed[at + e]);
    }

    fn values(self: *const Mix, acc: ?c.Accum, order: g.Order, out: []f64, mag: ?[]f64) void {
        const n = self.shape.n;
        const ss = self.shape.k;
        var t: [8]f64 = undefined;
        for (0..self.shape.rows) |r| for (0..n) |d| {
            for (0..ss) |st| {
                const at = (r * ss + st) * n + d;
                const u: f64 = c.fromBf16(self.up[at]);
                const sig = 1.0 / (1.0 + @exp(-u));
                t[st] = sig * @as(f64, c.fromBf16(self.normed[at]));
            }
            const total = if (acc) |ac| contract.rows.sumIn(ac, order, t[0..ss]) else blk: {
                var x: f64 = 0;
                for (t[0..ss]) |v| x += v;
                break :blk x;
            };
            out[r * n + d] = total / @as(f64, @floatFromInt(ss));
            if (mag) |m| {
                var x: f64 = 0;
                for (t[0..ss]) |v| x += @abs(v);
                m[r * n + d] = x / @as(f64, @floatFromInt(ss));
            }
        };
    }

    /// The bf16 roundings of each sigmoid and term (half an ulp each, a bit more) and of the mean, on the sums' bound.
    pub fn bounds(self: *Mix, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (sumsKind(self.kind)) return; // a group sum rounds nothing to bf16: the fp32 bound alone
        const half = std.math.ldexp(@as(f64, 1), -8);
        for (allow, want, mag) |*l, w, m| l.* += 2.01 * half * m + half * @abs(w) + std.math.floatMin(f32);
    }

    pub fn reference(self: *Mix, want: []f64, mag: []f64) !void {
        if (!sumsKind(self.kind)) return self.values(null, .sequential, want, mag);
        var t: [32]f64 = undefined;
        for (want, mag, 0..) |*w, *m, o| {
            try self.group(o, &t);
            w.* = 0;
            m.* = 0;
            for (t) |v| {
                w.* += v;
                m.* += @abs(v);
            }
        }
    }

    pub fn simulate(self: *Mix, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (!sumsKind(self.kind)) return self.values(acc, order, out, null);
        var t: [32]f64 = undefined;
        for (out, 0..) |*o, i| {
            try self.group(i, &t);
            o.* = contract.rows.sumIn(acc, order, &t);
        }
    }
};

test "dense mix group sums: gate 4's groups sum to 1 or -1 in fp32 in every order, 0 in bf16 and fp16" {
    const a = std.testing.allocator;
    try std.testing.expectEqual(@as(f64, 1), c.fromBf16(c.toBf16(@floatCast(1.0 / (1.0 + @exp(-16.0))))));
    const shape: c.Shape = .{ .rows = 1, .n = 2560, .k = 4 };
    var m = Mix{ .rig = undefined, .kind = .sums, .new_fn = undefined, .old_fn = undefined, .shape = shape, .fill = .stress };
    m.up = try a.alloc(u16, shape.rows * shape.k * shape.n);
    defer a.free(m.up);
    m.normed = try a.alloc(u16, m.up.len);
    defer a.free(m.normed);
    m.cancel();
    const n = m.count();
    try std.testing.expectEqual(@as(usize, 80), n);
    const buf = try a.alloc(f64, 3 * n);
    defer a.free(buf);
    const want = buf[0..n];
    const got = buf[2 * n ..];
    try m.reference(want, buf[n .. 2 * n]);
    for (want, 0..) |w, i| try std.testing.expectEqual(@as(f64, if (i % 2 == 0) 1 else -1), w);
    for ([_]g.Order{ .sequential, .pairwise, .strided }) |o| {
        try m.simulate(.f32, o, got);
        try std.testing.expectEqualSlices(f64, want, got);
        for ([_]c.Accum{ .f16, .bf16 }) |acc| {
            try m.simulate(acc, o, got);
            for (got) |v| try std.testing.expectEqual(@as(f64, 0), v);
        }
    }
}
