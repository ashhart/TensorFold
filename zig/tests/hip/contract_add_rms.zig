//! tf_add_rms under the contract on HIP: the residual add and norm in one launch, bit-equal to the launches it merges.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;

/// What the residual adds: a row of the kind, an fp32 rank sum, or a MoE layer's six expert rows and a shared row.
pub const Mode = enum(c_int) { act = 0, wide = 1, moe = 2 };

const slots = 6;
/// The launch's addend argument, act.hip's AddRmsIn.
const In = @typeInfo(@TypeOf(hip.Launcher.tf_add_rms)).@"fn".param_types[2].?;
const eps: f32 = 1e-5;

/// rms_kernel's sum of squares, a lane every 32nd, then the wave's tree; a MoE row's six products and adds first.
fn depth(s: c.Shape) usize {
    return (s.k + 31) / 32 + 6 + 2 * slots + 1;
}

const why = "a 16-bit output follows the sum of squares as ss^-1/2: a bf16 sum's error is a few output ulps";
const why_bf16 = why ++ "; fp16's overflowing sums give 0, which the bound's residual roundings put at 51x";

pub fn spec(mode: Mode, k: kit.Kind) c.Spec {
    const names = [3][2][]const u8{
        .{ "hip tf_add_rms_f16 (y fp16)", "hip tf_add_rms_bf16 (y bf16)" },
        .{ "hip tf_add_rms_f16 (y fp32)", "hip tf_add_rms_bf16 (y fp32)" },
        .{ "hip tf_add_rms_f16 (the MoE rows)", "hip tf_add_rms_bf16 (the MoE rows)" },
    };
    return .{
        .name = names[@intCast(@backingInt(mode))][@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        // the residual rounded before the norm reads it, directly and through the sum; a MoE row's sum rounded too
        .mids = if (mode == .moe) 4 else 2,
        .out = kit.elem(k),
        .replaces = "the launches it merges",
        .tiles = .{ .{}, .{}, .{ .size = 2048, .min = 1025 } },
        // Nano's 2,688 at a decode row and at a window's rows, and the widest row the launch takes
        .served = &.{
            .{ .rows = 1, .n = 2688, .k = 2688 },
            .{ .rows = 16, .n = 2688, .k = 2688 },
            .{ .rows = 3, .n = 8192, .k = 8192 },
        },
        .sums = if (k == .bf16) .{ .f16 = 40, .bf16 = 0, .why = why_bf16 } else .{ .bf16 = 0, .why = why },
    };
}

/// shape.rows rows of shape.k: x [rows][k] of the kind updated in place, normed [rows][k] of the kind.
pub const AddRms = struct {
    rig: rig_.Rig,
    mode: Mode,
    l: *const hip.Launcher,
    k: kit.Kind,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    x: []f64 = &.{},
    y: []f64 = &.{}, // [rows][k], or the MoE's [rows][slots][k]
    wts: []f64 = &.{},
    shared: []f64 = &.{},
    w: []f64 = &.{},
    x_bytes: []u8 = &.{},
    in: [4]u64 = @splat(0), // y, wts, shared, the norm's weight
    out: [2]u64 = @splat(0), // x, normed
    base: [3]u64 = @splat(0), // the merged launches' x, normed and MoE sum

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, m: Mode) AddRms {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .mode = m, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *AddRms) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *AddRms) g.Kernel {
        return .{ .spec = spec(self.mode, self.k), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(AddRms);

    fn signed(rng: std.Random, lo: f64, hi: f64) f64 {
        const v = lo + rng.float(f64) * (hi - lo);
        return if (rng.boolean()) v else -v;
    }

    /// x in +-[1, 2) and an addend of x's sign at a quarter to a half of it, so the add never cancels.
    pub fn prepare(self: *AddRms, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k <= 1024 or shape.k > 8192) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const k = self.k;
        const n = shape.rows * shape.k;
        self.x = try a.alloc(f64, n);
        self.y = try a.alloc(f64, if (self.mode == .moe) n * slots else n);
        self.wts = try a.alloc(f64, shape.rows * slots);
        self.shared = try a.alloc(f64, n);
        self.w = try kit.floats(a, rng, shape.k, 0.75, 1.25);
        for (self.x) |*v| v.* = kit.round(k, signed(rng, 1, 2));
        for (self.wts) |*v| v.* = @as(f32, @floatCast(0.1 + rng.float(f64) * 0.1)); // six slots of y at most 2
        for (0..n) |i| {
            const sign: f64 = if (self.x[i] < 0) -1 else 1;
            const b = sign * (0.25 + rng.float(f64) * 0.25) * @abs(self.x[i]);
            switch (self.mode) {
                .act => self.y[i] = kit.round(k, b),
                .wide => self.y[i] = @as(f32, @floatCast(b)),
                .moe => {
                    self.shared[i] = @as(f32, @floatCast(b / 2));
                    const r = i / shape.k;
                    const part: f32 = @floatCast(b / 2 / 0.9);
                    for (0..slots) |j| self.y[(r * slots + j) * shape.k + i % shape.k] = part;
                },
            }
        }
        if (fill == .stress) self.stress(seed);
        if (fill == .nonfinite) self.plant();
        self.x_bytes = try a.alloc(u8, n * kit.size(k));
        for (self.x, 0..) |v, i| kit.put(k, self.x_bytes, i, v);
        const yb = if (self.mode == .act) blk: {
            const b = try a.alloc(u8, n * kit.size(k));
            for (self.y, 0..) |v, i| kit.put(k, b, i, v);
            break :blk b;
        } else try kit.f32Bytes(a, self.y);
        const ins = [_][]const u8{
            yb,
            try kit.f32Bytes(a, self.wts),
            try kit.f32Bytes(a, self.shared),
            try kit.f32Bytes(a, self.w),
        };
        for (ins, &self.in) |b, *at| at.* = try self.rig.input(b);
        for (&self.out) |*at| at.* = try self.rig.output(self.x_bytes.len);
        for (&self.base) |*at| at.* = try self.rig.scratch(self.x_bytes.len);
        return true;
    }

    /// Gate 4: the norm's rows of integers (rows.zig's) and nothing added, so its sums are exact in fp32.
    fn stress(self: *AddRms, seed: u64) void {
        const s = self.shape;
        var r = rows.Rows.init(self.arena.allocator(), .rms, .{ .rows = s.rows, .n = s.k, .k = s.k }) catch unreachable;
        r.fill(.stress, seed);
        for (self.x, r.x) |*v, b| v.* = c.fromBf16(b);
        @memset(self.y, 0);
        @memset(self.shared, 0);
    }

    /// NaN in row 0's x, +Inf in row 1's addend and -Inf in row 2's, an overflowing sum in row 3.
    fn plant(self: *AddRms) void {
        const s = self.shape;
        const big: f64 = if (self.k == .f16) 60000 else 3e38;
        self.x[5] = std.math.nan(f64);
        const at = [_]usize{ s.k + 9, 2 * s.k + 1 };
        for ([_]f64{ std.math.inf(f64), -std.math.inf(f64) }, at) |v, i| if (i < self.x.len) switch (self.mode) {
            .moe => self.y[(i / s.k * slots + 1) * s.k + i % s.k] = v,
            else => self.y[i] = v,
        };
        if (3 * s.k + 2 < self.x.len) self.x[3 * s.k + 2] = big;
        if (3 * s.k + 2 < self.x.len and self.mode != .moe) {
            self.y[3 * s.k + 2] = kit.round(if (self.mode == .act) self.k else .f32, big);
        }
    }

    fn ptr(at: u64) ?*anyopaque {
        return @ptrFromInt(at);
    }

    /// x back to the prepared rows, then the merged launch.
    pub fn run(self: *AddRms) !void {
        try self.rig.put(self.out[0], self.x_bytes);
        const s = self.shape;
        const in: In = .{
            .y = self.in[0],
            .mode = @backingInt(self.mode),
            .wts = self.in[1],
            .shared = self.in[2],
            .slots = slots,
        };
        try self.l.tf_add_rms(
            ptr(self.out[0]),
            in,
            @ptrFromInt(self.in[3]),
            ptr(self.out[1]),
            @backingInt(self.k),
            @intCast(s.rows),
            @intCast(s.k),
            eps,
            self.rig.s.handle,
        );
        try self.rig.s.synchronize();
    }

    /// The launches it merges on the same inputs: the add (after tf_nemo_combine for the MoE rows), then the norm.
    pub fn replaced(self: *AddRms) ![]const u8 {
        try self.rig.put(self.base[0], self.x_bytes);
        const s = self.shape;
        const l = self.l;
        const st = self.rig.s.handle;
        const kind = @backingInt(self.k);
        const n: c_longlong = @intCast(s.rows * s.k);
        const x = ptr(self.base[0]);
        switch (self.mode) {
            .act => try l.tf_add(x, ptr(self.in[0]), x, kind, n, st),
            .wide => try l.tf_add_wide(x, @ptrFromInt(self.in[0]), x, kind, n, st),
            .moe => {
                var a: hip.launch.Args = .{};
                for ([_]u64{ self.in[0], self.in[1], self.in[2], self.base[2] }) |v| try a.add(v);
                for ([_]c_int{ kind, slots, @intCast(s.k) }) |v| try a.add(v);
                const grid: hip.launch.Dim3 = .{ .x = @intCast((s.k + 255) / 256), .y = @intCast(s.rows) };
                try hip.launch.launch(l.nemo.combine, .{ .grid = grid, .block = .{ .x = 256 } }, self.rig.s, &a);
                try l.tf_add(x, ptr(self.base[2]), x, kind, n, st);
            },
        }
        try l.tf_rms(x, @ptrFromInt(self.in[3]), ptr(self.base[1]), kind, @intCast(s.rows), @intCast(s.k), eps, st);
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.bytes(self.base[0], self.x_bytes.len));
        const normed = try self.rig.bytes(self.base[1], self.x_bytes.len);
        return std.mem.concat(a, u8, &.{ out, normed });
    }

    pub fn count(self: *AddRms) usize {
        return 2 * self.x.len;
    }

    /// Row i's addend in float64, and the magnitude of the terms the MoE rows' sum carries into it.
    fn addend(self: *AddRms, i: usize) struct { f64, f64 } {
        const s = self.shape;
        if (self.mode != .moe) return .{ self.y[i], 0 };
        const r = i / s.k;
        var acc: f64 = 0;
        var abs: f64 = 0;
        for (0..slots) |j| {
            const t = self.y[(r * slots + j) * s.k + i % s.k] * self.wts[r * slots + j];
            acc += t;
            abs += @abs(t);
        }
        return .{ acc + self.shared[i], abs + @abs(self.shared[i]) };
    }

    /// x + the addend, then rms(x + addend) * w, in float64; the norm's magnitude as rows.zig's.
    pub fn reference(self: *AddRms, want: []f64, mag: []f64) !void {
        const s = self.shape;
        const n = self.x.len;
        for (0..n) |i| {
            const b, const m = self.addend(i);
            const v = self.x[i] + b;
            want[i] = if (std.math.isInf(kit.round(self.k, v))) kit.round(self.k, v) else v; // past the kind's range
            mag[i] = m;
        }
        for (0..s.rows) |r| {
            const row = want[r * s.k ..][0..s.k];
            var ss: f64 = 0;
            for (row) |v| ss += v * v;
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(s.k)) + eps);
            for (row, 0..) |v, j| {
                const i = n + r * s.k + j;
                want[i] = v * inv * self.w[j];
                mag[i] = @abs(want[i]) / 2 + mag[r * s.k + j] * inv * self.w[j];
            }
        }
    }

    pub fn simulate(self: *AddRms, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = self.shape;
        const n = self.x.len;
        const t = try self.rig.a.alloc(f64, s.k);
        defer self.rig.a.free(t);
        for (0..s.rows) |r| {
            const row = self.x[r * s.k ..][0..s.k];
            for (t, row) |*v, x| v.* = x * x;
            const ss = rows.sumIn(acc, order, t);
            const inv = 1 / @sqrt(ss / @as(f64, @floatFromInt(s.k)) + eps);
            for (row, 0..) |x, j| {
                out[r * s.k + j] = x;
                out[n + r * s.k + j] = x * inv * self.w[j];
            }
        }
    }
};
