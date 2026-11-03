//! The joint schema head's pointwise kernels under the contract on HIP: BF16 widening and rounding, the gated logit.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Fns = hip.joint_schema.Fns;

pub const Which = enum { widen, round, combine, combine_unrounded };

fn none(_: c.Shape) usize {
    return 0;
}

/// rows x n values: V of a 16,384-row memory, a query block's context, a record's 1,024 options.
const wide = [_]c.Shape{ .{ .rows = 16384, .n = 1024, .k = 1 }, .{ .rows = 64, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 3, .k = 1 } };
const options = [_]c.Shape{ .{ .rows = 1, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 9, .k = 1 } };
const flat_tiles = [3]c.Tile{ .{}, .{ .size = 256 }, .{} };
/// Clef-Flash's scale and gate as the head holds them: bf16(exp(2.296875)) and bf16(sigmoid(-2)).
const joint_scale: f32 = 9.9375;
const gate: f32 = 0.11914062;

pub fn spec(which: Which) c.Spec {
    return switch (which) {
        .widen => .{ .name = "hip joint schema tf_js_widen", .accum = .none, .depth = none, .out = .f32, .tiles = flat_tiles, .served = &wide },
        .round => .{ .name = "hip joint schema tf_js_round", .accum = .none, .depth = none, .out = .bf16, .tiles = flat_tiles, .served = &wide },
        .combine, .combine_unrounded => .{
            .name = if (which == .combine) "hip joint schema tf_js_combine" else "control: gated logit missing its roundings",
            .accum = .none,
            .depth = none,
            .out = .bf16,
            .tiles = flat_tiles,
            .served = &options,
        },
    };
}

pub fn control(which: Which) ?g.Gate {
    return if (which == .combine_unrounded) .shapes else null;
}

pub const Point = struct {
    rig: rig_.Rig,
    which: Which,
    fns: Fns,
    /// The control's kernel: the gated logit with its intermediate roundings left out.
    unrounded: ?hip.Function = null,
    arena: std.heap.ArenaAllocator,
    n: usize = 0,
    x: []f32 = &.{}, // widen and round: the input values; combine: prior, cosine and residual in turn
    in: [3]u64 = .{ 0, 0, 0 },
    out: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, fns: Fns, ctl: hip.Module, which: Which) !Point {
        const f = if (which == .combine_unrounded) try ctl.function("tf_ctl_js_combine_unrounded") else null;
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .which = which, .fns = fns, .unrounded = f, .arena = .init(a) };
    }

    pub fn deinit(self: *Point) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Point) g.Kernel {
        return .{ .spec = spec(self.which), .ctx = self, .vt = &vt, .control = control(self.which) };
    }

    const vt = g.hooks(Point);

    pub fn prepare(self: *Point, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.n = shape.rows * shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const combine = self.which == .combine or self.which == .combine_unrounded;
        const parts: usize = if (combine) 3 else 1;
        self.x = try a.alloc(f32, parts * self.n);
        for (self.x, 0..) |*v, i| {
            const f = (rng.float(f32) * 2 - 1) * (if (i % 9 == 4) @as(f32, 300) else 3);
            // widening and the gated logit read BF16 values; rounding reads any FP32 value
            v.* = if (self.which == .round) f else c.fromBf16(c.toBf16(f));
        }
        if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |p, i| {
            for (0..parts) |part| self.x[part * self.n + (i * 5 + part) % self.n] = p;
        };
        switch (self.which) {
            .widen => {
                const h = try a.alloc(u16, self.n);
                for (h, self.x) |*o, v| o.* = c.toBf16(v);
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(h));
                self.out = try self.rig.output(self.n * 4);
            },
            .round => {
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.x));
                self.out = try self.rig.output(self.n * 2);
            },
            .combine, .combine_unrounded => {
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.x[0..self.n]));
                self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.x[self.n..][0..self.n]));
                const res = try a.alloc(u16, self.n);
                for (res, self.x[2 * self.n ..]) |*o, v| o.* = c.toBf16(v);
                self.in[2] = try self.rig.input(std.mem.sliceAsBytes(res));
                self.out = try self.rig.output(self.n * 2);
            },
        }
        return true;
    }

    pub fn run(self: *Point) !void {
        switch (self.which) {
            .widen, .round => try self.fns.convert(self.rig.s, self.which == .widen, self.in[0], self.out, self.n),
            .combine => try self.fns.combine(self.rig.s, self.in[0], self.in[1], self.in[2], self.n, joint_scale, gate, self.out),
            .combine_unrounded => {
                var a: hip.launch.Args = .{};
                for ([_]u64{ self.in[0], self.in[1], self.in[2] }) |p| try a.add(p);
                try a.add(@as(c_int, @intCast(self.n)));
                try a.add(joint_scale);
                try a.add(gate);
                try a.add(self.out);
                try hip.launch.launch(self.unrounded.?, .{ .grid = .{ .x = @intCast((self.n + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &a);
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Point) usize {
        return self.n;
    }

    pub fn reference(self: *Point, want: []f64, mag: []f64) !void {
        @memset(mag, 0);
        for (want, 0..) |*w, i| w.* = switch (self.which) {
            .widen, .round => self.x[i],
            .combine, .combine_unrounded => gated(self.x[i], self.x[self.n + i], self.x[2 * self.n + i]),
        };
    }

    /// The gated logit holds the spec's rounding points exactly: every output must equal the reference's.
    pub fn bounds(self: *Point, _: []const f64, _: []const f64, allow: []f64) !void {
        if (self.which != .combine and self.which != .combine_unrounded) return;
        @memset(allow, 0x1p-133);
    }
};

fn r16(v: f32) f32 {
    return c.fromBf16(c.toBf16(v));
}

/// bf16(prior + bf16(gate bf16(bf16(js cos) + res))) in FP32 between its BF16 roundings, as the head computes it.
fn gated(prior: f32, cosine: f32, res: f32) f64 {
    const joint = r16(r16(joint_scale * cosine) + res);
    return r16(prior + r16(gate * joint));
}
