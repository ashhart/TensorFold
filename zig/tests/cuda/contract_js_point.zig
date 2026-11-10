//! The joint schema head's CUDA pointwise contract: BF16 conversions, gated logits, residual adds and broadcasts.
const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const c = core.contract.spec;
const g = core.contract.gates;
const Fns = @import("qwen35").joint_schema.Fns;

pub const Which = enum { widen, round, combine, add, broadcast };

fn none(_: c.Shape) usize {
    return 0;
}

/// rows x n values: V of a 16,384-row memory, a query block's context, a record's 1,024 options.
const wide = [_]c.Shape{ .{ .rows = 16384, .n = 1024, .k = 1 }, .{ .rows = 64, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 3, .k = 1 } };
const options = [_]c.Shape{ .{ .rows = 1, .n = 1024, .k = 1 }, .{ .rows = 1, .n = 9, .k = 1 } };
const point = [_]c.Shape{ .{ .rows = 64, .n = 4096, .k = 1 }, .{ .rows = 17, .n = 1024, .k = 1 }, .{ .rows = 3, .n = 257, .k = 1 } };
const flat_tiles = [3]c.Tile{ .{}, .{ .size = 256 }, .{} };
/// Clef-Flash's scale and gate as the head holds them: bf16(exp(2.296875)) and bf16(sigmoid(-2)).
const joint_scale: f32 = 9.9375;
const gate: f32 = 0.11914062;

pub fn spec(which: Which, control: bool) c.Spec {
    return .{
        .name = if (control) switch (which) {
            .widen => "control shapes: joint schema widen reads FP16",
            .round => "control shapes: joint schema round truncates",
            .combine => "control shapes: joint schema gated logit missing its roundings",
            .add => "control shapes: joint schema add truncates its sum",
            .broadcast => "control shapes: joint schema broadcast repeats first column",
        } else switch (which) {
            .widen => "cuda joint schema tf_js_widen",
            .round => "cuda joint schema tf_js_round",
            .combine => "cuda joint schema tf_js_combine",
            .add => "cuda joint schema tf_js_add",
            .broadcast => "cuda joint schema tf_js_broadcast",
        },
        .accum = .none,
        .depth = none,
        .out = if (which == .widen) .f32 else .bf16,
        .tiles = flat_tiles,
        .served = switch (which) {
            .widen, .round => &wide,
            .combine => &options,
            .add, .broadcast => &point,
        },
    };
}

pub const Point = struct {
    rig: rig_.Rig,
    which: Which,
    control: bool,
    fns: Fns,
    arena: std.heap.ArenaAllocator,
    n: usize = 0,
    width: usize = 0,
    x: []f32 = &.{}, // combine: prior, cosine and residual; add: x then y; broadcast: one row
    in: [3]u64 = .{ 0, 0, 0 },
    out: u64 = 0,

    pub fn init(gpu: check.Gpu, s: cuda.Stream, fns: Fns, ctl: cuda.Module, which: Which, control: bool) !Point {
        var f = fns;
        if (control) switch (which) {
            .widen => f.k.widen = try ctl.function("tf_ctl_js_widen_half"),
            .round => f.k.round = try ctl.function("tf_ctl_js_round_truncate"),
            .combine => f.k.combine = try ctl.function("tf_ctl_js_combine_unrounded"),
            .add => f.k.add = try ctl.function("tf_ctl_js_add_truncate"),
            .broadcast => f.k.broadcast = try ctl.function("tf_ctl_js_broadcast_first"),
        };
        return .{ .rig = .{ .gpu = gpu, .s = s }, .which = which, .control = control, .fns = f, .arena = .init(gpu.gpa) };
    }

    pub fn deinit(self: *Point) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Point) g.Kernel {
        return .{ .spec = spec(self.which, self.control), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    const vt = g.hooks(Point);

    fn bfInput(self: *Point, x: []const f32) !u64 {
        const h = try self.arena.allocator().alloc(u16, x.len);
        for (h, x) |*o, v| o.* = c.toBf16(v);
        return self.rig.input(std.mem.sliceAsBytes(h));
    }

    pub fn prepare(self: *Point, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.n = shape.rows * shape.n;
        self.width = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const parts: usize = switch (self.which) {
            .combine => 3,
            .add => 2,
            else => 1,
        };
        const len = if (self.which == .broadcast) self.width else self.n;
        self.x = try a.alloc(f32, parts * len);
        for (self.x, 0..) |*v, i| {
            const f = (rng.float(f32) * 2 - 1) * (if (i % 9 == 4) @as(f32, 300) else 3);
            v.* = if (self.which == .round) f else r16(f);
        }
        // A sum above its midpoint distinguishes round-to-nearest-even from truncation, also at the smallest edge.
        if (self.which == .add) {
            self.x[0] = 1;
            self.x[self.n] = 3.0 / 256.0;
        }
        if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |p, i| {
            for (0..parts) |part| self.x[part * len + (i * 5 + part) % len] = p;
        };
        switch (self.which) {
            .widen, .broadcast => self.in[0] = try self.bfInput(self.x),
            .round => self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.x)),
            .add => {
                self.in[0] = try self.bfInput(self.x[0..self.n]);
                self.in[1] = try self.bfInput(self.x[self.n..]);
            },
            .combine => {
                self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.x[0..self.n]));
                self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.x[self.n..][0..self.n]));
                self.in[2] = try self.bfInput(self.x[2 * self.n ..]);
            },
        }
        self.out = try self.rig.output(self.n * @as(usize, if (self.which == .widen) 4 else 2));
        return true;
    }

    pub fn run(self: *Point) !void {
        switch (self.which) {
            .widen, .round => try self.fns.convert(self.rig.s, self.which == .widen, self.in[0], self.out, self.n),
            .combine => try self.fns.combine(self.rig.s, self.in[0], self.in[1], self.in[2], self.n, joint_scale, gate, self.out),
            .add => try self.fns.add(self.rig.s, self.in[0], self.in[1], self.out, self.n),
            .broadcast => try self.fns.broadcast(self.rig.s, self.in[0], self.out, self.n / self.width, self.width),
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
            .combine => gated(self.x[i], self.x[self.n + i], self.x[2 * self.n + i]),
            .add => r16(self.x[i] + self.x[self.n + i]),
            .broadcast => self.x[i % self.width],
        };
    }

    /// Gated logits and residual adds hold the specified BF16 points exactly; a broadcast preserves its input value.
    pub fn bounds(self: *Point, _: []const f64, _: []const f64, allow: []f64) !void {
        if (self.which == .combine or self.which == .add or self.which == .broadcast) @memset(allow, 0x1p-133);
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
