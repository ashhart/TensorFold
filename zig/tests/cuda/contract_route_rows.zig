//! route_rows' router logits under the contract: tf_route_logits against float64, its other row forms bit-equal.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Dense = contract.dense.Dense;
const rr = cuda.route_rows;

pub const Kind = enum { logits, few, wide };

/// One fp32 MMA chain over the inputs from zero.
fn chain(s: c.Shape) usize {
    return s.k / 16 + 16;
}

/// Flash Next's router: 2560 inputs, 512 experts and the shared one.
const fn_e = 513;
const fn_d = 2560;

pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .logits => .{ .name = "cuda route_rows tf_route_logits", .accum = .f32, .depth = chain, .out = .f32, .tiles = .{ .{ .size = 16 }, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 128, .step = 16, .min = 16 } }, .served = &.{ .{ .rows = 32, .n = fn_e, .k = fn_d }, .{ .rows = 17, .n = fn_e, .k = fn_d }, .{ .rows = 63, .n = fn_e, .k = fn_d } } },
        .few => .{ .name = "cuda route_rows tf_route_logits_few", .accum = .f32, .depth = chain, .out = .f32, .replaces = "tf_route_logits", .tiles = .{ .{ .size = 16 }, .{ .size = 32, .step = 1, .min = 1 }, .{ .size = 256, .step = 16, .min = 16 } }, .served = &.{ .{ .rows = 16, .n = fn_e, .k = fn_d }, .{ .rows = 1, .n = fn_e, .k = fn_d } } },
        .wide => .{ .name = "cuda route_rows tf_route_logits_wide", .accum = .f32, .depth = chain, .out = .f32, .replaces = "tf_route_logits", .tiles = .{ .{ .size = 128 }, .{ .size = 64, .step = 1, .min = 1 }, .{ .size = 64, .step = 16, .min = 16 } }, .served = &.{ .{ .rows = 64, .n = fn_e, .k = fn_d }, .{ .rows = 300, .n = fn_e, .k = fn_d } } },
    };
}

pub const Logits = struct {
    rig: rig_.Rig,
    kind: Kind,
    route: *const rr.RouteRows,
    d: ?Dense = null,
    x: u64 = 0,
    w: u64 = 0,
    base: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Logits) void {
        if (self.d) |d| d.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Logits) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Logits);

    pub fn prepare(self: *Logits, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (self.d) |d| d.deinit();
        self.d = null;
        self.rig.reset();
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 16 != 0) return false;
        const a = self.rig.gpu.gpa;
        const d = try Dense.init(a, shape, 1);
        self.d = d;
        d.fillX(fill, seed, 1);
        const w = try a.alloc(u16, shape.n * shape.k);
        defer a.free(w);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x726f7574);
        const rng = prng.random();
        const k = shape.k;
        for (0..shape.n) |j| for (0..k) |i| {
            // gate 4: integers mirrored into the second half, so fp32 sums are exact in any order
            const v: f32 = if (fill != .stress) (rng.float(f32) - 0.5) * 0.1 else if (i < k / 2) @floatFromInt(1 + rng.uintLessThan(u32, 7)) else -c.fromBf16(w[j * k + i - k / 2]);
            w[j * k + i] = c.toBf16(v);
            d.w[j * k + i] = c.fromBf16(w[j * k + i]);
        };
        self.x = try self.rig.input(std.mem.sliceAsBytes(d.x));
        self.w = try self.rig.input(std.mem.sliceAsBytes(w));
        self.out = try self.rig.output(shape.rows * shape.n * 4);
        if (self.kind != .logits) self.base = try self.rig.scratch(shape.rows * shape.n * 4);
        return true;
    }

    pub fn run(self: *Logits) !void {
        try self.launch(self.kind, self.out);
        try self.rig.s.synchronize();
    }

    /// RouteRows.logits' launch of one form, whatever the row count.
    fn launch(self: *Logits, kind: Kind, out: u64) !void {
        const sh = self.d.?.shape;
        var a: cuda.Args = .{};
        a.add(self.x);
        a.add(@as(i32, @intCast(sh.k)));
        a.add(self.w);
        a.add(out);
        for ([_]usize{ sh.rows, sh.k, sh.n }) |v| a.add(@as(i32, @intCast(v)));
        const r = self.route;
        switch (kind) {
            .wide => try cuda.launch.launch(r.wide_fn, .{ .grid = .{ .x = @intCast((sh.rows + 127) / 128), .y = @intCast((sh.n + 63) / 64) }, .block = .{ .x = 256 }, .shared = rr.wide_shared }, self.rig.s, &a),
            .few => try cuda.launch.launch(r.few_fn, .{ .grid = .{ .x = @intCast((sh.rows + 15) / 16), .y = @intCast((sh.n + 31) / 32) }, .block = .{ .x = 128 }, .shared = rr.few_shared }, self.rig.s, &a),
            .logits => try cuda.launch.launch(r.logits_fn, .{ .grid = .{ .x = @intCast((sh.rows + 15) / 16), .y = @intCast((sh.n + 31) / 32) }, .block = .{ .x = 128 }, .shared = rr.many_shared }, self.rig.s, &a),
        }
    }

    /// tf_route_logits on the same inputs.
    pub fn replaced(self: *Logits) ![]const u8 {
        if (self.kind == .logits) return error.NotAReplacement;
        try self.launch(.logits, self.base);
        return self.rig.bytes(self.base, self.count() * 4);
    }

    pub fn count(self: *Logits) usize {
        return self.d.?.outputs();
    }

    pub fn reference(self: *Logits, want: []f64, mag: []f64) !void {
        return self.d.?.reference(want, mag);
    }

    pub fn simulate(self: *Logits, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.d.?.simulate(acc, order, out);
    }
};
