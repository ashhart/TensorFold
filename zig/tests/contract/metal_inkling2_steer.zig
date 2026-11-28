//! Inkling 2 Mini --glp on the GPU: the steered visit output and its reference add, bit for bit in the host's order.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const p = @import("inkling2_runtime_plan");
const sources = @import("inkling2_metal");

const c = contract.spec;
const g = contract.gates;
const hidden = p.hidden;
const Args = extern struct { count: u32, alpha: f32 };

/// The steered visit output, or the add the separately ordered reference launches after main's mlp_partial.
pub const Kind = enum {
    partial,
    add,

    fn symbol(k: Kind) []const u8 {
        return switch (k) {
            .partial => "tf_ink2_mlp_partial_steer",
            .add => "tf_ink2_steer_add",
        };
    }
};

const controls = @import("metal_inkling2_steer_controls.zig");
const Variant = controls.Variant;
const source = controls.source;

pub const Entry = struct { name: []const u8, kind: Kind, alpha: f32, variant: Variant = .production, control: ?g.Gate = null };

/// The served strengths (0.1 and 0.3 round their products, 1 does not) and one control per planted fault.
pub const entries = [_]Entry{
    .{ .name = "metal inkling2 steer mlp_partial_steer tf_ink2_mlp_partial_steer", .kind = .partial, .alpha = 0.1 },
    .{ .name = "metal inkling2 steer mlp_partial_steer alpha 0.3", .kind = .partial, .alpha = 0.3 },
    .{ .name = "metal inkling2 steer mlp_partial_steer alpha 1", .kind = .partial, .alpha = 1 },
    .{ .name = "metal inkling2 steer steer_add tf_ink2_steer_add", .kind = .add, .alpha = 0.1 },
    .{ .name = "control: metal inkling2 steer contracted", .kind = .partial, .alpha = 0.1, .variant = .contracted, .control = .shapes },
    .{ .name = "control: metal inkling2 steer nan", .kind = .partial, .alpha = 0.1, .variant = .nan, .control = .nonfinite },
    .{ .name = "control: metal inkling2 steer unrounded residual", .kind = .partial, .alpha = 0.1, .variant = .unrounded, .control = .shapes },
    .{ .name = "control: metal inkling2 steer shifted direction", .kind = .partial, .alpha = 1, .variant = .shifted, .control = .shapes },
};

/// One kernel at one strength: its inputs, the host's bits for them and the launch.
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    e: Entry,
    rows: usize = 0,
    t: []f32 = &.{},
    x: []u16 = &.{},
    dir: []f32 = &.{},
    refs: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    const vt = g.hooks(Probe);
    const served = [_]c.Shape{ .{ .rows = 1, .n = hidden, .k = 1 }, .{ .rows = 5, .n = hidden, .k = 1 }, .{ .rows = 32, .n = hidden, .k = 1 }, .{ .rows = 256, .n = hidden, .k = 1 } };

    fn depth(_: c.Shape) usize {
        return 0;
    }

    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .control = self.e.control, .spec = .{
            .name = self.e.name,
            .accum = .none,
            .depth = depth,
            .ops = 2,
            .out = .bf16,
            .served = &served,
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "one product and two single additions have no reduction accumulator to discriminate" },
        } };
    }

    fn free(self: *Probe) void {
        self.a.free(self.t);
        self.a.free(self.x);
        self.a.free(self.dir);
        self.t = &.{};
        self.x = &.{};
        self.dir = &.{};
    }

    pub fn deinit(self: *Probe) void {
        self.free();
        self.rig.deinit();
        self.pipe.deinit();
    }

    pub fn count(self: *Probe) usize {
        return self.rows * hidden;
    }

    /// Rows half random, half built so the separately rounded add and a fused multiply-add differ in every element.
    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != hidden or shape.k != 1 or shape.rows == 0 or shape.rows > p.max_rows) return false;
        self.rig.reset();
        self.free();
        self.rows = shape.rows;
        const n = shape.rows * hidden;
        self.t = try self.a.alloc(f32, n);
        self.x = try self.a.alloc(u16, n);
        self.dir = try self.a.alloc(f32, hidden);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.dir) |*d| d.* = rng.float(f32) * 2 - 1;
        for (self.t, self.x) |*t, *x| {
            t.* = rng.float(f32) * 0.5 - 0.25;
            x.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
            for (put, 0..) |v, i| switch (self.e.kind) {
                .partial => self.t[(i * 997) % n] = v,
                .add => self.x[(i * 997) % n] = c.toBf16(v),
            };
        } else try self.order(rng);
        try self.upload();
        return true;
    }

    /// The odd rows (the only row of one): t zero, x the BF16 head of -alpha*d, so the sum is alpha*d's tail.
    fn order(self: *Probe, rng: std.Random) !void {
        const alpha = self.e.alpha;
        // a power-of-two strength makes every product exact: no row tells the two orders apart
        if (@as(u32, @bitCast(alpha)) & 0x7fffff == 0) return;
        for (0..hidden) |i| {
            const d, const head = for (0..1 << 20) |_| {
                const d = rng.float(f32) * 2 - 1;
                const head: u16 = @truncate(@as(u32, @bitCast(-(alpha * d))) >> 16);
                const x = c.fromBf16(head);
                if (p.mlpPartialSteer(0, x, alpha, d) != p.mlpPartialSteerContracted(0, x, alpha, d)) break .{ d, head };
            } else return error.OrderInvisible;
            self.dir[i] = d;
            var r: usize = if (self.rows == 1) 0 else 1;
            while (r < self.rows) : (r += 2) {
                self.t[r * hidden + i] = 0;
                self.x[r * hidden + i] = head;
            }
        }
    }

    fn upload(self: *Probe) !void {
        const n = self.count();
        switch (self.e.kind) {
            .partial => {
                self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(self.t));
                self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(self.x));
            },
            .add => self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(self.x)),
        }
        self.refs[2] = try self.rig.input(std.mem.sliceAsBytes(self.dir));
        self.out = try self.rig.output(n * 2);
    }

    pub fn run(self: *Probe) !void {
        return self.runWith(self.pipe, self.e.alpha);
    }

    /// The launch the engine makes; the add first copies its p into the output, which it then steers in place.
    pub fn runWith(self: *Probe, pipe: mtl.Pipeline, alpha: f32) !void {
        const n = self.count();
        if (self.e.kind == .add) @memcpy(self.rig.bytes(self.out, n * 2), std.mem.sliceAsBytes(self.x));
        const pass = self.rig.begin();
        const e = pass.e;
        e.setPipeline(pipe);
        switch (self.e.kind) {
            .partial => {
                inline for (.{ self.refs[0], self.refs[1], self.out, self.refs[2] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
                e.setValue(Args{ .count = @intCast(n), .alpha = alpha }, 4);
            },
            .add => {
                e.setBuffer(self.out.buf, self.out.off, 0);
                e.setBuffer(self.refs[2].buf, self.refs[2].off, 1);
                e.setValue(Args{ .count = @intCast(n), .alpha = alpha }, 2);
            },
        }
        e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(p.threads, 1, 1));
        try self.rig.finish(pass);
    }

    /// The host's value for element `i` at the strength `alpha`.
    fn want(self: *const Probe, i: usize, alpha: f32) f64 {
        const x = c.fromBf16(self.x[i]);
        const d = self.dir[i % hidden];
        return switch (self.e.kind) {
            .partial => p.mlpPartialSteer(self.t[i], x, alpha, d),
            .add => p.steerAdd(x, alpha, d),
        };
    }

    pub fn reference(self: *Probe, out: []f64, mag: []f64) !void {
        for (out, mag, 0..) |*w, *m, i| {
            w.* = self.want(i, self.e.alpha);
            m.* = @abs(w.*);
        }
    }

    /// Elements whose bits differ from the host's separately rounded order; NaN matches NaN.
    fn differ(self: *Probe, alpha: f32) usize {
        const got = std.mem.bytesAsSlice(u16, self.rig.bytes(self.out, self.count() * 2));
        var n: usize = 0;
        for (got, 0..) |bits, i| {
            const w = self.want(i, alpha);
            if (std.math.isNan(w)) {
                if (!std.math.isNan(c.fromBf16(bits))) n += 1;
            } else if (bits != c.toBf16(@floatCast(w))) n += 1;
        }
        return n;
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes and gate != .nonfinite) return "numeric hooks must be used";
        if (self.differ(self.e.alpha) != 0) return "steered bits differ from the separately rounded host order";
        if (gate == .nonfinite) {
            var planted: usize = 0;
            for (0..self.count()) |i| planted += @intFromBool(!std.math.isFinite(self.want(i, self.e.alpha)));
            if (planted == 0 or planted == self.count()) return "the planted NaN does not discriminate";
        }
        return null;
    }
};

/// The probes, and the cells that tie the fused op to main's two launches and a loaded zero to main.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8, gates: bool, seed: u64) !void {
        for (entries) |e| {
            if (only) |o| if (std.mem.indexOf(u8, e.name, o) == null) continue;
            const text = try source(s.a, e.variant);
            defer s.a.free(text);
            const pipe = try rig_.pipeline(s.device, text, e.kind.symbol());
            errdefer pipe.deinit();
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .pipe = pipe, .e = e };
            try s.probes.append(s.a, probe);
            try s.kernels.append(s.a, probe.kernel());
        }
        if (gates and s.probes.items.len > 0) try s.cells(seed);
    }

    /// For each production probe: fused == main's mlp_partial then the add; zero directions and alpha 0 == main.
    fn cells(s: *Suite, seed: u64) !void {
        const main = try rig_.pipeline(s.device, sources.runtime, "tf_ink2_mlp_partial");
        defer main.deinit();
        const add = try rig_.pipeline(s.device, sources.steer, Kind.add.symbol());
        defer add.deinit();
        var pairs: usize = 0;
        for (s.probes.items) |probe| {
            if (probe.e.variant != .production or probe.e.kind != .partial) continue;
            defer probe.rig.reset();
            for (Probe.served, 0..) |shape, i| {
                if (!try probe.prepare(shape, .random, seed +% i)) return error.RefusedServedShape;
                try probe.run();
                if (probe.differ(probe.e.alpha) != 0) return error.SteeredAddNotSeparatelyRounded;
                const fused = try s.a.dupe(u8, probe.rig.bytes(probe.out, probe.count() * 2));
                defer s.a.free(fused);
                if (!std.mem.eql(u8, fused, try s.twoLaunches(probe, main, add, probe.e.alpha))) return error.FusedDiffersFromTwoLaunches;
                // alpha 0, and a loaded zero direction at the probe's strength, leave main's bits
                try probe.runWith(probe.pipe, 0);
                const off = try s.twoLaunches(probe, main, null, 0);
                if (!std.mem.eql(u8, probe.rig.bytes(probe.out, probe.count() * 2), off)) return error.AlphaZeroDiffersFromMain;
                @memset(probe.rig.bytes(probe.refs[2], hidden * 4), 0);
                try probe.runWith(probe.pipe, probe.e.alpha);
                if (!std.mem.eql(u8, probe.rig.bytes(probe.out, probe.count() * 2), off)) return error.ZeroDirectionDiffersFromMain;
                // the rounded residual may be a negative zero: a +0 direction must keep its bits
                try s.signedZero(probe, main);
                pairs += probe.count();
            }
        }
        for (s.probes.items) |probe| {
            if (probe.e.variant != .production or probe.e.kind != .add) continue;
            defer probe.rig.reset();
            for (Probe.served, 0..) |shape, i| {
                if (!try probe.prepare(shape, .random, seed +% i)) return error.RefusedServedShape;
                try s.signedZero(probe, main);
            }
        }
        std.debug.print("PASS inkling2 steer: {d} elements fused == mlp_partial then steer_add, bit for bit;" ++
            " alpha 0 and a zero direction equal main's mlp_partial\n", .{pairs});
    }

    /// A negative-zero residual with a +0 direction keeps main's bits: adding +0 must not turn -0 into +0.
    fn signedZero(s: *Suite, probe: *Probe, main: mtl.Pipeline) !void {
        const n = probe.count();
        const neg: f32 = @bitCast(@as(u32, 0x8000_0000));
        // the add kernel has no t: it steers the p copied from x, so only the partial probe writes one
        if (probe.e.kind == .partial) {
            for (std.mem.bytesAsSlice(f32, probe.rig.bytes(probe.refs[0], n * 4))) |*v| v.* = neg;
        }
        for (probe.x) |*v| v.* = 0x8000;
        @memset(std.mem.bytesAsSlice(u16, probe.rig.bytes(probe.refs[1], n * 2)), 0x8000);
        @memset(std.mem.bytesAsSlice(f32, probe.rig.bytes(probe.refs[2], hidden * 4)), 0);
        if (probe.e.kind == .partial) {
            const want = try s.twoLaunches(probe, main, null, 0);
            try probe.runWith(probe.pipe, probe.e.alpha);
            if (!std.mem.eql(u8, probe.rig.bytes(probe.out, n * 2), want)) return error.SignedZeroDiffersFromMain;
        } else {
            // the add kernel steers p in place, so a zero step must leave every element's bits alone
            try probe.runWith(probe.pipe, probe.e.alpha);
            if (!std.mem.eql(u8, probe.rig.bytes(probe.out, n * 2), probe.rig.bytes(probe.refs[1], n * 2))) return error.SignedZeroDiffersFromMain;
        }
    }

    /// Main's mlp_partial into scratch, then (with `add`) the reference add over it: the separately ordered bytes.
    fn twoLaunches(s: *Suite, probe: *Probe, main: mtl.Pipeline, add: ?mtl.Pipeline, alpha: f32) ![]const u8 {
        _ = s;
        const n = probe.count();
        const out = try probe.rig.scratch(n * 2);
        const pass = probe.rig.begin();
        const e = pass.e;
        e.setPipeline(main);
        inline for (.{ probe.refs[0], probe.refs[1], out }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(p.RuntimeArgs{ .count = @intCast(n), .boundary = 0 }, 3);
        e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(p.threads, 1, 1));
        if (add) |pipe| {
            e.setPipeline(pipe);
            e.setBuffer(out.buf, out.off, 0);
            e.setBuffer(probe.refs[2].buf, probe.refs[2].off, 1);
            e.setValue(Args{ .count = @intCast(n), .alpha = alpha }, 2);
            e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(p.threads, 1, 1));
        }
        try probe.rig.finish(pass);
        return probe.rig.bytes(out, n * 2);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};
