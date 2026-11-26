//! Inkling 2 Mini --glp's fused visit output and mix, byte for byte against the steered output, main's widen and AR.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.glue;
const p = @import("inkling2_runtime_plan");
const sources = @import("inkling2_metal");
const controls = @import("metal_inkling2_steer_fused_controls.zig");

const c = contract.spec;
const g = contract.gates;
const hidden = p.hidden;
const Args = extern struct { count: u32, alpha: f32 };
const ArArgs = extern struct { rows: u32, sources: u32, queries: u32, alpha: f32 };
const symbol = "tf_ink2_mlp_partial_ar_steer";

pub const Entry = struct { name: []const u8, alpha: f32, variant: controls.Variant = .production, control: ?g.Gate = null };

/// The served strengths and one control per planted fault.
pub const entries = [_]Entry{
    .{ .name = "metal inkling2 steer mlp_partial_ar_steer tf_ink2_mlp_partial_ar_steer", .alpha = 0.1 },
    .{ .name = "metal inkling2 steer mlp_partial_ar_steer alpha 0.3", .alpha = 0.3 },
    .{ .name = "metal inkling2 steer mlp_partial_ar_steer alpha 1", .alpha = 1 },
    .{ .name = "control: metal inkling2 steer fused contracted", .alpha = 0.1, .variant = .contracted, .control = .shapes },
    .{ .name = "control: metal inkling2 steer fused nan", .alpha = 0.1, .variant = .nan, .control = .nonfinite },
    .{ .name = "control: metal inkling2 steer fused unrounded residual", .alpha = 0.1, .variant = .unrounded, .control = .shapes },
    .{ .name = "control: metal inkling2 steer fused shifted direction", .alpha = 1, .variant = .shifted, .control = .shapes },
};

fn at(r: rig_.Ref, off: usize) host.Ref {
    return .{ .buf = r.buf, .off = r.off + off };
}

/// One fused launch at one strength: its inputs, and the steered output, main's widen and main's AR as its reference.
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    steer: mtl.Pipeline,
    widen: mtl.Pipeline,
    base: host.Pipelines,
    e: Entry,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    t: []f32 = &.{},
    x: []u16 = &.{},
    dir: []f32 = &.{},
    ins: [6]rig_.Ref = undefined,
    outs: [3]rig_.Ref = undefined,

    const vt = g.hooks(Probe);
    /// rows, queries in n, sources in k: decode and verify rows, the deepest mixes, and whole prompt chunks
    pub const served = [_]c.Shape{
        .{ .rows = 1, .n = hidden, .k = 1 },       .{ .rows = 3, .n = 2 * hidden, .k = 2 }, .{ .rows = 5, .n = 3 * hidden, .k = 3 },
        .{ .rows = 8, .n = hidden, .k = 4 },       .{ .rows = 2, .n = 2 * hidden, .k = 5 }, .{ .rows = 32, .n = hidden, .k = 5 },
        .{ .rows = 256, .n = 2 * hidden, .k = 3 }, .{ .rows = 256, .n = hidden, .k = 5 },
    };

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
            .sums = .{ .f16 = 0, .bf16 = 0, .why = "the steered add has no accumulator; the mix's sums are main's AR, gated by its own entry" },
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
        self.base.deinit();
        self.widen.deinit();
        self.steer.deinit();
        self.pipe.deinit();
    }

    pub fn count(self: *Probe) usize {
        return self.shape.rows * hidden;
    }

    fn queries(self: *const Probe) usize {
        return self.shape.n / hidden;
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n % hidden != 0 or shape.n == 0 or shape.n > 3 * hidden or shape.k == 0 or shape.k > 5) return false;
        if (shape.rows == 0 or shape.rows > p.max_rows) return false;
        self.rig.reset();
        self.free();
        self.shape = shape;
        const rows = shape.rows;
        const n = rows * hidden;
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
            for (put, 0..) |v, i| self.t[(i * 997) % n] = v;
        } else try self.order(rng);
        const blocks = try self.a.alloc(u16, rows * 4 * hidden);
        defer self.a.free(blocks);
        for (blocks) |*b| b.* = c.toBf16(rng.float(f32) * 2 - 1);
        const inv = try self.a.alloc(f32, rows * 4);
        defer self.a.free(inv);
        for (inv) |*v| v.* = 0.5 + rng.float(f32);
        const qs = try self.a.alloc(f32, self.queries() * hidden);
        defer self.a.free(qs);
        for (qs) |*v| v.* = (rng.float(f32) * 2 - 1) * 0.05;
        self.ins = .{
            try self.rig.input(std.mem.sliceAsBytes(self.t)), try self.rig.input(std.mem.sliceAsBytes(self.x)),
            try self.rig.input(std.mem.sliceAsBytes(blocks)), try self.rig.input(std.mem.sliceAsBytes(inv)),
            try self.rig.input(std.mem.sliceAsBytes(qs)),     try self.rig.input(std.mem.sliceAsBytes(self.dir)),
        };
        self.outs = .{ try self.rig.output(n * 2), try self.rig.output(n * 4), try self.rig.output(self.queries() * n * 4) };
        return true;
    }

    /// The odd rows (the only row of one): t zero, x the BF16 head of -alpha*d, so p is alpha*d's tail: an FMA shows.
    fn order(self: *Probe, rng: std.Random) !void {
        const alpha = self.e.alpha;
        if (@as(u32, @bitCast(alpha)) & 0x7fffff == 0) return;
        for (0..hidden) |i| {
            const d, const head = for (0..1 << 20) |_| {
                const d = rng.float(f32) * 2 - 1;
                const head: u16 = @truncate(@as(u32, @bitCast(-(alpha * d))) >> 16);
                const x = c.fromBf16(head);
                if (p.mlpPartialSteer(0, x, alpha, d) != p.mlpPartialSteerContracted(0, x, alpha, d)) break .{ d, head };
            } else return error.OrderInvisible;
            self.dir[i] = d;
            var r: usize = if (self.shape.rows == 1) 0 else 1;
            while (r < self.shape.rows) : (r += 2) {
                self.t[r * hidden + i] = 0;
                self.x[r * hidden + i] = head;
            }
        }
    }

    pub fn run(self: *Probe) !void {
        return self.runWith(self.pipe, self.e.alpha, self.outs);
    }

    /// The engine's launch: main's mlpPartialAr buffers, then the direction and the strength.
    pub fn runWith(self: *Probe, pipe: mtl.Pipeline, alpha: f32, outs: [3]rig_.Ref) !void {
        const pass = self.rig.begin();
        const e = pass.e;
        e.setPipeline(pipe);
        inline for (.{ self.ins[0], self.ins[1], self.ins[2], self.ins[3], self.ins[4], outs[0], outs[1], outs[2], self.ins[5] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(ArArgs{ .rows = @intCast(self.shape.rows), .sources = @intCast(self.shape.k), .queries = @intCast(self.queries()), .alpha = alpha }, 9);
        e.dispatchGroups(mtl.Size.of(self.shape.rows, 1, 1), mtl.Size.of(256, 1, 1));
        try self.rig.finish(pass);
    }

    /// The separately ordered launches: the steered output alone, main's widen, then main's AR per row and query.
    fn separate(self: *Probe, outs: [3]rig_.Ref) !void {
        const rows = self.shape.rows;
        const n = rows * hidden;
        const pass = self.rig.begin();
        const e = pass.e;
        e.setPipeline(self.steer);
        inline for (.{ self.ins[0], self.ins[1], outs[0], self.ins[5] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(Args{ .count = @intCast(n), .alpha = self.e.alpha }, 4);
        e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(p.threads, 1, 1));
        e.setPipeline(self.widen);
        inline for (.{ outs[0], outs[1] }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(p.RuntimeArgs{ .count = @intCast(n), .boundary = 0 }, 2);
        e.dispatchThreads(mtl.Size.of(n, 1, 1), mtl.Size.of(p.threads, 1, 1));
        for (0..self.queries()) |q| for (0..rows) |row| {
            host.cachedAr(&self.base, e, at(self.ins[2], row * 4 * hidden * 2), at(outs[1], row * hidden * 4), at(self.ins[3], row * 16), at(self.ins[4], q * hidden * 4), at(outs[2], (q * rows + row) * hidden * 4), @intCast(self.shape.k), 1);
        };
        try self.rig.finish(pass);
    }

    pub fn reference(self: *Probe, out: []f64, mag: []f64) !void {
        for (out, mag, 0..) |*w, *m, i| {
            w.* = p.mlpPartialSteer(self.t[i], c.fromBf16(self.x[i]), self.e.alpha, self.dir[i % hidden]);
            m.* = @abs(w.*);
        }
    }

    fn lens(self: *const Probe) [3]usize {
        const n = self.shape.rows * hidden;
        return .{ n * 2, n * 4, self.queries() * n * 4 };
    }

    /// The reference's outputs in scratch buffers.
    fn separated(self: *Probe) ![3]rig_.Ref {
        const l = self.lens();
        const outs = [3]rig_.Ref{ try self.rig.scratch(l[0]), try self.rig.scratch(l[1]), try self.rig.scratch(l[2]) };
        try self.separate(outs);
        return outs;
    }

    pub fn check(self: *Probe, gate: g.Gate) !?[]const u8 {
        if (gate != .shapes and gate != .nonfinite) return "numeric hooks must be used";
        const got = std.mem.bytesAsSlice(u16, self.rig.bytes(self.outs[0], self.count() * 2));
        for (got, 0..) |bits, i| {
            const w = p.mlpPartialSteer(self.t[i], c.fromBf16(self.x[i]), self.e.alpha, self.dir[i % hidden]);
            const ok = if (std.math.isNan(w)) std.math.isNan(c.fromBf16(bits)) else bits == c.toBf16(@floatCast(w));
            if (!ok) return "steered p differs from the separately rounded host order";
        }
        const ref = try self.separated();
        for (self.outs, ref, self.lens()) |o, r, len| {
            if (!std.mem.eql(u8, self.rig.bytes(o, len), self.rig.bytes(r, len))) return "fused bytes differ from steer, widen, then AR";
        }
        return null;
    }
};

/// The probes, and the cells that tie a loaded zero and alpha 0 to main's fused mlp_partial_ar.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8, gates: bool, seed: u64) !void {
        for (entries) |e| {
            if (only) |o| if (std.mem.indexOf(u8, e.name, o) == null) continue;
            const text = try controls.source(s.a, e.variant);
            defer s.a.free(text);
            const pipe = try rig_.pipeline(s.device, text, symbol);
            errdefer pipe.deinit();
            const steer = try rig_.pipeline(s.device, sources.steer, "tf_ink2_mlp_partial_steer");
            errdefer steer.deinit();
            const widen = try rig_.pipeline(s.device, sources.runtime, "tf_ink2_widen");
            errdefer widen.deinit();
            var base = try host.Pipelines.load(s.device);
            errdefer base.deinit();
            const probe = try s.a.create(Probe);
            errdefer s.a.destroy(probe);
            probe.* = .{ .a = s.a, .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .pipe = pipe, .steer = steer, .widen = widen, .base = base, .e = e };
            try s.probes.append(s.a, probe);
            try s.kernels.append(s.a, probe.kernel());
        }
        if (gates and s.probes.items.len > 0) try s.cells(seed);
    }

    /// For each production probe and served shape: fused == separate launches; alpha 0 and a zero direction == main.
    fn cells(s: *Suite, seed: u64) !void {
        var rows: usize = 0;
        for (s.probes.items) |probe| {
            if (probe.e.variant != .production) continue;
            defer probe.rig.reset();
            for (Probe.served, 0..) |shape, i| {
                if (!try probe.prepare(shape, .random, seed +% i)) return error.RefusedServedShape;
                try probe.run();
                if (try probe.check(.shapes)) |why| {
                    std.debug.print("FAIL inkling2 steer fused: {s}\n", .{why});
                    return error.FusedSteerDiffers;
                }
                const l = probe.lens();
                const main = [3]rig_.Ref{ try probe.rig.scratch(l[0]), try probe.rig.scratch(l[1]), try probe.rig.scratch(l[2]) };
                const pass = probe.rig.begin();
                host.mlpPartialAr(&probe.base, pass.e, at(probe.ins[0], 0), at(probe.ins[1], 0), at(probe.ins[2], 0), at(probe.ins[3], 0), at(probe.ins[4], 0), at(main[0], 0), at(main[1], 0), at(main[2], 0), @intCast(shape.rows), @intCast(shape.k), @intCast(shape.n / hidden));
                try probe.rig.finish(pass);
                try probe.runWith(probe.pipe, 0, probe.outs);
                for (probe.outs, main, l) |o, m, len| if (!std.mem.eql(u8, probe.rig.bytes(o, len), probe.rig.bytes(m, len))) return error.AlphaZeroDiffersFromMain;
                @memset(probe.rig.bytes(probe.ins[5], hidden * 4), 0);
                try probe.runWith(probe.pipe, probe.e.alpha, probe.outs);
                for (probe.outs, main, l) |o, m, len| if (!std.mem.eql(u8, probe.rig.bytes(o, len), probe.rig.bytes(m, len))) return error.ZeroDirectionDiffersFromMain;
                rows += shape.rows;
            }
        }
        std.debug.print("PASS inkling2 steer fused: {d} rows fused == steered p, main's widen and main's AR, bit for bit;" ++
            " alpha 0 and a zero direction equal main's mlp_partial_ar\n", .{rows});
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};
