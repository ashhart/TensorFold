//! HY4 sink finish and split MLA merge under the raw-output kernel contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const rows = core.contract.rows;
const latent = 512;
const Kind = enum { sink, merge };

const sink_shapes = [_]c.Shape{ .{ .rows = 1, .n = 1, .k = 64 }, .{ .rows = 2, .n = 3, .k = 128 }, .{ .rows = 4, .n = 2, .k = latent } };
const merge_shapes = [_]c.Shape{ .{ .rows = 1, .n = 1, .k = 1 }, .{ .rows = 2, .n = 3, .k = 4 }, .{ .rows = 4, .n = 2, .k = 32 } };

fn noDepth(_: c.Shape) usize {
    return 0;
}
fn mergeDepth(s: c.Shape) usize {
    return s.k + 1;
}

pub const Entry = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    kind: Kind,
    shape: c.Shape = undefined,
    maxima: []f32 = &.{},
    mass: []f32 = &.{},
    numerator: []f32 = &.{},
    sinks: []f32 = &.{},
    dev: [4]u64 = @splat(0),
    out: u64 = 0,

    pub fn kernel(self: *Entry) g.Kernel {
        const name = if (self.kind == .sink) "cuda hy4 tf_hy4_finish_attention" else "cuda hy4 tf_hy4_attend_merge";
        return .{ .spec = if (self.kind == .sink)
            .{ .name = name, .accum = .none, .depth = noDepth, .ops = 8, .out = .f32, .served = &sink_shapes }
        else
            .{ .name = name, .accum = .f32, .depth = mergeDepth, .ops = 8, .mids = 1, .out = .bf16, .served = &merge_shapes, .sums = .{ .why = "split partials use fp32 mass/numerator merges before the bf16 output" } }, .ctx = self, .vt = &vt };
    }
    const vt = g.hooks(Entry);

    fn width(self: *const Entry) usize {
        return if (self.kind == .sink) self.shape.k else latent;
    }
    fn plane(self: *const Entry) usize {
        return self.shape.rows * self.shape.n;
    }
    fn chunks(self: *const Entry) usize {
        return if (self.kind == .merge) self.shape.k else 1;
    }
    pub fn count(self: *const Entry) usize {
        return self.plane() * self.width();
    }

    fn free(self: *Entry) void {
        const a = self.rig.gpu.gpa;
        for ([_][]f32{ self.maxima, self.mass, self.numerator, self.sinks }) |x| a.free(x);
        self.maxima = &.{};
        self.mass = &.{};
        self.numerator = &.{};
        self.sinks = &.{};
    }
    pub fn deinit(self: *Entry) void {
        self.free();
        self.rig.deinit();
    }

    pub fn prepare(self: *Entry, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        self.maxima = try a.alloc(f32, self.plane() * self.chunks());
        self.mass = try a.alloc(f32, self.maxima.len);
        self.numerator = try a.alloc(f32, self.maxima.len * self.width());
        self.sinks = try a.alloc(f32, shape.n);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.maxima) |*x| x.* = @floatCast(rng.floatNorm(f64));
        for (self.mass) |*x| x.* = @floatCast(0.5 + rng.float(f64));
        for (self.numerator) |*x| x.* = @floatCast(rng.floatNorm(f64));
        for (self.sinks) |*x| x.* = @floatCast(rng.floatNorm(f64));
        switch (fill) {
            .random => {},
            .nonfinite => self.plantNonfinite(),
            .stress => try self.stress(rng),
        }
        if (fill == .random) self.plantAdversarial();
        const bytes = [_][]const u8{ std.mem.sliceAsBytes(self.maxima), std.mem.sliceAsBytes(self.mass), std.mem.sliceAsBytes(self.numerator), std.mem.sliceAsBytes(self.sinks) };
        for (bytes, &self.dev) |x, *p| p.* = try self.rig.input(x);
        self.out = try self.rig.output(self.count() * if (self.kind == .sink) 4 else 2);
        return true;
    }

    fn plantNonfinite(self: *Entry) void {
        self.numerator[5] = std.math.nan(f32);
        self.numerator[9] = std.math.inf(f32);
        if (self.plane() > 1) {
            const at = self.width();
            for (self.numerator[at..][0..self.width()]) |*x| x.* = std.math.nan(f32);
        }
    }

    fn plantAdversarial(self: *Entry) void {
        if (self.kind == .sink) {
            self.sinks[0] = 5;
            self.maxima[0] = 5;
            self.numerator[0] = -0.0;
            self.numerator[1] = @bitCast(@as(u32, 1));
            if (self.plane() > 1) {
                self.maxima[1] = 3;
                self.numerator[self.width()] = -0.0;
            }
            if (self.shape.n > 1) self.sinks[1] = 5;
        } else {
            for (self.maxima) |*x| x.* = 4;
            self.maxima[0] = 5;
            if (self.maxima.len > 1) self.maxima[self.plane()] = 5;
            @memset(self.sinks, 0);
            self.numerator[0] = -0.0;
            self.numerator[1] = @bitCast(@as(u32, 1));
            self.numerator[2] = @bitCast(@as(u32, 0x80000001));
        }
    }

    fn stress(self: *Entry, rng: std.Random) !void {
        @memset(self.maxima, 0);
        @memset(self.mass, 1);
        @memset(self.sinks, 0);
        if (self.kind == .sink) return;
        const a = self.rig.gpu.gpa;
        const terms = try a.alloc(f32, self.shape.k);
        defer a.free(terms);
        for (0..self.plane()) |rh| for (0..latent) |col| {
            attn.mirrored(rng, terms);
            for (terms, 0..) |x, chunk| self.numerator[(chunk * self.plane() + rh) * latent + col] = x;
        };
    }

    pub fn run(self: *Entry) !void {
        var args: cuda.Args = .{};
        if (self.kind == .sink) for (self.dev) |p| args.add(p) else for ([_]usize{ 2, 0, 1, 3 }) |i| args.add(self.dev[i]);
        args.add(self.out);
        const s = self.shape;
        if (self.kind == .sink) {
            for ([_]usize{ s.rows, s.n, s.k }) |x| args.add(@as(c_uint, @intCast(x)));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.plane()), .y = @intCast((s.k + 255) / 256) }, .block = .{ .x = 256 } }, self.rig.s, &args);
        } else {
            args.add(@as(i64, @intCast(s.rows)));
            args.add(@as(i64, @intCast(s.n)));
            args.add(@as(c_int, @intCast(s.k)));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(s.rows), .y = @intCast(s.n) }, .block = .{ .x = 128 } }, self.rig.s, &args);
        }
        try self.rig.s.synchronize();
    }

    pub fn reference(self: *Entry, want: []f64, mag: []f64) !void {
        for (want, mag, 0..) |*w, *m, i| {
            const rh = i / self.width();
            const head = rh % self.shape.n;
            const col = i % self.width();
            var top: f64 = self.sinks[head];
            if (self.kind == .sink) top = @max(top, self.maxima[rh]) else for (0..self.chunks()) |chunk| top = @max(top, self.maxima[chunk * self.plane() + rh]);
            var denom: f64 = @exp(self.sinks[head] - top);
            var acc: f64 = 0;
            var absolute: f64 = 0;
            for (0..self.chunks()) |chunk| {
                const base = chunk * self.plane() + rh;
                const weight = @exp(self.maxima[base] - top);
                const value = self.numerator[base * self.width() + col] * weight;
                denom += self.mass[base] * weight;
                acc += value;
                absolute += @abs(value);
            }
            w.* = acc / denom;
            m.* = absolute / denom;
        }
    }

    pub fn simulate(self: *Entry, acc: c.Accum, order: g.Order, out: []f64) !void {
        if (self.kind == .sink) return error.NoSums;
        const a = self.rig.gpu.gpa;
        const terms = try a.alloc(f64, self.chunks() + 1);
        defer a.free(terms);
        const ones = try a.alloc(f64, self.chunks() + 1);
        defer a.free(ones);
        @memset(ones, 1);
        for (out, 0..) |*o, i| {
            const rh = i / latent;
            const col = i % latent;
            for (0..self.chunks()) |chunk| terms[chunk] = self.numerator[(chunk * self.plane() + rh) * latent + col];
            terms[self.chunks()] = 0;
            o.* = rows.sumIn(acc, order, terms) / rows.sumIn(acc, order, ones);
        }
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    sink_mod: cuda.Module,
    merge_mod: cuda.Module,
    items: [2]Entry,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const sink_mod = try cuda.Module.load(gpu.d, cuda.kernels.hy4_attention_sink);
        errdefer sink_mod.unload();
        const merge_mod = try cuda.Module.load(gpu.d, cuda.kernels.hy4_mla_attention);
        errdefer merge_mod.unload();
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .sink_mod = sink_mod, .merge_mod = merge_mod, .items = undefined };
        self.items[0] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try self.sink_mod.function("tf_hy4_finish_attention"), .kind = .sink };
        self.items[1] = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try self.merge_mod.function("tf_hy4_attend_merge"), .kind = .merge };
        for (&self.items) |*item| try kernels.append(a, item.kernel());
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.items) |*item| item.deinit();
        self.sink_mod.unload();
        self.merge_mod.unload();
        self.a.destroy(self);
    }
};
