//! tf_qv_attention64/72: segmented full image attention under the raw-output contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const served = [_]c.Shape{
    .{ .rows = 1, .n = 1, .k = 1 },
    .{ .rows = 1, .n = 130, .k = 130 },
    .{ .rows = 2, .n = 192, .k = 64 },
    .{ .rows = 2, .n = 196, .k = 65 },
    .{ .rows = 16, .n = 130, .k = 130 },
};

fn spec(name: []const u8, hd: usize) c.Spec {
    return .{ .name = name, .accum = .f32, .depth = struct {
        fn depth(s: c.Shape) usize {
            return s.k + 64;
        }
    }.depth, .ops = 6, .out = .bf16, .tiles = .{ .{}, .{ .size = 64 }, .{ .size = hd } }, .served = &served, .sums = .{ .why = attn.why } };
}

pub const Attention = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    name: []const u8,
    hd: usize,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    qkv: std.ArrayList(u16) = .empty,
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn kernel(self: *Attention) g.Kernel {
        return .{ .spec = spec(self.name, self.hd), .ctx = self, .vt = &vt };
    }
    const vt = g.hooks(Attention);

    pub fn deinit(self: *Attention) void {
        if (self.r) |*r| r.deinit();
        self.qkv.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    fn at(self: *Attention, which: usize, row: usize, head: usize) []u16 {
        const plane = self.shape.rows * self.shape.n * self.hd;
        const offset = which * plane + (head * self.shape.n + row) * self.hd;
        return self.qkv.items[offset..][0..self.hd];
    }

    pub fn prepare(self: *Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k > shape.n) return false;
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, self.hd, shape.k);
        try self.qkv.resize(a, 3 * self.count());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.qkv.items) |*x| x.* = attn.normal(rng);
        switch (fill) {
            .random => {},
            .nonfinite => self.plantNonfinite(),
            .stress => try self.stress(rng),
        }
        if (fill == .random) self.plantAdversarial();
        for (0..3) |which| self.dev[which] = try self.rig.input(std.mem.sliceAsBytes(self.qkv.items[which * self.count() ..][0..self.count()]));
        self.out = try self.rig.output(self.count() * 2);
        return true;
    }

    fn plantNonfinite(self: *Attention) void {
        const last = self.shape.n - 1;
        if (self.count() > self.hd) self.at(0, 0, 0)[3] = c.toBf16(std.math.nan(f32));
        self.at(2, last, 0)[7] = c.toBf16(std.math.inf(f32));
        self.at(2, self.shape.n / 2, 0)[9] = c.toBf16(-std.math.inf(f32));
        if (self.shape.rows > 1) self.at(1, self.shape.n / 3, 1)[5] = c.toBf16(std.math.nan(f32));
    }

    fn plantAdversarial(self: *Attention) void {
        const q = self.at(0, 0, 0);
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        for (0..self.shape.n) |row| @memset(self.at(1, row, 0), 0);
        const tile = @min(64, (self.shape.k - 1) / 2);
        if (tile == 0) return;
        const spots = if (self.shape.k > 65)
            [3]usize{ 63, 64, @min(128, self.shape.k - 1) }
        else
            [3]usize{ tile - 1, tile, @min(2 * tile, self.shape.k - 1) };
        for (spots, 0..) |row, i| {
            const key = self.at(1, row, 0);
            key[0] = c.toBf16(if (i < 2) 16 else 32);
            key[1] = if (i == 0) 0x8000 else 0;
            key[2] = if (i == 0) 0x0001 else 0x8001;
            key[3] = if (i == 0) 0x807f else 0x007f;
            const value = self.at(2, row, 0);
            value[0] = c.toBf16(1 + 2 * @as(f32, @floatFromInt(i)));
            value[1] = if (i == 0) 0x8000 else 0;
            value[2] = if (i == 0) 0x0001 else 0x8001;
            value[3] = if (i == 0) 0x8001 else 0x007f;
            value[4] = if (i == 0) 0x007f else 0x807f;
        }
    }

    fn stress(self: *Attention, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const key = try a.alloc(u16, self.hd);
        defer a.free(key);
        const col = try a.alloc(f32, self.shape.k);
        defer a.free(col);
        for (0..self.shape.rows) |head| {
            for (key) |*x| x.* = attn.normal(rng);
            for (0..self.shape.n) |row| @memcpy(self.at(1, row, head), key);
            var start: usize = 0;
            while (start < self.shape.n) {
                const len = @min(self.shape.k, self.shape.n - start);
                for (0..self.hd) |e| {
                    attn.mirrored(rng, col[0..len]);
                    for (col[0..len], 0..) |x, j| self.at(2, start + j, head)[e] = c.toBf16(x);
                }
                start += len;
            }
        }
    }

    pub fn run(self: *Attention) !void {
        var start: usize = 0;
        while (start < self.shape.n) {
            const len = @min(self.shape.k, self.shape.n - start);
            var args: cuda.Args = .{};
            for (self.dev) |p| args.add(p);
            args.add(self.out);
            for ([_]usize{ self.shape.n, self.shape.rows, start, len }) |x| args.add(@as(c_int, @intCast(x)));
            args.add(@as(f32, 1.0));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast((len + 63) / 64), .y = @intCast(self.shape.rows) }, .block = .{ .x = 64 } }, self.rig.s, &args);
            start += len;
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Attention) usize {
        return self.shape.n * self.shape.rows * self.hd;
    }

    pub fn load(self: *Attention, i: usize, hd: *attn.Head) !void {
        const row = i / self.shape.rows;
        const head = i % self.shape.rows;
        const start = (row / self.shape.k) * self.shape.k;
        const len = @min(self.shape.k, self.shape.n - start);
        for (hd.q, self.at(0, row, head)) |*y, x| y.* = c.fromBf16(x);
        for (0..len) |j| for (0..self.hd) |e| {
            hd.k[j * self.hd + e] = c.fromBf16(self.at(1, start + j, head)[e]);
            hd.v[j * self.hd + e] = c.fromBf16(self.at(2, start + j, head)[e]);
        };
        hd.n = len;
    }

    pub fn check(self: *Attention, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.shape.n * self.shape.rows, 1.0, try self.rig.outputBytes(), gate);
    }

    pub fn reference(self: *Attention, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.shape.n * self.shape.rows, 1.0, want, mag);
    }

    pub fn simulate(self: *Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.shape.n * self.shape.rows, 1.0, acc, order, out);
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    items: [2]Attention,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .mod = try cuda.Module.load(gpu.d, cuda.kernels.qwen_vision), .items = undefined };
        errdefer self.mod.unload();
        const names = [2][]const u8{ "cuda qwen vision tf_qv_attention64", "cuda qwen vision tf_qv_attention72" };
        const symbols = [2][]const u8{ "tf_qv_attention64", "tf_qv_attention72" };
        for (&self.items, 0..) |*item, i| {
            item.* = .{ .rig = .{ .gpu = gpu, .s = s }, .f = try self.mod.function(symbols[i]), .name = names[i], .hd = if (i == 0) 64 else 72 };
            errdefer item.rig.deinit();
            try kernels.append(a, item.kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (&self.items) |*item| item.deinit();
        self.mod.unload();
        self.a.destroy(self);
    }
};
