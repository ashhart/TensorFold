//! prefill_attention.cu's remaining full-attention exports, at their serving scales and row tiles.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");
const check = @import("check.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const Kind = enum { siglip, gemma, radio };
const Form = struct { kind: Kind, control: bool = false };
const forms = [_]Form{
    .{ .kind = .siglip }, .{ .kind = .siglip, .control = true },
    .{ .kind = .gemma },  .{ .kind = .gemma, .control = true },
    .{ .kind = .radio },  .{ .kind = .radio, .control = true },
};

const siglip_rows = [_]c.Shape{
    .{ .rows = 16, .n = 72, .k = 1 },
    .{ .rows = 16, .n = 72, .k = 7 },
    .{ .rows = 16, .n = 72, .k = 129 },
    .{ .rows = 16, .n = 72, .k = 308 },
    .{ .rows = 2, .n = 72, .k = 1024 }, // two heads stand for the full tile's 16
};
const gemma_rows = [_]c.Shape{
    .{ .rows = 16, .n = 72, .k = 9 },
    .{ .rows = 16, .n = 72, .k = 63 },
    .{ .rows = 16, .n = 72, .k = 126 },
    .{ .rows = 16, .n = 72, .k = 252 },
    .{ .rows = 2, .n = 72, .k = 2520 },
};
const radio_rows = [_]c.Shape{
    .{ .rows = 16, .n = 80, .k = 16 },
    .{ .rows = 16, .n = 80, .k = 65 },
    .{ .rows = 16, .n = 80, .k = 129 },
    .{ .rows = 16, .n = 80, .k = 576 },
    .{ .rows = 2, .n = 80, .k = 2304 },
};

fn dim(kind: Kind) usize {
    return if (kind == .radio) 80 else 72;
}

fn scale(kind: Kind) f32 {
    return switch (kind) {
        .gemma => 1.0,
        .siglip => 1.0 / @sqrt(@as(f32, 72)),
        .radio => 1.0 / @sqrt(@as(f32, 80)),
    };
}

fn served(kind: Kind) []const c.Shape {
    return switch (kind) {
        .siglip => &siglip_rows,
        .gemma => &gemma_rows,
        .radio => &radio_rows,
    };
}

fn name(f: Form) []const u8 {
    if (f.control) return switch (f.kind) {
        .siglip => "control: SigLIP2 full_kernel<72, 8, 8> doubling its score scale",
        .gemma => "control: Gemma4 full_kernel<72, 8, 8> doubling its unit score scale",
        .radio => "control: RADIO vision_kernel<80, 8> doubling its score scale",
    };
    return switch (f.kind) {
        .siglip => "cuda SigLIP2 prefill full_kernel<72, 8, 8>, 16 heads",
        .gemma => "cuda Gemma4 prefill full_kernel<72, 8, 8>, unit scale",
        .radio => "cuda RADIO prefill vision_kernel<80, 8>, 16 heads",
    };
}

fn keyChain(s: c.Shape) usize {
    return s.k + 64;
}

fn spec(f: Form) c.Spec {
    return .{
        .name = name(f),
        .accum = .f32,
        .depth = keyChain,
        .ops = 1,
        .out = .bf16,
        .tiles = .{ .{}, .{}, .{ .size = 64 } },
        .served = served(f.kind),
        .sums = .{ .why = attn.why },
    };
}

const Full = struct {
    rig: rig_.Rig,
    att: *const cuda.vit.Attention,
    form: Form,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    qkv: std.ArrayList(u16) = .empty,
    dev: u64 = 0,
    out: u64 = 0,

    fn kernel(self: *Full) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }
    const vt = g.hooks(Full);

    fn deinit(self: *Full) void {
        if (self.r) |*r| r.deinit();
        self.qkv.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    fn stride(self: *const Full) usize {
        return 3 * self.shape.rows * dim(self.form.kind);
    }

    fn at(self: *Full, row: usize, which: usize, head: usize) []u16 {
        const d = dim(self.form.kind);
        return self.qkv.items[row * self.stride() + (which * self.shape.rows + head) * d ..][0..d];
    }

    fn prepare(self: *Full, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const d = dim(self.form.kind);
        if (shape.n != d or shape.k == 0) return false;
        const a = self.rig.gpu.gpa;
        self.shape = shape;
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(a, d, shape.k);
        try self.qkv.resize(a, shape.k * self.stride());
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        for (self.qkv.items) |*x| x.* = attn.normal(rng);
        if (fill == .random) self.plantAdversarial();
        switch (fill) {
            .random => {},
            .nonfinite => self.plantNonfinite(),
            .stress => try self.stress(rng),
        }
        self.dev = try self.rig.input(std.mem.sliceAsBytes(self.qkv.items));
        self.out = try self.rig.output(shape.k * shape.rows * d * 2);
        return true;
    }

    fn plantNonfinite(self: *Full) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        const rows = self.shape.k;
        self.at(0, 0, 0)[3] = nan;
        self.at(rows - 1, 2, 0)[7] = inf;
        self.at(rows / 2, 2, 0)[9] = ninf;
        if (rows >= 3) {
            self.at(1, 2, 0)[11] = inf;
            self.at(2, 2, 0)[11] = ninf;
        }
        if (self.shape.rows > 1) self.at(rows / 3, 1, 1)[5] = nan;
    }

    /// Equal maxima straddle a 64-key tile, a later maximum changes, and exact signed/subnormal bits survive upload.
    fn plantAdversarial(self: *Full) void {
        const q = self.at(0, 0, 0);
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        for (0..self.shape.k) |row| @memset(self.at(row, 1, 0), 0);
        const tile = @min(64, (self.shape.k - 1) / 2);
        if (tile == 0) return;
        const spots = [3]usize{ tile - 1, tile, @min(2 * tile, self.shape.k - 1) };
        for (spots, 0..) |row, i| {
            const key = self.at(row, 1, 0);
            key[0] = c.toBf16(if (i < 2) 16 else 32);
            key[1] = if (i == 0) 0x8000 else 0;
            key[2] = if (i == 0) 0x0001 else 0x8001;
            key[3] = if (i == 0) 0x807f else 0x007f;
            const value = self.at(row, 2, 0);
            value[0] = c.toBf16(1 + 2 * @as(f32, @floatFromInt(i)));
            value[1] = if (i == 0) 0x8000 else 0;
            value[2] = if (i == 0) 0x0001 else 0x8001;
            value[3] = if (i == 0) 0x8001 else 0x007f;
            value[4] = if (i == 0) 0x007f else 0x807f;
        }
    }

    fn stress(self: *Full, rng: std.Random) !void {
        const a = self.rig.gpu.gpa;
        const d = dim(self.form.kind);
        const rows = self.shape.k;
        const key = try a.alloc(u16, d);
        defer a.free(key);
        const col = try a.alloc(f32, rows);
        defer a.free(col);
        for (key) |*x| x.* = attn.normal(rng);
        for (0..self.shape.rows) |h| {
            for (0..rows) |row| @memcpy(self.at(row, 1, h), key);
            for (0..d) |e| {
                attn.mirrored(rng, col);
                for (col, 0..) |x, row| self.at(row, 2, h)[e] = c.toBf16(x);
            }
        }
    }

    fn run(self: *Full) !void {
        const d = dim(self.form.kind);
        const sc = scale(self.form.kind) * (if (self.form.control) @as(f32, 2) else 1);
        try self.att.full(self.rig.s, self.dev, self.out, self.shape.k, self.shape.rows, d, sc);
        try self.rig.s.synchronize();
    }

    fn count(self: *Full) usize {
        return self.shape.k * self.shape.rows * dim(self.form.kind);
    }

    fn heads(self: *Full) usize {
        return self.shape.k * self.shape.rows;
    }

    fn load(self: *Full, i: usize, hd: *attn.Head) !void {
        const d = dim(self.form.kind);
        const row = i / self.shape.rows;
        const head = i % self.shape.rows;
        for (hd.q, self.at(row, 0, head)) |*y, x| y.* = c.fromBf16(x);
        for (0..self.shape.k) |j| for (0..d) |e| {
            hd.k[j * d + e] = c.fromBf16(self.at(j, 1, head)[e]);
            hd.v[j * d + e] = c.fromBf16(self.at(j, 2, head)[e]);
        };
        hd.n = self.shape.k;
    }

    fn check(self: *Full, gate: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.heads(), scale(self.form.kind), try self.rig.outputBytes(), gate);
    }
    fn reference(self: *Full, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.heads(), scale(self.form.kind), want, mag);
    }
    fn simulate(self: *Full, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.heads(), scale(self.form.kind), acc, order, out);
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    att: cuda.vit.Attention,
    items: [forms.len]Full = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        self.* = .{ .a = a, .att = try cuda.vit.Attention.load(gpu.d) };
        errdefer self.att.unload();
        errdefer for (self.items[0..self.made]) |*item| item.deinit();
        for (forms) |form| {
            self.items[self.made] = .{ .rig = .{ .gpu = gpu, .s = s }, .att = &self.att, .form = form };
            self.made += 1;
            try kernels.append(a, self.items[self.made - 1].kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*item| item.deinit();
        self.att.unload();
        self.a.destroy(self);
    }
};
