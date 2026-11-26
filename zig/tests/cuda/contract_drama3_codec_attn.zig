//! Drama 3 codec causal/windowed attention under the raw-output contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const H = 16;
const CODEC_D = 64;
const CODEC_WIDTH = H * CODEC_D;
const CODEC_QKV = 3 * CODEC_WIDTH;
const codec_scale = 0.125;

/// Encoder uses 512-wide history, decoder prefill/ring use 128; stream k = pos0 + rows, otherwise k = rows.
pub const CodecForm = enum {
    encoder512,
    decoder_prefill128,
    decoder_stream128,
    stream_control,

    fn window(f: CodecForm) usize {
        return if (f == .encoder512) 512 else 128;
    }

    fn stream(f: CodecForm) bool {
        return f == .decoder_stream128 or f == .stream_control;
    }

    fn control(f: CodecForm) bool {
        return f == .stream_control;
    }
};

const codec_encoder = [_]c.Shape{
    .{ .rows = 8, .n = H, .k = 8 },
    .{ .rows = 1, .n = H, .k = 1 },
    .{ .rows = 17, .n = H, .k = 17 },
    .{ .rows = 128, .n = H, .k = 128 },
    .{ .rows = 512, .n = H, .k = 512 },
};
const codec_prefill = [_]c.Shape{
    .{ .rows = 8, .n = H, .k = 8 },
    .{ .rows = 1, .n = H, .k = 1 },
    .{ .rows = 128, .n = H, .k = 128 },
    .{ .rows = 129, .n = H, .k = 129 },
    .{ .rows = 257, .n = H, .k = 257 },
};
const codec_stream = [_]c.Shape{
    .{ .rows = 8, .n = H, .k = 520 },
    .{ .rows = 1, .n = H, .k = 1 },
    .{ .rows = 4, .n = H, .k = 130 },
    .{ .rows = 4, .n = H, .k = 514 },
    .{ .rows = 8, .n = H, .k = 1540 },
};

fn codecDepth(s: c.Shape) usize {
    return CODEC_D + @min(s.k, 512) + 16;
}

fn codecSpec(f: CodecForm) c.Spec {
    return .{
        .name = switch (f) {
            .encoder512 => "cuda drama3 codec c_attn, encoder full causal 512",
            .decoder_prefill128 => "cuda drama3 codec c_attn, decoder prefill causal 128",
            .decoder_stream128 => "cuda drama3 codec c_attn, decoder windowed 512-row ring",
            .stream_control => "control: drama3 codec c_attn starts its window one key late",
        },
        .accum = .f32,
        .depth = codecDepth,
        .ops = 8,
        .out = .f32,
        .served = switch (f) {
            .encoder512 => &codec_encoder,
            .decoder_prefill128 => &codec_prefill,
            .decoder_stream128, .stream_control => &codec_stream,
        },
        .sums = .{ .why = "fp32 Q/K/V and one-warp online softmax reductions" },
    };
}

pub const CodecAttention = struct {
    rig: rig_.Rig,
    form: CodecForm,
    f: cuda.Function,
    ref: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    pos0: usize = 0,
    ring: usize = 0,
    qkv: []f32 = &.{},
    keys: []f32 = &.{},
    values: []f32 = &.{},
    dev: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(rig: rig_.Rig, form: CodecForm, mod: cuda.Module) !CodecAttention {
        return .{ .rig = rig, .form = form, .f = try mod.function("c_attn") };
    }

    pub fn deinit(self: *CodecAttention) void {
        self.clear();
        self.rig.deinit();
    }

    fn clear(self: *CodecAttention) void {
        if (self.ref) |*r| r.deinit();
        self.ref = null;
        const a = self.rig.gpu.gpa;
        inline for (.{ "qkv", "keys", "values" }) |name| {
            a.free(@field(self, name));
            @field(self, name) = &.{};
        }
    }

    pub fn kernel(self: *CodecAttention) g.Kernel {
        return .{ .spec = codecSpec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control()) .shapes else null };
    }

    const vt = g.hooks(CodecAttention);

    pub fn prepare(self: *CodecAttention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or shape.rows == 0 or shape.k < shape.rows) return false;
        self.clear();
        self.rig.reset();
        errdefer self.clear();
        self.shape = shape;
        self.pos0 = shape.k - shape.rows;
        self.ring = if (self.form.stream()) 512 else shape.k;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.ref = try attn.Ref.init(a, CODEC_D, @min(shape.k, self.form.window()));
        self.ref.?.out = .f32;
        self.qkv = try a.alloc(f32, shape.rows * CODEC_QKV);
        self.keys = try a.alloc(f32, self.ring * CODEC_WIDTH);
        self.values = try a.alloc(f32, self.ring * CODEC_WIDTH);
        for (self.qkv) |*x| x.* = rng.floatNorm(f32);
        for (self.keys) |*x| x.* = rng.floatNorm(f32);
        for (self.values) |*x| x.* = rng.floatNorm(f32);
        switch (fill) {
            .random => {},
            .nonfinite => self.plantNonfinite(),
            .stress => try self.plantStress(rng),
        }
        if (fill == .random) self.plantAdversarial();
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.qkv));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.keys));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.values));
        self.out = try self.rig.output(self.count() * @sizeOf(f32));
        return true;
    }

    fn qAt(self: *CodecAttention, row: usize, head: usize) []f32 {
        return self.qkv[row * CODEC_QKV + head * CODEC_D ..][0..CODEC_D];
    }

    fn cacheAt(self: *CodecAttention, values: bool, pos: usize, head: usize) []f32 {
        const x = if (values) self.values else self.keys;
        return x[(pos % self.ring) * CODEC_WIDTH + head * CODEC_D ..][0..CODEC_D];
    }

    fn plantNonfinite(self: *CodecAttention) void {
        self.qAt(0, 0)[3] = std.math.nan(f32);
        const p = self.pos0 + self.shape.rows - 1;
        self.cacheAt(true, p, 8)[7] = std.math.inf(f32);
        if (p > 0) self.cacheAt(true, p - 1, 8)[7] = -std.math.inf(f32);
        self.cacheAt(true, p, 15)[9] = std.math.inf(f32);
    }

    fn plantStress(self: *CodecAttention, rng: std.Random) !void {
        for (0..self.shape.rows) |row| for (0..H) |h| @memset(self.qAt(row, h), 0);
        @memset(self.keys, 0);
        const a = self.rig.gpu.gpa;
        const col = try a.alloc(f32, self.ring);
        defer a.free(col);
        for (0..H) |h| for (0..CODEC_D) |e| {
            attn.mirrored(rng, col);
            for (0..self.ring) |p| self.values[p * CODEC_WIDTH + h * CODEC_D + e] = col[p];
        };
    }

    /// The last row ties one-warp scores then changes the maximum; special values reach Q/K/V at wrapped ring rows.
    fn plantAdversarial(self: *CodecAttention) void {
        const row = self.shape.rows - 1;
        const q = self.qAt(row, 0);
        @memset(q, 0);
        q[0] = 1;
        q[1] = -0.0;
        q[2] = @bitCast(@as(u32, 1));
        q[3] = @bitCast(@as(u32, 0x80000001));
        q[4] = @bitCast(@as(u32, 0x007fffff));
        q[5] = @bitCast(@as(u32, 0x807fffff));
        for (0..self.ring) |p| @memset(self.keys[p * CODEC_WIDTH ..][0..CODEC_D], 0);
        const pos = self.pos0 + row;
        const n = @min(pos + 1, self.form.window());
        const first = pos + 1 - n;
        const p0 = first;
        const p1 = first + @min(8, n - 1);
        const p2 = first + @min(16, n - 1);
        for ([3]usize{ p0, p1, p2 }, 0..) |p, i| {
            if (i > 0 and p == @as(usize, if (i == 1) p0 else p1)) continue;
            const changed = if (p2 != p1) i == 2 else i == 1;
            const k = self.cacheAt(false, p, 0);
            k[0] = if (changed) 16 else 8;
            k[1] = -0.0;
            k[2] = @bitCast(@as(u32, 1));
            k[3] = @bitCast(@as(u32, 0x80000001));
            k[4] = @bitCast(@as(u32, 0x007fffff));
            k[5] = @bitCast(@as(u32, 0x807fffff));
            const v = self.cacheAt(true, p, 0);
            v[0] = if (i == 0) 1 else if (changed) 5 else 3;
            v[1] = if (i == 0) -0.0 else 0.0;
            v[2] = @bitCast(if (i == 0) @as(u32, 1) else @as(u32, 0x80000001));
            v[3] = @bitCast(if (i == 0) @as(u32, 0x80000001) else @as(u32, 0x007fffff));
            v[4] = @bitCast(if (i == 0) @as(u32, 0x007fffff) else @as(u32, 0x807fffff));
        }
    }

    pub fn run(self: *CodecAttention) !void {
        var a: cuda.Args = .{};
        for (self.dev) |x| a.add(x);
        a.add(self.out);
        a.add(@as(c_int, @intCast(self.pos0)));
        a.add(@as(c_int, @intCast(self.form.window() - @intFromBool(self.form.control()))));
        a.add(@as(c_int, @intCast(self.ring)));
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = 2, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *CodecAttention) usize {
        return self.shape.rows * CODEC_WIDTH;
    }

    fn heads(self: *CodecAttention) usize {
        return self.shape.rows * H;
    }

    pub fn load(self: *CodecAttention, i: usize, h: *attn.Head) !void {
        const row = i / H;
        const head = i % H;
        const src = self.qkv[row * CODEC_QKV + head * CODEC_D ..][0..CODEC_D];
        for (h.q, src) |*dst, x| dst.* = x;
        const pos = self.pos0 + row;
        const n = @min(pos + 1, self.form.window());
        const first = pos + 1 - n;
        h.n = n;
        for (0..n) |j| {
            const p = first + j;
            const at = (p % self.ring) * CODEC_WIDTH + head * CODEC_D;
            for (0..CODEC_D) |e| {
                h.k[j * CODEC_D + e] = self.keys[at + e];
                h.v[j * CODEC_D + e] = self.values[at + e];
            }
        }
    }

    pub fn check(self: *CodecAttention, at: g.Gate) !?[]const u8 {
        return self.ref.?.check(self, self.heads(), codec_scale, try self.rig.outputBytes(), at);
    }

    pub fn reference(self: *CodecAttention, want: []f64, mag: []f64) !void {
        return self.ref.?.reference(self, self.heads(), codec_scale, want, mag);
    }

    pub fn simulate(self: *CodecAttention, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.ref.?.simulate(self, self.heads(), codec_scale, acc, order, out);
    }
};

const codec_forms = [_]CodecForm{ .encoder512, .decoder_prefill128, .decoder_stream128, .stream_control };

/// Codec module and registrations, owned at stable addresses for the whole gate/raw run.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: ?cuda.Module = null,
    items: [codec_forms.len]CodecAttention = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.drama3_codec);
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&self.items, codec_forms) |*x, form| {
            x.* = try CodecAttention.init(rig, form, self.mod.?);
            self.made += 1;
            try kernels.append(a, x.kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.items[0..self.made]) |*x| x.deinit();
        if (self.mod) |m| m.unload();
        self.a.destroy(self);
    }
};
