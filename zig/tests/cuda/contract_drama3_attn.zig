//! Drama 3 direct/split gated attention under the raw-output contract.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const aref = core.act_ref;
const attention_ref = core.attention_ref;

const H = 16;
const D3_D = 256;
const D3_KV = 2;
const D3_PS = 9216;
const D3_SPLITS = 16;
const d3_scale = 0.0625;

/// Both projection layouts and changed decode paths are covered; controls omit the newest key only during launch.
pub const D3Form = enum {
    decode_backbone,
    decode_tower,
    split_backbone,
    split_tower,
    decode_control,
    split_control,

    fn split(f: D3Form) bool {
        return f == .split_backbone or f == .split_tower or f == .split_control;
    }

    fn tower(f: D3Form) bool {
        return f == .decode_tower or f == .split_tower;
    }

    fn control(f: D3Form) bool {
        return f == .decode_control or f == .split_control;
    }
};

/// k sets pos0 = k - rows; decode is one row and split shapes exercise accepted causal rows up to max_split_rows.
const d3_served = [_]c.Shape{
    .{ .rows = 1, .n = H, .k = 511 },
    .{ .rows = 1, .n = H, .k = 1 },
    .{ .rows = 1, .n = H, .k = 9 },
    .{ .rows = 1, .n = H, .k = 128 },
    .{ .rows = 4, .n = H, .k = 132 },
    .{ .rows = 1, .n = H, .k = 4096 },
};

fn d3Depth(s: c.Shape) usize {
    return D3_D + s.k + s.k / 8 + D3_SPLITS + 16;
}

fn d3Spec(f: D3Form) c.Spec {
    const name = switch (f) {
        .decode_backbone => "cuda drama3 d3_attn_decode, backbone gate layout",
        .decode_tower => "cuda drama3 d3_attn_decode, tower gate layout",
        .split_backbone => "cuda drama3 d3_attn_partial + d3_attn_merge, backbone gate layout",
        .split_tower => "cuda drama3 d3_attn_partial + d3_attn_merge, tower gate layout",
        .decode_control => "control: drama3 direct decode omits the newest key",
        .split_control => "control: drama3 split decode omits the newest key",
    };
    return .{
        .name = name,
        .accum = .f32,
        .depth = d3Depth,
        .ops = 12,
        .mids = 2,
        .out = .bf16,
        .served = &d3_served,
        .sums = .{ .why = "bf16 Q/K/V, online softmax partials and their merge reduce in fp32" },
    };
}

fn bf(x: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(x)));
}

pub const D3Attention = struct {
    rig: rig_.Rig,
    form: D3Form,
    decode_fn: cuda.Function,
    partial_fn: cuda.Function,
    merge_fn: cuda.Function,
    ref: ?attn.Ref = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    pos0: usize = 0,
    q: []u16 = &.{},
    keys: []u16 = &.{},
    values: []u16 = &.{},
    proj: []u16 = &.{},
    dev: [5]u64 = @splat(0),
    pml: u64 = 0,
    pacc: u64 = 0,
    out: u64 = 0,
    msg: [240]u8 = undefined,

    pub fn init(rig: rig_.Rig, form: D3Form, mod: cuda.Module) !D3Attention {
        return .{
            .rig = rig,
            .form = form,
            .decode_fn = try mod.function("d3_attn_decode"),
            .partial_fn = try mod.function("d3_attn_partial"),
            .merge_fn = try mod.function("d3_attn_merge"),
        };
    }

    pub fn deinit(self: *D3Attention) void {
        self.clear();
        self.rig.deinit();
    }

    fn clear(self: *D3Attention) void {
        if (self.ref) |*r| r.deinit();
        self.ref = null;
        const a = self.rig.gpu.gpa;
        inline for (.{ "q", "keys", "values", "proj" }) |name| {
            a.free(@field(self, name));
            @field(self, name) = &.{};
        }
    }

    pub fn kernel(self: *D3Attention) g.Kernel {
        return .{ .spec = d3Spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control()) .shapes else null };
    }

    const vt = g.hooks(D3Attention);

    pub fn prepare(self: *D3Attention, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != H or shape.rows == 0 or shape.k < shape.rows) return false;
        if (self.form.split() and shape.rows > 8) return false;
        self.clear();
        self.rig.reset();
        errdefer self.clear();
        self.shape = shape;
        self.pos0 = shape.k - shape.rows;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.ref = try attn.Ref.init(a, D3_D, shape.k);
        self.ref.?.out = .bf16;
        self.q = try a.alloc(u16, shape.rows * H * D3_D);
        self.keys = try a.alloc(u16, shape.k * D3_KV * D3_D);
        self.values = try a.alloc(u16, shape.k * D3_KV * D3_D);
        self.proj = try a.alloc(u16, shape.rows * D3_PS);
        for (self.q) |*x| x.* = attn.normal(rng);
        for (self.keys) |*x| x.* = attn.normal(rng);
        for (self.values) |*x| x.* = attn.normal(rng);
        for (self.proj) |*x| x.* = attn.normal(rng);
        switch (fill) {
            .random => {},
            .nonfinite => self.plantNonfinite(),
            .stress => try self.plantStress(rng),
        }
        if (fill == .random) self.plantAdversarial();
        self.dev[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dev[1] = try self.rig.input(std.mem.sliceAsBytes(self.keys));
        self.dev[2] = try self.rig.input(std.mem.sliceAsBytes(self.values));
        self.dev[3] = try self.rig.input(std.mem.sliceAsBytes(self.proj));
        const launch_pos: i32 = if (self.form.control() and self.pos0 == 0) -1 else @intCast(self.pos0 - @intFromBool(self.form.control()));
        const pos = [1]i32{launch_pos};
        self.dev[4] = try self.rig.input(std.mem.sliceAsBytes(&pos));
        if (self.form.split()) {
            const slots = shape.rows * H * D3_SPLITS;
            self.pml = try self.rig.scratch(slots * 2 * @sizeOf(f32));
            self.pacc = try self.rig.scratch(slots * D3_D * @sizeOf(f32));
        }
        self.out = try self.rig.output(self.count() * @sizeOf(u16));
        return true;
    }

    fn gateIndex(self: *const D3Attention, row: usize, head: usize, e: usize) usize {
        return row * D3_PS + (if (self.form.tower()) D3_D * H + head * D3_D else D3_D + head * 2 * D3_D) + e;
    }

    fn qAt(self: *D3Attention, row: usize, head: usize) []u16 {
        return self.q[(row * H + head) * D3_D ..][0..D3_D];
    }

    fn cacheAt(self: *D3Attention, values: bool, pos: usize, head: usize) []u16 {
        const x = if (values) self.values else self.keys;
        return x[(pos * D3_KV + head) * D3_D ..][0..D3_D];
    }

    /// NaN in one query, cancelling infinities in one value column and a positive infinity in another.
    fn plantNonfinite(self: *D3Attention) void {
        self.qAt(0, 0)[3] = 0x7fc0;
        const second = @min(1, self.shape.k - 1);
        self.cacheAt(true, 0, 0)[7] = 0x7f80;
        self.cacheAt(true, second, 0)[7] = 0xff80;
        self.cacheAt(true, self.shape.k - 1, 1)[9] = 0x7f80;
        self.proj[self.gateIndex(0, H - 1, 11)] = 0x7fc0;
    }

    /// Equal scores make weights one; mirrored values fill the live prefix and later keys zero for exact fp32 sums.
    fn plantStress(self: *D3Attention, rng: std.Random) !void {
        @memset(self.q, 0);
        @memset(self.keys, 0);
        @memset(self.proj, 0);
        const a = self.rig.gpu.gpa;
        const col = try a.alloc(f32, self.shape.k);
        defer a.free(col);
        @memset(col, 0);
        const live = @min(self.shape.k, 2048);
        for (0..D3_KV) |h| for (0..D3_D) |e| {
            attn.mirrored(rng, col[0..live]);
            for (0..self.shape.k) |p| self.cacheAt(true, p, h)[e] = c.toBf16(col[p]);
        };
    }

    /// The last row ties two maxima then two larger maxima in direct and split merges; special BF16 bits reach Q/K/V.
    fn plantAdversarial(self: *D3Attention) void {
        const row = self.shape.rows - 1;
        const q = self.qAt(row, 0);
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        for (0..self.shape.k) |p| @memset(self.cacheAt(false, p, 0), 0);
        const chunk = (self.shape.k + D3_SPLITS - 1) / D3_SPLITS;
        const p0 = 0;
        const p1 = @min(if (self.form.split()) chunk else 1, self.shape.k - 1);
        const p2 = @min(if (self.form.split()) 2 * chunk else 8, self.shape.k - 1);
        const p3 = @min(if (self.form.split()) 3 * chunk else 9, self.shape.k - 1);
        var previous = p0;
        for ([4]usize{ p0, p1, p2, p3 }, 0..) |p, i| {
            if (i > 0 and p == previous) continue;
            previous = p;
            const changed = i >= 2;
            const k = self.cacheAt(false, p, 0);
            k[0] = c.toBf16(if (changed) 32 else 16);
            k[1] = 0x8000;
            k[2] = 0x0001;
            k[3] = 0x8001;
            k[4] = 0x007f;
            k[5] = 0x807f;
            const v = self.cacheAt(true, p, 0);
            v[0] = c.toBf16(1 + 2 * @as(f32, @floatFromInt(i)));
            v[1] = if (i == 0) 0x8000 else 0;
            v[2] = if (i == 0) 0x0001 else 0x8001;
            v[3] = if (i == 0) 0x8001 else 0x007f;
            v[4] = if (i == 0) 0x007f else 0x807f;
        }
        self.proj[self.gateIndex(row, 0, 0)] = 0;
    }

    pub fn run(self: *D3Attention) !void {
        if (self.form.split()) {
            var a: cuda.Args = .{};
            for (self.dev[0..3]) |x| a.add(x);
            a.add(self.dev[4]);
            a.add(self.pml);
            a.add(self.pacc);
            try cuda.launch.launch(self.partial_fn, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = H, .z = D3_SPLITS }, .block = .{ .x = 256 } }, self.rig.s, &a);
            var b: cuda.Args = .{};
            b.add(self.pml);
            b.add(self.pacc);
            b.add(@as(c_int, D3_SPLITS));
            b.add(self.dev[3]);
            b.add(@as(c_int, D3_PS));
            b.add(@as(c_int, if (self.form.tower()) H * D3_D else D3_D));
            b.add(@as(c_int, if (self.form.tower()) D3_D else 2 * D3_D));
            b.add(self.out);
            try cuda.launch.launch(self.merge_fn, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = H, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &b);
        } else {
            var a: cuda.Args = .{};
            for (self.dev[0..3]) |x| a.add(x);
            a.add(self.dev[4]);
            a.add(self.dev[3]);
            a.add(@as(c_int, D3_PS));
            a.add(@as(c_int, if (self.form.tower()) H * D3_D else D3_D));
            a.add(@as(c_int, if (self.form.tower()) D3_D else 2 * D3_D));
            a.add(self.out);
            try cuda.launch.launch(self.decode_fn, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = H, .z = 1 }, .block = .{ .x = 256 } }, self.rig.s, &a);
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *D3Attention) usize {
        return self.shape.rows * H * D3_D;
    }

    fn heads(self: *D3Attention) usize {
        return self.shape.rows * H;
    }

    /// output head i: row i / H, head i % H, causal keys zero through pos0 + row.
    pub fn load(self: *D3Attention, i: usize, h: *attn.Head) !void {
        const row = i / H;
        const head = i % H;
        const kh = head / 8;
        const src = self.q[(row * H + head) * D3_D ..][0..D3_D];
        for (h.q, src) |*dst, x| dst.* = c.fromBf16(x);
        h.n = self.pos0 + row + 1;
        for (0..h.n) |p| for (0..D3_D) |e| {
            h.k[p * D3_D + e] = c.fromBf16(self.keys[(p * D3_KV + kh) * D3_D + e]);
            h.v[p * D3_D + e] = c.fromBf16(self.values[(p * D3_KV + kh) * D3_D + e]);
        };
    }

    fn gate(self: *const D3Attention, output: usize, e: usize) f64 {
        const row = output / H;
        const head = output % H;
        return c.fromBf16(self.proj[self.gateIndex(row, head, e)]);
    }

    fn applyGate(self: *const D3Attention, out: []f64, mag: ?[]f64) void {
        for (out, 0..) |*x, i| {
            const sg = bf(aref.sigmoid(self.gate(i / D3_D, i % D3_D)));
            x.* = bf(bf(x.*) * sg);
            if (mag) |m| m[i] = bf(m[i] * @abs(sg));
        }
    }

    /// The float64 attention endpoints, then the kernel's BF16 attention, BF16 sigmoid and BF16 output.
    pub fn check(self: *D3Attention, at: g.Gate) !?[]const u8 {
        const r = &self.ref.?;
        const out = try self.rig.outputBytes();
        var planted: usize = 0;
        for (0..self.heads()) |i| {
            try self.load(i, &r.head);
            attention_ref.head(r.head.q, r.head.k, r.head.v, r.keys[0..r.head.n], D3_D, d3_scale, r.want, r.tol);
            for (0..D3_D) |e| {
                const sg = bf(aref.sigmoid(self.gate(i, e)));
                const want = bf(bf(r.want[e]) * sg);
                const got = c.fromBf16(std.mem.readInt(u16, out[2 * (i * D3_D + e) ..][0..2], .little));
                if (c.classOf(want) != .finite) planted += 1;
                if (c.classOf(got) != c.classOf(want)) return try std.fmt.bufPrint(&self.msg, "head {d} element {d}: {t} where float64 gives {t}", .{ i, e, c.classOf(got), c.classOf(want) });
                if (at != .nonfinite and std.math.isFinite(want)) {
                    const allow = @abs(sg) * r.tol[e] + aref.tolerance(want);
                    if (!(@abs(got - want) <= allow)) return try std.fmt.bufPrint(&self.msg, "head {d} element {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of its bound", .{ i, e, got, want, @abs(got - want) / allow });
                }
            }
        }
        if (at == .nonfinite and (planted == 0 or planted == self.count())) return try std.fmt.bufPrint(&self.msg, "the planted values make {d} of {d} outputs non-finite: no proof", .{ planted, self.count() });
        return null;
    }

    pub fn reference(self: *D3Attention, want: []f64, mag: []f64) !void {
        try self.ref.?.reference(self, self.heads(), d3_scale, want, mag);
        self.applyGate(want, mag);
    }

    pub fn simulate(self: *D3Attention, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.ref.?.simulate(self, self.heads(), d3_scale, acc, order, out);
        self.applyGate(out, null);
    }
};

const d3_forms = [_]D3Form{ .decode_backbone, .decode_tower, .split_backbone, .split_tower, .decode_control, .split_control };

/// The real CUDA symbols and every registration, at stable addresses owned for the whole gate/raw run.
pub const Group = struct {
    a: std.mem.Allocator,
    mod: ?cuda.Module = null,
    items: [d3_forms.len]D3Attention = undefined,
    made: usize = 0,

    pub fn create(a: std.mem.Allocator, gpu: check.Gpu, s: cuda.Stream, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        self.* = .{ .a = a };
        errdefer self.destroy();
        self.mod = try cuda.Module.load(gpu.d, cuda.kernels.drama3);
        const rig: rig_.Rig = .{ .gpu = gpu, .s = s };
        for (&self.items, d3_forms) |*x, form| {
            x.* = try D3Attention.init(rig, form, self.mod.?);
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
