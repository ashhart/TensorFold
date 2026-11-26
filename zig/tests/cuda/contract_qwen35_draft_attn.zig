//! Qwen3.5 draft attention under the contract: the legacy block kernel and both batched frame kinds.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const attn = @import("contract_attn.zig");

const c = core.contract.spec;
const g = core.contract.gates;

const hd = 128;
const heads = 32;
const kv_heads = 8;
const group = heads / kv_heads;
const width = kv_heads * hd;

const Variant = enum { legacy, dflash2, dspine };

pub const Form = struct { variant: Variant, dspine_window: usize = 0, control: bool = false };
pub const forms = [_]Form{
    .{ .variant = .legacy },                                         .{ .variant = .dflash2 },
    .{ .variant = .dspine, .dspine_window = 16 },                    .{ .variant = .dspine, .dspine_window = 4096 },
    .{ .variant = .dspine, .dspine_window = 8192 },                  .{ .variant = .legacy, .control = true },
    .{ .variant = .dflash2, .control = true },                       .{ .variant = .dspine, .dspine_window = 16, .control = true },
    .{ .variant = .dspine, .dspine_window = 4096, .control = true }, .{ .variant = .dspine, .dspine_window = 8192, .control = true },
};

/// qwen35_draft_frame.cuh's DraftSegment.
const Segment = extern struct { ring: u64, begin: i64, position: i64 };

const legacy_served = [_]c.Shape{
    .{ .rows = 16, .n = heads, .k = 257 },  .{ .rows = 16, .n = heads, .k = 0 },
    .{ .rows = 16, .n = heads, .k = 63 },   .{ .rows = 16, .n = heads, .k = 64 },
    .{ .rows = 16, .n = heads, .k = 65 },   .{ .rows = 16, .n = heads, .k = 255 },
    .{ .rows = 16, .n = heads, .k = 256 },  .{ .rows = 16, .n = heads, .k = 2047 },
    .{ .rows = 16, .n = heads, .k = 2048 },
};
const dflash2_served = [_]c.Shape{
    .{ .rows = 8, .n = heads, .k = 257 },  .{ .rows = 1, .n = heads, .k = 0 },
    .{ .rows = 4, .n = heads, .k = 63 },   .{ .rows = 4, .n = heads, .k = 64 },
    .{ .rows = 4, .n = heads, .k = 65 },   .{ .rows = 4, .n = heads, .k = 255 },
    .{ .rows = 4, .n = heads, .k = 256 },  .{ .rows = 1, .n = heads, .k = 2047 },
    .{ .rows = 1, .n = heads, .k = 2048 }, .{ .rows = 8, .n = heads, .k = 700 },
    .{ .rows = 64, .n = heads, .k = 63 },
};
const dspine16_served = [_]c.Shape{
    .{ .rows = 4, .n = heads, .k = 16 },  .{ .rows = 1, .n = heads, .k = 0 },
    .{ .rows = 1, .n = heads, .k = 1 },   .{ .rows = 4, .n = heads, .k = 15 },
    .{ .rows = 64, .n = heads, .k = 16 },
};
const dspine4096_served = [_]c.Shape{
    .{ .rows = 4, .n = heads, .k = 257 },  .{ .rows = 1, .n = heads, .k = 0 },
    .{ .rows = 4, .n = heads, .k = 63 },   .{ .rows = 4, .n = heads, .k = 64 },
    .{ .rows = 4, .n = heads, .k = 65 },   .{ .rows = 2, .n = heads, .k = 255 },
    .{ .rows = 2, .n = heads, .k = 256 },  .{ .rows = 1, .n = heads, .k = 4095 },
    .{ .rows = 1, .n = heads, .k = 4096 }, .{ .rows = 4, .n = heads, .k = 700 },
    .{ .rows = 64, .n = heads, .k = 63 },
};
const dspine_served = [_]c.Shape{
    .{ .rows = 4, .n = heads, .k = 257 },  .{ .rows = 1, .n = heads, .k = 0 },
    .{ .rows = 4, .n = heads, .k = 63 },   .{ .rows = 4, .n = heads, .k = 64 },
    .{ .rows = 4, .n = heads, .k = 65 },   .{ .rows = 2, .n = heads, .k = 255 },
    .{ .rows = 2, .n = heads, .k = 256 },  .{ .rows = 1, .n = heads, .k = 4096 },
    .{ .rows = 1, .n = heads, .k = 8192 }, .{ .rows = 4, .n = heads, .k = 700 },
    .{ .rows = 64, .n = heads, .k = 63 },
};

const DepthFn = *const fn (c.Shape) usize;

fn legacyDepth(s: c.Shape) usize {
    return @min(s.k + s.rows - 1, 2048) + s.rows + 16;
}
fn dflash2Depth(s: c.Shape) usize {
    return @min(s.k + 15, 2048) + 16 + 9;
}
fn dspine16Depth(s: c.Shape) usize {
    return @min(s.k + 15, 16) + 16 + 1;
}
fn dspine4096Depth(s: c.Shape) usize {
    return @min(s.k + 15, 4096) + 16 + 17;
}
fn dspine8192Depth(s: c.Shape) usize {
    return @min(s.k + 15, 8192) + 16 + 33;
}

pub fn spec(f: Form) c.Spec {
    const name = switch (f.variant) {
        .legacy => if (f.control) "control qwen35 legacy draft attention leaves its output unwritten" else "cuda qwen35 draft block attention tf_q35d_attn",
        .dflash2 => if (f.control) "control qwen35 DFlash2 draft attention leaves its output unwritten" else "cuda qwen35 DFlash2 frame attention tf_q35d_attn_partial, tf_q35d_attn_merge",
        .dspine => switch (f.dspine_window) {
            16 => if (f.control) "control qwen35 DSpine 16-window attention leaves its output unwritten" else "cuda qwen35 DSpine 16-window causal attention tf_q35d_attn_partial, tf_q35d_attn_merge",
            4096 => if (f.control) "control qwen35 DSpine 4096-window attention leaves its output unwritten" else "cuda qwen35 DSpine 4096-window causal attention tf_q35d_attn_partial, tf_q35d_attn_merge",
            else => if (f.control) "control qwen35 DSpine 8192-window attention leaves its output unwritten" else "cuda qwen35 DSpine 8192-window causal attention tf_q35d_attn_partial, tf_q35d_attn_merge",
        },
    };
    const selection = switch (f.variant) {
        .legacy => .{ @as([]const c.Shape, &legacy_served), @as(DepthFn, legacyDepth) },
        .dflash2 => .{ @as([]const c.Shape, &dflash2_served), @as(DepthFn, dflash2Depth) },
        .dspine => switch (f.dspine_window) {
            16 => .{ @as([]const c.Shape, &dspine16_served), @as(DepthFn, dspine16Depth) },
            4096 => .{ @as([]const c.Shape, &dspine4096_served), @as(DepthFn, dspine4096Depth) },
            else => .{ @as([]const c.Shape, &dspine_served), @as(DepthFn, dspine8192Depth) },
        },
    };
    return .{ .name = name, .accum = .f32, .depth = selection[1], .ops = 3, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = selection[0], .sums = .{ .why = attn.why } };
}

pub const Draft = struct {
    rig: rig_.Rig,
    form: Form,
    legacy_fn: cuda.Function,
    partial: cuda.Function,
    merge: cuda.Function,
    arena: std.heap.ArenaAllocator,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    rows: usize = 0,
    ns: usize = 0,
    q: []u16 = &.{},
    kb: []u16 = &.{},
    vb: []u16 = &.{},
    ring: []u16 = &.{},
    dq: u64 = 0,
    dkb: u64 = 0,
    dvb: u64 = 0,
    dring: u64 = 0,
    segments: u64 = 0,
    part: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(rig: rig_.Rig, form: Form, mod: cuda.Module) !Draft {
        return .{ .rig = rig, .form = form, .legacy_fn = try mod.function("tf_q35d_attn"), .partial = try mod.function("tf_q35d_attn_partial"), .merge = try mod.function("tf_q35d_attn_merge"), .arena = .init(rig.gpu.gpa) };
    }

    pub fn deinit(self: *Draft) void {
        if (self.r) |*r| r.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Draft) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }
    const vt = g.hooks(Draft);

    fn window(self: *const Draft) usize {
        return if (self.form.variant == .dspine) (if (self.form.dspine_window == 0) 8192 else self.form.dspine_window) else 2048;
    }
    fn block(self: *const Draft) usize {
        return if (self.form.variant == .legacy) self.shape.rows else 16;
    }
    fn ringRows(self: *const Draft) usize {
        return self.window() + 16;
    }
    fn chunks(self: *const Draft) usize {
        return (self.ringRows() + 255) / 256;
    }
    /// Unequal served positions, all crossing the ring's wrap for contexts of at least three rows.
    fn begin(self: *const Draft) usize {
        return self.ringRows() - 3;
    }
    fn context(self: *const Draft, si: usize) usize {
        return self.shape.k -| si * self.shape.k / self.ns;
    }
    fn position(self: *const Draft, si: usize) usize {
        return self.begin() + self.context(si);
    }

    fn first(self: *const Draft, m: usize) usize {
        const local = if (self.form.variant == .legacy) m else m % 16;
        return @max(self.begin(), self.position(m / self.block()) + local -| (self.window() - 1));
    }
    fn ringTerms(self: *const Draft, m: usize) usize {
        return self.position(m / self.block()) - self.first(m);
    }
    fn blockTerms(self: *const Draft, m: usize) usize {
        return if (self.form.variant == .dspine) m % 16 + 1 else self.block();
    }

    fn term(self: *Draft, m: usize, j: usize, kvh: usize, values: bool) []u16 {
        const nr = self.ringTerms(m);
        if (j < nr) {
            const at = ((self.first(m) + j) % self.ringRows()) * width + kvh * hd;
            return self.ring[at + @as(usize, if (values) self.ringRows() * width else 0) ..][0..hd];
        }
        const si = m / self.block();
        const r = si * self.block() + (if (self.form.variant == .legacy) j - nr else j - nr);
        const at = r * width + kvh * hd;
        return (if (values) self.vb else self.kb)[at..][0..hd];
    }

    pub fn prepare(self: *Draft, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.free_all);
        if (shape.n != heads or shape.rows == 0 or shape.k > self.window()) return false;
        if (self.form.variant == .legacy) {
            if (shape.rows > 16) return false;
            self.ns = 1;
            self.rows = shape.rows;
        } else {
            if (shape.rows > 64) return false; // cuda_draft_frame.max_streams = cuda_engine.max_segs
            self.ns = shape.rows;
            self.rows = shape.rows * 16;
        }
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(self.rig.gpu.gpa, hd, shape.k + 32);
        self.q = try a.alloc(u16, self.rows * heads * hd);
        self.kb = try a.alloc(u16, self.rows * width);
        self.vb = try a.alloc(u16, self.rows * width);
        self.ring = try a.alloc(u16, 2 * self.ringRows() * width);
        for (self.q) |*x| x.* = attn.normal(rng);
        for (self.kb) |*x| x.* = attn.normal(rng);
        for (self.vb) |*x| x.* = attn.normal(rng);
        for (self.ring) |*x| x.* = attn.normal(rng);
        switch (fill) {
            .random => {},
            .nonfinite => self.plantNonfinite(),
            .stress => try self.stress(rng),
        }
        if (fill == .random) self.plantAdversarial();

        self.dq = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.dkb = try self.rig.input(std.mem.sliceAsBytes(self.kb));
        self.dvb = try self.rig.input(std.mem.sliceAsBytes(self.vb));
        self.dring = try self.rig.input(std.mem.sliceAsBytes(self.ring));
        if (self.form.variant != .legacy) {
            const sg = try a.alloc(Segment, self.ns);
            for (sg, 0..) |*x, si| x.* = .{ .ring = self.dring, .begin = @intCast(self.begin()), .position = @intCast(self.position(si)) };
            self.segments = try self.rig.input(std.mem.sliceAsBytes(sg));
            const slots = self.rows * heads * self.chunks();
            for (&self.part, [_]usize{ 4, 4, 4 * hd }) |*p, w| p.* = try self.rig.scratch(slots * w);
        }
        self.out = try self.rig.output(self.rows * heads * hd * 2);
        return true;
    }

    /// The query NaN and both infinities reach every frame kind; a NaN block key reaches causal rows too.
    fn plantNonfinite(self: *Draft) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        self.q[3] = nan;
        self.vb[7] = inf;
        self.vb[hd + 9] = ninf;
        if (self.rows > 1) self.vb[width + 11] = inf;
        if (self.rows > 2) self.vb[2 * width + 11] = ninf;
        if (self.context(0) > 0) {
            const last = self.position(0) - 1;
            const at = (last % self.ringRows()) * width + hd;
            self.ring[at + 5] = inf;
            self.kb[2 * width] = nan;
        }
    }

    /// Gate 4: every score is zero, with exact mirrored committed values and zero block values.
    fn stress(self: *Draft, rng: std.Random) !void {
        const a = self.arena.allocator();
        @memset(self.q, 0);
        @memset(self.kb, 0);
        @memset(self.vb, 0);
        @memset(self.ring, 0);
        const col = try a.alloc(f32, self.shape.k);
        defer a.free(col);
        for (0..kv_heads) |kvh| for (0..hd) |e| {
            attn.mirrored(rng, col);
            for (0..self.shape.k) |j| {
                const at = ((self.begin() + j) % self.ringRows()) * width + kvh * hd + e;
                self.ring[self.ringRows() * width + at] = c.toBf16(col[j]);
            }
        };
    }

    /// Keys 63 and 64 tie across warp/slice paths and key 256 changes the maximum; special values reach Q/K/V.
    fn plantAdversarial(self: *Draft) void {
        const q = self.q[0..hd];
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        const n = self.ringTerms(0) + self.blockTerms(0);
        for (0..n) |j| self.term(0, j, 0, false)[0] = 0;
        const tie0 = if (n > 64) 63 else 0;
        const tie1 = if (n > 64) 64 else @min(1, n - 1);
        const changed = if (n > 256) 256 else @min(2, n - 1);
        for ([2]usize{ tie0, tie1 }, 0..) |j, i| {
            const k = self.term(0, j, 0, false);
            const v = self.term(0, j, 0, true);
            k[0] = c.toBf16(16);
            k[1] = if (i == 0) 0x8000 else 0;
            k[2] = if (i == 0) 0x0001 else 0x8001;
            k[3] = if (i == 0) 0x807f else 0x007f;
            v[0] = c.toBf16(if (i == 0) 1 else 3);
            v[1] = if (i == 0) 0x8000 else 0;
            v[2] = if (i == 0) 0x0001 else 0x8001;
            v[3] = if (i == 0) 0x8001 else 0x007f;
            v[4] = if (i == 0) 0x007f else 0x807f;
        }
        const k = self.term(0, changed, 0, false);
        const v = self.term(0, changed, 0, true);
        k[0] = c.toBf16(32);
        k[1] = 0x8000;
        k[2] = 0x007f;
        k[3] = 0x8001;
        v[0] = c.toBf16(5);
        v[1] = 0x8000;
        v[2] = 0x0001;
        v[3] = 0x807f;
    }

    pub fn run(self: *Draft) !void {
        if (!self.form.control) switch (self.form.variant) {
            .legacy => {
                var a: cuda.Args = .{};
                inline for (.{ self.dq, self.dkb, self.dvb, self.dring, self.dring + self.ringRows() * width * 2, @as(c_int, @intCast(self.rows)), @as(c_int, heads), @as(c_int, kv_heads), @as(c_int, @intCast(self.ringRows())), @as(c_longlong, @intCast(self.begin())), @as(c_longlong, @intCast(self.position(0))), @as(c_int, @intCast(self.window())), self.out }) |v| a.add(v);
                try cuda.launch.launch(self.legacy_fn, .{ .grid = .{ .x = @intCast(self.rows), .y = heads }, .block = .{ .x = 32 } }, self.rig.s, &a);
            },
            .dflash2, .dspine => {
                var a: cuda.Args = .{};
                inline for (.{ self.dq, self.dkb, self.dvb, self.segments, @as(c_int, 16), @as(c_int, heads), @as(c_int, kv_heads), @as(c_int, @intCast(self.ringRows())), @as(c_int, @intCast(self.window())), @as(c_int, 0), @as(c_int, @intCast(self.chunks())), @as(c_int, @intFromBool(self.form.variant == .dspine)), self.part[0], self.part[1], self.part[2] }) |v| a.add(v);
                try cuda.launch.launch(self.partial, .{ .grid = .{ .x = @intCast(self.rows), .y = heads, .z = @intCast(self.chunks()) }, .block = .{ .x = 128 } }, self.rig.s, &a);
                var m: cuda.Args = .{};
                inline for (.{ self.part[0], self.part[1], self.part[2], @as(c_int, heads), @as(c_int, @intCast(self.chunks())), self.out }) |v| m.add(v);
                try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.rows), .y = heads }, .block = .{ .x = 32 } }, self.rig.s, &m);
            },
        };
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Draft) usize {
        return self.rows * heads * hd;
    }

    /// Output head i: its visible wrapping ring keys in position order, then its whole or causal block.
    pub fn load(self: *Draft, i: usize, h: *attn.Head) !void {
        const m = i / heads;
        const head = i % heads;
        const kvh = head / group;
        for (h.q, 0..) |*x, e| x.* = c.fromBf16(self.q[(m * heads + head) * hd + e]);
        const nr = self.ringTerms(m);
        const n = nr + self.blockTerms(m);
        for (0..n) |j| {
            const k = self.term(m, j, kvh, false);
            const v = self.term(m, j, kvh, true);
            for (0..hd) |e| {
                h.k[j * hd + e] = c.fromBf16(k[e]);
                h.v[j * hd + e] = c.fromBf16(v[e]);
            }
        }
        h.n = n;
    }

    fn scale() f64 {
        return 1 / @sqrt(@as(f64, hd));
    }
    pub fn check(self: *Draft, at: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.rows * heads, scale(), try self.rig.outputBytes(), at);
    }
    pub fn reference(self: *Draft, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.rows * heads, scale(), want, mag);
    }
    pub fn simulate(self: *Draft, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.rows * heads, scale(), acc, order, out);
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    items: []Draft,

    pub fn create(a: std.mem.Allocator, rig: rig_.Rig, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        const mod = try cuda.Module.load(rig.gpu.d, cuda.kernels.qwen35_draft);
        errdefer mod.unload();
        const items = try a.alloc(Draft, forms.len);
        errdefer a.free(items);
        self.* = .{ .a = a, .mod = mod, .items = items };
        var initialized: usize = 0;
        errdefer for (self.items[0..initialized]) |*item| item.deinit();
        for (forms) |form| {
            self.items[initialized] = try Draft.init(rig, form, self.mod);
            initialized += 1;
            try kernels.append(a, self.items[initialized - 1].kernel());
        }
        return self;
    }

    pub fn destroy(self: *Group) void {
        for (self.items) |*item| item.deinit();
        self.a.free(self.items);
        self.mod.unload();
        self.a.destroy(self);
    }
};
