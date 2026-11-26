//! GLM DFlash2 attention under the contract: the whole block kernel and the served 16-way split plus merge.
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
const block = 8;
const window = 2048;
const chunks = 16;
const glm_scale: f32 = 0.08838834764831845;

pub const Form = struct { split: bool, control: bool = false };
pub const forms = [_]Form{
    .{ .split = false },                  .{ .split = true },
    .{ .split = false, .control = true }, .{ .split = true, .control = true },
};

/// rows is streams S (up to 16), n is query heads, k is stream 0's context; all begin 3 slots before the ring wrap.
const served = [_]c.Shape{
    .{ .rows = 8, .n = heads, .k = 257 },  .{ .rows = 1, .n = heads, .k = 0 },
    .{ .rows = 4, .n = heads, .k = 63 },   .{ .rows = 4, .n = heads, .k = 64 },
    .{ .rows = 4, .n = heads, .k = 65 },   .{ .rows = 4, .n = heads, .k = 255 },
    .{ .rows = 4, .n = heads, .k = 256 },  .{ .rows = 1, .n = heads, .k = 2047 },
    .{ .rows = 1, .n = heads, .k = 2048 }, .{ .rows = 16, .n = heads, .k = 300 },
    .{ .rows = 4, .n = heads, .k = 700 },
};

fn depth(s: c.Shape) usize {
    return @min(s.k + block - 1, window) + block + chunks;
}

pub fn spec(f: Form) c.Spec {
    const name = if (f.control)
        (if (f.split) "control GLM DFlash2 split attention leaves its output unwritten" else "control GLM DFlash2 whole attention leaves its output unwritten")
    else
        (if (f.split) "cuda GLM DFlash2 served attention df_attn_part, df_attn_merge" else "cuda GLM DFlash2 whole-block attention df_attn");
    return .{ .name = name, .accum = .f32, .depth = depth, .ops = 3, .out = .bf16, .tiles = .{ .{}, .{}, .{ .size = 256 } }, .served = &served, .sums = .{ .why = attn.why } };
}

pub const Dflash = struct {
    rig: rig_.Rig,
    form: Form,
    whole: cuda.Function,
    part: cuda.Function,
    merge: cuda.Function,
    arena: std.heap.ArenaAllocator,
    r: ?attn.Ref = null,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    kb: []u16 = &.{},
    vb: []u16 = &.{},
    ring: []u16 = &.{},
    dq: u64 = 0,
    dkb: u64 = 0,
    dvb: u64 = 0,
    dring: u64 = 0,
    rings: u64 = 0,
    starts: u64 = 0,
    froms: u64 = 0,
    partial: [3]u64 = @splat(0),
    out: u64 = 0,

    pub fn init(rig: rig_.Rig, form: Form, mod: cuda.Module) !Dflash {
        return .{ .rig = rig, .form = form, .whole = try mod.function("df_attn"), .part = try mod.function("df_attn_part"), .merge = try mod.function("df_attn_merge"), .arena = .init(rig.gpu.gpa) };
    }

    pub fn deinit(self: *Dflash) void {
        if (self.r) |*r| r.deinit();
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Dflash) g.Kernel {
        return .{ .spec = spec(self.form), .ctx = self, .vt = &vt, .control = if (self.form.control) .shapes else null };
    }
    const vt = g.hooks(Dflash);

    fn rows(self: *const Dflash) usize {
        return self.shape.rows * block;
    }
    fn context(self: *const Dflash, si: usize) usize {
        return self.shape.k -| si * self.shape.k / self.shape.rows;
    }
    fn start(self: *const Dflash, si: usize) usize {
        return window - 3 + self.context(si);
    }
    fn low(self: *const Dflash, m: usize) usize {
        const si = m / block;
        return @max(window - 3, self.start(si) + m % block -| (window - 1));
    }
    fn ringTerms(self: *const Dflash, m: usize) usize {
        return self.start(m / block) - self.low(m);
    }

    fn term(self: *Dflash, m: usize, j: usize, kvh: usize, values: bool) []u16 {
        const nr = self.ringTerms(m);
        if (j < nr) {
            const at = ((self.low(m) + j) % window) * width + kvh * hd;
            return self.ring[at + @as(usize, if (values) window * width else 0) ..][0..hd];
        }
        const r = m / block * block + j - nr;
        const at = r * width + kvh * hd;
        return (if (values) self.vb else self.kb)[at..][0..hd];
    }

    pub fn prepare(self: *Dflash, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.free_all);
        if (shape.rows == 0 or shape.rows > 16 or shape.n != heads or shape.k > window) return false;
        self.shape = shape;
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.r) |*r| r.deinit();
        self.r = null;
        self.r = try attn.Ref.init(self.rig.gpu.gpa, hd, shape.k + 2 * block);
        self.q = try a.alloc(u16, self.rows() * heads * hd);
        self.kb = try a.alloc(u16, self.rows() * width);
        self.vb = try a.alloc(u16, self.rows() * width);
        self.ring = try a.alloc(u16, 2 * window * width);
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
        const rp = try a.alloc(u64, shape.rows);
        const st = try a.alloc(i64, shape.rows);
        const fr = try a.alloc(i64, shape.rows);
        for (rp, st, fr, 0..) |*p, *s, *f, si| {
            p.* = self.dring;
            s.* = @intCast(self.start(si));
            f.* = @intCast(self.begin());
        }
        self.rings = try self.rig.input(std.mem.sliceAsBytes(rp));
        self.starts = try self.rig.input(std.mem.sliceAsBytes(st));
        self.froms = try self.rig.input(std.mem.sliceAsBytes(fr));
        if (self.form.split) {
            const slots = self.rows() * heads * chunks;
            for (&self.partial, [_]usize{ 4, 4, 4 * hd }) |*p, w| p.* = try self.rig.scratch(slots * w);
        }
        self.out = try self.rig.output(self.rows() * heads * hd * 2);
        return true;
    }

    fn plantNonfinite(self: *Dflash) void {
        const nan = c.toBf16(std.math.nan(f32));
        const inf = c.toBf16(std.math.inf(f32));
        const ninf = c.toBf16(-std.math.inf(f32));
        self.q[3] = nan;
        self.vb[7] = inf;
        self.vb[hd + 9] = ninf;
        self.vb[width + 11] = inf;
        self.vb[2 * width + 11] = ninf;
        self.kb[2 * width] = nan;
        if (self.context(0) > 0) {
            const last = self.start(0) - 1;
            self.ring[(last % window) * width + hd + 5] = inf;
        }
    }

    /// Gate 4: zero scores everywhere, exact mirrored committed values and zero block values.
    fn stress(self: *Dflash, rng: std.Random) !void {
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
                const at = ((self.begin() + j) % window) * width + kvh * hd + e;
                self.ring[window * width + at] = c.toBf16(col[j]);
            }
        };
    }

    /// Keys 0 and L = ceil(T / 16) tie across chunks; key 2L or the tail raises the maximum with special Q/K/V.
    fn plantAdversarial(self: *Dflash) void {
        const q = self.q[0..hd];
        @memset(q, 0);
        q[0] = c.toBf16(1);
        q[1] = 0x8000;
        q[2] = 0x0001;
        q[3] = 0x8001;
        q[4] = 0x007f;
        q[5] = 0x807f;
        const n = self.ringTerms(0) + block;
        for (0..n) |j| self.term(0, j, 0, false)[0] = 0;
        const l = (n + chunks - 1) / chunks;
        const tie0 = 0;
        const tie1 = @min(l, n - 1);
        const changed = @min(2 * l, n - 1);
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

    pub fn run(self: *Dflash) !void {
        if (!self.form.control) {
            var a: cuda.Args = .{};
            inline for (.{ self.dq, self.dkb, self.dvb, self.rings, self.starts, self.froms, @as(c_int, 0), glm_scale }) |v| a.add(v);
            if (self.form.split) {
                for (self.partial) |p| a.add(p);
                try cuda.launch.launch(self.part, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = kv_heads, .z = chunks }, .block = .{ .x = 256 } }, self.rig.s, &a);
                var m: cuda.Args = .{};
                inline for (.{ self.partial[0], self.partial[1], self.partial[2], self.out }) |v| m.add(v);
                try cuda.launch.launch(self.merge, .{ .grid = .{ .x = @intCast(self.rows()), .y = heads }, .block = .{ .x = hd } }, self.rig.s, &m);
            } else {
                a.add(self.out);
                try cuda.launch.launch(self.whole, .{ .grid = .{ .x = @intCast(self.shape.rows), .y = kv_heads }, .block = .{ .x = 256 } }, self.rig.s, &a);
            }
        }
        try self.rig.s.synchronize();
    }

    pub fn count(self: *const Dflash) usize {
        return self.rows() * heads * hd;
    }

    /// Output head i: context in position order through start - 1, then all eight block keys.
    pub fn load(self: *Dflash, i: usize, h: *attn.Head) !void {
        const m = i / heads;
        const head = i % heads;
        const kvh = head / group;
        for (h.q, 0..) |*x, e| x.* = c.fromBf16(self.q[(m * heads + head) * hd + e]);
        const n = self.ringTerms(m) + block;
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
        return glm_scale;
    }
    pub fn check(self: *Dflash, at: g.Gate) !?[]const u8 {
        return self.r.?.check(self, self.rows() * heads, scale(), try self.rig.outputBytes(), at);
    }
    pub fn reference(self: *Dflash, want: []f64, mag: []f64) !void {
        return self.r.?.reference(self, self.rows() * heads, scale(), want, mag);
    }
    pub fn simulate(self: *Dflash, acc: c.Accum, order: g.Order, out: []f64) !void {
        return self.r.?.simulate(self, self.rows() * heads, scale(), acc, order, out);
    }
};

pub const Group = struct {
    a: std.mem.Allocator,
    mod: cuda.Module,
    items: []Dflash,

    pub fn create(a: std.mem.Allocator, rig: rig_.Rig, kernels: *std.ArrayList(g.Kernel)) !*Group {
        const self = try a.create(Group);
        errdefer a.destroy(self);
        const mod = try cuda.Module.load(rig.gpu.d, cuda.kernels.glm_dflash);
        errdefer mod.unload();
        const items = try a.alloc(Dflash, forms.len);
        errdefer a.free(items);
        self.* = .{ .a = a, .mod = mod, .items = items };
        var initialized: usize = 0;
        errdefer for (self.items[0..initialized]) |*item| item.deinit();
        for (forms) |form| {
            self.items[initialized] = try Dflash.init(rig, form, self.mod);
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
