//! experts3's split decode pair (split_kernel<2, 2, 2, 4>) under the contract on CUDA: bit-equal to the pair kernel.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Gpu = @import("check.zig").Gpu;
const rig_ = @import("contract_rig.zig");
const e3 = @import("experts3_tests.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const A3 = cuda.experts3.Affine3Experts;
const A4 = cuda.experts.Affine4Experts;

/// The split form itself, then two planted faults that gate 3 must catch.
pub const Fault = enum { none, drop_tail, reorder };

const routed = 12; // 3-bit experts; the 4-bit shared one is expert 12
const picks = 8; // distinct routed picks a token row, as the 35B-A3B routes
const slots = picks + 1; // the shared expert in the last slot
const warps = 4; // split_kernel's warps: a round of K groups is a group a warp
const tile = 16; // pairs a decode plan item
const rule = "as the pair kernel it replaces gives them: NaN and infinite token inputs";

/// A group's 64 products, its input sum, two fmas a group.
fn depth(s: c.Shape) usize {
    return 64 + 64 + 2 * (s.k / 64);
}

const tiles = [3]c.Tile{ .{ .size = tile }, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = 512, .step = 64, .min = 64 } };
/// rows = token rows: the 35B-A3B's 512-wide pair on 2048 dims at 1 to 4 rows, a full item, and five K groups.
const served = [_]c.Shape{
    .{ .rows = 4, .n = 512, .k = 2048 }, .{ .rows = 1, .n = 512, .k = 2048 },  .{ .rows = 2, .n = 512, .k = 2048 },
    .{ .rows = 3, .n = 512, .k = 2048 }, .{ .rows = 16, .n = 512, .k = 2048 }, .{ .rows = 4, .n = 512, .k = 320 },
};

pub fn spec(fault: Fault) c.Spec {
    const name = switch (fault) {
        .none => "cuda experts3 decode split pair (SwiGLU, a column block a CTA)",
        .drop_tail => "control: experts3 split pair drops a last round short of the warps",
        .reorder => "control: experts3 split pair chains the K groups in reverse",
    };
    // exp's 2 ulps, 1 + e, the quotient and the product; silu rounded to bf16 before the product
    return .{ .name = name, .accum = .f32, .depth = depth, .ops = 5, .mids = 1, .out = .bf16, .replaces = "experts3's decode pair kernel", .nan = .{ .stated = rule }, .tiles = tiles, .served = &served };
}

/// The modules experts3_tests loads and the kernels resolved from them.
pub const Loaded = struct {
    mods: [5]cuda.Module,
    k: e3.Kernels,

    pub fn init(gpu: Gpu) !Loaded {
        var l: Loaded = undefined;
        const images = [_][]const u8{ cuda.kernels.experts, cuda.kernels.experts_prefill, cuda.kernels.experts_pack, cuda.kernels.experts3, cuda.kernels.experts3_prefill };
        for (images, &l.mods, 0..) |img, *m, i| {
            errdefer for (l.mods[0..i]) |*o| o.unload();
            m.* = try cuda.Module.load(gpu.d, img);
        }
        errdefer for (&l.mods) |*m| m.unload();
        const sms: usize = @intCast(try gpu.ctx.attribute(.multiprocessor_count));
        l.k = .{ .a3 = try A3.resolve(l.mods[3], l.mods[4], sms), .a4 = try A4.resolve(l.mods[0], l.mods[1], l.mods[2], sms), .router = try cuda.grouped.Router.resolve(l.mods[0]) };
        return l;
    }

    pub fn deinit(l: *Loaded) void {
        for (&l.mods) |*m| m.unload();
    }
};

/// Each run of `kg` chunks of `chunk` bytes in reverse: the K groups of a token row or of a column block's table.
fn reverseGroups(bytes: []u8, chunk: usize, kg: usize) void {
    var at: usize = 0;
    while (at < bytes.len) : (at += chunk * kg) {
        for (0..kg / 2) |i| for (bytes[at + i * chunk ..][0..chunk], bytes[at + (kg - 1 - i) * chunk ..][0..chunk]) |*p, *q| std.mem.swap(u8, p, q);
    }
}

fn swiglu(gv: f64, uv: f64) f64 {
    return gv / (1 + @exp(-gv)) * uv;
}

fn bf(v: f64) f64 {
    return c.fromBf16(c.toBf16(@floatCast(v)));
}

/// shape.rows token rows, `slots` pairs each; the gate and up tables of the routed experts and the shared one.
pub const Split = struct {
    rig: rig_.Rig,
    k: *const e3.Kernels,
    fault: Fault = .none,
    shape: c.Shape = undefined,
    aff: std.ArrayList(Affine) = .empty, // gate then up, each routed experts then the shared one, on one set of rows
    ids: []i32 = &.{},
    x: [2]u64 = undefined, // the token rows: the pair kernel's, then the run's (a control's own)
    t3: [2]u64 = undefined, // the 3-bit pair tables, likewise
    t4: [2]u64 = undefined, // the shared expert's 4-bit pair table, likewise
    plan: cuda.grouped.Plan = undefined,
    items: usize = 0,
    out: u64 = 0,
    base: u64 = 0,
    msg: [160]u8 = undefined,

    pub fn deinit(self: *Split) void {
        self.drop();
        self.aff.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    fn drop(self: *Split) void {
        for (self.aff.items) |f| f.deinit();
        self.aff.clearRetainingCapacity();
        self.rig.gpu.gpa.free(self.ids);
        self.ids = &.{};
        self.rig.reset();
    }

    pub fn kernel(self: *Split) g.Kernel {
        return .{ .spec = spec(self.fault), .ctx = self, .vt = &vt, .control = if (self.fault == .none) null else .shapes };
    }

    const vt = g.hooks(Split);

    fn pairs(self: *const Split) usize {
        return self.shape.rows * slots;
    }

    /// Matrix m (0 gate, 1 up) of pair p's expert.
    fn matrix(self: *const Split, m: usize, p: usize) Affine {
        return self.aff.items[m * (routed + 1) + @as(usize, @intCast(self.ids[p]))];
    }

    /// The affine family's weights and rows (random, non-finite rows, or gate 4's integers), routed and packed.
    pub fn prepare(self: *Split, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        self.drop();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        for (0..2 * (routed + 1)) |i| {
            const f = try Affine.init(a, shape, if (i % (routed + 1) == routed) 4 else 3, 64);
            self.aff.append(a, f) catch |err| {
                f.deinit();
                return err;
            };
            try f.fill(fill, seed + i);
            if (i > 0) @memcpy(f.x, self.aff.items[0].x);
        }
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.ids = try a.alloc(i32, self.pairs());
        for (0..shape.rows) |r| {
            var taken: [routed]bool = @splat(false);
            for (0..picks) |i| {
                var e = rng.uintLessThan(usize, routed);
                while (taken[e]) e = rng.uintLessThan(usize, routed);
                taken[e] = true;
                self.ids[r * slots + i] = @intCast(e);
            }
            self.ids[r * slots + picks] = routed;
        }
        const t3 = try self.table(3, routed, A3.pairBytes(routed, shape.n, shape.k));
        defer a.free(t3);
        const t4 = try self.table(4, 1, 2 * A4.bytes(1, shape.n, shape.k));
        defer a.free(t4);
        const x = try a.dupe(u8, std.mem.sliceAsBytes(self.aff.items[0].x));
        defer a.free(x);
        self.x[0] = try self.rig.input(x);
        self.t3[0] = try self.rig.input(t3);
        self.t4[0] = try self.rig.input(t4);
        const kg = shape.k / 64;
        switch (self.fault) {
            .none => {},
            // zeroed token inputs make a group's terms exactly 0, as a kernel that skips a short last round would sum
            .drop_tail => for (0..shape.rows) |r| @memset(x[(r * kg + kg - kg % warps) * 128 ..][0 .. kg % warps * 128], 0),
            // the same terms, each chain over the groups in reverse
            .reorder => {
                reverseGroups(x, 128, kg);
                reverseGroups(t3, 2 * A3.bytes(1, 32, 64), kg);
                reverseGroups(t4, 2 * A4.bytes(1, 32, 64), kg);
            },
        }
        self.x[1] = if (self.fault == .none) self.x[0] else try self.rig.input(x);
        self.t3[1] = if (self.fault == .reorder) try self.rig.input(t3) else self.t3[0];
        self.t4[1] = if (self.fault == .reorder) try self.rig.input(t4) else self.t4[0];
        const ids = try self.rig.input(std.mem.sliceAsBytes(self.ids));
        self.items = cuda.grouped.maxItems(self.pairs(), routed + 1, tile);
        const zeros = try a.alloc(u8, @max(self.pairs() * 4, self.items * 12));
        defer a.free(zeros);
        @memset(zeros, 0);
        const blocks = (self.pairs() + 1023) / 1024;
        self.plan = .{ .members = try self.rig.input(zeros[0 .. self.pairs() * 4]), .items = try self.rig.input(zeros[0 .. self.items * 12]), .counts = try self.rig.input(zeros[0..8]), .rank = try self.rig.scratch(self.pairs() * 4), .hist = try self.rig.scratch(blocks * (routed + 1) * 4) };
        try self.k.router.route(self.rig.s, ids, self.pairs(), routed + 1, tile, self.plan);
        try self.rig.s.synchronize();
        self.out = try self.rig.output(self.count() * 2);
        self.base = try self.rig.scratch(self.count() * 2);
        return true;
    }

    /// Gate and up tables (interleaved a group) of `experts` experts at `bits`, packed as the engine packs them.
    fn table(self: *Split, bits: u8, experts: usize, len: usize) ![]u8 {
        const a = self.rig.gpu.gpa;
        const sh = self.shape;
        const first: usize = if (bits == 4) routed else 0;
        var dev = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, len);
        defer dev.free();
        for (0..2) |m| {
            const q = try a.alloc(u8, experts * sh.n * sh.k);
            defer a.free(q);
            const s = try a.alloc(u16, experts * sh.n * sh.k / 64);
            defer a.free(s);
            const b = try a.alloc(u16, s.len);
            defer a.free(b);
            for (self.aff.items[m * (routed + 1) + first ..][0..experts], 0..) |f, e| {
                @memcpy(q[e * f.codes.len ..][0..f.codes.len], f.codes);
                @memcpy(s[e * f.scales.len ..][0..f.scales.len], f.scales);
                @memcpy(b[e * f.biases.len ..][0..f.biases.len], f.biases);
            }
            const codes: e3.Codes = .{ .q = q, .scales = s, .biases = b, .count = experts, .n = sh.n, .k = sh.k };
            try e3.packTable(self.rig.gpu, self.k.*, self.rig.s, codes, bits, dev.ptr, 2, m, 0);
        }
        const out = try a.alloc(u8, len);
        errdefer a.free(out);
        try dev.download(0, out);
        return out;
    }

    /// Affine3Experts.run's pair launch: the split form, or (`split` false) the pair kernel it replaces.
    fn launch(self: *Split, split: bool, at: usize, out: u64) !void {
        var a3 = self.k.a3;
        a3.split_below = if (split) std.math.maxInt(usize) else 0;
        const sh = self.shape;
        const l: cuda.experts.Layer = .{ .format = .affine3, .up = self.t3[at], .down = 0, .width = sh.n, .dims = sh.k, .experts = routed + 1, .swiglu = true, .shared = routed, .shared_up = self.t4[at] };
        try a3.run(self.rig.s, true, .{ .x = self.x[at], .stride = sh.k, .slots = slots }, l, self.plan, out, self.items);
    }

    pub fn run(self: *Split) !void {
        try self.launch(true, 1, self.out);
        try self.rig.s.synchronize();
    }

    pub fn replaced(self: *Split) ![]const u8 {
        try self.launch(false, 0, self.base);
        return self.rig.bytes(self.base, self.count() * 2);
    }

    /// Gate 1: the pair kernel's bytes, and some output non-finite: the planted values.
    pub fn check(self: *Split, _: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        const want = try self.replaced();
        if (std.mem.indexOfDiff(u8, got, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} differs from the pair kernel's", .{ at, got.len });
        for (0..got.len / 2) |i| if (c.classOf(c.read(.bf16, got, i)) != .finite) return null;
        return try std.fmt.bufPrint(&self.msg, "no output is non-finite: the planted values prove nothing", .{});
    }

    pub fn count(self: *Split) usize {
        return self.pairs() * self.shape.n;
    }

    /// SwiGLU on the bf16-rounded float64 sums; mag 0: as a replacement, gate 3 is bit-equality and gate 4 exact.
    pub fn reference(self: *Split, want: []f64, mag: []f64) !void {
        const n = self.shape.n;
        @memset(mag, 0);
        for (0..self.pairs()) |p| for (0..n) |j| {
            want[p * n + j] = swiglu(bf(self.matrix(0, p).one(p / slots, j)[0]), bf(self.matrix(1, p).one(p / slots, j)[0]));
        };
    }

    /// Each matrix's sums as an accumulator leaves them (the affine family's structures), a pair's from its expert.
    pub fn simulate(self: *Split, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.rig.gpu.gpa;
        const n = self.shape.n;
        const per = self.shape.rows * n;
        const all = try a.alloc(f64, self.aff.items.len * per);
        defer a.free(all);
        for (self.aff.items, 0..) |f, i| try f.simulate(acc, order, all[i * per ..][0..per]);
        for (0..self.pairs()) |p| for (0..n) |j| {
            const e: usize = @intCast(self.ids[p]);
            const at = p / slots * n + j;
            out[p * n + j] = swiglu(bf(all[e * per + at]), bf(all[(routed + 1 + e) * per + at]));
        };
    }
};
