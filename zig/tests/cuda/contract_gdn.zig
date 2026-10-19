//! gdn_prep.cu (#797) under the contract on CUDA: a row's conv window, SiLU, the q/k norms, v, g and beta.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const ref = core.gdn_prep_ref;
const gp = cuda.gdn_prep;

/// Which outputs an entry reads (the others are working memory): q and k, v, or g and beta.
pub const Part = enum { qk, v, gb };

/// The conv's taps (one fma chain).
fn taps(s: c.Shape) usize {
    return s.k;
}

fn noSums(_: c.Shape) usize {
    return 0;
}

const qk_why = "these stress inputs do not separate bf16 conv/square sums beyond the normed output bound; fp16 overflows";
const v_why = "these stress inputs do not separate bf16 tap sums beyond the rounded output bound; fp16 overflows";
const gb_rule = "own check, the PR's float64 bound: g = exp(-y) carries y times y's relative error, beyond a fixed op count";

/// Shape: rows x key heads (n; 3 v heads a key head) x conv taps (k).
pub fn spec(dk: usize, part: Part) c.Spec {
    const served: []const c.Shape = switch (dk) {
        128 => &.{ .{ .rows = 1, .n = 16, .k = 4 }, .{ .rows = 64, .n = 16, .k = 4 } }, // Qwen3.8-27B: 16 key heads, 4 taps
        64 => &.{ .{ .rows = 1, .n = 2, .k = 4 }, .{ .rows = 40, .n = 2, .k = 4 } },
        else => &.{ .{ .rows = 1, .n = 1, .k = 2 }, .{ .rows = 40, .n = 1, .k = 2 } },
    };
    const tiles: [3]c.Tile = .{ .{ .size = 16 }, .{ .size = 4 }, .{ .size = 8, .min = 2 } }; // small blocks to 16 rows; 4 heads a block
    const name: []const u8 = switch (part) {
        inline else => |p| switch (dk) {
            64 => "cuda gdn_prep tf_gdn_prep_64 " ++ @tagName(p),
            128 => "cuda gdn_prep tf_gdn_prep_128 " ++ @tagName(p),
            else => "cuda gdn_prep tf_gdn_prep_256 " ++ @tagName(p),
        },
    };
    return switch (part) {
        // silu 8, the norm's sum and rsqrt 12, the scale and products 4; c rounded to bf16, and the norm through it
        .qk => .{ .name = name, .accum = .f32, .depth = taps, .ops = 24, .mids = 2, .out = .bf16, .tiles = tiles, .served = served, .sums = .{ .bf16 = 0, .why = qk_why } },
        .v => .{ .name = name, .accum = .f32, .depth = taps, .ops = 8, .out = .bf16, .tiles = tiles, .served = served, .sums = .{ .bf16 = 0, .why = v_why } },
        .gb => .{ .name = name, .accum = .none, .depth = noSums, .out = .f32, .nan = .{ .stated = gb_rule }, .tiles = tiles, .served = served },
    };
}

const vh_per = 3;

fn bf(h: u16) f64 {
    return c.fromBf16(h);
}

pub const Gdn = struct {
    rig: rig_.Rig,
    dk: usize,
    part: Part,
    k: *const gp.GdnPrep,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    sh: gp.Shape = .{ .kh = 0, .vh = 0, .dk = 0, .keep = 0 },
    qkv: []u16 = &.{},
    state: []u16 = &.{},
    conv: []u16 = &.{},
    av: []u16 = &.{},
    bv: []u16 = &.{},
    a_log: []f32 = &.{},
    dtb: []f32 = &.{},
    sid: []i32 = &.{},
    win: []i32 = &.{},
    io: gp.Io = undefined,
    msg: [200]u8 = undefined,

    pub fn deinit(self: *Gdn) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Gdn) g.Kernel {
        return .{ .spec = spec(self.dk, self.part), .ctx = self, .vt = if (self.part == .gb) &vt_gb else &vt };
    }

    const vt = g.hooks(Gdn);
    const vt_gb = blk: {
        var v = g.hooks(Gdn);
        v.check = checkGb;
        break :blk v;
    };

    fn chans(self: *const Gdn) usize {
        return (2 * self.sh.kh + self.sh.vh) * self.dk;
    }

    fn ld(self: *const Gdn) usize {
        return self.chans() + 64;
    }

    pub fn prepare(self: *Gdn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        if (shape.rows == 0 or shape.n == 0 or shape.k < 2 or shape.k > 8) return false;
        self.shape = shape;
        self.sh = .{ .kh = shape.n, .vh = vh_per * shape.n, .dk = self.dk, .keep = shape.k - 1 };
        const a = self.arena.allocator();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const r = shape.rows;
        const cc = self.chans();
        const keep = self.sh.keep;
        const vh = self.sh.vh;
        self.qkv = try a.alloc(u16, r * self.ld());
        self.state = try a.alloc(u16, 2 * keep * cc);
        self.conv = try a.alloc(u16, cc * shape.k);
        self.av = try a.alloc(u16, r * (vh + 3));
        self.bv = try a.alloc(u16, r * (vh + 3));
        for ([_][]u16{ self.qkv, self.state, self.conv, self.av, self.bv }, [_]f32{ 1, 1, 0.4, 2, 2 }) |v, sc| for (v) |*x| {
            x.* = c.toBf16(rng.floatNorm(f32) * sc);
        };
        self.a_log = try a.alloc(f32, vh);
        self.dtb = try a.alloc(f32, vh);
        for (self.a_log, self.dtb) |*x, *y| {
            x.* = rng.float(f32) * 2 - 1;
            y.* = rng.floatNorm(f32);
        }
        self.sid = try a.alloc(i32, r);
        self.win = try a.alloc(i32, r * shape.k);
        for (self.sid, 0..) |*x, i| x.* = @intCast(i % 2);
        for (self.win, 0..) |*w, i| { // kept rows or any qkv row (a tree's window is any rows), the row itself last
            w.* = if (rng.uintLessThan(u32, 3) == 0) @intCast(rng.uintLessThan(usize, keep)) else @intCast(keep + rng.uintLessThan(usize, r));
            if (i % shape.k == shape.k - 1) w.* = @intCast(keep + i / shape.k);
        }
        if (fill == .nonfinite) {
            const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
            const at = [_][2]usize{ .{ 0, cc / 3 }, .{ 1, cc / 2 }, .{ 2, cc - 1 }, .{ 3, 0 }, .{ 3, cc - 1 } };
            for (at, put) |p, v| if (p[0] < r) {
                self.qkv[p[0] * self.ld() + p[1]] = c.toBf16(v);
            };
            for ([_][]u16{ self.av, self.bv }, 0..) |v, i| if (i < r) {
                v[i * (vh + 3) + i % vh] = c.toBf16(put[i]);
            };
        }
        if (fill == .stress) {
            // every input 256, taps 255 but the last one 250 to 254: big integer sums, fp16 overflows past 65504
            @memset(self.qkv, c.toBf16(256));
            @memset(self.state, c.toBf16(256));
            for (0..cc) |i| for (0..shape.k) |t| {
                self.conv[i * shape.k + t] = c.toBf16(if (t + 1 < shape.k) 255 else @floatFromInt(250 + i % 5));
            };
        }
        self.io = .{ .qkv = try self.rig.input(std.mem.sliceAsBytes(self.qkv)), .qkv_ld = @intCast(self.ld()), .state = try self.rig.input(std.mem.sliceAsBytes(self.state)), .conv = try self.rig.input(std.mem.sliceAsBytes(self.conv)), .win = try self.rig.input(std.mem.sliceAsBytes(self.win)), .sid = try self.rig.input(std.mem.sliceAsBytes(self.sid)), .a = try self.rig.input(std.mem.sliceAsBytes(self.av)), .a_ld = @intCast(vh + 3), .b = try self.rig.input(std.mem.sliceAsBytes(self.bv)), .b_ld = @intCast(vh + 3), .a_log = try self.rig.input(std.mem.sliceAsBytes(self.a_log)), .dt_bias = try self.rig.input(std.mem.sliceAsBytes(self.dtb)), .q = 0, .k = 0, .v = 0, .g = 0, .beta = 0 };
        const qk_bytes = r * self.sh.kh * self.dk * 2;
        const kinds = [5]bool{ self.part == .qk, self.part == .qk, self.part == .v, self.part == .gb, self.part == .gb };
        const sizes = [5]usize{ qk_bytes, qk_bytes, r * vh * self.dk * 2, r * vh * 4, r * vh * 4 };
        var ptrs: [5]u64 = undefined;
        for (kinds, sizes, &ptrs) |out, n, *p| p.* = if (out) try self.rig.output(n) else try self.rig.scratch(n);
        self.io.q = ptrs[0];
        self.io.k = ptrs[1];
        self.io.v = ptrs[2];
        self.io.g = ptrs[3];
        self.io.beta = ptrs[4];
        return true;
    }

    pub fn run(self: *Gdn) !void {
        try self.k.run(self.rig.s, self.io, self.shape.rows, self.sh);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Gdn) usize {
        const r = self.shape.rows;
        return switch (self.part) {
            .qk => 2 * r * self.sh.kh * self.dk,
            .v => r * self.sh.vh * self.dk,
            .gb => 2 * r * self.sh.vh,
        };
    }

    /// Output o's row, head (q heads, k heads, then v heads) and channel.
    fn locate(self: *const Gdn, o: usize) [3]usize {
        const r = self.shape.rows;
        const kh = self.sh.kh;
        const dk = self.dk;
        if (self.part == .v) return .{ o / (self.sh.vh * dk), 2 * kh + o / dk % self.sh.vh, o % dk };
        const k_half = o >= r * kh * dk;
        const i = if (k_half) o - r * kh * dk else o;
        return .{ i / (kh * dk), (if (k_half) kh else 0) + i / dk % kh, i % dk };
    }

    /// Head hd of row r: each channel's tap terms into t [dk][taps].
    fn terms(self: *const Gdn, r: usize, hd: usize, t: []f64) void {
        const n = self.shape.k;
        const keep = self.sh.keep;
        for (0..n) |tp| {
            const src: usize = @intCast(self.win[r * n + tp]);
            for (0..self.dk) |i| {
                const ch = hd * self.dk + i;
                const x = if (src < keep) bf(self.state[(@as(usize, @intCast(self.sid[r])) * keep + src) * self.chans() + ch]) else bf(self.qkv[(src - keep) * self.ld() + ch]);
                t[i * n + tp] = x * bf(self.conv[ch * n + tp]);
            }
        }
    }

    /// Head hd of row r's outputs and magnitudes (dk each), its conv sums in `acc` and `order` (f32 none: float64).
    fn head(self: *Gdn, r: usize, hd: usize, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const a = self.arena.allocator();
        const n = self.shape.k;
        const t = try a.alloc(f64, self.dk * n);
        const cv = try a.alloc(f64, self.dk);
        self.terms(r, hd, t);
        for (cv, mag, 0..) |*o, *m, i| {
            const row = t[i * n ..][0..n];
            var s: f64 = 0;
            m.* = 0;
            for (row) |v| {
                s += v;
                m.* += 1.1 * @abs(v); // silu' is at most 1.1
            }
            if (acc) |ac| s = rows.sumIn(ac, order, row);
            o.* = s / (1 + @exp(-s));
        }
        if (hd >= 2 * self.sh.kh) {
            @memcpy(want, cv);
            return;
        }
        const is_q = hd < self.sh.kh;
        const dkf: f64 = @floatFromInt(self.dk);
        var f = ref.norm(cv, is_q, self.sh.eps, want);
        if (acc) |ac| {
            const sq = try a.alloc(f64, self.dk);
            for (sq, cv) |*q, v| q.* = v * v;
            f = 1 / @sqrt(rows.sumIn(ac, order, sq) / dkf + self.sh.eps) * (if (is_q) 1 / dkf else 1 / @sqrt(dkf));
            for (want, cv) |*o, v| o.* = v * f;
        }
        for (mag) |*m| m.* *= @abs(f);
    }

    fn outputs(self: *Gdn, acc: ?c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        const a = self.arena.allocator();
        const hw = try a.alloc(f64, self.dk);
        const hm = try a.alloc(f64, self.dk);
        var o: usize = 0;
        while (o < want.len) : (o += self.dk) {
            const at = self.locate(o);
            try self.head(at[0], at[1], acc, order, hw, hm);
            @memcpy(want[o..][0..self.dk], hw);
            @memcpy(mag[o..][0..self.dk], hm);
        }
    }

    pub fn reference(self: *Gdn, want: []f64, mag: []f64) !void {
        if (self.part == .gb) {
            for (want, mag, 0..) |*w, *m, i| {
                w.* = self.gbWant(i);
                m.* = @abs(w.*);
            }
            return;
        }
        return self.outputs(null, .sequential, want, mag);
    }

    pub fn simulate(self: *Gdn, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.arena.allocator().alloc(f64, out.len);
        return self.outputs(acc, order, out, mag);
    }

    /// g (first rows x vh) then beta, in float64.
    fn gbWant(self: *const Gdn, i: usize) f64 {
        const vh = self.sh.vh;
        const half = self.shape.rows * vh;
        const j = i % half;
        const r = j / vh;
        const hv = j % vh;
        if (i < half) return ref.gate(bf(self.av[r * (vh + 3) + hv]), self.dtb[hv], self.a_log[hv]);
        return 1 / (1 + @exp(-bf(self.bv[r * (vh + 3) + hv])));
    }

    /// gdn_prep_tests' bounds: g within 2^-20 (4 + 4 exp(a_log) softplus) of itself, beta within 2^-19.
    fn checkGb(ctx: *anyopaque, at: g.Gate) anyerror!?[]const u8 {
        const self: *Gdn = @ptrCast(@alignCast(ctx));
        const out = try self.rig.outputBytes();
        const vh = self.sh.vh;
        const half = self.shape.rows * vh;
        var planted: usize = 0;
        const n = self.count();
        for (0..n) |i| {
            const want = self.gbWant(i);
            const got = c.read(.f32, out, i);
            if (c.classOf(want) != .finite) planted += 1;
            if (c.classOf(got) != c.classOf(want)) return try std.fmt.bufPrint(&self.msg, "output {d} is {t} where float64 gives {t}", .{ i, c.classOf(got), c.classOf(want) });
            if (at == .nonfinite or c.classOf(want) != .finite) continue;
            const j = i % half;
            const x = bf(self.av[j / vh * (vh + 3) + j % vh]) + self.dtb[j % vh];
            const sp = if (x > 20) x else std.math.log1p(@exp(x));
            const tol = if (i < half) @abs(want) * std.math.pow(f64, 2, -20) * (4 + 4 * @exp(@as(f64, self.a_log[j % vh])) * sp) + 1e-30 else want * std.math.pow(f64, 2, -19) + 1e-30;
            if (!(@abs(got - want) <= tol)) return try std.fmt.bufPrint(&self.msg, "output {d} is {e:.6} where float64 gives {e:.6}: {d:.2} of its bound", .{ i, got, want, @abs(got - want) / tol });
        }
        if (at == .nonfinite and (planted == 0 or planted == n)) return try std.fmt.bufPrint(&self.msg, "the planted values make {d} of {d} outputs non-finite: no proof", .{ planted, n });
        return null;
    }
};
