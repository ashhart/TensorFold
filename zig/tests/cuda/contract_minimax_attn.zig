//! MiniMax-M3.1-Flash's CUDA attention under the contract: kept-block, every-block and draft-window forms.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const mm3 = @import("minimax_m3_cuda");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const sumIn = contract.rows.sumIn;
const D = 128;
const hk = 2; // a TP=2 rank's key heads, 16 query heads each
const hq = 16 * hk;
const topk = 16; // kept blocks a (row, key head)
const scale: f32 = 1.0 / @sqrt(@as(f32, D)); // as the forward passes it
const soft_why = "online softmax over 64-key chunks with bf16 probabilities into P.V: no input makes its sums exact in fp32";

pub const Form = enum {
    /// tf_mm3_attn_prompt_tc over each row's 16 kept blocks (bitmaps): prompt chunks and decode windows alike
    kept,
    /// tf_mm3_attn_prompt_tc over every block (words 0: a row that sees 16 blocks or fewer)
    every,
    /// tf_mm3_attn_prompt_tc's fixed window: every row over keys [lo, hi], as a DSpark block attends
    window,
};

/// A key's 128 products, then the row's keys, the chunks' rescales and the merge.
fn attnDepth(s: c.Shape) usize {
    return D + s.k + s.k / 64 + 16;
}

/// rows x a row's outputs (32 heads x 128) x the keys the last row sees (window: the window's keys).
pub fn spec(comptime f: Form) c.Spec {
    const served: []const c.Shape = switch (f) {
        .kept => &.{ .{ .rows = 64, .n = hq * D, .k = 2112 }, .{ .rows = 29, .n = hq * D, .k = 3000 }, .{ .rows = 1, .n = hq * D, .k = 3100 }, .{ .rows = 4, .n = hq * D, .k = 2300 }, .{ .rows = 16, .n = hq * D, .k = 2300 } },
        .every => &.{ .{ .rows = 64, .n = hq * D, .k = 1000 }, .{ .rows = 13, .n = hq * D, .k = 300 }, .{ .rows = 1, .n = hq * D, .k = 900 }, .{ .rows = 5, .n = hq * D, .k = 2048 } },
        .window => &.{ .{ .rows = 16, .n = hq * D, .k = 4096 }, .{ .rows = 16, .n = hq * D, .k = 700 }, .{ .rows = 16, .n = hq * D, .k = 1500 } },
    };
    const name = switch (f) {
        .kept => "cuda minimax tf_mm3_attn_prompt_tc over kept blocks",
        .every => "cuda minimax tf_mm3_attn_prompt_tc over every block",
        .window => "cuda minimax tf_mm3_attn_prompt_tc over a draft block's window",
    };
    return .{ .name = name, .accum = .f32, .depth = attnDepth, .ops = 8, .out = .bf16, .served = served, .sums = .{ .f16 = 0, .bf16 = 0, .exact = false, .why = soft_why } };
}

/// One attention call's queries, key and value cache rows and each (row, key head)'s kept blocks.
pub const Attn = struct {
    rig: rig_.Rig,
    k: *const mm3.kernels.Kernels,
    form: Form,
    /// control: one kept block left out of the device's selection (the reference keeps it)
    drop: bool = false,
    /// control: the window starts one key late on the device (the reference keeps its first key)
    late: bool = false,
    shape: c.Shape = undefined,
    q: []u16 = &.{},
    keys: []u16 = &.{}, // [key][hk][D], as the cache holds them
    values: []u16 = &.{},
    sel: []i32 = &.{}, // [row][hk][topk] ascending, -1 past the last
    tol: []f64 = &.{},
    pos: usize = 0,
    lo: usize = 0,
    hi: usize = 0, // window: the last key
    words: usize = 0,
    in: [4]u64 = @splat(0),
    out: u64 = 0,

    pub fn deinit(self: *Attn) void {
        self.free();
        self.rig.deinit();
    }

    fn free(self: *Attn) void {
        const a = self.rig.gpu.gpa;
        inline for (.{ "q", "keys", "values" }) |name| {
            a.free(@field(self, name));
            @field(self, name) = &.{};
        }
        a.free(self.sel);
        a.free(self.tol);
        self.sel = &.{};
        self.tol = &.{};
    }

    pub fn kernel(self: *Attn) g.Kernel {
        var declared = switch (self.form) {
            inline else => |f| spec(f),
        };
        if (self.drop) declared.name = "control: MiniMax prompt attention drops a kept block";
        if (self.late) declared.name = "control: MiniMax draft-window attention starts a key late";
        return .{ .spec = declared, .ctx = self, .vt = &vt, .control = if (self.drop or self.late) .shapes else null };
    }

    const vt = g.hooks(Attn);

    fn window(self: *const Attn) bool {
        return self.form == .window;
    }

    fn picked(self: *const Attn) bool {
        return self.form == .kept;
    }

    /// The last key row r sees.
    fn last(self: *const Attn, r: usize) usize {
        return if (self.window()) self.hi else self.pos + r;
    }

    /// q in [-4, 4) (peaked: a lost block shows), k and v in [-1, 1); stress: 1/8 steps; non-finite: NaN q, inf k.
    pub fn prepare(self: *Attn, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.n != hq * D or shape.rows == 0 or shape.k < shape.rows) return false;
        self.free();
        self.rig.reset();
        self.shape = shape;
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        if (self.window()) {
            // lo % 128 >= 64 at 4,096 and 700 keys (the old NaN case), below 64 at 1,500
            self.hi = switch (shape.k) {
                4096 => 5060,
                700 => 899,
                else => shape.k + 30,
            };
            self.lo = self.hi + 1 - shape.k;
            self.pos = 0;
        } else {
            self.pos = shape.k - shape.rows;
            self.lo = 0;
        }
        const n_keys = self.last(shape.rows - 1) + 1;
        self.q = try a.alloc(u16, shape.rows * hq * D);
        self.keys = try a.alloc(u16, n_keys * hk * D);
        self.values = try a.alloc(u16, n_keys * hk * D);
        const amp: f32 = 4;
        for (self.q) |*v| v.* = c.toBf16(if (fill == .stress) amp * grid(rng) else (rng.float(f32) * 2 - 1) * amp);
        for ([_][]u16{ self.keys, self.values }) |x| for (x) |*v| {
            v.* = c.toBf16(if (fill == .stress) grid(rng) else rng.float(f32) * 2 - 1);
        };
        if (fill == .nonfinite) {
            self.q[3] = c.toBf16(std.math.nan(f32));
            self.keys[(9 * hk + 1) * D + 5] = c.toBf16(std.math.inf(f32));
        }
        self.sel = try a.alloc(i32, shape.rows * hk * topk);
        for (0..shape.rows) |r| for (0..hk) |h| self.choose(rng, r, self.sel[(r * hk + h) * topk ..][0..topk]);
        const dev = try a.dupe(i32, self.sel);
        defer a.free(dev);
        if (self.drop) for (0..topk - 1) |i| { // row 0, key head 0: its first kept block other than its own goes
            dev[i] = dev[i + 1];
            if (dev[i] < 0) break;
        };
        if (self.drop) dev[topk - 1] = -1;
        self.words = blocks(n_keys) / 32 + 1;
        const bits = try a.alloc(u32, shape.rows * hk * self.words);
        defer a.free(bits);
        @memset(bits, 0);
        for (0..shape.rows * hk) |p| for (dev[p * topk ..][0..topk]) |b| if (b >= 0) {
            const at: usize = @intCast(b);
            bits[p * self.words + at / 32] |= @as(u32, 1) << @intCast(at % 32);
        };
        self.in[0] = try self.rig.input(std.mem.sliceAsBytes(self.q));
        self.in[1] = try self.rig.input(std.mem.sliceAsBytes(self.keys));
        self.in[2] = try self.rig.input(std.mem.sliceAsBytes(self.values));
        self.in[3] = try self.rig.input(std.mem.sliceAsBytes(bits));
        self.out = try self.rig.output(shape.rows * hq * D * 2);
        return true;
    }

    /// A value on a 1/8 grid in [-1, 1]: bf16-exact, so stress sums start from exact inputs.
    fn grid(rng: std.Random) f32 {
        return @as(f32, @floatFromInt(@as(i32, @intCast(rng.uintLessThan(u32, 17))) - 8)) / 8;
    }

    fn blocks(keys: usize) usize {
        return (keys + 127) / 128;
    }

    /// Row r's kept blocks for a key head: picked forms keep its own and 15 others at random once it sees more than 16.
    fn choose(self: *const Attn, rng: std.Random, r: usize, out: []i32) void {
        @memset(out, -1);
        if (!self.picked()) return;
        const own = self.last(r) / 128;
        if (own + 1 <= topk) {
            for (0..own + 1) |b| out[b] = @intCast(b);
            return;
        }
        out[0] = @intCast(own);
        var n: usize = 1;
        while (n < topk) {
            const b: i32 = @intCast(rng.uintLessThan(usize, own));
            if (std.mem.indexOfScalar(i32, out[0..n], b) != null) continue;
            out[n] = b;
            n += 1;
        }
        std.mem.sort(i32, out, {}, std.sort.asc(i32));
    }

    pub fn run(self: *Attn) !void {
        const s = self.rig.s;
        const rows = self.shape.rows;
        var a: cuda.Args = .{};
        switch (self.form) {
            .kept, .every, .window => {
                const words: usize = if (self.form == .kept) self.words else 0;
                inline for (.{ self.in[0], self.in[1], self.in[2], self.in[3], self.out }) |v| a.add(v);
                for ([_]usize{ rows, self.pos, hq, hk, words }) |v| a.add(@as(c_int, @intCast(v)));
                a.add(scale);
                a.add(@as(c_int, @intCast(self.lo + @intFromBool(self.late))));
                a.add(@as(c_int, if (self.window()) @intCast(self.hi) else -1));
                try cuda.launch.launch(self.k.attn_prompt_tc, .{ .grid = .{ .x = @intCast((rows + 7) / 8), .y = hk }, .block = .{ .x = 256 } }, s, &a);
            },
        }
        try s.synchronize();
    }

    pub fn count(self: *Attn) usize {
        return self.shape.rows * hq * D;
    }

    /// The keys row r sees through key head h, ascending, into `out` (its length returned).
    fn seen(self: *const Attn, r: usize, h: usize, out: []usize) usize {
        const top = self.last(r);
        var n: usize = 0;
        if (self.picked()) {
            for (self.sel[(r * hk + h) * topk ..][0..topk]) |b| if (b >= 0) {
                const at: usize = @intCast(b);
                for (at * 128..@min(at * 128 + 128, top + 1)) |key| {
                    out[n] = key;
                    n += 1;
                }
            };
            return n;
        }
        for (self.lo..top + 1) |key| {
            out[n] = key;
            n += 1;
        }
        return n;
    }

    /// Row r, head h's softmax(q k scale) v, sums in `acc` (null: f64); `tol`: bf16 p and out, fp32 dots and exp.
    fn attend(self: *Attn, r: usize, h: usize, acc: ?c.Accum, order: g.Order, o: []f64, tol: ?[]f64) !void {
        const a = self.rig.gpu.gpa;
        const kv = h / 16;
        const list = try a.alloc(usize, self.last(r) + 1);
        defer a.free(list);
        const n = self.seen(r, kv, list);
        const s = try a.alloc(f64, 3 * n);
        defer a.free(s);
        const ds = s[n .. 2 * n];
        const p = s[2 * n ..];
        var terms: [D]f64 = undefined;
        var most = -std.math.inf(f64);
        const sc: f64 = scale;
        const qr = self.q[(r * hq + h) * D ..][0..D];
        for (list[0..n], 0..) |key, i| {
            const kr = self.keys[(key * hk + kv) * D ..][0..D];
            var abs: f64 = 0;
            for (&terms, qr, kr) |*x, qv, kvv| {
                x.* = @as(f64, c.fromBf16(qv)) * c.fromBf16(kvv);
                abs += @abs(x.*);
            }
            s[i] = (if (acc) |k| sumIn(k, order, &terms) else sumIn(.none, .sequential, &terms)) * sc;
            ds[i] = (abs * sc * (D + 16) + @abs(s[i]) * 8) * 0x1p-24 + 0x1p-20;
            most = maxNan(most, s[i]);
        }
        for (p, s[0..n]) |*x, v| x.* = @exp(v - most);
        const l = if (acc) |k| sumIn(k, order, p) else sumIn(.none, .sequential, p);
        const chain = @as(f64, @floatFromInt(n + n / 64 + 32)) * 0x1p-24; // fp32 sums of p and p v, the chunks' rescales
        const col = try a.alloc(f64, n);
        defer a.free(col);
        for (o, 0..) |*od, d| {
            var spread: f64 = 0;
            for (col, p, list[0..n], ds) |*x, w, key, e| {
                const v: f64 = c.fromBf16(self.values[(key * hk + kv) * D + d]);
                x.* = w * v;
                spread += w * @abs(v) * (0x1p-8 + e + chain);
            }
            od.* = (if (acc) |k| sumIn(k, order, col) else sumIn(.none, .sequential, col)) / l;
            if (tol) |t| t[d] = @abs(od.*) * (0x1p-8 + chain) + spread / l + 0x1p-120;
        }
    }

    /// float64 o = sum p v / sum p, p = exp(s - max), over each row's kept keys; bound as attend states.
    pub fn reference(self: *Attn, want: []f64, mag: []f64) !void {
        const a = self.rig.gpu.gpa;
        a.free(self.tol);
        self.tol = try a.alloc(f64, want.len);
        for (0..self.shape.rows) |r| for (0..hq) |h| {
            const at = (r * hq + h) * D;
            try self.attend(r, h, null, .sequential, want[at..][0..D], self.tol[at..][0..D]);
        };
        for (mag, want) |*m, w| m.* = @abs(w);
    }

    pub fn bounds(self: *Attn, _: []const f64, _: []const f64, allow: []f64) !void {
        for (allow, self.tol) |*l, t| l.* = if (std.math.isFinite(t)) t else std.math.floatMax(f64);
    }

    pub fn simulate(self: *Attn, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..self.shape.rows) |r| for (0..hq) |h| {
            const at = (r * hq + h) * D;
            try self.attend(r, h, acc, order, out[at..][0..D], null);
            for (out[at..][0..D]) |*v| v.* = c.fromBf16(c.toBf16(@floatCast(v.*)));
        };
    }
};

/// max that keeps a NaN, as float64 arithmetic on the formula would.
fn maxNan(x: f64, y: f64) f64 {
    return if (std.math.isNan(x) or std.math.isNan(y)) std.math.nan(f64) else @max(x, y);
}
