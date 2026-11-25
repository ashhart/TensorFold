//! Drama 3's compact router (d3_router_compact) under the contract on CUDA: picks and weights against FP64.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const Gpu = @import("check.zig").Gpu;
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const experts = 256;
const top = 8;
// drama3/config.zig's moe_proj rows: [shared gate 512 | up 512 | router 256 | shared gate 1 | pad 7]
const stride: usize = 1288;
const r_off: usize = 1024;

fn noSums(_: c.Shape) usize {
    return 0;
}

const rule = "only held logits count; a NaN or +inf held logit, or none finite, gives NaN weights over the lowest held ids";

/// Rows: tokens; n: experts held of 256 (8 up); k: the router's 256 logits. Prefill, decode, a whole map.
pub fn spec() c.Spec {
    return .{
        .name = "cuda drama3 compact router d3_router_compact",
        .accum = .none,
        .depth = noSums,
        .out = .raw,
        .nan = .{ .stated = rule },
        .tiles = .{ .{}, .{ .size = 9, .step = 1, .min = 8 }, .{} },
        .served = &.{ .{ .rows = 64, .n = 64, .k = experts }, .{ .rows = 1, .n = 64, .k = experts }, .{ .rows = 64, .n = 256, .k = experts }, .{ .rows = 64, .n = 8, .k = experts } },
    };
}

pub const Router = struct {
    rig: rig_.Rig,
    f: cuda.Function,
    control: bool = false, // f is d3_router: no mask, original ids, must fail gate 3
    shape: c.Shape = undefined,
    logits: []u16 = &.{},
    slots: [experts]i32 = undefined,
    held: [experts]u8 = undefined,
    sp: u64 = 0,
    sl: u64 = 0,
    picks: u64 = 0,
    weights: u64 = 0,
    msg: [200]u8 = undefined,

    pub fn init(gpu: Gpu, s: cuda.Stream, mod: cuda.Module, control: bool) !Router {
        return .{ .rig = .{ .gpu = gpu, .s = s }, .f = try mod.function(if (control) "d3_router" else "d3_router_compact"), .control = control };
    }

    pub fn deinit(self: *Router) void {
        self.rig.gpu.gpa.free(self.logits);
        self.rig.deinit();
    }

    pub fn kernel(self: *Router) g.Kernel {
        var k: g.Kernel = .{ .spec = spec(), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
        if (self.control) k.spec.name = "cuda drama3 compact router control: d3_router, unmasked";
        return k;
    }

    const vt = g.hooks(Router);

    fn logit(self: *const Router, t: usize, id: usize) f64 {
        return c.fromBf16(self.logits[t * stride + r_off + id]);
    }

    /// Held: a random n of 256 in shuffled slots; excluded logits above every held one; held on a 1/4 grid (ties).
    pub fn prepare(self: *Router, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != experts or shape.n < top or shape.n > experts) return false;
        self.rig.reset();
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        var ids: [experts]u8 = undefined;
        for (&ids, 0..) |*id, i| id.* = @intCast(i);
        rng.shuffle(u8, &ids);
        @memcpy(self.held[0..shape.n], ids[0..shape.n]);
        self.slots = @splat(-1);
        for (self.held[0..shape.n], 0..) |id, slot| self.slots[id] = @intCast(slot);
        const a = self.rig.gpu.gpa;
        a.free(self.logits);
        self.logits = &.{};
        self.logits = try a.alloc(u16, shape.rows * stride);
        @memset(self.logits, c.toBf16(std.math.nan(f32))); // outside the router window: a misread offset poisons
        for (0..shape.rows) |t| for (0..experts) |id| {
            const v: f32 = if (self.slots[id] >= 0) @as(f32, @floatFromInt(rng.uintLessThan(u32, 24))) / 4 - 3 else 8 + @as(f32, @floatFromInt(rng.uintLessThan(u32, 8)));
            self.logits[t * stride + r_off + id] = c.toBf16(v);
        };
        if (fill == .nonfinite) {
            const h = self.held[0..shape.n];
            const ex = for (0..experts) |id| {
                if (self.slots[id] < 0) break id;
            } else null;
            if (ex) |id| self.logits[r_off + id] = c.toBf16(std.math.nan(f32)); // row 0: masked away
            self.logits[stride + r_off + h[1]] = c.toBf16(std.math.nan(f32));
            self.logits[2 * stride + r_off + h[2]] = c.toBf16(std.math.inf(f32));
            for (h) |id| self.logits[3 * stride + r_off + id] = c.toBf16(-std.math.inf(f32));
            self.logits[4 * stride + r_off + h[0]] = c.toBf16(-std.math.inf(f32)); // one -inf held: finite row
        }
        self.sp = try self.rig.input(std.mem.sliceAsBytes(self.logits));
        self.sl = try self.rig.input(std.mem.sliceAsBytes(&self.slots));
        self.picks = try self.rig.output(shape.rows * top * 4);
        self.weights = try self.rig.output(shape.rows * top * 4);
        return true;
    }

    pub fn run(self: *Router) !void {
        var a: cuda.Args = .{};
        a.add(self.sp);
        a.add(@as(c_int, stride));
        a.add(@as(c_int, r_off));
        a.add(self.picks);
        a.add(self.weights);
        a.add(self.sl);
        try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(self.shape.rows) }, .block = .{ .x = experts } }, self.rig.s, &a);
        try self.rig.s.synchronize();
    }

    pub fn count(self: *Router) usize {
        return 2 * self.shape.rows * top;
    }

    pub fn reference(_: *Router, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Row t's eight picks in FP64: held ids by logit, ties to the lower original id; the weights, NaN if poisoned.
    fn fp64(self: *const Router, t: usize, ids: *[top]usize, weights: *[top]f64) bool {
        var order: [experts]usize = undefined;
        const h = order[0..self.shape.n];
        for (h, self.held[0..self.shape.n]) |*o, id| o.* = id;
        var max = -std.math.inf(f64);
        var poisoned = false;
        for (h) |id| {
            const v = self.logit(t, id);
            poisoned = poisoned or std.math.isNan(v) or v == std.math.inf(f64);
            max = @max(max, v);
        }
        poisoned = poisoned or max == -std.math.inf(f64);
        const Cmp = struct {
            fn lt(_: *const Router, a: usize, b: usize) bool {
                return a < b; // poisoned: every held prob reads 0, so the lowest ids win
            }
            fn before(r: struct { *const Router, usize }, a: usize, b: usize) bool {
                const la = r[0].logit(r[1], a);
                const lb = r[0].logit(r[1], b);
                return la > lb or (la == lb and a < b);
            }
        };
        if (poisoned) std.sort.pdq(usize, h, self, Cmp.lt) else std.sort.pdq(usize, h, .{ self, t }, Cmp.before);
        var sum: f64 = 0;
        for (h[0..top]) |id| sum += @exp(self.logit(t, id) - max);
        for (h[0..top], ids, weights) |id, *i, *w| {
            i.* = id;
            w.* = if (poisoned) std.math.nan(f64) else @exp(self.logit(t, id) - max) / sum;
        }
        return poisoned;
    }

    pub fn check(self: *Router, at: g.Gate) !?[]const u8 {
        const n = self.shape.rows * top;
        const got_p = try self.rig.gpu.gpa.alloc(i32, n);
        defer self.rig.gpu.gpa.free(got_p);
        @memcpy(std.mem.sliceAsBytes(got_p), try self.rig.bytes(self.picks, n * 4));
        const got_w: []align(1) const f32 = std.mem.bytesAsSlice(f32, try self.rig.bytes(self.weights, n * 4));
        var poisoned: usize = 0;
        for (0..self.shape.rows) |t| {
            var ids: [top]usize = undefined;
            var weights: [top]f64 = undefined;
            poisoned += @intFromBool(self.fp64(t, &ids, &weights));
            for (0..top) |k| {
                const p = got_p[t * top + k];
                const w: f64 = got_w[t * top + k];
                const slot = self.slots[ids[k]];
                const same = if (std.math.isNan(weights[k])) std.math.isNan(w) else @abs(w - weights[k]) <= 1e-6;
                if (p != slot or !same) return try std.fmt.bufPrint(&self.msg, "row {d} pick {d}: slot {d} weight {e}, FP64 slot {d} (id {d}) weight {e}", .{ t, k, p, w, slot, ids[k], weights[k] });
            }
        }
        if (at == .nonfinite and poisoned != 3) return try std.fmt.bufPrint(&self.msg, "{d} poisoned rows, not the 3 planted: no proof", .{poisoned});
        return null;
    }
};
