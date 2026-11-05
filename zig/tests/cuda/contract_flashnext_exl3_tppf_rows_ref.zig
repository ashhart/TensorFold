//! Host cases of Flash Next EXL3's TP prefill rotate (tf_exl3pf_rotate) and place (tf_exl3pf_tp_place) launches.
const std = @import("std");
const core = @import("core");
const flash = @import("flashnext");
const ref = @import("contract_flashnext_exl3_tppf_ref.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const pk = flash.exl3.prefill_k;

const Fault = ref.Fault;
const slots = ref.slots;
const dims = ref.dims;

/// Experts of one layer's table: 15 routed and the shared one as id 15.
pub const table_n = 16;
const had32: f32 = 0.08838834764831845;
const had64: f64 = 1.0 / @sqrt(128.0);
const block = 128;

/// The Hadamard transform of a block in the kernel's order: level h = 1, 2, ..., 64, each pair (a + b, a - b).
fn fwht(comptime T: type, v: *[block]T, fault: Fault) void {
    var h: usize = 1;
    while (h < block) : (h *= 2) for (0..block) |i| {
        if (i & h != 0) continue;
        const a = v[i];
        const b = v[i | h];
        v[i] = if (T == f32 and fault == .bf16_adds) ref.bf16(a + b) else a + b;
        v[i | h] = if (T == f32 and fault == .bf16_adds) ref.bf16(a - b) else a - b;
    };
}

/// Gate 4's inputs: Hadamard rows 3 and 5 summed (4,096 a sign) plus 64 m, m below 16: exact fp32 sums, most cancel.
fn stressX(k: usize) f32 {
    const s3: f32 = if (@popCount(3 & k) % 2 == 1) -1 else 1;
    const s5: f32 = if (@popCount(5 & k) % 2 == 1) -1 else 1;
    return 4096 * (s3 + s5) + 64 * @as(f32, @floatFromInt((k * 5 + 3) % 16));
}

/// 7 adds, the product and one scale: the roundings a value passes (the 7 levels and the multiply).
fn rotateDepth(_: c.Shape) usize {
    return 8;
}

const rotate_served = [_]c.Shape{
    .{ .rows = 5, .n = dims, .k = block },
    .{ .rows = 1, .n = dims, .k = block },
    .{ .rows = 130, .n = dims, .k = block },
    .{ .rows = 1408, .n = dims, .k = block },
};
const max_rows = 1408;

/// Member c of the pass: xg[c], xu[c] = fp16 of its row's bf16 x times suh[expert], rotated by the 128 Hadamard.
pub const Rotate = struct {
    gpa: std.mem.Allocator,
    fault: Fault = .none,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x: []u16 = &.{},
    picks: []i32 = &.{},
    plan: pk.Plan = undefined,
    suh: [2][]u16 = .{ &.{}, &.{} },
    msg: [200]u8 = undefined,

    pub const n_in = 5;
    pub const n_out = 2;

    pub fn deinit(self: *Rotate) void {
        if (self.arena) |*a| a.deinit();
    }

    pub fn spec(self: *const Rotate) c.Spec {
        const name = switch (self.fault) {
            .none => "cuda Flash Next EXL3 prefill tf_exl3pf_rotate",
            .nan_dropped => "control: tf_exl3pf_rotate with NaN and infinite inputs zeroed",
            .writes_input => "control: tf_exl3pf_rotate writing its input",
            .wrong => "control: tf_exl3pf_rotate taking the up table's signs from the gate table",
            .bf16_adds => "control: tf_exl3pf_rotate with bf16 butterfly adds",
            .flat => unreachable,
        };
        return .{
            .name = name,
            .accum = .f32,
            .depth = rotateDepth,
            // the normalization constant is rounded once and the product once
            .ops = 2,
            .out = .f16,
            .nan = .{ .stated = "a NaN or infinite input reaches the 128 outputs of its block, as the host transform gives them" },
            .tiles = ref.edges,
            .served = &rotate_served,
        };
    }

    pub fn accepts(shape: c.Shape) bool {
        return shape.rows >= 1 and shape.rows <= max_rows and shape.n >= 256 and shape.n % 256 == 0 and shape.n <= dims and shape.k == block;
    }

    pub fn prepare(self: *Rotate, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!accepts(shape)) return false;
        self.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.gpa);
        const a = self.arena.?.allocator();
        self.shape = shape;
        const rows = shape.rows;
        const n = shape.n;
        var prng = std.Random.DefaultPrng.init(seed ^ 0x726f7461);
        const rng = prng.random();
        const stress = fill == .stress;
        self.x = try a.alloc(u16, rows * n);
        for (self.x, 0..) |*v, i| v.* = c.toBf16(if (stress) stressX(i % block) else (rng.float(f32) - 0.5) * 2);
        if (fill == .nonfinite) {
            self.x[(rows - 1) * n] = c.toBf16(std.math.nan(f32));
            if (rows > 1) self.x[(rows - 2) * n + 1] = c.toBf16(std.math.inf(f32));
        }
        for (&self.suh) |*t| {
            t.* = try a.alloc(u16, table_n * n);
            for (t.*) |*h| {
                const m: f16 = @floatCast(0.5 + rng.float(f32));
                h.* = @bitCast(if (stress) @as(f16, 1) else if (rng.boolean()) m else -m);
            }
        }
        self.picks = try a.alloc(i32, rows * slots);
        for (0..rows) |r| {
            var ids: [table_n - 1]i32 = undefined;
            for (&ids, 0..) |*v, i| v.* = @intCast(i);
            rng.shuffle(i32, &ids);
            @memcpy(self.picks[r * slots ..][0 .. slots - 1], ids[0 .. slots - 1]);
            self.picks[r * slots + slots - 1] = table_n - 1;
        }
        self.plan = try pk.Plan.init(a, rows * slots, table_n);
        var counts: [table_n]u32 = undefined;
        pk.build(self.picks, null, &counts, &self.plan);
        return true;
    }

    pub fn inputs(self: *const Rotate) [n_in][]const u8 {
        return .{ std.mem.sliceAsBytes(self.x), std.mem.sliceAsBytes(self.picks), self.plan.memberBytes(), std.mem.sliceAsBytes(self.suh[0]), std.mem.sliceAsBytes(self.suh[1]) };
    }

    pub fn outputLens(self: *const Rotate) [n_out]usize {
        return .{ self.plan.n_members * self.shape.n * 2, self.plan.n_members * self.shape.n * 2 };
    }

    pub fn count(self: *const Rotate) usize {
        return 2 * self.plan.n_members * self.shape.n;
    }

    /// The block's products x * suh (fp32 for the kernel, float64 for the reference) of member `m` in matrix `mat`.
    fn products(self: *const Rotate, comptime T: type, fault: Fault, mat: usize, m: usize, blk: usize, v: *[block]T) void {
        const n = self.shape.n;
        const p: usize = @intCast(self.plan.members[m]);
        const e: usize = @intCast(self.picks[p]);
        const table = self.suh[if (fault == .wrong and mat == 1) 0 else mat];
        for (v, 0..) |*t, j| {
            const col = blk * block + j;
            var xv = c.fromBf16(self.x[p / slots * n + col]);
            if (fault == .nan_dropped and !std.math.isFinite(xv)) xv = 0;
            const sv: f32 = @as(f16, @bitCast(table[e * n + col]));
            t.* = @as(T, xv) * @as(T, sv);
        }
    }

    /// The kernel's 128 fp16 outputs of one block, bit for bit.
    fn outBlock(self: *const Rotate, fault: Fault, mat: usize, m: usize, blk: usize, bits: *[block]u16) void {
        var v: [block]f32 = undefined;
        self.products(f32, fault, mat, m, blk, &v);
        fwht(f32, &v, fault);
        for (bits, v) |*b, x| b.* = @bitCast(@as(f16, @floatCast(x * had32)));
    }

    /// Float64's transform of the block's products, and the sum of their magnitudes (the bound's mag).
    fn refBlock(self: *const Rotate, mat: usize, m: usize, blk: usize, want: *[block]f64, mag: *f64) void {
        var t: [block]f64 = undefined;
        self.products(f64, .none, mat, m, blk, &t);
        mag.* = 0;
        for (t) |x| mag.* += @abs(x) * had64;
        fwht(f64, &t, .none);
        for (want, t) |*w, x| w.* = x * had64;
    }

    fn at(self: *const Rotate, mat: usize, m: usize, blk: usize) usize {
        return (mat * self.plan.n_members + m) * self.shape.n + blk * block;
    }

    pub fn reference(self: *const Rotate, want: []f64, mag: []f64) !void {
        var w: [block]f64 = undefined;
        var s: f64 = 0;
        for (0..2) |mat| for (0..self.plan.n_members) |m| for (0..self.shape.n / block) |blk| {
            self.refBlock(mat, m, blk, &w, &s);
            @memcpy(want[self.at(mat, m, blk)..][0..block], &w);
            @memset(mag[self.at(mat, m, blk)..][0..block], s);
        };
    }

    /// Output `j` of a block summed in `acc` in `order`: the products with the transform's signs.
    pub fn simulate(self: *const Rotate, acc: c.Accum, order: g.Order, out: []f64) !void {
        var t: [block]f64 = undefined;
        var s: [block]f64 = undefined;
        for (0..2) |mat| for (0..self.plan.n_members) |m| for (0..self.shape.n / block) |blk| {
            self.products(f64, .none, mat, m, blk, &t);
            for (0..block) |j| {
                for (&s, t, 0..) |*y, x, k| y.* = if (@popCount(j & k) % 2 == 1) -x else x;
                out[self.at(mat, m, blk) + j] = core.contract.rows.sumIn(acc, order, &s) * had64;
            }
        };
    }

    /// Each output against the host's fp16 bits, float64's class and the declared bound (gate 1: finite and not).
    pub fn verify(self: *Rotate, out: []const u8, gate: g.Gate) !?[]const u8 {
        const s = self.spec();
        var bits: [block]u16 = undefined;
        var want: [block]f64 = undefined;
        var mag: f64 = 0;
        var finite: usize = 0;
        for (0..2) |mat| for (0..self.plan.n_members) |m| for (0..self.shape.n / block) |blk| {
            self.outBlock(.none, mat, m, blk, &bits);
            self.refBlock(mat, m, blk, &want, &mag);
            for (bits, want, 0..) |host, w, j| {
                const i = self.at(mat, m, blk) + j;
                const got = std.mem.readInt(u16, out[2 * i ..][0..2], .little);
                const same = got == host or (std.math.isNan(@as(f16, @bitCast(got))) and std.math.isNan(@as(f16, @bitCast(host))));
                if (!same) return try std.fmt.bufPrint(&self.msg, "output {d}: {x:0>4} where the host transform gives {x:0>4}", .{ i, got, host });
                const have: f64 = @as(f16, @bitCast(got));
                if (c.classOf(have) != c.classOf(w)) return try std.fmt.bufPrint(&self.msg, "output {d} is {t} where float64 gives {t}", .{ i, c.classOf(have), c.classOf(w) });
                if (!std.math.isFinite(w)) continue;
                finite += 1;
                if (@abs(have - w) > c.bound(s, self.shape, w, mag)) return try std.fmt.bufPrint(&self.msg, "output {d}: {d} is outside float64's bound around {d}", .{ i, have, w });
            }
        };
        if (gate == .nonfinite and (finite == 0 or finite == self.count())) return "the planted values leave all outputs finite or none";
        return null;
    }

    pub fn emulate(self: *Rotate, fault: Fault, out: []u8) void {
        var bits: [block]u16 = undefined;
        for (0..2) |mat| for (0..self.plan.n_members) |m| for (0..self.shape.n / block) |blk| {
            self.outBlock(fault, mat, m, blk, &bits);
            for (bits, 0..) |b, j| std.mem.writeInt(u16, out[2 * (self.at(mat, m, blk) + j) ..][0..2], b, .little);
        };
        if (fault == .writes_input) self.x[0] +%= 1;
    }
};

fn noSums(_: c.Shape) usize {
    return 0;
}

const place_served = [_]c.Shape{
    .{ .rows = 5, .n = dims, .k = 0 },
    .{ .rows = 1, .n = dims, .k = 0 },
    .{ .rows = 130, .n = dims, .k = 0 },
    .{ .rows = 1024, .n = dims, .k = 0 },
};
const max_place_rows = 1024;

/// y[r, 0, :] = sum[r] and the weights {1, 0, ..., 0}: the other slots of y are left as they were (poison).
pub const Place = struct {
    gpa: std.mem.Allocator,
    fault: Fault = .none,
    arena: ?std.heap.ArenaAllocator = null,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    sum: []f32 = &.{},
    msg: [200]u8 = undefined,

    pub const n_in = 1;
    pub const n_out = 2;

    pub fn deinit(self: *Place) void {
        if (self.arena) |*a| a.deinit();
    }

    pub fn spec(self: *const Place) c.Spec {
        const name = switch (self.fault) {
            .none => "cuda Flash Next EXL3 prefill tf_exl3pf_tp_place",
            .nan_dropped => "control: tf_exl3pf_tp_place with NaN zeroed",
            .writes_input => "control: tf_exl3pf_tp_place writing its input",
            .wrong => "control: tf_exl3pf_tp_place leaving every weight 1",
            else => unreachable,
        };
        return .{
            .name = name,
            .accum = .none,
            .depth = noSums,
            .out = .raw,
            .nan = .{ .stated = "y[r, 0] holds the branch's bits (NaN payloads and infinities included), weights are 1, 0, ..., 0" },
            .tiles = ref.edges,
            .served = &place_served,
        };
    }

    pub fn accepts(shape: c.Shape) bool {
        return shape.rows >= 1 and shape.rows <= max_place_rows and shape.n >= 256 and shape.n % 256 == 0 and shape.n <= dims;
    }

    pub fn prepare(self: *Place, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (!accepts(shape)) return false;
        self.deinit();
        self.arena = std.heap.ArenaAllocator.init(self.gpa);
        self.shape = shape;
        self.sum = try self.arena.?.allocator().alloc(f32, shape.rows * shape.n);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x706c6163);
        const rng = prng.random();
        for (self.sum) |*v| v.* = rng.floatNorm(f32) * 3;
        // a row each of NaN (with a payload), +Inf and -Inf every 7th column
        if (fill == .nonfinite) for (self.sum, 0..) |*v, i| if (i % 7 == 0) {
            v.* = switch ((i / shape.n) % 3) {
                0 => @bitCast(@as(u32, 0x7fc01234)),
                1 => std.math.inf(f32),
                else => -std.math.inf(f32),
            };
        };
        return true;
    }

    pub fn inputs(self: *const Place) [n_in][]const u8 {
        return .{std.mem.sliceAsBytes(self.sum)};
    }

    fn yLen(self: *const Place) usize {
        return self.shape.rows * slots * self.shape.n * 4;
    }

    pub fn outputLens(self: *const Place) [n_out]usize {
        return .{ self.yLen(), self.shape.rows * slots * 4 };
    }

    pub fn count(self: *const Place) usize {
        return self.yLen() + self.shape.rows * slots * 4;
    }

    /// The bytes y and the weights hold after a launch (a planted fault's too); everything else keeps the poison.
    fn build(self: *const Place, fault: Fault, out: []u8) void {
        @memset(out, 0xff);
        const n = self.shape.n;
        for (0..self.shape.rows) |r| for (0..n) |col| {
            const v = self.sum[r * n + col];
            ref.put(out, r * slots * n + col, if (fault == .nan_dropped and std.math.isNan(v)) 0 else v);
        };
        for (0..self.shape.rows * slots) |i| ref.put(out, self.yLen() / 4 + i, if (i % slots == 0 or fault == .wrong) 1 else 0);
    }

    pub fn reference(self: *Place, want: []f64, mag: []f64) !void {
        const bytes = try self.arena.?.allocator().alloc(u8, self.count());
        self.build(.none, bytes);
        for (want, mag, bytes) |*w, *m, b| {
            w.* = @floatFromInt(b);
            m.* = w.*;
        }
    }

    pub fn simulate(_: *const Place, _: c.Accum, _: g.Order, _: []f64) !void {
        return error.NoSums;
    }

    /// The branch's bytes in y[r, 0, :], the weights, and poison elsewhere (gate 1: a non-finite value shows).
    pub fn verify(self: *Place, out: []const u8, gate: g.Gate) !?[]const u8 {
        const want = try self.arena.?.allocator().alloc(u8, self.count());
        self.build(.none, want);
        if (std.mem.indexOfDiff(u8, out, want)) |at| return try std.fmt.bufPrint(&self.msg, "byte {d} of {d} is {x:0>2} where the branch's copy has {x:0>2}", .{ at, out.len, out[at], want[at] });
        if (gate != .nonfinite) return null;
        var bad: usize = 0;
        for (self.sum) |v| bad += @intFromBool(!std.math.isFinite(v));
        return if (bad == 0 or bad == self.sum.len) "the planted values leave all of the branch finite or none" else null;
    }

    pub fn emulate(self: *Place, fault: Fault, out: []u8) void {
        self.build(fault, out);
        if (fault == .writes_input) self.sum[0] += 1;
    }
};
