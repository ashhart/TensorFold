//! Flash Next's prompt experts on 64-pair factored tiles (flashnext_fexp.cu), bit-equal to expert_rows.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const flash = @import("flashnext");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Affine = contract.affine.Affine;
const parts = flash.batch_state;
const tk = flash.tile_kernels;

pub const Kind = enum { down, up };

/// The packed layout's blocks (experts_pack<1>): 32 columns by 32 inputs, then their scales and biases.
const block_bytes: usize = 160 * 4;
pub const pack_symbol: [:0]const u8 = "_ZN15tf_experts_pack11pack_kernelILi1EEEvPKjPKtS4_Pjiiiiii";
const experts_n = 4;

/// A token's pairs in the up form (ten experts and the shared one); one at a few pairs, so planted rows stay a few.
fn slotsOf(pairs: usize) usize {
    return if (pairs < 44) 1 else 11;
}

/// A group's 32 products from zero, its input sum, two fmas a group.
fn depth(s: c.Shape) usize {
    return 32 + 32 + 2 * (s.k / 32);
}

/// The up form's sums are given (expert_rows' fp32 form): only its SwiGLU epilogue rounds.
fn epilogue(_: c.Shape) usize {
    return 0;
}

const tiles = [3]c.Tile{ .{ .size = 64 }, .{ .size = 128, .step = 32, .min = 32 }, .{ .size = 128, .step = 64, .min = 64 } };
/// rows = expert pairs (11 a token for up), n = outputs, k = inputs: Flash Next's 640-wide experts on 2560 dims.
const down_shapes = [_]c.Shape{ .{ .rows = 704, .n = 2560, .k = 640 }, .{ .rows = 1100, .n = 2560, .k = 640 } };
const up_shapes = [_]c.Shape{ .{ .rows = 704, .n = 640, .k = 2560 }, .{ .rows = 1100, .n = 640, .k = 2560 } };

pub fn spec(kind: Kind) c.Spec {
    return switch (kind) {
        .down => .{ .name = "cuda flashnext factored experts down 64 pairs, fp32 out", .accum = .f32, .depth = depth, .ops = 2, .out = .f32, .replaces = "expert_rows' down form", .tiles = tiles, .served = &down_shapes },
        // exp's 2 ulps, 1 + e, the quotient and the product; silu rounded to bf16 before the product
        .up => .{ .name = "cuda flashnext factored experts gate and up 64 pairs, SwiGLU", .accum = .f32, .depth = epilogue, .ops = 5, .mids = 1, .out = .bf16, .replaces = "expert_rows' up form", .tiles = tiles, .served = &up_shapes },
    };
}

pub const Fns = struct { up: cuda.Function, down: cuda.Function, rows_up: cuda.Function, rows_down: cuda.Function, pack: cuda.Function };

const Plan = struct { items: u64 = 0, counts: u64 = 0, members: u64 = 0, n: usize = 0 };

pub const Fexp = struct {
    rig: rig_.Rig,
    kind: Kind,
    f: *const Fns,
    a: std.mem.Allocator = undefined,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    /// Each (matrix, expert)'s affine weights on the shared token rows (gate then up for the up form).
    aff: std.ArrayList(Affine) = .empty,
    expert: []usize = &.{},
    gs: []f32 = &.{},
    us: []f32 = &.{},
    plan64: Plan = .{},
    plan32: Plan = .{},
    dx: u64 = 0,
    dw: u64 = 0,
    base: u64 = 0,
    out: u64 = 0,

    pub fn deinit(self: *Fexp) void {
        self.drop();
        self.aff.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    fn drop(self: *Fexp) void {
        const a = self.rig.gpu.gpa;
        for (self.aff.items) |f| f.deinit();
        self.aff.clearRetainingCapacity();
        for ([_][]const u8{ std.mem.sliceAsBytes(self.expert), std.mem.sliceAsBytes(self.gs), std.mem.sliceAsBytes(self.us) }) |b| if (b.len > 0) a.free(b);
        self.expert = &.{};
        self.gs = &.{};
        self.us = &.{};
        self.rig.reset();
    }

    pub fn kernel(self: *Fexp) g.Kernel {
        return .{ .spec = spec(self.kind), .ctx = self, .vt = &vt };
    }

    const vt = g.hooks(Fexp);

    fn slots(self: *const Fexp) usize {
        return if (self.kind == .up) slotsOf(self.shape.rows) else 0;
    }

    fn tokens(self: *const Fexp) usize {
        return if (self.kind == .up) (self.shape.rows + self.slots() - 1) / self.slots() else self.shape.rows;
    }

    fn tokenOf(self: *const Fexp, p: usize) usize {
        return if (self.kind == .up) p / self.slots() else p;
    }

    pub fn prepare(self: *Fexp, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.drop();
        // two groups a stage (K in 64s), outputs in 32-column blocks
        if (shape.rows == 0 or shape.n == 0 or shape.n % 32 != 0 or shape.k == 0 or shape.k % 64 != 0) return false;
        const a = self.rig.gpu.gpa;
        self.a = a;
        self.shape = shape;
        const n = shape.n;
        const k = shape.k;
        const kg = k / 32;
        const mats: usize = if (self.kind == .up) 2 else 1;
        self.expert = try a.alloc(usize, shape.rows);
        if (self.kind == .up) {
            self.gs = try a.alloc(f32, shape.rows * n);
            self.us = try a.alloc(f32, shape.rows * n);
        }
        const words = try a.alloc(u32, mats * experts_n * n * k / 8);
        defer a.free(words);
        const scales = try a.alloc(u16, mats * experts_n * n * kg);
        defer a.free(scales);
        const biases = try a.alloc(u16, scales.len);
        defer a.free(biases);
        // each matrix of each expert from the affine family, every one on the first one's token rows
        for (0..mats * experts_n) |m| {
            const f = try Affine.init(a, .{ .rows = self.tokens(), .n = n, .k = k }, 4, 32);
            try self.aff.append(a, f);
            try f.fill(fill, seed + m);
            if (m > 0) @memcpy(f.x, self.aff.items[0].x);
            f.words(words[m * n * k / 8 ..][0 .. n * k / 8]);
            @memcpy(scales[m * n * kg ..][0 .. n * kg], f.scales);
            @memcpy(biases[m * n * kg ..][0 .. n * kg], f.biases);
        }
        // pairs to experts unevenly (one expert past a tile, one with none at small counts)
        for (self.expert, 0..) |*e, p| e.* = if (p % 5 < 3) 0 else if (p % 5 == 3) 1 else 3;
        self.plan64 = try self.plan(64);
        self.plan32 = try self.plan(cuda.experts.RowsExperts.tile);
        self.dx = try self.rig.input(std.mem.sliceAsBytes(self.aff.items[0].x));
        // pack each matrix as the engine does (experts_pack<1>), gate and up blocks side by side for the up form
        const per = experts_n * (n / 32) * kg * block_bytes;
        const packed_w = try a.alloc(u8, mats * per);
        defer a.free(packed_w);
        for (0..mats) |m| try self.pack(words[m * experts_n * n * k / 8 ..][0 .. experts_n * n * k / 8], scales[m * experts_n * n * kg ..][0 .. experts_n * n * kg], biases[m * experts_n * n * kg ..][0 .. experts_n * n * kg], packed_w[m * per ..][0..per]);
        if (self.kind == .up) {
            try self.upSums(packed_w[0..per], self.gs);
            try self.upSums(packed_w[per..], self.us);
            const stacked = try a.alloc(u8, 2 * per);
            defer a.free(stacked);
            for (0..per / block_bytes) |b| {
                @memcpy(stacked[2 * b * block_bytes ..][0..block_bytes], packed_w[b * block_bytes ..][0..block_bytes]);
                @memcpy(stacked[(2 * b + 1) * block_bytes ..][0..block_bytes], packed_w[per + b * block_bytes ..][0..block_bytes]);
            }
            self.dw = try self.rig.input(stacked);
        } else self.dw = try self.rig.input(packed_w);
        const esize = c.size(spec(self.kind).out);
        self.out = try self.rig.output(shape.rows * n * esize);
        self.base = try self.rig.scratch(shape.rows * n * esize);
        return true;
    }

    /// A grouped plan of `tile` pairs an item (expert, first member, count), its members expert by expert.
    fn plan(self: *Fexp, tile: usize) !Plan {
        const a = self.a;
        const members = try a.alloc(i32, self.shape.rows);
        defer a.free(members);
        var items: std.ArrayList(i32) = .empty;
        defer items.deinit(a);
        var at: usize = 0;
        for (0..experts_n) |e| {
            const first = at;
            for (self.expert, 0..) |pe, p| if (pe == e) {
                members[at] = @intCast(p);
                at += 1;
            };
            var i = first;
            while (i < at) : (i += tile) try items.appendSlice(a, &.{ @intCast(e), @intCast(i), @intCast(@min(tile, at - i)) });
        }
        const n_items = [_]i32{@intCast(items.items.len / 3)};
        return .{ .items = try self.rig.input(std.mem.sliceAsBytes(items.items)), .counts = try self.rig.input(std.mem.sliceAsBytes(&n_items)), .members = try self.rig.input(std.mem.sliceAsBytes(members)), .n = items.items.len / 3 };
    }

    fn upload(self: *Fexp, bytes: []const u8) !u64 {
        const p = try self.rig.scratch(bytes.len);
        const b: cuda.DeviceBuffer = .{ .d = self.rig.gpu.d, .ptr = p, .len = bytes.len };
        try b.upload(0, bytes);
        return p;
    }

    /// experts_pack<1> on the device, read back: [experts][n / 32][k / 32] blocks.
    fn pack(self: *Fexp, words: []const u32, scales: []const u16, biases: []const u16, out: []u8) !void {
        const n = self.shape.n;
        const kg = self.shape.k / 32;
        const dst = try self.rig.scratch(out.len);
        var a: cuda.Args = .{};
        a.add(try self.upload(std.mem.sliceAsBytes(words)));
        a.add(try self.upload(std.mem.sliceAsBytes(scales)));
        a.add(try self.upload(std.mem.sliceAsBytes(biases)));
        a.add(dst);
        for ([_]usize{ n, self.shape.k / 8, kg, n / 32, 1, 0 }) |v| a.add(@as(c_int, @intCast(v))); // one table, the 0th
        try cuda.launch.launch(self.f.pack, .{ .grid = .{ .x = @intCast(kg), .y = @intCast(n / 32), .z = experts_n }, .block = .{ .x = 160 } }, self.rig.s, &a);
        @memcpy(out, try self.rig.bytes(dst, out.len));
    }

    /// One matrix's fp32 sums [pairs, n] by expert_rows' down form on the token rows, read back.
    fn upSums(self: *Fexp, w: []const u8, out: []f32) !void {
        const dw = try self.upload(w);
        const dst = try self.rig.scratch(out.len * 4);
        try self.rows(self.f.rows_down, self.slots(), dw, dst);
        @memcpy(std.mem.sliceAsBytes(out), try self.rig.bytes(dst, out.len * 4));
    }

    fn args(self: *Fexp, slot_n: usize, w: u64, p: Plan, out: u64) cuda.Args {
        const sh = self.shape;
        return cuda.experts.Affine4Experts.argsOf(self.dx, sh.k, slot_n, w, sh.k / 32, sh.n / 32, .{ .items = p.items, .counts = p.counts, .members = p.members, .rank = 0, .hist = 0 }, out, sh.n);
    }

    /// RowsExperts.run's launch: items of 32 pairs, four column blocks a warp's share, at most 4096 blocks.
    fn rows(self: *Fexp, f: cuda.Function, slot_n: usize, w: u64, out: u64) !void {
        var a = self.args(slot_n, w, self.plan32, out);
        const nb = self.shape.n / 32;
        const grid = @min((self.plan32.n * nb + 3) / 4, 4096);
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = 128 } }, self.rig.s, &a);
    }

    /// cuda_batch.zig's fexp launch: items of 64 pairs, wn column blocks a CTA.
    fn launch(self: *Fexp, out: u64) !void {
        const wn: usize = if (self.kind == .up) 2 else 4;
        const f = if (self.kind == .up) self.f.up else self.f.down;
        var a = self.args(self.slots(), self.dw, self.plan64, out);
        const nb = self.shape.n / 32;
        try cuda.launch.launch(f, .{ .grid = .{ .x = @intCast(self.plan64.n * ((nb + wn - 1) / wn)) }, .block = .{ .x = 256 }, .shared = tk.fexp_smem }, self.rig.s, &a);
    }

    pub fn run(self: *Fexp) !void {
        try self.launch(self.out);
        try self.rig.s.synchronize();
    }

    /// expert_rows on the same inputs and pairs (its own plan: items of 32 over the same members).
    pub fn replaced(self: *Fexp) ![]const u8 {
        const esize = c.size(spec(self.kind).out);
        try self.rows(if (self.kind == .up) self.f.rows_up else self.f.rows_down, self.slots(), self.dw, self.base);
        try self.rig.s.synchronize();
        return self.rig.bytes(self.base, self.count() * esize);
    }

    pub fn count(self: *Fexp) usize {
        return self.shape.rows * self.shape.n;
    }

    /// SwiGLU on the bf16 gate and up values in float64 (the kernel's rounding of silu is the spec's one mid).
    fn swiglu(gv: f64, uv: f64) f64 {
        return gv / (1 + @exp(-gv)) * uv;
    }

    fn bf(v: f64) f64 {
        return c.fromBf16(c.toBf16(@floatCast(v)));
    }

    pub fn reference(self: *Fexp, want: []f64, mag: []f64) !void {
        const n = self.shape.n;
        if (self.kind == .up) {
            for (want, mag, self.gs, self.us) |*w, *m, gv, uv| {
                w.* = swiglu(bf(gv), bf(uv));
                m.* = 0;
            }
            return;
        }
        for (0..self.shape.rows) |p| for (0..n) |j| {
            const o = self.aff.items[self.expert[p]].one(p, j);
            want[p * n + j] = o[0];
            mag[p * n + j] = o[1];
        };
    }

    /// Each matrix's sums as an accumulator leaves them (the affine family's structures), per pair from its expert.
    pub fn simulate(self: *Fexp, acc: c.Accum, order: g.Order, out: []f64) !void {
        const a = self.a;
        const n = self.shape.n;
        const t = self.tokens();
        const all = try a.alloc(f64, self.aff.items.len * t * n);
        defer a.free(all);
        for (self.aff.items, 0..) |f, m| try f.simulate(acc, order, all[m * t * n ..][0 .. t * n]);
        for (0..self.shape.rows) |p| for (0..n) |j| {
            const at = self.tokenOf(p) * n + j;
            const s0 = all[self.expert[p] * t * n + at];
            out[p * n + j] = if (self.kind == .up) swiglu(bf(s0), bf(all[(experts_n + self.expert[p]) * t * n + at])) else s0;
        };
    }
};
