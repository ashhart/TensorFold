//! GLM-5.3-Flash M3 prompt index and KDA contracts: byte-exact against unvectorized baseline, plus planted controls.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const hi: usize = 32;
const di: usize = 128;
const d_kda: usize = 128;
const taps: usize = 4;

const ScoreArgs = extern struct { p0: u32, q_stride: u32, w_stride: u32, s_stride: u32, rows: u32 };

fn bfVal(x: f32) u16 {
    return c.toBf16(x);
}

fn fillBf(dst: []u16, rng: std.Random, scale: f32) void {
    for (dst) |*v| v.* = bfVal((rng.float(f32) * 2 - 1) * scale);
}

fn fillF32(dst: []f32, rng: std.Random, lo: f32, hi_v: f32) void {
    for (dst) |*v| v.* = lo + rng.float(f32) * (hi_v - lo);
}

fn packRefs(a: std.mem.Allocator, rig: *const rig_.Rig, refs: []const rig_.Ref, lens: []const usize, out: *std.ArrayList(u8)) ![]const u8 {
    out.clearRetainingCapacity();
    for (refs, lens) |r, len| try out.appendSlice(a, rig.bytes(r, len));
    return out.items;
}

fn allocOutputSet(rig: *rig_.Rig, lens: []const usize, refs: []rig_.Ref, bases: []rig_.Ref) !void {
    for (lens, 0..) |len, i| {
        refs[i] = try rig.output(len);
        bases[i] = try rig.scratch(len);
        @memset(rig.bytes(bases[i], len), 0xff);
    }
}

pub const IndexKind = enum { scores, decode };

pub const Index = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: IndexKind,
    control: bool,
    name: []const u8,
    pipe: mtl.Pipeline,
    base_pipe: mtl.Pipeline,
    rows: usize = 0,
    p0: usize = 0,
    most: usize = 0,
    s_stride: usize = 0,
    refs: [3]rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    base: rig_.Ref = undefined,
    bytes: std.ArrayList(u8) = .empty,
    base_bytes: std.ArrayList(u8) = .empty,

    fn depth(s: c.Shape) usize {
        _ = s;
        return 0;
    }

    pub fn spec(kind: IndexKind, control: bool, name: []const u8) c.Spec {
        const served: []const c.Shape = switch (kind) {
            .scores => &.{ .{ .rows = 1, .n = di, .k = 16 }, .{ .rows = 33, .n = di, .k = 128 }, .{ .rows = 65, .n = di, .k = 260 } },
            .decode => &.{ .{ .rows = 1, .n = di, .k = 16 }, .{ .rows = 4, .n = di, .k = 516 }, .{ .rows = 16, .n = di, .k = 1028 } },
        };
        const tiles: [3]c.Tile = switch (kind) {
            .scores => .{ .{ .size = 32, .step = 1, .min = 1 }, .{ .size = di, .step = di, .min = di }, .{ .size = 256, .step = 4, .min = 16 } },
            .decode => .{ .{ .size = 4, .step = 1, .min = 1 }, .{ .size = di, .step = di, .min = di }, .{ .size = 256, .step = 4, .min = 16 } },
        };
        return .{
            .name = name,
            .accum = .none,
            .depth = depth,
            .out = .f32,
            .replaces = if (control) null else "unvectorized M3 glm_index_nax kernel",
            .nan = .{ .stated = "every output bit matches the unvectorized baseline, non-finite inputs included" },
            .tiles = tiles,
            .served = served,
            .sums = .{ .why = "bit-preserving layout optimization; arithmetic and accumulation order match the baseline" },
        };
    }

    const vt = blk: {
        var v = g.hooks(Index);
        v.output = output;
        v.intact = intact;
        v.check = checkNonfinite;
        break :blk v;
    };

    pub fn kernel(self: *Index) g.Kernel {
        return .{ .spec = spec(self.kind, self.control, self.name), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    pub fn deinit(self: *Index) void {
        self.rig.deinit();
        self.bytes.deinit(self.a);
        self.base_bytes.deinit(self.a);
    }

    pub fn prepare(self: *Index, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        if (shape.rows == 0 or shape.n != di or shape.k + shape.rows < 4) return false;
        self.rows = shape.rows;
        self.p0 = shape.k;
        self.most = (self.p0 + self.rows) / 4;
        const q_stride = hi * di + 64;
        const w_stride = hi + 16;
        self.s_stride = self.most + 5;
        const iq = try self.a.alloc(u16, self.rows * q_stride);
        defer self.a.free(iq);
        const iw = try self.a.alloc(u16, self.rows * w_stride);
        defer self.a.free(iw);
        const pool = try self.a.alloc(u16, self.most * di);
        defer self.a.free(pool);
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        fillBf(iq, rng, 1.5);
        fillBf(iw, rng, 1.5);
        fillBf(pool, rng, 1.5);
        if (fill == .nonfinite) {
            iq[0] = bfVal(std.math.nan(f32));
            iq[di + 3] = bfVal(std.math.inf(f32));
            iq[2 * di + 7] = bfVal(-std.math.inf(f32));
            iw[1] = bfVal(std.math.inf(f32));
            pool[0] = bfVal(std.math.nan(f32));
            pool[(self.most - 1) * di + 5] = bfVal(-std.math.inf(f32));
        }
        self.refs[0] = try self.rig.input(std.mem.sliceAsBytes(iq));
        self.refs[1] = try self.rig.input(std.mem.sliceAsBytes(iw));
        self.refs[2] = try self.rig.input(std.mem.sliceAsBytes(pool));
        self.out = try self.rig.output(self.rows * self.s_stride * 4);
        self.base = try self.rig.scratch(self.rows * self.s_stride * 4);
        @memset(self.rig.bytes(self.base, self.rows * self.s_stride * 4), 0xff);
        return true;
    }

    fn encode(self: *const Index, e: mtl.ComputeEncoder, pipe: mtl.Pipeline, out_ref: rig_.Ref, rm: u32, bs: u32) void {
        const q_stride: u32 = hi * di + 64;
        const w_stride: u32 = hi + 16;
        const n: u32 = @intCast(self.rows);
        const most: u32 = @intCast(self.most);
        e.setPipeline(pipe);
        e.setBuffer(self.refs[0].buf, self.refs[0].off, 0);
        e.setBuffer(self.refs[1].buf, self.refs[1].off, 1);
        e.setBuffer(self.refs[2].buf, self.refs[2].off, 2);
        e.setBuffer(out_ref.buf, out_ref.off, 3);
        const args = ScoreArgs{ .p0 = @intCast(self.p0), .q_stride = q_stride, .w_stride = w_stride, .s_stride = @intCast(self.s_stride), .rows = n };
        e.setBytes(std.mem.asBytes(&args), 4);
        const gx = if (self.kind == .scores) (most + 63) / 64 else (most + bs - 1) / bs;
        const gy = if (self.kind == .scores) (n + rm - 1) / rm else n;
        e.dispatchGroups(mtl.Size.of(gx, gy, 1), mtl.Size.of(128, 1, 1));
    }

    pub fn run(self: *Index) !void {
        const p = self.rig.begin();
        const m3 = !self.rig.device.tensorUnits();
        const rm: u32 = if (self.control) (if (m3) 64 else 128) else if (m3) 32 else 64;
        const bs: u32 = if (self.control) (if (m3) 128 else 256) else if (m3) 64 else 128;
        self.encode(p.e, self.pipe, self.out, rm, bs);
        try self.rig.finish(p);
    }

    fn writtenCount(self: *const Index) usize {
        var total: usize = 0;
        for (0..self.rows) |r| total += (self.p0 + r + 1) / 4;
        return total;
    }

    pub fn count(self: *Index) usize {
        return self.writtenCount();
    }

    fn pack(self: *const Index, from: rig_.Ref, dst: *std.ArrayList(u8)) ![]const u8 {
        dst.clearRetainingCapacity();
        const raw = self.rig.bytes(from, self.rows * self.s_stride * 4);
        for (0..self.rows) |r| {
            const b = (self.p0 + r + 1) / 4;
            try dst.appendSlice(self.a, raw[r * self.s_stride * 4 ..][0 .. b * 4]);
        }
        return dst.items;
    }

    fn intact(ctx: *anyopaque) !bool {
        const self: *Index = @ptrCast(@alignCast(ctx));
        if (!try self.rig.intact()) return false;
        const raw = self.rig.bytes(self.out, self.rows * self.s_stride * 4);
        for (0..self.rows) |r| {
            const b = (self.p0 + r + 1) / 4;
            for (raw[(r * self.s_stride + b) * 4 .. (r + 1) * self.s_stride * 4]) |byte| if (byte != 0xff) return false;
        }
        return true;
    }

    fn output(ctx: *anyopaque) ![]const u8 {
        const self: *Index = @ptrCast(@alignCast(ctx));
        if (!try intact(self)) return error.WroteOutsideOutputs;
        return self.pack(self.out, &self.bytes);
    }

    pub fn replaced(self: *Index) ![]const u8 {
        const p = self.rig.begin();
        self.encode(p.e, self.base_pipe, self.base, 64, 128);
        try self.rig.finish(p);
        return self.pack(self.base, &self.base_bytes);
    }

    fn checkNonfinite(ctx: *anyopaque, gate: g.Gate) !?[]const u8 {
        const self: *Index = @ptrCast(@alignCast(ctx));
        const want = try self.replaced();
        const got = try output(self);
        if (!std.mem.eql(u8, got, want)) {
            return if (gate == .nonfinite) "index outputs differ from baseline on nonfinite inputs" else "index outputs differ from baseline";
        }
        return null;
    }

    pub fn reference(_: *Index, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
};

pub const KdaKind = enum { prep, scan, post };

pub const Kda = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    kind: KdaKind,
    heads: u32,
    control: bool,
    name: []const u8,
    pipe: mtl.Pipeline,
    base_pipe: mtl.Pipeline,
    rows: usize = 0,
    r0: usize = 0,
    in_refs: [8]rig_.Ref = undefined,
    out_refs: [5]rig_.Ref = undefined,
    base_refs: [5]rig_.Ref = undefined,
    out_lens: [5]usize = undefined,
    n_outs: usize = 0,
    bytes: std.ArrayList(u8) = .empty,
    base_bytes: std.ArrayList(u8) = .empty,

    fn depth(s: c.Shape) usize {
        _ = s;
        return 0;
    }

    fn kdaServed(comptime w: usize, kind: KdaKind) []const c.Shape {
        return switch (kind) {
            .prep => &.{ .{ .rows = 1, .n = w, .k = 0 }, .{ .rows = 5, .n = w, .k = 4 }, .{ .rows = 16, .n = w, .k = 0 } },
            .scan => &.{ .{ .rows = 1, .n = w, .k = d_kda }, .{ .rows = 24, .n = w, .k = d_kda }, .{ .rows = 25, .n = w, .k = d_kda } },
            .post => &.{ .{ .rows = 1, .n = w, .k = 0 }, .{ .rows = 2, .n = w, .k = 0 }, .{ .rows = 5, .n = w, .k = 0 } },
        };
    }

    pub fn spec(kind: KdaKind, heads: u32, control: bool, name: []const u8) c.Spec {
        const w: usize = @as(usize, heads) * d_kda;
        const served = if (heads == 32) kdaServed(32 * d_kda, kind) else kdaServed(64 * d_kda, kind);
        const tiles: [3]c.Tile = switch (kind) {
            .prep => .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = w, .step = w, .min = w }, .{ .size = 4, .step = 4, .min = 0 } },
            .scan => .{ .{ .size = 24, .step = 1, .min = 1 }, .{ .size = w, .step = w, .min = w }, .{ .size = d_kda, .step = d_kda, .min = d_kda } },
            .post => .{ .{ .size = 4, .step = 1, .min = 1 }, .{ .size = w, .step = w, .min = w }, .{} },
        };
        return .{
            .name = name,
            .accum = .none,
            .depth = depth,
            .out = .raw,
            .replaces = if (control) null else "unvectorized M3 glm_kda_prompt kernel",
            .nan = .{ .stated = "every output bit matches the unvectorized baseline, non-finite inputs included" },
            .tiles = tiles,
            .served = served,
            .sums = .{ .why = "bit-preserving layout optimization; arithmetic and recurrence order match the baseline" },
        };
    }

    const vt = blk: {
        var v = g.hooks(Kda);
        v.output = output;
        v.check = checkNonfinite;
        break :blk v;
    };

    pub fn kernel(self: *Kda) g.Kernel {
        return .{ .spec = spec(self.kind, self.heads, self.control, self.name), .ctx = self, .vt = &vt, .control = if (self.control) .shapes else null };
    }

    pub fn deinit(self: *Kda) void {
        self.rig.deinit();
        self.bytes.deinit(self.a);
        self.base_bytes.deinit(self.a);
    }

    fn allocBfIn(self: *Kda, len: usize, rng: std.Random, scale: f32, idx: usize, nf: bool) !void {
        const buf = try self.a.alloc(u16, len);
        defer self.a.free(buf);
        fillBf(buf, rng, scale);
        if (nf and len > 0) buf[0] = bfVal(std.math.nan(f32));
        if (nf and len > 4) buf[4] = bfVal(std.math.inf(f32));
        if (nf and len > 9) buf[9] = bfVal(-std.math.inf(f32));
        self.in_refs[idx] = try self.rig.input(std.mem.sliceAsBytes(buf));
    }

    fn allocF32In(self: *Kda, len: usize, rng: std.Random, lo: f32, hi_v: f32, idx: usize, nf: bool) !void {
        const buf = try self.a.alloc(f32, len);
        defer self.a.free(buf);
        fillF32(buf, rng, lo, hi_v);
        if (nf and len > 0) buf[0] = std.math.inf(f32);
        if (nf and len > 3) buf[3] = std.math.nan(f32);
        self.in_refs[idx] = try self.rig.input(std.mem.sliceAsBytes(buf));
    }

    pub fn prepare(self: *Kda, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        self.rig.reset();
        const w: usize = @as(usize, self.heads) * d_kda;
        if (shape.rows == 0 or shape.n != w) return false;
        self.rows = shape.rows;
        self.r0 = if (self.kind == .prep) shape.k else 0;
        const c3 = 3 * w;
        const ps = c3 + 2 * d_kda + self.heads;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nf = fill == .nonfinite;
        switch (self.kind) {
            .prep => {
                try self.allocBfIn((self.r0 + self.rows) * ps, rng, 0.5, 0, nf);
                try self.allocBfIn((taps - 1) * c3, rng, 0.5, 1, nf);
                try self.allocF32In(taps * c3, rng, -0.5, 0.5, 2, false);
                try self.allocBfIn(self.rows * w, rng, 0.5, 3, nf);
                try self.allocF32In(self.heads, rng, -1.5, -0.2, 4, nf);
                try self.allocF32In(w, rng, -0.5, 0.5, 5, false);
                self.out_lens[0..5].* = .{ self.rows * w * 2, self.rows * w * 2, self.rows * w * 2, self.rows * w * 4, self.rows * self.heads * 4 };
                self.n_outs = 5;
            },
            .scan => {
                for (0..3) |i| try self.allocBfIn(self.rows * w, rng, 0.4, i, nf);
                try self.allocF32In(self.rows * w, rng, 0.5, 1.0, 3, nf);
                try self.allocF32In(self.rows * self.heads, rng, 0.1, 0.9, 4, nf);
                try self.allocF32In(@as(usize, self.heads) * d_kda * d_kda, rng, -0.5, 0.5, 5, nf);
                self.out_lens[0..2].* = .{ @as(usize, self.heads) * d_kda * d_kda * 4, self.rows * w * 2 };
                self.n_outs = 2;
            },
            .post => {
                try self.allocBfIn(self.rows * w, rng, 0.8, 0, nf);
                try self.allocBfIn(self.rows * w, rng, 0.8, 1, nf);
                try self.allocF32In(d_kda, rng, 0.5, 1.5, 2, nf);
                try self.allocBfIn(self.rows * ps, rng, 0.5, 3, nf);
                try self.allocBfIn((taps - 1) * c3, rng, 0.5, 4, nf);
                self.out_lens[0..2].* = .{ self.rows * w * 2, (taps - 1) * c3 * 2 };
                self.n_outs = 2;
            },
        }
        try allocOutputSet(&self.rig, self.out_lens[0..self.n_outs], self.out_refs[0..self.n_outs], self.base_refs[0..self.n_outs]);
        return true;
    }

    fn encode(self: *const Kda, e: mtl.ComputeEncoder, pipe: mtl.Pipeline, outs: []const rig_.Ref, m3: bool, ctrl: bool) void {
        const w: usize = @as(usize, self.heads) * d_kda;
        const ps: i32 = @intCast(3 * w + 2 * d_kda + self.heads);
        e.setPipeline(pipe);
        switch (self.kind) {
            .prep => {
                const shape = [3]i32{ @intCast(self.rows), ps, @intCast(self.r0) };
                const lb: f32 = -1.0;
                e.setBuffer(self.in_refs[0].buf, self.in_refs[0].off, 0);
                e.setBytes(std.mem.asBytes(&shape), 1);
                for (self.in_refs[1..6], 2..) |r, idx| e.setBuffer(r.buf, r.off, idx);
                e.setBytes(std.mem.asBytes(&lb), 7);
                for (outs[0..5], 8..) |r, idx| e.setBuffer(r.buf, r.off, idx);
                const gy: usize = if (ctrl) self.heads / 8 else self.heads / 4;
                e.dispatchGroups(mtl.Size.of(self.rows, gy, 1), mtl.Size.of(128, 1, 1));
            },
            .scan => {
                const r_val: i32 = @intCast(self.rows);
                for (self.in_refs[0..6], 0..) |r, idx| e.setBuffer(r.buf, r.off, idx);
                e.setBuffer(outs[0].buf, outs[0].off, 6);
                e.setBuffer(outs[1].buf, outs[1].off, 7);
                e.setBytes(std.mem.asBytes(&r_val), 8);
                const sg_cols: usize = if (m3) 16 else 64;
                const th: usize = if (m3) 128 else 512;
                const gx = if (ctrl) (d_kda / sg_cols) / 2 else d_kda / sg_cols;
                e.dispatchGroups(mtl.Size.of(gx, self.heads, 1), mtl.Size.of(th, 1, 1));
            },
            .post => {
                const shape = [2]i32{ @intCast(self.rows), ps };
                const eps: f32 = 1e-5;
                for (self.in_refs[0..3], 0..) |r, idx| e.setBuffer(r.buf, r.off, idx);
                e.setBytes(std.mem.asBytes(&eps), 3);
                e.setBuffer(outs[0].buf, outs[0].off, 4);
                e.setBuffer(self.in_refs[3].buf, self.in_refs[3].off, 5);
                e.setBytes(std.mem.asBytes(&shape), 6);
                e.setBuffer(self.in_refs[4].buf, self.in_refs[4].off, 7);
                e.setBuffer(outs[1].buf, outs[1].off, 8);
                const hb: usize = if (m3) 4 else 1;
                const th: usize = if (m3) 128 else 32;
                const gy = if (ctrl) (self.heads / hb) / 2 else self.heads / hb;
                e.dispatchGroups(mtl.Size.of(self.rows, gy, 1), mtl.Size.of(th, 1, 1));
            },
        }
    }

    pub fn run(self: *Kda) !void {
        const p = self.rig.begin();
        const m3 = !self.rig.device.tensorUnits();
        self.encode(p.e, self.pipe, self.out_refs[0..self.n_outs], m3, self.control);
        try self.rig.finish(p);
    }

    pub fn count(self: *Kda) usize {
        var total: usize = 0;
        for (self.out_lens[0..self.n_outs]) |len| total += len;
        return total;
    }

    fn output(ctx: *anyopaque) ![]const u8 {
        const self: *Kda = @ptrCast(@alignCast(ctx));
        return packRefs(self.a, &self.rig, self.out_refs[0..self.n_outs], self.out_lens[0..self.n_outs], &self.bytes);
    }

    pub fn replaced(self: *Kda) ![]const u8 {
        const p = self.rig.begin();
        self.encode(p.e, self.base_pipe, self.base_refs[0..self.n_outs], false, false);
        try self.rig.finish(p);
        return packRefs(self.a, &self.rig, self.base_refs[0..self.n_outs], self.out_lens[0..self.n_outs], &self.base_bytes);
    }

    fn checkNonfinite(ctx: *anyopaque, gate: g.Gate) !?[]const u8 {
        const self: *Kda = @ptrCast(@alignCast(ctx));
        const want = try self.replaced();
        const got = try output(self);
        if (!std.mem.eql(u8, got, want)) {
            return if (gate == .nonfinite) "kda outputs differ from baseline on nonfinite inputs" else "kda outputs differ from baseline";
        }
        return null;
    }

    pub fn reference(_: *Kda, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }
};

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const text = only orelse return true;
    return std.mem.indexOf(u8, name, text) != null;
}

const kda_specs = [_]struct { kind: KdaKind, heads: u32, fn_name: []const u8, control: bool, name: []const u8 }{
    .{ .kind = .prep, .heads = 64, .fn_name = "glm_kda_prep", .control = false, .name = "metal glm m3 kda_prep" },
    .{ .kind = .prep, .heads = 64, .fn_name = "glm_kda_prep", .control = true, .name = "control: metal glm m3 kda_prep, wrong head tile grid" },
    .{ .kind = .scan, .heads = 64, .fn_name = "glm_kda_scan", .control = false, .name = "metal glm m3 kda_scan" },
    .{ .kind = .scan, .heads = 64, .fn_name = "glm_kda_scan", .control = true, .name = "control: metal glm m3 kda_scan, half value-column grid" },
    .{ .kind = .scan, .heads = 32, .fn_name = "glm_kda_scan_tp", .control = false, .name = "metal glm m3 kda_scan_tp" },
    .{ .kind = .post, .heads = 64, .fn_name = "glm_kda_post", .control = false, .name = "metal glm m3 kda_post" },
    .{ .kind = .post, .heads = 64, .fn_name = "glm_kda_post", .control = true, .name = "control: metal glm m3 kda_post, wrong head tile grid" },
};

const index_specs = [_]struct { kind: IndexKind, fn_name: []const u8, control: bool, name: []const u8 }{
    .{ .kind = .scores, .fn_name = "glm_index_scores_nax", .control = false, .name = "metal glm m3 index_scores_nax" },
    .{ .kind = .scores, .fn_name = "glm_index_scores_nax", .control = true, .name = "control: metal glm m3 index_scores_nax, wrong row tile height" },
    .{ .kind = .decode, .fn_name = "glm_index_decode", .control = false, .name = "metal glm m3 index_decode" },
    .{ .kind = .decode, .fn_name = "glm_index_decode", .control = true, .name = "control: metal glm m3 index_decode, wrong block tile width" },
};

pub const Suite = struct {
    a: std.mem.Allocator,
    pipes: std.ArrayList(mtl.Pipeline) = .empty,
    indices: [index_specs.len]Index = undefined,
    index_made: usize = 0,
    kdas: [kda_specs.len]Kda = undefined,
    kda_made: usize = 0,

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, out: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        var need_idx = false;
        for (index_specs) |e| if (wanted(only, e.name)) {
            need_idx = true;
        };
        var need_kda = false;
        for (kda_specs) |e| if (wanted(only, e.name)) {
            need_kda = true;
        };
        const m3_def = if (!device.tensorUnits()) "#define TF_GLM_M3_FAST 1\n" else "";
        if (need_idx) {
            const idx_raw = try std.mem.concat(a, u8, &.{ m3_def, sources.glm_index_nax });
            defer a.free(idx_raw);
            const idx_src = try tf.frags.source(device, a, idx_raw);
            defer a.free(idx_src);
            const idx_base_src = try tf.frags.source(device, a, sources.glm_index_nax);
            defer a.free(idx_base_src);
            const idx_lib = try mtl.Library.fromSource(device, idx_src, mtl.CompileOptions.mlx());
            defer idx_lib.deinit();
            const base_lib = try mtl.Library.fromSource(device, idx_base_src, mtl.CompileOptions.mlx());
            defer base_lib.deinit();
            for (index_specs) |e| {
                if (!wanted(only, e.name)) continue;
                const pipe = try mtl.Pipeline.init(device, idx_lib, e.fn_name, false);
                try s.pipes.append(a, pipe);
                const base_pipe = try mtl.Pipeline.init(device, base_lib, e.fn_name, false);
                try s.pipes.append(a, base_pipe);
                s.indices[s.index_made] = .{
                    .a = a,
                    .rig = .{ .a = a, .device = device, .queue = queue },
                    .kind = e.kind,
                    .control = e.control,
                    .name = e.name,
                    .pipe = pipe,
                    .base_pipe = base_pipe,
                };
                try out.append(a, s.indices[s.index_made].kernel());
                s.index_made += 1;
            }
        }
        if (need_kda) {
            const kda_prefix = comptime tf.glm.kernels.generatedSource("kda_rows");
            const kda_src = try std.mem.concat(a, u8, &.{ kda_prefix, m3_def, sources.glm_kda_prompt });
            defer a.free(kda_src);
            const kda_base_src = kda_prefix ++ sources.glm_kda_prompt;
            const kda_lib = try mtl.Library.fromSource(device, kda_src, mtl.CompileOptions.mlx());
            defer kda_lib.deinit();
            const base_lib = try mtl.Library.fromSource(device, kda_base_src, mtl.CompileOptions.mlx());
            defer base_lib.deinit();
            for (kda_specs) |e| {
                if (!wanted(only, e.name)) continue;
                const pipe = try mtl.Pipeline.init(device, kda_lib, e.fn_name, false);
                try s.pipes.append(a, pipe);
                const base_pipe = try mtl.Pipeline.init(device, base_lib, e.fn_name, false);
                try s.pipes.append(a, base_pipe);
                s.kdas[s.kda_made] = .{
                    .a = a,
                    .rig = .{ .a = a, .device = device, .queue = queue },
                    .kind = e.kind,
                    .heads = e.heads,
                    .control = e.control,
                    .name = e.name,
                    .pipe = pipe,
                    .base_pipe = base_pipe,
                };
                try out.append(a, s.kdas[s.kda_made].kernel());
                s.kda_made += 1;
            }
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.indices[0..s.index_made]) |*it| it.deinit();
        for (s.kdas[0..s.kda_made]) |*it| it.deinit();
        for (s.pipes.items) |p| p.deinit();
        s.pipes.deinit(s.a);
        s.a.destroy(s);
    }
};
