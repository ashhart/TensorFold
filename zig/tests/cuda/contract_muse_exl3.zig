//! Muse Spark's EXL3 launch forms as replacements of the shared kernels: each must equal their bytes on any input.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const mu = @import("contract_muse_util.zig");

const c = core.contract.spec;
const g = core.contract.gates;
const shared = cuda.exl3;

const rule = "as the shared kernel it replaces gives them: a non-finite input reaches its row through the rotation";

fn noDepth(_: c.Shape) usize {
    return 0;
}

/// muse_linear2_k{K2}_w{WK} (32-row passes) for tf_exl3_linear_{K2}_2_{WK}; muse_linear_multi_k{K2} for 4 of them.
pub const LinEntry = struct { k2: u32, wk: u32, multi: bool };
pub const lin_entries = [_]LinEntry{
    .{ .k2 = 4, .wk = 8, .multi = false }, .{ .k2 = 4, .wk = 4, .multi = false }, .{ .k2 = 6, .wk = 4, .multi = false },
    .{ .k2 = 3, .wk = 8, .multi = false }, .{ .k2 = 4, .wk = 4, .multi = true },  .{ .k2 = 6, .wk = 4, .multi = true },
};

fn linName(e: LinEntry) []const u8 {
    inline for (lin_entries) |x| if (x.k2 == e.k2 and x.wk == e.wk and x.multi == e.multi) {
        return if (x.multi) std.fmt.comptimePrint("cuda muse muse_linear_multi_k{d} (q, k, v, gate in one launch)", .{x.k2}) else std.fmt.comptimePrint("cuda muse muse_linear2_k{d}_w{d} (32-row passes)", .{ x.k2, x.wk });
    };
    unreachable;
}

/// No sums declared (gate 4 N/A): bit-equal to the replaced kernel's sums; rotated inputs cannot hold stress integers.
pub fn linSpec(e: LinEntry) c.Spec {
    // the served shapes that plan to this entry's warps (exl3.plan): Muse Spark's o, pre / post and the head slice
    const wide = &[_]c.Shape{ .{ .rows = 32, .n = 8192, .k = 8192 }, .{ .rows = 17, .n = 8192, .k = 4096 }, .{ .rows = 128, .n = 16384, .k = 8192 } };
    const narrow = &[_]c.Shape{ .{ .rows = 32, .n = 2048, .k = 8192 }, .{ .rows = 24, .n = 4096, .k = 8192 }, .{ .rows = 128, .n = 512, .k = 8192 } };
    const qkvg = &[_]c.Shape{ .{ .rows = 8, .n = 512, .k = 8192 }, .{ .rows = 1, .n = 512, .k = 8192 }, .{ .rows = 16, .n = 1024, .k = 8192 } };
    return .{
        .name = linName(e),
        .accum = .none,
        .depth = noDepth,
        .out = if (e.multi) .f16 else .f32,
        .replaces = if (e.multi) "tf_exl3_linear on each of the four" else "tf_exl3_linear",
        .nan = .{ .stated = rule },
        .tiles = .{ .{ .size = if (e.multi) 16 else 32, .step = 1, .min = 1 }, .{ .size = 128, .step = 128, .min = 128 }, .{} },
        .served = if (e.multi) qkvg else if (e.wk == 8) wide else narrow,
    };
}

/// Random EXL3 weights for one layer: trellis words (any bits decode to a finite fp16), scales near 1.
fn layer(rig: *rig_.Rig, a: std.mem.Allocator, rng: std.Random, k: usize, n: usize, k2: u32) !shared.Weight {
    const tw = 4 * @as(usize, k2);
    const words = try a.alloc(u32, (k / 16) * (n / 16) * tw);
    defer a.free(words);
    for (words) |*w| w.* = rng.int(u32);
    const sc = try a.alloc(f16, k + n);
    defer a.free(sc);
    for (sc) |*v| v.* = @floatCast(0.5 + rng.float(f32));
    const st = shared.strides(k, n, k2, true);
    return .{ .trellis = try rig.input(std.mem.sliceAsBytes(words)), .suh = try rig.input(std.mem.sliceAsBytes(sc[0..k])), .svh = try rig.input(std.mem.sliceAsBytes(sc[k..])), .k = @intCast(k), .n = @intCast(n), .halves = k2, .codebook = .mul1, .stride_k = st.k, .stride_nb = st.nb };
}

pub const Lin = struct {
    rig: rig_.Rig,
    entry: LinEntry,
    lin: *const shared.Linear,
    f: cuda.Function,
    rot: cuda.Function,
    w: [4]shared.Weight = undefined,
    jobs: usize = 1,
    m: usize = 0,
    x: u64 = 0,
    xh: u64 = 0,
    z: u64 = 0,
    counters: u64 = 0,
    y: u64 = 0,
    base: std.ArrayList(u8) = .empty,
    msg: [160]u8 = undefined,

    pub fn init(rig: rig_.Rig, e: LinEntry, mod: cuda.Module, lin: *const shared.Linear) !Lin {
        const f = if (e.multi) switch (e.k2) {
            inline 3, 4, 6 => |k2| try mod.function(std.fmt.comptimePrint("muse_linear_multi_k{d}", .{k2})),
            else => return error.Unregistered,
        } else switch (e.k2 * 16 + e.wk) {
            inline 3 * 16 + 8, 4 * 16 + 8, 4 * 16 + 4, 6 * 16 + 4 => |v| try mod.function(std.fmt.comptimePrint("muse_linear2_k{d}_w{d}", .{ v / 16, v % 16 })),
            else => return error.Unregistered,
        };
        return .{ .rig = rig, .entry = e, .lin = lin, .f = f, .rot = try mod.function("muse_rot_multi") };
    }

    pub fn deinit(self: *Lin) void {
        self.base.deinit(self.rig.gpu.gpa);
        self.rig.deinit();
    }

    const vt = g.hooks(Lin);

    pub fn kernel(self: *Lin) g.Kernel {
        return .{ .spec = linSpec(self.entry), .ctx = self, .vt = &vt };
    }

    fn ysize(self: *const Lin) usize {
        return self.m * self.w[0].n * (if (self.entry.multi) @as(usize, 2) else 4);
    }

    pub fn prepare(self: *Lin, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.rows > 128 or shape.k % 128 != 0 or shape.n % 128 != 0 or shape.k == 0 or shape.n == 0) return false;
        const p = shared.plan(shape.k, shape.n, 192, 8);
        if (p.wk != self.entry.wk or (shape.k / 16) % (p.sk * p.wk) != 0) return false;
        self.rig.reset();
        const a = self.rig.gpu.gpa;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        self.jobs = if (self.entry.multi) 4 else 1;
        self.m = shape.rows;
        for (0..self.jobs) |j| self.w[j] = try layer(&self.rig, a, rng, shape.k, shape.n, self.entry.k2);
        const x = try a.alloc(f16, shape.rows * shape.k);
        defer a.free(x);
        for (x) |*v| v.* = @floatCast(rng.float(f32) * 2 - 1);
        if (fill == .nonfinite) {
            x[0] = std.math.nan(f16);
            if (shape.rows > 2) x[2 * shape.k + 5] = std.math.inf(f16);
            if (shape.rows > 3) x[3 * shape.k + shape.k - 1] = -std.math.inf(f16);
        }
        self.x = try self.rig.input(std.mem.sliceAsBytes(x));
        self.xh = try self.rig.scratch(4 * shape.rows * shape.k * 2);
        self.z = try self.rig.scratch(@max(4 * @as(usize, p.sk) * shape.rows * shape.n * 4, 4));
        self.counters = try self.rig.scratch(4 * shared.Linear.counterBytes(self.w[0]));
        self.y = try self.rig.output(self.jobs * self.ysize());
        return true;
    }

    /// The family's launches: ops.qkvg's rot_multi + linear_multi, or ops.proj's rot_in + linear2.
    pub fn run(self: *Lin) !void {
        const s = self.rig.s;
        const w0 = self.w[0];
        const p = shared.plan(w0.k, w0.n, 192, 8);
        const smem: u32 = @intCast(p.wk * @min(self.m, 8) * 128 * 4);
        if (self.entry.multi) {
            const Job = extern struct { xh: u64, t: u64, stride_k: i64, stride_nb: i64, svh: u64, bias: u64, y: u64, z: u64, counters: u64, y_dtype: c_int, k: c_int, n: c_int, sk: c_int, b0: c_int, pad: c_int = 0 };
            const Batch = extern struct { j: [4]Job, n: c_int, m: c_int, pad: [2]c_int = .{ 0, 0 } };
            const Rot = extern struct { suh: [4]u64, xh: [4]u64 };
            var rot: Rot = undefined;
            var bt: Batch = .{ .j = undefined, .n = 4, .m = @intCast(self.m) };
            const zr = @as(usize, p.sk) * self.m * w0.n * 4;
            const cr = shared.Linear.counterBytes(w0);
            for (0..4) |j| {
                const w = self.w[j];
                rot.suh[j] = w.suh;
                rot.xh[j] = self.xh + j * self.m * w.k * 2;
                bt.j[j] = .{ .xh = rot.xh[j], .t = w.trellis, .stride_k = @intCast(w.stride_k), .stride_nb = @intCast(w.stride_nb), .svh = w.svh, .bias = 0, .y = self.y + j * self.ysize(), .z = if (p.sk > 1) self.z + j * zr else 0, .counters = self.counters + j * cr, .y_dtype = 0, .k = @intCast(w.k), .n = @intCast(w.n), .sk = @intCast(p.sk), .b0 = @intCast(j * (w.n / 128) * p.sk) };
            }
            var ra: cuda.Args = .{};
            ra.add(self.x);
            ra.add(@as(c_int, 0));
            ra.add(rot);
            ra.add(@as(c_int, @intCast(w0.k)));
            try cuda.launch.launch(self.rot, .{ .grid = .{ .x = @intCast((w0.k / 128 + 3) / 4), .y = @intCast(self.m), .z = 4 }, .block = .{ .x = 128 } }, s, &ra);
            var la: cuda.Args = .{};
            la.add(bt);
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(4 * (w0.n / 128) * p.sk), .y = 1, .z = 1 }, .block = .{ .x = 128 }, .shared = smem }, s, &la);
        } else {
            var ra: cuda.Args = .{};
            inline for (.{ self.x, w0.suh, self.xh }) |v| ra.add(v);
            inline for (.{ 0, self.m, w0.k }) |v| ra.add(@as(i32, @intCast(v)));
            try cuda.launch.launch(self.lin.rot_in, .{ .grid = .{ .x = @intCast((w0.k / 128 + 3) / 4), .y = @intCast(self.m), .z = 1 }, .block = .{ .x = 128 } }, s, &ra);
            var la: cuda.Args = .{};
            inline for (.{ self.xh, w0.trellis }) |v| la.add(v);
            la.add(@as(i64, @intCast(w0.stride_k)));
            la.add(@as(i64, @intCast(w0.stride_nb)));
            inline for (.{ w0.svh, @as(u64, 0), self.y }) |v| la.add(v);
            la.add(@as(c_int, 2));
            la.add(if (p.sk > 1) self.z else @as(u64, 0));
            la.add(self.counters);
            inline for (.{ self.m, w0.k, w0.n, p.sk }) |v| la.add(@as(c_int, @intCast(v)));
            try cuda.launch.launch(self.f, .{ .grid = .{ .x = @intCast(w0.n / 128), .y = p.sk, .z = 1 }, .block = .{ .x = p.wk * 32 }, .shared = smem }, s, &la);
        }
        try s.synchronize();
    }

    pub fn count(self: *Lin) usize {
        return self.jobs * self.m * self.w[0].n;
    }

    /// Never asked: a replacement's gates 1 and 3 compare bytes (check, replaced), and it declares no sums for gate 4.
    pub fn reference(_: *Lin, _: []f64, _: []f64) !void {
        return error.ReplacementHasNoFloat64Reference;
    }

    /// The shared Linear.matmul on the same inputs, each job into its region.
    pub fn replaced(self: *Lin) ![]const u8 {
        const a = self.rig.gpu.gpa;
        const bytes = self.jobs * self.ysize();
        var tmp = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, bytes);
        defer tmp.free();
        var z = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, @as(usize, shared.plan(self.w[0].k, self.w[0].n, 192, 8).sk) * self.m * self.w[0].n * 4 + 4);
        defer z.free();
        var cn = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, shared.Linear.counterBytes(self.w[0]));
        defer cn.free();
        try cn.fill8(0, self.rig.s.handle);
        var xh = try cuda.DeviceBuffer.alloc(self.rig.gpu.d, self.m * self.w[0].k * 2);
        defer xh.free();
        for (0..self.jobs) |j| try self.lin.matmul(self.rig.s, self.w[j], self.x, .f16, self.m, tmp.ptr + j * self.ysize(), if (self.entry.multi) .f16 else .f32, xh.ptr, z.ptr, cn.ptr);
        try self.rig.s.synchronize();
        try self.base.resize(a, bytes);
        try tmp.download(0, self.base.items);
        return self.base.items;
    }

    /// Gate 1: tf_exl3_linear's bytes on the planted inputs, some of them non-finite.
    pub fn check(self: *Lin, at: g.Gate) !?[]const u8 {
        const a = self.rig.gpu.gpa;
        const got = try a.dupe(u8, try self.rig.outputBytes());
        defer a.free(got);
        return mu.sameBytes(&self.msg, got, try self.replaced(), "tf_exl3_linear", linSpec(self.entry).out, at);
    }
};

/// muse_grouped2 (32-member tiles) for tf_exl3_grouped_2: gate / up (2 matrices, sk 4) and down (1, sk 1).
pub const GrpEntry = struct { mats: u32, k2s: [2]u32 };
pub const grp_entries = [_]GrpEntry{ .{ .mats = 2, .k2s = .{ 4, 4 } }, .{ .mats = 1, .k2s = .{ 3, 4 } }, .{ .mats = 2, .k2s = .{ 6, 3 } } };

pub fn grpSpec(e: GrpEntry) c.Spec {
    const names = [_][]const u8{ "cuda muse muse_grouped2 (gate / up, 2 bits)", "cuda muse muse_grouped2 (down, 1.5 and 2 bits)", "cuda muse muse_grouped2 (gate / up, 3 and 1.5 bits)" };
    const i = for (grp_entries, 0..) |x, j| {
        if (x.mats == e.mats and x.k2s[0] == e.k2s[0] and x.k2s[1] == e.k2s[1]) break j;
    } else unreachable;
    const gu = &[_]c.Shape{ .{ .rows = 64, .n = 2048, .k = 4096 }, .{ .rows = 17, .n = 2048, .k = 4096 } };
    const dn = &[_]c.Shape{ .{ .rows = 64, .n = 4096, .k = 2048 }, .{ .rows = 33, .n = 4096, .k = 2048 } };
    return .{ .name = names[i], .accum = .none, .depth = noDepth, .out = .f32, .replaces = "tf_exl3_grouped_2", .nan = .{ .stated = rule }, .tiles = .{ .{ .size = 32, .step = 1, .min = 1 }, .{}, .{} }, .served = if (e.mats == 2) gu else dn };
}

pub const Grp = @import("contract_muse_grouped.zig").Grp;
