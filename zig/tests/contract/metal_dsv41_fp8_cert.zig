//! DeepSeek V4.1 opt-in FP64-certified BF16 dense FP8 kernels under the contract: prompt, decode, tile, equality.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_fp8_ref.zig");

const c = contract.spec;
const g = contract.gates;
const mma = tf.dsv41_metal.fp8_mma_cert;
const rows_k = tf.dsv41_metal.fp8_rows;
const tile_k = tf.dsv41_metal.fp8_tile;
const base = "dsv41_fp8_cert ";

pub const Path = enum { mma, rows, tile };
pub const Variant = struct { path: Path, out: mma.Out = .bf16, fixture: bool = false, ctl: u8 = 0 };

const served = [_]c.Shape{
    .{ .rows = 16, .n = 1280, .k = 5120 },
    .{ .rows = 1, .n = 512, .k = 5120 },
    .{ .rows = 4, .n = 512, .k = 8192 },
    .{ .rows = 8, .n = 5120, .k = 2304 },
    .{ .rows = 16, .n = 128, .k = 512 },
    .{ .rows = 16, .n = 256, .k = 1280 },
    .{ .rows = 7, .n = 1024, .k = 4096 },
    .{ .rows = 8, .n = 256, .k = 6144 },
    .{ .rows = 1, .n = 64, .k = 1536 },
    .{ .rows = 9, .n = 64, .k = 1536 },
};

/// Direct round-to-nearest-even from f64 to BF16 without double rounding through f32.
fn f64ToBf16(x: f64) u16 {
    if (std.math.isNan(x)) return 0x7fc0;
    const f: f32 = @floatCast(x);
    if (!std.math.isFinite(f)) return c.toBf16(f);
    const u: u32 = @bitCast(f);
    if ((u & 0xffff) == 0x8000) {
        const diff = x - @as(f64, f);
        if (diff != 0) {
            const same_sign = (diff < 0) == ((u >> 31) != 0);
            const trunc: u16 = @intCast(u >> 16);
            return if (same_sign) trunc +% 1 else trunc;
        }
    }
    const lsb: u32 = (u >> 16) & 1;
    return @intCast((u +% 0x7fff +% lsb) >> 16);
}

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = .f32,
        .depth = ref.depth,
        .out = if (v.out == .f32) .f32 else .bf16,
        .tiles = if (v.fixture) .{ .{}, .{}, .{} } else .{ .{ .size = 16 }, .{ .size = 32, .step = 32, .min = 32 }, .{ .size = 256, .step = 32, .min = 32 } },
        .served = if (v.fixture) &.{ref.fixture.shape} else &served,
    };
}

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    mma_p: ?mma.Pipelines = null,
    rows_p: ?rows_k.Pipelines = null,
    tile_p: ?tile_k.Pipelines = null,
    f: ?ref.Fp8 = null,
    fill: g.Fill = .random,
    real_fixture: bool = false,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var p: Proj = .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name };
        const src = switch (v.path) {
            .mma => sources.dsv41.fp8_mma_cert,
            .rows => sources.dsv41.fp8_rows_cert,
            .tile => sources.dsv41.fp8_tile_cert,
        };
        const text = if (v.ctl == 0) null else try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, src });
        defer if (text) |t| a.free(t);
        const body = text orelse src;
        switch (v.path) {
            .mma => p.mma_p = try mma.Pipelines.loadSource(a, device, body),
            .rows => p.rows_p = try rows_k.Pipelines.loadSource(device, body),
            .tile => p.tile_p = try tile_k.Pipelines.loadSource(device, body),
        }
        return p;
    }

    pub fn deinit(p: *Proj) void {
        p.drop();
        if (p.mma_p) |*q| q.deinit();
        if (p.rows_p) |*q| q.deinit();
        if (p.tile_p) |*q| q.deinit();
        p.rig.deinit();
    }

    fn drop(p: *Proj) void {
        if (p.f) |f| f.deinit();
        p.f = null;
        p.rig.reset();
        if (p.mma_p) |*q| q.clearNormCache();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        const control: ?g.Gate = switch (p.v.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        };
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = control };
    }

    const vt = g.hooks(Proj);

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        const k: u32 = @intCast(shape.k);
        const n: u32 = @intCast(shape.n);
        const r: u32 = @intCast(shape.rows);
        switch (p.v.path) {
            .mma => mma.check(.{ .k = k, .n = n, .rows = r, .out = p.v.out }) catch return false,
            .rows => rows_k.check(.{ .k = k, .n = n, .rows = r, .split = rows_k.splitOf(k) }) catch return false,
            .tile => if (!tile_k.fits(.{ .k = k, .n = n, .rows = r })) return false,
        }
        const f = try ref.Fp8.init(p.a, shape);
        p.f = f;
        p.fill = fill;
        p.real_fixture = p.v.fixture and fill == .random and std.meta.eql(shape, ref.fixture.shape);
        if (p.real_fixture) try ref.fixture.load(f, null) else f.fill(fill, seed);
        const bytes = shape.rows * shape.n * c.size(spec(p.v, p.name).out);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        p.w = try p.rig.input(f.codes);
        p.s = try p.rig.input(f.scales);
        p.y = try p.rig.output(bytes);
        return true;
    }

    pub fn run(p: *Proj) !void {
        const f = p.f.?;
        const k: u32 = @intCast(f.shape.k);
        const n: u32 = @intCast(f.shape.n);
        const r: u32 = @intCast(f.shape.rows);
        const pass = p.rig.begin();
        switch (p.v.path) {
            .mma => try mma.encode(&p.mma_p.?, pass.e, .{ .buf = p.x.buf, .off = p.x.off }, .{ .buf = p.w.buf, .off = p.w.off }, .{ .buf = p.s.buf, .off = p.s.off }, .{ .buf = p.y.buf, .off = p.y.off }, .{ .k = k, .n = n, .rows = r, .out = p.v.out }),
            .rows => try rows_k.encodeWhole(&p.rows_p.?, pass.e, .{ .buf = p.x.buf, .off = p.x.off }, .{ .buf = p.w.buf, .off = p.w.off }, .{ .buf = p.s.buf, .off = p.s.off }, .{ .buf = p.y.buf, .off = p.y.off }, .{ .k = k, .n = n, .rows = r, .split = rows_k.splitOf(k), .out = if (p.v.out == .f32) .f32 else .bf16 }, .{}),
            .tile => try tile_k.encode(&p.tile_p.?, pass.e, .{ .buf = p.x.buf, .off = p.x.off }, .{ .buf = p.w.buf, .off = p.w.off }, .{ .buf = p.s.buf, .off = p.s.off }, .{ .buf = p.y.buf, .off = p.y.off }, .{ .k = k, .n = n, .rows = r, .out = if (p.v.out == .f32) .f32 else .bf16 }, .{}),
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.f.?.outputs();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        const f = p.f.?;
        if (p.real_fixture) {
            try ref.npyF64(ref.fixture.ref_npy, want);
            if (p.v.out == .bf16) for (want) |*w| {
                w.* = c.fromBf16(f64ToBf16(w.*));
            };
            f.magnitudes(mag);
            return;
        }
        const leaves = try p.a.alloc(f32, ref.levelsOf(f.shape.k / 32));
        defer p.a.free(leaves);
        for (0..f.shape.rows) |r| for (0..f.shape.n) |j| {
            const o = f.one(r, j, leaves);
            const at = r * f.shape.n + j;
            want[at] = if (p.v.out == .bf16) c.fromBf16(f64ToBf16(o.exact)) else @as(f64, @as(f32, @floatCast(o.exact)));
            mag[at] = o.mag;
        };
    }

    /// Certified BF16 outputs must match independent FP64 at 0.0 ULP on every input (including stress and fixture).
    pub fn bounds(p: *Proj, _: []const f64, _: []const f64, allow: []f64) !void {
        if (p.v.out == .bf16) @memset(allow, 0);
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        return p.f.?.simulate(acc, order, out);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Proj) = .empty,

    const list = [_]struct { v: Variant, name: []const u8 }{
        .{ .v = .{ .path = .mma, .out = .bf16 }, .name = base ++ "mma bf16" },
        .{ .v = .{ .path = .mma, .out = .f32 }, .name = base ++ "mma f32" },
        .{ .v = .{ .path = .mma, .out = .bf16, .fixture = true }, .name = base ++ "mma release wkv fixture" },
        .{ .v = .{ .path = .mma, .out = .bf16, .ctl = 1 }, .name = base ++ "mma control: non-finite inputs flushed" },
        .{ .v = .{ .path = .mma, .out = .bf16, .ctl = 2 }, .name = base ++ "mma control: last K group dropped" },
        .{ .v = .{ .path = .mma, .out = .bf16, .ctl = 3 }, .name = base ++ "mma control: group sums rounded to fp16" },
        .{ .v = .{ .path = .mma, .out = .bf16, .ctl = 4 }, .name = base ++ "mma control: writes past its output" },
        .{ .v = .{ .path = .rows, .out = .bf16 }, .name = base ++ "rows bf16" },
        .{ .v = .{ .path = .rows, .out = .f32 }, .name = base ++ "rows f32" },
        .{ .v = .{ .path = .rows, .out = .bf16, .ctl = 1 }, .name = base ++ "rows control: non-finite inputs flushed" },
        .{ .v = .{ .path = .rows, .out = .bf16, .ctl = 2 }, .name = base ++ "rows control: last K group dropped" },
        .{ .v = .{ .path = .rows, .out = .bf16, .ctl = 3 }, .name = base ++ "rows control: group sums rounded to fp16" },
        .{ .v = .{ .path = .rows, .out = .bf16, .ctl = 4 }, .name = base ++ "rows control: writes past its output" },
        .{ .v = .{ .path = .tile, .out = .bf16 }, .name = base ++ "tile bf16" },
        .{ .v = .{ .path = .tile, .out = .f32 }, .name = base ++ "tile f32" },
        .{ .v = .{ .path = .tile, .out = .bf16, .ctl = 1 }, .name = base ++ "tile control: non-finite inputs flushed" },
        .{ .v = .{ .path = .tile, .out = .bf16, .ctl = 2 }, .name = base ++ "tile control: last K group dropped" },
        .{ .v = .{ .path = .tile, .out = .bf16, .ctl = 3 }, .name = base ++ "tile control: group sums rounded to fp16" },
        .{ .v = .{ .path = .tile, .out = .bf16, .ctl = 4 }, .name = base ++ "tile control: writes past its output" },
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a, .device = device, .queue = queue };
        if (!mma.supported(device)) return s;
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            try s.probes.append(a, p);
            try kernels.append(a, p.kernel());
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    /// Prompt (mma), decode (rows) and verify tile (tile) produce byte-equal BF16 outputs across rows and NaN/Inf.
    pub fn equality(s: *Suite, seed: u64) !usize {
        if (!mma.supported(s.device) or s.probes.items.len == 0) return 0;
        var mp = try mma.Pipelines.load(s.a, s.device);
        defer mp.deinit();
        var rp = try rows_k.Pipelines.loadSource(s.device, sources.dsv41.fp8_rows_cert);
        defer rp.deinit();
        var tp = try tile_k.Pipelines.loadSource(s.device, sources.dsv41.fp8_tile_cert);
        defer tp.deinit();
        var rig: rig_.Rig = .{ .a = s.a, .device = s.device, .queue = s.queue };
        defer rig.deinit();
        var bad: usize = 0;
        var launches: usize = 0;
        const shapes = [_][2]u32{ .{ 5120, 1280 }, .{ 5120, 512 }, .{ 4096, 1024 }, .{ 8192, 512 }, .{ 2304, 1280 }, .{ 6144, 256 } };
        for (shapes) |kn| for ([_]g.Fill{ .random, .nonfinite }) |fill| {
            rig.reset();
            mp.clearNormCache();
            const k = kn[0];
            const n = kn[1];
            const rows: u32 = 16;
            const f = try ref.Fp8.init(s.a, .{ .rows = rows, .n = n, .k = k });
            defer f.deinit();
            f.fill(fill, seed);
            const bytes = @as(usize, rows) * n * 2;
            const x = try rig.input(std.mem.sliceAsBytes(f.x));
            const w = try rig.input(f.codes);
            const sc = try rig.input(f.scales);
            const ym = try rig.output(bytes);
            const yr = try rig.output(bytes);
            const yt = try rig.output(bytes);
            const pass = rig.begin();
            try mma.encode(&mp, pass.e, .{ .buf = x.buf, .off = x.off }, .{ .buf = w.buf, .off = w.off }, .{ .buf = sc.buf, .off = sc.off }, .{ .buf = ym.buf, .off = ym.off }, .{ .k = k, .n = n, .rows = rows, .out = .bf16 });
            try rows_k.encodeWhole(&rp, pass.e, .{ .buf = x.buf, .off = x.off }, .{ .buf = w.buf, .off = w.off }, .{ .buf = sc.buf, .off = sc.off }, .{ .buf = yr.buf, .off = yr.off }, .{ .k = k, .n = n, .rows = rows, .split = rows_k.splitOf(k), .out = .bf16 }, .{});
            try tile_k.encode(&tp, pass.e, .{ .buf = x.buf, .off = x.off }, .{ .buf = w.buf, .off = w.off }, .{ .buf = sc.buf, .off = sc.off }, .{ .buf = yt.buf, .off = yt.off }, .{ .k = k, .n = n, .rows = rows, .out = .bf16 }, .{});
            try rig.finish(pass);
            launches += 3;
            const bm = rig.bytes(ym, bytes);
            const br = rig.bytes(yr, bytes);
            const bt = rig.bytes(yt, bytes);
            if (!bf16Equal(bm, br) or !bf16Equal(bm, bt)) bad += 1;
        };
        std.debug.print("{s} {s}prompt/decode/tile equality: {d} shapes x random/nonfinite ({d} launches), {d} differ\n", .{ if (bad == 0) "PASS" else "FAIL", base, shapes.len, launches, bad });
        return bad;
    }
};

fn bf16Equal(a: []const u8, b: []const u8) bool {
    const au: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, a));
    const bu: []const u16 = @alignCast(std.mem.bytesAsSlice(u16, b));
    for (au, bu) |x, y| {
        const both_nan = (x & 0x7fff) > 0x7f80 and (y & 0x7fff) > 0x7f80;
        if (x != y and !both_nan) return false;
    }
    return true;
}
