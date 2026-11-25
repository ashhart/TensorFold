//! SenseNova-Loop BF16 and AWQ W4A16 g128 Metal kernels under Contract Gates 1-5 and row-invariance checks.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const limits_k = @import("metal_limits.zig");
const bf16_k = @import("metal_sensenova_bf16.zig");

const c = contract.spec;
const g = contract.gates;
const rows_in = contract.rows;
const awq = tf.sensenova_loop.awq;
const lgpu = tf.sensenova_loop.linear_gpu;
const records = contract.sensenova_linear_records;

pub const Bf16Variant = bf16_k.Bf16Variant;
pub const Bf16Cfg = bf16_k.Bf16Cfg;
pub const Bf16Probe = bf16_k.Bf16Probe;

fn depthAwq(s: c.Shape) usize {
    return 128 + s.k / 8 + 32;
}

pub const AwqVariant = enum { m5_lane, m5_mv, m5_gemm, m3_mv, m3_qmm, xsum };
pub const AwqCfg = struct { v: AwqVariant, tier: u32 = 1, sk: u32 = 1, pre: bool = false, rb: bool = false, ctl: u8 = 0, name: []const u8 };

fn isM5Variant(v: AwqVariant) bool {
    return v == .m5_lane or v == .m5_mv or v == .m5_gemm;
}

pub const AwqProbe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    xsum_pipe: ?mtl.Pipeline = null,
    cfg: AwqCfg,
    served_buf: [2]c.Shape = undefined,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    x_host: []u16 = &.{},
    q_host: []u4 = &.{},
    sb_host: []awq.Float2 = &.{},
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    sb: rig_.Ref = undefined,
    xs: ?rig_.Ref = null,
    y: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, dev: mtl.Device, q: mtl.Queue, lib: mtl.Library, cfg: AwqCfg) !AwqProbe {
        var buf: [64]u8 = undefined;
        const fun: [:0]const u8 = switch (cfg.v) {
            .m5_lane => try std.fmt.bufPrintSentinel(&buf, "sn_awq_lane_sk{d}_p{d}_r{d}", .{ cfg.sk, @intFromBool(cfg.pre), @intFromBool(cfg.rb) }, 0),
            .m5_mv => try std.fmt.bufPrintSentinel(&buf, "sn_awq_m5_mv_r{d}_sk{d}", .{ cfg.tier, cfg.sk }, 0),
            .m5_gemm => "sn_awq_gemm_m5",
            .m3_mv => try std.fmt.bufPrintSentinel(&buf, "sn_awq_row_mv_r{d}_sk{d}", .{ cfg.tier, cfg.sk }, 0),
            .m3_qmm => "sn_awq_qmm_m3",
            .xsum => "sn_awq_xsum_128",
        };
        const pipe = try mtl.Pipeline.init(dev, lib, fun, false);
        errdefer pipe.deinit();
        const xp = if ((cfg.v == .m5_lane and cfg.pre) or cfg.v == .m5_gemm) try mtl.Pipeline.init(dev, lib, "sn_awq_xsum_128", false) else null;
        return .{ .a = a, .rig = .{ .a = a, .device = dev, .queue = q }, .pipe = pipe, .xsum_pipe = xp, .cfg = cfg };
    }

    pub fn deinit(p: *AwqProbe) void {
        p.a.free(p.x_host);
        p.a.free(p.q_host);
        p.a.free(p.sb_host);
        p.rig.deinit();
        p.pipe.deinit();
        if (p.xsum_pipe) |xp| xp.deinit();
    }

    const vt = g.hooks(AwqProbe);

    pub fn kernel(p: *AwqProbe) g.Kernel {
        const rt: usize = switch (p.cfg.v) {
            .m5_lane => if (p.cfg.rb) 32 else 16,
            .m5_mv, .m3_mv => p.cfg.tier,
            .m5_gemm, .m3_qmm => 64,
            .xsum => 16,
        };
        p.served_buf = switch (p.cfg.v) {
            .m5_lane => .{ .{ .rows = if (p.cfg.rb) 24 else 8, .n = 64, .k = 4096 }, .{ .rows = if (p.cfg.rb) 33 else 16, .n = 128, .k = 12288 } },
            .m5_mv, .m3_mv => .{ .{ .rows = p.cfg.tier, .n = 64, .k = 4096 }, .{ .rows = p.cfg.tier, .n = 512, .k = 12288 } },
            .m5_gemm => .{ .{ .rows = 96, .n = 192, .k = 4096 }, .{ .rows = 65, .n = 40, .k = 512 } },
            .m3_qmm => .{ .{ .rows = 96, .n = 192, .k = 4096 }, .{ .rows = 64, .n = 128, .k = 512 } },
            .xsum => .{ .{ .rows = 8, .n = 1, .k = 4096 }, .{ .rows = 33, .n = 1, .k = 12288 } },
        };
        const nt: c.Tile = if (p.cfg.v == .xsum) .{} else .{ .size = 64, .step = 8, .min = 8 };
        const tiles: [3]c.Tile = .{ .{ .size = rt, .step = 1, .min = 1 }, nt, .{ .size = 256, .step = 128, .min = 128 } };
        return .{
            .spec = .{ .name = p.cfg.name, .accum = .f32, .depth = depthAwq, .out = if (p.cfg.v == .xsum) .f32 else .bf16, .tiles = tiles, .served = &p.served_buf },
            .ctx = p,
            .vt = &vt,
            .control = bf16_k.ctlGate(p.cfg.ctl),
        };
    }

    pub fn prepare(p: *AwqProbe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.k == 0 or s.k % 128 != 0) return false;
        if (p.cfg.v != .xsum and (s.n == 0 or s.n % 8 != 0)) return false;
        if (p.cfg.v == .m5_lane and !p.cfg.rb and s.rows > 16) return false;
        if ((p.cfg.v == .m3_mv or p.cfg.v == .m5_mv) and s.rows > 32) return false;
        p.a.free(p.x_host);
        p.a.free(p.q_host);
        p.a.free(p.sb_host);
        p.q_host = &.{};
        p.sb_host = &.{};
        p.rig.reset();
        p.shape = s;
        const kg = s.k / 128;
        p.x_host = try p.a.alloc(u16, s.rows * s.k);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x4444);
        const rng = prng.random();
        const h = s.k / 2;
        const xmax: u64 = @min(32, ((1 << 24) - 1) / (16 * s.k));
        for (0..s.rows) |r| {
            if (fill == .stress and p.cfg.v == .xsum) {
                for (0..kg) |gi| {
                    const b = r * s.k + gi * 128;
                    @memset(p.x_host[b .. b + 63], c.toBf16(32768.0));
                    p.x_host[b + 63] = c.toBf16(1.0);
                    @memset(p.x_host[b + 64 .. b + 128], c.toBf16(-32768.0));
                }
            } else if (fill == .stress) {
                for (0..h) |i| {
                    const v = 1 + rng.uintLessThan(u64, xmax);
                    p.x_host[r * s.k + i] = c.toBf16(@floatFromInt(v));
                    p.x_host[r * s.k + h + i] = c.toBf16(@floatFromInt(v + @intFromBool(i < 8)));
                }
            } else for (0..s.k) |i| p.x_host[r * s.k + i] = if (i % 29 == 0) 0 else c.toBf16(rng.float(f32) * 4 - 2);
        }
        if (fill == .nonfinite) bf16_k.injectNonFinite(p.x_host, s.rows, s.k);
        p.x = try p.rig.input(std.mem.sliceAsBytes(p.x_host));
        if (p.cfg.v == .xsum) {
            p.y = try p.rig.output(kg * s.rows * 4);
            return true;
        }
        const n8 = s.n / 8;
        p.q_host = try p.a.alloc(u4, s.n * s.k);
        const qw = try p.a.alloc(u32, s.k * n8);
        const qz = try p.a.alloc(u32, kg * n8);
        const dt: awq.ScaleDType = if (seed % 2 == 0) .f16 else .bf16;
        const sc = try p.a.alloc(u8, kg * s.n * 2);
        defer {
            p.a.free(qw);
            p.a.free(qz);
            p.a.free(sc);
        }
        for (0..kg) |gi| for (0..n8) |c8| {
            var zs: [8]u4 = undefined;
            for (0..8) |ci| {
                zs[ci] = if (fill == .stress) 8 else @intCast(1 + rng.uintLessThan(u32, 15));
                const sf: f32 = if (fill == .stress) 1.0 else (rng.float(f32) * 0.02 + 0.002);
                const bits: u16 = if (dt == .f16) @bitCast(@as(f16, @floatCast(sf))) else c.toBf16(sf);
                std.mem.writeInt(u16, sc[2 * (gi * s.n + c8 * 8 + ci) ..][0..2], bits, .little);
            }
            qz[gi * n8 + c8] = awq.packAwqWord(zs);
        };
        for (0..s.k) |ki| for (0..n8) |c8| {
            var cols: [8]u4 = undefined;
            for (0..8) |ci| {
                const ni = c8 * 8 + ci;
                const sq: u4 = if (ki < h) @intCast(9 + (ki * 7 + ni) % 6) else @intCast(7 - ((ki - h) * 7 + ni) % 6);
                cols[ci] = if (fill != .stress) @intCast(rng.uintLessThan(u32, 16)) else sq;
                p.q_host[ni * s.k + ki] = cols[ci];
            }
            qw[ki * n8 + c8] = awq.packAwqWord(cols);
        };
        const part = awq.Part{ .n = s.n, .k = s.k, .qweight = qw, .qzeros = qz, .scales = sc, .scales_dtype = dt };
        const layout: awq.Layout = if (isM5Variant(p.cfg.v)) .m5_tile else .m3_row;
        const pw = try p.a.alloc(u32, awq.weightWords(s.n, s.k, layout));
        defer p.a.free(pw);
        p.sb_host = try p.a.alloc(awq.Float2, awq.metaCount(s.n, s.k));
        try awq.packParts(&.{part}, layout, pw, p.sb_host);
        p.w = try p.rig.input(std.mem.sliceAsBytes(pw));
        p.sb = try p.rig.input(std.mem.sliceAsBytes(p.sb_host));
        if ((p.cfg.v == .m5_lane and p.cfg.pre) or p.cfg.v == .m5_gemm) {
            p.xs = try p.rig.scratch(kg * std.mem.alignForward(usize, s.rows, 16) * 4);
        }
        p.y = try p.rig.output(s.rows * s.n * 2);
        return true;
    }

    pub fn count(p: *AwqProbe) usize {
        return if (p.cfg.v == .xsum) (p.shape.k / 128) * p.shape.rows else p.shape.rows * p.shape.n;
    }

    pub fn run(p: *AwqProbe) !void {
        const s = p.shape;
        const pass = p.rig.begin();
        const e = pass.e;
        if (p.cfg.v == .xsum) {
            e.setPipeline(p.pipe);
            e.setBuffer(p.x.buf, p.x.off, 0);
            e.setValue([4]i32{ @intCast(s.rows), @intCast(s.rows), @intCast(s.k), 0 }, 1);
            e.setBuffer(p.y.buf, p.y.off, 2);
            e.dispatchThreads(mtl.Size.of(p.count(), 1, 1), mtl.Size.of(256, 1, 1));
        } else {
            const stride = std.mem.alignForward(usize, s.rows, 16);
            if (p.xsum_pipe) |xp| {
                e.setPipeline(xp);
                e.setBuffer(p.x.buf, p.x.off, 0);
                e.setValue([4]i32{ @intCast(s.rows), @intCast(stride), @intCast(s.k), 0 }, 1);
                e.setBuffer(p.xs.?.buf, p.xs.?.off, 2);
                e.dispatchThreads(mtl.Size.of(stride * (s.k / 128), 1, 1), mtl.Size.of(256, 1, 1));
                e.barrier();
            }
            const step: usize = if (p.cfg.v == .m3_mv or p.cfg.v == .m5_mv) p.cfg.tier else s.rows;
            var r0: usize = 0;
            while (r0 < s.rows) : (r0 += step) {
                const rm: i32 = @intCast(@min(step, s.rows - r0));
                e.setPipeline(p.pipe);
                e.setBuffer(p.x.buf, p.x.off + r0 * s.k * 2, 0);
                e.setBuffer(p.w.buf, p.w.off, 1);
                e.setBuffer(p.sb.buf, p.sb.off, 2);
                e.setValue(lgpu.AwqArgs{ .m = rm, .n = @intCast(s.n), .k = @intCast(s.k), .mp = @intCast(stride), .ldy = @intCast(s.n) }, 3);
                e.setBuffer(p.y.buf, p.y.off + r0 * s.n * 2, 4);
                if (p.xs) |xs_ref| e.setBuffer(xs_ref.buf, xs_ref.off, 5);
                switch (p.cfg.v) {
                    .m5_lane => e.dispatchGroups(mtl.Size.of((s.n + 31) / 32, if (p.cfg.rb) (s.rows + 15) / 16 else 1, 1), mtl.Size.of(p.cfg.sk * 32, 1, 1)),
                    .m5_mv => e.dispatchGroups(mtl.Size.of((s.n + 7) / 8, 1, 1), mtl.Size.of(p.cfg.sk * 32, 1, 1)),
                    .m5_gemm => e.dispatchGroups(mtl.Size.of(((s.n + 127) / 128) * 4, (((s.rows + 63) / 64) + 3) / 4, 1), mtl.Size.of(256, 1, 1)),
                    .m3_mv => blk: {
                        const no = (8 / p.cfg.sk) * lgpu.awqM3MvCols(p.cfg.tier);
                        e.dispatchGroups(mtl.Size.of((s.n + no - 1) / no, 1, 1), mtl.Size.of(256, 1, 1));
                        break :blk;
                    },
                    .m3_qmm => e.dispatchGroups(mtl.Size.of((s.n + 63) / 64, (s.rows + 63) / 64, 1), mtl.Size.of(128, 1, 1)),
                    .xsum => unreachable,
                }
            }
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *AwqProbe, want: []f64, mag: []f64) !void {
        const s = p.shape;
        const kg = s.k / 128;
        if (p.cfg.v == .xsum) {
            for (0..kg) |gi| for (0..s.rows) |r| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..128) |i| {
                    const v = @as(f64, c.fromBf16(p.x_host[r * s.k + gi * 128 + i]));
                    sum += v;
                    abs += @abs(v);
                }
                want[gi * s.rows + r] = sum;
                mag[gi * s.rows + r] = abs;
            };
            return;
        }
        const m5_math = isM5Variant(p.cfg.v);
        for (0..s.rows) |r| for (0..s.n) |j| {
            var y: f64 = 0;
            var a: f64 = 0;
            for (0..kg) |gi| {
                const sb = if (m5_math) p.sb_host[gi * s.n + j] else p.sb_host[j * kg + gi];
                var qx: f64 = 0;
                var xs: f64 = 0;
                for (gi * 128..(gi + 1) * 128) |ki| {
                    const x = @as(f64, c.fromBf16(p.x_host[r * s.k + ki]));
                    const q: f64 = @floatFromInt(p.q_host[j * s.k + ki]);
                    const wv = q * @as(f64, sb.scale) + @as(f64, sb.bias);
                    qx += if (m5_math) q * x else wv * x;
                    xs += x;
                    a += if (m5_math) @abs(sb.scale) * @abs(q * x) + @abs(sb.bias) * @abs(x) else @abs(wv * x);
                }
                y += if (m5_math) @as(f64, sb.scale) * qx + @as(f64, sb.bias) * xs else qx;
            }
            want[r * s.n + j] = y;
            mag[r * s.n + j] = a;
        };
    }

    pub fn simulate(p: *AwqProbe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const s = p.shape;
        const kg = s.k / 128;
        if (p.cfg.v == .xsum) {
            var t128: [128]f64 = undefined;
            for (0..kg) |gi| for (0..s.rows) |r| {
                for (0..128) |i| t128[i] = c.fromBf16(p.x_host[r * s.k + gi * 128 + i]);
                out[gi * s.rows + r] = rows_in.sumIn(acc, order, &t128);
            };
            return;
        }
        const m5_math = isM5Variant(p.cfg.v);
        const buf = try p.a.alloc(f64, 2 * s.k + kg);
        defer p.a.free(buf);
        const t = buf[0 .. 2 * s.k];
        const gs = buf[2 * s.k ..];
        for (0..s.rows) |r| for (0..s.n) |j| {
            for (0..s.k) |ki| {
                const x = @as(f64, c.fromBf16(p.x_host[r * s.k + ki]));
                const q: f64 = @floatFromInt(p.q_host[j * s.k + ki]);
                const sb = if (m5_math) p.sb_host[(ki / 128) * s.n + j] else p.sb_host[j * kg + ki / 128];
                t[ki] = if (m5_math) q * x else (q * @as(f64, sb.scale) + @as(f64, sb.bias)) * x;
                t[s.k + ki] = x;
            }
            if (m5_math) {
                for (gs, 0..) |*gv, gi| {
                    const sb = p.sb_host[gi * s.n + j];
                    gv.* = rows_in.sumIn(acc, .sequential, t[gi * 128 ..][0..128]) * @as(f64, sb.scale) +
                        rows_in.sumIn(acc, .sequential, t[s.k + gi * 128 ..][0..128]) * @as(f64, sb.bias);
                }
                out[r * s.n + j] = rows_in.sumIn(acc, order, gs);
            } else out[r * s.n + j] = rows_in.sumIn(acc, order, t[0..s.k]);
        };
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    libs: [2][5]?mtl.Library = @splat(@splat(null)),
    bf16_list: std.ArrayList(*Bf16Probe) = .empty,
    awq_list: std.ArrayList(*AwqProbe) = .empty,

    fn freeProbes(a: std.mem.Allocator, list: anytype) void {
        for (list.items) |p| {
            p.deinit();
            a.destroy(p);
        }
        list.deinit(a);
    }

    pub fn deinit(s: *Suite) void {
        freeProbes(s.a, &s.bf16_list);
        freeProbes(s.a, &s.awq_list);
        for (&s.libs) |*row| for (row) |*opt| if (opt.*) |lib| lib.deinit();
    }

    fn getLib(s: *Suite, is_awq: bool, ctl: u8) !mtl.Library {
        const slot = &s.libs[@intFromBool(is_awq)][ctl];
        if (slot.*) |lib| return lib;
        const prefix = try std.fmt.allocPrint(s.a, "#define TF_CTL {d}\n", .{ctl});
        defer s.a.free(prefix);
        const raw = if (is_awq) @import("kernel_sources").qwen27_awq_w4a16 else @import("kernel_sources").qwen27_bf16_linear;
        const body = try tf.frags.source(s.device, s.a, raw);
        defer s.a.free(body);
        const full = try std.mem.concat(s.a, u8, &.{ prefix, body });
        defer s.a.free(full);
        slot.* = try mtl.Library.fromSource(s.device, full, mtl.CompileOptions.mlx());
        return slot.*.?;
    }

    fn addProbe(s: *Suite, comptime T: type, list: *std.ArrayList(*T), comptime is_awq: bool, only: ?[]const u8, cfg: if (is_awq) AwqCfg else Bf16Cfg) !void {
        if (is_awq and ((isM5Variant(cfg.v) and !s.device.tensorUnits()) or (cfg.v == .m3_qmm and s.device.tensorUnits()))) return;
        if (only) |needle| if (std.mem.indexOf(u8, cfg.name, needle) == null) return;
        const p = try s.a.create(T);
        errdefer s.a.destroy(p);
        p.* = try T.init(s.a, s.device, s.queue, try s.getLib(is_awq, cfg.ctl), cfg);
        errdefer p.deinit();
        try @import("contract").register(s.a, &list, &s.kernels, p, p.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]Bf16Cfg{
            .{ .v = .mv, .tier = 1, .sk = 1, .name = "sensenova linear sn_bf16_mv_r1_sk1" },
            .{ .v = .mv, .tier = 4, .sk = 2, .name = "sensenova linear sn_bf16_mv_r4_sk2" },
            .{ .v = .mv, .tier = 8, .sk = 4, .name = "sensenova linear sn_bf16_mv_r8_sk4" },
            .{ .v = .mv, .tier = 32, .sk = 8, .name = "sensenova linear sn_bf16_mv_r32_sk8" },
            .{ .v = .gemm, .tier = 64, .sk = 1, .name = "sensenova linear sn_bf16_gemm_00" },
            .{ .v = .splitk, .tier = 64, .sk = 2, .name = "sensenova linear sn_bf16_gemm_splitk_00" },
        }) |cs| try s.addProbe(Bf16Probe, &s.bf16_list, false, only, cs);
        const ctags = [_][]const u8{ "nonfinite->0", "repeat", "tail skip", "fp16 sum" };
        inline for ([_]u8{ 1, 2, 3, 4 }, ctags) |ctl, tag| {
            try s.addProbe(Bf16Probe, &s.bf16_list, false, only, .{ .v = .mv, .tier = 8, .sk = 2, .ctl = ctl, .name = "control: sensenova bf16_mv " ++ tag });
            if (ctl != 2) try s.addProbe(Bf16Probe, &s.bf16_list, false, only, .{ .v = .gemm, .tier = 64, .ctl = ctl, .name = "control: sensenova bf16_gemm " ++ tag });
            try s.addProbe(AwqProbe, &s.awq_list, true, only, .{ .v = .m5_lane, .tier = 16, .sk = 4, .pre = true, .ctl = ctl, .name = "control: sensenova awq_lane " ++ tag });
            try s.addProbe(AwqProbe, &s.awq_list, true, only, .{ .v = .m5_mv, .tier = 8, .sk = 2, .ctl = ctl, .name = "control: sensenova awq_m5_mv " ++ tag });
            if (ctl != 2) try s.addProbe(AwqProbe, &s.awq_list, true, only, .{ .v = .m5_gemm, .tier = 64, .ctl = ctl, .name = "control: sensenova awq_gemm_m5 " ++ tag });
            try s.addProbe(AwqProbe, &s.awq_list, true, only, .{ .v = .m3_mv, .tier = 8, .sk = 2, .ctl = ctl, .name = "control: sensenova awq_row_mv " ++ tag });
            if (ctl != 2) try s.addProbe(AwqProbe, &s.awq_list, true, only, .{ .v = .m3_qmm, .tier = 64, .ctl = ctl, .name = "control: sensenova awq_qmm " ++ tag });
        }
        for ([_]AwqCfg{
            .{ .v = .m5_lane, .tier = 16, .sk = 1, .name = "sensenova awq-m5 sn_awq_lane_sk1_p0_r0" },
            .{ .v = .m5_lane, .tier = 16, .sk = 4, .pre = true, .name = "sensenova awq-m5 sn_awq_lane_sk4_p1_r0" },
            .{ .v = .m5_lane, .tier = 32, .sk = 8, .pre = true, .rb = true, .name = "sensenova awq-m5 sn_awq_lane_sk8_p1_r1" },
            .{ .v = .m5_mv, .tier = 1, .sk = 1, .name = "sensenova awq-m5 sn_awq_m5_mv_r1_sk1" },
            .{ .v = .m5_mv, .tier = 4, .sk = 2, .name = "sensenova awq-m5 sn_awq_m5_mv_r4_sk2" },
            .{ .v = .m5_mv, .tier = 8, .sk = 4, .name = "sensenova awq-m5 sn_awq_m5_mv_r8_sk4" },
            .{ .v = .m5_mv, .tier = 32, .sk = 8, .name = "sensenova awq-m5 sn_awq_m5_mv_r32_sk8" },
            .{ .v = .m5_gemm, .tier = 64, .sk = 1, .name = "sensenova awq-m5 sn_awq_gemm_m5" },
            .{ .v = .m3_mv, .tier = 1, .sk = 1, .name = "sensenova awq sn_awq_row_mv_r1_sk1" },
            .{ .v = .m3_mv, .tier = 4, .sk = 2, .name = "sensenova awq sn_awq_row_mv_r4_sk2" },
            .{ .v = .m3_mv, .tier = 8, .sk = 4, .name = "sensenova awq sn_awq_row_mv_r8_sk4" },
            .{ .v = .m3_mv, .tier = 32, .sk = 8, .name = "sensenova awq sn_awq_row_mv_r32_sk8" },
            .{ .v = .m3_qmm, .tier = 64, .sk = 1, .name = "sensenova awq sn_awq_qmm_m3" },
            .{ .v = .xsum, .tier = 16, .name = "sensenova awq sn_awq_xsum_128" },
        }) |cs| try s.addProbe(AwqProbe, &s.awq_list, true, only, cs);
    }

    fn checkInv(s: *Suite, comptime P: type, lib: mtl.Library, v: anytype, sk: u32) !void {
        const n: usize = 256;
        const k: usize = 1024;
        const pre = if (P == AwqProbe) (v == .m5_lane or v == .m5_gemm) else false;
        const is_m5_m = if (P == AwqProbe) isM5Variant(v) else false;
        const tier7: u32 = if (v == .m5_lane) 16 else if (v == .m5_gemm) 64 else 8;
        const c7 = if (P == Bf16Probe) Bf16Cfg{ .v = .mv, .tier = 8, .sk = sk, .name = "i" } else AwqCfg{ .v = v, .tier = tier7, .sk = sk, .pre = pre, .name = "i" };
        var p7 = try P.init(s.a, s.device, s.queue, lib, c7);
        defer p7.deinit();
        _ = try p7.prepare(.{ .rows = 7, .n = n, .k = k }, .random, 99 + sk);
        try p7.run();
        const ref = try s.a.dupe(u8, p7.rig.bytes(p7.y, 7 * n * 2));
        defer s.a.free(ref);
        for ([_]usize{ 1, 3 }) |rows| {
            const tier: u32 = if (v == .m5_lane) 16 else if (v == .m5_gemm) 64 else if (rows == 1) 1 else 4;
            const c1 = if (P == Bf16Probe) Bf16Cfg{ .v = .mv, .tier = tier, .sk = sk, .name = "s" } else AwqCfg{ .v = v, .tier = tier, .sk = sk, .pre = pre, .name = "s" };
            var p1 = try P.init(s.a, s.device, s.queue, lib, c1);
            defer p1.deinit();
            _ = try p1.prepare(.{ .rows = rows, .n = n, .k = k }, .random, 99 + sk);
            @memcpy(p1.rig.bytes(p1.x, rows * k * 2), p7.rig.bytes(p7.x, rows * k * 2));
            if (P == Bf16Probe) {
                @memcpy(p1.rig.bytes(p1.w, n * k * 2), p7.rig.bytes(p7.w, n * k * 2));
            } else {
                const wb = awq.weightWords(n, k, if (is_m5_m) .m5_tile else .m3_row) * 4;
                @memcpy(p1.rig.bytes(p1.w, wb), p7.rig.bytes(p7.w, wb));
                @memcpy(p1.rig.bytes(p1.sb, awq.metaCount(n, k) * 8), p7.rig.bytes(p7.sb, awq.metaCount(n, k) * 8));
            }
            try p1.run();
            if (!std.mem.eql(u8, ref[0 .. rows * n * 2], p1.rig.bytes(p1.y, rows * n * 2))) return error.RowInvarianceFailed;
        }
    }

    fn checkEncodeAwqInv(s: *Suite) !void {
        var gpu = lgpu.LinearGpu.init(s.a, s.device);
        defer gpu.deinit();
        try gpu.reserve(64, true);
        const n: u32 = 256;
        const k: u32 = 1024;
        const opts = mtl.ResourceOptions.shared;
        const w_buf = try s.device.buffer(@as(usize, n) * k * 2, opts);
        defer w_buf.deinit();
        const x_buf = try s.device.buffer(@as(usize, 33) * k * 2, opts);
        defer x_buf.deinit();
        const y_ref = try s.device.buffer(@as(usize, 33) * n * 2, opts);
        defer y_ref.deinit();
        const y_cur = try s.device.buffer(@as(usize, 16) * n * 2, opts);
        defer y_cur.deinit();
        var prng = std.Random.DefaultPrng.init(0x3333);
        for (w_buf.slice(u16, @as(usize, n) * k)) |*d| d.* = c.toBf16((prng.random().float(f32) * 2.0 - 1.0) * 0.25);
        for (x_buf.slice(u16, @as(usize, 33) * k)) |*d| d.* = c.toBf16((prng.random().float(f32) * 2.0 - 1.0) * 1.5);
        var mat = try bf16_k.packBf16AsAwq(&gpu, w_buf.slice(u16, @as(usize, n) * k), n, k);
        defer mat.deinit();
        for ([_]u32{ 33, 1, 4, 8, 16 }) |rows| {
            const dst = if (rows == 33) y_ref else y_cur;
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = s.queue.commandBuffer();
            const e = cb.compute(.serial);
            try gpu.encodeAwq(e, mat, .{ .buf = x_buf }, null, .{ .buf = dst }, rows, n, false, null);
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |_| return error.GpuFailed;
            if (rows != 33 and !std.mem.eql(u16, y_ref.slice(u16, rows * n), y_cur.slice(u16, rows * n))) {
                return error.RowInvarianceFailed;
            }
        }
    }

    /// Gate 3 row-invariance: 1, 3, 7 rows per launch produce identical row bytes to single-row launches.
    pub fn rowCountInvariant(s: *Suite) !void {
        const bf_lib = try s.getLib(false, 0);
        const awq_lib = try s.getLib(true, 0);
        for ([_]u32{ 1, 2, 4, 8 }) |sk| {
            try s.checkInv(Bf16Probe, bf_lib, .mv, sk);
            try s.checkInv(AwqProbe, awq_lib, AwqVariant.m3_mv, sk);
            if (s.device.tensorUnits()) {
                try s.checkInv(AwqProbe, awq_lib, AwqVariant.m5_mv, sk);
                try s.checkInv(AwqProbe, awq_lib, AwqVariant.m5_lane, sk);
            }
        }
        if (s.device.tensorUnits()) try s.checkInv(AwqProbe, awq_lib, AwqVariant.m5_gemm, 1);
        try s.checkEncodeAwqInv();
        try bf16_k.checkBf16Split(s.a, s.device, s.queue);
        try bf16_k.checkAwqSplitAndQuant(s.a, s.device, s.queue);
        std.debug.print("PASS row-invariance sensenova bf16_mv, awq_row_mv{s}, encodeAwq (1,4,8,16,33), awq split + quant\n", .{if (s.device.tensorUnits()) ", awq_m5_mv, awq_lane, awq_gemm_m5" else ""});
    }
};

pub fn limits(a: std.mem.Allocator, device: mtl.Device, only: ?[]const u8) !usize {
    const tensor = device.tensorUnits();
    var list: std.ArrayList(contract.limits.MetalLaunch) = .empty;
    defer list.deinit(a);
    for (records.records) |d| {
        if (only) |needle| if (std.mem.indexOf(u8, d.name, needle) == null) continue;
        if (records.isM5Only(d.name) and !tensor) continue;
        try list.append(a, d);
    }
    var probs = try contract.registry.checkMetal(a, list.items, false);
    const bf_src = try tf.frags.source(device, a, @import("kernel_sources").qwen27_bf16_linear);
    defer a.free(bf_src);
    const awq_src = try tf.frags.source(device, a, @import("kernel_sources").qwen27_awq_w4a16);
    defer a.free(awq_src);
    for (list.items) |d| {
        const fun = d.name[std.mem.lastIndexOfScalar(u8, d.name, ' ').? + 1 ..];
        const src = if (std.mem.startsWith(u8, d.name, "sensenova linear ")) bf_src else awq_src;
        probs += try limits_k.onThisGpu(a, device, .{ .decl = d, .function = fun, .here = src, .pre_m5 = src }, false);
    }
    const bad = contract.limits.MetalLaunch{ .name = "control: sensenova 40 KiB tg", .threads = 256, .tg_memory = 40 * 1024 };
    if (try contract.registry.checkMetal(a, &.{bad}, true) == 0) probs += 1;
    return probs;
}
