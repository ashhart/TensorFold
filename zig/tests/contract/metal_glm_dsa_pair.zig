//! Full GLM-5.3's fp32 pair kernels: full-width epochs, rank-order residuals, RMS and vocabulary ties.
const std = @import("std");
const mtl = @import("metal");
const ct = @import("contract");
const c = ct.spec;
const g = ct.gates;
const rig = @import("rig.zig");
const source = @import("kernel_sources").core_pair;
const max_rows = @import("tensorfold").glm_dsa.gpu.prompt_rows;
const width = 6144;
const vocab = 77440;
const epoch: u64 = 0x1_0000_0003;
const stale: u64 = 3;
const eps: f32 = 1e-5;
const absent = std.math.maxInt(u32);

pub const Kind = enum {
    pair_stage,
    pair_add_rms,
    pair_argmax,
    pair_pick,

    fn threads(k: Kind) u32 {
        return switch (k) {
            .pair_stage, .pair_add_rms => 256,
            .pair_argmax => 1024,
            .pair_pick => 32,
        };
    }

    fn memory(k: Kind) usize {
        return switch (k) {
            .pair_add_rms => 32,
            .pair_argmax => 256,
            else => 0,
        };
    }

    fn spec(k: Kind, label: []const u8) c.Spec {
        return .{
            .name = label,
            .accum = if (k == .pair_add_rms) .f32 else .none,
            .depth = depth,
            .ops = if (k == .pair_add_rms) 8 else 0,
            .out = if (k == .pair_add_rms) .f32 else .raw,
            .served = if (k == .pair_argmax) &head_shapes else &row_shapes,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
            .nan = .{ .stated = switch (k) {
                .pair_stage => "copies all bits; a stale previous epoch leaves the send slot unchanged",
                .pair_add_rms => "IEEE residual and RMS classes; a stale epoch zeros both residual and normalized rows",
                .pair_argmax => "NaN never wins; greater values win, lower ids break ties, all minus infinity has no id",
                .pair_pick => "larger peer value or lower peer id wins; a stale epoch writes safe token zero",
            } },
            .sums = .{ .f16 = 10, .bf16 = 10 },
        };
    }
};

const row_shapes = [_]c.Shape{
    .{ .rows = 1, .n = width, .k = width },
    .{ .rows = 16, .n = width, .k = width },
    .{ .rows = 32, .n = width, .k = width },
    .{ .rows = 256, .n = width, .k = width },
    .{ .rows = max_rows, .n = width, .k = width },
};
const head_shapes = [_]c.Shape{
    .{ .rows = 1, .n = vocab, .k = 1 },
    .{ .rows = 16, .n = vocab, .k = 1 },
};

fn depth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 5 + 8 + 2;
}

pub fn name(comptime kind: Kind) []const u8 {
    return "metal glm dsa pair " ++ @tagName(kind);
}

fn caseName(comptime kind: Kind, control: bool) []const u8 {
    const normal = comptime name(kind);
    return if (control) "control: " ++ normal ++ " planted guard or tie" else normal;
}

const Args = extern struct {
    width: u32 = width,
    stride: u32 = width,
    rank: u32,
    x: u32 = 0,
    eps: f32 = eps,
    pad: [3]f32 = @splat(0),
};

fn floats(r: *const rig.Rig, ref: rig.Ref, n: usize) []f32 {
    return @ptrCast(@alignCast(r.bytes(ref, n * 4)));
}

fn words(r: *const rig.Rig, ref: rig.Ref, n: usize) []u32 {
    return @ptrCast(@alignCast(r.bytes(ref, n * 4)));
}

const Probe = struct {
    rig: rig.Rig,
    pipe: mtl.Pipeline,
    kind: Kind,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    seed: u64 = 0,
    ready: [2]rig.Ref = undefined,
    data: [3]rig.Ref = undefined,
    weight: rig.Ref = undefined,
    h: [2]rig.Ref = undefined,
    out: [2]rig.Ref = undefined,
    const vt = g.hooks(Probe);

    fn kernel(p: *Probe, label: []const u8, control: bool) g.Kernel {
        return .{ .spec = p.kind.spec(label), .ctx = p, .vt = &vt, .control = if (control) .shapes else null };
    }

    fn elements(p: *const Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    fn value(p: *const Probe, part: usize, i: usize) f32 {
        if (p.fill == .stress) {
            if (part != 0) return 0;
            return @floatFromInt(@as(i32, @intCast(std.hash.int(i +% p.seed) % 65)) - 32);
        }
        if (p.fill == .nonfinite and part == 0 and i % width == 7 and i / width < 4)
            return ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32) })[i / width];
        if (p.fill == .nonfinite and part == 1 and i == 3 * width + 7) return -std.math.inf(f32);
        const h = std.hash.int(i +% p.seed +% (part + 1) *% 0x9e3779b9);
        return @as(f32, @floatFromInt(@as(i32, @as(u16, @truncate(h))) - 32768)) / 8192;
    }

    fn weightAt(p: *const Probe, j: usize) f32 {
        return if (p.fill == .stress) 1 else c.fromBf16(c.toBf16(0.75 + @as(f32, @floatFromInt(j % 17)) / 32));
    }

    fn logit(p: *const Probe, r: usize, j: usize) f32 {
        if (p.fill == .nonfinite) {
            if (r == 3) return std.math.nan(f32);
            if (r == 4) return -std.math.inf(f32);
            if (j == 0) return ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) })[r % 3];
        }
        if (j == 31 or j == 32 or j == 1055 or j == 2047) return 32;
        return p.value(0, r * p.shape.n + j);
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > max_rows) return false;
        if (shape.n != (if (p.kind == .pair_argmax) @as(usize, vocab) else width)) return false;
        p.rig.reset();
        p.shape = shape;
        p.fill = fill;
        p.seed = seed;
        p.ready[0] = try p.rig.input(std.mem.asBytes(&epoch));
        p.ready[1] = try p.rig.input(std.mem.asBytes(&stale));
        if (p.kind == .pair_pick) {
            const payload = try p.rig.a.alloc(u32, 2 * shape.rows);
            defer p.rig.a.free(payload);
            for (0..2) |rank| {
                for (0..shape.rows) |r| {
                    const v: f32 = if (fill == .nonfinite and r < 3)
                        ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) })[r]
                    else
                        @floatFromInt((r + rank * (r % 3)) % 5);
                    payload[2 * r] = @bitCast(v);
                    payload[2 * r + 1] = @intCast(rank * vocab + r);
                }
                p.data[rank] = try p.rig.input(std.mem.sliceAsBytes(payload));
            }
            p.out[0] = try p.rig.output(shape.rows * 4);
            return true;
        }
        const data = try p.rig.a.alloc(f32, p.elements());
        defer p.rig.a.free(data);
        if (p.kind == .pair_argmax) {
            for (data, 0..) |*v, i| v.* = p.logit(i / shape.n, i % shape.n);
            p.data[0] = try p.rig.input(std.mem.sliceAsBytes(data));
            p.out[0] = try p.rig.output(shape.rows * 8);
        } else {
            for (0..(if (p.kind == .pair_stage) @as(usize, 1) else 3)) |part| {
                for (data, 0..) |*v, i| v.* = p.value(part, i);
                p.data[part] = try p.rig.input(std.mem.sliceAsBytes(data));
            }
            if (p.kind == .pair_stage) {
                p.out[0] = try p.rig.output(p.elements() * 4);
            } else {
                const w = try p.rig.a.alloc(u16, width);
                defer p.rig.a.free(w);
                for (w, 0..) |*v, j| v.* = c.toBf16(p.weightAt(j));
                p.weight = try p.rig.input(std.mem.sliceAsBytes(w));
                for (0..2) |rank| {
                    p.h[rank] = try p.rig.output(p.elements() * 4);
                    p.out[rank] = try p.rig.output(p.elements() * 4);
                }
            }
        }
        return true;
    }

    fn dispatch(p: *Probe, valid: bool) !void {
        const ready = p.ready[@intFromBool(!valid)];
        if (p.kind == .pair_add_rms) {
            for (0..2) |rank| {
                @memcpy(p.rig.bytes(p.h[rank], p.elements() * 4), p.rig.bytes(p.data[2], p.elements() * 4));
                @memset(p.rig.bytes(p.out[rank], p.elements() * 4), 0xff);
            }
        } else @memset(p.rig.bytes(p.out[0], p.outBytes()), 0xff);
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        switch (p.kind) {
            .pair_stage => {
                e.setBuffer(ready.buf, ready.off, 0);
                e.setValue(epoch, 1);
                e.setBuffer(p.data[0].buf, p.data[0].off, 2);
                e.setBuffer(p.out[0].buf, p.out[0].off, 3);
                e.setValue(@as(u32, @intCast(p.elements())), 4);
                e.dispatchThreads(.of(p.elements(), 1, 1), .of(256, 1, 1));
            },
            .pair_add_rms => for (0..2) |rank| {
                e.setBuffer(ready.buf, ready.off, 0);
                e.setValue(epoch, 1);
                e.setValue(Args{ .rank = @intCast(rank) }, 2);
                e.setBuffer(p.data[rank].buf, p.data[rank].off, 3);
                e.setBuffer(p.data[1 - rank].buf, p.data[1 - rank].off, 4);
                e.setBuffer(p.h[rank].buf, p.h[rank].off, 5);
                e.setBuffer(p.weight.buf, p.weight.off, 6);
                e.setBuffer(p.out[rank].buf, p.out[rank].off, 7);
                e.dispatchGroups(.of(p.shape.rows, 1, 1), .of(256, 1, 1));
            },
            .pair_argmax => {
                e.setBuffer(p.data[0].buf, p.data[0].off, 0);
                e.setValue([4]u32{ @intCast(p.shape.n), vocab, @intCast(p.shape.n), 0 }, 1);
                e.setBuffer(p.out[0].buf, p.out[0].off, 2);
                e.dispatchGroups(.of(p.shape.rows, 1, 1), .of(1024, 1, 1));
            },
            .pair_pick => {
                e.setBuffer(ready.buf, ready.off, 0);
                e.setValue(epoch, 1);
                e.setValue(@as(u32, @intCast(p.shape.rows)), 2);
                e.setBuffer(p.data[0].buf, p.data[0].off, 3);
                e.setBuffer(p.data[1].buf, p.data[1].off, 4);
                e.setBuffer(p.out[0].buf, p.out[0].off, 5);
                e.dispatchGroups(.of(1, 1, 1), .of(32, 1, 1));
            },
        }
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        return p.dispatch(true);
    }

    fn outBytes(p: *const Probe) usize {
        return switch (p.kind) {
            .pair_stage, .pair_add_rms => p.elements() * 4,
            .pair_argmax => p.shape.rows * 8,
            .pair_pick => p.shape.rows * 4,
        };
    }

    pub fn count(p: *const Probe) usize {
        return if (p.kind == .pair_add_rms) 4 * p.elements() else p.outBytes();
    }

    fn residual(p: *const Probe, i: usize) f64 {
        return @as(f64, p.value(2, i)) + (@as(f64, p.value(0, i)) + p.value(1, i));
    }

    fn evaluate(p: *Probe, acc: c.Accum, order: g.Order, want: []f64, mag: []f64) !void {
        if (p.kind != .pair_add_rms) {
            @memset(want, 0);
            @memset(mag, 0);
            return;
        }
        var terms: [width]f64 = undefined;
        const n = p.elements();
        for (0..p.shape.rows) |r| {
            for (&terms, 0..) |*t, j| {
                const x = p.residual(r * width + j);
                t.* = x * x;
            }
            const sum = ct.rows.sumIn(acc, order, &terms);
            const inv = 1 / @sqrt(sum / width + eps);
            for (0..width) |j| {
                const i = r * width + j;
                const x = p.residual(i);
                for (0..2) |rank| {
                    want[rank * 2 * n + i] = x;
                    want[(rank * 2 + 1) * n + i] = p.weightAt(j) * (x * inv);
                    mag[rank * 2 * n + i] = @abs(x);
                    mag[(rank * 2 + 1) * n + i] = @abs(want[(rank * 2 + 1) * n + i]);
                }
            }
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        return p.evaluate(.none, .sequential, want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.rig.a.alloc(f64, out.len);
        defer p.rig.a.free(mag);
        return p.evaluate(acc, order, out, mag);
    }

    fn residualBound(p: *const Probe, i: usize) f64 {
        if (p.fill == .stress) return 0;
        return c.gamma(2, c.unit(.f32)) * (@abs(@as(f64, p.value(0, i))) + @abs(@as(f64, p.value(1, i))) + @abs(@as(f64, p.value(2, i))));
    }

    pub fn bounds(p: *Probe, want: []const f64, _: []const f64, allow: []f64) !void {
        if (p.kind != .pair_add_rms) {
            @memset(allow, 0);
            return;
        }
        const n = p.elements();
        for (0..p.shape.rows) |r| {
            var q: f64 = 0;
            var input_error: f64 = 0;
            var upper_q: f64 = 0;
            for (0..width) |j| {
                const i = r * width + j;
                const x = @abs(p.residual(i));
                const b = p.residualBound(i);
                q += x * x;
                input_error += 2 * x * b + b * b;
                upper_q += (x + b) * (x + b);
            }
            const reduction = if (p.fill == .stress) 0 else c.gamma(@floatFromInt(depth(p.shape)), c.unit(.f32)) * upper_q;
            const d = q / width + eps;
            const dd = (input_error + reduction) / width + c.gamma(2, c.unit(.f32)) * d;
            const lo = 1 / @sqrt(d + dd);
            const hi = 1 / @sqrt(d - dd);
            const center = 1 / @sqrt(d);
            for (0..width) |j| {
                const i = r * width + j;
                const b = p.residualBound(i);
                const propagated = @abs(@as(f64, p.weightAt(j))) * (b * hi + @abs(p.residual(i)) * @max(hi - center, center - lo));
                for (0..2) |rank| {
                    allow[rank * 2 * n + i] = b + 0x1p-149;
                    const at = (rank * 2 + 1) * n + i;
                    allow[at] = propagated + 8 * c.unit(.f32) * @abs(want[at]) + 0x1p-149;
                }
            }
        }
    }

    fn checkRms(p: *Probe, valid: bool) !?[]const u8 {
        const n = p.elements();
        if (!valid) {
            for (0..2) |rank| for (0..n) |i| {
                if (floats(&p.rig, p.h[rank], n)[i] != 0 or floats(&p.rig, p.out[rank], n)[i] != 0)
                    return "a stale full-width epoch consumed a residual or peer payload";
            };
            return null;
        }
        const values = try p.rig.a.alloc(f64, 3 * p.count());
        defer p.rig.a.free(values);
        const want = values[0..p.count()];
        const mag = values[p.count() .. 2 * p.count()];
        const allow = values[2 * p.count() ..];
        try p.reference(want, mag);
        if (p.fill != .nonfinite) try p.bounds(want, mag, allow);
        for (0..2) |rank| for (0..n) |i| {
            const h = floats(&p.rig, p.h[rank], n)[i];
            const y = floats(&p.rig, p.out[rank], n)[i];
            const at = (rank * 2 + 1) * n + i;
            const ordered: f32 = p.value(2, i) + (p.value(0, i) + p.value(1, i));
            if (p.fill == .nonfinite) {
                if (c.classOf(h) != c.classOf(ordered) or c.classOf(y) != c.classOf(want[at]))
                    return "residual or RMS non-finite classes differ from independent float64";
            } else {
                if (@as(u32, @bitCast(h)) != @as(u32, @bitCast(ordered))) return "residual differs from rank-ordered fp32";
                if (!(@abs(@as(f64, y) - want[at]) <= allow[at])) return "RMS exceeds its propagated float64 bound";
            }
        };
        return null;
    }

    fn compare(p: *Probe, valid: bool) !?[]const u8 {
        if (!try p.rig.intact()) return "a launch overwrote a guard";
        switch (p.kind) {
            .pair_add_rms => return p.checkRms(valid),
            .pair_stage => {
                const got = p.rig.bytes(p.out[0], p.outBytes());
                if (!valid) return if (std.mem.allEqual(u8, got, 0xff)) null else "stale previous epoch changed send bytes";
                if (!std.mem.eql(u8, got, p.rig.bytes(p.data[0], p.outBytes()))) return "stage changed payload bits";
            },
            .pair_argmax => for (0..p.shape.rows) |r| {
                var best: f32 = -std.math.inf(f32);
                var id: u32 = absent;
                for (0..p.shape.n) |j| {
                    const v = p.logit(r, j);
                    if (v > best) {
                        best = v;
                        id = vocab + @as(u32, @intCast(j));
                    }
                }
                const got = words(&p.rig, p.out[0], 2 * p.shape.rows);
                if (got[2 * r] != @as(u32, @bitCast(best)) or got[2 * r + 1] != id) return "argmax value or lower-id tie differs";
            },
            .pair_pick => for (0..p.shape.rows) |r| {
                const own = words(&p.rig, p.data[0], 2 * p.shape.rows);
                const peer = words(&p.rig, p.data[1], 2 * p.shape.rows);
                const vm: f32 = @bitCast(own[2 * r]);
                const vp: f32 = @bitCast(peer[2 * r]);
                const am = own[2 * r + 1];
                const ap = peer[2 * r + 1];
                const expected = if (!valid) 0 else if (vp > vm or (vp == vm and ap < am)) ap else am;
                if (words(&p.rig, p.out[0], p.shape.rows)[r] != expected) return "pick differs from value/tie or stale sentinel rule";
            },
        }
        return null;
    }

    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        if (try p.compare(true)) |why| return why;
        if (p.kind != .pair_argmax) {
            try p.dispatch(false);
            if (try p.compare(false)) |why| return why;
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: [8]Probe = undefined,
    len: usize = 0,

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        var libs: [2]?mtl.Library = @splat(null);
        defer for (libs) |lib| if (lib) |l| l.deinit();
        inline for (std.enums.values(Kind)) |kind| {
            for ([_]bool{ false, true }, &libs) |control, *lib| {
                const label = caseName(kind, control);
                if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
                if (lib.* == null) lib.* = try s.library(control);
                const pipe = try mtl.Pipeline.init(s.device, lib.*.?, @tagName(kind), false);
                errdefer pipe.deinit();
                if (pipe.maxThreads() < kind.threads() or rig.tgMemory(pipe) != kind.memory()) return error.UnexpectedLaunchLimits;
                try kernels.ensureUnusedCapacity(s.a, 1);
                s.probes[s.len] = .{ .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .pipe = pipe, .kind = kind };
                kernels.appendAssumeCapacity(s.probes[s.len].kernel(label, control));
                s.len += 1;
            }
        }
    }

    fn library(s: *const Suite, control: bool) !mtl.Library {
        if (!control) return mtl.Library.fromSource(s.device, source, mtl.CompileOptions.mlx());
        if (std.mem.count(u8, source, "ready[0] != previous") != 1 or std.mem.count(u8, source, "ready[0] != expected") != 2 or
            std.mem.count(u8, source, "ob == best && oa < at") != 2) return error.ControlEditMissing;
        const a = try std.mem.replaceOwned(u8, s.a, source, "ready[0] != previous", "uint(ready[0]) != uint(previous)");
        defer s.a.free(a);
        const b = try std.mem.replaceOwned(u8, s.a, a, "ready[0] != expected", "uint(ready[0]) != uint(expected)");
        defer s.a.free(b);
        const d = try std.mem.replaceOwned(u8, s.a, b, "ob == best && oa < at", "ob == best && oa > at");
        defer s.a.free(d);
        return mtl.Library.fromSource(s.device, d, mtl.CompileOptions.mlx());
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.len]) |*p| {
            p.pipe.deinit();
            p.rig.deinit();
        }
    }
};
