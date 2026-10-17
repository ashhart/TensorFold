//! Gemma's private shaders enter the ordinary contract tally through the family catalogue.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const math = @import("gemma4_math.zig");
const c = contract.spec;
const g = contract.gates;
const family = tf.gemma4.contracts;
const lane = tf.lane_projection;
const mm = tf.gemma4.affine_mm;
const fused = tf.gemma4.fused_lane;
const I = math.Interval;
pub const entries = family.entries;
pub const requireComplete = family.requireComplete;
fn depth(s: c.Shape) usize {
    return 128 + 2 * (s.k / 32) + 32;
}
fn ref(r: rig_.Ref) lane.Ref {
    return .{ .buf = r.buf, .off = r.off };
}
fn offset(r: rig_.Ref, n: usize) lane.Ref {
    return .{ .buf = r.buf, .off = r.off + n };
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    v: family.Variant,
    corrupt: bool = false,
    bytes: std.ArrayList(u8) = .empty,
    served: [2]c.Shape,
    f: ?contract.affine.Affine = null,
    dense: ?mm.Pipes = null,
    projection: ?fused.Projection = null,
    aux: [5][]u16 = undefined,
    hidden: []u16 = &.{},
    xb: []I = &.{},
    hb: []I = &.{},
    ar: [5]rig_.Ref = undefined,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    s: rig_.Ref = undefined,
    b: rig_.Ref = undefined,
    y: rig_.Ref = undefined,
    h: rig_.Ref = undefined,
    tap: rig_.Ref = undefined,
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: family.Variant, corrupt: bool) !Probe {
        var p = Probe{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .corrupt = corrupt, .served = .{ .{ .rows = if (v.kind == 0) 65 else 5, .n = 128, .k = v.k }, .{ .rows = 1, .n = 256, .k = v.k } } };
        if (v.kind == 0) {
            const src = try mm.source(device, a, .{});
            defer a.free(src);
            p.dense = .{try rig_.pipeline(device, src, "tf_affine_mm")};
        }
        return p;
    }
    fn drop(p: *Probe) void {
        if (p.f) |f| {
            f.deinit();
            if (p.v.kind == 1 or p.v.kind == 2) {
                for (p.aux) |v| p.a.free(v);
                p.a.free(p.hidden);
                p.a.free(p.xb);
                p.a.free(p.hb);
            }
        }
        p.f = null;
        if (p.projection) |x| x.deinit();
        p.projection = null;
        p.rig.reset();
    }
    pub fn deinit(p: *Probe) void {
        p.drop();
        if (p.dense) |pipes| for (pipes) |x| x.deinit();
        p.rig.deinit();
        p.bytes.deinit(p.a);
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = .{ .name = p.v.name, .accum = .f32, .depth = depth, .out = .bf16, .served = &p.served, .tiles = .{ .{ .size = 64 }, .{ .size = 64, .step = 64, .min = 64 }, .{} }, .mids = if (p.v.kind == 2) 10 else 0, .sums = if (p.v.kind == 2) .{ .f16 = 0, .bf16 = 0, .why = "staged GeGLU can mask accumulator errors; margins reported, no precision qualification claimed" } else .{} }, .ctx = p, .vt = &vt, .control = if (p.corrupt) .shapes else null };
    }
    const vt = blk: {
        var v = g.hooks(Probe);
        v.output = output;
        v.intact = intact;
        break :blk v;
    };
    fn output(ctx: *anyopaque) ![]const u8 {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        p.bytes.clearRetainingCapacity();
        try p.bytes.appendSlice(p.a, p.rig.bytes(p.y, p.mainCount() * 2));
        if (p.v.kind == 1 or p.v.kind == 2) {
            try p.bytes.appendSlice(p.a, p.rig.bytes(p.h, p.hidden.len * 2));
            if (p.v.tap) try p.bytes.appendSlice(p.a, p.rig.bytes(p.tap, p.hidden.len * 2));
        }
        return p.bytes.items;
    }
    fn intact(ctx: *anyopaque) !bool {
        const p: *Probe = @ptrCast(@alignCast(ctx));
        if (!try p.rig.intact()) return false;
        if (p.v.kind == 2) for (p.rig.bytes(p.y, p.f.?.outputs() * 2)[p.mainCount() * 2 ..]) |byte| {
            if (byte != 0xff) return false;
        };
        return true;
    }
    fn layout(p: *Probe, n: usize) fused.Layout {
        var l = p.v.layout();
        l.projection.n = n;
        return l;
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        p.drop();
        if (shape.k != p.v.k or shape.rows == 0 or shape.rows > 65 or shape.n == 0 or shape.n % 64 != 0) return false;
        if (p.v.kind != 0) p.layout(shape.n).validate() catch return false;
        var f = try contract.affine.Affine.init(p.a, shape, 4, 64);
        var transferred = false;
        errdefer if (!transferred) f.deinit();
        try f.fill(fill, seed);
        if (p.v.kind == 2 and fill == .stress) for (0..shape.n) |j| {
            const row = f.codes[j * shape.k ..][0..shape.k];
            for (0..shape.k / 2) |i| std.mem.swap(u8, &row[i], &row[i + shape.k / 2]);
        };
        if (p.v.kind == 1 or p.v.kind == 2) try p.prologue(f, fill, seed);
        p.f = f;
        transferred = true;
        const words = try p.a.alloc(u32, shape.n * shape.k / 8);
        defer p.a.free(words);
        f.words(words);
        p.x = try p.rig.input(std.mem.sliceAsBytes(f.x));
        if (p.v.kind == 0) {
            p.w = try p.rig.input(std.mem.sliceAsBytes(words));
            p.s = try p.rig.input(std.mem.sliceAsBytes(f.scales));
            p.b = try p.rig.input(std.mem.sliceAsBytes(f.biases));
        } else {
            const l = p.layout(shape.n);
            p.projection = try fused.Projection.init(p.a, p.rig.device, l);
            const pw = try p.a.alloc(u32, l.projection.weightWords());
            defer p.a.free(pw);
            const sb = try p.a.alloc(u16, l.projection.metadataElements());
            defer p.a.free(sb);
            try lane.pack(l.projection, words, f.scales, f.biases, pw, sb);
            p.w = try p.rig.input(std.mem.sliceAsBytes(pw));
            p.s = try p.rig.input(std.mem.sliceAsBytes(sb));
            try lane.regOrder(p.rig.device, &.{.{ .w = ref(p.w), .n = shape.n, .k = shape.k }});
        }
        // encodeFused conservatively validates N outputs; GeGLU's unused half remains an inner guard.
        p.y = try p.rig.output(f.outputs() * 2);
        if (p.v.kind == 1 or p.v.kind == 2) {
            p.h = try p.rig.output(f.x.len * 2);
            p.tap = if (p.v.tap) try p.rig.output(f.x.len * 2) else try p.rig.input(std.mem.sliceAsBytes(p.aux[0]));
        } else {
            p.h = p.x;
            p.tap = p.x;
        }
        return true;
    }
    fn prologue(p: *Probe, f: contract.affine.Affine, fill: g.Fill, seed: u64) !void {
        const len = f.x.len;
        var made: usize = 0;
        errdefer for (p.aux[0..made]) |v| p.a.free(v);
        for (&p.aux) |*v| {
            v.* = try p.a.alloc(u16, len);
            made += 1;
        }
        p.hidden = try p.a.alloc(u16, len);
        errdefer p.a.free(p.hidden);
        p.xb = try p.a.alloc(I, len);
        errdefer p.a.free(p.xb);
        p.hb = try p.a.alloc(I, len);
        errdefer p.a.free(p.hb);
        var prng = std.Random.DefaultPrng.init(seed ^ 0x59);
        const rng = prng.random();
        for (0..len) |i| {
            p.aux[0][i] = c.toBf16(rng.float(f32) * 2 - 1);
            p.aux[1][i] = f.x[i];
            p.aux[2][i] = c.toBf16(0.5 + rng.float(f32));
            p.aux[3][i] = c.toBf16(0.75);
            p.aux[4][i] = c.toBf16(0.5 + rng.float(f32));
            if (p.v.kind == 2) p.aux[3][i] = p.aux[4][i];
            if (fill == .stress) {
                p.aux[0][i] = c.toBf16(1);
                p.aux[1][i] = 0;
                p.aux[2][i] = 0;
                p.aux[3][i] = if (p.v.kind == 1) c.toBf16(1) else f.x[i];
                p.aux[4][i] = f.x[i];
            }
        }
        const k = f.shape.k;
        for (0..f.shape.rows) |row| {
            const start = row * k;
            var aux: [5][]const u16 = undefined;
            for (p.aux, 0..) |v, i| aux[i] = v[start..][0..k];
            math.prologue(p.v.kind, aux, f.x[start..][0..k], p.hidden[start..][0..k], p.xb[start..][0..k], p.hb[start..][0..k], 1e-6);
        }
        for (p.aux, 0..) |v, i| p.ar[i] = try p.rig.input(std.mem.sliceAsBytes(v));
    }
    fn mainCount(p: *Probe) usize {
        const f = p.f.?;
        return f.outputs() / (if (p.v.kind == 2) @as(usize, 2) else 1);
    }
    pub fn count(p: *Probe) usize {
        return p.mainCount() + if (p.v.kind == 1 or p.v.kind == 2) p.hidden.len * (1 + @as(usize, @intFromBool(p.v.tap))) else @as(usize, 0);
    }
    pub fn run(p: *Probe) !void {
        const f = p.f.?;
        const shape = f.shape;
        const pass = p.rig.begin();
        if (p.v.kind == 0) {
            mm.dense(pass.e, p.dense.?, ref(p.x), .{ .w = ref(p.w), .s = ref(p.s), .b = ref(p.b), .n = @as(u32, @intCast(shape.n)), .k = @as(u32, @intCast(shape.k)) }, ref(p.y), @intCast(shape.rows));
        } else for (0..shape.rows) |row| {
            var args = fused.Args{ .a = @splat(ref(p.x)), .o = .{ offset(p.h, row * shape.k * 2), offset(p.tap, row * shape.k * 2) }, .f = .{ 1e-6, 0, 0, 0 }, .i = .{ @intFromBool(p.v.tap), 0, 0, 0 } };
            if (p.v.kind != 3) for (p.ar, 0..) |v, i| {
                args.a[i] = offset(v, row * shape.k * 2);
            };
            try p.projection.?.encodeFused(pass.e, offset(p.x, row * shape.k * 2), ref(p.w), ref(p.s), offset(p.y, row * shape.n * 2 / (if (p.v.kind == 2) @as(usize, 2) else 1)), args, 1);
        }
        try p.rig.finish(pass);
        if (p.corrupt) std.mem.writeInt(u16, p.rig.bytes(p.y, 2)[0..2], c.toBf16(std.math.nan(f32)), .little);
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const f = p.f.?;
        const raw = try p.a.alloc(f64, 2 * f.outputs());
        defer p.a.free(raw);
        const values = raw[0..f.outputs()];
        const mags = raw[f.outputs()..];
        try f.reference(values, mags);
        if (p.v.kind == 0) for (0..f.shape.rows) |r| {
            for (0..f.shape.n) |j| {
                var y: f64 = 0;
                var m: f64 = 0;
                for (0..f.shape.k) |i| {
                    const at = j * (f.shape.k / 64) + i / 64;
                    const w = math.roundedWeight(c.fromBf16(f.scales[at]), f.codes[j * f.shape.k + i], c.fromBf16(f.biases[at]));
                    const term = w * c.fromBf16(f.x[r * f.shape.k + i]);
                    y += term;
                    m += @abs(term);
                }
                values[r * f.shape.n + j] = y;
                mags[r * f.shape.n + j] = m;
            }
        };
        p.finishReference(values, mags, want, mag);
    }
    fn finishReference(p: *Probe, values: []const f64, mags: []const f64, want: []f64, mag: []f64) void {
        const f = p.f.?;
        for (want[0..p.mainCount()], mag[0..p.mainCount()], 0..) |*w, *m, i| {
            const j = if (p.v.kind == 2) (i / (f.shape.n / 2)) * f.shape.n + (i % (f.shape.n / 2)) / 16 * 32 + i % 16 else i;
            w.* = if (p.v.kind == 2) math.geglu(values[j], values[j + 16]) else values[j];
            m.* = if (p.v.kind == 2) mags[j] * @max(1, @abs(values[j + 16])) + mags[j + 16] * @max(1, @abs(values[j])) else mags[j];
        }
        for (want[p.mainCount()..], mag[p.mainCount()..], 0..) |*w, *m, i| {
            w.* = c.fromBf16(p.hidden[i % p.hidden.len]);
            m.* = @abs(w.*);
        }
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const f = p.f.?;
        const raw = try p.a.alloc(f64, 2 * f.outputs() + out.len);
        defer p.a.free(raw);
        try f.simulate(acc, order, raw[0..f.outputs()]);
        @memset(raw[f.outputs()..], 0);
        p.finishReference(raw[0..f.outputs()], raw[f.outputs() .. 2 * f.outputs()], out, raw[2 * f.outputs() ..]);
    }
    pub fn check(p: *Probe, at: g.Gate) !?[]const u8 {
        const f = p.f.?;
        const n = p.count();
        const buf = try p.a.alloc(f64, 2 * n);
        defer p.a.free(buf);
        const want = buf[0..n];
        const mag = buf[n..];
        try p.reference(want, mag);
        const got = try output(p);
        if (!try intact(p)) return "private output overwrote a guard or GeGLU padding";
        if (at == .nonfinite) {
            var planted: usize = 0;
            for (want, 0..) |w, i| {
                planted += @intFromBool(!std.math.isFinite(w));
                if (c.classOf(w) != c.classOf(c.read(.bf16, got, i))) return "private output class differs from staged reference";
            }
            return if (planted == 0 or planted == n) "nonfinite fixture lacks mixed classes" else null;
        }
        const raw_bounds = try p.a.alloc(I, f.outputs());
        defer p.a.free(raw_bounds);
        for (raw_bounds, 0..) |*range, j| {
            const row = j / f.shape.n;
            const col = j % f.shape.n;
            var sum = I.point(0);
            var magnitude: f64 = 0;
            for (0..f.shape.k) |i| {
                const k = col * (f.shape.k / 64) + i / 64;
                const s = c.fromBf16(f.scales[k]);
                const b = c.fromBf16(f.biases[k]);
                const q: f64 = @floatFromInt(f.codes[col * f.shape.k + i]);
                const weight = if (p.v.kind == 0) math.roundedWeight(s, @intFromFloat(q), b) else s * q + b;
                const x = c.fromBf16(f.x[row * f.shape.k + i]);
                const xr = if (p.v.kind == 1 or p.v.kind == 2) p.xb[row * f.shape.k + i] else I.point(x);
                sum = sum.add(xr.mul(I.point(weight)));
                magnitude += (@abs(s * q) + @abs(b)) * @max(@abs(xr.lo), @abs(xr.hi));
            }
            const error_bound = c.gamma(@floatFromInt(depth(f.shape)), 0x1p-24) * magnitude;
            range.* = (I{ .lo = sum.lo - error_bound, .hi = sum.hi + error_bound });
        }
        for (want, 0..) |_, i| {
            var range: I = undefined;
            if (i < p.mainCount()) {
                const j = if (p.v.kind == 2) (i / (f.shape.n / 2)) * f.shape.n + (i % (f.shape.n / 2)) / 16 * 32 + i % 16 else i;
                range = if (p.v.kind == 2) math.gegluBound(raw_bounds[j], raw_bounds[j + 16]) else raw_bounds[j].rounded();
            } else range = p.hb[(i - p.mainCount()) % p.hidden.len];
            if (!range.contains(c.read(.bf16, got, i))) return "private output outside staged rounding interval";
        }
        return null;
    }
};
pub fn limits(a: std.mem.Allocator, device: mtl.Device, only: ?[]const u8) !usize {
    var problems: usize = 0;
    for (entries) |v| {
        if (only) |s| {
            if (std.mem.indexOf(u8, v.name, s) == null) continue;
        }
        const text = if (v.kind == 0) try mm.source(device, a, .{}) else try fused.source(a, v.layout());
        defer a.free(text);
        const source = if (v.kind == 0) try a.dupe(u8, text) else try tf.gemma4.contracts.adapt(device, a, text);
        defer a.free(source);
        const d = contract.limits.MetalLaunch{ .name = v.name, .threads = if (v.kind == 0) 128 else @intCast(v.layout().threads), .tg_memory = @intCast(v.threadgroupBytes()), .offline = false };
        var notes: std.ArrayList(u8) = .empty;
        defer notes.deinit(a);
        for (contract.limits.apple) |chip| problems += try contract.limits.metal(d, chip, &notes, a);
        std.debug.print("{s}", .{notes.items});
        problems += try @import("metal_limits.zig").onThisGpu(a, device, .{ .decl = d, .function = if (v.kind == 0) "tf_affine_mm" else "tf_lane_reg", .here = source, .pre_m5 = "" }, false);
    }
    return problems;
}

pub fn append(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: family.Variant, corrupt: bool, owners: *std.ArrayList(*Probe), kernels: *std.ArrayList(g.Kernel)) !void {
    const p = try a.create(Probe);
    errdefer a.destroy(p);
    p.* = try Probe.init(a, device, queue, v, corrupt);
    errdefer p.deinit();
    try kernels.ensureUnusedCapacity(a, 1);
    try owners.append(a, p);
    kernels.appendAssumeCapacity(p.kernel());
}
