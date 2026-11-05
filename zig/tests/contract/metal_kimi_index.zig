//! K3.1's assumed biased key norm and scores through independent binary64 contract comparisons.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("kimi_index_ref.zig");
const c = contract.spec;
const g = contract.gates;
const ix = k3.index;

pub const Which = enum { key, scores };
const shapes_key = [_]c.Shape{ .{ .rows = 8, .n = 128, .k = 128 }, .{ .rows = 1, .n = 128, .k = 128 }, .{ .rows = 256, .n = 128, .k = 128 } };
const shapes_scores = [_]c.Shape{
    .{ .rows = 8, .n = 257, .k = 128 },   .{ .rows = 1, .n = 1, .k = 128 },
    .{ .rows = 3, .n = 127, .k = 128 },   .{ .rows = 2, .n = 128, .k = 128 },
    .{ .rows = 1, .n = 129, .k = 128 },   .{ .rows = 2, .n = 2047, .k = 128 },
    .{ .rows = 1, .n = 2048, .k = 128 },  .{ .rows = 1, .n = 2049, .k = 128 },
    .{ .rows = 1, .n = 65536, .k = 128 },
};
fn depth(_: c.Shape) usize {
    return 128 + 32;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    ctl: u8,
    name: []const u8,
    pipe: mtl.Pipeline,
    shape: c.Shape = undefined,
    x: rig_.Ref = undefined,
    w: rig_.Ref = undefined,
    k: rig_.Ref = undefined,
    meta: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    mode: g.Fill = .random,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which, ctl: u8, name: []const u8) !Probe {
        var text = try a.dupe(u8, ix.source);
        defer a.free(text);
        const edit: ?[2][]const u8 = switch (ctl) {
            1 => .{ "isnan(scaled) ? scaled : max(scaled, 0.0f)", "max(scaled, 0.0f)" },
            2 => .{ "h < K31_IH; ++h", "h + 1 < K31_IH; ++h" },
            3 => .{ "dot = fma(query[h * K31_ID + col], float(key[col]), dot)", "dot = float(half(fma(query[h * K31_ID + col], float(key[col]), dot)))" },
            4 => .{ "scores[ulong(r) * a.score_stride + k] = score;", "scores[ulong(r) * a.score_stride + k] = score; if (r == 0 && k == 0) scores[ulong(a.rows) * a.score_stride] = 1.0f;" },
            5 => .{ "v += simd_shuffle_xor(v, d)", "v = float(half(v + simd_shuffle_xor(v, d)))" },
            6 => .{ "v += simd_shuffle_xor(v, d)", "v = float(bfloat(v + simd_shuffle_xor(v, d)))" },
            else => null,
        };
        if (edit) |pair| {
            if (std.mem.count(u8, text, pair[0]) != 1) return error.ControlSourceChanged;
            const changed = try std.mem.replaceOwned(u8, a, text, pair[0], pair[1]);
            a.free(text);
            text = changed;
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .which = which, .ctl = ctl, .name = name, .pipe = try rig_.pipeline(device, text, if (which == .key) "k3_index_key" else "k3_index_scores") };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        p.pipe.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        const spec: c.Spec = .{
            .name = p.name,
            .accum = .f32,
            .depth = depth,
            .ops = 4,
            .out = if (p.which == .key) .bf16 else .f32,
            .served = if (p.which == .key) &shapes_key else &shapes_scores,
            .tiles = .{ .{ .size = 16 }, .{}, .{} },
        };
        return .{ .spec = spec, .ctx = p, .vt = &vt, .control = switch (p.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            5, 6 => .sums,
            else => null,
        } };
    }

    const vt = g.hooks(Probe);

    fn values(p: *Probe, n: usize, rng: std.Random) !rig_.Ref {
        const words = try p.a.alloc(u16, n);
        defer p.a.free(words);
        for (words) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 2);
        return p.rig.input(std.mem.sliceAsBytes(words));
    }

    fn bits(p: *Probe, r: rig_.Ref, n: usize) []u16 {
        return @alignCast(std.mem.bytesAsSlice(u16, p.rig.bytes(r, n * 2)));
    }

    pub fn prepare(p: *Probe, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows < 1 or shape.rows > 256 or shape.k != 128 or shape.n < 1 or shape.n > 65536 or (p.which == .key and shape.n != 128)) return false;
        p.rig.reset();
        p.shape = shape;
        p.mode = mode;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const nr = shape.rows;
        p.x = try p.values(nr * (if (p.which == .key) @as(usize, 128) else 4096), rng);
        p.w = try p.values(if (p.which == .key) 128 else nr * 32, rng);
        p.k = try p.values(if (p.which == .key) 128 else shape.n * 128, rng);
        const meta = try p.a.alloc(ix.Row, nr);
        defer p.a.free(meta);
        for (meta, 0..) |*row, r| row.* = .{ .slot = 0, .pos = @intCast(if (p.which == .key) r else shape.n - 1), .list = 0 };
        p.meta = try p.rig.input(std.mem.sliceAsBytes(meta));
        p.out = try p.rig.output(p.count() * (if (p.which == .key) @as(usize, 2) else 4));
        const x = p.bits(p.x, nr * (if (p.which == .key) @as(usize, 128) else 4096));
        if (mode == .nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7fc1 }, 0..) |v, r| {
            if (r < nr) x[r * (if (p.which == .key) @as(usize, 128) else 4096)] = v;
        };
        if (mode == .stress) {
            // The small mean offset is lost in either reduced accumulator, but visible after normalization.
            for (x, 0..) |*v, i| v.* = c.toBf16(if (p.which == .key) @as(f32, if (i % 128 == 0) 132 else 128) else 100);
            @memset(p.bits(p.w, if (p.which == .key) 128 else nr * 32), c.toBf16(1));
            @memset(p.bits(p.k, if (p.which == .key) 128 else shape.n * 128), c.toBf16(if (p.which == .key) @as(f32, 0) else 100));
        }
        return true;
    }

    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn encode(p: *Probe, e: mtl.ComputeEncoder, first: usize, nr: usize) void {
        e.setPipeline(p.pipe);
        e.setBuffer(p.x.buf, p.x.off + first * (if (p.which == .key) @as(usize, 128) else 4096) * 2, 0);
        e.setBuffer(p.w.buf, p.w.off + (if (p.which == .key) 0 else first * 32 * 2), 1);
        e.setBuffer(p.k.buf, p.k.off, 2);
        e.setBuffer(p.meta.buf, p.meta.off + first * @sizeOf(ix.Row), 3);
        e.setValue(ix.Args{ .rows = @intCast(nr), .slot_keys = @intCast(if (p.which == .key) p.shape.rows else p.shape.n), .score_stride = @intCast(p.shape.n) }, 4);
        e.setBuffer(p.out.buf, p.out.off + (if (p.which == .key) 0 else first * p.shape.n * 4), 5);
        e.dispatchGroups(mtl.Size.of(if (p.which == .key) nr else (p.shape.n + 127) / 128, if (p.which == .key) 1 else nr, 1), mtl.Size.of(if (p.which == .key) 32 else 128, 1, 1));
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.shape.rows);
        try p.rig.finish(pass);
    }

    pub fn independentRows(p: *Probe) !void {
        if (!try p.prepare(.{ .rows = 8, .n = if (p.which == .key) 128 else 2051, .k = 128 }, .random, 973)) return error.IndexFixtureRefused;
        const meta: []ix.Row = @alignCast(std.mem.bytesAsSlice(ix.Row, p.rig.bytes(p.meta, 8 * @sizeOf(ix.Row))));
        if (p.which == .scores) for (meta, 0..) |*row, i| {
            row.pos = @intCast(([_]usize{ 0, 126, 127, 128, 511, 2046, 2047, 2050 })[i]);
        };
        try p.run();
        const width: usize = if (p.which == .key) 2 else 4;
        const bytes = try p.a.dupe(u8, p.rig.bytes(p.out, p.count() * width));
        defer p.a.free(bytes);
        const want = try p.a.alloc(f64, p.count());
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, p.count());
        defer p.a.free(mag);
        const allow = try p.a.alloc(f64, p.count());
        defer p.a.free(allow);
        try p.reference(want, mag);
        try p.bounds(want, mag, allow);
        for (0..8) |r| for (0..p.shape.n) |j| {
            const i = r * p.shape.n + j;
            if (p.which == .scores and j > meta[r].pos) {
                if (std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little) != 0xffffffff) return error.IndexFutureKeyWritten;
            } else {
                const got = c.read(p.kernel().spec.out, bytes, i);
                if (!(@abs(got - want[i]) <= allow[i])) return error.IndexCausalReference;
            }
        };
        @memset(p.rig.bytes(p.out, p.count() * width), 0xff);
        const pass = p.rig.begin();
        for (0..8) |i| p.encode(pass.e, 7 - i, 1);
        try p.rig.finish(pass);
        if (!std.mem.eql(u8, bytes, p.rig.bytes(p.out, bytes.len)) or !try p.rig.intact()) return error.IndexRowOrder;
    }

    fn evaluate(p: *Probe, sum: ref.Sum, want: []f64, mag: []f64) void {
        const nr = p.shape.rows;
        if (p.which == .key) {
            const x = p.bits(p.x, nr * 128);
            for (0..nr) |r| ref.key(x[r * 128 ..][0..128], p.bits(p.w, 128), p.bits(p.k, 128), sum, want[r * 128 ..][0..128], mag[r * 128 ..][0..128]);
        } else {
            const x = p.bits(p.x, nr * 4096);
            const w = p.bits(p.w, nr * 32);
            const keys = p.bits(p.k, p.shape.n * 128);
            for (0..nr) |r| for (0..p.shape.n) |j| {
                const result = ref.score(x[r * 4096 ..][0..4096], keys[j * 128 ..][0..128], w[r * 32 ..][0..32], sum);
                want[r * p.shape.n + j] = result.value;
                mag[r * p.shape.n + j] = result.magnitude;
            };
        }
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        p.evaluate(.{}, want, mag);
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.a.alloc(f64, out.len);
        defer p.a.free(mag);
        p.evaluate(.{ .acc = acc, .order = order }, out, mag);
    }

    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (p.which == .scores) {
            const spec = p.kernel().spec;
            for (allow, want, mag) |*out, w, m| out.* = if (p.mode == .stress) c.boundExact(spec, w) else c.bound(spec, p.shape, w, m);
            return;
        }
        const x = p.bits(p.x, p.shape.rows * 128);
        const w = p.bits(p.w, 128);
        for (0..p.shape.rows) |r| {
            const xr = x[r * 128 ..][0..128];
            var mean: f64 = 0;
            var abs_mean: f64 = 0;
            for (xr) |v| {
                mean += ref.f(v) / 128;
                abs_mean += @abs(ref.f(v)) / 128;
            }
            var variance: f64 = 0;
            var centered_abs: f64 = 0;
            for (xr) |v| {
                variance += std.math.pow(f64, ref.f(v) - mean, 2) / 128;
                centered_abs += @abs(ref.f(v) - mean) / 128;
            }
            const dm = c.gamma(9, 0x1p-24) * abs_mean;
            const dv = c.gamma(12, 0x1p-24) * variance + (2 * dm * centered_abs + dm * dm) * (1 + c.gamma(12, 0x1p-24));
            const base = variance + @as(f64, @as(f32, 1e-5));
            if (dv >= base) return error.IllConditionedIndexFixture;
            const inv = 1 / @sqrt(base);
            const di = inv * (dv / (2 * (base - dv)) + 2 * 0x1p-24);
            for (0..128) |j| {
                const delta = @abs(ref.f(xr[j]) - mean);
                const dx = dm + 0x1p-24 * (delta + dm);
                const work = @abs(ref.f(w[j])) * (dx * inv + (delta + dx) * di + 0x1p-24 * (delta + dx) * (inv + di)) + 0x1p-24 * @abs(want[r * 128 + j]);
                allow[r * 128 + j] = work + 0x1p-8 * (@abs(want[r * 128 + j]) + work) + 0x1p-133;
            }
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Probe) = .empty,
    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        const entries = .{
            .{ Which.key, @as(u8, 0), "kimi index key k3_index_key" },
            .{ Which.scores, @as(u8, 0), "kimi index scores k3_index_scores" },
            .{ Which.scores, @as(u8, 1), "kimi index control: NaN suppressed" },
            .{ Which.scores, @as(u8, 2), "kimi index control: last head dropped" },
            .{ Which.scores, @as(u8, 3), "kimi index control: FP16 dot" },
            .{ Which.scores, @as(u8, 4), "kimi index control: output overrun" },
            .{ Which.key, @as(u8, 5), "kimi index key control: FP16 shuffle sums" },
            .{ Which.key, @as(u8, 6), "kimi index key control: BF16 shuffle sums" },
        };
        errdefer s.destroy();
        inline for (entries) |entry| {
            if (only == null or std.mem.indexOf(u8, entry[2], only.?) != null) {
                const p = try a.create(Probe);
                p.* = Probe.init(a, device, queue, entry[0], entry[1], entry[2]) catch |err| {
                    a.destroy(p);
                    return err;
                };
                s.probes.append(a, p) catch |err| {
                    p.deinit();
                    a.destroy(p);
                    return err;
                };
                try kernels.append(a, p.kernel());
            }
        }
        return s;
    }
    pub fn invariance(s: *Suite) !void {
        for (s.probes.items) |p| if (p.ctl == 0) {
            try p.independentRows();
            if (p.which == .scores) std.debug.print("PASS kimi index scores: causal binary64 outputs; future keys untouched; reverse single-row launches equal batch bytes\n", .{}) else std.debug.print("PASS kimi index key: binary64 outputs; reverse single-row launches equal batch bytes\n", .{});
        };
    }
    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }
};
