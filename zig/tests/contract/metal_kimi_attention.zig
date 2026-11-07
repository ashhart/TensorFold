//! Selected K3.1 MLA: independent binary64 outputs, guarded repeatability and a supplemental dense-byte oracle.
const std = @import("std");
const mtl = @import("metal");
const k3 = @import("kimi_k3");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("kimi_index_ref.zig");
const c = contract.spec;
const g = contract.gates;
const ix = k3.index;
const shapes = [_]c.Shape{
    .{ .rows = 8, .n = 2 * 512, .k = 17 },   .{ .rows = 1, .n = 512, .k = 1 },
    .{ .rows = 2, .n = 3 * 512, .k = 15 },   .{ .rows = 1, .n = 33 * 512, .k = 16 },
    .{ .rows = 1, .n = 96 * 512, .k = 513 }, .{ .rows = 1, .n = 512, .k = 2047 },
    .{ .rows = 1, .n = 512, .k = 2048 },     .{ .rows = 1, .n = 512, .k = 2049 },
    .{ .rows = 1, .n = 512, .k = 65536 },
};
fn depth(s: c.Shape) usize {
    return 5 * @min(s.k, 2048) + 32;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: [4]mtl.Pipeline,
    shape: c.Shape = undefined,
    ctl: u8 = 0,
    mode: g.Fill = .random,
    q: rig_.Ref = undefined,
    qlat: rig_.Ref = undefined,
    cache: rig_.Ref = undefined,
    compact: rig_.Ref = undefined,
    meta: rig_.Ref = undefined,
    dense_meta: rig_.Ref = undefined,
    list: rig_.Ref = undefined,
    partial: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    dense_out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, ctl: u8) !Probe {
        const dense = try std.mem.replaceOwned(u8, a, ix.dense_source, "#include \"kimi_common.h\"", ix.common_source);
        defer a.free(dense);
        const edit: ?[2][]const u8 = switch (ctl) {
            1 => .{ "const float sc = simd_sum(p) * a.scale;", "const float sc0 = simd_sum(p) * a.scale; const float sc = isfinite(sc0) ? sc0 : 0.0f;" },
            2 => .{ "const uint key = list[row.list + k0 + j];", "const uint key = list[row.list + k0];" },
            3 => .{ "o[i] = fma(e, float(key[16 * lane + i]), o[i] * corr);", "o[i] = float(half(fma(e, float(key[16 * lane + i]), o[i] * corr)));" },
            4 => .{ "acc[i] / L;", "acc[i] / L; if (t == 0 && r == 0 && hl == 0) olat[a.rows * a.heads * K31_LAT] = 1.0f;" },
            else => null,
        };
        const indexed = if (edit) |pair| try std.mem.replaceOwned(u8, a, ix.attention_source, pair[0], pair[1]) else try a.dupe(u8, ix.attention_source);
        defer a.free(indexed);
        if (ctl != 0 and std.mem.eql(u8, indexed, ix.attention_source)) return error.ControlSourceChanged;
        const text = try std.mem.concat(a, u8, &.{ dense, "\n", indexed });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        var pipes: [4]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |p| p.deinit();
        for ([_][]const u8{ "k3_index_attend", "k3_index_merge", "k3_mla_attend", "k3_mla_merge" }, 0..) |function_name, i| {
            pipes[i] = try mtl.Pipeline.init(device, lib, function_name, false);
            made += 1;
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .ctl = ctl };
    }
    pub fn deinit(p: *Probe) void {
        p.rig.deinit();
        for (p.pipes) |pipe| pipe.deinit();
    }
    pub fn name(ctl: u8) []const u8 {
        return switch (ctl) {
            1 => "kimi index attention control: NaN score suppressed",
            2 => "kimi index attention control: tile repeats first key",
            3 => "kimi index attention control: FP16 numerator",
            4 => "kimi index attention control: output overrun",
            else => "kimi index attention k3_index_attend k3_index_merge",
        };
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = .{ .name = name(p.ctl), .accum = .f32, .depth = depth, .ops = 8, .out = .f32, .served = &shapes }, .ctx = p, .vt = &vt, .control = switch (p.ctl) {
            1 => .nonfinite,
            2 => .shapes,
            3 => .sums,
            4 => .repeat,
            else => null,
        } };
    }

    const vt = g.hooks(Probe);

    fn heads(p: *const Probe) usize {
        return p.shape.n / 512;
    }
    fn kept(p: *const Probe) usize {
        return @min(p.shape.k, 2048);
    }
    fn slice(p: *Probe, comptime T: type, r: rig_.Ref, n: usize) []T {
        return @alignCast(std.mem.bytesAsSlice(T, p.rig.bytes(r, n * @sizeOf(T))));
    }

    pub fn prepare(p: *Probe, shape: c.Shape, mode: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 256 or shape.n == 0 or shape.n % 512 != 0 or shape.n / 512 > 96 or shape.k == 0 or shape.k > 65536) return false;
        p.rig.reset();
        p.shape = shape;
        p.mode = mode;
        const nr = shape.rows;
        const nh = p.heads();
        const nk = p.kept();
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const query = try p.a.alloc(u16, nr * nh * 192);
        defer p.a.free(query);
        const qlat = try p.a.alloc(f32, nr * nh * 512);
        defer p.a.free(qlat);
        const keys = try p.a.alloc(u16, nr * shape.k * 576);
        defer p.a.free(keys);
        for (query) |*v| v.* = c.toBf16(if (mode == .stress) 0 else rng.float(f32) * 0.04 - 0.02);
        for (qlat) |*v| v.* = if (mode == .stress) 0 else rng.float(f32) * 0.04 - 0.02;
        for (keys, 0..) |*v, j| v.* = c.toBf16(if (mode == .stress) (@as(f32, @floatFromInt(rng.uintLessThan(u32, 1001))) - 500) * 256 else rng.float(f32) * 4 - 2 + @as(f32, @floatFromInt(j % 3)) * 0.0625);
        if (mode == .nonfinite) for ([_]u16{ 0x7fc0, 0x7f80, 0xff80, 0x7fc1 }, 0..) |v, r| {
            if (r < nr) query[r * nh * 192 + 128] = v;
        };
        p.q = try p.rig.input(std.mem.sliceAsBytes(query));
        p.qlat = try p.rig.input(std.mem.sliceAsBytes(qlat));
        p.cache = try p.rig.input(std.mem.sliceAsBytes(keys));
        const list = try p.a.alloc(u32, nr * nk);
        defer p.a.free(list);
        const compact = try p.a.alloc(u16, nr * nk * 576);
        defer p.a.free(compact);
        const meta = try p.a.alloc(ix.Row, nr);
        defer p.a.free(meta);
        const dense_meta = try p.a.alloc(k3.kernels.MlaRow, nr);
        defer p.a.free(dense_meta);
        for (0..nr) |r| {
            meta[r] = .{ .slot = @intCast(r), .pos = @intCast(shape.k - 1), .list = @intCast(r * nk), .nkeys = @intCast(nk) };
            dense_meta[r] = .{ .slot = @intCast(r), .pos = @intCast(nk - 1) };
            for (0..nk) |j| {
                const pos = if (nk == shape.k) j else j * (shape.k - 1) / (nk - 1);
                list[r * nk + j] = @intCast(pos);
                @memcpy(compact[(r * nk + j) * 576 ..][0..576], keys[(r * shape.k + pos) * 576 ..][0..576]);
            }
        }
        p.compact = try p.rig.input(std.mem.sliceAsBytes(compact));
        p.list = try p.rig.input(std.mem.sliceAsBytes(list));
        p.meta = try p.rig.input(std.mem.sliceAsBytes(meta));
        p.dense_meta = try p.rig.input(std.mem.sliceAsBytes(dense_meta));
        p.partial = try p.rig.scratch(nr * nh * 16 * 514 * 4);
        p.out = try p.rig.output(p.count() * 4);
        p.dense_out = try p.rig.scratch(p.count() * 4);
        return true;
    }

    fn call(p: *Probe, dense: bool) k3.kernels.MlaArgs {
        const cache = if (dense) p.compact else p.cache;
        return .{ .cache = cache.buf.gpuAddress() + cache.off, .slot_keys = @intCast(if (dense) p.kept() else p.shape.k), .heads = @intCast(p.heads()), .head0 = 0, .rows = @intCast(p.shape.rows), .q_stride = @intCast(p.heads() * 192), .kv_stride = 576, .gate_stride = 0, .out_stride = 0, .eps = 1e-5, .scale = 1.0 / @sqrt(@as(f32, 192)) };
    }

    fn launch(p: *Probe, dense: bool) !void {
        const pass = p.rig.begin();
        const first: usize = if (dense) 2 else 0;
        const meta = if (dense) p.dense_meta else p.meta;
        const out = if (dense) p.dense_out else p.out;
        pass.e.useResource((if (dense) p.compact else p.cache).buf, 1);
        pass.e.setPipeline(p.pipes[first]);
        const inputs = .{ p.q, p.qlat, meta, p.partial };
        inline for (inputs, 0..) |r, i| pass.e.setBuffer(r.buf, r.off, i);
        pass.e.setValue(p.call(dense), 4);
        if (!dense) pass.e.setBuffer(p.list.buf, p.list.off, 5);
        const fit = @min(32, p.pipes[first].maxThreads() / 32);
        const groups = (p.heads() + fit - 1) / fit;
        const per = (p.heads() + groups - 1) / groups;
        pass.e.dispatchGroups(mtl.Size.of(groups, 16, p.shape.rows), mtl.Size.of(per * 32, 1, 1));
        pass.e.barrier();
        pass.e.setPipeline(p.pipes[first + 1]);
        pass.e.setBuffer(p.partial.buf, p.partial.off, 0);
        pass.e.setBuffer(meta.buf, meta.off, 1);
        pass.e.setBuffer(out.buf, out.off, 2);
        pass.e.setValue(p.call(dense), 3);
        pass.e.dispatchGroups(mtl.Size.of(p.heads(), p.shape.rows, 1), mtl.Size.of(256, 1, 1));
        try p.rig.finish(pass);
    }
    pub fn run(p: *Probe) !void {
        try p.launch(false);
    }
    pub fn count(p: *Probe) usize {
        return p.shape.rows * p.shape.n;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const nh = p.heads();
        const nr = p.shape.rows;
        const nk = p.kept();
        const query = p.slice(u16, p.q, nr * nh * 192);
        const qlat = p.slice(f32, p.qlat, nr * nh * 512);
        const keys = p.slice(u16, p.cache, nr * p.shape.k * 576);
        const list = p.slice(u32, p.list, nr * nk);
        for (0..nr) |r| for (0..nh) |h| {
            const offset = (r * nh + h) * 512;
            ref.attend(query[(r * nh + h) * 192 ..][0..192], qlat[offset..][0..512], keys[r * p.shape.k * 576 ..][0 .. p.shape.k * 576], list[r * nk ..][0..nk], p.call(false).scale, want[offset..][0..512], mag[offset..][0..512]);
        };
    }

    pub fn bounds(p: *Probe, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (p.mode == .stress) {
            for (allow, want) |*a, w| a.* = c.boundExact(p.kernel().spec, w);
            return;
        }
        const nh = p.heads();
        const nr = p.shape.rows;
        const nk = p.kept();
        const query = p.slice(u16, p.q, nr * nh * 192);
        const qlat = p.slice(f32, p.qlat, nr * nh * 512);
        const keys = p.slice(u16, p.compact, nr * nk * 576);
        for (0..nr) |r| for (0..nh) |h| {
            var most: f64 = 0;
            var value: f64 = 0;
            for (0..nk) |j| {
                const k = keys[(r * nk + j) * 576 ..][0..576];
                var sum: f64 = 0;
                for (0..512) |d| {
                    sum += @abs(@as(f64, qlat[(r * nh + h) * 512 + d]) * ref.f(k[d]));
                    value = @max(value, @abs(ref.f(k[d])));
                }
                for (0..64) |d| sum += @abs(ref.f(query[(r * nh + h) * 192 + 128 + d]) * ref.f(k[512 + d]));
                most = @max(most, sum * p.call(false).scale);
            }
            // Dot: 18 lane terms, five shuffle sums and a scale. Softmax sensitivity is bounded by 2*score error.
            const ds = c.gamma(25, 0x1p-24) * most;
            const work = value * (2 * ds + c.gamma(@floatFromInt(5 * nk + 32), 0x1p-24));
            for (0..512) |d| {
                const i = (r * nh + h) * 512 + d;
                allow[i] = work + 0x1p-24 * @abs(want[i]) + 0x1p-149;
            }
        };
        _ = mag;
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        // Stress queries are zero, so each latent is the uniform mean, with no score approximation.
        const terms = try p.a.alloc(f64, p.kept());
        defer p.a.free(terms);
        const nk = p.kept();
        const keys = p.slice(u16, p.compact, p.shape.rows * nk * 576);
        for (0..p.shape.rows) |r| for (0..512) |d| {
            for (terms, 0..) |*term, j| term.* = ref.f(keys[(r * nk + j) * 576 + d]);
            const sum = contract.rows.sumIn(acc, order, terms);
            @memset(terms, 1);
            const denominator = contract.rows.sumIn(acc, order, terms);
            for (0..p.heads()) |h| out[(r * p.heads() + h) * 512 + d] = sum / denominator;
        };
    }

    /// Supplemental byte equality never replaces the generic runner's binary64 comparisons.
    pub fn denseBytes(p: *Probe) !void {
        try p.run();
        const before = try p.a.dupe(u8, p.rig.bytes(p.out, p.count() * 4));
        defer p.a.free(before);
        try p.launch(true);
        if (!std.mem.eql(u8, before, p.rig.bytes(p.dense_out, p.count() * 4))) return error.IndexDenseBytesDiffer;
    }
};
