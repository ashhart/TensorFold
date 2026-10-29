//! Production NVFP4 gate/up and down: guarded indirect tables, routed prompt rows and independent FP64 ranges.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const source = @import("metal_drama3.zig").source;
const ref = @import("drama3_nvfp4_ref.zig");
pub const Variant = ref.Variant;

pub fn spec(variant: Variant) c.Spec {
    return .{
        .name = if (variant == .up) "metal drama3 d3_nvfp4_up (block16, BF16 SwiGLU)" else "metal drama3 d3_nvfp4_down (block16, FP32)",
        .accum = .f32,
        .depth = ref.depth,
        .ops = if (variant == .up) 10 else 1,
        .out = if (variant == .up) .bf16 else .f32,
        .tiles = .{ .{ .size = 16 }, .{ .size = 4, .step = 4, .min = 4 }, .{ .size = 512, .step = 16, .min = 16 } },
        .served = if (variant == .up) &.{
            .{ .rows = 1, .n = 512, .k = 2048 },
            .{ .rows = 3, .n = 512, .k = 2048 },
            .{ .rows = 17, .n = 512, .k = 2048 },
        } else &.{
            .{ .rows = 1, .n = 2048, .k = 512 },
            .{ .rows = 3, .n = 2048, .k = 512 },
            .{ .rows = 17, .n = 2048, .k = 512 },
        },
    };
}

const Args = extern struct { K: u32, N: u32, prompt: u32, cap: u32, slots: u32 };
const Packed = struct { w: rig_.Ref, s: rig_.Ref, global: rig_.Ref };
const Sample = struct { want: f64, magnitude: f64, range: ref.Interval };

pub const Proj = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipe: mtl.Pipeline,
    variant: Variant,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    host: []u16 = &.{},
    samples: []Sample = &.{},
    weights: std.ArrayList(ref.Weights) = .empty,
    packed_buffers: [16]Packed = undefined,
    x: rig_.Ref = undefined,
    table: rig_.Ref = undefined,
    picks: rig_.Ref = undefined,
    counts: rig_.Ref = undefined,
    members: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, variant: Variant) !Proj {
        const symbol = if (variant == .up) "d3_nvfp4_up" else "d3_nvfp4_down";
        const label = if (variant == .up) "drama3 nvfp4 d3_nvfp4_up" else "drama3 nvfp4 d3_nvfp4_down";
        const src = try source(a, label);
        defer a.free(src);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipe = try rig_.pipeline(device, src, symbol), .variant = variant };
    }

    fn clear(p: *Proj) void {
        p.a.free(p.host);
        p.host = &.{};
        p.a.free(p.samples);
        p.samples = &.{};
        for (p.weights.items) |w| w.deinit();
        p.weights.clearRetainingCapacity();
        p.rig.reset();
    }

    pub fn deinit(p: *Proj) void {
        p.clear();
        p.weights.deinit(p.a);
        p.rig.deinit();
        p.pipe.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        return .{ .spec = spec(p.variant), .ctx = p, .vt = &vt };
    }

    const vt = g.hooks(Proj);

    fn perExpert(p: *const Proj) usize {
        return if (p.variant == .up) 2 else 1;
    }

    fn expert(pair: usize) usize {
        return (3 * (pair % ref.slots) + pair / ref.slots) % ref.slots;
    }

    fn address(b: rig_.Ref) u64 {
        return b.buf.gpuAddress() + b.off;
    }

    fn triple(w: Packed) [3]u64 {
        return .{ address(w.w), address(w.s), address(w.global) };
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const largest = spec(p.variant).served[2];
        if (shape.rows == 0 or shape.rows > largest.rows or shape.n == 0 or shape.n > largest.n or
            shape.n % 4 != 0 or shape.k == 0 or shape.k > largest.k or shape.k % 16 != 0) return false;
        p.clear();
        p.shape = shape;
        p.fill = fill;
        const pairs = shape.rows * ref.slots;
        p.host = try p.a.alloc(u16, (if (p.variant == .up) shape.rows else pairs) * shape.k);
        ref.fillX(p.host, shape.k, p.variant, fill, seed);
        p.x = try p.rig.input(std.mem.sliceAsBytes(p.host));
        try p.weights.ensureTotalCapacity(p.a, ref.slots * p.perExpert());
        for (0..ref.slots * p.perExpert()) |i| {
            const projection = if (p.variant == .up) i % 2 else 2;
            const w = try ref.Weights.init(p.a, shape, fill, seed +% (i * 13), projection);
            p.weights.appendAssumeCapacity(w);
            p.packed_buffers[i] = .{
                .w = try p.rig.input(w.codes),
                .s = try p.rig.input(w.scales),
                .global = try p.rig.input(std.mem.asBytes(&w.global)),
            };
        }
        var table: [ref.experts][9]u64 = undefined;
        for (&table, 0..) |*entry, e| {
            const base = (e % ref.slots) * p.perExpert();
            const first = triple(p.packed_buffers[base]);
            const second = triple(p.packed_buffers[base + p.perExpert() - 1]);
            @memcpy(entry[0..3], &first);
            @memcpy(entry[3..6], &second);
            @memcpy(entry[6..9], &first);
        }
        p.table = try p.rig.input(std.mem.asBytes(&table));
        const picks = try p.a.alloc(i32, pairs);
        defer p.a.free(picks);
        const members = try p.a.alloc(i32, ref.experts * pairs);
        defer p.a.free(members);
        @memset(members, -1);
        var counts: [ref.experts]i32 = @splat(0);
        for (picks, 0..) |*pick, pair| pick.* = @intCast(expert(pair));
        // Reverse order proves the prompt scatter is not accidentally reading contiguous token rows.
        for (0..pairs) |i| {
            const pair = pairs - 1 - i;
            const e = expert(pair);
            members[e * pairs + @as(usize, @intCast(counts[e]))] = @intCast(pair);
            counts[e] += 1;
        }
        p.picks = try p.rig.input(std.mem.sliceAsBytes(picks));
        p.counts = try p.rig.input(std.mem.asBytes(&counts));
        p.members = try p.rig.input(std.mem.sliceAsBytes(members));
        p.out = try p.rig.output(p.count() * c.size(spec(p.variant).out));
        p.samples = try p.a.alloc(Sample, p.count());
        p.computeReference();
        return true;
    }

    pub fn run(p: *Proj) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        for (p.packed_buffers[0..p.weights.items.len]) |w| {
            e.useResource(w.w.buf, 1);
            e.useResource(w.s.buf, 1);
            e.useResource(w.global.buf, 1);
        }
        for ([_]rig_.Ref{ p.x, p.table, p.picks, p.counts, p.members, p.out }, 0..) |b, i|
            e.setBuffer(b.buf, b.off, i);
        e.setValue(Args{
            .K = @intCast(p.shape.k),
            .N = @intCast(p.shape.n),
            .prompt = @intFromBool(p.shape.rows > 1),
            .cap = @intCast(p.shape.rows * ref.slots),
            .slots = ref.slots,
        }, 6);
        const groups_y: usize = if (p.shape.rows > 1) ref.experts else ref.slots;
        e.dispatchGroups(mtl.Size.of(p.shape.n / 4, groups_y, 1), mtl.Size.of(128, 1, 1));
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.shape.rows * ref.slots * p.shape.n;
    }

    fn inputRow(p: *const Proj, pair: usize) []const u16 {
        const r = if (p.variant == .up) pair / ref.slots else pair;
        return p.host[r * p.shape.k ..][0..p.shape.k];
    }

    fn computeReference(p: *Proj) void {
        for (0..p.shape.rows * ref.slots) |pair| {
            const base = expert(pair) * p.perExpert();
            const x = p.inputRow(pair);
            for (0..p.shape.n) |n| {
                const first = p.weights.items[base].dot(n, x);
                const range = ref.projectionRange(first, p.shape, p.fill == .stress);
                var sample = Sample{ .want = first.value, .magnitude = first.magnitude, .range = range };
                if (p.variant == .up) {
                    const second = p.weights.items[base + 1].dot(n, x);
                    sample.want = ref.swiglu(first.value, second.value);
                    sample.magnitude += second.magnitude;
                    sample.range = ref.upRange(range, ref.projectionRange(second, p.shape, p.fill == .stress));
                }
                p.samples[pair * p.shape.n + n] = sample;
            }
        }
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        for (p.samples, want, mag) |sample, *w, *m| {
            w.* = sample.want;
            m.* = sample.magnitude;
        }
    }

    pub fn bounds(p: *Proj, want: []const f64, _: []const f64, allow: []f64) !void {
        for (p.samples, want, allow) |sample, w, *limit|
            limit.* = @max(@abs(sample.range.lo - w), @abs(sample.range.hi - w));
    }

    pub fn intervals(p: *Proj, low: []f64, high: []f64) !void {
        for (p.samples, low, high) |sample, *lo, *hi| {
            lo.* = sample.range.lo;
            hi.* = sample.range.hi;
        }
    }

    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (0..p.shape.rows * ref.slots) |pair| {
            const base = expert(pair) * p.perExpert();
            const x = p.inputRow(pair);
            for (0..p.shape.n) |n| {
                const first = p.weights.items[base].simulate(n, x, acc, order);
                out[pair * p.shape.n + n] = if (p.variant == .up)
                    ref.swiglu(first, p.weights.items[base + 1].simulate(n, x, acc, order))
                else
                    first;
            }
        }
    }
};
