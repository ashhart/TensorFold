//! Byte-exact Gate 1-4 contract for Inkling 2 Mini's multi-segment dense projections vs core sp_matvec.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const controls = @import("metal_inkling2_dense_controls.zig");
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.glue;
const glue_src = @import("inkling2_metal").glue;
const sp_sources = @import("kernel_sources");

const c = contract.spec;
const g = contract.gates;
const dense_in = contract.dense;

const MvArgs = extern struct { rows: u32, n: u32, k: u32, ldy: u32 };

fn denseDepth(s: c.Shape) usize {
    return s.k / 32 + 8;
}

const dense_served = [_]c.Shape{
    .{ .rows = 5, .n = 8, .k = 512 },
    .{ .rows = 1, .n = 24, .k = 512 },
    .{ .rows = 2, .n = 8, .k = 512 },
    .{ .rows = 4, .n = 8, .k = 512 },
    .{ .rows = 8, .n = 8, .k = 512 },
    .{ .rows = 16, .n = 8, .k = 512 },
    .{ .rows = 32, .n = 8, .k = 512 },
    .{ .rows = 33, .n = 8, .k = 512 },
    .{ .rows = 1, .n = 8, .k = 264 },
    .{ .rows = 1, .n = 8, .k = 2304 },
};

fn segWidths(segs: u8, n: usize) [6]u32 {
    if (n == 24) return .{ 3072, 384, 24, 3072, 1024, 1024 };
    const base: u32 = @intCast(n);
    return .{ base + 3, base, 3, base + 1, if (segs == 6) base + 5 else 0, if (segs == 6) base + 2 else 0 };
}

fn fillSegmentWeights(w16: []u16, d: dense_in.Dense, s: c.Shape, fill: g.Fill, seed: u64) void {
    var prng = std.Random.DefaultPrng.init(seed ^ 0x5151);
    const rng = prng.random();
    const h = s.k / 2;
    for (0..s.n) |j| for (0..s.k) |i| {
        var v: f32 = if (fill != .stress) rng.float(f32) * 2 - 1 else if (i < h) @floatFromInt(1 + (i * 7 + j) % 6) else -@as(f32, @floatFromInt(1 + ((i - h) * 7 + j) % 6));
        if (fill == .nonfinite and v == 0) v = 0.25;
        w16[j * s.k + i] = c.toBf16(v);
        d.w[j * s.k + i] = c.fromBf16(w16[j * s.k + i]);
    };
}

fn plantOrderWitness(x: []u16, w16: []u16, d: dense_in.Dense) void {
    const sets = [_]struct { i: usize, xv: f32, wv: f32 }{
        .{ .i = 0, .xv = 1.0, .wv = 0x1p26 },
        .{ .i = 256, .xv = 1.0, .wv = -0x1p26 },
        .{ .i = 260, .xv = 1.0, .wv = 2.0 },
    };
    for (0..8) |o| {
        x[o] = c.toBf16(0);
        x[256 + o] = c.toBf16(0);
        w16[o] = c.toBf16(0);
        w16[256 + o] = c.toBf16(0);
        d.w[o] = 0;
        d.w[256 + o] = 0;
    }
    for (sets) |p| {
        x[p.i] = c.toBf16(p.xv);
        w16[p.i] = c.toBf16(p.wv);
        d.w[p.i] = c.fromBf16(w16[p.i]);
    }
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    base: [6]mtl.Pipeline,
    segs: u8,
    ctl: controls.Control,
    shape: c.Shape = undefined,
    n_dims: [6]u32 = @splat(0),
    models: [6]?dense_in.Dense = @splat(null),
    x_refs: [3]rig_.Ref = undefined,
    w_refs: [6]rig_.Ref = undefined,
    y_refs: [6]rig_.Ref = undefined,
    base_out: [6]rig_.Ref = undefined,
    base_bytes: std.ArrayList(u8) = .empty,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, segs: u8, ctl: controls.Control) !Probe {
        const text = try controls.source(a, glue_src, ctl);
        defer a.free(text);
        var pipes = try host.Pipelines.loadSource(device, text);
        errdefer pipes.deinit();
        const lib = try mtl.Library.fromSource(device, sp_sources.spark25_glue, .{ .language = mtl.CompileOptions.mlx().language, .math_functions = .precise });
        defer lib.deinit();
        var base: [6]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (base[0..made]) |p| p.deinit();
        for ([_][:0]const u8{ "sp_matvec_1", "sp_matvec_2", "sp_matvec_4", "sp_matvec_8", "sp_matvec_16", "sp_matvec_32" }, &base) |name, *p| {
            p.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .base = base, .segs = segs, .ctl = ctl };
    }

    fn drop(p: *Probe) void {
        for (&p.models) |*slot| {
            if (slot.*) |d| d.deinit();
            slot.* = null;
        }
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.base_bytes.deinit(p.a);
        p.rig.deinit();
        for (p.base) |b| b.deinit();
        p.pipes.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        return .{
            .ctx = p,
            .vt = &vt,
            .control = if (p.ctl == .none) null else .shapes,
            .spec = .{
                .name = controls.label(p.ctl, p.segs),
                .replaces = "core sp_matvec_{1,2,4,8,16,32}",
                .accum = .f32,
                .depth = denseDepth,
                .out = .bf16,
                .tiles = .{ .{ .size = 8, .step = 1, .min = 1 }, .{ .size = 8, .step = 8, .min = 8 }, .{ .size = 256, .step = 8, .min = 256 } },
                .served = &dense_served,
            },
        };
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (s.rows == 0 or s.n == 0 or s.k == 0 or s.k % 8 != 0) return false;
        p.drop();
        p.shape = s;
        p.n_dims = segWidths(p.segs, s.n);
        const w_max: usize = s.k * @max(p.n_dims[0], @max(p.n_dims[4], p.n_dims[5]));
        const w16 = try p.a.alloc(u16, w_max);
        defer p.a.free(w16);
        for (0..p.segs) |idx| {
            const ns = p.n_dims[idx];
            const d = try dense_in.Dense.init(p.a, .{ .rows = s.rows, .n = ns, .k = s.k }, 1);
            p.models[idx] = d;
            if (idx == 0 or idx >= 4) {
                d.fillX(fill, seed ^ (@as(u64, idx) * 0x9e3779b9), if (fill == .stress) 16 else 2);
                if (fill == .stress) for (0..s.rows) |row| for (0..8) |i| {
                    const base_x = c.fromBf16(d.x[row * s.k + i]);
                    d.x[row * s.k + s.k / 2 + i] = c.toBf16(base_x + 1);
                };
            } else @memcpy(d.x, p.models[0].?.x);
            fillSegmentWeights(w16[0 .. ns * s.k], d, .{ .rows = s.rows, .n = ns, .k = s.k }, fill, seed +% idx *% 17);
            if (idx == 0 and fill == .random and s.k >= 512) plantOrderWitness(d.x, w16[0 .. ns * s.k], d);
            if (idx >= 1 and idx <= 3 and fill == .random and s.k >= 512) @memcpy(d.x, p.models[0].?.x);
            p.w_refs[idx] = try p.rig.input(std.mem.sliceAsBytes(w16[0 .. ns * s.k]));
            p.y_refs[idx] = try p.rig.output(s.rows * ns * 2);
            p.base_out[idx] = try p.rig.scratch(s.rows * ns * 2);
        }
        const x_indices = [3]usize{ 0, 4, 5 };
        const nx: usize = if (p.segs == 4) 1 else 3;
        for (0..nx) |xi| p.x_refs[xi] = try p.rig.input(std.mem.sliceAsBytes(p.models[x_indices[xi]].?.x));
        return true;
    }

    pub fn count(p: *Probe) usize {
        var total: usize = 0;
        for (0..p.segs) |i| total += p.models[i].?.outputs();
        return total;
    }

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        if (p.segs == 4) {
            host.dense4(&p.pipes, pass.e, hr(p.x_refs[0]), .{ hr(p.w_refs[0]), hr(p.w_refs[1]), hr(p.w_refs[2]), hr(p.w_refs[3]) }, p.n_dims[0..4].*, .{ hr(p.y_refs[0]), hr(p.y_refs[1]), hr(p.y_refs[2]), hr(p.y_refs[3]) }, @intCast(p.shape.rows), @intCast(p.shape.k));
        } else {
            host.dense6(&p.pipes, pass.e, hr(p.x_refs[0]), hr(p.x_refs[1]), hr(p.x_refs[2]), .{ hr(p.w_refs[0]), hr(p.w_refs[1]), hr(p.w_refs[2]), hr(p.w_refs[3]), hr(p.w_refs[4]), hr(p.w_refs[5]) }, p.n_dims, .{ hr(p.y_refs[0]), hr(p.y_refs[1]), hr(p.y_refs[2]), hr(p.y_refs[3]), hr(p.y_refs[4]), hr(p.y_refs[5]) }, @intCast(p.shape.rows), @intCast(p.shape.k));
        }
        try p.rig.finish(pass);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        const s = p.shape;
        p.base_bytes.clearRetainingCapacity();
        for (0..p.segs) |idx| @memset(p.rig.bytes(p.base_out[idx], s.rows * p.n_dims[idx] * 2), 0xff);
        const pass = p.rig.begin();
        const e = pass.e;
        for (0..p.segs) |idx| {
            const x_slot: usize = if (idx == 4) 1 else if (idx == 5) 2 else 0;
            const n = p.n_dims[idx];
            var at: usize = 0;
            while (at < s.rows) {
                const left = s.rows - at;
                const take: usize = if (left >= 32) 32 else if (left >= 16) 16 else if (left >= 8) 8 else if (left >= 4) 4 else if (left >= 2) 2 else 1;
                const variant: usize = switch (take) {
                    1 => 0,
                    2 => 1,
                    4 => 2,
                    8 => 3,
                    16 => 4,
                    32 => 5,
                    else => unreachable,
                };
                e.setPipeline(p.base[variant]);
                e.setBuffer(p.x_refs[x_slot].buf, p.x_refs[x_slot].off + at * s.k * 2, 0);
                e.setBuffer(p.w_refs[idx].buf, p.w_refs[idx].off, 1);
                e.setValue(MvArgs{ .rows = @intCast(take), .n = n, .k = @intCast(s.k), .ldy = n }, 2);
                e.setBuffer(p.base_out[idx].buf, p.base_out[idx].off + at * n * 2, 3);
                e.dispatchGroups(mtl.Size.of((n + 7) / 8, 1, 1), mtl.Size.of(256, 1, 1));
                at += take;
            }
        }
        try p.rig.finish(pass);
        for (0..p.segs) |idx| try p.base_bytes.appendSlice(p.a, p.rig.bytes(p.base_out[idx], s.rows * p.n_dims[idx] * 2));
        return p.base_bytes.items;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        var off: usize = 0;
        for (0..p.segs) |i| {
            const d = p.models[i].?;
            const len = d.outputs();
            try d.reference(want[off..][0..len], mag[off..][0..len]);
            off += len;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        var off: usize = 0;
        for (0..p.segs) |i| {
            const d = p.models[i].?;
            const len = d.outputs();
            try d.simulate(acc, order, out[off..][0..len]);
            off += len;
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    fn add(s: *Suite, segs: u8, ctl: controls.Control, only: ?[]const u8) !void {
        const name = controls.label(ctl, segs);
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        probe.* = try Probe.init(s.a, s.device, s.queue, segs, ctl);
        errdefer probe.deinit();
        try contract.register(s.a, &s.probes, &s.kernels, probe, probe.kernel());
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]controls.Control{ .none, .order, .dense4_seg, .ldy }) |ctl| try s.add(4, ctl, only);
        for ([_]controls.Control{ .none, .order, .kv_input, .ldy }) |ctl| try s.add(6, ctl, only);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
