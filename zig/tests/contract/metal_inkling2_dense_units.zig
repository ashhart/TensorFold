//! Inkling 2 Mini BF16 dense tensor-unit kernels under Gates 1-4, width-seam bit-equality, and planted controls.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.dense_units;
const controls = @import("metal_inkling2_dense_units_controls.zig");
const src = @import("inkling2_metal");

pub const Kind = enum { dense_16, dense_32, dense_64, dense6_16, dense6_32, dense6_64 };

const served_shapes = [_]c.Shape{
    .{ .rows = 1, .n = 24, .k = 2304 },
    .{ .rows = 5, .n = 384, .k = 2304 },
    .{ .rows = 17, .n = 1024, .k = 2304 },
    .{ .rows = 65, .n = 384, .k = 3072 },
    .{ .rows = 5, .n = 2304, .k = 3072 },
};

fn isFused6(k: Kind) bool {
    return switch (k) {
        .dense6_16, .dense6_32, .dense6_64 => true,
        else => false,
    };
}

fn band(k: Kind) host.Band {
    return switch (k) {
        .dense_16, .dense6_16 => .b16,
        .dense_32, .dense6_32 => .b32,
        .dense_64, .dense6_64 => .b64,
    };
}

fn symbol(k: Kind) []const u8 {
    return switch (k) {
        .dense_16 => "tf_ink2_dense_units_16",
        .dense_32 => "tf_ink2_dense_units_32",
        .dense_64 => "tf_ink2_dense_units_64",
        .dense6_16 => "tf_ink2_dense6_units_16",
        .dense6_32 => "tf_ink2_dense6_units_32",
        .dense6_64 => "tf_ink2_dense6_units_64",
    };
}

fn mutation(k: Kind) controls.Mutation {
    return switch (k) {
        .dense_16 => .dense_16,
        .dense_32 => .dense_32,
        .dense_64 => .dense_64,
        .dense6_16 => .dense6_16,
        .dense6_32 => .dense6_32,
        .dense6_64 => .dense6_64,
    };
}

fn depth(s: c.Shape) usize {
    return s.k / 128 * contract.attention_range.mma_step + 8;
}

pub fn spec(k: Kind, control: bool) c.Spec {
    const base = if (control) mutation(k).name() else switch (k) {
        .dense_16 => "metal inkling2 dense_units tf_ink2_dense_units_16",
        .dense_32 => "metal inkling2 dense_units tf_ink2_dense_units_32",
        .dense_64 => "metal inkling2 dense_units tf_ink2_dense_units_64",
        .dense6_16 => "metal inkling2 dense_units tf_ink2_dense6_units_16",
        .dense6_32 => "metal inkling2 dense_units tf_ink2_dense6_units_32",
        .dense6_64 => "metal inkling2 dense_units tf_ink2_dense6_units_64",
    };
    return .{
        .name = base,
        .accum = .f32,
        .depth = depth,
        .out = .bf16,
        .tiles = .{
            .{ .size = host.bandRows(band(k)) },
            .{ .size = 32 },
            .{ .size = 256, .step = 128, .min = 128 },
        },
        .served = &served_shapes,
    };
}

fn xIndex6(proj: usize) usize {
    return switch (proj) {
        1 => 1,
        2 => 2,
        else => 0,
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    kind: Kind,
    ctl: controls.Mutation,
    shape: c.Shape = undefined,
    fill: g.Fill = .random,
    ns6: [6]usize = .{ 0, 0, 0, 0, 0, 0 },
    xs: [3][]u16 = .{ &.{}, &.{}, &.{} },
    ws: [6][]u16 = .{ &.{}, &.{}, &.{}, &.{}, &.{}, &.{} },
    in_refs: [9]rig_.Ref = undefined,
    out_refs: [6]rig_.Ref = undefined,
    seam_enabled: bool = false,
    seam_y1: rig_.Ref = undefined,
    seam_y17: rig_.Ref = undefined,
    seam_y65: rig_.Ref = undefined,
    seam_512: bool = false,
    seam_x512: rig_.Ref = undefined,
    seam_y512: rig_.Ref = undefined,
    seam_y1_r17: rig_.Ref = undefined,
    seam_y1_r64: rig_.Ref = undefined,
    seam_y6_1: [6]rig_.Ref = undefined,
    seam_y6_sep: [6]rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, control: bool) !Probe {
        const mut = if (control) mutation(kind) else .none;
        const text = try controls.apply(a, mut);
        defer a.free(text);
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipes = try host.Pipelines.loadSource(device, text),
            .kind = kind,
            .ctl = mut,
        };
    }

    fn clear(p: *Probe) void {
        for (&p.xs) |*x| {
            p.a.free(x.*);
            x.* = &.{};
        }
        for (&p.ws) |*w| {
            p.a.free(w.*);
            w.* = &.{};
        }
        p.seam_enabled = false;
        p.seam_512 = false;
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.kind, p.ctl != .none), .ctx = p, .vt = &vt, .control = p.ctl.gate() };
    }
    const vt = g.hooks(Probe);

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    fn fillX(dst: []u16, rows: usize, k: usize, fill: g.Fill, rng: std.Random) void {
        for (dst) |*v| v.* = c.toBf16(rng.float(f32) * 4 - 2);
        switch (fill) {
            .random => {},
            .nonfinite => {
                const at = [_][2]usize{ .{ 0, k / 3 }, .{ 1, k / 2 }, .{ 2, k - 1 }, .{ 3, 0 }, .{ 3, k - 1 } };
                const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32), std.math.inf(f32), -std.math.inf(f32) };
                for (at, put) |pos, val| if (pos[0] < rows) {
                    dst[pos[0] * k + pos[1]] = c.toBf16(val);
                };
            },
            .stress => {
                const h = k / 2;
                for (0..rows) |r| for (0..h) |i| {
                    const v: f32 = @floatFromInt(2 * (128 + rng.uintLessThan(u32, 127)));
                    dst[r * k + i] = c.toBf16(v);
                    dst[r * k + h + i] = c.toBf16(v + if (i < 8) @as(f32, 2) else 0);
                };
            },
        }
    }

    fn fillW(dst: []u16, n: usize, k: usize, fill: g.Fill, rng: std.Random) void {
        switch (fill) {
            .random => {
                for (dst) |*v| v.* = c.toBf16(rng.float(f32) - 0.5);
            },
            .nonfinite => {
                for (dst) |*v| {
                    const r = rng.float(f32) - 0.5;
                    const bits = c.toBf16(r);
                    v.* = if ((bits & 0x7fff) == 0) c.toBf16(0.25) else bits;
                }
            },
            .stress => {
                const h = k / 2;
                for (0..n) |j| for (0..h) |i| {
                    const q: f32 = @floatFromInt(1 + rng.uintLessThan(u32, 8));
                    dst[j * k + i] = c.toBf16(q);
                    dst[j * k + h + i] = c.toBf16(-q);
                };
            },
        }
    }

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 128 != 0) return false;
        p.clear();
        p.shape = shape;
        p.fill = fill;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const fused = isFused6(p.kind);
        const nx: usize = if (fused) 3 else 1;
        const np: usize = if (fused) 6 else 1;
        if (fused) {
            p.ns6 = .{
                shape.n,
                @max(@as(usize, 1), shape.n / 3),
                @max(@as(usize, 1), shape.n / 2),
                24,
                24,
                shape.n,
            };
        } else {
            p.ns6[0] = shape.n;
        }
        for (0..nx) |i| {
            p.xs[i] = try p.a.alloc(u16, shape.rows * shape.k);
            fillX(p.xs[i], shape.rows, shape.k, fill, rng);
            p.in_refs[i] = try p.rig.input(std.mem.sliceAsBytes(p.xs[i]));
        }
        for (0..np) |i| {
            p.ws[i] = try p.a.alloc(u16, p.ns6[i] * shape.k);
            fillW(p.ws[i], p.ns6[i], shape.k, fill, rng);
            p.in_refs[nx + i] = try p.rig.input(std.mem.sliceAsBytes(p.ws[i]));
            p.out_refs[i] = try p.rig.output(shape.rows * p.ns6[i] * 2);
        }
        if (p.ctl == .none and fill == .random and shape.rows == 65) {
            p.seam_enabled = true;
            if (!fused) {
                p.seam_y1 = try p.rig.scratch(shape.n * 2);
                p.seam_y17 = try p.rig.scratch(17 * shape.n * 2);
                p.seam_y65 = try p.rig.scratch(65 * shape.n * 2);
                if (shape.n == 24 and shape.k == 2304) {
                    p.seam_512 = true;
                    p.seam_x512 = try p.rig.scratch(512 * shape.k * 2);
                    const dst = std.mem.bytesAsSlice(u16, p.rig.bytes(p.seam_x512, 512 * shape.k * 2));
                    const row_len = shape.rows * shape.k;
                    var pos: usize = 0;
                    while (pos < 512 * shape.k) {
                        const take = @min(row_len, 512 * shape.k - pos);
                        @memcpy(dst[pos..][0..take], p.xs[0][0..take]);
                        pos += take;
                    }
                    p.seam_y512 = try p.rig.scratch(512 * shape.n * 2);
                    p.seam_y1_r17 = try p.rig.scratch(shape.n * 2);
                    p.seam_y1_r64 = try p.rig.scratch(shape.n * 2);
                }
            } else {
                for (0..6) |i| {
                    p.seam_y6_1[i] = try p.rig.scratch(p.ns6[i] * 2);
                    p.seam_y6_sep[i] = try p.rig.scratch(65 * p.ns6[i] * 2);
                }
            }
        }
        return true;
    }

    fn bufs6(p: *const Probe, outs: [6]rig_.Ref) host.Buffers6 {
        return .{
            .xq = hr(p.in_refs[0]),
            .xk = hr(p.in_refs[1]),
            .xv = hr(p.in_refs[2]),
            .wq = hr(p.in_refs[3]),
            .wk = hr(p.in_refs[4]),
            .wv = hr(p.in_refs[5]),
            .wr = hr(p.in_refs[6]),
            .ws = hr(p.in_refs[7]),
            .wg = hr(p.in_refs[8]),
            .yq = hr(outs[0]),
            .yk = hr(outs[1]),
            .yv = hr(outs[2]),
            .yr = hr(outs[3]),
            .ys = hr(outs[4]),
            .yg = hr(outs[5]),
        };
    }

    fn call6(p: *const Probe, rows: u32) host.Call6 {
        return .{
            .rows = rows,
            .k = @intCast(p.shape.k),
            .nq = @intCast(p.ns6[0]),
            .nk = @intCast(p.ns6[1]),
            .nv = @intCast(p.ns6[2]),
            .nr = @intCast(p.ns6[3]),
            .ns = @intCast(p.ns6[4]),
            .ng = @intCast(p.ns6[5]),
        };
    }

    fn runSeamChecks(p: *Probe) !void {
        const pass = p.rig.begin();
        if (!isFused6(p.kind)) {
            const x = hr(p.in_refs[0]);
            const w = hr(p.in_refs[1]);
            const n: u32 = @intCast(p.shape.n);
            const k: u32 = @intCast(p.shape.k);
            try host.encode(&p.pipes, pass.e, x, w, hr(p.seam_y1), .{ .rows = 1, .n = n, .k = k });
            try host.encode(&p.pipes, pass.e, x, w, hr(p.seam_y17), .{ .rows = 17, .n = n, .k = k });
            try host.encode(&p.pipes, pass.e, x, w, hr(p.seam_y65), .{ .rows = 65, .n = n, .k = k });
            if (p.seam_512) {
                const x17 = host.Ref{ .buf = p.in_refs[0].buf, .off = p.in_refs[0].off + 17 * p.shape.k * 2 };
                const x64 = host.Ref{ .buf = p.in_refs[0].buf, .off = p.in_refs[0].off + 64 * p.shape.k * 2 };
                try host.encode(&p.pipes, pass.e, hr(p.seam_x512), w, hr(p.seam_y512), .{ .rows = 512, .n = n, .k = k });
                try host.encode(&p.pipes, pass.e, x17, w, hr(p.seam_y1_r17), .{ .rows = 1, .n = n, .k = k });
                try host.encode(&p.pipes, pass.e, x64, w, hr(p.seam_y1_r64), .{ .rows = 1, .n = n, .k = k });
            }
            try p.rig.finish(pass);
            const row_bytes = p.shape.n * 2;
            const y1 = p.rig.bytes(p.seam_y1, row_bytes);
            const y_probe = p.rig.bytes(p.out_refs[0], row_bytes);
            const y17 = p.rig.bytes(p.seam_y17, row_bytes);
            const y65 = p.rig.bytes(p.seam_y65, row_bytes);
            if (!std.mem.eql(u8, y1, y_probe) or !std.mem.eql(u8, y1, y17) or !std.mem.eql(u8, y1, y65)) {
                std.debug.print("FAIL seam {s} {d}x{d}: rows=1 differs from rows=17/65 row 0\n", .{ symbol(p.kind), p.shape.n, p.shape.k });
                return error.WidthSeamMismatch;
            }
            if (p.seam_512) {
                const y512 = p.rig.bytes(p.seam_y512, 512 * row_bytes);
                const r17 = p.rig.bytes(p.seam_y1_r17, row_bytes);
                const r64 = p.rig.bytes(p.seam_y1_r64, row_bytes);
                if (!std.mem.eql(u8, y1, y512[0..row_bytes]) or
                    !std.mem.eql(u8, r17, y512[17 * row_bytes ..][0..row_bytes]) or
                    !std.mem.eql(u8, r64, y512[64 * row_bytes ..][0..row_bytes]))
                {
                    std.debug.print("FAIL seam {s} 512x{d}x{d}: rows=1 slice differs from rows=512\n", .{ symbol(p.kind), p.shape.n, p.shape.k });
                    return error.WidthSeamMismatch;
                }
            }
            return;
        }
        try host.encode6(&p.pipes, pass.e, p.bufs6(p.seam_y6_1), p.call6(1));
        for (0..6) |i| {
            const x = hr(p.in_refs[xIndex6(i)]);
            const w = hr(p.in_refs[3 + i]);
            try host.encode(&p.pipes, pass.e, x, w, hr(p.seam_y6_sep[i]), .{ .rows = 65, .n = @intCast(p.ns6[i]), .k = @intCast(p.shape.k) });
        }
        try p.rig.finish(pass);
        for (0..6) |i| {
            const row_bytes = p.ns6[i] * 2;
            const total_bytes = 65 * row_bytes;
            const y65 = p.rig.bytes(p.out_refs[i], total_bytes);
            const y1 = p.rig.bytes(p.seam_y6_1[i], row_bytes);
            const y_sep = p.rig.bytes(p.seam_y6_sep[i], total_bytes);
            if (!std.mem.eql(u8, y1, y65[0..row_bytes]) or !std.mem.eql(u8, y65, y_sep)) {
                std.debug.print("FAIL dense6 seam/eq {s} proj {d}: differs from rows=1 or separate dense_units\n", .{ symbol(p.kind), i });
                return error.WidthSeamMismatch;
            }
        }
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        if (!isFused6(p.kind)) {
            try host.encodeBand(&p.pipes, pass.e, band(p.kind), hr(p.in_refs[0]), hr(p.in_refs[1]), hr(p.out_refs[0]), .{
                .rows = @intCast(p.shape.rows),
                .n = @intCast(p.shape.n),
                .k = @intCast(p.shape.k),
            });
        } else {
            try host.encode6Band(&p.pipes, pass.e, band(p.kind), p.bufs6(p.out_refs), p.call6(@intCast(p.shape.rows)));
        }
        try p.rig.finish(pass);
        if (p.seam_enabled) try p.runSeamChecks();
    }

    pub fn count(p: *const Probe) usize {
        if (!isFused6(p.kind)) return p.shape.rows * p.shape.n;
        var sum: usize = 0;
        for (p.ns6) |n| sum += p.shape.rows * n;
        return sum;
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const k = p.shape.k;
        const np: usize = if (isFused6(p.kind)) 6 else 1;
        var base: usize = 0;
        for (0..np) |proj| {
            const nj = if (isFused6(p.kind)) p.ns6[proj] else p.shape.n;
            const x_slice = p.xs[if (isFused6(p.kind)) xIndex6(proj) else 0];
            const w_slice = p.ws[proj];
            for (0..p.shape.rows) |r| for (0..nj) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..k) |col| {
                    const term = @as(f64, c.fromBf16(x_slice[r * k + col])) * c.fromBf16(w_slice[j * k + col]);
                    sum += term;
                    abs += @abs(term);
                }
                want[base + r * nj + j] = sum;
                mag[base + r * nj + j] = abs;
            };
            base += p.shape.rows * nj;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const k = p.shape.k;
        const t = try p.a.alloc(f64, k);
        defer p.a.free(t);
        const np: usize = if (isFused6(p.kind)) 6 else 1;
        var base: usize = 0;
        for (0..np) |proj| {
            const nj = if (isFused6(p.kind)) p.ns6[proj] else p.shape.n;
            const x_slice = p.xs[if (isFused6(p.kind)) xIndex6(proj) else 0];
            const w_slice = p.ws[proj];
            for (0..p.shape.rows) |r| for (0..nj) |j| {
                for (0..k) |col| {
                    t[col] = @as(f64, c.fromBf16(x_slice[r * k + col])) * c.fromBf16(w_slice[j * k + col]);
                }
                out[base + r * nj + j] = contract.rows.sumIn(acc, order, t);
            };
            base += p.shape.rows * nj;
        }
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [12]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    /// Runs only on GPUs with tensor units; elsewhere M3 kernels serve every dense width.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for (std.enums.values(Kind)) |kind| for ([_]bool{ false, true }) |control| {
            const label = if (control) mutation(kind).name() else spec(kind, false).name;
            if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, kind, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        };
    }
};
