//! Ling MLA extras under the contract: GPT-J RoPE on cache and q, and per-head gate expand.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const controls = @import("metal_ling_mla_controls.zig");

const c = contract.spec;
const g = contract.gates;
const Which = controls.Which;
const Mutation = controls.Mutation;

const LAT: u32 = 512;
const ROPE: u32 = 64;
const KEY: u32 = LAT + ROPE;
const NOPE: u32 = 128;
const VD: u32 = 128;
const QD: u32 = NOPE + ROPE;
const NFREQ: u32 = ROPE / 2;
const HEADS: u32 = 16;
const THETA: f64 = 6_000_000;
const no_sums: c.Sums = .{ .f16 = 0, .bf16 = 0, .why = "no sums: elementwise rotate or a broadcast" };

const MlaRow = extern struct { slot: u32, pos: u32 };
const MlaArgs = extern struct {
    cache: u64,
    slot_keys: u32,
    heads: u32,
    head0: u32,
    rows: u32,
    q_stride: u32,
    kv_stride: u32,
    gate_stride: u32,
    out_stride: u32,
    eps: f32,
    scale: f32,
};
comptime {
    if (@sizeOf(MlaArgs) != 48) @compileError("MlaArgs must stay 48 bytes, matching k3");
}

fn none(_: c.Shape) usize {
    return 0;
}

fn invFreq(i: usize) f64 {
    return 1.0 / std.math.pow(f64, THETA, @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(ROPE)));
}

pub fn spec(w: Which) c.Spec {
    const rope = [_]c.Shape{ .{ .rows = 8, .n = HEADS, .k = ROPE }, .{ .rows = 1, .n = HEADS, .k = ROPE } };
    const cache = [_]c.Shape{ .{ .rows = 8, .n = 1, .k = ROPE }, .{ .rows = 1, .n = 1, .k = ROPE } };
    const gate = [_]c.Shape{ .{ .rows = 8, .n = HEADS * VD, .k = HEADS }, .{ .rows = 1, .n = HEADS * VD, .k = HEADS } };
    return switch (w) {
        .rope_cache => .{ .name = "metal ling mla tf_ling_mla_rope_cache", .accum = .none, .depth = none, .ops = 2, .mids = 1, .out = .bf16, .in_place = true, .served = &cache, .sums = no_sums },
        .rope_q => .{ .name = "metal ling mla tf_ling_mla_rope_q", .accum = .none, .depth = none, .ops = 2, .mids = 1, .out = .bf16, .in_place = true, .served = &rope, .sums = no_sums },
        .gate_expand => .{ .name = "metal ling mla tf_ling_mla_gate_expand", .accum = .none, .depth = none, .out = .bf16, .served = &gate, .sums = no_sums },
    };
}

fn random16(rng: std.Random, out: []u16, lo: f32, hi: f32) void {
    for (out) |*v| v.* = c.toBf16(lo + rng.float(f32) * (hi - lo));
}

fn plant(v: []u16, stride: usize, rows: usize, col: usize) void {
    const put = [_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) };
    for (put, 0..) |f, r| if (r < rows) {
        v[r * stride + col] = c.toBf16(f);
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    which: Which,
    mutation: Mutation,
    lib: mtl.Library,
    pipe: mtl.Pipeline,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    q: []u16 = &.{},
    cache: []u16 = &.{},
    gate: []u16 = &.{},
    cs: []f32 = &.{},
    rows: []MlaRow = &.{},
    in_cs: rig_.Ref = undefined,
    in_rows: rig_.Ref = undefined,
    in_g: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, which: Which, mutation: Mutation) !Probe {
        const plain = @import("ling_metal").mla;
        const src = try controls.apply(a, plain, mutation);
        defer a.free(src);
        const lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer lib.deinit();
        return .{
            .a = a,
            .rig = .{ .a = a, .device = device, .queue = queue },
            .which = which,
            .mutation = mutation,
            .lib = lib,
            .pipe = try mtl.Pipeline.init(device, lib, controls.function(which), false),
        };
    }

    fn drop(p: *Probe) void {
        for ([_]*[]u16{ &p.q, &p.cache, &p.gate }) |s| {
            p.a.free(s.*);
            s.* = &.{};
        }
        p.a.free(p.cs);
        p.cs = &.{};
        p.a.free(p.rows);
        p.rows = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.drop();
        p.rig.deinit();
        p.pipe.deinit();
        p.lib.deinit();
    }

    const vt = g.hooks(Probe);

    pub fn kernel(p: *Probe) g.Kernel {
        var k: g.Kernel = .{ .spec = spec(p.which), .ctx = p, .vt = &vt };
        if (p.mutation != .none) {
            k.spec.name = p.mutation.name();
            k.control = p.mutation.gate();
        }
        return k;
    }

    pub fn prepare(p: *Probe, s: c.Shape, fill: g.Fill, seed: u64) !bool {
        switch (p.which) {
            .rope_cache => if (s.k != ROPE or s.n != 1) return false,
            .rope_q => if (s.k != ROPE or s.n != HEADS) return false,
            .gate_expand => if (s.n != HEADS * VD or s.k != HEADS) return false,
        }
        if (s.rows == 0) return false;
        p.drop();
        p.shape = s;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows: u32 = @intCast(s.rows);
        p.rows = try p.a.alloc(MlaRow, rows);
        for (p.rows, 0..) |*r, t| r.* = .{ .slot = 0, .pos = @intCast(t) };
        p.cs = try p.a.alloc(f32, rows * NFREQ * 2);
        for (p.rows, 0..) |r, t| for (0..NFREQ) |i| {
            const ang = @as(f64, @floatFromInt(r.pos)) * invFreq(i);
            const at = (t * NFREQ + i) * 2;
            p.cs[at] = @floatCast(@cos(ang));
            p.cs[at + 1] = @floatCast(@sin(ang));
        };
        p.in_cs = try p.rig.input(std.mem.sliceAsBytes(p.cs));
        switch (p.which) {
            .rope_cache => {
                p.cache = try p.a.alloc(u16, rows * KEY);
                random16(rng, p.cache, -3, 3);
                if (fill == .nonfinite) plant(p.cache, KEY, s.rows, LAT);
                p.out = try p.rig.output(p.cache.len * 2);
            },
            .rope_q => {
                p.q = try p.a.alloc(u16, rows * HEADS * QD);
                random16(rng, p.q, -3, 3);
                if (fill == .nonfinite) plant(p.q, HEADS * QD, s.rows, NOPE);
                p.out = try p.rig.output(p.q.len * 2);
            },
            .gate_expand => {
                p.gate = try p.a.alloc(u16, rows * HEADS);
                random16(rng, p.gate, -2, 2);
                if (fill == .nonfinite) plant(p.gate, HEADS, s.rows, 0);
                p.in_g = try p.rig.input(std.mem.sliceAsBytes(p.gate));
                p.out = try p.rig.output(rows * HEADS * VD * 2);
            },
        }
        if (p.which == .rope_cache) p.in_rows = try p.rig.input(std.mem.sliceAsBytes(p.rows));
        return true;
    }

    pub fn count(p: *Probe) usize {
        const rows = p.shape.rows;
        return switch (p.which) {
            .rope_cache => rows * KEY,
            .rope_q => rows * HEADS * QD,
            .gate_expand => rows * HEADS * VD,
        };
    }

    fn args(p: *Probe) MlaArgs {
        const rows: u32 = @intCast(p.shape.rows);
        return .{
            .cache = if (p.which == .rope_cache) p.out.buf.gpuAddress() + p.out.off else 0,
            .slot_keys = rows,
            .heads = HEADS,
            .head0 = 0,
            .rows = rows,
            .q_stride = HEADS * QD,
            .kv_stride = KEY,
            .gate_stride = HEADS * VD,
            .out_stride = HEADS * VD,
            .eps = 1e-6,
            .scale = 1,
        };
    }

    pub fn run(p: *Probe) !void {
        const rows = p.shape.rows;
        switch (p.which) {
            .rope_cache => @memcpy(p.rig.bytes(p.out, p.cache.len * 2), std.mem.sliceAsBytes(p.cache)),
            .rope_q => @memcpy(p.rig.bytes(p.out, p.q.len * 2), std.mem.sliceAsBytes(p.q)),
            .gate_expand => {},
        }
        const pass = p.rig.begin();
        const e = pass.e;
        e.setPipeline(p.pipe);
        const aa = p.args();
        switch (p.which) {
            .rope_cache => {
                e.setBuffer(p.in_rows.buf, p.in_rows.off, 0);
                e.setValue(aa, 1);
                e.setBuffer(p.in_cs.buf, p.in_cs.off, 2);
                e.useResource(p.out.buf, mtl.ResourceUsage.read | mtl.ResourceUsage.write);
                e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
            },
            .rope_q => {
                e.setBuffer(p.out.buf, p.out.off, 0);
                e.setValue(aa, 1);
                e.setBuffer(p.in_cs.buf, p.in_cs.off, 2);
                e.dispatchGroups(mtl.Size.of(HEADS, rows, 1), mtl.Size.of(32, 1, 1));
            },
            .gate_expand => {
                e.setBuffer(p.in_g.buf, p.in_g.off, 0);
                e.setBuffer(p.out.buf, p.out.off, 1);
                e.setValue(aa, 2);
                e.dispatchThreads(mtl.Size.of(HEADS * VD, rows, 1), mtl.Size.of(128, 1, 1));
            },
        }
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        const rows = p.shape.rows;
        switch (p.which) {
            .rope_cache => for (0..rows) |t| {
                const z = p.cache[t * KEY ..][0..KEY];
                for (0..LAT) |i| {
                    want[t * KEY + i] = c.fromBf16(z[i]);
                    mag[t * KEY + i] = @abs(want[t * KEY + i]);
                }
                rotateRef(want[t * KEY + LAT ..][0..ROPE], mag[t * KEY + LAT ..][0..ROPE], z[LAT..][0..ROPE], p.cs[t * NFREQ * 2 ..][0 .. NFREQ * 2]);
            },
            .rope_q => for (0..rows) |t| for (0..HEADS) |h| {
                const base = (t * HEADS + h) * QD;
                const z = p.q[base..][0..QD];
                for (0..NOPE) |i| {
                    want[base + i] = c.fromBf16(z[i]);
                    mag[base + i] = @abs(want[base + i]);
                }
                rotateRef(want[base + NOPE ..][0..ROPE], mag[base + NOPE ..][0..ROPE], z[NOPE..][0..ROPE], p.cs[t * NFREQ * 2 ..][0 .. NFREQ * 2]);
            },
            .gate_expand => for (0..rows) |t| for (0..HEADS) |h| {
                const v = c.fromBf16(p.gate[t * HEADS + h]);
                const lo = (t * HEADS + h) * VD;
                for (0..VD) |d| {
                    want[lo + d] = v;
                    mag[lo + d] = @abs(v);
                }
            },
        }
    }
};

fn rotateRef(want: []f64, mag: []f64, z: []const u16, cs: []const f32) void {
    for (0..NFREQ) |i| {
        const x0: f64 = c.fromBf16(z[2 * i]);
        const x1: f64 = c.fromBf16(z[2 * i + 1]);
        const cv: f64 = cs[i * 2];
        const sv: f64 = cs[i * 2 + 1];
        want[i] = x0 * cv - x1 * sv;
        want[NFREQ + i] = x1 * cv + x0 * sv;
        mag[i] = @abs(x0) * @abs(cv) + @abs(x1) * @abs(sv);
        mag[NFREQ + i] = mag[i];
    }
}

fn wanted(only: ?[]const u8, name: []const u8) bool {
    const needle = only orelse return true;
    return std.mem.indexOf(u8, name, needle) != null;
}

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    probes: std.ArrayList(*Probe) = .empty,

    pub fn collect(s: *Suite, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !void {
        for (std.enums.values(Which)) |w| {
            const name = spec(w).name;
            if (!wanted(only, name)) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, w, .none);
            errdefer p.deinit();
            try @import("contract").register(s.a, &s.probes, &kernels, p, p.kernel());
        }
        for (std.enums.values(Mutation)) |m| {
            if (m == .none or !wanted(only, m.name())) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, m.which(), m);
            errdefer p.deinit();
            try @import("contract").register(s.a, &s.probes, &kernels, p, p.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
