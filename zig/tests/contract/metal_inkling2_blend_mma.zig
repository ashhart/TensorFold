//! Inkling 2 Mini's loop blend on simdgroup matrices: each output equals the scalar blend's, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const loop = @import("tensorfold").inkling2_metal_host.loop;
const controls = @import("metal_inkling2_blend_mma_controls.zig");
const src = @import("inkling2_metal");

const name = "metal inkling2 mma tf_ink2_loop_blend_mma";
const hidden = 2304;
const rank = 64;
/// The scalar kernel's arguments: rows, and floats from one row's message to the next.
const Args = extern struct { rows: u32, stride: u32 };
const stride = 2 * hidden;

fn depth(_: c.Shape) usize {
    return rank;
}

pub fn spec(control: bool) c.Spec {
    return .{
        .name = if (control) controls.Mutation.ranks.name() else name,
        .accum = .none,
        .depth = depth,
        .out = .f32,
        .replaces = "tf_inkling2_loop_blend, on the same tables, zeta and rows",
        .nan = .{ .stated = "every output equals the scalar blend's: a non-finite input has the same class" },
        // Rows go 16 a tile; 37 rows leave the last tile part full.
        .tiles = .{ .{ .size = 16 }, .{}, .{} },
        .served = &.{ .{ .rows = 16, .n = hidden, .k = rank }, .{ .rows = 37, .n = hidden, .k = rank } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "the scalar blend's fp32 sums are the baseline, bit for bit" },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: loop.Pipelines,
    control: bool,
    shape: c.Shape = undefined,
    refs: [6]rig_.Ref = undefined,
    baseline: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: bool) !Probe {
        const text = try controls.apply(a, if (control) .ranks else .none);
        defer a.free(text);
        var pipes = try loop.Pipelines.loadSource(device, src.loop);
        errdefer pipes.deinit();
        try pipes.loadMma(device, text);
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .control = control };
    }

    pub fn deinit(p: *Probe) void {
        p.rig.reset();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .spec = spec(p.control), .ctx = p, .vt = &vt, .control = if (p.control) .shapes else null };
    }
    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n != hidden or shape.k != rank) return false;
        p.rig.reset();
        p.shape = shape;
        const rows = shape.rows;
        const padded = (rows + 15) / 16 * 16;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const h = try p.a.alloc(f32, rows * hidden);
        defer p.a.free(h);
        const m = try p.a.alloc(f32, rows * stride);
        defer p.a.free(m);
        // Zeta's tile rows past the call are read and not stored.
        const zeta = try p.a.alloc(u16, padded * rank);
        defer p.a.free(zeta);
        const forget = try p.a.alloc(u16, hidden * rank);
        defer p.a.free(forget);
        const gain = try p.a.alloc(u16, hidden * rank);
        defer p.a.free(gain);
        for (h) |*v| v.* = (rng.float(f32) * 2 - 1) * 3;
        for (m) |*v| v.* = (rng.float(f32) * 2 - 1) * 3;
        for (zeta) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (forget) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.4);
        for (gain) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.4);
        if (fill == .nonfinite) {
            h[0] = std.math.nan(f32);
            m[1] = std.math.inf(f32);
            zeta[2] = c.toBf16(-std.math.inf(f32));
        }
        const bytes = std.mem.sliceAsBytes;
        const out = try p.rig.output(p.count() * 4);
        p.refs = .{ try p.rig.input(bytes(h)), try p.rig.input(bytes(m)), try p.rig.input(bytes(zeta)), try p.rig.input(bytes(forget)), try p.rig.input(bytes(gain)), out };
        p.baseline = try p.rig.scratch(p.count() * 4);
        return true;
    }

    fn launch(p: *Probe, out: rig_.Ref, scalar: bool) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        for (p.refs[0..5], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setBuffer(out.buf, out.off, 5);
        e.setValue(Args{ .rows = @intCast(p.shape.rows), .stride = stride }, 6);
        if (scalar) {
            e.setPipeline(p.pipes.blend);
            e.dispatchThreads(mtl.Size.of(hidden, p.shape.rows, 1), mtl.Size.of(32, 1, 1));
        } else {
            e.setPipeline(p.pipes.blend_mma.?);
            e.dispatchGroups(mtl.Size.of(hidden / 16, (p.shape.rows + 15) / 16, 1), mtl.Size.of(32, 1, 1));
        }
        try p.rig.finish(pass);
    }

    pub fn run(p: *Probe) !void {
        try p.launch(p.refs[5], false);
    }

    pub fn replaced(p: *Probe) ![]const u8 {
        try p.launch(p.baseline, true);
        return p.rig.bytes(p.baseline, p.count() * 4);
    }

    pub fn count(p: *const Probe) usize {
        return p.shape.rows * hidden;
    }

    pub fn reference(_: *Probe, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Non-finite inputs: every output has the scalar blend's class, and a finite one its bytes.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        for (0..p.count()) |i| {
            const a: f32 = @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little));
            const b: f32 = @bitCast(std.mem.readInt(u32, want[4 * i ..][0..4], .little));
            if (std.math.isNan(a) and std.math.isNan(b)) continue;
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return "an output differs from the scalar blend's";
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [2]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    /// The entry exists where the GPU has no tensor units, the only GPUs whose loader builds the kernel.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (s.device.tensorUnits()) return;
        for ([_]bool{ false, true }) |control| {
            if (only) |needle| if (std.mem.indexOf(u8, spec(control).name, needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        }
    }
};
