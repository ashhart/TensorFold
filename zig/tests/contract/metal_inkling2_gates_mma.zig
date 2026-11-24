//! Inkling 2 Mini's loop gates on simdgroup matrices: each output equals the scalar gates', byte for byte.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const loop = @import("tensorfold").inkling2_metal_host.loop;
const controls = @import("metal_inkling2_gates_mma_controls.zig");
const src = @import("inkling2_metal");

const name = "metal inkling2 mma tf_inkling2_loop_gates_mma";
const hidden = 2304;
const rank = 64;
/// Floats from one row's message to the next: the scalar kernel's rows are `hidden` apart, the forward's are further.
const stride = 3 * hidden;

fn depth(_: c.Shape) usize {
    return 2 * hidden;
}

fn label(control: ?controls.Mutation) []const u8 {
    return if (control) |m| m.name() else name;
}

pub fn spec(control: ?controls.Mutation) c.Spec {
    return .{
        .name = label(control),
        .accum = .none,
        .depth = depth,
        .out = .f32,
        .replaces = "tf_inkling2_loop_gates, on the same tables and rows",
        .nan = .{ .stated = "every output equals the scalar gates': a non-finite input has the same class" },
        // Rows go 8 a threadgroup; 37 rows leave the last one part full.
        .tiles = .{ .{ .size = 8 }, .{}, .{} },
        .served = &.{ .{ .rows = 8, .n = hidden, .k = rank }, .{ .rows = 37, .n = hidden, .k = rank } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "the scalar gates' fp32 sums are the baseline, bit for bit" },
    };
}

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: loop.Pipelines,
    control: ?controls.Mutation,
    shape: c.Shape = undefined,
    refs: [6]rig_.Ref = undefined,
    baseline: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, control: ?controls.Mutation) !Probe {
        const text = try controls.apply(a, control orelse .none);
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
        return .{ .spec = spec(p.control), .ctx = p, .vt = &vt, .control = if (p.control) |m| m.gate() else null };
    }
    const vt = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n != hidden or shape.k != rank) return false;
        p.rig.reset();
        p.shape = shape;
        const rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const h = try p.a.alloc(f32, rows * hidden);
        defer p.a.free(h);
        const m = try p.a.alloc(f32, rows * stride);
        defer p.a.free(m);
        const down = try p.a.alloc(u16, rank * 2 * hidden);
        defer p.a.free(down);
        const forget = try p.a.alloc(u16, hidden * rank);
        defer p.a.free(forget);
        const gain = try p.a.alloc(u16, hidden * rank);
        defer p.a.free(gain);
        for (h) |*v| v.* = (rng.float(f32) * 2 - 1) * 3;
        for (m) |*v| v.* = (rng.float(f32) * 2 - 1) * 3;
        for (down) |*v| v.* = c.toBf16(rng.float(f32) * 2 - 1);
        for (forget) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.4);
        for (gain) |*v| v.* = c.toBf16((rng.float(f32) * 2 - 1) * 0.4);
        // Equal huge inputs with opposite huge weights cancel; the sums keep what each order of adding lost.
        const marks = [_][2]usize{ .{ 100, 1900 }, .{ hidden + 300, hidden + 2100 } };
        for (marks) |pair| {
            for (0..rows) |row| for (pair) |d| {
                if (d < hidden) h[row * hidden + d] = 1000 else m[row * stride + d - hidden] = 1000;
            };
            for (0..rank) |r| {
                down[r * 2 * hidden + pair[0]] = c.toBf16(16384);
                down[r * 2 * hidden + pair[1]] = c.toBf16(-16384);
            }
        }
        if (fill == .nonfinite) {
            h[0] = std.math.nan(f32);
            m[1] = std.math.inf(f32);
            down[2] = c.toBf16(-std.math.inf(f32));
        }
        const bytes = std.mem.sliceAsBytes;
        const out = try p.rig.output(p.count() * 4);
        p.refs = .{ try p.rig.input(bytes(h)), try p.rig.input(bytes(m)), try p.rig.input(bytes(down)), try p.rig.input(bytes(forget)), try p.rig.input(bytes(gain)), out };
        p.baseline = try p.rig.scratch(p.count() * 4);
        return true;
    }

    fn hr(r: rig_.Ref, bytes: usize) loop.Ref {
        return .{ .buf = r.buf, .off = r.off + bytes };
    }

    fn launch(p: *Probe, out: rig_.Ref, scalar: bool) !void {
        const pass = p.rig.begin();
        const e = pass.e;
        const rows: u32 = @intCast(p.shape.rows);
        if (scalar) {
            for (0..rows) |row| loop.gates(&p.pipes, e, hr(p.refs[0], row * hidden * 4), hr(p.refs[1], row * stride * 4), hr(p.refs[2], 0), hr(p.refs[3], 0), hr(p.refs[4], 0), hr(out, row * hidden * 4), 1);
        } else {
            try loop.gatesRows(&p.pipes, e, hr(p.refs[0], 0), hr(p.refs[1], 0), stride, hr(p.refs[2], 0), hr(p.refs[3], 0), hr(p.refs[4], 0), hr(out, 0), rows);
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

    /// Non-finite inputs: every output has the scalar gates' class, and a finite one its bytes.
    pub fn check(p: *Probe, _: g.Gate) !?[]const u8 {
        const got = try p.rig.outputBytes();
        const want = try p.replaced();
        for (0..p.count()) |i| {
            const a: f32 = @bitCast(std.mem.readInt(u32, got[4 * i ..][0..4], .little));
            const b: f32 = @bitCast(std.mem.readInt(u32, want[4 * i ..][0..4], .little));
            if (std.math.isNan(a) and std.math.isNan(b)) continue;
            if (@as(u32, @bitCast(a)) != @as(u32, @bitCast(b))) return "an output differs from the scalar gates'";
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [3]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    /// The entries exist where the GPU has no tensor units, the only GPUs whose loader builds the kernel.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (s.device.tensorUnits()) return;
        for ([_]?controls.Mutation{ null, .chunks, .ranks }) |control| {
            if (only) |needle| if (std.mem.indexOf(u8, label(control), needle) == null) continue;
            s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, control);
            s.made += 1;
            try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
        }
    }
};
