//! qwen27's norm_xs, norm_input and mlp_xs built without Q27_XS (#2900): the rows' bits of the XS build, no XS written.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const contract = @import("contract");
const rig_ = @import("rig.zig");

const c = contract.spec;
const g = contract.gates;
const quant = tf.qwen27.quant_gpu;

/// The 27B's hidden width (the norms' K) and its MLP width (mlp_xs' N).
const hidden = 5120;
const width = 17408;

pub const Which = enum { norm, input_norm, mlp };

pub const Reg = struct { which: Which, control: bool = false };

const rule = "bits only: every output equals the XS build's, NaN and both infinities included";

fn noSums(_: c.Shape) usize {
    return 0;
}

pub fn name(r: Reg) []const u8 {
    if (r.control) return "control: qwen27 norm without XS drops its normed rows";
    return switch (r.which) {
        .norm => "metal qwen27 norm without XS q27_norm_xs",
        .input_norm => "metal qwen27 input norm without XS q27_norm_input",
        .mlp => "metal qwen27 mlp without XS q27_mlp_xs",
    };
}

fn kind(w: Which) quant.Kind {
    return switch (w) {
        .norm => .norm,
        .input_norm => .input_norm,
        .mlp => .mlp,
    };
}

/// The rows a decode window, a verify window and a prompt chunk give these kernels, padded or not.
fn served(n: usize) []const c.Shape {
    return switch (n) {
        inline hidden, width => |x| comptime &.{ .{ .rows = 1, .n = x, .k = x }, .{ .rows = 5, .n = x, .k = x }, .{ .rows = 16, .n = x, .k = x }, .{ .rows = 17, .n = x, .k = x }, .{ .rows = 64, .n = x, .k = x } },
        else => unreachable,
    };
}

pub fn spec(r: Reg) c.Spec {
    const n: usize = if (r.which == .mlp) width else hidden;
    return .{ .name = name(r), .accum = .none, .depth = noSums, .out = .raw, .in_place = false, .nan = .{ .stated = rule }, .tiles = .{ .{ .size = 16 }, .{}, .{} }, .served = served(n) };
}

pub const Xs = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    reg: Reg,
    /// The kernel without Q27_XS (the one under test), and the XS build it must equal.
    plain: mtl.Pipeline,
    sums: mtl.Pipeline,
    rows: usize = 0,
    ins: [5]rig_.Ref = undefined,
    outs: [2]rig_.Ref = undefined,
    /// The XS build's outputs and sums, and the plain build's XS buffer (filled with a pattern it must keep).
    base: [2]rig_.Ref = undefined,
    base_xs: rig_.Ref = undefined,
    xs: rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, r: Reg) !Xs {
        const key: quant.Key = .{ .kind = kind(r.which), .k = if (r.which == .mlp) width else hidden, .n = width };
        const with = try quant.specializedSource(a, key, true);
        defer a.free(with);
        var without = try quant.specializedSource(a, key, false);
        defer a.free(without);
        if (r.control) { // the plain build's normed rows go missing, as a misplaced #endif would drop them
            const write = "XO[base + i] = x;";
            const at = std.mem.indexOf(u8, without, write) orelse return error.ControlEditMissing;
            @memset(without[at..][0..write.len], ' ');
        }
        const sums = try rig_.pipeline(device, with, quant.function(key.kind));
        errdefer sums.deinit();
        const plain = try rig_.pipeline(device, without, quant.function(key.kind));
        errdefer plain.deinit();
        const threads: usize = if (r.which == .mlp) 64 else hidden / 16;
        if (plain.maxThreads() < threads or sums.maxThreads() < threads) return error.XsLimits;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .reg = r, .plain = plain, .sums = sums };
    }

    pub fn deinit(t: *Xs) void {
        t.rig.deinit();
        t.plain.deinit();
        t.sums.deinit();
    }

    pub fn kernel(t: *Xs) g.Kernel {
        return .{ .spec = spec(t.reg), .ctx = t, .vt = &vt, .control = if (t.reg.control) .shapes else null };
    }

    const vt = g.hooks(Xs);

    fn groups(t: *const Xs) usize {
        return (if (t.reg.which == .mlp) width else hidden) / 64;
    }

    fn padded(rows: usize) usize {
        return quant.padded(@intCast(rows));
    }

    /// bf16 inputs in [-2, 2), NaN and both infinities at gate 1; dims are the frame's [rows, padded rows].
    pub fn prepare(t: *Xs, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64) return false;
        t.rig.reset();
        t.rows = shape.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const rows = shape.rows;
        const n: usize = if (t.reg.which == .mlp) 2 * width else hidden; // mlp_xs reads fused gate|up rows
        const sizes = [_]usize{ rows * n, rows * hidden, hidden };
        var host: [3][]u16 = undefined;
        for (&host, sizes) |*h, len| {
            h.* = try t.a.alloc(u16, len);
            for (h.*) |*x| x.* = c.toBf16(rng.float(f32) * 4 - 2);
        }
        defer for (host) |h| t.a.free(h);
        if (fill == .nonfinite) for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }, 0..) |v, j| {
            host[0][(j * 131 + 7) % host[0].len] = c.toBf16(v);
        };
        const dims = [8]i32{ @intCast(rows), @intCast(padded(rows)), 0, 0, 0, 0, 0, 0 };
        const eps: f32 = 1e-6;
        t.ins[0] = try t.rig.input(std.mem.sliceAsBytes(host[0])); // H, or GATE|UP
        t.ins[1] = try t.rig.input(std.mem.sliceAsBytes(host[1])); // R
        t.ins[2] = try t.rig.input(std.mem.sliceAsBytes(host[2])); // Wt
        t.ins[3] = try t.rig.input(std.mem.asBytes(&eps));
        t.ins[4] = try t.rig.input(std.mem.asBytes(&dims));
        const out_n: usize = if (t.reg.which == .mlp) width else hidden;
        const outs: usize = if (t.reg.which == .norm) 2 else 1;
        for (t.outs[0..outs], t.base[0..outs]) |*o, *b| {
            o.* = try t.rig.output(rows * out_n * 2);
            b.* = try t.rig.scratch(rows * out_n * 2);
        }
        const xs_bytes = t.groups() * padded(rows) * 4;
        t.xs = try t.rig.scratch(xs_bytes);
        t.base_xs = try t.rig.scratch(xs_bytes);
        return true;
    }

    fn encode(t: *Xs, e: mtl.ComputeEncoder, p: mtl.Pipeline, outs: []const rig_.Ref, xs: rig_.Ref, rows: usize) void {
        e.setPipeline(p);
        const in_refs: []const rig_.Ref = switch (t.reg.which) {
            .norm => &.{ t.ins[0], t.ins[1], t.ins[2], t.ins[3], t.ins[4] },
            .input_norm => &.{ t.ins[0], t.ins[2], t.ins[3], t.ins[4] },
            .mlp => &.{ t.ins[0], t.ins[0], t.ins[4] },
        };
        var slot: usize = 0;
        for (in_refs) |r| {
            e.setBuffer(r.buf, r.off, slot);
            slot += 1;
        }
        if (t.reg.which == .mlp) e.setBuffer(t.ins[0].buf, t.ins[0].off + width * 2, 1); // UP: the second half of a row
        for (outs) |o| {
            e.setBuffer(o.buf, o.off, slot);
            slot += 1;
        }
        e.setBuffer(xs.buf, xs.off, slot);
        if (t.reg.which == .mlp) e.dispatchGroups(.of(width / 64, rows, 1), .of(64, 1, 1)) else e.dispatchGroups(.of(1, rows, 1), .of(hidden / 16, 1, 1));
    }

    fn outputs(t: *const Xs) usize {
        return if (t.reg.which == .norm) 2 else 1;
    }

    /// The build under test at the rows alone, as quant_gpu dispatches it where `xs` is off.
    pub fn run(t: *Xs) !void {
        @memset(t.rig.bytes(t.xs, t.groups() * padded(t.rows) * 4), 0xa5);
        const p = t.rig.begin();
        t.encode(p.e, t.plain, t.outs[0..t.outputs()], t.xs, t.rows);
        try t.rig.finish(p);
    }

    pub fn count(_: *Xs) usize {
        return 0;
    }

    pub fn reference(_: *Xs, _: []f64, _: []f64) !void {}

    fn say(t: *Xs, comptime fmt: []const u8, args: anytype) ![]const u8 {
        return std.fmt.bufPrint(&t.msg, fmt, args);
    }

    /// The XS build over the padded rows on the same inputs: every output byte equal, and no XS byte written here.
    pub fn check(t: *Xs, _: g.Gate) !?[]const u8 {
        const p = t.rig.begin();
        t.encode(p.e, t.sums, t.base[0..t.outputs()], t.base_xs, padded(t.rows));
        try t.rig.finish(p);
        const out_n: usize = if (t.reg.which == .mlp) width else hidden;
        for (t.outs[0..t.outputs()], t.base[0..t.outputs()], 0..) |o, b, i| {
            const got = t.rig.bytes(o, t.rows * out_n * 2);
            const want = t.rig.bytes(b, t.rows * out_n * 2);
            if (std.mem.indexOfDiff(u8, got, want)) |at| return try t.say("output {d} differs from the XS build's from byte {d} (row {d})", .{ i, at, at / (out_n * 2) });
        }
        for (t.rig.bytes(t.xs, t.groups() * padded(t.rows) * 4), 0..) |x, at| if (x != 0xa5) return try t.say("the build without Q27_XS wrote XS byte {d}", .{at});
        return null;
    }
};

/// The three kernels and the planted control, registered beside the other qwen27 probes.
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Xs) = .empty,

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        const regs = [_]Reg{ .{ .which = .norm }, .{ .which = .input_norm }, .{ .which = .mlp }, .{ .which = .norm, .control = true } };
        for (regs) |r| {
            if (only) |o| if (std.mem.indexOf(u8, name(r), o) == null) continue;
            const x = try s.a.create(Xs);
            errdefer s.a.destroy(x);
            x.* = try Xs.init(s.a, s.device, s.queue, r);
            try s.probes.append(s.a, x);
            try s.kernels.append(s.a, x.kernel());
        }
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |x| {
            x.deinit();
            s.a.destroy(x);
        }
        s.probes.deinit(s.a);
    }
};
