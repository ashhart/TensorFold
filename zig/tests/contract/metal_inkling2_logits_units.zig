//! Inkling 2 Mini's logits on the tensor units: the glue entry's FP64 dot, BF16 round and cap of 30.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const bf = contract.bf16_round.bf16;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.experts;
const glue_host = @import("tensorfold").inkling2_metal_host.glue;
const controls = @import("metal_inkling2_units_controls.zig");
const src = @import("inkling2_metal");

const hidden = 2304;
pub const name = "metal inkling2 units tf_ink2_logits_units";
pub const argmax_name = "metal inkling2 units tf_ink2_logits_units argmax";

fn depth(s: c.Shape) usize {
    return s.k + 3;
}

pub fn spec(argmax: bool, control: bool) c.Spec {
    if (argmax) return .{
        .name = if (control) controls.Mutation.logits_argmax.name() else argmax_name,
        .accum = .none,
        .depth = depth,
        .ops = 8,
        .mids = 1,
        .out = .raw,
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 90, .k = hidden }, .{ .rows = 8, .n = 131, .k = hidden }, .{ .rows = 20, .n = 65, .k = hidden } },
        .nan = .{ .stated = "NaN at token 0 wins; later NaNs lose to finite tokens; finite ties pick the lowest id" },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "exact integer argmax ties and NaN rules" },
    };
    return .{
        .name = if (control) controls.Mutation.logits_cap.name() else name,
        .accum = .f32,
        .depth = depth,
        .ops = 8,
        .mids = 1,
        .out = .f32,
        // Rows go 16 a band and vocabulary rows 32 a tile; 90 vocabulary rows leave the last tile part full.
        .tiles = .{ .{ .size = 16 }, .{ .size = 32 }, .{} },
        .served = &.{ .{ .rows = 1, .n = 64, .k = hidden }, .{ .rows = 8, .n = 90, .k = hidden }, .{ .rows = 20, .n = 32, .k = hidden } },
        .sums = .{ .f16 = 0, .bf16 = 0, .why = "independent FP64 terms, an explicit BF16 node and a tanh" },
    };
}

const Sim = struct { acc: c.Accum, order: g.Order };
const Sample = struct { value: f64, magnitude: f64 };

pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    glue: ?glue_host.Pipelines = null,
    argmax: bool,
    control: bool,
    write_both: bool = false,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    w: []u16 = &.{},
    refs: [6]rig_.Ref = undefined,
    msg: [160]u8 = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, argmax: bool, control: bool) !Probe {
        const mut: controls.Mutation = if (!control) .none else if (argmax) .logits_argmax else .logits_cap;
        const text = try controls.apply(a, mut);
        defer a.free(text);
        var pipes = try host.Pipelines.loadSource(device, src.experts);
        errdefer pipes.deinit();
        try pipes.loadUnits(device, text);
        const glue = if (argmax) try glue_host.Pipelines.load(device) else null;
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .pipes = pipes, .glue = glue, .argmax = argmax, .control = control };
    }

    fn clear(p: *Probe) void {
        p.a.free(p.x);
        p.a.free(p.w);
        p.x = &.{};
        p.w = &.{};
        p.rig.reset();
    }

    pub fn deinit(p: *Probe) void {
        p.clear();
        if (p.glue) |*gp| gp.deinit();
        p.pipes.deinit();
        p.rig.deinit();
    }

    pub fn kernel(p: *Probe) g.Kernel {
        const mut: controls.Mutation = if (p.argmax) .logits_argmax else .logits_cap;
        return .{ .spec = spec(p.argmax, p.control), .ctx = p, .vt = if (p.argmax) &vt_argmax else &vt_logits, .control = if (p.control) mut.gate() else null };
    }
    const vt_logits = blk: {
        var table = g.hooks(Probe);
        table.check = null;
        break :blk table;
    };
    const vt_argmax = g.hooks(Probe);

    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const max_n: usize = if (p.argmax) 256 else 128;
        if (shape.rows == 0 or shape.rows > 40 or shape.n == 0 or shape.n > max_n or shape.k != hidden) return false;
        p.clear();
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        p.x = try p.a.alloc(u16, shape.rows * hidden);
        p.w = try p.a.alloc(u16, shape.n * hidden);
        if (!p.argmax) {
            for (p.x, 0..) |*v, i| v.* = c.toBf16((rng.float(f32) * 2 - 1) * (if (i % 7 == 0) @as(f32, 3) else 1));
            for (p.w) |*v| v.* = c.toBf16(rng.float(f32) * 0.5 - 0.25);
            if (fill == .stress) {
                // Whole numbers times a power of two: every product and partial sum is exact, and the cap stays live.
                for (p.x, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(128 * (1 + i % 13))) * (if (i >= p.x.len / 2) @as(f32, -1) else 1));
                for (p.w, 0..) |*v, i| v.* = c.toBf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0x1p-20);
            }
            if (fill == .nonfinite and p.x.len >= 3) {
                p.x[0] = c.toBf16(std.math.nan(f32));
                p.x[p.x.len / 2] = c.toBf16(std.math.inf(f32));
                p.x[p.x.len - 1] = c.toBf16(-std.math.inf(f32));
            }
            p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
            p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.w));
            p.refs[2] = try p.rig.output(p.count() * 4);
            return true;
        }
        const r = shape.rows;
        const n = shape.n;
        p.write_both = (fill != .nonfinite and r == 8 and n == 131);
        @memset(p.x, c.toBf16(0));
        @memset(p.w, c.toBf16(0));
        if (fill != .nonfinite) {
            for (0..r) |row| {
                for (0..8) |d| p.x[row * hidden + d] = c.toBf16(if (d < 4) 1 else -0.5);
                if (row % 3 == 2) for (8..16) |d| {
                    p.x[row * hidden + d] = c.toBf16(128);
                };
            }
            for (0..n) |v| {
                const pat = rng.uintLessThan(u32, 16);
                for (0..4) |d| p.w[v * hidden + d] = c.toBf16(@floatFromInt(@as(i32, @intCast((pat >> @intCast(d)) & 1)) * 2 - 1));
                if (v == 1 or v == 2 or (n > 8 and v == n - 1)) for (0..4) |d| {
                    p.w[v * hidden + d] = c.toBf16(2);
                };
                if (v >= 16 and v < 20) {
                    @memset(p.w[v * hidden ..][0..4], c.toBf16(0));
                    p.w[v * hidden + 8] = c.toBf16(8);
                }
            }
        } else {
            for (0..n) |v| {
                p.w[v * hidden + 0] = c.toBf16(std.math.inf(f32));
                p.w[v * hidden + 1] = c.toBf16(-std.math.inf(f32));
            }
            if (n > 1) {
                p.w[hidden + 0] = c.toBf16(std.math.inf(f32));
                p.w[hidden + 1] = c.toBf16(std.math.inf(f32));
            }
            if (n > 2) {
                p.w[2 * hidden + 0] = c.toBf16(0);
                p.w[2 * hidden + 1] = c.toBf16(0);
                p.w[2 * hidden + 2] = c.toBf16(1);
            }
            if (n > 33) {
                p.w[33 * hidden + 0] = c.toBf16(std.math.inf(f32));
                p.w[33 * hidden + 1] = c.toBf16(std.math.inf(f32));
            }
            if (n > 34) {
                p.w[34 * hidden + 0] = c.toBf16(0);
                p.w[34 * hidden + 1] = c.toBf16(0);
                p.w[34 * hidden + 3] = c.toBf16(1);
            }
            for (0..r) |row| {
                if (row == 0) {
                    p.x[0] = c.toBf16(1);
                    p.x[1] = c.toBf16(1);
                } else if (row == 1) {
                    p.x[row * hidden + 0] = c.toBf16(-1);
                    p.x[row * hidden + 1] = c.toBf16(1);
                    p.x[row * hidden + 2] = c.toBf16(10);
                } else if (row == 2) {
                    p.x[row * hidden + 0] = c.toBf16(-1);
                    p.x[row * hidden + 1] = c.toBf16(1);
                    p.x[row * hidden + 2] = c.toBf16(-1000);
                    p.x[row * hidden + 3] = c.toBf16(-1000);
                } else if (row == 3) {
                    p.x[row * hidden + 0] = c.toBf16(-1);
                    p.x[row * hidden + 1] = c.toBf16(1);
                    p.x[row * hidden + 2] = c.toBf16(-1000);
                    p.x[row * hidden + 3] = c.toBf16(10);
                } else {
                    p.x[row * hidden + 0] = c.toBf16(-1);
                    p.x[row * hidden + 1] = c.toBf16(1);
                }
            }
        }
        p.refs[0] = try p.rig.input(std.mem.sliceAsBytes(p.x));
        p.refs[1] = try p.rig.input(std.mem.sliceAsBytes(p.w));
        p.refs[2] = try p.rig.scratch(r * n * 4);
        const parts = (n + 31) / 32;
        p.refs[3] = try p.rig.scratch(r * parts * 4);
        p.refs[4] = try p.rig.scratch(r * parts * 4);
        p.refs[5] = try p.rig.output(r * 4);
        return true;
    }

    fn hr(b: rig_.Ref) host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }
    fn ghr(b: rig_.Ref) glue_host.Ref {
        return .{ .buf = b.buf, .off = b.off };
    }

    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        if (!p.argmax) {
            try host.logitsUnits(&p.pipes, pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), @intCast(p.shape.rows), @intCast(p.shape.n));
        } else {
            const n_parts = try host.logitsUnitsArgmax(&p.pipes, pass.e, hr(p.refs[0]), hr(p.refs[1]), hr(p.refs[2]), hr(p.refs[3]), hr(p.refs[4]), @intCast(p.shape.rows), @intCast(p.shape.n), p.write_both);
            pass.e.barrier();
            glue_host.argmax(&p.glue.?, pass.e, ghr(p.refs[3]), ghr(p.refs[4]), ghr(p.refs[5]), @intCast(p.shape.rows), n_parts);
        }
        try p.rig.finish(pass);
    }

    pub fn count(p: *const Probe) usize {
        return if (p.argmax) p.shape.rows else p.shape.rows * p.shape.n;
    }

    fn evaluate(p: *const Probe, at: usize, sim: ?Sim) Sample {
        const row = at / p.shape.n;
        const v = at % p.shape.n;
        var terms: [hidden]f64 = undefined;
        var magnitude: f64 = 0;
        for (&terms, 0..) |*term, d| {
            term.* = @as(f64, c.fromBf16(p.w[v * hidden + d])) * @as(f64, c.fromBf16(p.x[row * hidden + d]));
            magnitude += @abs(term.*);
        }
        const sum = if (sim) |s| rows.sumIn(s.acc, s.order, &terms) else blk: {
            var total: f64 = 0;
            for (terms) |term| total += term;
            break :blk total;
        };
        return .{ .value = 30 * std.math.tanh(bf(sum) / 30), .magnitude = magnitude };
    }

    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        if (p.argmax) return error.CheckedByItsOwnRule;
        for (want, mag, 0..) |*v, *m, i| {
            const sample = p.evaluate(i, null);
            v.* = sample.value;
            m.* = sample.magnitude;
        }
    }

    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        for (out, 0..) |*v, i| v.* = p.evaluate(i, .{ .acc = acc, .order = order }).value;
    }

    pub fn check(p: *Probe, gate: g.Gate) !?[]const u8 {
        _ = gate;
        if (!p.argmax) return "numeric hooks must be used";
        const r = p.shape.rows;
        const n = p.shape.n;
        const logits_bytes = p.rig.bytes(p.refs[2], r * n * 4);
        if (!p.write_both) {
            for (logits_bytes) |b| if (b != 0xa5) return "write_logits=false touched out logits buffer";
        }
        const got_ids = std.mem.bytesAsSlice(u32, try p.rig.outputBytes());
        for (0..r) |row| {
            var best_val: f32 = 0;
            var best_id: ?usize = null;
            for (0..n) |v| {
                const y: f32 = @floatCast(p.evaluate(row * n + v, null).value);
                if (p.write_both) {
                    const got_y = c.read(.f32, logits_bytes, row * n + v);
                    if (@abs(got_y - y) > 1e-4) {
                        return try std.fmt.bufPrint(&p.msg, "row {d} v {d} both-mode logit {d} != {d}", .{ row, v, got_y, y });
                    }
                }
                if (best_id == null or y > best_val) {
                    best_val = y;
                    best_id = v;
                }
            }
            if (got_ids[row] != best_id.?) {
                return try std.fmt.bufPrint(&p.msg, "row {d} picked id {d} where exact argmax picks {d}", .{ row, got_ids[row], best_id.? });
            }
        }
        return null;
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: [4]Probe = undefined,
    made: usize = 0,

    pub fn deinit(s: *Suite) void {
        for (s.probes[0..s.made]) |*probe| probe.deinit();
    }

    /// The entry exists only where the GPU has tensor units; elsewhere the glue's kernel serves every width.
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for ([_]bool{ false, true }) |argmax| {
            for ([_]bool{ false, true }) |control| {
                const label = spec(argmax, control).name;
                if (only) |needle| if (std.mem.indexOf(u8, label, needle) == null) continue;
                s.probes[s.made] = try Probe.init(s.a, s.device, s.queue, argmax, control);
                s.made += 1;
                try s.kernels.append(s.a, s.probes[s.made - 1].kernel());
            }
        }
    }
};

test "the logits entry resolves to its symbol in the units source" {
    try std.testing.expect(std.mem.startsWith(u8, name, "metal inkling2 units "));
    try std.testing.expectEqual(@as(usize, 1), std.mem.count(u8, src.experts_units, "tf_ink2_logits_units("));
}
