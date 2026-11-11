//! Group sums of the producers' stored BF16 rows, bit-equal to tf_lane_sums and independently summed in FP64.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const sources = @import("kernel_sources");
const rr = @import("rig.zig");
const c = ct.spec;
const g = ct.gates;
const q = tf.qwen27;
const K = 6144;
const groups = K / 64;
pub const Kind = enum { gdn, attention };
fn depth(_: c.Shape) usize {
    return 64;
}
pub fn label(kind: Kind, bad: bool) []const u8 {
    return switch (kind) {
        inline else => |k| if (bad) "control: q27 out sums " ++ @tagName(k) ++ " reversed" else "metal q27 out sums " ++ @tagName(k),
    };
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rr.Rig,
    kind: Kind,
    bad: bool,
    old: mtl.Pipeline,
    fused: mtl.Pipeline,
    sums: tf.lane_projection.Projection,
    rows: u32 = 0,
    stride: u32 = 0,
    p: q.gdn_contract.Params = undefined,
    ins: [8]rr.Ref = undefined,
    stored: [2]rr.Ref = undefined,
    xs: rr.Ref = undefined,
    gate: ?rr.Ref = null,
    expected: []u8 = &.{},
    const vt = g.hooks(Probe);
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kind: Kind, bad: bool) !Probe {
        const original = if (kind == .gdn) try a.dupe(u8, sources.core_deltanet) else try q.simd_attention.library(a, true, true);
        defer a.free(original);
        const from = "for (int i = 0; i < 64; i++) acc += float(row[";
        if (std.mem.count(u8, original, from) != 1) return error.OutSumsControlEdit;
        const text = if (bad) try std.mem.replaceOwned(u8, a, original, from, "for (int i = 63; i >= 0; i--) acc += float(row[") else try a.dupe(u8, original);
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const old = try mtl.Pipeline.init(device, lib, if (kind == .gdn) "tf_delta_gdn_post" else "q27_attn_merge_single", false);
        errdefer old.deinit();
        const fused = try mtl.Pipeline.init(device, lib, if (kind == .gdn) "tf_delta_gdn_post_sums" else "q27_attn_merge_single_sums", false);
        errdefer fused.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .kind = kind, .bad = bad, .old = old, .fused = fused, .sums = try tf.lane_projection.Projection.init(a, device, .{ .n = 32, .k = K, .format = .{ .bits = 4, .group = 64 }, .precompute_sums = true, .reg = true, .rows = 128 }) };
    }
    pub fn deinit(p: *Probe) void {
        p.a.free(p.expected);
        p.rig.deinit();
        p.old.deinit();
        p.fused.deinit();
        p.sums.deinit();
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.bad) .shapes else null, .spec = .{ .name = label(p.kind, p.bad), .accum = .f32, .depth = depth, .out = .f32, .replaces = "tf_lane_sums over the unchanged producer's stored BF16 rows", .served = &.{ .{ .rows = 1, .n = groups, .k = 64 }, .{ .rows = 5, .n = groups, .k = 64 }, .{ .rows = 16, .n = groups, .k = 64 }, .{ .rows = 128, .n = groups, .k = 64 } }, .tiles = .{ .{ .size = 1 }, .{ .step = groups, .min = groups }, .{ .size = 64, .step = 64, .min = 64 } } } };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 128 or shape.n != groups or shape.k != 64) return false;
        p.rig.reset();
        p.a.free(p.expected);
        p.expected = &.{};
        p.gate = null;
        p.rows = @intCast(shape.rows);
        p.stride = p.rows;
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        var target: [128]u16 = undefined;
        if (fill == .stress) {
            const witness = try @import("q27_out_sums_inputs.zig").stress(seed);
            for (&target, 0..) |*v, i| v.* = witness[i % 64];
        } else for (&target, 0..) |*v, i| {
            const x = (random.float(f32) * 2 - 1) * (if (i % 17 == 0) @as(f32, 1e6) else 1);
            v.* = if (i % 31 == 0) 0x8000 else if (i % 31 == 1) 1 else c.toBf16(x);
        }
        p.stored = .{ try p.rig.scratch(shape.rows * K * 2), try p.rig.scratch(shape.rows * K * 2) };
        p.xs = try p.rig.output(groups * shape.rows * 4);
        if (p.kind == .gdn) try p.gdnInputs(&target, fill) else try p.attentionInputs(&target, fill);
        try p.baseline();
        return true;
    }
    fn gdnInputs(p: *Probe, target: *const [128]u16, fill: g.Fill) !void {
        const n = @as(usize, p.rows) * K;
        const rec = try p.a.alloc(u16, n);
        defer p.a.free(rec);
        @memset(rec, 0x3f80);
        const gate = try p.a.alloc(u16, n);
        defer p.a.free(gate);
        @memset(gate, c.toBf16(32));
        var norm: [128]u16 = undefined;
        for (&norm, target) |*v, x| v.* = c.toBf16(c.fromBf16(x) / 32);
        if (fill == .nonfinite) {
            norm[0] = 0x3f80;
            norm[1] = 0x3f80;
            for (0..@min(p.rows, 4)) |row| {
                gate[row * K] = if (row == 0) 0x7fc0 else 0x7f80;
                if (row == 2) rec[row * K] = 0xbf80;
                if (row == 3) {
                    gate[row * K + 1] = 0x7f80;
                    rec[row * K + 1] = 0xbf80;
                }
            }
        }
        p.ins[0] = try p.rig.input(std.mem.sliceAsBytes(rec));
        p.ins[1] = try p.rig.input(std.mem.sliceAsBytes(gate));
        p.ins[2] = try p.rig.input(std.mem.asBytes(&norm));
        p.p = .{ .rows = p.rows, .nk = 16, .nv = 48, .dk = 128, .dv = 128, .taps = 4, .slots = 1, .weight_flags = 0, .eps = 1e-6 };
    }
    fn attentionInputs(p: *Probe, target: *const [128]u16, fill: g.Fill) !void {
        const n = @as(usize, 4 * p.rows) * 16 * 256;
        const partial = try p.a.alloc(f32, n);
        defer p.a.free(partial);
        @memset(partial, 0);
        for (0..4) |hk| for (0..p.rows) |row| for (0..6) |head| for (0..256) |ch| {
            const at = ((hk * p.rows + row) * 16 + head) * 256 + ch;
            partial[at] = c.fromBf16(target[ch % 128]);
            if (fill == .nonfinite and head == 0 and hk == 0) {
                if (ch == 0 and row < 4) partial[at] = if (row == 0) std.math.nan(f32) else if (row == 2) -std.math.inf(f32) else std.math.inf(f32);
                if (ch == 1 and row == 3) partial[at] = -std.math.inf(f32);
            }
        };
        var meta: [20]i32 = @splat(0);
        meta[3] = 1;
        meta[4] = @intCast(p.rows);
        meta[13] = 1;
        const nodes = try p.a.alloc(i32, p.rows * 4);
        defer p.a.free(nodes);
        @memset(nodes, 0);
        const scores = try p.a.alloc(f32, 4 * p.rows * 16);
        defer p.a.free(scores);
        @memset(scores, 0);
        const totals = try p.a.alloc(f32, scores.len);
        defer p.a.free(totals);
        @memset(totals, 1);
        const zero: [16]u8 = @splat(0);
        p.ins[0] = try p.rig.input(&zero);
        p.ins[1] = p.ins[0];
        p.ins[2] = p.ins[0];
        p.ins[3] = try p.rig.input(std.mem.sliceAsBytes(partial));
        p.ins[4] = try p.rig.input(std.mem.sliceAsBytes(scores));
        p.ins[5] = try p.rig.input(std.mem.sliceAsBytes(totals));
        p.ins[6] = try p.rig.input(std.mem.asBytes(&meta));
        p.ins[7] = try p.rig.input(std.mem.sliceAsBytes(nodes));
    }
    pub fn encode(p: *Probe, e: mtl.ComputeEncoder, fused: bool) void {
        e.setPipeline(if (fused) p.fused else p.old);
        const out = p.stored[@intFromBool(fused)];
        if (p.kind == .gdn) {
            for (p.ins[0..3], 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setValue(p.p, 3);
            e.setBuffer(out.buf, out.off, 4);
            if (fused) {
                e.setBuffer(p.xs.buf, p.xs.off, 5);
                e.setValue(p.stride, 6);
            }
            e.dispatchGroups(.of(1, 48, p.rows), .of(32, 1, 1));
        } else {
            for (p.ins, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
            e.setBuffer(out.buf, out.off, 8);
            const gate = p.gate orelse out;
            e.setBuffer(gate.buf, gate.off, 9);
            e.setValue(@as(u32, @intFromBool(p.gate != null)), 10);
            if (fused) {
                e.setBuffer(p.xs.buf, p.xs.off, 11);
                e.setValue(p.stride, 12);
            }
            e.dispatchGroups(.of(4, p.rows * 6, 1), .of(32, 1, 1));
        }
        e.barrier();
    }
    pub fn baseline(p: *Probe) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, false);
        const baseline_sums = try p.sums.encodeOwnSums(pass.e, .{ .buf = p.stored[0].buf, .off = p.stored[0].off }, p.rows);
        try p.rig.finish(pass);
        if (p.expected.len == 0) p.expected = try p.a.alloc(u8, groups * p.rows * 4);
        for (0..groups) |group| for (0..p.rows) |row| {
            const at = (group * baseline_sums.stride + row) * 4;
            @memcpy(p.expected[(group * p.rows + row) * 4 ..][0..4], baseline_sums.ref.buf.contents()[baseline_sums.ref.off + at ..][0..4]);
        };
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        p.encode(pass.e, true);
        try p.rig.finish(pass);
        if (!std.mem.eql(u8, p.rig.bytes(p.stored[0], p.rows * K * 2), p.rig.bytes(p.stored[1], p.rows * K * 2))) return error.OutSumsProducerBytesDiffer;
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        return p.expected;
    }
    pub fn count(p: *Probe) usize {
        return groups * p.rows;
    }
    fn evaluate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64, mag: ?[]f64) !void {
        const stored: [*]const u16 = @ptrCast(@alignCast(p.stored[0].buf.contents() + p.stored[0].off));
        for (out, 0..) |*value, i| {
            const group = i / p.rows;
            const row = i % p.rows;
            var terms: [64]f64 = undefined;
            var sum: f64 = 0;
            var magnitude: f64 = 0;
            for (&terms, 0..) |*v, ch| {
                v.* = c.fromBf16(stored[row * K + group * 64 + ch]);
                sum += v.*;
                magnitude += @abs(v.*);
            }
            value.* = if (acc == .none) sum else ct.rows.sumIn(acc, order, &terms);
            if (mag) |m| m[i] = magnitude;
        }
    }
    pub fn reference(p: *Probe, want: []f64, mag: []f64) !void {
        try p.evaluate(.none, .sequential, want, mag);
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        try p.evaluate(acc, order, out, null);
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        if (!s.device.tensorUnits()) return;
        for ([_]Kind{ .gdn, .attention }) |kind| for ([_]bool{ false, true }) |bad| {
            if (only) |f| if (std.mem.indexOf(u8, label(kind, bad), f) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, kind, bad);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
        };
    }
    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
    }
};
