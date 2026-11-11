//! Fused head norm and RoPE: frozen T622 replacement bytes, independent FP64 norm nodes and accumulator stress.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const ct = @import("contract");
const rr = @import("rig.zig");
const q = tf.qwen27;
const c = ct.spec;
const g = ct.gates;
const D = 256;
fn depth(_: c.Shape) usize {
    return 14;
}
pub fn label(bad: bool) []const u8 {
    return if (bad) "control: q27 qk missing first native rounding" else "metal q27 qk norm rope";
}
pub const Probe = struct {
    a: std.mem.Allocator,
    rig: rr.Rig,
    bad: bool,
    zero_angles: bool = false,
    pipe: mtl.Pipeline,
    old_norm: mtl.Pipeline,
    old_rope: mtl.Pipeline,
    r: ?ct.rows.Rows = null,
    args: q.glue.QkNormRope = undefined,
    buffers: [9]rr.Ref = undefined,
    before: std.ArrayList(u8) = .empty,
    const vt = g.hooks(Probe);
    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, bad: bool) !Probe {
        const raw = q.runtime_sources.glue;
        const needle = "q27_round(value * inverse[0], 0)";
        if (std.mem.count(u8, raw, needle) != 1) return error.QkControlEdit;
        const body = if (bad) try std.mem.replaceOwned(u8, a, raw, needle, "(value * inverse[0])") else try a.dupe(u8, raw);
        defer a.free(body);
        const text = try std.mem.concat(a, u8, &.{ body, @embedFile("q27_qk_t622.metal") });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const pipe = try mtl.Pipeline.init(device, lib, "q27_qk_norm_rope", false);
        errdefer pipe.deinit();
        const old_norm = try mtl.Pipeline.init(device, lib, "q27_head_norm_t622", false);
        errdefer old_norm.deinit();
        return .{ .a = a, .rig = .{ .a = a, .device = device, .queue = queue }, .bad = bad, .pipe = pipe, .old_norm = old_norm, .old_rope = try mtl.Pipeline.init(device, lib, "q27_rope_t622", false) };
    }
    pub fn deinit(p: *Probe) void {
        if (p.r) |r| r.deinit();
        p.pipe.deinit();
        p.old_norm.deinit();
        p.old_rope.deinit();
        p.rig.deinit();
        p.before.deinit(p.a);
    }
    pub fn kernel(p: *Probe) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (p.bad) .shapes else null, .spec = .{
            .name = label(p.bad),
            .replaces = "c3f943f3dc head norms and RoPE",
            .accum = .f32,
            .depth = depth,
            .ops = 8,
            .mids = 1,
            .out = .bf16,
            .served = &.{ .{ .rows = 1, .n = 28, .k = D }, .{ .rows = 16, .n = 28, .k = D }, .{ .rows = 16, .n = 14, .k = D }, .{ .rows = 16, .n = 34, .k = D }, .{ .rows = 128, .n = 20, .k = D }, .{ .rows = 2048, .n = 28, .k = D } },
            .tiles = .{ .{ .size = 16, .min = 1 }, .{ .step = 2, .min = 14 }, .{ .size = D, .step = D, .min = D } },
        } };
    }
    pub fn prepare(p: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 2048 or shape.k != D) return false;
        const hk: u32 = switch (shape.n) {
            28, 20 => 4,
            14, 34 => 2,
            else => return false,
        };
        const h: u32 = @intCast(shape.n - hk);
        if (p.r) |r| r.deinit();
        p.r = null;
        p.rig.reset();
        p.before.clearRetainingCapacity();
        const rows: u32 = @intCast(shape.rows);
        // Isolate the accumulator stress from transcendental error; nonzero rotations retain strict byte gates.
        p.zero_angles = fill == .stress;
        const r = try ct.rows.Rows.init(p.a, .rms, .{ .rows = shape.rows * shape.n, .n = D, .k = D });
        p.r = r;
        r.fill(fill, seed);
        if (fill == .stress) @import("q27_mixed_inputs.zig").stress(r.x, r.w);
        const nq: usize = rows * h * D;
        const nk: usize = rows * hk * D;
        const qg = try p.a.alloc(u16, 2 * nq);
        defer p.a.free(qg);
        const mixed = try p.a.alloc(u16, 2 * nk);
        defer p.a.free(mixed);
        @memset(qg, 0x7fc5);
        @memset(mixed, 0x7fc5);
        for (0..rows) |row| {
            for (0..h) |head| @memcpy(qg[(row * h + head) * 2 * D ..][0..D], r.x[(row * (h + hk) + head) * D ..][0..D]);
            @memcpy(mixed[row * 2 * hk * D ..][0 .. hk * D], r.x[(row * (h + hk) + h) * D ..][0 .. hk * D]);
        }
        p.buffers[0] = try p.rig.input(std.mem.sliceAsBytes(qg));
        p.buffers[1] = try p.rig.input(std.mem.sliceAsBytes(mixed));
        p.buffers[2] = try p.rig.input(std.mem.sliceAsBytes(r.w));
        p.buffers[3] = try p.rig.input(std.mem.sliceAsBytes(r.w));
        const positions = try p.a.alloc(i32, rows * 3);
        defer p.a.free(positions);
        for (0..3) |axis| for (0..rows) |row| {
            positions[axis * rows + row] = if (p.zero_angles) 0 else @intCast(519 + axis * 1009 + row);
        };
        p.buffers[4] = try p.rig.input(std.mem.sliceAsBytes(positions));
        p.buffers[5] = try p.rig.output(nq * 2);
        p.buffers[6] = try p.rig.output(nk * 2);
        p.buffers[7] = try p.rig.output(nq * 2);
        p.buffers[8] = try p.rig.output(nk * 2);
        p.args = .{ .width = D, .heads = h, .kv_heads = hk, .rows = rows, .rotary = 64, .key_stride = hk * 2 * D, .eps = r.eps, .theta = 10000000, .h_end = 12, .w_end = 10, .plane = rows };
        const pass = p.rig.begin();
        @import("q27_qk_reference.zig").encode(pass.e, p.old_norm, p.old_rope, p.params());
        try p.rig.finish(pass);
        try p.before.appendSlice(p.a, try p.rig.outputBytes());
        for (p.buffers[5..]) |b| @memset(p.rig.bytes(b, if (b.buf.id == p.buffers[5].buf.id or b.buf.id == p.buffers[7].buf.id) nq * 2 else nk * 2), 0xff);
        return true;
    }
    fn params(p: *Probe) q.attention.Preprocess {
        const b = p.buffers;
        return .{ .qg = ref(b[0]), .mixed = ref(b[1]), .q_gain = ref(b[2]), .k_gain = ref(b[3]), .positions = ref(b[4]), .norm_q = ref(b[5]), .norm_k = ref(b[6]), .query = ref(b[7]), .rotated_key = ref(b[8]), .rows = p.args.rows, .eps = p.args.eps, .theta = p.args.theta, .h_end = p.args.h_end, .w_end = p.args.w_end, .plane = p.args.plane, .geometry = .{ .heads = p.args.heads, .kv_heads = p.args.kv_heads } };
    }
    pub fn run(p: *Probe) !void {
        const pass = p.rig.begin();
        pass.e.setPipeline(p.pipe);
        for (p.buffers[0..5], 0..) |b, i| pass.e.setBuffer(b.buf, b.off, i);
        pass.e.setValue(p.args, 5);
        for (p.buffers[5..], 6..) |b, i| pass.e.setBuffer(b.buf, b.off, i);
        pass.e.dispatchGroups(.of(p.args.heads + p.args.kv_heads, p.args.rows, 1), .of(64, 1, 1));
        try p.rig.finish(pass);
        if (!p.bad) try p.independent();
    }
    fn independent(p: *Probe) !void {
        const n = p.count();
        const want = try p.a.alloc(f64, n);
        defer p.a.free(want);
        const mag = try p.a.alloc(f64, n);
        defer p.a.free(mag);
        try p.reference(want, mag);
        const output = try p.rig.outputBytes();
        const spec = p.kernel().spec;
        const shape = c.Shape{ .rows = p.args.rows, .n = p.args.heads + p.args.kv_heads, .k = D };
        // Norm nodes use the existing RMS bound; all four output nodes require frozen replacement bytes.
        for (want[0 .. n / 2], mag[0 .. n / 2], 0..) |w, m, i| {
            const got = c.read(.bf16, output, i);
            if (c.classOf(got) != c.classOf(w)) return error.QkReferenceClass;
            if (std.math.isFinite(w) and @abs(got - w) > c.bound(spec, shape, w, m)) {
                std.debug.print("QK_FP64 index={d} got={d} want={d} bound={d}\n", .{ i, got, w, c.bound(spec, shape, w, m) });
                return error.QkReferenceBound;
            }
        }
    }
    pub fn replaced(p: *Probe) ![]const u8 {
        return p.before.items;
    }
    pub fn count(p: *Probe) usize {
        return 2 * p.r.?.outputs();
    }
    pub fn reference(p: *Probe, out: []f64, mag: []f64) !void {
        const raw = try p.a.alloc(f64, p.r.?.outputs());
        defer p.a.free(raw);
        const rm = try p.a.alloc(f64, raw.len);
        defer p.a.free(rm);
        try p.r.?.reference(raw, rm);
        p.arrange(raw, out, mag);
    }
    fn arrange(p: *Probe, raw: []const f64, out: []f64, mag: ?[]f64) void {
        const h = p.args.heads;
        const hk = p.args.kv_heads;
        const rows = p.args.rows;
        const nq: usize = rows * h * D;
        const nk: usize = rows * hk * D;
        for (0..rows) |row| for (0..h + hk) |head| for (0..D) |col| {
            const src = (row * (h + hk) + head) * D + col;
            const normal = raw[src];
            const dst = if (head < h) (row * h + head) * D + col else nq + (row * hk + head - h) * D + col;
            out[dst] = normal;
            if (mag) |m| m[dst] = @abs(normal) / 2;
            var rotated = normal;
            var magnitude = @abs(normal);
            if (col < 64) {
                const second = col >= 32;
                const d = if (second) col - 32 else col;
                const axis: usize = if (d % 3 == 1 and d < 12) 1 else if (d % 3 == 2 and d < 10) 2 else 0;
                const angle = @as(f64, @floatFromInt(if (p.zero_angles) @as(usize, 0) else 519 + axis * 1009 + row)) * std.math.pow(f64, 10000000, -@as(f64, @floatFromInt(d)) / 32);
                const partner = raw[src - col + (if (second) d else d + 32)];
                const x = normal * @cos(angle);
                const y = partner * @sin(angle);
                rotated = if (second) x + y else x - y;
                magnitude = @abs(x) + @abs(y);
            }
            out[nq + nk + dst] = rotated;
            if (mag) |m| m[nq + nk + dst] = 2 * magnitude;
        };
    }
    pub fn simulate(p: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        const raw = try p.a.alloc(f64, p.r.?.outputs());
        defer p.a.free(raw);
        try p.r.?.simulate(acc, order, raw);
        p.arrange(raw, out, null);
    }
};
fn ref(r: rr.Ref) q.projection.Ref {
    return .{ .buffer = r.buf, .offset = r.off };
}
pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,
    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        for ([_]bool{ false, true }) |bad| {
            if (only) |f| if (std.mem.indexOf(u8, label(bad), f) == null) continue;
            const p = try s.a.create(Probe);
            errdefer s.a.destroy(p);
            p.* = try Probe.init(s.a, s.device, s.queue, bad);
            try s.probes.append(s.a, p);
            try s.kernels.append(s.a, p.kernel());
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
