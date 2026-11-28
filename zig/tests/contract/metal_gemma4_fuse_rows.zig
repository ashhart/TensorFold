//! Exact comparisons against Gemma's unfused Metal glue and reg projection at served widths.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const g4 = tf.gemma4;
const lane = tf.lane_projection;
const fused = g4.fused_lane;
const g = @import("contract").gates;
const r = @import("rig.zig");

fn ref(v: r.Ref) lane.Ref {
    return .{ .buf = v.buf, .off = v.off };
}

fn bind(e: mtl.ComputeEncoder, buffers: []const r.Ref) void {
    for (buffers, 0..) |v, i| e.setBuffer(v.buf, v.off, i);
}

const Pipes = struct {
    plain: lane.Projection,
    one: fused.Projection,
    multi: fused.Projection,
    tail: ?mtl.Pipeline = null,
    geglu: ?mtl.Pipeline = null,

    fn init(a: std.mem.Allocator, device: mtl.Device, v: g4.contracts.Variant) !Pipes {
        const one = try fused.Projection.init(a, device, v.layout());
        errdefer one.deinit();
        const multi = try fused.Projection.init(a, device, v.layoutFor(true));
        errdefer multi.deinit();
        if (r.tgMemory(multi.pipe) > 32768) return error.ThreadgroupMemoryLimit;
        const plain = try lane.Projection.init(a, device, one.layout.projection);
        errdefer plain.deinit();
        var p: Pipes = .{ .plain = plain, .one = one, .multi = multi };
        if (v.kind != 3) {
            const src = try g4.kernels.glueSource(a, v.geo, false, false, 4);
            defer a.free(src);
            p.tail = try r.pipeline(device, src, if (v.kind == 1) "gemma_dense_tail" else "gemma_attn_tail");
            errdefer p.tail.?.deinit();
            if (v.kind == 2) p.geglu = try r.pipeline(device, src, "gemma_geglu");
        }
        return p;
    }

    fn deinit(p: Pipes) void {
        if (p.geglu) |pipe| pipe.deinit();
        if (p.tail) |pipe| pipe.deinit();
        p.plain.deinit();
        p.one.deinit();
        p.multi.deinit();
    }
};

const Pack = struct {
    w: r.Ref,
    sb: r.Ref,

    fn make(rig: *r.Rig, l: lane.Layout, rng: std.Random) !Pack {
        const a = rig.a;
        const words = try a.alloc(u32, l.n * l.k / 8);
        defer a.free(words);
        for (words) |*word| word.* = rng.int(u32);
        const scales = try a.alloc(u16, l.n * l.k / 64);
        defer a.free(scales);
        const biases = try a.alloc(u16, scales.len);
        defer a.free(biases);
        for (scales, biases) |*scale, *bias| {
            scale.* = 0x3b80 + @as(u16, rng.int(u6));
            bias.* = 0xbc80 + @as(u16, rng.int(u6));
        }
        const packed_w = try a.alloc(u32, l.weightWords());
        defer a.free(packed_w);
        const packed_sb = try a.alloc(u16, l.metadataElements());
        defer a.free(packed_sb);
        try lane.pack(l, words, scales, biases, packed_w, packed_sb);
        const w = try rig.input(std.mem.sliceAsBytes(packed_w));
        try lane.regOrder(rig.device, &.{.{ .w = ref(w), .n = @intCast(l.n), .k = @intCast(l.k) }});
        return .{ .w = w, .sb = try rig.input(std.mem.sliceAsBytes(packed_sb)) };
    }
};

fn values(rig: *r.Rig, count: usize, rng: std.Random, special: bool) !r.Ref {
    const data = try rig.a.alloc(u16, count);
    defer rig.a.free(data);
    for (data) |*x| x.* = @import("contract").spec.toBf16(rng.float(f32) - 0.5);
    if (special) {
        data[0] = 0x7fc0;
        data[1] = 0x7f80;
    }
    return rig.input(std.mem.sliceAsBytes(data));
}

const Buffers = struct {
    x: r.Ref,
    h: r.Ref,
    y: r.Ref,
    tap: r.Ref,
    norm0: r.Ref,
    norm1: r.Ref,
    scalar: r.Ref,
    sums: r.Ref,
    normed: r.Ref,
    gu: r.Ref,
    got_h: r.Ref,
    got_y: r.Ref,
    got_tap: r.Ref,
};

fn cell(rig: *r.Rig, p: Pipes, weights: Pack, v: g4.contracts.Variant, rows: u32, rng: std.Random, special: bool, corrupt: bool) !bool {
    defer rig.reset();
    const hidden = rows * v.k * 2;
    const tap_len = hidden * 2;
    const tap_stride: i32 = if (v.tap) @intCast(2 * v.k) else 0;
    const out = rows * v.n * (if (v.kind == 2) @as(usize, 1) else 2);
    const b: Buffers = .{
        .x = try values(rig, rows * v.k, rng, special and v.kind == 3),
        .h = try values(rig, rows * v.k, rng, special and v.kind != 3),
        .y = try rig.output(out),
        .tap = try rig.output(tap_len),
        .norm0 = try values(rig, v.k, rng, false),
        .norm1 = try values(rig, v.k, rng, false),
        .scalar = try rig.input(&.{ 0x40, 0x3f }),
        .sums = try rig.scratch((rows * (v.k / 64) + 16) * 4),
        .normed = try rig.scratch(hidden),
        .gu = try rig.scratch(rows * v.n * 2),
        .got_h = try rig.output(hidden),
        .got_y = try rig.output(out),
        .got_tap = try rig.output(tap_len),
    };
    const h = try rig.output(hidden);
    const pass = rig.begin();
    const e = pass.e;
    if (v.kind == 3) {
        try p.plain.encode(e, ref(b.x), ref(weights.w), ref(weights.sb), ref(b.y), rows);
    } else {
        const eps = g4.config.eps;
        const stride: i32 = @intCast(rows);
        e.setPipeline(p.tail.?);
        if (v.kind == 1) {
            bind(e, &.{ b.h, b.x, b.norm0, b.scalar, b.norm1 });
            e.setValue(eps, 5);
            e.setBuffer(h.buf, h.off, 6);
            e.setBuffer(b.normed.buf, b.normed.off, 7);
            e.setBuffer(b.sums.buf, b.sums.off, 8);
            e.setValue(stride, 9);
            e.setBuffer(b.tap.buf, b.tap.off, 10);
            e.setValue(tap_stride, 11);
        } else {
            bind(e, &.{ b.h, b.x, b.norm0, b.norm1 });
            e.setValue(eps, 4);
            e.setBuffer(h.buf, h.off, 5);
            e.setBuffer(b.normed.buf, b.normed.off, 6);
            e.setBuffer(b.sums.buf, b.sums.off, 7);
            e.setValue(stride, 8);
        }
        e.dispatchGroups(.{ .width = rows }, .{ .width = 256 });
        try p.plain.encodeSums(e, ref(b.normed), ref(weights.w), ref(weights.sb), .{ .ref = ref(b.sums), .stride = rows }, ref(if (v.kind == 1) b.y else b.gu), rows);
        if (v.kind == 2) {
            e.setPipeline(p.geglu.?);
            bind(e, &.{ b.gu, b.y });
            e.setValue(@as(u32, @intCast(v.n / 2)), 2);
            e.setBuffer(b.sums.buf, b.sums.off, 3);
            e.setValue(@as(i32, 0), 4);
            e.setValue(@as(i32, 1), 5);
            e.dispatchGroups(.{ .width = v.n / 128, .height = rows }, .{ .width = 64 });
        }
    }
    const f = if (rows == 1) p.one else p.multi;
    const a3 = if (v.kind == 1) b.scalar else b.norm1;
    try f.encodeFused(e, ref(b.x), ref(weights.w), ref(weights.sb), ref(b.got_y), .{
        .a = .{ ref(b.h), ref(b.x), ref(b.norm0), ref(a3), ref(b.norm1), ref(b.x) },
        .o = .{ ref(b.got_h), ref(b.got_tap) },
        .f = .{ g4.config.eps, 0, 0, 0 },
        .i = .{ tap_stride, 0, 0, 0 },
    }, rows);
    try rig.finish(pass);
    if (!try rig.intact()) return error.FusedRowsGuardChanged;
    const want = rig.bytes(b.y, out);
    const got = rig.bytes(b.got_y, out);
    var nonfinite: usize = 0;
    var finite: usize = 0;
    for (0..out / 2) |i| {
        const bits = std.mem.readInt(u16, want[i * 2 ..][0..2], .little);
        if (bits == 0xffff) return error.FusedRowsUnwrittenOutput;
        if (bits & 0x7f80 == 0x7f80) nonfinite += 1 else finite += 1;
    }
    if (special and (nonfinite == 0 or (rows > 1 and finite == 0))) return error.FusedRowsNonfiniteFixture;
    if (!special and nonfinite != 0) return error.FusedRowsFiniteFixture;
    const equal = std.mem.eql(u8, want, got) and
        (v.kind == 3 or std.mem.eql(u8, rig.bytes(h, hidden), rig.bytes(b.got_h, hidden))) and
        std.mem.eql(u8, rig.bytes(b.tap, tap_len), rig.bytes(b.got_tap, tap_len));
    if (!corrupt) return equal;
    if (!equal) return error.ControlBaselineMismatch;
    got[0] ^= 1;
    return std.mem.eql(u8, want, got);
}

pub fn gates(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, only: ?[]const u8, seed: u64, t: *g.Tally) !void {
    if (only) |s| if (std.mem.indexOf(u8, "gemma4 fused rows", s) == null) return;
    if (!@import("metal_support.zig").tensor(device.tensorUnits(), "gemma4 fused rows")) return;
    var weights: r.Rig = .{ .a = a, .device = device, .queue = queue };
    defer weights.deinit();
    var rig: r.Rig = .{ .a = a, .device = device, .queue = queue };
    defer rig.deinit();
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    for (g4.contracts.entries) |v| {
        if (v.kind == 0) continue;
        defer weights.reset();
        const p = try Pipes.init(a, device, v);
        defer p.deinit();
        const pack = try Pack.make(&weights, v.layout().projection, rng);
        for ([_]u32{ 1, 2, 4, 8, 16 }) |rows| for ([_]bool{ false, true }) |special| {
            const equal = try cell(&rig, p, pack, v, rows, rng, special, false);
            t.failed += @intFromBool(!equal);
            std.debug.print("{s} gemma4 fused rows: {s}, R={d}, {s}\n", .{ if (equal) "PASS" else "FAIL", v.name, rows, if (special) "NaN/Inf" else "finite" });
        };
        if (v.kind == 2 and std.mem.eql(u8, v.geo.name, "31B")) {
            const equal = try cell(&rig, p, pack, v, 16, rng, false, true);
            t.controls += 1;
            if (equal) t.missed += 1 else t.caught += 1;
            std.debug.print("{s} gemma4 fused rows: 31B GeGLU output bit flipped\n", .{if (equal) "CONTROL MISSED" else "CONTROL RED"});
        }
    }
}
