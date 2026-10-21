//! moe_glue.cu against core/moe_ref.zig: picks exact, weights and sums in float64 tolerance, bits at any row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const dump = @import("byte_dump.zig");

const ref = core.moe_ref;
const counts = [_]usize{ 1, 3, 17, 300 };
const most = counts[counts.len - 1];

const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    g: *const cuda.moe_glue.MoeGlue,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    worst: f64 = 0,

    fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn out(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, @max(bytes, 4));
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0xff, r.s.handle);
        return b.ptr;
    }

    fn back(r: *Rig, ptr: u64, len: usize) ![]u8 {
        const o = try r.a.alloc(u8, len);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        try b.download(0, o);
        return o;
    }
};

fn f32At(raw: []const u8, i: usize) f32 {
    return @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little));
}

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn topkCase(r: *Rig, route: cuda.moe_glue.Route, with_bias: bool) !void {
    const ne = route.experts;
    const logits = try r.a.alloc(f32, most * ne);
    for (logits, 0..) |*l, i| {
        l.* = r.rng.floatNorm(f32) * 2;
        if (i % 7 == 3) l.* = logits[i - 1]; // equal neighbours: the tie rule decides
    }
    const bias = try r.a.alloc(f32, ne);
    for (bias, 0..) |*b, i| b.* = if (i % 5 == 0) 0.25 else 0;
    const slots = route.topk + @intFromBool(route.shared);
    const dl = try r.dev(std.mem.sliceAsBytes(logits));
    const db = try r.dev(std.mem.sliceAsBytes(bias));
    const want_p = try r.a.alloc(u32, most * slots);
    const want_w = try r.a.alloc(f64, most * slots);
    const rr: ref.Route = .{ .experts = ne, .topk = route.topk, .shared = route.shared, .normalize = route.normalize, .scale = route.scale };
    for (0..most) |row| ref.topk(logits[row * ne ..][0..ne], if (with_bias) bias else null, rr, want_p[row * slots ..][0..slots], want_w[row * slots ..][0..slots]);
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "topk {d} of {d}{s}{s}{s}", .{ route.topk, ne, if (route.shared) " + shared" else "", if (with_bias) ", bias" else "", if (route.normalize) ", normalized" else "" });
    const ps = try r.a.alloc(u64, counts.len);
    const ws = try r.a.alloc(u64, counts.len);
    for (counts, ps, ws) |m, *p, *w| {
        p.* = try r.out(m * slots * 4);
        w.* = try r.out(m * slots * 4);
        try r.g.topk(r.s, dl, if (with_bias) db else null, route, p.*, w.*, m);
    }
    const gp = try r.back(ps[counts.len - 1], most * slots * 4);
    const gw = try r.back(ws[counts.len - 1], most * slots * 4);
    try dump.output(r.a, "moe-glue", label, "picks", &.{ most, slots }, .u32, .{ .bias = with_bias }, gp);
    try dump.output(r.a, "moe-glue", label, "weights", &.{ most, slots }, .f32, .{ .bias = with_bias }, gw);
    const picks = std.mem.bytesAsSlice(u32, @as([]align(1) u8, gp));
    var q_most: f64 = 0;
    for (want_p, 0..) |p, i| try check.expect(picks[i] == p, "{s}: pick {d} is {d}, want {d}", .{ label, i, picks[i], p });
    for (want_w, 0..) |w, i| q_most = @max(q_most, @abs(f32At(gw, i) - w) / (@abs(w) * std.math.pow(f64, 2, -20) + 1e-30));
    r.worst = @max(r.worst, q_most);
    try check.expect(q_most <= 1, "{s}: weights within 2^-20 (worst {d:.3})", .{ label, q_most });
    for (counts[0 .. counts.len - 1], ps[0 .. counts.len - 1], ws[0 .. counts.len - 1]) |m, p, w| {
        const got_p = try r.back(p, m * slots * 4);
        const got_w = try r.back(w, m * slots * 4);
        try dump.output(r.a, "moe-glue", label, "picks", &.{ m, slots }, .u32, .{ .bias = with_bias }, got_p);
        try dump.output(r.a, "moe-glue", label, "weights", &.{ m, slots }, .f32, .{ .bias = with_bias }, got_w);
        try check.sameBytes(label, got_p, gp[0 .. m * slots * 4]);
        try check.sameBytes(label, got_w, gw[0 .. m * slots * 4]);
    }
}

fn combineCase(r: *Rig, slots: usize, dims: usize, y_bf16: bool) !void {
    const n = most * slots * dims;
    const yv = try r.a.alloc(f64, n);
    const ybytes = try r.a.alloc(u8, n * @as(usize, if (y_bf16) 2 else 4));
    for (yv, 0..) |*v, i| {
        const x = r.rng.floatNorm(f32) * (if (i % 97 == 0) @as(f32, 30) else 1);
        if (y_bf16) {
            const h = bf16(x);
            std.mem.writeInt(u16, ybytes[2 * i ..][0..2], h, .little);
            v.* = @as(f32, @bitCast(@as(u32, h) << 16));
        } else {
            std.mem.writeInt(u32, ybytes[4 * i ..][0..4], @bitCast(x), .little);
            v.* = x;
        }
    }
    const w = try r.a.alloc(f32, most * slots);
    for (w) |*x| x.* = r.rng.float(f32);
    const wv = try r.a.alloc(f64, w.len);
    for (w, wv) |x, *v| v.* = x;
    const dy = try r.dev(ybytes);
    const dw = try r.dev(std.mem.sliceAsBytes(w));
    const want = try r.a.alloc(f64, most * dims);
    const tol = try r.a.alloc(f64, most * dims);
    for (0..most) |row| {
        ref.combine(yv[row * slots * dims ..][0 .. slots * dims], wv[row * slots ..][0..slots], dims, want[row * dims ..][0..dims]);
        for (0..dims) |d| { // bf16 half-ulp of the sum, plus fp32 fma error over the slots
            var mag: f64 = 0;
            for (0..slots) |k| mag += @abs(wv[row * slots + k] * yv[(row * slots + k) * dims + d]);
            tol[row * dims + d] = @abs(want[row * dims + d]) * std.math.pow(f64, 2, -8) + mag * @as(f64, @floatFromInt(slots)) * std.math.pow(f64, 2, -24) + 1e-30;
        }
    }
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "combine {d} slots x {d} ({s} -> bf16)", .{ slots, dims, if (y_bf16) "bf16" else "fp32" });
    const outs = try r.a.alloc(u64, counts.len);
    for (counts, outs) |m, *o| {
        o.* = try r.out(m * dims * 2);
        try r.g.combine(r.s, dy, if (y_bf16) .bf16 else .f32, dw, o.*, .bf16, m, slots, dims);
    }
    const wide = try r.back(outs[counts.len - 1], most * dims * 2);
    try dump.output(r.a, "moe-glue", label, "sum", &.{ most, dims }, .bf16, .{ .slots = slots, .bf16_input = y_bf16 }, wide);
    var q_most: f64 = 0;
    for (want, tol, 0..) |x, t, i| {
        const got: f64 = @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, wide[2 * i ..][0..2], .little)) << 16));
        q_most = @max(q_most, @abs(got - x) / t);
    }
    r.worst = @max(r.worst, q_most);
    try check.expect(q_most <= 1, "{s}: within its float64 tolerance (worst {d:.3})", .{ label, q_most });
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, o| {
        const got = try r.back(o, m * dims * 2);
        try dump.output(r.a, "moe-glue", label, "sum", &.{ m, dims }, .bf16, .{ .slots = slots, .bf16_input = y_bf16 }, got);
        try check.sameBytes(label, got, wide[0 .. m * dims * 2]);
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var g = try cuda.moe_glue.MoeGlue.load(gpu.d);
    defer g.unload();
    var prng = std.Random.DefaultPrng.init(23);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .g = &g, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    try topkCase(&r, .{ .experts = 384, .topk = 6, .shared = true }, true); // Kolibri 1
    try topkCase(&r, .{ .experts = 256, .topk = 8, .shared = false, .normalize = true, .scale = 2.5 }, true); // DeepSeek-style
    try topkCase(&r, .{ .experts = 512, .topk = 32, .shared = true }, true); // the warp form's widest
    try topkCase(&r, .{ .experts = 513, .topk = 6, .shared = true }, true); // the block form's narrowest
    try topkCase(&r, .{ .experts = 4096, .topk = 64, .shared = false }, false);
    try combineCase(&r, 7, 2560, false);
    try combineCase(&r, 7, 2560, true);
    try combineCase(&r, 9, 7168, false);
    check.pass("moe glue: picks exact, weights and sums within float64 tolerance (worst {d:.3}), every row count's bits equal", .{r.worst});
}
