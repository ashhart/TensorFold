//! norms.cu against core/norms_ref.zig's float64 rows, and every row's bits at every row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const ref = core.norms_ref;
const Kind = cuda.norms.Kind;

const widths = [_]usize{ 64, 2560, 5120, 7168, 9216 }; // 9,216: past 8,192, the re-reading path
const counts = [_]usize{ 1, 3, 17, 300 };
const most = counts[counts.len - 1];
const eps: f32 = 1e-6;

const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
    n: *const cuda.norms.Norms,
    rng: std.Random,
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    worst: f64 = 0,

    fn dev(r: *Rig, bytes: []const u8) !u64 {
        const b = try cuda.DeviceBuffer.fromHost(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        return b.ptr;
    }

    fn out(r: *Rig, bytes: usize) !u64 {
        const b = try cuda.DeviceBuffer.alloc(r.gpu.d, bytes);
        try r.bufs.append(r.gpu.gpa, b);
        try b.fill8(0xff, r.s.handle); // a row the kernel misses reads NaN
        return b.ptr;
    }

    fn back(r: *Rig, ptr: u64, len: usize) ![]u8 {
        const o = try r.a.alloc(u8, len);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = len };
        try b.download(0, o);
        return o;
    }

    /// Values a residual stream holds: mostly within [-1, 1), one channel in 64 twenty times larger.
    fn values(r: *Rig, n: usize) []f32 {
        const v = r.a.alloc(f32, n) catch unreachable;
        for (v, 0..) |*x, i| x.* = (r.rng.float(f32) * 2 - 1) * (if (i % 64 == 7) @as(f32, 20) else 1);
        return v;
    }
};

fn size(k: Kind) usize {
    return switch (k) {
        .f32 => 4,
        .f16, .bf16 => 2,
    };
}

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

/// `v` stored as `k` (bytes) and the values it then holds (f64).
fn stored(a: std.mem.Allocator, v: []const f32, k: Kind) !struct { bytes: []u8, vals: []f64 } {
    const bytes = try a.alloc(u8, v.len * size(k));
    const vals = try a.alloc(f64, v.len);
    for (v, 0..) |x, i| switch (k) {
        .f32 => {
            std.mem.writeInt(u32, bytes[4 * i ..][0..4], @bitCast(x), .little);
            vals[i] = x;
        },
        .bf16 => {
            const h = bf16(x);
            std.mem.writeInt(u16, bytes[2 * i ..][0..2], h, .little);
            vals[i] = @as(f32, @bitCast(@as(u32, h) << 16));
        },
        .f16 => {
            const h: f16 = @floatCast(x);
            std.mem.writeInt(u16, bytes[2 * i ..][0..2], @bitCast(h), .little);
            vals[i] = h;
        },
    };
    return .{ .bytes = bytes, .vals = vals };
}

fn read(raw: []const u8, k: Kind, i: usize) f64 {
    return switch (k) {
        .f32 => @as(f32, @bitCast(std.mem.readInt(u32, raw[4 * i ..][0..4], .little))),
        .bf16 => @as(f32, @bitCast(@as(u32, std.mem.readInt(u16, raw[2 * i ..][0..2], .little)) << 16)),
        .f16 => @as(f16, @bitCast(std.mem.readInt(u16, raw[2 * i ..][0..2], .little))),
    };
}

/// Every element of `raw` (kind `k`) within `tol` of `want` (null: core's tolerance of a once-rounded value).
fn close(r: *Rig, what: []const u8, raw: []const u8, k: Kind, want: []const f64, tol: ?[]const f64) !void {
    var q_most: f64 = 0;
    for (want, 0..) |w, i| {
        const t = if (tol) |tt| tt[i] else ref.tolerance(w, @fromBackingInt(@intCast(@backingInt(k))));
        const q = @abs(read(raw, k, i) - w) / t;
        if (!(q <= q_most)) q_most = q;
    }
    r.worst = @max(r.worst, q_most);
    try check.expect(q_most <= 1, "{s}: within its float64 tolerance (worst {d:.3})", .{ what, q_most });
}

fn rmsCase(r: *Rig, width: usize, xk: Kind, yk: Kind, weighted: bool) !void {
    const x = try stored(r.a, r.values(most * width), xk);
    const wv = r.values(width);
    for (wv) |*v| v.* = 0.5 + @abs(v.*) / 20;
    const w = try stored(r.a, wv, .f32);
    const dx = try r.dev(x.bytes);
    const dw = try r.dev(w.bytes);
    const want = try r.a.alloc(f64, most * width);
    for (0..most) |row| ref.rms(x.vals[row * width ..][0..width], if (weighted) w.vals else null, eps, want[row * width ..][0..width]);
    var name: [96]u8 = undefined;
    const outs = try r.a.alloc(u64, counts.len);
    for (counts, outs) |m, *o| {
        o.* = try r.out(m * width * size(yk));
        try r.n.rms(r.s, dx, xk, if (weighted) dw else null, o.*, yk, m, width, eps);
    }
    const label = try std.fmt.bufPrint(&name, "rms {t} -> {t}, width {d}{s}", .{ xk, yk, width, if (weighted) "" else ", no weight" });
    const wide = try r.back(outs[counts.len - 1], most * width * size(yk));
    try close(r, label, wide, yk, want, null);
    for (counts[0 .. counts.len - 1], outs[0 .. counts.len - 1]) |m, o|
        try check.sameBytes(label, try r.back(o, m * width * size(yk)), wide[0 .. m * width * size(yk)]);
}

fn addCase(r: *Rig, width: usize, sandwich: bool) !void {
    const x = try stored(r.a, r.values(most * width), .bf16);
    const res0 = try stored(r.a, r.values(most * width), .f32);
    const w1v = r.values(width);
    const w2v = r.values(width);
    for (w1v, w2v) |*p, *q| {
        p.* = 0.5 + @abs(p.*) / 20;
        q.* = 0.5 + @abs(q.*) / 20;
    }
    const w1 = try stored(r.a, w1v, .f32);
    const w2 = try stored(r.a, w2v, .f32);
    const dx = try r.dev(x.bytes);
    const dw1 = try r.dev(w1.bytes);
    const dw2 = try r.dev(w2.bytes);
    const want_res = try r.a.dupe(f64, res0.vals);
    const want_y = try r.a.alloc(f64, most * width);
    const added = try r.a.alloc(f64, most * width);
    const res_tol = try r.a.alloc(f64, most * width);
    const y_tol = try r.a.alloc(f64, most * width);
    for (0..most) |row| {
        const at = row * width;
        const xs = x.vals[at..][0..width];
        if (sandwich) ref.rms(xs, w1.vals, eps, added[at..][0..width]) else @memcpy(added[at..][0..width], xs);
        ref.addRms(xs, want_res[at..][0..width], if (sandwich) w1.vals else null, w2.vals, eps, want_y[at..][0..width]);
        ref.addRmsBounds(res0.vals[at..][0..width], added[at..][0..width], w2.vals, want_y[at..][0..width], want_res[at..][0..width], .bf16, res_tol[at..][0..width], y_tol[at..][0..width]);
    }
    var name: [96]u8 = undefined;
    const label = try std.fmt.bufPrint(&name, "add_rms bf16 + fp32 residual, width {d}{s}", .{ width, if (sandwich) ", sandwich" else "" });
    const ys = try r.a.alloc(u64, counts.len);
    const rs = try r.a.alloc(u64, counts.len);
    for (counts, ys, rs) |m, *y, *rr| {
        rr.* = try r.dev(res0.bytes[0 .. m * width * 4]);
        y.* = try r.out(m * width * 2);
        try r.n.addRms(r.s, dx, .bf16, rr.*, if (sandwich) dw1 else null, dw2, y.*, .bf16, m, width, eps);
    }
    const wide_y = try r.back(ys[counts.len - 1], most * width * 2);
    const wide_r = try r.back(rs[counts.len - 1], most * width * 4);
    try close(r, label, wide_y, .bf16, want_y, y_tol);
    try close(r, label, wide_r, .f32, want_res, res_tol);
    for (counts[0 .. counts.len - 1], ys[0 .. counts.len - 1], rs[0 .. counts.len - 1]) |m, y, rr| {
        try check.sameBytes(label, try r.back(y, m * width * 2), wide_y[0 .. m * width * 2]);
        try check.sameBytes(label, try r.back(rr, m * width * 4), wide_r[0 .. m * width * 4]);
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var n = try cuda.norms.Norms.load(gpu.d);
    defer n.unload();
    var prng = std.Random.DefaultPrng.init(17);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .n = &n, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    for (widths) |w| {
        try rmsCase(&r, w, .f32, .bf16, true);
        try rmsCase(&r, w, .bf16, .bf16, true);
        try rmsCase(&r, w, .f16, .f32, false);
        try addCase(&r, w, true);
        try addCase(&r, w, false);
    }
    // a bf16 row goes to the 16-byte tree by its width alone; a misaligned buffer is refused, not run on the other tree
    const buf = try r.out(2 * 512 + 16);
    const refused = if (n.rms(s, buf + 2, .bf16, null, buf + 2, .bf16, 1, 256, 1e-6)) |_| false else |e| e == error.Misaligned;
    try check.expect(refused, "norms: a misaligned bf16 row is refused", .{});
    check.pass("norms: rms and add_rms within float64 tolerance (worst {d:.3}), every row count's bits equal, a misaligned bf16 row refused", .{r.worst});
}
