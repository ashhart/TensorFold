//! `ling-weights-real DIR REFDIR`: real FP8 layers 0, 4, 7 (experts 0-3) vs float64 and torch (oracle/ling_weights.py).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const ling = @import("ling_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const lw = @import("ling_weights_tests.zig");

const Rig = qr.RigT;
const h = qr.helpers;
const W = ling.weights;
const rows = 8;

/// Torch's rows for a module: x [8][k] bf16, y and sum |x w| [8][n] float64.
const Ref = struct { x: []const u16, y: []const f64, abs: []const f64 };

fn readRef(r: *Rig, dir: []const u8, module: []const u8, k: usize, n: usize) !Ref {
    const io = r.gpu.io;
    var b: [256]u8 = undefined;
    const x = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&b, "{s}/x{d}.bin", .{ dir, k }), r.a, .unlimited);
    const y = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&b, "{s}/{s}.y", .{ dir, module }), r.a, .unlimited);
    const abs = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&b, "{s}/{s}.abs", .{ dir, module }), r.a, .unlimited);
    if (x.len != rows * k * 2 or y.len != rows * n * 8 or abs.len != y.len) return error.BadReference;
    const xs = try r.a.alloc(u16, rows * k);
    @memcpy(std.mem.sliceAsBytes(xs), x);
    const ys = try r.a.alloc(f64, rows * n);
    @memcpy(std.mem.sliceAsBytes(ys), y);
    const as = try r.a.alloc(f64, rows * n);
    @memcpy(std.mem.sliceAsBytes(as), abs);
    return .{ .x = xs, .y = ys, .abs = as };
}

/// Agreement with torch: the largest difference as a fraction of the bound, and the relative L2 error.
const Stat = struct {
    worst: f64 = 0,
    rel: f64 = 0, // the largest relative L2 over the checks
    n: usize = 0,

    fn add(s: *Stat, name: []const u8, got: []const f64, want: []const f64, tol: []const f64) !void {
        var most: f64 = 0;
        var e2: f64 = 0;
        var y2: f64 = 0;
        for (got, want, tol) |g, w, t| {
            most = @max(most, @abs(g - w) / t);
            e2 += (g - w) * (g - w);
            y2 += w * w;
        }
        const rel = @sqrt(e2 / @max(y2, 1e-300));
        std.debug.print("RESULT torch {s}: {d:.3} of its bound, rel L2 {e:.3}\n", .{ name, most, rel });
        s.worst = @max(s.worst, most);
        s.rel = @max(s.rel, rel);
        s.n += 1;
        try check.expect(most <= 1, "{s} within its bound against torch", .{name});
    }
};

/// The bf16 (or fp32) output at `ptr` as float64.
fn values(r: *Rig, ptr: u64, count: usize, f32_out: bool) ![]f64 {
    const raw = try r.back(ptr, count * @as(usize, if (f32_out) 4 else 2));
    const out = try r.a.alloc(f64, count);
    for (out, 0..) |*v, i| v.* = if (f32_out) std.mem.bytesToValue(f32, raw[4 * i ..][0..4]) else h.bfOf(std.mem.bytesToValue(u16, raw[2 * i ..][0..2]));
    return out;
}

/// The projection named `module` among the loaded layers, or null (routed experts and the router are not projections).
fn find(w: *const W.Weights, module: []const u8) ?W.Proj {
    var b: [192]u8 = undefined;
    for (w.layers) |*l| {
        var list: [16]lw.Item = undefined;
        for (list[0..lw.projections(l, &list)]) |e| if (std.mem.eql(u8, ling.names.moduleName(&b, l.index, e[0], null) catch continue, module)) return e[1];
    }
    return null;
}

fn bound(ref: Ref, i: usize, fp32: bool) f64 {
    return (if (fp32) 0 else @abs(ref.y[i]) * std.math.pow(f64, 2, -7)) + ref.abs[i] * std.math.pow(f64, 2, -12) + 1e-30;
}

/// A projection's 8 decode rows against torch's.
fn projection(r: *Rig, kit: *const lw.Kit, st: *Stat, dir: []const u8, module: []const u8, p: W.Proj) !void {
    const n = p.outputs();
    const k = p.inputs();
    const ref = try readRef(r, dir, module, k, n);
    const out = try r.zeros(rows * n * 2);
    try kit.linear().decode(r.s, p, .{ .x = try r.dev(std.mem.sliceAsBytes(ref.x)), .ldx = k, .part = try r.zeros(8 * rows * n * 4) }, out, rows);
    const tol = try r.a.alloc(f64, rows * n);
    for (tol, 0..) |*t, i| t.* = bound(ref, i, false);
    try st.add(module, try values(r, out, rows * n, false), ref.y, tol);
}

/// The router as loaded (bf16 [experts][hidden]) times torch's rows in float64, against torch's.
fn router(r: *Rig, st: *Stat, dir: []const u8, module: []const u8, l: *const W.Layer, c: *const ling.config.Config) !void {
    const n: usize = c.experts;
    const k: usize = c.hidden;
    const ref = try readRef(r, dir, module, k, n);
    const w = try r.back(l.ffn.moe.router, n * k * 2);
    const got = try r.a.alloc(f64, rows * n);
    for (0..rows) |row| for (0..n) |j| {
        var s: f64 = 0;
        for (0..k) |i| s += h.bfOf(ref.x[row * k + i]) * h.bfOf(std.mem.bytesToValue(u16, w[2 * (j * k + i) ..][0..2]));
        got[row * n + j] = s;
    };
    const tol = try r.a.alloc(f64, rows * n);
    for (tol, ref.abs) |*t, a| t.* = a * std.math.pow(f64, 2, -40) + 1e-30; // float64 sums both sides
    try st.add(module, got, ref.y, tol);
}

/// Experts 0-3 of layer i: gate-up on torch's rows 0-1 of x (SwiGLU of torch's gate and up), down on its 8 pair rows.
fn expertsVsTorch(r: *Rig, kit: *const lw.Kit, st: *Stat, dir: []const u8, i: u32, l: cuda.experts.Layer) !void {
    const slots = 4;
    const pairs = 2 * slots;
    const ni = l.width;
    const d = l.dims;
    const pk = try r.a.alloc(i32, pairs);
    for (pk, 0..) |*e, p| e.* = lw.pick(p, slots, l.experts);
    const items = cuda.grouped.maxItems(pairs, l.experts, 16);
    const plan: cuda.grouped.Plan = .{ .members = try r.zeros(pairs * 4), .items = try r.zeros(items * 12), .counts = try r.zeros(8), .rank = try r.zeros(4), .hist = try r.zeros(4) };
    try kit.router.route(r.s, try r.dev(std.mem.sliceAsBytes(pk)), pairs, l.experts, 16, plan);
    var nolimit = l;
    nolimit.limit = 0;
    var b: [192]u8 = undefined;
    const want = try r.a.alloc(f64, pairs * ni);
    const tol = try r.a.alloc(f64, pairs * ni);
    const dwant = try r.a.alloc(f64, pairs * d);
    const dtol = try r.a.alloc(f64, pairs * d);
    var x: []const u16 = &.{};
    var xd: []const u16 = &.{};
    for (0..pairs) |p| {
        const e: u32 = @intCast(pk[p]);
        const g = try readRef(r, dir, try ling.names.moduleName(&b, i, .expert_gate, e), d, ni);
        const u = try readRef(r, dir, try ling.names.moduleName(&b, i, .expert_up, e), d, ni);
        const dn = try readRef(r, dir, try ling.names.moduleName(&b, i, .expert_down, e), ni, d);
        x = g.x;
        xd = dn.x;
        const row = p / slots;
        for (0..ni) |j| {
            const gy = g.y[row * ni + j];
            want[p * ni + j] = lw.swiglu(gy, u.y[row * ni + j]);
            const gt = bound(g, row * ni + j, false);
            const ut = bound(u, row * ni + j, false);
            tol[p * ni + j] = @abs(want[p * ni + j]) * std.math.pow(f64, 2, -6) + 1.1 * @abs(u.y[row * ni + j]) * gt + @abs(gy / (1 + @exp(-gy))) * ut + 1e-30;
        }
        for (0..d) |j| {
            dwant[p * d + j] = dn.y[p * d + j];
            dtol[p * d + j] = bound(dn, p * d + j, true);
        }
    }
    const act = try r.zeros(pairs * ni * 2);
    try kit.grouped().decode(r.s, .up, .{ .x = try r.dev(std.mem.sliceAsBytes(x[0 .. 2 * d])), .stride = d, .slots = slots }, nolimit, plan, act, items, -1);
    const y = try r.zeros(pairs * d * 4);
    try kit.grouped().decode(r.s, .down_f32, .{ .x = try r.dev(std.mem.sliceAsBytes(xd)), .stride = ni }, nolimit, plan, y, items, -1);
    var name: [96]u8 = undefined;
    try st.add(try std.fmt.bufPrint(&name, "layer {d} experts 0-3 gate-up (SwiGLU)", .{i}), try values(r, act, pairs * ni, false), want, tol);
    try st.add(try std.fmt.bufPrint(&name, "layer {d} experts 0-3 down", .{i}), try values(r, y, pairs * d, true), dwant, dtol);
}

pub fn run(gpu: check.Gpu, dir: []const u8, refdir: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    var s = try cuda.Stream.init(gpu.d, false);
    defer s.deinit();
    var prng = std.Random.DefaultPrng.init(0x11a6_0047);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = s, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*x| x.free();
        r.bufs.deinit(gpu.gpa);
    }
    var kit: lw.Kit = undefined;
    try kit.init(gpu);
    defer kit.deinit();
    var why: ling.config.Why = .{};
    var o = W.open(gpu.gpa, gpu.io, dir, &why) catch |e| {
        std.debug.print("{s}: {s}\n", .{ dir, why.text() });
        return e;
    };
    defer o.ck.close();
    const t0 = check.now(gpu.io);
    var w = try W.load(gpu.gpa, gpu.d, s, &kit.pack, &o.ck, o.c, &.{ 0, 4, 7 }, .{ .globals = false, .experts = 4 });
    defer w.deinit();
    const secs = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e9;
    var bytes: usize = 0;
    for (w.store.named.items) |n| bytes += n.len;
    std.debug.print("RESULT loaded layers 0, 4, 7 (4 experts each) in {d:.1} s: {d} MiB on the device\n", .{ secs, bytes >> 20 });
    const n = try lw.layers(&r, &kit, &o.ck, &w);
    var st: Stat = .{};
    const manifest = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fmt.allocPrint(r.a, "{s}/manifest.txt", .{refdir}), r.a, .unlimited);
    var lines = std.mem.tokenizeScalar(u8, manifest, '\n');
    while (lines.next()) |line| {
        const module = line[0 .. std.mem.indexOfScalar(u8, line, ' ') orelse return error.BadReference];
        if (std.mem.indexOf(u8, module, ".experts.") != null) continue;
        if (std.mem.endsWith(u8, module, ".mlp.gate")) {
            const i = try std.fmt.parseInt(u32, module["model.layers.".len..std.mem.indexOfScalarPos(u8, module, "model.layers.".len, '.').?], 10);
            try router(&r, &st, refdir, module, w.layer(i).?, &w.config);
            continue;
        }
        try projection(&r, &kit, &st, refdir, module, find(&w, module) orelse return error.NotLoaded);
    }
    for ([_]u32{ 4, 7 }) |i| try expertsVsTorch(&r, &kit, &st, refdir, i, w.layer(i).?.ffn.moe.routed.grouped);
    check.pass("ling-weights-real: {d} projections within float64 bounds from the file (worst {d:.3}), {d} small tensors byte-equal; {d} checks against torch, worst {d:.3} of the bound, rel L2 at most {e:.3}", .{ n.proj, r.worst, n.small, st.n, st.worst, st.rel });
}
