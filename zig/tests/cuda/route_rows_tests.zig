//! route_rows.cu's router logits and softmax top-k rows against float64, bits equal at every row count.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");

const rref = core.route_ref;
const aref = core.act_ref;
const counts = [_]usize{ 1, 3, 16, 17, 40 };
const most = counts[counts.len - 1];

fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    return @truncate((u + 0x7fff + ((u >> 16) & 1)) >> 16);
}

fn bf(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

const Rig = struct {
    gpu: check.Gpu,
    a: std.mem.Allocator,
    s: cuda.Stream,
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
        try b.fill8(0xff, r.s.handle);
        return b.ptr;
    }

    fn back(r: *Rig, comptime T: type, ptr: u64, n: usize) ![]T {
        const o = try r.a.alloc(T, n);
        try r.s.synchronize();
        const b: cuda.DeviceBuffer = .{ .d = r.gpu.d, .ptr = ptr, .len = n * @sizeOf(T) };
        try b.download(0, std.mem.sliceAsBytes(o));
        return o;
    }

    fn bfs(r: *Rig, n: usize, scale: f32) []u16 {
        const v = r.a.alloc(u16, n) catch unreachable;
        for (v) |*x| x.* = bf16((r.rng.float(f32) * 2 - 1) * scale);
        return v;
    }

    fn within(r: *Rig, what: []const u8, got: f64, want: f64, tol: f64) !void {
        const q = @abs(got - want) / tol;
        if (!(q <= 1)) try check.expect(false, "{s}: within float64 tolerance (got {d}, want {d}, {d:.3} tolerances)", .{ what, got, want, q });
        r.worst = @max(r.worst, q);
    }
};

/// Router logits at every row count (the few-row and the many-row launches): fp64 dots, then the same bits a row.
fn logits(r: *Rig, g: *const cuda.route_rows.RouteRows, d: usize, e: usize) !void {
    const ld = d + 64; // rows with something after them
    const x = r.bfs(most * ld, 1);
    const w = r.bfs(e * d, 0.05);
    const dx = try r.dev(std.mem.sliceAsBytes(x));
    const dw = try r.dev(std.mem.sliceAsBytes(w));
    var outs: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        outs[c] = try r.out(m * e * 4);
        try g.logits(r.s, dx, ld, dw, outs[c], m, d, e);
    }
    const got = try r.back(f32, outs[counts.len - 1], most * e);
    const xv = try r.a.alloc(f64, d);
    const wv = try r.a.alloc(f64, d);
    for (0..most) |row| {
        for (xv, 0..) |*o, i| o.* = bf(x[row * ld + i]);
        for (0..e) |ex| {
            for (wv, 0..) |*o, i| o.* = bf(w[ex * d + i]);
            try r.within("route logits", got[row * e + ex], rref.dot(xv, wv), rref.dotTolerance(xv, wv));
        }
    }
    for (counts[0 .. counts.len - 1], 0..) |m, c| try check.sameBytes("route logits rows", std.mem.sliceAsBytes(try r.back(f32, outs[c], m * e)), std.mem.sliceAsBytes(got[0 .. m * e]));
}

/// Top-k of each row (a row of ties too): the picks the fp64 rule makes, weights and the shared gate within bf16.
fn topk(r: *Rig, g: *const cuda.route_rows.RouteRows, experts: usize, k: usize, shared: bool) !void {
    const ld = experts + 1;
    const slots = k + @intFromBool(shared);
    const l = try r.a.alloc(f32, most * ld);
    for (l) |*v| v.* = (r.rng.float(f32) * 2 - 1) * 4;
    for (0..experts) |i| l[i] = @floatFromInt(i % 3); // row 0: every pick a tie broken by the lower id
    const dl = try r.dev(std.mem.sliceAsBytes(l));
    var picks: [counts.len]u64 = undefined;
    var wts: [counts.len]u64 = undefined;
    for (counts, 0..) |m, c| {
        picks[c] = try r.out(m * slots * 4);
        wts[c] = try r.out(m * slots * 4);
        try g.topk(r.s, dl, ld, experts, k, picks[c], wts[c], slots, m, shared);
    }
    const p = try r.back(i32, picks[counts.len - 1], most * slots);
    const wt = try r.back(f32, wts[counts.len - 1], most * slots);
    const row = try r.a.alloc(f64, experts);
    const ids = try r.a.alloc(usize, k);
    const want = try r.a.alloc(f64, k);
    for (0..most) |i| {
        for (row, 0..) |*o, j| o.* = l[i * ld + j];
        rref.topk(row, ids, want);
        for (0..k) |j| {
            try check.expect(p[i * slots + j] == @as(i32, @intCast(ids[j])), "route top-k row {d} pick {d}: expert {d}, the rule picks {d}", .{ i, j, p[i * slots + j], ids[j] });
            try r.within("route top-k weight", wt[i * slots + j], want[j], aref.tolerance(want[j]));
        }
        if (shared) {
            try check.expect(p[i * slots + k] == @as(i32, @intCast(experts)), "route top-k row {d}: the shared slot names expert {d}", .{ i, p[i * slots + k] });
            const gate = aref.sigmoid(bf(bf16(l[i * ld + experts])));
            try r.within("route shared gate", wt[i * slots + k], gate, aref.tolerance(gate));
        }
    }
    for (counts[0 .. counts.len - 1], 0..) |m, c| {
        try check.sameBytes("route top-k picks", std.mem.sliceAsBytes(try r.back(i32, picks[c], m * slots)), std.mem.sliceAsBytes(p[0 .. m * slots]));
        try check.sameBytes("route top-k weights", std.mem.sliceAsBytes(try r.back(f32, wts[c], m * slots)), std.mem.sliceAsBytes(wt[0 .. m * slots]));
    }
}

pub fn run(gpu: check.Gpu) !void {
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    var prng = std.Random.DefaultPrng.init(41);
    var r: Rig = .{ .gpu = gpu, .a = arena.allocator(), .s = try cuda.Stream.init(gpu.d, false), .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
        r.s.deinit();
    }
    var g = try cuda.route_rows.RouteRows.load(gpu.d);
    defer g.unload();
    try logits(&r, &g, 2560, 513); // Flash Next: 512 experts and the shared expert's gate row
    try logits(&r, &g, 2688, 128);
    try logits(&r, &g, 1040, 37); // a K tail past the staged 256s, and experts past a block
    try topk(&r, &g, 512, 10, true);
    try topk(&r, &g, 128, 8, false);
    check.pass("route rows: logits and top-k within float64 tolerance (worst {d:.3}), the fp64 rule's picks, every row count's bits equal", .{r.worst});
}
