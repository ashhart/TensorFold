//! Deterministic device gate: packed projections, SwiGLU, pruning and mixed FP8/NVFP4 vs independent FP64.
const std = @import("std");
const Gpu = @import("gpu.zig").Gpu;
const k = @import("kernels.zig");
const q = @import("quant_format.zig");
const ref = @import("quant_reference.zig");

const H = 2048;
const I = 512;
const E = 8;
const max_rows = 3;
const pairs = max_rows * E;

const Score = struct {
    count: usize = 0,
    worst: f64 = 0,
    worst_column: usize = 0,
    squared: f64 = 0,
    fn add(s: *Score, got: f64, want: f64, bound: f64, col: usize) !void {
        if (!std.math.isFinite(got) or !std.math.isFinite(want)) return error.NonfiniteQuantResult;
        const delta = @abs(got - want);
        s.count += 1;
        s.squared += delta * delta;
        if (delta > s.worst) {
            s.worst = delta;
            s.worst_column = col;
        }
        if (delta > bound) {
            std.debug.print("FAIL quant column={d} got={d} FP64={d} abs={d} bound={d}\n", .{ col, got, want, delta, bound });
            return error.QuantReferenceMismatch;
        }
    }
    fn report(s: Score, label: []const u8) void {
        std.debug.print("SUMMARY {s} PASS count={d} max_abs={d} worst_column={d} rms={d}\n", .{ label, s.count, s.worst, s.worst_column, @sqrt(s.squared / @as(f64, @floatFromInt(s.count))) });
    }
};

fn put(g: *Gpu, bytes: []const u8, indirect: bool) !u64 {
    const a = try g.alloc(bytes.len);
    @memcpy(g.host(a, bytes.len), bytes);
    if (indirect) try g.indirect.append(g.gpa, g.ref(a).buf);
    return a;
}

fn bfRows(g: *Gpu, a: u64, n: usize) []align(1) u16 {
    return std.mem.bytesAsSlice(u16, @as([]align(1) u8, g.host(a, n * 2)));
}

const Projection = struct {
    w: u64,
    s: u64,
    global: u64,
    n: usize,
    dim: usize,
    fn init(g: *Gpu, n: usize, dim: usize, seed: usize) !Projection {
        const w = try g.alloc(n * dim / 2);
        const s = try g.alloc(n * dim / 16);
        for (g.host(w, n * dim / 2), 0..) |*b, j| b.* = @truncate((j * 37 + seed * 19) % 256);
        for (g.host(s, n * dim / 16), 0..) |*b, j| b.* = @intCast((j + seed * 11) % 127);
        const global: f32 = @as(f32, @floatFromInt(seed + 1)) / 1024;
        const gp = try put(g, std.mem.asBytes(&global), true);
        try g.indirect.append(g.gpa, g.ref(w).buf);
        try g.indirect.append(g.gpa, g.ref(s).buf);
        return .{ .w = w, .s = s, .global = gp, .n = n, .dim = dim };
    }
    fn scalar(p: Projection, g: *Gpu, row: usize, x: []align(1) const u16) f64 {
        const global: f32 = @bitCast(std.mem.readInt(u32, g.host(p.global, 4)[0..4], .little));
        return ref.projection(g.host(p.w, p.n * p.dim / 2), g.host(p.s, p.n * p.dim / 16), global, row, p.dim, x);
    }
};

/// Run only in an exclusive GPU window. The caller opens the newly built drama3 metallib, not a model checkpoint.
pub fn run(g: *Gpu) !void {
    try g.flush();
    const first = g.bufs.items.len;
    const indirect_first = g.indirect.items.len;
    defer {
        g.flush() catch {};
        g.indirect.shrinkRetainingCapacity(indirect_first);
        g.releaseFrom(first);
    }
    const gate = try Projection.init(g, I, H, 0);
    const up = try Projection.init(g, I, H, 1);
    const down = try Projection.init(g, H, I, 2);
    var pointers: [E * 9]u64 = undefined;
    for (0..E) |e| for ([_]Projection{ gate, up, down }, 0..) |p, j| {
        pointers[e * 9 + j * 3] = g.gpuAddress(p.w);
        pointers[e * 9 + j * 3 + 1] = g.gpuAddress(p.s);
        pointers[e * 9 + j * 3 + 2] = g.gpuAddress(p.global);
    };
    const table = try put(g, std.mem.sliceAsBytes(&pointers), false);
    const x = try g.alloc(max_rows * H * 2);
    for (bfRows(g, x, max_rows * H), 0..) |*b, j| b.* = q.bf16(@as(f32, @floatFromInt(@as(i32, @intCast((j * 13) % 29)) - 14)) / 512);
    const logits = try g.alloc(max_rows * 256 * 2);
    const original = [_]usize{ 1, 7, 19, 32, 65, 127, 200, 255 };
    var slot_map: [256]i32 = @splat(-1);
    for (original, 0..) |id, slot| slot_map[id] = @intCast(slot);
    const slots = try put(g, std.mem.sliceAsBytes(&slot_map), false);
    // Excluded experts deliberately have the largest logits; retained ones tie, proving original-ID tie order.
    for (bfRows(g, logits, max_rows * 256), 0..) |*b, j| b.* = q.bf16(if (slot_map[j % 256] >= 0) -2 else 100);
    const picks = try g.alloc(pairs * 4);
    const rw = try g.alloc(pairs * 4);
    const counts = try g.alloc(256 * 4);
    const members = try g.alloc(256 * pairs * 4);
    const gu = try g.alloc(pairs * I * 2);
    const output = try g.alloc(pairs * H * 4);
    var up_score: Score = .{};
    var down_score: Score = .{};
    for ([_]usize{ 1, max_rows }) |rows| {
        k.routerCompact(g, logits, 256, 0, picks, rw, slots, rows);
        if (rows > 1) k.plan(g, picks, rows * E, counts, members, pairs);
        const route: k.Route = .{ .table = table, .picks = picks, .counts = counts, .members = members, .cap = pairs, .pairs = rows * E, .prompt = rows > 1, .experts = E };
        k.expertsNvfp4(g, true, x, route, H, I, gu);
        k.expertsNvfp4(g, false, gu, route, I, H, output);
        try g.flush();
        for (0..rows * E) |pair| {
            const picked = std.mem.readInt(i32, g.host(picks + pair * 4, 4)[0..4], .little);
            if (picked != pair % E) return error.PrunedRoutingMismatch;
            const weight: f32 = @bitCast(std.mem.readInt(u32, g.host(rw + pair * 4, 4)[0..4], .little));
            if (weight != 0.125) return error.PrunedRoutingMismatch;
            const xr = bfRows(g, x + (pair / E) * H * 2, H);
            const gr = bfRows(g, gu + pair * I * 2, I);
            for (0..I) |col| {
                const want = ref.swiglu(gate.scalar(g, col, xr), up.scalar(g, col, xr));
                try up_score.add(ref.fromBf16(gr[col]), want, 0.000002 + @abs(want) / 64, col);
            }
            for (0..H) |col| {
                const want = down.scalar(g, col, gr);
                const got: f32 = @bitCast(std.mem.readInt(u32, g.host(output + (pair * H + col) * 4, 4)[0..4], .little));
                try down_score.add(got, want, 0.0002 + @abs(want) * 0.00002, col);
            }
        }
        try combine(g, output, rw, x, rows);
        try rawProjection(g, route, gate, x, output, rows, "drama3 NVFP4 gate FP32");
        try rawProjection(g, route, up, x, output, rows, "drama3 NVFP4 up FP32");
        try fp8Layer(g, route, x, gu, output, rows);
    }
    up_score.report("drama3 NVFP4 gate/up SwiGLU decode+prompt");
    down_score.report("drama3 NVFP4 down FP32 decode+prompt");
}

fn combine(g: *Gpu, down: u64, weights: u64, x: u64, rows: usize) !void {
    const residual = try g.alloc(rows * H * 2);
    const shared = try g.alloc(rows * H * 2);
    const wg = try g.alloc(H * 2);
    for (bfRows(g, residual, rows * H), 0..) |*v, i| v.* = q.bf16(@as(f32, @floatFromInt(i % 13)) / 32);
    for (bfRows(g, shared, rows * H), 0..) |*v, i| v.* = q.bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 16);
    for (bfRows(g, wg, H), 0..) |*v, i| v.* = q.bf16(@as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 64);
    k.moeCombineNvfp4(g, residual, down, weights, shared, x, wg, H, rows);
    try g.flush();
    var score: Score = .{};
    for (0..rows) |row| {
        var dot: f64 = 0;
        for (0..H) |j| dot += ref.fromBf16(bfRows(g, x, rows * H)[row * H + j]) * ref.fromBf16(bfRows(g, wg, H)[j]);
        const gate = 1 / (1 + @exp(-dot));
        for (0..H) |col| {
            var sum: f64 = 0;
            for (0..E) |slot| {
                const weight: f32 = @bitCast(std.mem.readInt(u32, g.host(weights + (row * E + slot) * 4, 4)[0..4], .little));
                const d: f32 = @bitCast(std.mem.readInt(u32, g.host(down + ((row * E + slot) * H + col) * 4, 4)[0..4], .little));
                sum += ref.roundBf16(@as(f32, weight * d));
            }
            const sh = ref.fromBf16(bfRows(g, shared, rows * H)[row * H + col]);
            const old = ref.roundBf16(@as(f64, @floatFromInt((row * H + col) % 13)) / 32);
            const want = ref.roundBf16(old + ref.roundBf16(gate * sh + ref.roundBf16(sum)));
            const got = ref.fromBf16(bfRows(g, residual, rows * H)[row * H + col]);
            try score.add(got, want, 0.00001 + @abs(want) / 128, col);
        }
    }
    score.report("drama3 NVFP4 pair rounding shared gate residual");
}

fn rawProjection(g: *Gpu, original_route: k.Route, p: Projection, x: u64, output: u64, rows: usize, label: []const u8) !void {
    var ptrs: [E * 9]u64 = @splat(0);
    for (0..E) |e| {
        ptrs[e * 9 + 6] = g.gpuAddress(p.w);
        ptrs[e * 9 + 7] = g.gpuAddress(p.s);
        ptrs[e * 9 + 8] = g.gpuAddress(p.global);
    }
    var route = original_route;
    route.table = try put(g, std.mem.sliceAsBytes(&ptrs), false);
    const repeated = try g.alloc(rows * E * H * 2);
    for (0..rows * E) |pair|
        @memcpy(g.host(repeated + pair * H * 2, H * 2), g.host(x + (pair / E) * H * 2, H * 2));
    k.expertsNvfp4(g, false, repeated, route, H, I, output);
    try g.flush();
    var score: Score = .{};
    for (0..rows * E) |pair| for (0..I) |col| {
        const want = p.scalar(g, col, bfRows(g, repeated + pair * H * 2, H));
        const got: f32 = @bitCast(std.mem.readInt(u32, g.host(output + (pair * I + col) * 4, 4)[0..4], .little));
        try score.add(got, want, 0.00002 + @abs(want) * 0.00002, col);
    };
    score.report(label);
}

fn fp8Layer(g: *Gpu, original_route: k.Route, x: u64, gu: u64, output: u64, rows: usize) !void {
    const w = try g.alloc(H * I);
    for (g.host(w, H * I), 0..) |*b, j| b.* = @intCast(0x20 + j % 32);
    const scales = try g.alloc((H / 128) * (I / 128) * 4);
    const fs = std.mem.bytesAsSlice(f32, @as([]align(1) u8, g.host(scales, (H / 128) * (I / 128) * 4)));
    @memset(fs, 0.0078125);
    try g.indirect.append(g.gpa, g.ref(w).buf);
    try g.indirect.append(g.gpa, g.ref(scales).buf);
    var ptrs: [E * 6]u64 = undefined;
    for (0..E * 3) |i| {
        ptrs[i * 2] = g.gpuAddress(w);
        ptrs[i * 2 + 1] = g.gpuAddress(scales);
    }
    var route = original_route;
    route.table = try put(g, std.mem.sliceAsBytes(&ptrs), false);
    const xq = try g.alloc(rows * H * 2);
    const xs = try g.alloc(rows * (H / 128) * 4);
    const gq = try g.alloc(rows * E * I * 2);
    const gs = try g.alloc(rows * E * (I / 128) * 4);
    k.actQuant(g, x, H, xq, H, xs, rows, H);
    k.experts(g, true, xq, xs, route, H, I, gu);
    k.actQuant(g, gu, I, gq, I, gs, rows * E, I);
    k.experts(g, false, gq, gs, route, I, H, output);
    try g.flush();
    var score: Score = .{};
    for (0..rows * E) |pair| for (0..H) |col| {
        var want: f64 = 0;
        const act = bfRows(g, gq + pair * I * 2, I);
        for (0..I) |j| {
            const scale: f32 = @bitCast(std.mem.readInt(u32, g.host(gs + (pair * (I / 128) + j / 128) * 4, 4)[0..4], .little));
            want += ref.fp8(g.host(w, H * I)[col * I + j]) * ref.fromBf16(act[j]) * scale * 0.0078125;
        }
        const got: f32 = @bitCast(std.mem.readInt(u32, g.host(output + (pair * H + col) * 4, 4)[0..4], .little));
        try score.add(got, want, 0.0001 + @abs(want) * 0.00002, col);
    };
    score.report("drama3 mixed-layer legacy FP8 down");
}
