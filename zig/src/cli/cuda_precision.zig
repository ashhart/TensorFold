//! check-precision: ModelOpt projections on real weights and activations against fp64, beside NVIDIA's own math.

const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const nemotron = @import("nemotron");

/// bf16 rows as read back from a dump file.
const Rows16 = []align(1) const u16;

fn e4m3(b: u8) f64 {
    const sign: f64 = if (b & 0x80 != 0) -1 else 1;
    const e: i32 = (b >> 3) & 0xF;
    const m: f64 = @floatFromInt(b & 7);
    return sign * if (e == 0) m / 8.0 * 0.015625 else (1 + m / 8.0) * std.math.pow(f64, 2, @floatFromInt(e - 7));
}

/// x rounded to the nearest e4m3 (ties to even), saturating at 448, as an FP8 activation quantizer does.
fn toE4m3(x: f64) f64 {
    const a = @min(@abs(x), 448.0);
    if (a == 0) return 0;
    const e = @max(std.math.floor(std.math.log2(a)), -6.0); // below 2^-6 the step is the subnormals' 2^-9
    const step = std.math.pow(f64, 2, e - 3);
    var q = @round(a / step);
    if (@abs(a / step - std.math.trunc(a / step) - 0.5) < 1e-12 and @mod(q, 2) == 1) q -= 1;
    return std.math.copysign(@min(q * step, 448.0), x);
}

fn e2m1(n: u8) f64 {
    const mag = [8]f64{ 0, 0.5, 1, 1.5, 2, 3, 4, 6 };
    return (if (n & 8 != 0) @as(f64, -1) else 1) * mag[n & 7];
}

fn bf(v: u16) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, v) << 16)));
}

fn round16(x: f64) f64 {
    const u: u32 = @bitCast(@as(f32, @floatCast(x)));
    return bf(@truncate((u + 0x7FFF + ((u >> 16) & 1)) >> 16));
}

/// A checkpoint matrix [n, k] as stored: bf16, e4m3 times one scale, or NVFP4 codes, block scales and one scale.
const Matrix = struct {
    kind: enum { bf16, fp8, nvfp4 },
    w: []const u8,
    blocks: []const u8 = &.{},
    scale: f64 = 1,
    in_scale: ?f64 = null, // FP8's static activation scale (NVIDIA's W8A8)
    n: usize,
    k: usize,

    fn open(ck: *core.Checkpoint, pre: []const u8) !Matrix {
        var buf: [192]u8 = undefined;
        const w = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight", .{pre}));
        switch (w.dtype) {
            .bf16 => return .{ .kind = .bf16, .w = w.bytes, .n = w.dim(0), .k = w.dim(1) },
            .f8_e4m3 => return .{ .kind = .fp8, .w = w.bytes, .scale = try one(ck, &buf, pre, "weight_scale"), .in_scale = try one(ck, &buf, pre, "input_scale"), .n = w.dim(0), .k = w.dim(1) },
            .u8 => {
                const b = try ck.get(try std.fmt.bufPrint(&buf, "{s}.weight_scale", .{pre}));
                return .{ .kind = .nvfp4, .w = w.bytes, .blocks = b.bytes, .scale = try one(ck, &buf, pre, "weight_scale_2"), .n = w.dim(0), .k = w.dim(1) * 2 };
            },
            else => return error.UnsupportedQuantization,
        }
    }

    fn one(ck: *core.Checkpoint, buf: []u8, pre: []const u8, leaf: []const u8) !f64 {
        const t = try ck.get(try std.fmt.bufPrint(buf, "{s}.{s}", .{ pre, leaf }));
        return @floatCast(@as(f32, @bitCast(std.mem.readInt(u32, t.bytes[0..4], .little))));
    }

    /// Row r's values in fp64 over inputs [c0, c0 + out.len).
    fn row(m: Matrix, r: usize, c0: usize, out: []f64) void {
        for (out, c0..) |*v, c| v.* = switch (m.kind) {
            .bf16 => bf(std.mem.readInt(u16, m.w[(r * m.k + c) * 2 ..][0..2], .little)),
            .fp8 => e4m3(m.w[r * m.k + c]) * m.scale,
            .nvfp4 => blk: {
                const byte = m.w[r * (m.k / 2) + c / 2];
                break :blk e2m1(if (c & 1 == 0) byte & 0xF else byte >> 4) * e4m3(m.blocks[r * (m.k / 16) + c / 16]) * m.scale;
            },
        };
    }
};

/// Error sums of one projection: fp64 reference against ours (decode rows, prompt rows) and NVIDIA's math.
const Stats = struct {
    ref: f64 = 0,
    err: [3]f64 = .{ 0, 0, 0 },
    worst: [3]f64 = .{ 0, 0, 0 },
    count: usize = 0,

    fn add(s: *Stats, ref: f64, got: [3]f64) void {
        s.ref += ref * ref;
        s.count += 1;
        for (got, &s.err, &s.worst) |g, *e, *w| {
            e.* += (g - ref) * (g - ref);
            w.* = @max(w.*, @abs(g - ref));
        }
    }

    fn print(s: Stats, name: []const u8, kind: []const u8) void {
        const rms = @sqrt(s.ref / @as(f64, @floatFromInt(s.count)));
        var rel: [3]f64 = undefined;
        for (s.err, &rel) |e, *r| r.* = @sqrt(e / @as(f64, @floatFromInt(s.count))) / rms;
        std.debug.print("{s:<22} {s:<6} {d:>9} {e:>10.3} | {e:>9.3} {e:>9.3} {e:>9.3} | {e:>9.3} {e:>9.3} {e:>9.3}\n", .{ name, kind, s.count, rms, rel[0], rel[1], rel[2], s.worst[0] / rms, s.worst[1] / rms, s.worst[2] / rms });
    }
};

fn readBin(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, block: usize, leaf: []const u8) ![]u8 {
    var buf: [512]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "{s}/{d:0>3}_{s}.bin", .{ dir, block, leaf });
    return std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .unlimited);
}

const Rig = struct { gpa: std.mem.Allocator, e: *nemotron.Engine, x: cuda.DeviceBuffer, y: cuda.DeviceBuffer };

/// One dense projection over `rows` rows of x [rows, k] (bf16): our decode and prompt rows against fp64 and NVIDIA's.
fn dense(r: *Rig, ck: *core.Checkpoint, label: []const u8, pres: []const []const u8, q: nemotron.kernels.QLinear, x: Rows16, rows: usize) !void {
    const o = r.e.ops();
    const n = q.outputs();
    const k = q.inputs();
    try o.upload(r.x.ptr, std.mem.sliceAsBytes(x[0 .. rows * k]));
    var ours: [2][]u16 = undefined;
    for (0..2) |path| {
        if (path == 0) try o.dense(r.x.ptr, 0, q, r.y.ptr, rows) else try o.prefillDense(r.x.ptr, q, r.y.ptr, rows);
        ours[path] = try r.gpa.alloc(u16, rows * n);
        try o.download(std.mem.sliceAsBytes(ours[path]), r.y.ptr);
        try o.s.synchronize();
    }
    defer for (ours) |b| r.gpa.free(b);
    var s: Stats = .{};
    var kind: []const u8 = "";
    const w = try r.gpa.alloc(f64, k);
    defer r.gpa.free(w);
    const xs = try r.gpa.alloc(f64, 2 * rows * k); // the rows, then as NVIDIA's math feeds them
    defer r.gpa.free(xs);
    var at: usize = 0;
    for (pres) |pre| {
        const m = try Matrix.open(ck, pre);
        kind = @tagName(m.kind);
        for (xs[0 .. rows * k], xs[rows * k ..], x[0 .. rows * k]) |*a, *a8, v| {
            a.* = bf(v);
            a8.* = if (m.in_scale) |si| toE4m3(a.* / si) * si else a.*;
        }
        for (0..m.n) |c| {
            m.row(c, 0, w);
            for (0..rows) |i| {
                var ref: f64 = 0;
                var nv: f64 = 0;
                for (xs[i * k ..][0..k], xs[(rows + i) * k ..][0..k], w) |a, a8, wv| {
                    ref += a * wv;
                    nv += a8 * wv;
                }
                s.add(ref, .{ bf(ours[0][i * n + at + c]), bf(ours[1][i * n + at + c]), round16(nv) });
            }
        }
        at += m.n;
    }
    s.print(label, kind);
}

/// A MoE layer's experts on the forward's own pairs: up (relu^2) from y, down from our own activations.
fn experts(gpa: std.mem.Allocator, io: std.Io, ck: *core.Checkpoint, dir: []const u8, block: usize, c: nemotron.Config, rows: usize) !void {
    var raw: [4][]u8 = undefined;
    for (&raw, [_][]const u8{ "y", "pick", "act", "ymoe" }) |*b, leaf| b.* = try readBin(gpa, io, dir, block, leaf);
    defer for (raw) |b| gpa.free(b);
    const y: Rows16 = std.mem.bytesAsSlice(u16, raw[0]);
    const pick: []align(1) const i32 = std.mem.bytesAsSlice(i32, raw[1]);
    const act: Rows16 = std.mem.bytesAsSlice(u16, raw[2]);
    const ymoe: Rows16 = std.mem.bytesAsSlice(u16, raw[3]);
    const W = c.expert_width;
    const D = c.hidden;
    var up: Stats = .{};
    var down: Stats = .{};
    const w = try gpa.alloc(f64, @max(W, D));
    defer gpa.free(w);
    var buf: [192]u8 = undefined;
    for (0..rows * c.slots()) |p| {
        const ex: usize = @intCast(pick[p]);
        const shared = ex >= c.experts;
        const mid = if (shared) "shared_experts" else try std.fmt.bufPrint(&buf, "experts.{d}", .{ex});
        var pb: [2][192]u8 = undefined;
        const mu = try Matrix.open(ck, try std.fmt.bufPrint(&pb[0], "backbone.layers.{d}.mixer.{s}.up_proj", .{ block, mid }));
        const md = try Matrix.open(ck, try std.fmt.bufPrint(&pb[1], "backbone.layers.{d}.mixer.{s}.down_proj", .{ block, mid }));
        const half = if (shared) ex - c.experts else 0;
        const xr = y[(p / c.slots()) * D ..][0..D];
        for (0..W) |j| {
            mu.row(half * W + j, 0, w[0..D]);
            var pre: f64 = 0;
            for (xr, w[0..D]) |a, b| pre += bf(a) * b;
            const relu = @max(pre, 0);
            const nv = @max(round16(pre), 0);
            const got = bf(act[p * W + j]);
            up.add(relu * relu, .{ got, got, round16(nv * nv) });
        }
        for (0..D) |i| {
            md.row(i, half * W, w[0..W]);
            var ref: f64 = 0;
            for (act[p * W ..][0..W], w[0..W]) |a, b| ref += bf(a) * b;
            const got = bf(ymoe[p * D + i]);
            down.add(ref, .{ got, got, round16(ref) });
        }
    }
    var name: [32]u8 = undefined;
    up.print(try std.fmt.bufPrint(&name, "{d}.experts.up", .{block}), "nvfp4");
    down.print(try std.fmt.bufPrint(&name, "{d}.experts.down", .{block}), "nvfp4");
}

/// Prefills `tokens` with the projection inputs dumped to `dir`, then checks the first and last block of each kind.
pub fn check(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, model: []const u8, tokens: []const u32, dir: []const u8) !u8 {
    if (e.c.format != .modelopt) return error.NotModelopt;
    var d: nemotron.Dump = .{ .gpa = gpa, .io = io, .dir = dir };
    _ = try e.prefill(tokens, &d, null);
    var ck = try core.Checkpoint.openModel(gpa, io, model);
    defer ck.close();
    const c = e.c;
    const rows = @min(tokens.len, 16);
    var r: Rig = .{ .gpa = gpa, .e = e, .x = try cuda.DeviceBuffer.alloc(e.ctx.d, 16 * 8192 * 2), .y = try cuda.DeviceBuffer.alloc(e.ctx.d, 16 * c.vocab * 2) };
    defer r.x.free();
    defer r.y.free();
    std.debug.print("{d} prompt rows, {d} checked; error / rms(ref): rms (decode, prompt, NVIDIA) | worst (decode, prompt, NVIDIA)\n", .{ tokens.len, rows });
    var seen = [3][2]?usize{ .{ null, null }, .{ null, null }, .{ null, null } };
    for (c.kinds[0..c.layers], 0..) |kind, i| {
        const s = &seen[@backingInt(kind)];
        if (s[0] == null) s[0] = i;
        s[1] = i;
    }
    var pb: [4][192]u8 = undefined;
    for (seen, 0..) |s, kind| for ([2]?usize{ s[0], if (s[1] != s[0]) s[1] else null }) |maybe| {
        const i = maybe orelse continue;
        const blk = e.w.blocks[i];
        var label: [64]u8 = undefined;
        const xb = try readBin(gpa, io, dir, i, "y");
        defer gpa.free(xb);
        const x: Rows16 = std.mem.bytesAsSlice(u16, xb);
        switch (@as(nemotron.Kind, @fromBackingInt(@intCast(kind)))) {
            .mamba => {
                try dense(&r, &ck, try std.fmt.bufPrint(&label, "{d}.in_proj", .{i}), &.{try std.fmt.bufPrint(&pb[0], "backbone.layers.{d}.mixer.in_proj", .{i})}, blk.mamba.in_proj, x, rows);
                const g = try readBin(gpa, io, dir, i, "g");
                defer gpa.free(g);
                try dense(&r, &ck, try std.fmt.bufPrint(&label, "{d}.out_proj", .{i}), &.{try std.fmt.bufPrint(&pb[0], "backbone.layers.{d}.mixer.out_proj", .{i})}, blk.mamba.out_proj, std.mem.bytesAsSlice(u16, g), rows);
            },
            .attention => {
                const pre = [3][]const u8{ try std.fmt.bufPrint(&pb[0], "backbone.layers.{d}.mixer.q_proj", .{i}), try std.fmt.bufPrint(&pb[1], "backbone.layers.{d}.mixer.k_proj", .{i}), try std.fmt.bufPrint(&pb[2], "backbone.layers.{d}.mixer.v_proj", .{i}) };
                try dense(&r, &ck, try std.fmt.bufPrint(&label, "{d}.qkv", .{i}), &pre, blk.attn.qkv, x, rows);
                const att = try readBin(gpa, io, dir, i, "att");
                defer gpa.free(att);
                try dense(&r, &ck, try std.fmt.bufPrint(&label, "{d}.o_proj", .{i}), &.{try std.fmt.bufPrint(&pb[3], "backbone.layers.{d}.mixer.o_proj", .{i})}, blk.attn.o, std.mem.bytesAsSlice(u16, att), rows);
            },
            .moe => try experts(gpa, io, &ck, dir, i, c, rows),
        }
    };
    const fin = try readBin(gpa, io, dir, c.layers, "y");
    defer gpa.free(fin);
    try dense(&r, &ck, "lm_head", &.{"lm_head"}, e.w.head, std.mem.bytesAsSlice(u16, fin), @min(rows, 4));
    return 0;
}
