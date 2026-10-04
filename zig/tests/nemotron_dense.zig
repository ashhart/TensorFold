//! Nemotron's tensor-unit dense projections at real shapes: schedule variants bit-checked against the engine's kernels, then timed.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
const MP = 16; // every window of up to 16 rows pads to one 16-row tile
const GS = 64;

/// A projection: its kernel, shape, how many a forward runs, and weight copies streamed per timing (so no copy stays cached).
const Shape = struct { key: []const u8, kernel: []const u8, n: usize, k: usize, per_step: usize, per_level: usize, copies: usize, split: bool, relu2: bool = false };
const shapes = [_]Shape{
    .{ .key = "in", .kernel = "coop_in_1_0_sk", .n = 10304, .k = 2688, .per_step = 23, .per_level = 0, .copies = 23, .split = true },
    .{ .key = "out", .kernel = "coop_out_1_0_sk", .n = 2688, .k = 4096, .per_step = 29, .per_level = 1, .copies = 45, .split = true },
    .{ .key = "down", .kernel = "coop_down_1_0", .n = 2688, .k = 3712, .per_step = 23, .per_level = 1, .copies = 46, .split = false },
    .{ .key = "up", .kernel = "up_relu2_1_0", .n = 3712, .k = 2688, .per_step = 23, .per_level = 1, .copies = 46, .split = false, .relu2 = true },
    .{ .key = "qkv", .kernel = "coop_qkv_1_0", .n = 4608, .k = 2688, .per_step = 6, .per_level = 1, .copies = 36, .split = false },
    .{ .key = "head", .kernel = "coop_head_1_0", .n = 131072, .k = 2688, .per_step = 1, .per_level = 0, .copies = 2, .split = false },
    .{ .key = "eh", .kernel = "coop_eh_1_0_sk", .n = 2688, .k = 5376, .per_step = 0, .per_level = 1, .copies = 32, .split = true },
    .{ .key = "draft", .kernel = "coop_draft_1_0", .n = 32768, .k = 2688, .per_step = 0, .per_level = 1, .copies = 6, .split = false },
};

/// Schedules of the same arithmetic (flip: split shapes whole, the others split a K slice a threadgroup).
const variants = [_][]const u8{ "base", "copy", "rows", "pair", "quad", "rows_pair", "rows_quad", "flip", "flip_pair", "m8" };
const row_counts = [_]usize{ 1, 2, 3, 8 };

fn swap(a: std.mem.Allocator, text: []const u8, old: []const u8, new: []const u8) ![]u8 {
    if (std.mem.indexOf(u8, text, old) == null) {
        std.debug.print("variant text not found: {s}\n", .{old});
        return error.VariantText;
    }
    return std.mem.replaceOwned(u8, a, text, old, new);
}

const loop_head = "  for (int g = g_begin; g < g_end; g++) {\n";
const loop_ab = "    auto a = tA.slice(g * GS, 0);\n    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(\n        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (64 * GS / 2), dextents<int32_t, 2>(GS, 64));\n";
const loop_run = "    op.run(a, b, P);\n";
const epi_body = "      const vec<bfloat, 2> sb = as_type<vec<bfloat, 2>>(sbw[g * N + n0 + ecol[i]]);\n      const float xs = !EDGE || rb + erow[i] < MP ? XS[g * MP + rb + erow[i]] : 0.0f;\n      C[i] = fma(float(sb[0]), P[i], fma(float(sb[1]), xs, C[i]));\n";
const epi_rows = "      if (rb + erow[i] < M) {\n" ++ epi_body ++ "      }\n";
const p_decl = "  auto P = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n";

/// The epilogue of group g + j from cooperative tensor P<j> (only valid rows when `rows`).
fn epilogue(a: std.mem.Allocator, j: usize, rows: bool) ![]u8 {
    const body = if (rows) epi_rows else epi_body;
    if (j == 0) return std.fmt.allocPrint(a, "    for (int i = 0; i < CAP; i++) {{\n{s}    }}\n", .{body});
    const gj = try std.fmt.allocPrint(a, "(g + {d})", .{j});
    var t = try swap(a, body, "sbw[g * N", try std.fmt.allocPrint(a, "sbw[{s} * N", .{gj}));
    t = try swap(a, t, "XS[g * MP", try std.fmt.allocPrint(a, "XS[{s} * MP", .{gj}));
    t = try swap(a, t, "P[i]", try std.fmt.allocPrint(a, "P{d}[i]", .{j}));
    return std.fmt.allocPrint(a, "    for (int i = 0; i < CAP; i++) {{\n{s}    }}\n", .{t});
}

/// `u` groups in flight: u matmuls into P, P1.., then their epilogues in group order; the remainder one at a time.
fn unrolled(a: std.mem.Allocator, src: []const u8, u: usize, rows: bool) ![]u8 {
    var body: std.ArrayList(u8) = .empty;
    try body.print(a, "  int g = g_begin;\n  for (; g + {d} <= g_end; g += {d}) {{\n", .{ u, u });
    for (0..u) |j| {
        if (j == 0) {
            try body.appendSlice(a, loop_ab);
            continue;
        }
        var ab = try swap(a, loop_ab, "auto a = tA.slice(g * GS", try std.fmt.allocPrint(a, "auto a{d} = tA.slice((g + {d}) * GS", .{ j, j }));
        ab = try swap(a, ab, "tensor_inline> b(", try std.fmt.allocPrint(a, "tensor_inline> b{d}(", .{j}));
        ab = try swap(a, ab, "KG + g)", try std.fmt.allocPrint(a, "KG + g + {d})", .{j}));
        try body.appendSlice(a, ab);
    }
    for (0..u) |j| {
        if (j == 0) try body.appendSlice(a, loop_run) else try body.print(a, "    op.run(a{d}, b{d}, P{d});\n", .{ j, j, j });
    }
    for (0..u) |j| try body.appendSlice(a, try epilogue(a, j, rows));
    try body.print(a, "  }}\n  for (; g < g_end; g++) {{\n{s}{s}{s}  }}\n", .{ loop_ab, loop_run, try epilogue(a, 0, rows) });
    const start = std.mem.indexOf(u8, src, loop_head) orelse return error.VariantText;
    const old_loop = try std.fmt.allocPrint(a, "{s}{s}{s}{s}  }}\n", .{ loop_head, loop_ab, loop_run, try epilogue(a, 0, false) });
    if (!std.mem.startsWith(u8, src[start..], old_loop)) return error.VariantText;
    var decls: std.ArrayList(u8) = .empty;
    try decls.appendSlice(a, p_decl);
    for (1..u) |j| try decls.print(a, "  auto P{d} = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n", .{j});
    const out = try std.mem.concat(a, u8, &.{ src[0..start], body.items, src[start + old_loop.len ..] });
    return swap(a, out, p_decl, decls.items);
}

const part_loop = "  for (int i = 0; i < CAP; i++) {\n    const int m = rb + erow[i], n = n0 + ecol[i];\n    if (m < M) PART[((int64_t)slice * MP + m) * N + n] = C[i];\n  }\n";

/// The one-tile-a-threadgroup kernel with each K slice its own threadgroup, writing fp32 partials (gen's _split).
fn splitSource(a: std.mem.Allocator, src: []const u8) ![]u8 {
    var t = try swap(a, src, "const ushort slice = sg >> 1;", "const ushort slice = threadgroup_position_in_grid.z;");
    const tip = std.mem.indexOf(u8, t, "  const ushort tip = ") orelse return error.VariantText;
    const tip_end = (std.mem.indexOfScalarPos(u8, t, tip, '\n') orelse return error.VariantText) + 1;
    t = try std.mem.concat(a, u8, &.{ t[0..tip], t[tip_end..] });
    t = try swap(a, t, "device bfloat16_t* Y [[buffer(5)]]", "device float* PART [[buffer(5)]]");
    const cut = std.mem.indexOf(u8, t, "  threadgroup float part[") orelse return error.VariantText;
    const last = std.mem.lastIndexOfScalar(u8, t, '}') orelse return error.VariantText;
    return std.mem.concat(a, u8, &.{ t[0..cut], part_loop, "\n", t[last..] });
}

const ta_line = "  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA((device bfloat*)X + (int64_t)rb * K, dextents<int32_t, 2>(K, M - rb));\n";
const ids_line = "  for (int i = 0; i < CAP; i++) { auto ids = P.get_multidimensional_index(i); ecol[i] = ids[0]; erow[i] = ids[1]; }\n";

/// The 16-row tile as 16/r tiles of r rows (C[j*r + i] is tile j's element i), tiles past the window's rows skipped (uniform).
fn tiles(a: std.mem.Allocator, src: []const u8, r: usize) ![]u8 {
    const n = 16 / r;
    var t = try swap(a, src, "matmul2d_descriptor(16 * TMR, 64, GS,", try std.fmt.allocPrint(a, "matmul2d_descriptor({d}, 64, GS,", .{r}));
    var more: std.ArrayList(u8) = .empty;
    try more.appendSlice(a, ta_line);
    for (1..n) |j| try more.print(a, "  tensor<device bfloat, dextents<int32_t, 2>, tensor_inline> tA{d}((device bfloat*)X + (int64_t)(rb + {d}) * K, dextents<int32_t, 2>(K, max(M - rb - {d}, 1)));\n", .{ j, j * r, j * r });
    t = try swap(a, t, ta_line, more.items);
    var decls: std.ArrayList(u8) = .empty;
    try decls.appendSlice(a, p_decl);
    for (1..n) |j| try decls.print(a, "  auto P{d} = op.template get_destination_cooperative_tensor<decltype(a0), decltype(b0), float>();\n", .{j});
    t = try swap(a, t, p_decl, decls.items);
    t = try swap(a, t, ids_line, try std.fmt.allocPrint(a, "  for (int i = 0; i < {d}; i++) {{ auto ids = P.get_multidimensional_index(i); for (int j = 0; j < {d}; j++) {{ ecol[j * {d} + i] = ids[0]; erow[j * {d} + i] = ids[1] + j * {d}; }} }}\n", .{ r, n, r, r, r }));
    var body: std.ArrayList(u8) = .empty;
    try body.appendSlice(a, loop_run);
    for (1..n) |j| try body.print(a, "    if (rb + {d} < M) {{\n      auto a{d}x = tA{d}.slice(g * GS, 0);\n      op.run(a{d}x, b, P{d});\n    }}\n", .{ j * r, j, j, j, j });
    for (0..n) |j| {
        var e: []const u8 = epi_body;
        if (j > 0) {
            const at = try std.fmt.allocPrint(a, "[{d} + i]", .{j * r});
            e = try swap(a, e, "ecol[i]", try std.fmt.allocPrint(a, "ecol{s}", .{at}));
            e = try swap(a, e, "erow[i]", try std.fmt.allocPrint(a, "erow{s}", .{at}));
            e = try swap(a, e, "C[i]", try std.fmt.allocPrint(a, "C{s}", .{at}));
            e = try swap(a, e, "P[i]", try std.fmt.allocPrint(a, "P{d}[i]", .{j}));
        }
        if (j > 0) try body.print(a, "    if (rb + {d} < M)\n", .{j * r});
        try body.print(a, "    for (int i = 0; i < {d}; i++) {{\n{s}    }}\n", .{ r, e });
    }
    return swap(a, t, try std.fmt.allocPrint(a, "{s}{s}", .{ loop_run, try epilogue(a, 0, false) }), body.items);
}

fn variantSource(a: std.mem.Allocator, src: []const u8, v: []const u8) ![]const u8 {
    const eq = std.mem.eql;
    if (eq(u8, v, "copy") or eq(u8, v, "flip")) return src;
    if (eq(u8, v, "flip_pair")) return unrolled(a, src, 2, false);
    if (eq(u8, v, "rows")) return swap(a, src, epi_body, epi_rows);
    if (eq(u8, v, "m8")) return tiles(a, src, 8); // MPP takes M in multiples of 8
    if (eq(u8, v, "pair")) return unrolled(a, src, 2, false);
    if (eq(u8, v, "quad")) return unrolled(a, src, 4, false);
    if (eq(u8, v, "rows_pair")) return unrolled(a, src, 2, true);
    if (eq(u8, v, "rows_quad")) return unrolled(a, src, 4, true);
    return error.UnknownVariant;
}

fn kernelOf(comptime key: []const u8) sources.nemotron.Kernel {
    inline for (sources.nemotron.all) |k| if (comptime std.mem.eql(u8, k.key, key)) return k;
    @compileError("no kernel " ++ key);
}

fn constOf(src: []const u8, name: []const u8) usize {
    var buf: [64]u8 = undefined;
    const pat = std.fmt.bufPrint(&buf, "constexpr int {s} = ", .{name}) catch unreachable;
    const at = (std.mem.indexOf(u8, src, pat) orelse unreachable) + pat.len;
    const end = std.mem.indexOfScalarPos(u8, src, at, ';') orelse unreachable;
    return std.fmt.parseInt(usize, src[at..end], 10) catch unreachable;
}

/// One shape's buffers: weight copies, inputs for 8 rows and for each row alone, outputs.
const Bufs = struct {
    w: []mtl.Buffer,
    sbt: []mtl.Buffer,
    x: mtl.Buffer, // [MP, K] bf16: rows 0..7 random
    xs: mtl.Buffer, // [K/GS, MP] f32: each row's group sums
    x1: [8]mtl.Buffer, // row r alone at row 0
    xs1: [8]mtl.Buffer,
    y: mtl.Buffer, // [MP, N] bf16
    xso: mtl.Buffer, // [N/64, MP] f32 (relu2's group sums)
    part: mtl.Buffer, // [SK, MP, N] f32

    fn init(a: std.mem.Allocator, d: mtl.Device, s: Shape, sk: usize, rng: std.Random) !Bufs {
        var b: Bufs = undefined;
        b.w = try a.alloc(mtl.Buffer, s.copies);
        b.sbt = try a.alloc(mtl.Buffer, s.copies);
        const kg = s.k / GS;
        for (b.w, b.sbt) |*w, *sb| {
            w.* = try d.buffer(s.n * s.k / 2, opts);
            rng.bytes(w.slice(u8, s.n * s.k / 2));
            sb.* = try d.buffer(kg * s.n * 4, opts);
            const pairs = sb.slice(u16, kg * s.n * 2);
            for (0..kg * s.n) |i| {
                pairs[2 * i] = bf16(0.002 + 0.018 * rng.float(f32));
                pairs[2 * i + 1] = bf16(-0.15 * rng.float(f32));
            }
        }
        b.x = try d.buffer(MP * s.k * 2, opts);
        const xv = b.x.slice(u16, MP * s.k);
        @memset(xv, 0);
        for (xv[0 .. 8 * s.k]) |*v| v.* = bf16(rng.floatNorm(f32));
        b.xs = try d.buffer(kg * MP * 4, opts);
        sums(xv, b.xs.slice(f32, kg * MP), s.k, 0, 8, 0);
        for (0..8) |r| {
            b.x1[r] = try d.buffer(MP * s.k * 2, opts);
            const x1 = b.x1[r].slice(u16, MP * s.k);
            @memset(x1, 0);
            @memcpy(x1[0..s.k], xv[r * s.k .. (r + 1) * s.k]);
            b.xs1[r] = try d.buffer(kg * MP * 4, opts);
            sums(xv, b.xs1[r].slice(f32, kg * MP), s.k, r, r + 1, 0);
        }
        b.y = try d.buffer(MP * s.n * 2, opts);
        b.xso = try d.buffer((s.n / 64) * MP * 4, opts);
        b.part = try d.buffer(sk * MP * s.n * 4, opts);
        return b;
    }
};

fn bf16(f: f32) u16 {
    return @truncate(@as(u32, @bitCast(f)) >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// XS[g, at + (r - first)] = row r's sum over group g, for rows first..last.
fn sums(x: []const u16, xs: []f32, k: usize, first: usize, last: usize, at: usize) void {
    @memset(xs, 0);
    for (first..last) |r| for (0..k / GS) |g| {
        var t: f32 = 0;
        for (x[r * k + g * GS .. r * k + (g + 1) * GS]) |v| t += f32of(v);
        xs[g * MP + at + r - first] = t;
    };
}

const Pipes = struct { main: mtl.Pipeline, combine: mtl.Pipeline };

/// Encode one projection of `rows` rows from copy `c` (inputs x/xs), as the engine does.
fn encode(enc: mtl.ComputeEncoder, p: Pipes, s: Shape, split: bool, sk: usize, b: *const Bufs, c: usize, x: mtl.Buffer, xs: mtl.Buffer, rows: usize) void {
    const md = [8]i32{ @intCast(rows), MP, 0, 0, 0, 0, 0, 0 };
    enc.setPipeline(p.main);
    enc.setBuffer(x, 0, 0);
    enc.setBuffer(xs, 0, 1);
    enc.setBuffer(b.w[c], 0, 2);
    enc.setBuffer(b.sbt[c], 0, 3);
    enc.setBytes(std.mem.asBytes(&md), 4);
    if (split) {
        enc.setBuffer(b.part, 0, 5);
        enc.dispatchThreads(mtl.Size.of(s.n / 64 * 64, 1, sk), mtl.Size.of(64, 1, 1));
        const dims = [8]i32{ @intCast(s.n), @intCast(rows), MP, @intCast(sk), 0, 0, 0, 0 };
        enc.setPipeline(p.combine);
        enc.setBuffer(b.part, 0, 0);
        enc.setBytes(std.mem.asBytes(&dims), 1);
        enc.setBuffer(b.y, 0, 2);
        enc.dispatchThreads(mtl.Size.of(s.n, rows, 1), mtl.Size.of(@min(s.n, 256), 1, 1));
    } else {
        enc.setBuffer(b.y, 0, 5);
        if (s.relu2) enc.setBuffer(b.xso, 0, 6);
        enc.dispatchThreads(mtl.Size.of(s.n / 64 * 64 * sk, 1, 1), mtl.Size.of(64 * sk, 1, 1));
    }
}

/// Outputs of rows 0..rows-1 (copy 0): y rows then relu2's group sums.
fn run(q: mtl.Queue, p: Pipes, s: Shape, split: bool, sk: usize, b: *const Bufs, x: mtl.Buffer, xs: mtl.Buffer, rows: usize, out: []u32) usize {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    @memset(b.y.slice(u8, MP * s.n * 2), 0xff);
    @memset(b.xso.slice(u8, (s.n / 64) * MP * 4), 0xff);
    const cb = q.commandBuffer();
    const enc = cb.compute(.serial);
    encode(enc, p, s, split, sk, b, 0, x, xs, rows);
    enc.end();
    cb.commit();
    cb.wait();
    var n: usize = 0;
    for (b.y.slice(u16, rows * s.n)) |v| {
        out[n] = v;
        n += 1;
    }
    if (s.relu2) for (0..s.n / 64) |t| for (0..rows) |r| {
        out[n] = @bitCast(b.xso.slice(f32, (s.n / 64) * MP)[t * MP + r]);
        n += 1;
    };
    return n;
}

/// Median GPU microseconds per projection over every copy, `reps` command buffers after warm-up.
fn time(q: mtl.Queue, p: Pipes, s: Shape, split: bool, sk: usize, b: *const Bufs, rows: usize, reps: usize) f64 {
    var t: [32]f64 = undefined;
    for (0..reps + 3) |i| {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = q.commandBuffer();
        const enc = cb.compute(.serial);
        for (0..s.copies) |c| encode(enc, p, s, split, sk, b, c, b.x, b.xs, rows);
        enc.end();
        cb.commit();
        cb.wait();
        if (i >= 3) t[i - 3] = cb.gpuSeconds() * 1e6 / @as(f64, @floatFromInt(s.copies));
    }
    std.mem.sort(f64, t[0..reps], {}, std.sort.asc(f64));
    return t[reps / 2];
}

/// Row r of the 8-row outputs against row r computed alone (rows 0..7), and rows 0..m-1 at m = 1, 2, 3 against the same.
fn rowExact(q: mtl.Queue, p: Pipes, s: Shape, split: bool, sk: usize, b: *const Bufs, full: []const u32, scratch: []u32) bool {
    for (0..8) |r| {
        _ = run(q, p, s, split, sk, b, b.x1[r], b.xs1[r], 1, scratch);
        if (!std.mem.eql(u32, scratch[0..s.n], full[r * s.n .. (r + 1) * s.n])) return false;
    }
    for ([_]usize{ 1, 2, 3 }) |m| {
        _ = run(q, p, s, split, sk, b, b.x, b.xs, m, scratch);
        if (!std.mem.eql(u32, scratch[0 .. m * s.n], full[0 .. m * s.n])) return false;
    }
    return true;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const args = try init.minimal.args.toSlice(a);
    const reps: usize = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 9;
    if (reps < 1 or reps > 32) return error.BadReps;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    const q = try device.queue();
    std.debug.print("device {s}; compiling the engine's kernels\n", .{device.name()});
    var engine = try tf.nemotron.kernels.load(gpa, device);
    defer engine.deinit();
    const combine = engine.get("tf_coop_combine");
    var prng: std.Random.DefaultPrng = .init(20261004);
    const rng = prng.random();
    var step: [variants.len][row_counts.len]f64 = @splat(@splat(0));
    var level: [variants.len][row_counts.len]f64 = @splat(@splat(0));
    var all_exact = true;
    var best_step: [row_counts.len]f64 = @splat(0);
    var best_level: [row_counts.len]f64 = @splat(0);
    inline for (shapes) |s| {
        const kern = comptime kernelOf(s.kernel);
        const sk = constOf(kern.source, "SK");
        const b = try Bufs.init(a, device, s, sk, rng);
        const words = 8 * s.n + (s.n / 64) * 8;
        const ref = try a.alloc(u32, words);
        const got = try a.alloc(u32, words);
        const scratch = try a.alloc(u32, words);
        const base: Pipes = .{ .main = engine.get(s.kernel), .combine = combine };
        var best: [row_counts.len]f64 = @splat(0);
        var base_us: [row_counts.len]f64 = @splat(0);
        defer for (best, 0..) |t, mi| {
            best_step[mi] += t * @as(f64, @floatFromInt(s.per_step));
            best_level[mi] += t * @as(f64, @floatFromInt(s.per_level));
        };
        const nref = run(q, base, s, s.split, sk, &b, b.x, b.xs, 8, ref);
        if (comptime std.mem.eql(u8, s.key, "in")) _ = time(q, base, s, s.split, sk, &b, 1, 32); // clocks up before the first timing
        std.debug.print("\n{s} (N {d}, K {d}, SK {d}, {d} copies, {d:.1} MB a copy): us a projection at rows 1 2 3 8, GB/s at 1 row\n", .{ s.key, s.n, s.k, sk, s.copies, @as(f64, @floatFromInt(s.n * s.k / 2 + (s.k / GS) * s.n * 4)) / 1e6 });
        for (variants, 0..) |v, vi| {
            var p = base;
            var lib: ?mtl.Library = null;
            const flip = std.mem.startsWith(u8, v, "flip");
            if (flip and (s.relu2 or sk == 1)) {
                for (0..row_counts.len) |mi| {
                    step[vi][mi] += base_us[mi] * @as(f64, @floatFromInt(s.per_step));
                    level[vi][mi] += base_us[mi] * @as(f64, @floatFromInt(s.per_level));
                }
                continue;
            }
            const split = s.split != flip;
            const whole = if (comptime s.split) comptime kernelOf("coop_" ++ s.key ++ "_1_0") else kern;
            if (flip and s.split and vi == 7) p.main = engine.get("coop_" ++ s.key ++ "_1_0");
            if (vi > 0 and !(flip and s.split and vi == 7)) {
                const src0 = if (flip) whole.source else kern.source;
                const src1 = variantSource(a, src0, v) catch |e| {
                    std.debug.print("  {s:<10} source: {s}\n", .{ v, @errorName(e) });
                    continue;
                };
                const text = if (flip and !s.split) splitSource(a, src1) catch |e| {
                    std.debug.print("  {s:<10} split source: {s}\n", .{ v, @errorName(e) });
                    continue;
                } else src1;
                lib = mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx()) catch {
                    std.debug.print("  {s:<10} compile failed\n", .{v});
                    continue;
                };
                p.main = try mtl.Pipeline.init(device, lib.?, if (flip) whole.function else kern.function, false);
            }
            defer if (lib) |l| l.deinit();
            const n = run(q, p, s, split, sk, &b, b.x, b.xs, 8, got);
            const same = n == nref and std.mem.eql(u32, got[0..n], ref[0..nref]);
            const rows_ok = rowExact(q, p, s, split, sk, &b, got[0..n], scratch);
            all_exact = all_exact and same and rows_ok;
            var us: [row_counts.len]f64 = undefined;
            for (row_counts, 0..) |m, mi| {
                us[mi] = time(q, p, s, split, sk, &b, m, reps);
                step[vi][mi] += us[mi] * @as(f64, @floatFromInt(s.per_step));
                level[vi][mi] += us[mi] * @as(f64, @floatFromInt(s.per_level));
                if (same and rows_ok and (best[mi] == 0 or us[mi] < best[mi])) best[mi] = us[mi];
            }
            if (vi == 0) base_us = us;
            const gbs = @as(f64, @floatFromInt(s.n * s.k / 2 + (s.k / GS) * s.n * 4)) / (us[0] * 1e3);
            std.debug.print("  {s:<10} bits {s} rows {s} | {d:8.2} {d:8.2} {d:8.2} {d:8.2} | {d:5.0}\n", .{ v, if (same) "==" else "DIFFER", if (rows_ok) "exact" else "VARIANT", us[0], us[1], us[2], us[3], gbs });
        }
    }
    std.debug.print("\ndense per forward (23 in, 29 out, 23 down, 23 up, 6 qkv, 1 head), ms at rows 1 2 3 8; head level (eh, qkv, out, up, down, draft); a shape without the variant counts at base:\n", .{});
    for (variants, 0..) |v, vi| std.debug.print("  {s:<10} step {d:6.3} {d:6.3} {d:6.3} {d:6.3} | level {d:6.3} {d:6.3} {d:6.3} {d:6.3}\n", .{ v, step[vi][0] / 1e3, step[vi][1] / 1e3, step[vi][2] / 1e3, step[vi][3] / 1e3, level[vi][0] / 1e3, level[vi][1] / 1e3, level[vi][2] / 1e3, level[vi][3] / 1e3 });
    std.debug.print("  {s:<10} step {d:6.3} {d:6.3} {d:6.3} {d:6.3} | level {d:6.3} {d:6.3} {d:6.3} {d:6.3}\n", .{ "best each", best_step[0] / 1e3, best_step[1] / 1e3, best_step[2] / 1e3, best_step[3] / 1e3, best_level[0] / 1e3, best_level[1] / 1e3, best_level[2] / 1e3, best_level[3] / 1e3 });
    std.debug.print("{s}\n", .{if (all_exact) "every variant: bits == the engine's kernel, every row exact alone and in windows" else "SOME VARIANT DIFFERS (see above)"});
    if (!all_exact) std.process.exit(1);
}
