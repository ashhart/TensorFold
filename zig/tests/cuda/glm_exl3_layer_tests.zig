//! One real GLM layer's EXL3 experts through the family loader, decode and prompt, against float64 unpacks.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const qr = @import("quant_ref_tests.zig");
const Gpu = check.Gpu;

const fmt = core.exl3_format;
const h = qr.helpers;
const Rig = qr.RigT;
const prefix = "model.language_model.";
const slots = glm.exl3.top_k + 1;
const pool = 16; // the experts the rows pick from: the float64 matrices are decoded for these alone

/// cuda_weights' loader as cuda_exl3.load sees it: tensors by name, the host bytes gathered, each upload kept.
const Loader = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    ck: *core.Checkpoint,
    host: std.ArrayList(u8) = .empty,
    name_buf: [192]u8 = undefined,
    last: []const u8 = "",
    bufs: std.ArrayList(cuda.DeviceBuffer) = .empty,
    bytes: usize = 0,

    pub fn get(L: *Loader, comptime f: []const u8, args: anytype) !core.checkpoint.Tensor {
        L.last = try std.fmt.bufPrint(&L.name_buf, prefix ++ f, args);
        return L.ck.get(L.last);
    }
    pub fn put(L: *Loader, comptime f: []const u8, args: anytype) !u64 {
        _ = f;
        _ = args;
        const b = try cuda.DeviceBuffer.fromHost(L.d, L.host.items);
        errdefer {
            var x = b;
            x.free();
        }
        try L.bufs.append(L.gpa, b);
        L.bytes += L.host.items.len;
        L.host.clearRetainingCapacity();
        return b.ptr;
    }
    fn deinit(L: *Loader) void {
        for (L.bufs.items) |*b| b.free();
        L.bufs.deinit(L.gpa);
        L.host.deinit(L.gpa);
    }
};

/// Rotate each 128-value block along `axis` of the row-major [rows, cols] matrix: fmt.rotate's transform, in n log n.
fn rotate(x: []f64, rows: usize, cols: usize, axis: usize) void {
    const hd = fmt.hadamard_dim;
    const inv = 1.0 / std.math.sqrt(@as(f64, hd));
    if (axis == 0) { // whole rows at a time: rows r and r + half of each block's butterflies
        var b: usize = 0;
        while (b < rows) : (b += hd) {
            var half: usize = 1;
            while (half < hd) : (half *= 2) {
                var i: usize = 0;
                while (i < hd) : (i += 2 * half) for (i..i + half) |j| {
                    const p = x[(b + j) * cols ..][0..cols];
                    const q = x[(b + j + half) * cols ..][0..cols];
                    for (p, q) |*u, *v| {
                        const s = u.* + v.*;
                        v.* = u.* - v.*;
                        u.* = s;
                    }
                };
            }
            for (x[b * cols ..][0 .. hd * cols]) |*v| v.* *= inv;
        }
    } else for (0..rows) |r| {
        var b: usize = 0;
        while (b < cols) : (b += hd) {
            const blk = x[r * cols + b ..][0..hd];
            var half: usize = 1;
            while (half < hd) : (half *= 2) {
                var i: usize = 0;
                while (i < hd) : (i += 2 * half) for (i..i + half) |j| {
                    const s = blk[j] + blk[j + half];
                    blk[j + half] = blk[j] - blk[j + half];
                    blk[j] = s;
                };
            }
            for (blk) |*v| v.* *= inv;
        }
    }
}

fn halfs(a: std.mem.Allocator, t: core.checkpoint.Tensor) ![]f16 {
    const out = try a.alloc(f16, t.bytes.len / 2);
    @memcpy(std.mem.sliceAsBytes(out), t.bytes);
    return out;
}

/// Expert `e`'s projection as the rank holds it, float64: gate/up cols [lo,hi), down rows; unpack, cut, scale.
fn matrix(a: std.mem.Allocator, ck: *core.Checkpoint, layer: u32, e: usize, comptime proj: []const u8, k: usize, n: usize, lo: usize, hi: usize, by_cols: bool) ![]f64 {
    var buf: [160]u8 = undefined;
    const t = try ck.get(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}." ++ proj ++ ".trellis", .{ layer, e }));
    const words = try a.alloc(i16, t.bytes.len / 2);
    defer a.free(words);
    @memcpy(std.mem.sliceAsBytes(words), t.bytes);
    const wq = try a.alloc(f16, k * n);
    defer a.free(wq);
    fmt.unpack(.{ .halves = @intCast(t.dim(2) / 8) }, .mcg, words, k / 16, n / 16, wq);
    const suh = try halfs(a, try ck.get(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}." ++ proj ++ ".suh", .{ layer, e })));
    defer a.free(suh);
    const svh = try halfs(a, try ck.get(try std.fmt.bufPrint(&buf, prefix ++ "layers.{d}.mlp.experts.{d}." ++ proj ++ ".svh", .{ layer, e })));
    defer a.free(svh);
    const rk = if (by_cols) k else hi - lo;
    const rn = if (by_cols) hi - lo else n;
    const w = try a.alloc(f64, rk * rn);
    for (0..rk) |r| {
        const src = if (by_cols) wq[r * n + lo ..][0..rn] else wq[(lo + r) * n ..][0..rn];
        for (w[r * rn ..][0..rn], src) |*d, s| d.* = @floatCast(s);
    }
    rotate(w, rk, rn, 0);
    for (0..rk) |r| {
        const s: f64 = @floatCast(suh[if (by_cols) r else lo + r]);
        for (w[r * rn ..][0..rn]) |*v| v.* *= s;
    }
    rotate(w, rk, rn, 1);
    for (0..rk) |r| for (w[r * rn ..][0..rn], 0..) |*v, c| {
        v.* *= @as(f64, @floatCast(svh[if (by_cols) lo + c else c]));
    };
    return w;
}

/// The kernels' SwiGLU: gate and up rounded to bf16 and limited, silu to bf16, the product to bf16.
fn act(g: f64, u: f64, limit: f32) f64 {
    const gg = @min(h.bfOf(h.bf16Of(@floatCast(g))), limit);
    const uu = @max(@min(h.bfOf(h.bf16Of(@floatCast(u))), limit), -limit);
    return h.bfOf(h.bf16Of(@floatCast(h.bfOf(h.bf16Of(@floatCast(gg / (1 + @exp(-gg))))) * uu)));
}

const Case = struct { rows: usize, prompt: bool };

/// `glm-exl3-layer <ckpt> <layer> <rank> <world>`: the layer's experts at the rank's share, all within tolerance.
pub fn run(gpu: Gpu, dir: []const u8, layer_s: []const u8, rank_s: []const u8, world_s: []const u8) !void {
    try checkRotation(gpu.gpa);
    const layer = try std.fmt.parseInt(u32, layer_s, 10);
    const rank = try std.fmt.parseInt(usize, rank_s, 10);
    const world = try std.fmt.parseInt(usize, world_s, 10);
    var arena: std.heap.ArenaAllocator = .init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 24));
    var why: core.modelopt.Why = .{};
    if (!glm.exl3.isExl3(gpu.gpa, text)) return error.NotExl3;
    glm.exl3.admit(gpu.gpa, text, &why) catch |err| {
        std.debug.print("FAIL admission: {s}\n", .{why.text()});
        return err;
    };
    const c = try glm.config.parse(gpu.gpa, text);
    const cut = (try glm.shard.Share.of(&c, rank, world)).moe;
    const D: usize = c.hidden;
    const W = cut.len();
    const E: usize = c.experts;
    const limit = c.swiglu_limit;
    var ck = try core.Checkpoint.openModelPrefix(gpu.gpa, gpu.io, dir, prefix);
    defer ck.close();

    var stream = try cuda.Stream.init(gpu.d, false);
    defer stream.deinit();
    var prng = std.Random.DefaultPrng.init(layer * 131 + rank);
    var r: Rig = .{ .gpu = gpu, .a = a, .s = stream, .rng = prng.random() };
    defer {
        for (r.bufs.items) |*b| b.free();
        r.bufs.deinit(gpu.gpa);
    }
    var L: Loader = .{ .gpa = gpu.gpa, .d = gpu.d, .ck = &ck };
    defer L.deinit();
    const w = try glm.exl3.load(&L, layer, E, D, c.moe_inter, cut);
    try stream.synchronize();
    std.debug.print("glm-exl3-layer: layer {d}, rank {d} of {d} (columns {d}..{d}): {d:.2} GiB uploaded; widths gate/up {d}, down {d} half-bits (0: mixed); one rotation {}\n", .{ layer, rank, world, cut.lo, cut.hi, @as(f64, @floatFromInt(L.bytes)) / (1 << 30), w.uniform_gu, w.uniform_d, w.shx });
    var k2s: [3][]i32 = undefined;
    for (&k2s, w.tables()) |*t, ptr| {
        t.* = try a.alloc(i32, E);
        const b: cuda.DeviceBuffer = .{ .d = gpu.d, .ptr = ptr, .len = E * 4 };
        try b.download(0, std.mem.sliceAsBytes(t.*));
    }
    std.debug.print("glm-exl3-layer: expert 0's half-bits gate {d}, up {d}, down {d}\n", .{ k2s[0][0], k2s[1][0], k2s[2][0] });
    const l = glm.exl3.layer(w, E, D, W, limit);
    std.debug.print("glm-exl3-layer: prompt kernels take the layer: {}\n", .{cuda.exl3_experts.prompt.eligible(l)});

    var pm = try cuda.Module.load(gpu.d, cuda.kernels.experts);
    defer pm.unload();
    var k = try glm.exl3.Kernels.load(gpu.d);
    defer k.unload();
    var ek = try glm.experts.Kernels.load(gpu.d, pm, @intCast(try gpu.ctx.attribute(.multiprocessor_count)));
    defer ek.unload();

    // the pool: the first and last experts and 14 more, distinct
    var ids: [pool]usize = undefined;
    ids[0] = 0;
    ids[1] = E - 1;
    var n_ids: usize = 2;
    while (n_ids < pool) {
        const e = r.rng.uintLessThan(usize, E);
        if (std.mem.indexOfScalar(usize, ids[0..n_ids], e) == null) {
            ids[n_ids] = e;
            n_ids += 1;
        }
    }
    const cases = [_]Case{ .{ .rows = 1, .prompt = false }, .{ .rows = 3, .prompt = false }, .{ .rows = 8, .prompt = false }, .{ .rows = 24, .prompt = true } };
    var xs: [cases.len][]u16 = undefined;
    var pks: [cases.len][]i32 = undefined;
    var outs: [cases.len][]u8 = undefined;
    for (cases, &xs, &pks, &outs) |cs, *x, *pk, *out| {
        x.* = try r.bfs(cs.rows * D, 1.0);
        pk.* = try a.alloc(i32, cs.rows * slots);
        for (0..cs.rows) |row| {
            var seen: [pool]bool = @splat(false);
            for (0..slots - 1) |j| {
                var p = r.rng.uintLessThan(usize, pool);
                while (seen[p]) p = r.rng.uintLessThan(usize, pool);
                seen[p] = true;
                pk.*[row * slots + j] = @intCast(ids[p]);
            }
            pk.*[row * slots + slots - 1] = @intCast(E); // the shared expert's slot, filled apart
        }
        const P = cs.rows * glm.exl3.top_k;
        const tile: usize = if (cs.prompt) glm.experts.staged_tile else glm.experts.tile;
        const wide = P > cuda.grouped.small;
        const plan: cuda.grouped.Plan = .{
            .members = try r.zeros(P * 4),
            .items = try r.zeros(cuda.grouped.maxItems(P, E, tile) * 12),
            .counts = try r.zeros(8),
            .rank = try r.zeros((if (wide) P else 1) * 4),
            .hist = try r.zeros((if (wide) (P + 1023) / 1024 * E else 1) * 4),
        };
        const sz = glm.exl3.scratchFor(cs.rows, D, W, cs.prompt, w);
        const sc: glm.exl3.Scratch = .{ .picks = try r.zeros(sz[0]), .xg = try r.zeros(sz[1]), .xu = try r.zeros(sz[2]), .z = try r.zeros(sz[3]) };
        const ey = try r.zeros(cs.rows * slots * D * 4);
        try glm.exl3.run(&k, &ek, stream, w, sc, plan, try r.dev(std.mem.sliceAsBytes(x.*)), D, try r.dev(std.mem.sliceAsBytes(pk.*)), slots, cs.rows, E, D, W, limit, cs.prompt, ey, 0);
        out.* = try r.back(ey, cs.rows * slots * D * 4);
    }

    // float64, one pool expert at a time: its three matrices at the rank's share, then every pick of it
    var worst: [cases.len]f64 = @splat(0);
    const g = try a.alloc(f64, W);
    const u = try a.alloc(f64, W);
    const act_v = try a.alloc(f64, W);
    const swapped = try a.alloc(f64, W); // the planted control: gate and up swapped must fail the same tolerance
    var control: f64 = 0;
    for (ids, 0..) |e, done| {
        const wg = try matrix(gpu.gpa, &ck, layer, e, "gate_proj", D, c.moe_inter, cut.lo, cut.hi, true);
        defer gpu.gpa.free(wg);
        const wu = try matrix(gpu.gpa, &ck, layer, e, "up_proj", D, c.moe_inter, cut.lo, cut.hi, true);
        defer gpu.gpa.free(wu);
        const wd = try matrix(gpu.gpa, &ck, layer, e, "down_proj", c.moe_inter, D, cut.lo, cut.hi, false);
        defer gpu.gpa.free(wd);
        ck.release();
        for (cases, xs, pks, outs, &worst) |cs, x, pk, out, *most| for (0..cs.rows) |row| for (0..slots - 1) |j| {
            if (pk[row * slots + j] != @as(i32, @intCast(e))) continue;
            @memset(g, 0);
            @memset(u, 0);
            for (0..D) |kk| {
                const xv = h.bfOf(x[row * D + kk]);
                for (g, u, wg[kk * W ..][0..W], wu[kk * W ..][0..W]) |*gv, *uv, a_, b_| {
                    gv.* += xv * a_;
                    uv.* += xv * b_;
                }
            }
            for (act_v, swapped, g, u) |*v, *sv, gv, uv| {
                v.* = act(gv, uv, limit);
                sv.* = act(uv, gv, limit);
            }
            for (0..D) |n| {
                var y: f64 = 0;
                var mag: f64 = 0;
                var y_swapped: f64 = 0;
                for (0..W) |kk| {
                    const p = act_v[kk] * wd[kk * D + n];
                    y += p;
                    mag += @abs(p);
                    y_swapped += swapped[kk] * wd[kk * D + n];
                }
                const tol = mag * (1.0 / 64.0) + 1e-6; // fp16 rotated inputs, bf16 activations, fp16 down inputs
                const at = (row * slots + j) * D + n;
                const got: f64 = @as(f32, @bitCast(std.mem.readInt(u32, out[at * 4 ..][0..4], .little)));
                const q = @abs(got - y) / tol;
                control = @max(control, @abs(got - y_swapped) / tol);
                if (!(q <= most.*)) {
                    most.* = q;
                    if (!(q <= 1)) std.debug.print("  {s} {d} rows: row {d} slot {d} (expert {d}) column {d}: {e} against {e} (tolerance {e})\n", .{ if (cs.prompt) "prompt" else "decode", cs.rows, row, j, e, n, got, y, tol });
                }
            }
        };
        if (done % 4 == 3) std.debug.print("glm-exl3-layer: {d} of {d} experts checked\n", .{ done + 1, pool });
    }
    for (cases, outs) |cs, out| for (0..cs.rows) |row| { // the shared slot keeps the fill's bytes
        const bits = std.mem.readInt(u32, out[((row * slots + slots - 1) * D) * 4 ..][0..4], .little);
        try check.expect(bits == 0xffffffff, "glm-exl3-layer: the shared slot left alone", .{});
    };
    std.debug.print("glm-exl3-layer: the planted control (gate and up swapped in the reference) reaches {d:.1} of its tolerance\n", .{control});
    try check.expect(control > 1, "glm-exl3-layer: the planted control fails its tolerance", .{});
    var all: f64 = 0;
    for (cases, worst) |cs, most| {
        std.debug.print("RESULT glm-exl3-layer {d} {s} ({d} rows): largest difference {d:.3} of its tolerance\n", .{ layer, if (cs.prompt) "prompt" else "decode", cs.rows, most });
        all = @max(all, most);
    }
    try check.expect(all <= 1, "glm-exl3-layer: layer {d} within its float64 tolerance", .{layer});
    check.pass("glm-exl3-layer: layer {d} rank {d} of {d}, decode and prompt slot rows within {d:.3} of their float64 tolerance", .{ layer, rank, world, all });
}

/// The n log n rotation against core's on a random matrix, both axes.
fn checkRotation(gpa: std.mem.Allocator) !void {
    var prng = std.Random.DefaultPrng.init(1);
    const x = try gpa.alloc(f64, 256 * 384);
    defer gpa.free(x);
    const y = try gpa.alloc(f64, x.len);
    defer gpa.free(y);
    const z = try gpa.alloc(f64, x.len);
    defer gpa.free(z);
    for (x) |*v| v.* = prng.random().float(f64) * 2 - 1;
    for ([_]usize{ 0, 1 }) |axis| {
        @memcpy(y, x);
        fmt.rotate(y, 256, 384, axis);
        @memcpy(z, x);
        rotate(z, 256, 384, axis);
        for (y, z) |p, q| try check.expect(@abs(p - q) <= 1e-12, "glm-exl3-layer: the n log n rotation is core's (axis {d})", .{axis});
    }
}
