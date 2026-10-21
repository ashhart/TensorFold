//! A Qwen-VL image tower on Metal (the Qwen3.5 tower at any output width): an image's patches to its merged rows.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../frags.zig");
const Config = @import("config.zig").Config;
const grid = @import("grid.zig");
const wts = @import("weights.zig");

pub const Ref = wts.Ref;

/// The axial rotary's base (the Qwen3.5 vision config's default).
const theta: f32 = 10000;

const Kernels = struct {
    lib: mtl.Library,
    gemm: [3][2][2]mtl.Pipeline, // [bf16 nt, f32 nt, f32 nn][M % 64 == 0][N % 128 == 0]
    embed: mtl.Pipeline,
    norm: mtl.Pipeline,
    qkv: mtl.Pipeline,
    softmax: mtl.Pipeline,
    heads: mtl.Pipeline,
    gelu: mtl.Pipeline,
    residual: mtl.Pipeline,
    bias: mtl.Pipeline,

    const names = .{ .{ "embed", "qwv_embed" }, .{ "norm", "qwv_layer_norm" }, .{ "qkv", "qwv_qkv" }, .{ "softmax", "qwv_softmax" }, .{ "heads", "qwv_heads" }, .{ "gelu", "qwv_bias_gelu" }, .{ "residual", "qwv_residual" }, .{ "bias", "qwv_bias" } };

    fn load(gpa: std.mem.Allocator, device: mtl.Device) !Kernels {
        const src = try frags.source(device, gpa, sources.core_qwen_vision);
        defer gpa.free(src);
        var k: Kernels = undefined;
        k.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer k.lib.deinit();
        var made: usize = 0; // a failed pipeline releases the ones made before it, then the library
        errdefer k.release(made);
        for ([_][]const u8{ "bf16_nt", "f32_nt", "f32_nn" }, 0..) |kind, ki| for (0..2) |m| for (0..2) |n| {
            var name: [32]u8 = undefined;
            k.gemm[ki][m][n] = try mtl.Pipeline.init(device, k.lib, try std.fmt.bufPrint(&name, "qwv_gemm_{s}_{d}{d}", .{ kind, m, n }), false);
            made += 1;
        };
        inline for (names) |kv| {
            @field(k, kv[0]) = try mtl.Pipeline.init(device, k.lib, kv[1], false);
            made += 1;
        }
        return k;
    }

    /// The first `n` pipelines in the order `load` makes them: the gemms, then the glue.
    fn release(k: *Kernels, n: usize) void {
        var i: usize = 0;
        for (&k.gemm) |*a| for (a) |*b| for (b) |p| {
            if (i == n) return;
            p.deinit();
            i += 1;
        };
        inline for (names) |kv| {
            if (i == n) return;
            @field(k, kv[0]).deinit();
            i += 1;
        }
    }

    fn deinit(k: *Kernels) void {
        k.release(3 * 2 * 2 + names.len);
        k.lib.deinit();
    }
};

/// One buffer's parts for `P` patches, each 256-aligned, and the query rows one attention pass scores.
const Scratch = struct {
    parts: [15]usize,
    chunk: u32,

    fn of(c: Config, P: usize) Scratch {
        const h: usize = c.hidden;
        const hd: usize = c.headDim();
        // the scores take at most 256 MiB: every head's `chunk` query rows by P keys in fp32
        const chunk: u32 = @intCast(std.math.clamp((256 << 20) / (c.heads * P * 4) / 64 * 64, 64, 4096));
        return .{ .chunk = chunk, .parts = .{ P * c.patchWidth() * 2, P * 4 * 4, P * 4 * 4, P * hd * 4, P * hd * 4, P * h * 4, P * h * 2, P * h * 2, P * 3 * h * 2, P * h * 4, P * h * 4, P * h * 4, @as(usize, c.heads) * chunk * P * 4, @as(usize, c.heads) * chunk * hd * 4, P * c.intermediate * 2 } };
    }

    fn bytes(s: Scratch) usize {
        var total: usize = 0;
        for (s.parts) |n| total += std.mem.alignForward(usize, n, 256);
        return total;
    }
};

/// What a tower for `max_tokens` image tokens a request holds: its weights and its scratch.
pub const Memory = struct { source: usize, weights: usize, scratch: usize };

pub fn memoryPlan(gpa: std.mem.Allocator, c: Config, max_tokens: u32) !Memory {
    const weights = try wts.plan(gpa, c);
    return .{ .source = weights.source, .weights = weights.device, .scratch = Scratch.of(c, @as(usize, max_tokens) * c.merge * c.merge).bytes() };
}

pub fn planBytes(gpa: std.mem.Allocator, c: Config, max_tokens: u32) !usize {
    const memory = try memoryPlan(gpa, c, max_tokens);
    return std.math.add(usize, memory.weights, memory.scratch);
}

pub const Tower = struct {
    gpa: std.mem.Allocator,
    c: Config,
    k: Kernels,
    w: wts.Weights,
    scratch: mtl.Buffer,
    inv: []f32, // the axial rotary's head_dim / 4 frequencies
    max_patches: u32,
    chunk: u32,
    bytes: usize,
    patches: Ref, // [P, patch width] bf16
    taps: Ref, // [P, 4] u32
    tap_w: Ref, // [P, 4] f32
    cos: Ref, // [P, head_dim] f32
    sin: Ref,
    x: Ref, // [P, hidden] fp32: the residual stream
    xn: Ref, // [P, hidden]: a norm's or a projection's output; the merger's input [T, merged]
    att: Ref, // [P, hidden]: the heads side by side; the merger's hidden [T, merged]
    qkv: Ref, // [P, 3 hidden] bf16
    q: Ref, // [heads, P, head_dim] f32
    k_: Ref,
    v: Ref,
    s: Ref, // [heads, chunk, P] f32
    o: Ref, // [heads, chunk, head_dim] f32
    mid: Ref, // [P, intermediate] bf16

    /// Kernels, the checkpoint's tower weights (bf16 as stored) and scratch for images of up to `max_tokens` tokens.
    pub fn load(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, c: Config, max_tokens: u32) !*Tower {
        return loadWith(gpa, io, device, dir, c, max_tokens, device);
    }

    pub fn loadWith(gpa: std.mem.Allocator, io: std.Io, device: mtl.Device, dir: []const u8, c: Config, max_tokens: u32, factory: anytype) !*Tower {
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        t.gpa = gpa;
        t.c = c;
        t.k = try Kernels.load(gpa, device);
        errdefer t.k.deinit();
        t.w = try wts.loadWith(gpa, io, factory, dir, c);
        errdefer t.w.deinit(gpa);
        t.inv = try gpa.alloc(f32, c.headDim() / 4);
        errdefer gpa.free(t.inv);
        grid.invFreq(t.inv, c.headDim(), theta);
        const P: usize = @as(usize, max_tokens) * c.merge * c.merge;
        t.max_patches = @intCast(P);
        const plan = Scratch.of(c, P);
        t.chunk = plan.chunk;
        t.scratch = try factory.buffer(plan.bytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer t.scratch.deinit();
        var at: usize = 0;
        const views = [_]*Ref{ &t.patches, &t.taps, &t.tap_w, &t.cos, &t.sin, &t.x, &t.xn, &t.att, &t.qkv, &t.q, &t.k_, &t.v, &t.s, &t.o, &t.mid };
        for (views, plan.parts) |view, n| {
            view.* = .{ .buf = t.scratch, .off = at };
            at += std.mem.alignForward(usize, n, 256);
        }
        t.bytes = plan.bytes() + t.w.buf.length();
        return t;
    }

    pub fn deinit(t: *Tower) void {
        t.scratch.deinit();
        t.gpa.free(t.inv);
        t.w.deinit(t.gpa);
        t.k.deinit();
        t.gpa.destroy(t);
    }

    /// Buffers to keep resident with the engine's.
    pub fn buffers(t: *const Tower) [2]mtl.Buffer {
        return .{ t.w.buf, t.scratch };
    }

    /// One image's patches ([gh * gw, patch width] fp32, merge order) to merged rows at `out` (bf16, out_hidden wide).
    pub fn image(t: *Tower, queue: mtl.Queue, pixels: []const f32, gh: u32, gw: u32, out: Ref) !void {
        const c = t.c;
        const M = c.merge;
        const P: u32 = gh * gw;
        if (P == 0 or gh % M != 0 or gw % M != 0 or pixels.len != @as(usize, P) * c.patchWidth()) return error.ImageGrid;
        if (P > t.max_patches) return error.ImageTooLarge;
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        t.tables(pixels, gh, gw);
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        t.encode(e, P, out);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("qwen vision: an image's tower pass failed on the GPU: {s}", .{msg});
            return error.TowerGpuFailure;
        }
    }

    /// The host's part: bf16 patches, each patch's four learned-position taps, its rotary cos and sin.
    fn tables(t: *Tower, pixels: []const f32, gh: u32, gw: u32) void {
        const P: usize = @as(usize, gh) * gw;
        const hd: usize = t.c.headDim();
        const dst: [*]u16 = @ptrCast(@alignCast(t.patches.addr()));
        for (pixels, 0..) |p, i| dst[i] = bf16(p);
        const idx: [*]u32 = @ptrCast(@alignCast(t.taps.addr()));
        const wt: [*]f32 = @ptrCast(@alignCast(t.tap_w.addr()));
        for (0..P) |i| {
            const at = grid.place(i, gw, t.c.merge);
            const tp = grid.taps(at[0], at[1], gh, gw, t.c.side());
            @memcpy(idx[i * 4 ..][0..4], &tp.index);
            @memcpy(wt[i * 4 ..][0..4], &tp.weight);
        }
        const cs: [*]f32 = @ptrCast(@alignCast(t.cos.addr()));
        const sn: [*]f32 = @ptrCast(@alignCast(t.sin.addr()));
        grid.rotary(cs[0 .. P * hd], sn[0 .. P * hd], gh, gw, t.c.merge, t.inv);
    }

    fn encode(t: *Tower, e: mtl.ComputeEncoder, P: u32, out: Ref) void {
        const c = t.c;
        const h = c.hidden;
        const T = P / (c.merge * c.merge);
        t.linear(e, t.patches, t.w.patch_w, t.xn, P, h, c.patchWidth());
        e.setPipeline(t.k.embed);
        for ([_]Ref{ t.xn, t.w.patch_b, t.w.pos, t.taps, t.tap_w }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(h, 5);
        e.setBuffer(t.x.buf, t.x.off, 6);
        e.dispatchThreads(mtl.Size.of(h, P, 1), mtl.Size.of(256, 1, 1));
        for (t.w.blocks) |b| t.block(e, b, P);
        t.norm(e, t.x, t.w.norm_w, t.w.norm_b, t.xn, P, h);
        // the merge: four neighbouring patches' rows, in merge order, are one row of xn read as [T, merged]
        t.linear(e, t.xn, t.w.fc1_w, t.att, T, c.merged(), c.merged());
        t.gelu(e, t.att, t.w.fc1_b, T, c.merged(), true);
        t.linear(e, t.att, t.w.fc2_w, out, T, c.out_hidden, c.merged());
        e.setPipeline(t.k.bias);
        e.setBuffer(out.buf, out.off, 0);
        e.setBuffer(t.w.fc2_b.buf, t.w.fc2_b.off, 1);
        e.setValue(c.out_hidden, 2);
        e.dispatchThreads(mtl.Size.of(c.out_hidden, T, 1), mtl.Size.of(256, 1, 1));
    }

    fn block(t: *Tower, e: mtl.ComputeEncoder, b: wts.Block, P: u32) void {
        const c = t.c;
        const h = c.hidden;
        const hd = c.headDim();
        t.norm(e, t.x, b.norm1_w, b.norm1_b, t.xn, P, h);
        t.linear(e, t.xn, b.qkv_w, t.qkv, P, 3 * h, h);
        e.setPipeline(t.k.qkv);
        for ([_]Ref{ t.qkv, b.qkv_b, t.cos, t.sin, t.q, t.k_, t.v }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue([3]u32{ P, c.heads, hd }, 7);
        e.dispatchThreads(mtl.Size.of(hd, 3 * c.heads, P), mtl.Size.of(hd, @max(1, 256 / hd), 1));
        var q0: u32 = 0;
        while (q0 < P) : (q0 += t.chunk) {
            const lc = @min(t.chunk, P - q0);
            const stride = @as(usize, P) * hd;
            // every head's scores of this chunk's queries against all keys, then their weighted values
            t.gemm(e, 1, t.q.at(@as(usize, q0) * hd * 4), t.k_, t.s, lc, P, hd, hd, hd, P, c.heads, stride, stride, @as(usize, lc) * P);
            e.setPipeline(t.k.softmax);
            e.setBuffer(t.s.buf, t.s.off, 0);
            e.setValue(P, 1);
            e.setValue(1.0 / @sqrt(@as(f32, @floatFromInt(hd))), 2);
            e.dispatchGroups(mtl.Size.of(@as(usize, c.heads) * lc, 1, 1), mtl.Size.of(256, 1, 1));
            t.gemm(e, 2, t.s, t.v, t.o, lc, hd, P, P, hd, hd, c.heads, @as(usize, lc) * P, stride, @as(usize, lc) * hd);
            e.setPipeline(t.k.heads);
            e.setBuffer(t.o.buf, t.o.off, 0);
            e.setBuffer(t.att.buf, t.att.off, 1);
            e.setValue([4]u32{ lc, q0, c.heads, hd }, 2);
            e.dispatchThreads(mtl.Size.of(h, lc, 1), mtl.Size.of(256, 1, 1));
        }
        t.linear(e, t.att, b.proj_w, t.xn, P, h, h);
        t.residual(e, t.xn, b.proj_b, P, h);
        t.norm(e, t.x, b.norm2_w, b.norm2_b, t.xn, P, h);
        t.linear(e, t.xn, b.fc1_w, t.mid, P, c.intermediate, h);
        t.gelu(e, t.mid, b.fc1_b, P, c.intermediate, false);
        t.linear(e, t.mid, b.fc2_w, t.xn, P, h, c.intermediate);
        t.residual(e, t.xn, b.fc2_b, P, h);
    }

    /// y [M, N] = x [M, K] w^T in bf16 with fp32 sums (w as stored, [N, K]).
    fn linear(t: *Tower, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: u32, N: u32, K: u32) void {
        t.gemm(e, 0, x, w, y, M, N, K, K, K, N, 1, 0, 0, 0);
    }

    /// tfp::gemm_nax: d [batch, M, N] = a x b (kind 0: bf16 a b^T, 1: fp32 a b^T, 2: fp32 a b), batches `s*` apart.
    fn gemm(t: *Tower, e: mtl.ComputeEncoder, kind: usize, a: Ref, b: Ref, d: Ref, M: u32, N: u32, K: u32, lda: u32, ldb: u32, ldd: u32, batch: u32, sa: usize, sb: usize, sd: usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(t.k.gemm[kind][@intFromBool(M % 64 == 0)][@intFromBool(N % 128 == 0)]);
        e.setBuffer(a.buf, a.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, lda, ldb, ldd, tn, tm, 2, K / 256, sa, sb, sd, 0, 0 };
        for (vals, 0..) |x, i| p[i] = @intCast(x);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(d.buf, d.off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, batch) * 2), mtl.Size.of(32, 4, 2));
    }

    fn norm(t: *Tower, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: u32, width: u32) void {
        e.setPipeline(t.k.norm);
        for ([_]Ref{ x, w, b, y }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(extern struct { rows: u32, width: u32, eps: f32 }{ .rows = rows, .width = width, .eps = 1e-6 }, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn gelu(t: *Tower, e: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, width: u32, exact: bool) void {
        e.setPipeline(t.k.gelu);
        e.setBuffer(y.buf, y.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        e.setValue([2]u32{ width, @intFromBool(exact) }, 2);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }

    /// x += y + b on the residual stream.
    fn residual(t: *Tower, e: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, width: u32) void {
        e.setPipeline(t.k.residual);
        for ([_]Ref{ t.x, y, b }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(width, 3);
        e.dispatchThreads(mtl.Size.of(width, rows, 1), mtl.Size.of(256, 1, 1));
    }
};

/// fp32 to bf16, round to nearest even (torch's and MLX's cast).
pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (b & 0x7fffffff > 0x7f800000) return @truncate((b >> 16) | 0x40); // NaN stays NaN
    return @truncate((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

test "a 4,096-token tower's scratch, and bf16's rounding" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const sample = @import("config.zig");
    const c = (try Config.parse(arena.allocator(), sample.sample_config, null, null)).?;
    const s = Scratch.of(c, 16384);
    try std.testing.expectEqual(@as(u32, 256), s.chunk);
    try std.testing.expectEqual(@as(usize, 961_675_264), s.bytes()); // 917 MiB, the scores' 256 MiB the largest part
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(1.0));
    try std.testing.expectEqual(@as(u16, 0x3f80), bf16(@bitCast(@as(u32, 0x3f808000)))); // a tie to even, down
    try std.testing.expectEqual(@as(u16, 0x3f82), bf16(@bitCast(@as(u32, 0x3f818000)))); // a tie to even, up
}

test "every tower pipeline compiles and loads on this GPU" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var k = try Kernels.load(std.testing.allocator, device);
    k.deinit();
}
