//! Step-5's image tower on Metal: patch embed, 47 perception-encoder blocks, two stride-2 convs and the projector.
const std = @import("std");
const mtl = @import("metal");
const ks = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
const ck = @import("../../core/checkpoint_metal.zig");
const rope_freq = @import("../../core/rope_freq.zig");
const image = @import("image.zig");

pub const width = 1536;
pub const heads = 16;
pub const head_dim = 96;
pub const depth = 47;
pub const mlp_inter = 8960;
pub const out_hidden = 4096; // the language model's width
const big = image.overview / image.patch; // 52: the grid the position table and the RoPE cache are made for
const small = image.crop / image.patch; // 36
const max_patches = big * big;
const chunk = 256; // query rows an attention pass scores at once

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
fn ref(t: ck.Tensor) Ref {
    return .{ .buf = t.buffer, .off = t.offset };
}

const glue_names = [_][:0]const u8{ "s5v_bias", "s5v_scale_add", "s5v_add", "s5v_layer_norm", "s5v_qkv", "s5v_softmax", "s5v_heads", "s5v_quick_gelu", "s5v_im2col" };
const G = enum { bias, scale_add, add, layer_norm, qkv, softmax, heads_, quick_gelu, im2col };

const Block = struct { ln1_w: Ref, ln1_b: Ref, ln2_w: Ref, ln2_b: Ref, qkv_w: Ref, qkv_b: Ref, out_w: Ref, out_b: Ref, fc_w: Ref, fc_b: Ref, proj_w: Ref, proj_b: Ref, ls1: Ref, ls2: Ref };

pub const Vision = struct {
    gpa: std.mem.Allocator,
    lib: mtl.Library,
    gemm: [3][2][2]mtl.Pipeline, // [bf16 nt, f32 nt, f32 nn][M % 64 == 0][N % 128 == 0]
    glue: [glue_names.len]mtl.Pipeline,
    conv1: Ref,
    ln_pre_w: Ref,
    ln_pre_b: Ref,
    blocks: [depth]Block,
    down1_b: Ref,
    down2_b: Ref,
    proj: Ref,
    owned: mtl.Buffer, // down1 and down2 reordered, the position tables, the RoPE tables
    down1: Ref,
    down2: Ref,
    pos: [2]Ref, // the 52 x 52 table, and resized to 36 x 36
    cos: [2]Ref,
    sin: [2]Ref,
    scratch: mtl.Buffer,
    patches: Ref,
    x: Ref,
    xn: Ref,
    qkv: Ref,
    q: Ref,
    k: Ref,
    v: Ref,
    s: Ref,
    o: Ref,
    att: Ref,
    h: Ref,

    pub fn load(gpa: std.mem.Allocator, device: mtl.Device, c: *const ck.Checkpoint, hidden: u32) !*Vision {
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        v.gpa = gpa;
        const tables = try readTensors(v, c, hidden);
        const d1 = tables.d1;
        const d2 = tables.d2;
        const pos = tables.pos;
        const src = try frags.source(device, gpa, ks.step5_vision);
        defer gpa.free(src);
        v.lib = try mtl.Library.fromSource(device, src, mtl.CompileOptions.mlx());
        errdefer v.lib.deinit();
        const kinds = [_][]const u8{ "bf16_nt", "f32_nt", "f32_nn" };
        const gemms: *[12]mtl.Pipeline = @ptrCast(&v.gemm); // [kind][m][n] in order
        var made: usize = 0; // pipelines created so far, gemms then glue, released if a later step fails
        errdefer for (0..made) |i| (if (i < gemms.len) gemms[i] else v.glue[i - gemms.len]).deinit();
        for (gemms, 0..) |*p, i| {
            var name: [64]u8 = undefined;
            const full = try std.fmt.bufPrint(&name, "s5v_gemm_{s}_{d}{d}", .{ kinds[i / 4], i / 2 % 2, i % 2 });
            p.* = try mtl.Pipeline.init(device, v.lib, full, false);
            made += 1;
        }
        for (glue_names, 0..) |n, i| {
            v.glue[i] = try mtl.Pipeline.init(device, v.lib, n, false);
            made += 1;
        }

        // owned: down1, down2 as [out, ky, kx, in]; the 36 x 36 position table; cos and sin for both grids
        const n_d1 = d1.count();
        const n_d2 = d2.count();
        const n_pos = small * small * width;
        const n_rope = (max_patches + small * small) * head_dim;
        v.owned = try device.buffer((n_d1 + n_d2 + n_pos + 2 * n_rope) * 2, mtl.ResourceOptions.shared);
        errdefer v.owned.deinit();
        const o = v.owned.slice(u16, n_d1 + n_d2 + n_pos + 2 * n_rope);
        reorder(d1.host(u16), o[0..n_d1], 2 * width, width);
        reorder(d2.host(u16), o[n_d1..][0..n_d2], 4 * width, 2 * width);
        resizePos(pos.host(u16), o[n_d1 + n_d2 ..][0..n_pos]);
        const at = n_d1 + n_d2 + n_pos;
        rope(o[at..][0 .. max_patches * head_dim], o[at + n_rope ..][0 .. max_patches * head_dim], big);
        rope(o[at + max_patches * head_dim ..][0 .. small * small * head_dim], o[at + n_rope + max_patches * head_dim ..][0 .. small * small * head_dim], small);
        v.down1 = .{ .buf = v.owned };
        v.down2 = .{ .buf = v.owned, .off = n_d1 * 2 };
        v.pos = .{ ref(pos), .{ .buf = v.owned, .off = (n_d1 + n_d2) * 2 } };
        v.cos = .{ .{ .buf = v.owned, .off = at * 2 }, .{ .buf = v.owned, .off = (at + max_patches * head_dim) * 2 } };
        v.sin = .{ .{ .buf = v.owned, .off = (at + n_rope) * 2 }, .{ .buf = v.owned, .off = (at + n_rope + max_patches * head_dim) * 2 } };
        // scratch, sized for the overview's 2704 patches
        const P: usize = max_patches;
        const sizes = [_]usize{ P * image.patch_width * 2, P * width * 2, P * width * 2, P * 3 * width * 2, heads * P * head_dim * 4, heads * P * head_dim * 4, heads * P * head_dim * 4, heads * chunk * P * 4, heads * chunk * head_dim * 4, P * width * 2, P * @max(mlp_inter, 9 * width) * 2 };
        var total: usize = 0;
        for (sizes) |s| total += std.mem.alignForward(usize, s, 256);
        v.scratch = try device.buffer(total, mtl.ResourceOptions.private);
        errdefer v.scratch.deinit();
        var off: usize = 0;
        const fields = [_]*Ref{ &v.patches, &v.x, &v.xn, &v.qkv, &v.q, &v.k, &v.v, &v.s, &v.o, &v.att, &v.h };
        for (fields, sizes) |f, s| {
            f.* = .{ .buf = v.scratch, .off = off };
            off += std.mem.alignForward(usize, s, 256);
        }
        // patches are written by the host: keep them in a shared buffer of their own
        v.patches = .{ .buf = try device.buffer(P * image.patch_width * 2, mtl.ResourceOptions.shared) };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        for (&v.gemm) |*a| for (a) |*b| for (b) |p| p.deinit();
        for (v.glue) |p| p.deinit();
        v.lib.deinit();
        v.owned.deinit();
        v.scratch.deinit();
        v.patches.buf.deinit();
        v.gpa.destroy(v);
    }

    /// One piece (patches [gh * gw, 588] f32, grid 52 or 36) through the tower into `out` ([gh * gw / 16, 4096]).
    pub fn encode(v: *Vision, e: mtl.ComputeEncoder, pixels: []const f32, gh: u32, gw: u32, out: Ref) !void {
        if (gh != gw or (gh != big and gh != small)) return error.Step5ImageGrid;
        const P: u32 = gh * gw;
        if (pixels.len != @as(usize, P) * image.patch_width) return error.Step5ImageGrid;
        const which: usize = if (gh == big) 0 else 1;
        const dst = v.patches.buf.slice(u16, @as(usize, P) * image.patch_width);
        for (pixels, dst) |p, *d| d.* = bf16(p);
        v.linear(e, v.patches, v.conv1, v.x, P, width, image.patch_width);
        v.run(e, .add, &.{ v.x, v.pos[which] }, .{ @as(usize, P) * width, 1, 1 }, .{ 256, 1, 1 }, false, null);
        v.layerNorm(e, v.x, v.ln_pre_w, v.ln_pre_b, v.xn, P);
        std.mem.swap(Ref, &v.x, &v.xn);
        for (v.blocks) |b| v.block(e, b, P, which);
        // the grid's two stride-2 convs as GEMMs over their windows, then the projector
        const g1 = (gh - 1) / 2 + 1;
        v.im2col(e, v.x, v.h, gh, width);
        v.linear(e, v.h, v.down1, v.qkv, g1 * g1, 2 * width, 9 * width);
        v.bias(e, v.qkv, v.down1_b, g1 * g1, 2 * width);
        const g2 = (g1 - 1) / 2 + 1;
        v.im2col(e, v.qkv, v.h, g1, 2 * width);
        v.linear(e, v.h, v.down2, v.att, g2 * g2, 4 * width, 9 * 2 * width);
        v.bias(e, v.att, v.down2_b, g2 * g2, 4 * width);
        v.linear(e, v.att, v.proj, out, g2 * g2, out_hidden, 4 * width);
    }

    fn block(v: *Vision, e: mtl.ComputeEncoder, b: Block, P: u32, which: usize) void {
        v.layerNorm(e, v.x, b.ln1_w, b.ln1_b, v.xn, P);
        v.linear(e, v.xn, b.qkv_w, v.qkv, P, 3 * width, width);
        v.bias(e, v.qkv, b.qkv_b, P, 3 * width);
        e.setPipeline(v.glue[@backingInt(G.qkv)]);
        for ([_]Ref{ v.qkv, v.cos[which], v.sin[which], v.q, v.k, v.v }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(P, 6);
        e.dispatchGroups(mtl.Size.of(@as(usize, P) * heads * 3, 1, 1), mtl.Size.of(32, 1, 1));
        var q0: u32 = 0;
        while (q0 < P) : (q0 += chunk) {
            const lc = @min(chunk, P - q0);
            v.gemmAt(e, 1, .{ .buf = v.q.buf, .off = v.q.off + @as(usize, q0) * head_dim * 4 }, v.k, v.s, lc, P, head_dim, head_dim, head_dim, P, heads, @as(usize, P) * head_dim, @as(usize, P) * head_dim, @as(usize, lc) * P);
            e.setPipeline(v.glue[@backingInt(G.softmax)]);
            e.setBuffer(v.s.buf, v.s.off, 0);
            e.setValue(P, 1);
            e.setValue(@as(f32, 1.0 / @sqrt(@as(f32, head_dim))), 2);
            e.dispatchGroups(mtl.Size.of(@as(usize, heads) * lc, 1, 1), mtl.Size.of(256, 1, 1));
            v.gemmAt(e, 2, v.s, v.v, v.o, lc, head_dim, P, P, head_dim, head_dim, heads, @as(usize, lc) * P, @as(usize, P) * head_dim, @as(usize, lc) * head_dim);
            e.setPipeline(v.glue[@backingInt(G.heads_)]);
            e.setBuffer(v.o.buf, v.o.off, 0);
            e.setBuffer(v.att.buf, v.att.off, 1);
            e.setValue([2]u32{ lc, q0 }, 2);
            e.dispatchThreads(mtl.Size.of(width, lc, 1), mtl.Size.of(256, 1, 1));
        }
        v.linear(e, v.att, b.out_w, v.xn, P, width, width);
        v.bias(e, v.xn, b.out_b, P, width);
        v.scaleAdd(e, v.x, v.xn, b.ls1, P);
        v.layerNorm(e, v.x, b.ln2_w, b.ln2_b, v.xn, P);
        v.linear(e, v.xn, b.fc_w, v.h, P, mlp_inter, width);
        v.bias(e, v.h, b.fc_b, P, mlp_inter);
        v.run(e, .quick_gelu, &.{v.h}, .{ @as(usize, P) * mlp_inter, 1, 1 }, .{ 256, 1, 1 }, false, null);
        v.linear(e, v.h, b.proj_w, v.xn, P, width, mlp_inter);
        v.bias(e, v.xn, b.proj_b, P, width);
        v.scaleAdd(e, v.x, v.xn, b.ls2, P);
    }

    fn run(v: *Vision, e: mtl.ComputeEncoder, g: G, bufs: []const Ref, grid: [3]usize, group: [3]usize, groups: bool, value: ?u32) void {
        e.setPipeline(v.glue[@backingInt(g)]);
        for (bufs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        if (value) |x| e.setValue(x, bufs.len);
        const a = mtl.Size.of(grid[0], grid[1], grid[2]);
        const b = mtl.Size.of(group[0], group[1], group[2]);
        if (groups) e.dispatchGroups(a, b) else e.dispatchThreads(a, b);
    }

    fn bias(v: *Vision, e: mtl.ComputeEncoder, y: Ref, b: Ref, rows: u32, n: u32) void {
        v.run(e, .bias, &.{ y, b }, .{ n, rows, 1 }, .{ 256, 1, 1 }, false, n);
    }

    fn scaleAdd(v: *Vision, e: mtl.ComputeEncoder, x: Ref, y: Ref, gamma: Ref, rows: u32) void {
        v.run(e, .scale_add, &.{ x, y, gamma }, .{ width, rows, 1 }, .{ 256, 1, 1 }, false, width);
    }

    fn layerNorm(v: *Vision, e: mtl.ComputeEncoder, x: Ref, w: Ref, b: Ref, y: Ref, rows: u32) void {
        e.setPipeline(v.glue[@backingInt(G.layer_norm)]);
        for ([_]Ref{ x, w, b, y }, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(extern struct { rows: u32, width: u32, eps: f32 }{ .rows = rows, .width = width, .eps = 1e-5 }, 4);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(256, 1, 1));
    }

    fn im2col(v: *Vision, e: mtl.ComputeEncoder, x: Ref, cols: Ref, g: u32, C: u32) void {
        const o = (g - 1) / 2 + 1;
        e.setPipeline(v.glue[@backingInt(G.im2col)]);
        e.setBuffer(x.buf, x.off, 0);
        e.setBuffer(cols.buf, cols.off, 1);
        e.setValue([4]u32{ g, g, C, 0 }, 2);
        e.dispatchThreads(mtl.Size.of(9 * C, o * o, 1), mtl.Size.of(256, 1, 1));
    }

    /// y [M, N] = x [M, K] w^T in bf16 (w as stored, [N, K]).
    fn linear(v: *Vision, e: mtl.ComputeEncoder, x: Ref, w: Ref, y: Ref, M: u32, N: u32, K: u32) void {
        v.gemmAt(e, 0, x, w, y, M, N, K, K, K, N, 1, 0, 0, 0);
    }

    /// tfp::gemm_nax: d [batch, M, N] = a x b (kind 0: bf16 a b^T, 1: fp32 a b^T, 2: fp32 a b), batches `s*` apart.
    fn gemmAt(v: *Vision, e: mtl.ComputeEncoder, kind: usize, a: Ref, b: Ref, d: Ref, M: u32, N: u32, K: u32, lda: u32, ldb: u32, ldd: u32, batch: u32, sa: usize, sb: usize, sd: usize) void {
        const tn = (N + 127) / 128;
        const tm = (M + 63) / 64;
        e.setPipeline(v.gemm[kind][@intFromBool(M % 64 == 0)][@intFromBool(N % 128 == 0)]);
        e.setBuffer(a.buf, a.off, 0);
        e.setBuffer(b.buf, b.off, 1);
        var p: [16]i32 = @splat(0);
        const vals = [_]usize{ M, N, K, lda, ldb, ldd, tn, tm, 2, K / 256, sa, sb, sd, 0, 0 };
        for (vals, 0..) |x, i| p[i] = @intCast(x);
        e.setBytes(std.mem.asBytes(&p), 2);
        e.setBuffer(d.buf, d.off, 3);
        e.dispatchThreads(mtl.Size.of(@as(usize, tn << 2) * 32, (tm + 3) / 4 * 4, @as(usize, batch) * 2), mtl.Size.of(32, 4, 2));
    }
};

/// fp32 to bf16 as core's bf16_round.bits rounds (NaN stays NaN); a copy, the contract tests build core as a module.
pub fn bf16(x: f32) u16 {
    const u: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @truncate((u >> 16) | 0x40);
    return @truncate((u +% 0x7fff +% ((u >> 16) & 1)) >> 16);
}

fn f32of(h: u16) f32 {
    return @bitCast(@as(u32, h) << 16);
}

/// A conv weight [out, in, 3, 3] as [out, ky, kx, in]: the GEMM over im2col's (ky, kx, c) windows.
fn reorder(src: []const u16, dst: []u16, out: usize, in: usize) void {
    for (0..out) |o| for (0..in) |c| for (0..9) |t| {
        dst[(o * 9 + t) * in + c] = src[(o * in + c) * 9 + t];
    };
}

/// The 52 x 52 position table at 36 x 36: F.interpolate bilinear, align_corners false, in fp32, rounded to bf16.
fn resizePos(src: []const u16, dst: []u16) void {
    const scale = @as(f32, big) / @as(f32, small);
    for (0..small) |i| for (0..small) |j| {
        const sy = @max((@as(f32, @floatFromInt(i)) + 0.5) * scale - 0.5, 0);
        const sx = @max((@as(f32, @floatFromInt(j)) + 0.5) * scale - 0.5, 0);
        const y0: usize = @intFromFloat(sy);
        const x0: usize = @intFromFloat(sx);
        const y1: usize = @min(y0 + 1, big - 1);
        const x1: usize = @min(x0 + 1, big - 1);
        const ly = sy - @as(f32, @floatFromInt(y0));
        const lx = sx - @as(f32, @floatFromInt(x0));
        for (0..width) |c| {
            const a = f32of(src[(y0 * big + x0) * width + c]);
            const b = f32of(src[(y0 * big + x1) * width + c]);
            const cc = f32of(src[(y1 * big + x0) * width + c]);
            const d = f32of(src[(y1 * big + x1) * width + c]);
            const top = a * (1 - lx) + b * lx;
            const bot = cc * (1 - lx) + d * lx;
            dst[(i * small + j) * width + c] = bf16(top * (1 - ly) + bot * ly);
        }
    };
}

/// The encoder's 2D RoPE for a g x g grid: dims [w | h] freqs, pairs repeated, bf16 angles, bf16 cos and sin.
fn rope(cs: []u16, sn: []u16, g: usize) void {
    const half = head_dim / 2; // 48 dims an axis: 24 frequencies, each twice
    var inv: [half / 2]f32 = undefined;
    rope_freq.inverseFrequencies(10000.0, half, &inv);
    for (0..g) |r| for (0..g) |c| for (0..head_dim) |d| {
        const axis_pos: f32 = @floatFromInt(if (d < half) c else r);
        const i = (d % half) / 2;
        const angle = f32of(bf16(axis_pos * inv[i]));
        cs[(r * g + c) * head_dim + d] = bf16(@cos(angle));
        sn[(r * g + c) * head_dim + d] = bf16(@sin(angle));
    };
}

const Tables = struct { d1: ck.Tensor, d2: ck.Tensor, pos: ck.Tensor };

pub fn checkedTensor(t: ck.Tensor, dims: []const usize) !ck.Tensor {
    const shape = @import("tensor_shape.zig");
    if (dims.len == 4 and dims[0] == width and dims[1] == 3 and dims[2] == image.patch and dims[3] == image.patch and t.rank == 2) {
        try shape.require(t, .bf16, &.{ width, image.patch_width });
    } else if (std.mem.eql(usize, dims, &.{ max_patches, width }) and t.rank == 3 and t.shape[0] == 1) {
        try shape.require(t, .bf16, &.{ 1, max_patches, width });
    } else try shape.require(t, .bf16, dims);
    return t;
}

pub fn requireHidden(hidden: u32) error{UnsupportedStep5Tower}!void {
    if (hidden != out_hidden) return error.UnsupportedStep5Tower;
}

fn readTensors(v: *Vision, c: *const ck.Checkpoint, hidden: u32) !Tables {
    try requireHidden(hidden);
    const t = struct {
        fn get(cp: *const ck.Checkpoint, comptime fmt: []const u8, args: anytype, dims: []const usize) !ck.Tensor {
            var b: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&b, fmt, args);
            const x = try cp.get(name);
            return checkedTensor(x, dims);
        }
    };
    v.conv1 = ref(try t.get(c, "vision_model.conv1.weight", .{}, &.{ width, 3, image.patch, image.patch }));
    v.ln_pre_w = ref(try t.get(c, "vision_model.ln_pre.weight", .{}, &.{width}));
    v.ln_pre_b = ref(try t.get(c, "vision_model.ln_pre.bias", .{}, &.{width}));
    for (&v.blocks, 0..) |*b, i| {
        const p = "vision_model.transformer.resblocks.{d}.";
        b.* = .{
            .ln1_w = ref(try t.get(c, p ++ "ln_1.weight", .{i}, &.{width})),
            .ln1_b = ref(try t.get(c, p ++ "ln_1.bias", .{i}, &.{width})),
            .ln2_w = ref(try t.get(c, p ++ "ln_2.weight", .{i}, &.{width})),
            .ln2_b = ref(try t.get(c, p ++ "ln_2.bias", .{i}, &.{width})),
            .qkv_w = ref(try t.get(c, p ++ "attn.in_proj_weight", .{i}, &.{ 3 * width, width })),
            .qkv_b = ref(try t.get(c, p ++ "attn.in_proj_bias", .{i}, &.{3 * width})),
            .out_w = ref(try t.get(c, p ++ "attn.out_proj.weight", .{i}, &.{ width, width })),
            .out_b = ref(try t.get(c, p ++ "attn.out_proj.bias", .{i}, &.{width})),
            .fc_w = ref(try t.get(c, p ++ "mlp.c_fc.weight", .{i}, &.{ mlp_inter, width })),
            .fc_b = ref(try t.get(c, p ++ "mlp.c_fc.bias", .{i}, &.{mlp_inter})),
            .proj_w = ref(try t.get(c, p ++ "mlp.c_proj.weight", .{i}, &.{ width, mlp_inter })),
            .proj_b = ref(try t.get(c, p ++ "mlp.c_proj.bias", .{i}, &.{width})),
            .ls1 = ref(try t.get(c, p ++ "ls_1.gamma", .{i}, &.{width})),
            .ls2 = ref(try t.get(c, p ++ "ls_2.gamma", .{i}, &.{width})),
        };
    }
    v.down1_b = ref(try t.get(c, "vision_model.vit_downsampler1.bias", .{}, &.{2 * width}));
    v.down2_b = ref(try t.get(c, "vision_model.vit_downsampler2.bias", .{}, &.{4 * width}));
    v.proj = ref(try t.get(c, "vit_large_projector.weight", .{}, &.{ out_hidden, 4 * width }));
    const d1 = try t.get(c, "vision_model.vit_downsampler1.weight", .{}, &.{ 2 * width, width, 3, 3 });
    const d2 = try t.get(c, "vision_model.vit_downsampler2.weight", .{}, &.{ 4 * width, 2 * width, 3, 3 });
    const pos = try t.get(c, "vision_model.positional_embedding", .{}, &.{ max_patches, width });
    return .{ .d1 = d1, .d2 = d2, .pos = pos };
}

pub fn validate(c: *const ck.Checkpoint, hidden: u32) !void {
    var v: Vision = undefined;
    _ = try readTensors(&v, c, hidden);
}

test {
    _ = @import("vision_test.zig");
}

/// The replaced frequency (rope before #1852), kept as a reference: Zig's f32 pow, an fp32 reciprocal.
fn replacedInv(i: usize) f32 {
    return 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(2 * i)) / @as(f32, head_dim / 2));
}

/// The replaced rope: its frequencies through the same bf16 angles and bf16 cos and sin.
fn replacedRope(cs: []u16, sn: []u16, g: usize) void {
    const half = head_dim / 2;
    for (0..g) |r| for (0..g) |c| for (0..head_dim) |d| {
        const axis_pos: f32 = @floatFromInt(if (d < half) c else r);
        const angle = f32of(bf16(axis_pos * replacedInv((d % half) / 2)));
        cs[(r * g + c) * head_dim + d] = bf16(@cos(angle));
        sn[(r * g + c) * head_dim + d] = bf16(@sin(angle));
    };
}

const Diff = struct { differ: usize, ulps: u32 };

/// Counts the words that differ; when `loud`, prints both tables side by side and the largest differences.
fn sideBySide(label: []const u8, old: []const f32, new: []const f32, loud: bool) Diff {
    var r: Diff = .{ .differ = 0, .ulps = 0 };
    var abs: f64 = 0;
    var rel: f64 = 0;
    if (loud) std.debug.print("{s}\n   i  old                         new                         ulps\n", .{label});
    for (old, new, 0..) |o, n, i| {
        const a: u32 = @bitCast(o);
        const b: u32 = @bitCast(n);
        const u = if (a > b) a - b else b - a;
        const d = @abs(@as(f64, n) - o);
        if (u != 0) r.differ += 1;
        r.ulps = @max(r.ulps, u);
        abs = @max(abs, d);
        rel = @max(rel, d / @abs(@as(f64, o)));
        const sign = if (b > a) "+" else if (b < a) "-" else " ";
        if (loud) std.debug.print("{d:>4}  0x{x:0>8} {e:<16}  0x{x:0>8} {e:<16}  {s}{d}\n", .{ i, a, o, b, n, sign, u });
    }
    if (loud) std.debug.print("{s}: {d} of {d} words differ, largest {d} ulp, abs {e:.3}, rel {e:.3}\n", .{
        label, r.differ, old.len, r.ulps, abs, rel,
    });
    return r;
}

/// The bf16 entries of two tables that differ; when `loud`, printed with the largest and the first patch hit.
fn tableDiff(label: []const u8, old: []const u16, new: []const u16, loud: bool) usize {
    var differ: usize = 0;
    var most: f32 = 0;
    var first: ?usize = null;
    for (old, new, 0..) |o, n, i| if (o != n) {
        differ += 1;
        most = @max(most, @abs(f32of(n) - f32of(o)));
        if (first == null) first = i / head_dim;
    };
    if (loud) std.debug.print("{s}: {d} of {d} entries differ, largest {e:.3}, first at patch {?d}\n", .{
        label, differ, old.len, most, first,
    });
    return differ;
}

test "step5 host: tower RoPE frequencies (theta 1e4 over 48) on rope_freq, beside the fp32-pow form" {
    // transformers' form in exact arithmetic, rn32(1 / rn32(1e4^rn32(2i / 48))): it gives rope_freq.zig's torch words
    const want = [24]u32{
        0x3f800000, 0x3f2e6928, 0x3eeda63c, 0x3ea1e89b, 0x3e5c9d35, 0x3e164d7a, 0x3dcccccd, 0x3d8b8754,
        0x3d3e1e95, 0x3d0186e3, 0x3cb07dc6, 0x3c707bf6, 0x3c23d70a, 0x3bdf3ee9, 0x3b98187a, 0x3b4f3e38,
        0x3b0d3169, 0x3ac0632e, 0x3a83126f, 0x3a3298ba, 0x39f35a5c, 0x39a5cb60, 0x3961e8a7, 0x3919e8f2,
    };
    var new: [head_dim / 4]f32 = undefined;
    rope_freq.inverseFrequencies(10000.0, head_dim / 2, &new);
    var old: [head_dim / 4]f32 = undefined;
    for (&old, 0..) |*f, i| f.* = replacedInv(i);
    const label = "Step-5 tower (theta 1e4, 48 dims an axis)";
    for (want, new, 0..) |w, f, i| std.testing.expectEqual(w, @as(u32, @bitCast(f))) catch |e| {
        std.debug.print("\n{s}: word {d}: expected 0x{x:0>8}, got 0x{x:0>8}\n", .{ label, i, w, @as(u32, @bitCast(f)) });
        _ = sideBySide(label, &old, &new, true);
        return e;
    };
    const got = sideBySide(label, &old, &new, false);
    std.testing.expectEqual(Diff{ .differ = 8, .ulps = 2 }, got) catch |e| {
        std.debug.print("\n{s}: expected 8 words to move by at most 2 ulp, got {d} by {d}\n", .{ label, got.differ, got.ulps });
        _ = sideBySide(label, &old, &new, true);
        return e;
    };
    const a = std.testing.allocator;
    var counts: [2][2]usize = undefined;
    for ([_]usize{ big, small }, &counts) |g, *count| {
        const n = g * g * head_dim;
        const t = try a.alloc(u16, 4 * n);
        defer a.free(t);
        rope(t[0..n], t[n..][0..n], g);
        replacedRope(t[2 * n ..][0..n], t[3 * n ..][0..n], g);
        var names: [2][64]u8 = undefined;
        const cos = try std.fmt.bufPrint(&names[0], "Step-5 tower bf16 cos, {d} x {d}", .{ g, g });
        const sin = try std.fmt.bufPrint(&names[1], "Step-5 tower bf16 sin, {d} x {d}", .{ g, g });
        count.* = .{ tableDiff(cos, t[2 * n ..][0..n], t[0..n], false), tableDiff(sin, t[3 * n ..][0..n], t[n..][0..n], false) };
        std.testing.expectEqual([2]usize{ 0, 0 }, count.*) catch |e| {
            _ = tableDiff(cos, t[2 * n ..][0..n], t[0..n], true);
            _ = tableDiff(sin, t[3 * n ..][0..n], t[n..][0..n], true);
            return e;
        };
    }
    try std.testing.expectEqual([2][2]usize{ .{ 0, 0 }, .{ 0, 0 } }, counts);
}
