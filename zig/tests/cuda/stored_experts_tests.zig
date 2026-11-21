//! Inkling 2 Mini's stored experts from the checkpoint: decoded weights bit for bit, outputs within the fp32 bound.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const check = @import("check.zig");
const cs = @import("contract_stored_experts.zig");

const Gpu = check.Gpu;
const grouped = cuda.grouped;
const ex_ = cuda.experts;
const mx = cuda.experts_mx;
const dec = core.expert_decode;
const c = core.contract.spec;

pub const hidden = 2304;
pub const width = 1152;
pub const experts = 128;
pub const top = 8;

/// One stored matrix of all experts: codes, scales and (NVFP4) the per-expert factor.
pub const Matrix = struct {
    codes: []const u8,
    scales: []const u8,
    factor: [experts]f32 = @splat(1),
    n: usize,
    k: usize,

    pub fn expertCodes(m: Matrix, e: usize) []const u8 {
        const per = m.codes.len / experts;
        return m.codes[e * per ..][0..per];
    }

    pub fn expertScales(m: Matrix, e: usize) []const u8 {
        const per = m.scales.len / experts;
        return m.scales[e * per ..][0..per];
    }

    /// Weight (row, i) of expert e as the kernels form it in fp32.
    pub fn f32At(m: Matrix, nv: bool, e: usize, row: usize, i: usize) f32 {
        if (nv) return dec.nvfp4F32(m.expertCodes(e), m.expertScales(e), m.k, row, i, m.factor[e]);
        return @floatCast(dec.mxfp8(m.expertCodes(e), m.expertScales(e), m.k, row, i));
    }

    /// Weight (row, i) of expert e in float64 (the reference's).
    pub fn f64At(m: Matrix, nv: bool, e: usize, row: usize, i: usize) f64 {
        if (nv) return dec.nvfp4(m.expertCodes(e), m.expertScales(e), m.k, row, i, m.factor[e]);
        return dec.mxfp8(m.expertCodes(e), m.expertScales(e), m.k, row, i);
    }
};

/// A layer's w13 (gate rows then up rows) and w2 from the mapped checkpoint.
pub const Stored = struct {
    nv: bool,
    w13: Matrix,
    w2: Matrix,

    pub fn read(f: *const core.safetensors.File, layer: usize) !Stored {
        var name: [96]u8 = undefined;
        const pre = "model.llm.layers.{d}.mlp.experts.";
        const codes13 = f.get(try std.fmt.bufPrint(&name, pre ++ "w13_weight", .{layer})) orelse return error.MissingTensor;
        const nv = codes13.dtype == .u8;
        var s: Stored = .{ .nv = nv, .w13 = undefined, .w2 = undefined };
        inline for (.{ "w13", "w2" }, .{ &s.w13, &s.w2 }) |w, m| {
            const t = f.get(try std.fmt.bufPrint(&name, pre ++ w ++ "_weight", .{layer})) orelse return error.MissingTensor;
            const sc = f.get(try std.fmt.bufPrint(&name, pre ++ w ++ "_weight_scale", .{layer})) orelse return error.MissingTensor;
            const k = if (nv) t.dim(2) * 2 else t.dim(2);
            m.* = .{ .codes = t.bytes, .scales = sc.bytes, .n = t.dim(1), .k = k };
            if (nv) {
                const amax = f.get(try std.fmt.bufPrint(&name, pre ++ w ++ "_weight_amax", .{layer})) orelse return error.MissingTensor;
                const den = f.get(try std.fmt.bufPrint(&name, pre ++ w ++ "_weight_outer_scale_denominator", .{layer})) orelse return error.MissingTensor;
                const d = std.mem.bytesToValue(f32, den.bytes[0..4]);
                try check.expect(d == 2688, "layer {d} {s}: outer scale denominator {d}, expected 2688", .{ layer, w, d });
                for (&m.factor, 0..) |*x, e| x.* = mx.nvfp4Factor(std.mem.bytesToValue(f32, amax.bytes[e * 4 ..][0..4]), d);
            }
        }
        try check.expect(s.w13.n == 2 * width and s.w13.k == hidden and s.w2.n == hidden and s.w2.k == width, "layer {d}: expert shapes", .{layer});
        return s;
    }

    /// The layer packed for the kernels: w13 as gate-up pairs, w2 alone, and NVFP4's factors per (expert, matrix).
    pub fn upload(s: Stored, gpu: Gpu, bufs: *[4]cuda.DeviceBuffer) !ex_.Layer {
        const a = gpu.gpa;
        const up_words = (if (s.nv) ex_.Nvfp4Experts.bytes(1, width, hidden, 2) else mx.Mxfp8Experts.bytes(1, width, hidden, 2)) / 4;
        const down_words = (if (s.nv) ex_.Nvfp4Experts.bytes(1, hidden, width, 1) else mx.Mxfp8Experts.bytes(1, hidden, width, 1)) / 4;
        const up = try a.alloc(u32, experts * up_words);
        defer a.free(up);
        const down = try a.alloc(u32, experts * down_words);
        defer a.free(down);
        const cb = s.w13.expertCodes(0).len; // w13 rows interleave gate and up (row 2j gate j, 2j + 1 up j)
        const sb = s.w13.expertScales(0).len;
        const split = try a.alloc(u8, cb + sb);
        defer a.free(split);
        for (0..experts) |e| {
            const rc = cb / (2 * width);
            const rs = sb / (2 * width);
            mx.splitGateUp(s.w13.expertCodes(e), rc, split[0 .. cb / 2], split[cb / 2 .. cb]);
            mx.splitGateUp(s.w13.expertScales(e), rs, split[cb..][0 .. sb / 2], split[cb + sb / 2 ..][0 .. sb / 2]);
            const gate = [2][]const u8{ split[0 .. cb / 2], split[cb..][0 .. sb / 2] };
            const upm = [2][]const u8{ split[cb / 2 .. cb], split[cb + sb / 2 ..][0 .. sb / 2] };
            const o = up[e * up_words ..][0..up_words];
            const d = down[e * down_words ..][0..down_words];
            if (s.nv) {
                ex_.Nvfp4Experts.packGateUp(o, gate, upm, width, hidden);
                ex_.Nvfp4Experts.packOne(d, s.w2.expertCodes(e), s.w2.expertScales(e), hidden, width);
            } else {
                mx.Mxfp8Experts.packGateUp(o, gate, upm, width, hidden);
                mx.Mxfp8Experts.packOne(d, s.w2.expertCodes(e), s.w2.expertScales(e), hidden, width);
            }
        }
        var f13: [experts * 2]f32 = undefined;
        for (0..experts) |e| f13[2 * e] = s.w13.factor[e];
        for (0..experts) |e| f13[2 * e + 1] = s.w13.factor[e]; // one amax covers the gate and the up rows
        bufs[0] = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(up));
        bufs[1] = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(down));
        bufs[2] = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&f13));
        bufs[3] = try cuda.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(&s.w2.factor));
        return .{ .format = if (s.nv) .nvfp4 else .mxfp8, .up = bufs[0].ptr, .down = bufs[1].ptr, .up_scale = bufs[2].ptr, .down_scale = bufs[3].ptr, .width = width, .dims = hidden, .experts = experts, .caps = cs.caps };
    }
};

/// Plan scratch for `pairs` picks over 128 experts in items of `tile`.
pub const PlanBufs = struct {
    bufs: [5]cuda.DeviceBuffer,
    plan: grouped.Plan,
    items: usize,

    pub fn init(gpu: Gpu, pairs: usize, tile: usize) !PlanBufs {
        var p: PlanBufs = undefined;
        p.items = grouped.maxItems(pairs, experts, tile);
        const sizes = [5]usize{ pairs * 4, p.items * 12, 8, pairs * 4, ((pairs + 1023) / 1024) * experts * 4 };
        var made: usize = 0;
        errdefer for (p.bufs[0..made]) |*b| b.free();
        for (sizes, &p.bufs) |n, *b| {
            b.* = try cuda.DeviceBuffer.alloc(gpu.d, n);
            made += 1;
        }
        p.plan = .{ .members = p.bufs[0].ptr, .items = p.bufs[1].ptr, .counts = p.bufs[2].ptr, .rank = p.bufs[3].ptr, .hist = p.bufs[4].ptr };
        return p;
    }

    pub fn deinit(p: *PlanBufs) void {
        for (&p.bufs) |*b| b.free();
    }
};

/// `tokens` rows' top-8 distinct experts, seeded.
pub fn picks(a: std.mem.Allocator, tokens: usize, seed: u64) ![]i32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try a.alloc(i32, tokens * top);
    for (0..tokens) |t| {
        var perm: [experts]i32 = undefined;
        for (&perm, 0..) |*v, i| v.* = @intCast(i);
        prng.random().shuffle(i32, &perm);
        @memcpy(out[t * top ..][0..top], perm[0..top]);
    }
    return out;
}

/// `ckpt DIR`: layers 0 (NVFP4) and 20 (MXFP8) of the checkpoint in DIR.
pub fn ckpt(gpu: Gpu, dir: []const u8) !void {
    const path = try std.fmt.allocPrint(gpu.gpa, "{s}/model-00001-of-00001.safetensors", .{dir});
    defer gpu.gpa.free(path);
    var f = try core.safetensors.File.openPrefix(gpu.gpa, gpu.io, path, "model.llm.layers.");
    defer f.close(gpu.io);
    try @import("stored_experts_check.zig").e8m0Edges(gpu);
    for ([_]usize{ 0, 20 }) |layer| {
        const s = try Stored.read(&f, layer);
        try invariants(s, layer);
        try @import("stored_experts_check.zig").decoded(gpu, s, layer);
        try @import("stored_experts_check.zig").outputs(gpu, s, layer);
    }
}

/// The stored data as the scale analysis found it: NVFP4's code x block scale peaks at exactly 6 x 448 in every
/// expert (so amax / 2688 is the factor), MXFP8 holds no NaN code and peaks at 448.
fn invariants(s: Stored, layer: usize) !void {
    for ([_]Matrix{ s.w13, s.w2 }, 0..) |m, mi| {
        var peaks: usize = 0;
        for (0..experts) |e| {
            var most: f64 = 0;
            const codes = m.expertCodes(e);
            const scales = m.expertScales(e);
            if (s.nv) {
                for (0..m.n) |row| for (0..m.k / 16) |b| {
                    const sv = dec.e4m3(scales[row * (m.k / 16) + b]);
                    for (0..16) |q| most = @max(most, @abs(dec.e2m1(dec.nibble(codes[row * (m.k / 2) ..][0 .. m.k / 2], b * 16 + q)) * sv));
                };
                try check.expect(most == 2688, "layer {d} matrix {d} expert {d}: peak code x scale {d}, expected 2688", .{ layer, mi, e, most });
            } else {
                for (codes) |b| {
                    try check.expect(b & 0x7F != 0x7F, "layer {d} matrix {d} expert {d}: a NaN code", .{ layer, mi, e });
                    most = @max(most, @abs(dec.e4m3(b)));
                }
                for (scales) |b| try check.expect(b != 255, "layer {d} matrix {d} expert {d}: a NaN scale", .{ layer, mi, e });
                if (most == 448) peaks += 1;
            }
        }
        if (s.nv) check.pass("layer {d} NVFP4 matrix {d}: every expert's code x block scale peaks at 2688 = 6 x 448", .{ layer, mi });
        if (!s.nv) check.pass("layer {d} MXFP8 matrix {d}: no NaN code or scale; {d} of 128 experts peak at 448", .{ layer, mi, peaks });
    }
}

test "picks are eight distinct experts a token" {
    const p = try picks(std.testing.allocator, 3, 7);
    defer std.testing.allocator.free(p);
    for (0..3) |t| for (0..top) |i| for (0..i) |j| try std.testing.expect(p[t * top + i] != p[t * top + j]);
}

comptime {
    _ = c;
}
