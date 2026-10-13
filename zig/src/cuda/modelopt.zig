//! ModelOpt checkpoints' tensors as stored, repacked on the GPU into the lane matmul's and NVFP4 experts' layouts.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const qmmf = @import("qmmf.zig");
const nvfp4 = @import("nvfp4.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Rows padded to whole 128-row blocks, as every lane format pads them.
pub const padded = nvfp4.padded;

/// modelopt_pack.cu's kernels; each matches a host packer (fp8.packCodes, qmmf.packBf16, nvfp4.pack*, Nvfp4Experts).
pub const Packer = struct {
    mod: Module,
    fp8_lane: Function,
    bf16_lane: Function,
    fp4_words: Function,
    fp4_scales: Function,
    fp4_experts: Function,

    pub fn load(d: *const Driver) !Packer {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.modelopt_pack);
        errdefer mod.unload();
        return .{
            .mod = mod,
            .fp8_lane = try mod.function("tf_pack_fp8_lane"),
            .bf16_lane = try mod.function("tf_pack_bf16_lane"),
            .fp4_words = try mod.function("tf_pack_fp4_words"),
            .fp4_scales = try mod.function("tf_pack_fp4_scales"),
            .fp4_experts = try mod.function("tf_pack_fp4_experts"),
        };
    }

    pub fn unload(p: *Packer) void {
        p.mod.unload();
    }

    fn flat(f: Function, s: Stream, src: u64, out: u64, n: usize, k: usize, total: usize) !void {
        var a: launch_.Args = .{};
        a.add(src);
        a.add(out);
        a.add(@as(i32, @intCast(n)));
        a.add(@as(i32, @intCast(k)));
        a.add(@as(i64, @intCast(total)));
        try launch_.launch(f, .{ .grid = .{ .x = @intCast((total + 255) / 256) }, .block = .{ .x = 256 } }, s, &a);
    }

    /// e4m3 codes [n, k] at `src` into `out` (padded(n) * k bytes): a `.fp8` lane weight's codes.
    pub fn fp8Lane(p: *const Packer, s: Stream, src: u64, out: u64, n: usize, k: usize) !void {
        return flat(p.fp8_lane, s, src, out, n, k, padded(n) * k);
    }

    /// bf16 [n, k] at `src` into `out` (padded(n) * k * 2 bytes): a `.bf16` lane weight.
    pub fn bf16Lane(p: *const Packer, s: Stream, src: u64, out: u64, n: usize, k: usize) !void {
        return flat(p.bf16_lane, s, src, out, n, k, padded(n) * k);
    }

    /// NVFP4 codes [n, k/2] and e4m3 scales [n, k/16] into an `.nvfp4` lane weight's words and scale tiles.
    pub fn fp4Lane(p: *const Packer, s: Stream, codes: u64, scales: u64, words: u64, tiles: u64, n: usize, k: usize) !void {
        try flat(p.fp4_words, s, codes, words, n, k, padded(n) / 64 * (k / 64) * 512);
        try flat(p.fp4_scales, s, scales, tiles, n, k, padded(n) * (k / 16));
    }

    /// `count` experts' codes [E][n][ks/2] and scales [E][n][ks/16], inputs k0.., as part m of `parts` (Nvfp4Experts).
    pub fn fp4Experts(p: *const Packer, s: Stream, codes: u64, scales: u64, out: u64, count: usize, n: usize, ks: usize, k0: usize, k: usize, parts: usize, m: usize) !void {
        if (n % 32 != 0 or k % 32 != 0 or k0 % 16 != 0 or k0 + k > ks) return error.UnexpectedTensor;
        var a: launch_.Args = .{};
        a.add(codes);
        a.add(scales);
        a.add(out);
        for ([_]usize{ n, ks, k0, k, parts, m }) |v| a.add(@as(i32, @intCast(v)));
        try launch_.launch(p.fp4_experts, .{ .grid = .{ .x = @intCast(k / 32), .y = @intCast(n / 32), .z = @intCast(count) }, .block = .{ .x = 144 } }, s, &a);
    }
};

/// The lane matmul's view of a packed per-tensor FP8 projection.
pub fn fp8Weight(codes: u64, scale: f32, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .fp8, .codes = codes, .scales = 0, .scale = scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

/// The lane matmul's view of a packed bf16 projection.
pub fn bf16Weight(w: u64, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .bf16, .codes = w, .scales = 0, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(padded(n)) };
}

test "lane views carry their mode, padded rows and scale" {
    const f = fp8Weight(1, 0.5, 10304, 2688);
    try std.testing.expect(f.mode == .fp8 and f.npad == 10368 and f.scale == 0.5);
    const b = bf16Weight(1, 256, 2688);
    try std.testing.expect(b.mode == .bf16 and b.npad == 256 and b.scale == 1.0);
}
