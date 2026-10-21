//! Dense NVFP4 x NVFP4 projections in the checkpoint's math (ck_dense4.cu): quant4's rows times tf_ck_pack4's tiles.
const std = @import("std");
const driver = @import("driver.zig");
const module = @import("module.zig");
const launch_ = @import("launch.zig");
const stream_ = @import("stream.zig");
const kernels = @import("kernels.zig");
const qmmf = @import("qmmf.zig");
const ck = @import("ck.zig");

const Driver = driver.Driver;
const Module = module.Module;
const Function = module.Function;
const Stream = stream_.Stream;

/// Rows from which a call takes the staged prompt kernel (the same bits as the warp kernel at every row count).
pub const prompt_rows = 17;

/// Row tiles a band of the staged kernel: ~4 MiB of NVFP4 rows, so its weight tiles stay in L2 (GB10: 2x the TFLOPS).
pub fn band(m: usize, k: usize) usize {
    const rows_t = (m + 127) / 128;
    return @max(1, @min(rows_t, (4 << 20) / (128 * k / 2)));
}

pub const Dense4 = struct {
    mod: Module,
    warp: Function,
    prompt: Function,
    resident: [2]usize, // blocks each kernel fills: per SM times SMs

    pub fn load(d: *const Driver, sms: usize) !Dense4 {
        if (!kernels.available) return error.BuiltWithoutKernels;
        var mod = try Module.load(d, kernels.ck_dense4);
        errdefer mod.unload();
        const warp = try mod.function("tf_ck4_dense");
        const prompt = try mod.function("tf_ck4_dense_prompt");
        return .{ .mod = mod, .warp = warp, .prompt = prompt, .resident = .{ @max(1, try warp.occupancy(128, 0)) * sms, @max(1, try prompt.occupancy(256, 0)) * sms } };
    }

    pub fn unload(k: *Dense4) void {
        k.mod.unload();
    }

    /// out [m, n] bf16 = alpha * (quant4's rows `x4` times ck-packed `w`); form 0 warp, 1 staged, null by rows.
    pub fn matmulForm(k: *const Dense4, s: Stream, x4: u64, m: usize, w: qmmf.Weight, out: u64, form: ?u1) !void {
        if (w.mode != .fp4 or w.n % 64 != 0 or w.k % 64 != 0) return error.FormatMismatch;
        if (m == 0) return;
        const staged = (form orelse @intFromBool(m >= prompt_rows)) == 1;
        const units = if (staged) (m + 127) / 128 * ((w.n / 64 + 1) / 2) else (m + 15) / 16 * (w.n / 32);
        const grid = @min(if (staged) units else (units + 3) / 4, k.resident[@intFromBool(staged)]);
        var a: launch_.Args = .{};
        a.add(x4);
        a.add(x4 + m * w.k / 2);
        a.add(w.codes);
        a.add(w.scales);
        a.add(ck.alpha(w.act, w.scale));
        for ([_]usize{ m, w.k, w.n }) |v| a.add(@as(i32, @intCast(v)));
        a.add(out);
        if (staged) a.add(@as(i32, @intCast(band(m, w.k))));
        try launch_.launch(if (staged) k.prompt else k.warp, .{ .grid = .{ .x = @intCast(grid) }, .block = .{ .x = if (staged) 256 else 128 } }, s, &a);
    }

    pub fn matmul(k: *const Dense4, s: Stream, x4: u64, m: usize, w: qmmf.Weight, out: u64) !void {
        return k.matmulForm(s, x4, m, w, out, null);
    }
};

/// The lane view of an NVFP4 projection packed for the checkpoint's math (tf_ck_pack4), under its input scale.
pub fn weight(words: u64, tiles: u64, global_scale: f32, act: f32, n: usize, k: usize) qmmf.Weight {
    return .{ .mode = .fp4, .codes = words, .scales = tiles, .scale = global_scale, .n = @intCast(n), .k = @intCast(k), .npad = @intCast(n), .act = act };
}

test "bands hold about 4 MiB of rows" {
    try std.testing.expectEqual(@as(usize, 12), band(4096, 5120));
    try std.testing.expectEqual(@as(usize, 3), band(4096, 17408));
    try std.testing.expectEqual(@as(usize, 1), band(100, 5120));
}

test "the checkpoint view keeps the input scale and pads nothing" {
    const w = weight(1, 2, 0.5, 0.25, 5120, 17408);
    try std.testing.expect(w.mode == .fp4 and w.npad == 5120 and w.act == 0.25);
    try std.testing.expectEqual(@as(f32, 0.125), ck.alpha(w.act, w.scale));
}
