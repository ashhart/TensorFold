//! Opt-in calibrated NVFP4 prompt rows; decode and verification retain their existing BF16 activations.
const std = @import("std");
const Driver = @import("driver.zig").Driver;
const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const Stream = @import("stream.zig").Stream;
const launch = @import("launch.zig");
const kernels = @import("kernels.zig");
const qmmf = @import("qmmf.zig");
const Tma = @import("nvfp4_tma.zig").Tma;

pub fn enabled(raw: ?[]const u8) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (std.mem.eql(u8, value, "1")) return true;
    return error.BadNativeNvfp4Prefill;
}

/// The native FP4 transport can be measured without changing affine/FP8 prompt arithmetic.
pub fn selectTma(raw: ?[]const u8, fast_prefill: bool) !bool {
    return if (raw) |value| enabled(value) else fast_prefill;
}

pub fn tmaRequested() !bool {
    const raw = try @import("sys").environment.given("TF_QWEN_NVFP4_TMA");
    return selectTma(raw, @import("fp8_prefill.zig").enabled());
}

pub fn validScale(scale: f32) bool {
    return std.math.isFinite(scale) and scale > 0 and std.math.isFinite(1.0 / scale);
}

pub fn supported(capability: u32) bool {
    return capability == 120 or capability == 121;
}

pub const Native = struct {
    mod: Module,
    quant: Function,
    bf16: Function,
    f32_out: Function,
    scratch: DeviceBuffer,
    rows: usize,
    width: usize,
    tma: ?Tma = null,
    prepared: [2]?DeviceBuffer = .{ null, null },
    workspace: ?DeviceBuffer = null,

    pub fn load(d: *const Driver, rows: usize, width: usize) !Native {
        if (rows == 0 or width == 0 or width % 64 != 0) return error.BadNativeNvfp4Shape;
        var mod = try Module.load(d, kernels.nvfp4_prompt);
        errdefer mod.unload();
        const quant = try mod.function("tf_nvfp4_quant");
        const bf16 = try mod.function("tf_nvfp4_prompt");
        const f32_out = try mod.function("tf_nvfp4_prompt_f32");
        const size = try std.math.mul(usize, rows, width);
        var tma = if (try tmaRequested()) try Tma.load(d) else null;
        errdefer if (tma) |*t| t.unload();
        return .{ .mod = mod, .quant = quant, .bf16 = bf16, .f32_out = f32_out, .scratch = try DeviceBuffer.alloc(d, size / 2 + size / 16), .rows = rows, .width = width, .tma = tma };
    }

    pub fn unload(n: *Native) void {
        for (&n.prepared) |*p| if (p.*) |*buffer| buffer.free();
        if (n.tma) |*t| t.unload();
        n.scratch.free();
        if (n.workspace) |*buffer| buffer.free();
        n.mod.unload();
    }

    pub fn reservePacked(n: *Native, bytes: usize) !void {
        if (n.tma == null or bytes == 0) return;
        if (n.workspace != null) return error.NativeNvfp4AlreadyReserved;
        const scratch_len = n.scratch.len;
        const first = (try std.math.add(usize, scratch_len, 255)) & ~@as(usize, 255);
        const second = (try std.math.add(usize, try std.math.add(usize, first, bytes), 255)) & ~@as(usize, 255);
        var buffer = try DeviceBuffer.alloc(n.mod.d, try std.math.add(usize, second, bytes));
        errdefer buffer.free();
        try n.mod.d.synchronize();
        n.scratch.free();
        n.scratch = .{ .d = n.mod.d, .ptr = buffer.ptr, .len = scratch_len, .borrowed = true };
        n.prepared[0] = .{ .d = n.mod.d, .ptr = buffer.ptr + first, .len = bytes, .borrowed = true };
        n.prepared[1] = .{ .d = n.mod.d, .ptr = buffer.ptr + second, .len = bytes, .borrowed = true };
        n.workspace = buffer;
    }

    fn quantize(n: *const Native, s: Stream, x: u64, ldx: usize, rows: usize, w: qmmf.Weight) !u64 {
        if (w.mode != .fp4 or !validScale(w.act)) return error.BadNativeNvfp4Scale;
        if (rows == 0 or rows > n.rows or w.k == 0 or w.k > n.width or w.k % 64 != 0 or
            w.n == 0 or w.npad < (w.n + 63) / 64 * 64 or ldx < w.k) return error.BadNativeNvfp4Shape;
        const scale_rows = n.scratch.ptr + rows * w.k / 2;
        var a: launch.Args = .{};
        a.add(x);
        a.add(n.scratch.ptr);
        a.add(scale_rows);
        for ([_]usize{ rows, w.k, ldx }) |value| a.add(@as(i32, @intCast(value)));
        a.add(@as(f32, 1.0 / w.act));
        try launch.launch(n.quant, .{
            .grid = .{ .x = @intCast((rows * (w.k / 16) + 255) / 256) },
            .block = .{ .x = 256 },
        }, s, &a);
        return scale_rows;
    }

    pub fn matmul(n: *const Native, s: Stream, x: u64, ldx: usize, rows: usize, w: qmmf.Weight, out: u64, kind: qmmf.Out) !void {
        const scale_rows = try n.quantize(s, x, ldx, rows, w);
        if (n.tma) |*t| if (Tma.supports(rows, w.k)) {
            var weight = w;
            const bytes = @as(usize, w.npad) * w.k / 2;
            const prepared = n.prepared[0] != null and n.prepared[0].?.len >= bytes;
            if (prepared) {
                weight.codes = n.prepared[0].?.ptr;
                try t.repack(s, w.codes, weight.codes, bytes);
            }
            return t.matmul(s, n.scratch.ptr, scale_rows, rows, weight, out, kind, prepared);
        };
        var b: launch.Args = .{};
        b.add(n.scratch.ptr);
        b.add(scale_rows);
        b.add(w.codes);
        b.add(w.scales);
        b.add(out);
        for ([_]usize{ rows, w.n, w.k, w.npad }) |value| b.add(@as(i32, @intCast(value)));
        b.add(@as(f32, w.act * w.scale));
        try launch.launch(if (kind == .f32) n.f32_out else n.bf16, .{
            .grid = .{ .x = @intCast((w.n + 127) / 128), .y = @intCast((rows + 63) / 64) },
            .block = .{ .x = 256 },
        }, s, &b);
    }

    pub fn swiglu(n: *const Native, s: Stream, x: u64, rows: usize, gate: qmmf.Weight, up: qmmf.Weight, out: u64) !bool {
        const t = if (n.tma) |*t| t else return false;
        if (!Tma.supports(rows, gate.k) or gate.n != up.n or gate.k != up.k or gate.act != up.act or gate.act <= 0)
            return false;
        const scales = try n.quantize(s, x, gate.k, rows, gate);
        var g = gate;
        var u = up;
        const bytes = @as(usize, gate.npad) * gate.k / 2;
        const prepared = n.prepared[0] != null and n.prepared[1] != null and n.prepared[0].?.len >= bytes;
        if (prepared) {
            g.codes = n.prepared[0].?.ptr;
            u.codes = n.prepared[1].?.ptr;
            try t.repack(s, gate.codes, g.codes, bytes);
            try t.repack(s, up.codes, u.codes, bytes);
        }
        try t.swiglu(s, n.scratch.ptr, scales, rows, g, u, out, prepared);
        return true;
    }
};

test "native NVFP4 prefill is explicit and Blackwell-only" {
    try std.testing.expect(!try enabled(null));
    try std.testing.expect(!try enabled("0"));
    try std.testing.expect(try enabled("1"));
    for ([_][]const u8{ "", "true", "2", "01" }) |v| try std.testing.expectError(error.BadNativeNvfp4Prefill, enabled(v));
    for ([_]u32{ 80, 89, 90, 100 }) |cap| try std.testing.expect(!supported(cap));
    try std.testing.expect(supported(120) and supported(121));
}

test "native NVFP4 calibration rejects invalid multipliers" {
    try std.testing.expect(validScale(0.001261393));
    for ([_]f32{ 0, -1, std.math.inf(f32), std.math.nan(f32), 1.0e-40 }) |scale|
        try std.testing.expect(!validScale(scale));
}

test "native FP4 transport selects independently without changing its default" {
    try std.testing.expect(!try selectTma(null, false));
    try std.testing.expect(try selectTma(null, true));
    try std.testing.expect(try selectTma("1", false));
    try std.testing.expect(!try selectTma("0", true));
    try std.testing.expectError(error.BadNativeNvfp4Prefill, selectTma("true", false));
}
