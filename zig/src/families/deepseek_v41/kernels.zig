//! DeepSeek-V4.1's CUDA kernels (zig/kernels/cuda/deepseek_v41) as embedded fatbins, and their launches.

const std = @import("std");
const cuda = @import("cuda");
const options = @import("dsv41_kernel_options");

pub const exl3 = @import("cuda").qlinear.block.experts;
pub const dense = @import("cuda").qlinear.block.dense;
pub const kv_glue = @import("kernels_kv_glue.zig");
pub const replay = @import("kernels_replay.zig");
pub const ops = @import("kernels_ops.zig");
/// sampling.cu (its own module, loaded by the keyed sampler)
pub const sampling = @import("sampling_kernels.zig");

fn Blob(comptime import_name: []const u8) type {
    return struct {
        pub const bytes align(16) = @embedFile(import_name).*;
    };
}

/// False in host-only builds (no nvcc, no prebuilt fatbins); every image is then empty.
pub const available = options.with_kernels;

/// The fatbins, by the names zig/build/cuda.zig gives them (dsv41_<file>).
pub const images = struct {
    pub const exl3_experts: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_exl3_experts").bytes else &.{};
    pub const x3ld: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3ld").bytes else &.{};
    pub const x3pf: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3pf").bytes else &.{};
    pub const x3gm: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3gm").bytes else &.{};
    pub const linear: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_linear").bytes else &.{};
    pub const x3seg: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3seg").bytes else &.{};
    pub const dense3: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_dense3").bytes else &.{};
    pub const attn: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_attn").bytes else &.{};
    pub const topk: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_topk").bytes else &.{};
    pub const mhc: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_mhc").bytes else &.{};
    pub const mhc_pf: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_mhc_pf").bytes else &.{};
    pub const router_gemv: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_router_gemv").bytes else &.{};
    pub const pfdense_k8: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k8").bytes else &.{};
    pub const pfdense_k10: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k10").bytes else &.{};
    pub const pfdense_k12: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k12").bytes else &.{};
    pub const pfdense_k16: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfdense_k16").bytes else &.{};
    pub const engram_gate: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_engram_gate").bytes else &.{};
    pub const l2pace: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_l2pace").bytes else &.{};
    pub const l2pf: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_l2pf").bytes else &.{};
    pub const x3gm_plan: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3gm_plan").bytes else &.{};
    pub const topk_keys: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_topk_keys").bytes else &.{};
    pub const pointwise: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pointwise").bytes else &.{};
    pub const kvsplit: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_kvsplit").bytes else &.{};
    pub const glue: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_glue").bytes else &.{};
    pub const x3gm3: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3gm3").bytes else &.{};
    pub const pfglue: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_pfglue").bytes else &.{};
    pub const kv_glue: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_kv_glue").bytes else &.{};
    pub const router_glue: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_router_glue").bytes else &.{};
    pub const x3ld_epi: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_x3ld_epi").bytes else &.{};
    pub const topk_b: []const u8 = if (available) &Blob("dsv41_fatbin_dsv41_topk_b").bytes else &.{};
};

/// lin2.cu (ours, TF_DSV41_LIN2): optional (zig/build/cuda.zig), an empty image when the build has none.
const Lin2Blob = struct {
    pub const bytes align(16) = @embedFile("dsv41_fatbin_dsv41_lin2").*;
};
pub const lin2_image: []const u8 = &Lin2Blob.bytes;

fn env(name: [:0]const u8) ?[]const u8 {
    const v = std.c.getenv(name) orelse return null;
    return std.mem.span(v);
}

pub const Kernels = struct {
    mods: [30]cuda.Module,
    kv_glue: kv_glue.Functions,
    exl3: exl3.Functions,
    dense: dense.Functions,
    ops: ops.Functions,
    sms: usize,

    /// Loads every module and resolves every instance the launches can use.
    pub fn load(ctx: *const cuda.Context) !Kernels {
        if (try ctx.capability() < 90) return error.DeepSeekRequiresSm90;
        if (!available) return error.BuiltWithoutKernels;
        const d = ctx.d;
        var k: Kernels = undefined;
        const list = [_][]const u8{ images.exl3_experts, images.x3ld, images.x3pf, images.x3gm, images.linear, images.x3seg, images.dense3, images.attn, images.topk, images.mhc, images.mhc_pf, images.router_gemv, images.pfdense_k8, images.pfdense_k10, images.pfdense_k12, images.pfdense_k16, images.engram_gate, images.l2pace, images.l2pf, images.x3gm_plan, images.topk_keys, images.pointwise, images.kvsplit, images.glue, images.x3gm3, images.pfglue, images.router_glue, images.kv_glue, images.x3ld_epi, images.topk_b };
        comptime std.debug.assert(list.len == @typeInfo(@FieldType(Kernels, "mods")).array.len);
        var loaded: usize = 0;
        errdefer for (k.mods[0..loaded]) |*m| m.unload();
        for (list, 0..) |img, i| {
            k.mods[i] = try cuda.Module.load(d, img);
            loaded += 1;
        }
        // the per-block shared memory opt-in (sm_120 / sm_121: 101,376 B): instances above it are refused one by one
        const optin: u32 = @intCast(try ctx.attribute(.max_shared_memory_per_block_optin));
        k.kv_glue = try kv_glue.Functions.resolve(k.mods[27]);
        k.exl3 = try exl3.Functions.resolve(k.mods[0], k.mods[1], k.mods[2], k.mods[3], k.mods[19], k.mods[24], k.mods[26], k.mods[28], optin);
        k.dense = try dense.Functions.resolve(k.mods[4], k.mods[5], k.mods[6]);
        const m = k.mods;
        k.ops = try ops.Functions.resolve(m[7], m[8], m[9], m[10], m[11], .{ m[12], m[13], m[14], m[15] }, m[16], m[17], m[18], m[20], m[21], m[22], m[23], m[25], m[29], optin);
        k.sms = @intCast(try ctx.attribute(.multiprocessor_count));
        try k.loadLin2(d, blk: {
            var lm = try dense.lin2ModeWk(env("TF_DSV41_LIN2"), env("TF_DSV41_LIN2_PF"), env("TF_DSV41_LIN2_WK"));
            lm.cluster = try dense.lin2Cluster(env("TF_DSV41_LIN2_CLUSTER"));
            break :blk lm;
        });
        return k;
    }

    /// TF_DSV41_LIN2: lin2.cu's module and instances (dense.Ops); refused when the build has no lin2 fatbin.
    pub fn loadLin2(k: *Kernels, d: *const cuda.Driver, mode: dense.Lin2Mode) !void {
        if (!mode.lanes and !mode.linear and !mode.cluster) return;
        if (lin2_image.len == 0) return error.Lin2KernelNotBuilt;
        var m = try cuda.Module.load(d, lin2_image);
        errdefer m.unload();
        k.dense.lin2 = try dense.Lin2.resolve(m, mode);
        std.log.scoped(.dsv41).info("lin2: 17-32-row {s} launches as two-tile variants (one pass over the weights){s}{s}", .{ if (mode.lanes and mode.linear) "dense3 and linear.cu" else if (mode.lanes) "dense3" else "linear.cu", if (mode.linear and !mode.pf) ", linear2 without the k-step prefetch" else "", if (mode.lanes and !mode.wk8) ", lanes2 at 4-warp CTAs only" else if (mode.lanes and !mode.wk4) ", lanes2 at 8-warp CTAs only" else "" });
        if (mode.cluster) std.log.scoped(.dsv41).info("lin2: dense3's split-K launches of up to 32 rows reduce their partials in a cluster's shared memory (TF_DSV41_LIN2_CLUSTER)", .{});
    }

    pub fn deinit(k: *Kernels) void {
        if (k.dense.lin2) |*l| l.module.unload();
        for (&k.mods) |*m| m.unload();
    }

    /// The routed-expert launches on one stream.
    pub fn experts(k: *const Kernels, s: cuda.Stream) exl3.Ops {
        return .{ .f = &k.exl3, .s = s, .sms = k.sms };
    }

    /// Attention, top-k, mHC, router, pfdense, Engram gate and L2 launches on one stream.
    pub fn others(k: *const Kernels, s: cuda.Stream) ops.Ops {
        return .{ .f = &k.ops, .s = s };
    }

    pub fn kvStore(k: *const Kernels, s: cuda.Stream) kv_glue.Ops {
        return .{ .f = &k.kv_glue, .stream = s };
    }

    /// The dense-linear launches on one stream.
    pub fn linears(k: *const Kernels, s: cuda.Stream) dense.Ops {
        return .{ .f = &k.dense, .s = s };
    }
};

test {
    std.testing.refAllDecls(kv_glue);
    std.testing.refAllDecls(kv_glue.Ops);
    std.testing.refAllDecls(exl3);
    std.testing.refAllDecls(exl3.Ops);
    std.testing.refAllDecls(exl3.Functions);
    std.testing.refAllDecls(dense);
    std.testing.refAllDecls(dense.Ops);
    std.testing.refAllDecls(dense.Functions);
    std.testing.refAllDecls(Kernels);
    std.testing.refAllDecls(replay);
    std.testing.refAllDecls(ops);
    std.testing.refAllDecls(ops.Ops);
    std.testing.refAllDecls(ops.Functions);
}

test {
    _ = ops;
    _ = replay;
    _ = dense;
    _ = exl3;
    _ = kv_glue;
    _ = @import("kernels_shapes.zig");
}

test "DeepSeek refuses pre-cluster GPUs before loading any image" {
    const abi = cuda.abi;
    const Fake = struct {
        var sm: u32 = 0;
        var loads: usize = 0;
        var failed: bool = false;
        fn attribute(out: *c_int, key: abi.DeviceAttribute, _: abi.Device) callconv(.c) abi.Result {
            if (failed) return abi.error_not_ready;
            out.* = switch (key) {
                .compute_capability_major => @intCast(sm / 10),
                .compute_capability_minor => @intCast(sm % 10),
                else => unreachable,
            };
            return abi.success;
        }
        fn shared(_: abi.Function, _: abi.FunctionAttribute, _: c_int) callconv(.c) abi.Result {
            return abi.error_not_ready;
        }
        fn load(_: *abi.Module, _: ?*const anyopaque) callconv(.c) abi.Result {
            loads += 1;
            return abi.error_not_ready;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuDeviceGetAttribute = Fake.attribute;
    driver.api.cuModuleLoadData = Fake.load;
    driver.api.cuFuncSetAttribute = Fake.shared;
    const ctx: cuda.Context = .{ .d = &driver, .device = 0, .handle = null };
    Fake.loads = 0;
    Fake.failed = false;
    for ([_]u32{ 80, 86, 89 }) |sm| {
        Fake.sm = sm;
        try std.testing.expectError(error.DeepSeekRequiresSm90, Kernels.load(&ctx));
        try std.testing.expectEqual(@as(usize, 0), Fake.loads);
    }
    for ([_]u32{ 90, 100, 120, 121 }) |sm| {
        Fake.sm = sm;
        const before = Fake.loads;
        try std.testing.expectError(if (available) error.NotReady else error.BuiltWithoutKernels, Kernels.load(&ctx));
        try std.testing.expectEqual(before + @intFromBool(available), Fake.loads);
    }
    const group: cuda.Function = .{ .d = &driver, .handle = @ptrFromInt(1) };
    try std.testing.expectError(error.NotReady, group.allowDynamicShared(48 * 1024));
    Fake.failed = true;
    const before = Fake.loads;
    try std.testing.expectError(error.NotReady, Kernels.load(&ctx));
    try std.testing.expectEqual(before, Fake.loads);
}
