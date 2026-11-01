//! Flash Next CUDA module ownership and optional expert staging are kept together.
const std = @import("std");
const cuda = @import("cuda");
const tk = @import("cuda_tile_kernels.zig");
const state = @import("cuda_batch_state.zig");
const cubin_qmm = state.cubin_qmm;
const qmm_smem = state.qmm_smem;
const chain_tpr_smem = state.chain_tpr_smem;
const gdn_fused_smem = state.gdn_fused_smem;
const Heads = @import("cuda_heads.zig").Heads;

/// The DeltaNet kernels of one head geometry: the full heads', or their `_h8` twins' (8 key and 24 value heads).
pub const Gdn = struct {
    front: cuda.Function,
    chain: cuda.Function,
    chain_to: cuda.Function,
    chain_tpr: cuda.Function,
    /// front + chain + gated norm in one kernel a value head (prompt passes of 64 rows or more).
    fused: cuda.Function,
    norm: cuda.Function,

    /// The entry point of `field` for the full heads (suffix "") or a twin set (suffix "_h8").
    pub fn symbol(comptime field: []const u8, comptime suffix: [:0]const u8) [:0]const u8 {
        const names = .{ .front = "fn_gdn_front", .chain = "fn_gdn_rows", .chain_to = "fn_gdn_rows_to", .chain_tpr = "fn_gdn_chain_tpr", .fused = "fn_gdn_fused", .norm = "fn_gdn_norm" };
        return @field(names, field) ++ suffix;
    }

    fn load(m: cuda.Module, comptime suffix: [:0]const u8) !Gdn {
        var g: Gdn = undefined;
        inline for (@typeInfo(Gdn).@"struct".field_names) |name| @field(g, name) = try m.function(symbol(name, suffix));
        try g.chain_tpr.allowDynamicShared(chain_tpr_smem);
        try g.fused.allowDynamicShared(gdn_fused_smem);
        return g;
    }
};

pub const Kernels = struct {
    qmm: cuda.Module,
    experts: cuda.Module,
    rows: cuda.Module,
    batch: cuda.Module,
    /// The core's expert plan and its rows kernels.
    router: cuda.grouped.Router,
    moe: cuda.experts.RowsExperts,
    lane: cuda.Function,
    /// The prompt tiles that keep #952's bits (cuda_tile_kernels.zig).
    tiles: tk.TileKernels,
    native_lane: cuda.qmmf.Lane,
    nvfp4: cuda.experts.Nvfp4Experts,
    /// Block-FP8 routed experts (the MTP head of NVIDIA's NVFP4 checkpoint; cuda_moe_native.zig).
    fp8: cuda.experts.Fp8Experts,
    expert_tile: ?cuda.experts_tile.Tile = null,
    /// The DeltaNet kernels for the whole layer's heads and for a rank's half under the head split (gdnSet).
    gdn: Gdn,
    gdn_h8: Gdn,
    sum: cuda.Function,
    moe_sum: cuda.Function,
    copy_cols: cuda.Function,
    ep_picks: cuda.Function,
    shift: cuda.Function,
    /// Each sparse row's top blocks (attention._select / _select_tiles), native.
    select: cuda.Function,

    pub fn load(d: *const cuda.Driver) !Kernels {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const tile_mode = try cuda.experts_tile.mode(if (std.c.getenv("TF_FLASHNEXT_PROMPT_EXPERT_TILE")) |v| std.mem.span(v) else null);
        var k: Kernels = undefined;
        k.expert_tile = null;
        k.qmm = try cuda.Module.load(d, cuda.kernels.qmm);
        errdefer k.qmm.unload();
        k.experts = try cuda.Module.load(d, cuda.kernels.experts);
        errdefer k.experts.unload();
        k.rows = try cuda.Module.load(d, cuda.kernels.experts_rows);
        errdefer k.rows.unload();
        k.batch = try cuda.Module.load(d, cuda.kernels.flashnext_batch);
        errdefer k.batch.unload();
        k.lane = try k.qmm.function(cubin_qmm);
        try k.lane.allowDynamicShared(qmm_smem);
        k.tiles = try tk.TileKernels.load(d);
        errdefer k.tiles.unload();
        k.router = try cuda.grouped.Router.resolve(k.experts);
        k.moe = try cuda.experts.RowsExperts.resolve(k.rows);
        k.native_lane = try cuda.qmmf.Lane.load(d, (try d.capability(0)) / 10);
        errdefer k.native_lane.unload();
        var device: c_int = 0;
        var sms: c_int = 0;
        try d.check(d.api.cuDeviceGet(&device, 0), "cuDeviceGet");
        try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, device), "cuDeviceGetAttribute");
        k.nvfp4 = try cuda.experts.Nvfp4Experts.load(d, @intCast(sms));
        errdefer k.nvfp4.unload();
        k.fp8 = try cuda.experts.Fp8Experts.load(d, @intCast(sms));
        errdefer k.fp8.unload();
        if (tile_mode != .off) k.expert_tile = try cuda.experts_tile.Tile.load(d, @intCast(sms), tile_mode);
        errdefer if (k.expert_tile) |*tile| tile.unload();
        k.sum = try k.batch.function("fn_sum_slices");
        k.moe_sum = try k.batch.function("fn_moe_sum");
        k.copy_cols = try k.batch.function("fn_copy_cols");
        k.ep_picks = try k.batch.function("fn_ep_picks");
        k.shift = try k.batch.function("fn_shift_window");
        k.select = try k.batch.function("fn_qsa_select");
        k.gdn = try Gdn.load(k.batch, "");
        k.gdn_h8 = try Gdn.load(k.batch, "_h8");
        return k;
    }

    /// The DeltaNet kernels for a layer's heads `h`: the full set, or the `_h8` twins for half; any other is refused.
    pub fn gdnSet(k: *const Kernels, h: Heads) error{RowShape}!*const Gdn {
        if (h.eql(.full)) return &k.gdn;
        if (h.eql(.half)) return &k.gdn_h8;
        return error.RowShape;
    }

    pub fn deinit(k: *Kernels) void {
        if (k.expert_tile) |*tile| tile.unload();
        k.fp8.unload();
        k.nvfp4.unload();
        k.native_lane.unload();
        k.batch.unload();
        k.rows.unload();
        k.experts.unload();
        k.tiles.unload();
        k.qmm.unload();
    }
};

test "a split layer's DeltaNet launches, the rollback replay's front and chain too, take the _h8 twins; a whole one today's" {
    const want = [_][:0]const u8{ "fn_gdn_front_h8", "fn_gdn_rows_h8", "fn_gdn_rows_to_h8", "fn_gdn_chain_tpr_h8", "fn_gdn_fused_h8", "fn_gdn_norm_h8" };
    const names = @typeInfo(Gdn).@"struct".field_names;
    inline for (names, want) |field, name| {
        try std.testing.expectEqualStrings(name, Gdn.symbol(field, "_h8"));
        try std.testing.expectEqualStrings(name[0 .. name.len - 3], Gdn.symbol(field, ""));
    }
    var driver: cuda.Driver = undefined;
    var k: Kernels = undefined;
    inline for (names, 0..) |field, i| {
        @field(k.gdn, field) = .{ .d = &driver, .handle = @ptrFromInt(0x100 + i) };
        @field(k.gdn_h8, field) = .{ .d = &driver, .handle = @ptrFromInt(0x200 + i) };
    }
    const half = try k.gdnSet(.half);
    const full = try k.gdnSet(.full);
    try std.testing.expectEqual(k.gdn_h8.front.handle, half.front.handle);
    try std.testing.expectEqual(k.gdn_h8.chain.handle, half.chain.handle);
    try std.testing.expectEqual(k.gdn.front.handle, full.front.handle);
    try std.testing.expectEqual(k.gdn.chain.handle, full.chain.handle);
    inline for (names) |field| try std.testing.expect(@field(half, field).handle != @field(full, field).handle);
    try std.testing.expectError(error.RowShape, k.gdnSet(.{ .nk = 4, .nv = 12, .q = 6, .kv = 1 }));
}
