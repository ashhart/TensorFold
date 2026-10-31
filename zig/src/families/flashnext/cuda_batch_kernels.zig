//! Flash Next CUDA module ownership and optional expert staging are kept together.
const std = @import("std");
const cuda = @import("cuda");
const tk = @import("cuda_tile_kernels.zig");
const state = @import("cuda_batch_state.zig");
const cubin_qmm = state.cubin_qmm;
const qmm_smem = state.qmm_smem;
const chain_tpr_smem = state.chain_tpr_smem;
const gdn_fused_smem = state.gdn_fused_smem;
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
    /// BF16 weights from prompt_rows rows on prompt16.cu's tile, the lane's bits (TF_FLASHNEXT_PROMPT_BF16=0: lane).
    prompt: ?cuda.nvfp4.Prompt,
    prompt_rows: usize,
    nvfp4: cuda.experts.Nvfp4Experts,
    expert_tile: ?cuda.experts_tile.Tile = null,
    front: cuda.Function,
    chain: cuda.Function,
    chain_to: cuda.Function,
    /// TF_FLASHNEXT_KEEP_LATE: the front into a stream's pending-row log, and the chain with the pending rows.
    front_log: cuda.Function,
    chain_late: cuda.Function,
    /// A rolled-back lane's conv windows and n-gram tail in one launch (pending rows).
    late_roll: cuda.Function,
    chain_tpr: cuda.Function,
    /// front + chain + gated norm in one kernel a value head (prompt passes of 64 rows or more).
    gdn_fused: cuda.Function,
    gnorm: cuda.Function,
    sum: cuda.Function,
    moe_sum: cuda.Function,
    copy_cols: cuda.Function,
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
        k.prompt_rows = envRows("TF_FLASHNEXT_PROMPT_ROWS", 1024);
        k.prompt = if (envOn("TF_FLASHNEXT_PROMPT_BF16")) try cuda.nvfp4.Prompt.load(d) else null;
        errdefer if (k.prompt) |*p| p.unload();
        var device: c_int = 0;
        var sms: c_int = 0;
        try d.check(d.api.cuDeviceGet(&device, 0), "cuDeviceGet");
        try d.check(d.api.cuDeviceGetAttribute(&sms, .multiprocessor_count, device), "cuDeviceGetAttribute");
        k.nvfp4 = try cuda.experts.Nvfp4Experts.load(d, @intCast(sms));
        errdefer k.nvfp4.unload();
        if (tile_mode != .off) k.expert_tile = try cuda.experts_tile.Tile.load(d, @intCast(sms), tile_mode);
        errdefer if (k.expert_tile) |*tile| tile.unload();
        k.sum = try k.batch.function("fn_sum_slices");
        k.moe_sum = try k.batch.function("fn_moe_sum");
        k.copy_cols = try k.batch.function("fn_copy_cols");
        k.shift = try k.batch.function("fn_shift_window");
        k.select = try k.batch.function("fn_qsa_select");
        k.front = try k.batch.function("fn_gdn_front");
        k.chain = try k.batch.function("fn_gdn_rows");
        k.chain_to = try k.batch.function("fn_gdn_rows_to");
        k.front_log = try k.batch.function("fn_gdn_front_log");
        k.chain_late = try k.batch.function("fn_gdn_rows_late");
        k.late_roll = try k.batch.function("fn_late_roll");
        k.chain_tpr = try k.batch.function("fn_gdn_chain_tpr");
        try k.chain_tpr.allowDynamicShared(chain_tpr_smem);
        k.gdn_fused = try k.batch.function("fn_gdn_fused");
        try k.gdn_fused.allowDynamicShared(gdn_fused_smem);
        k.gnorm = try k.batch.function("fn_gdn_norm");
        return k;
    }

    /// A native projection over m rows: a BF16 weight takes the prompt tile from prompt_rows rows, the rest the lane.
    pub fn nativeMatmul(k: *const Kernels, s: cuda.Stream, x: u64, ldx: usize, m: usize, w: cuda.qmmf.Weight, y: u64, part: u64, out: cuda.qmmf.Out) !void {
        if (k.prompt) |p| if (m >= k.prompt_rows and w.mode == .bf16) return p.matmulOut(s, x, ldx, m, w, y, out);
        return k.native_lane.matmulOut(s, x, ldx, m, w, y, out, part);
    }

    /// TF_FLASHNEXT_PROFILE=1: each chunk's wall time (synchronized) and its host n-gram gather on stderr.
    pub fn profile() bool {
        const v = std.c.getenv("TF_FLASHNEXT_PROFILE") orelse return false;
        return std.mem.eql(u8, std.mem.span(v), "1");
    }

    pub fn deinit(k: *Kernels) void {
        if (k.expert_tile) |*tile| tile.unload();
        k.nvfp4.unload();
        if (k.prompt) |*p| p.unload();
        k.native_lane.unload();
        k.batch.unload();
        k.rows.unload();
        k.experts.unload();
        k.tiles.unload();
        k.qmm.unload();
    }
};

/// A switch on unless its variable is "0".
fn envOn(name: [*:0]const u8) bool {
    const v = std.c.getenv(name) orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// A row count from the environment, or `default`.
fn envRows(name: [*:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}
