//! DeepSeek V4.1 opt-in FP64-certified BF16 routed experts on Apple M5 Ultra Metal 4.1 tensor units.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const ex = @import("experts.zig");
const fp8_cert = @import("fp8_mma_cert.zig");

pub const Ref = ex.Ref;
pub const Route = ex.Route;
pub const ExpertPtrs = ex.ExpertPtrs;
pub const use = ex.use;
pub const topk = ex.topk;
pub const row_tile = 32;

const XmArgs = extern struct { K: u32, N: u32, topk: u32, pairs: u32, limit: f32 };
const XuPatchArgs = extern struct { patch_cap: u32, ug_cap: u32 };
const DsUpArgs = extern struct { K: u32, N: u32, topk: u32, rows_max: u32, limit: f32 };
const DsDownArgs = extern struct { K: u32, N: u32, topk: u32, rows_max: u32, beta: f32 = 0, lin: f32 = 0 };

pub const Call = struct { rows: u32, k: u32 = ex.hidden, n: u32 = ex.width, experts: u32 = ex.experts };

const patch_cap_up: u32 = 1_048_576;
const patch_cap_dn: u32 = 2_097_152;
const ug_cap_rows: u32 = 65_536;
const patch_vals_bytes: usize = 512 << 20;
const patch_meta_bytes: usize = @as(usize, patch_cap_dn) * 16;
const unsafe_mask_bytes: usize = @as(usize, ug_cap_rows) * 32;

/// Whether this Mac satisfies the strict Apple M5 Ultra + Metal 4.1 gate.
pub fn supported(device: mtl.Device) bool {
    return fp8_cert.supported(device);
}

/// Most 32-row tiles a chunk can make: full tiles, plus one partial per active expert.
pub fn maxTiles(c: Call) u32 {
    const pairs = c.rows * topk;
    return pairs / row_tile + @min(c.experts, pairs);
}

/// Scratch bytes by buffer: tiles, codes (half), scales, act, out, and down's halves (+32 row pad) + scales.
pub fn bytes(c: Call) [6]usize {
    const pairs = @as(usize, c.rows) * topk;
    const codes = @as(usize, c.rows) * c.k * 2;
    const dq = 2 * (pairs + 32) * c.n + 4 * pairs * c.n / 32;
    return .{ 16 * (1 + maxTiles(c)), codes, 4 * @as(usize, c.rows) * c.k / 32, 2 * (pairs * c.n + 1), 2 * (pairs * c.k + 1), dq };
}

pub fn check(c: Call) !void {
    if (c.rows == 0 or c.k % 32 != 0 or c.k == 0 or c.k > 8192 or c.n % 32 != 0 or c.n == 0) return error.BadExpertShape;
}

/// Decode certified expert source (experts_kimi_cert + experts_cert), with optional header (TF_CTL).
pub fn decodeSource(a: std.mem.Allocator, head: []const u8) ![]u8 {
    const body = try std.mem.replaceOwned(u8, a, sources.dsv41.experts_cert, "#include \"experts_kimi_cert.metal\"\n", sources.dsv41.experts_kimi_cert);
    defer a.free(body);
    return std.mem.concat(a, u8, &.{ head, body });
}

/// Full M5 Ultra Metal 4.1 tensor-unit certified expert source (decode + Pass 1 + Pass 2 patch).
pub fn source(a: std.mem.Allocator, head: []const u8) ![]u8 {
    const dec = try decodeSource(a, head);
    defer a.free(dec);
    return std.mem.concat(a, u8, &.{ dec, "\n", sources.dsv41.xunits_cert, "\n", sources.dsv41.xunits_cert_patch });
}

const State = struct {
    a: std.mem.Allocator,
    patch_meta: mtl.Buffer,
    patch_vals: mtl.Buffer,
    patch_cnt: mtl.Buffer,
    unsafe_mask: mtl.Buffer,
    ind: mtl.Buffer,
    one_slots: mtl.Buffer,
    multi_pairs: mtl.Buffer,

    fn init(a: std.mem.Allocator, device: mtl.Device) !*State {
        const st = try a.create(State);
        errdefer a.destroy(st);
        const opt = mtl.ResourceOptions.shared;
        const patch_meta = try device.buffer(patch_meta_bytes, opt);
        errdefer patch_meta.deinit();
        const patch_vals = try device.buffer(patch_vals_bytes, opt);
        errdefer patch_vals.deinit();
        const patch_cnt = try device.buffer(64, opt);
        errdefer patch_cnt.deinit();
        const unsafe_mask = try device.buffer(unsafe_mask_bytes, opt);
        errdefer unsafe_mask.deinit();
        const ind = try device.buffer(256, opt);
        errdefer ind.deinit();
        const one_slots = try device.buffer(4096, opt);
        errdefer one_slots.deinit();
        const multi_pairs = try device.buffer(1024, opt);
        st.* = .{
            .a = a,
            .patch_meta = patch_meta,
            .patch_vals = patch_vals,
            .patch_cnt = patch_cnt,
            .unsafe_mask = unsafe_mask,
            .ind = ind,
            .one_slots = one_slots,
            .multi_pairs = multi_pairs,
        };
        return st;
    }

    fn deinit(st: *State) void {
        st.patch_meta.deinit();
        st.patch_vals.deinit();
        st.patch_cnt.deinit();
        st.unsafe_mask.deinit();
        st.ind.deinit();
        st.one_slots.deinit();
        st.multi_pairs.deinit();
        st.a.destroy(st);
    }
};

pub const Pipelines = struct {
    tiles: mtl.Pipeline,
    tiles_hybrid: mtl.Pipeline,
    quant: mtl.Pipeline,
    up: mtl.Pipeline,
    upc_one_xu: mtl.Pipeline,
    dprep: mtl.Pipeline,
    dprep_hybrid: mtl.Pipeline,
    down: mtl.Pipeline,
    down_one: mtl.Pipeline,
    zero_cnt: mtl.Pipeline,
    patch_up: mtl.Pipeline,
    patch_down: mtl.Pipeline,
    st: *State,

    pub fn load(a: std.mem.Allocator, device: mtl.Device) !Pipelines {
        const text = try source(a, "");
        defer a.free(text);
        return loadSource(a, device, text);
    }

    pub fn loadSource(a: std.mem.Allocator, device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const st = try State.init(a, device);
        errdefer st.deinit();
        var tiles_p = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_tiles", false);
        errdefer tiles_p.deinit();
        var tiles_hyb = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_tiles_hybrid", false);
        errdefer tiles_hyb.deinit();
        var quant_p = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_quant_ug", false);
        errdefer quant_p.deinit();
        var up_p = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_up", false);
        errdefer up_p.deinit();
        var upc_one_xu = try mtl.Pipeline.init(device, lib, "tf_dsv41_xp_upc_one_xu", false);
        errdefer upc_one_xu.deinit();
        var dprep_p = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_dprep_ug", false);
        errdefer dprep_p.deinit();
        var dprep_hyb = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_dprep_ug_hybrid", false);
        errdefer dprep_hyb.deinit();
        var down_p = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_down", false);
        errdefer down_p.deinit();
        var down_one = try mtl.Pipeline.init(device, lib, "tf_dsv41_xp_down_one", false);
        errdefer down_one.deinit();
        var zero_p = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_zero_cnt", false);
        errdefer zero_p.deinit();
        var pup = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_patch_up", false);
        errdefer pup.deinit();
        const pdn = try mtl.Pipeline.init(device, lib, "tf_dsv41_xu_patch_down", false);
        return .{
            .tiles = tiles_p,
            .tiles_hybrid = tiles_hyb,
            .quant = quant_p,
            .up = up_p,
            .upc_one_xu = upc_one_xu,
            .dprep = dprep_p,
            .dprep_hybrid = dprep_hyb,
            .down = down_p,
            .down_one = down_one,
            .zero_cnt = zero_p,
            .patch_up = pup,
            .patch_down = pdn,
            .st = st,
        };
    }

    /// Small indirect and compaction buffers held by State (for joining the target's residency set).
    pub fn hybridBufs(p: *const Pipelines) [4]mtl.Buffer {
        return .{ p.st.patch_cnt, p.st.ind, p.st.one_slots, p.st.multi_pairs };
    }

    pub fn deinit(p: *Pipelines) void {
        p.tiles.deinit();
        p.tiles_hybrid.deinit();
        p.quant.deinit();
        p.up.deinit();
        p.upc_one_xu.deinit();
        p.dprep.deinit();
        p.dprep_hybrid.deinit();
        p.down.deinit();
        p.down_one.deinit();
        p.zero_cnt.deinit();
        p.patch_up.deinit();
        p.patch_down.deinit();
        p.st.deinit();
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref) void {
    for (refs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
}

pub fn tiles(p: *const Pipelines, e: mtl.ComputeEncoder, r: Route, table: Ref) void {
    e.setPipeline(p.tiles);
    bind(e, &.{ r.slots, r.nslots, table });
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
}

pub fn quant(p: *const Pipelines, e: mtl.ComputeEncoder, x: Ref, codes: Ref, scales: Ref, c: Call) void {
    e.setPipeline(p.quant);
    bind(e, &.{ x, codes, scales });
    e.setValue([3]u32{ c.k, c.rows, ug_cap_rows }, 3);
    e.setBuffer(p.st.unsafe_mask, 0, 4);
    e.dispatchGroups(mtl.Size.of(1, c.rows, 1), mtl.Size.of(128, 1, 1));
}

pub fn up(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, table: Ref, r: Route, tile_table: Ref, act: Ref, c: Call) void {
    const args = XmArgs{ .K = c.k, .N = c.n, .topk = topk, .pairs = c.rows * topk, .limit = ex.swiglu_limit };
    e.setPipeline(p.zero_cnt);
    e.setBuffer(p.st.patch_cnt, 0, 0);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
    e.barrier();

    const pa = XuPatchArgs{ .patch_cap = patch_cap_up, .ug_cap = ug_cap_rows };
    e.setPipeline(p.up);
    bind(e, &.{ codes, scales, table, tile_table, r.pairs, act });
    e.setValue(args, 6);
    e.setBuffer(r.weights.buf, r.weights.off, 7);
    e.setBuffer(p.st.patch_meta, 0, 8);
    e.setBuffer(p.st.patch_vals, 0, 9);
    e.setBuffer(p.st.patch_cnt, 0, 10);
    e.setValue(pa, 11);
    e.dispatchGroups(mtl.Size.of(c.n / 32, maxTiles(c), 1), mtl.Size.of(128, 1, 1));
    e.barrier();

    e.setPipeline(p.patch_up);
    bind(e, &.{ codes, scales, table, r.pairs, r.weights, tile_table });
    e.setBuffer(p.st.patch_meta, 0, 6);
    e.setBuffer(p.st.patch_vals, 0, 7);
    e.setBuffer(p.st.patch_cnt, 0, 8);
    e.setBuffer(act.buf, act.off, 9);
    e.setValue(args, 10);
    e.setValue(pa, 11);
    e.setBuffer(p.st.unsafe_mask, 0, 12);
    e.dispatchIndirect(p.st.patch_cnt, 16, mtl.Size.of(128, 1, 1));
}

pub fn down(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, dq: Ref, table: Ref, tile_table: Ref, out: Ref, c: Call) void {
    const pairs = c.rows * topk;
    const args = XmArgs{ .K = c.n, .N = c.k, .topk = topk, .pairs = pairs, .limit = 0 };
    const ds: Ref = .{ .buf = dq.buf, .off = dq.off + 2 * @as(usize, pairs + 32) * c.n };
    e.setPipeline(p.dprep);
    bind(e, &.{ act, dq, ds });
    e.setValue([3]u32{ c.n, pairs, ug_cap_rows }, 3);
    e.setBuffer(p.st.unsafe_mask, 0, 4);
    e.dispatchGroups(mtl.Size.of(1, pairs, 1), mtl.Size.of(256, 1, 1));
    e.setPipeline(p.zero_cnt);
    e.setBuffer(p.st.patch_cnt, 0, 0);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
    e.barrier();

    const pa = XuPatchArgs{ .patch_cap = patch_cap_dn, .ug_cap = ug_cap_rows };
    e.setPipeline(p.down);
    bind(e, &.{ act, table, tile_table, out });
    e.setValue(args, 4);
    e.setBuffer(dq.buf, dq.off, 5);
    e.setBuffer(ds.buf, ds.off, 6);
    e.setBuffer(p.st.patch_meta, 0, 7);
    e.setBuffer(p.st.patch_vals, 0, 8);
    e.setBuffer(p.st.patch_cnt, 0, 9);
    e.setValue(pa, 10);
    e.dispatchGroups(mtl.Size.of(c.k / 32, maxTiles(c), 1), mtl.Size.of(64, 1, 1));
    e.barrier();

    e.setPipeline(p.patch_down);
    bind(e, &.{ act, table, tile_table, ds });
    e.setBuffer(p.st.patch_meta, 0, 4);
    e.setBuffer(p.st.patch_vals, 0, 5);
    e.setBuffer(p.st.patch_cnt, 0, 6);
    e.setBuffer(out.buf, out.off, 7);
    e.setValue(args, 8);
    e.setValue(pa, 9);
    e.setBuffer(p.st.unsafe_mask, 0, 10);
    e.dispatchIndirect(p.st.patch_cnt, 16, mtl.Size.of(128, 1, 1));
}

/// Hybrid decode tile splitter: count == 1 into one_slots, count >= 2 into tiles + multi_pairs, resets patch_cnt.
pub fn tilesHybrid(p: *const Pipelines, e: mtl.ComputeEncoder, r: Route, tile_table: Ref, c: Call) void {
    e.setPipeline(p.tiles_hybrid);
    bind(e, &.{ r.slots, r.nslots, tile_table });
    e.setBuffer(p.st.one_slots, 0, 3);
    e.setBuffer(p.st.multi_pairs, 0, 4);
    e.setBuffer(p.st.ind, 0, 5);
    e.setBuffer(p.st.patch_cnt, 0, 6);
    e.setValue([4]u32{ c.n / 32, c.k / 32, 2, 0 }, 7);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(32, 1, 1));
}

pub fn upHybrid(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, table: Ref, r: Route, tile_table: Ref, act: Ref, c: Call) void {
    e.setPipeline(p.upc_one_xu);
    e.setBuffer(codes.buf, codes.off, 0);
    e.setBuffer(table.buf, table.off, 1);
    e.setBuffer(p.st.one_slots, 0, 2);
    e.setBuffer(p.st.ind, 28 * 4, 3);
    e.setBuffer(r.pairs.buf, r.pairs.off, 4);
    e.setBuffer(act.buf, act.off, 5);
    e.setValue(DsUpArgs{ .K = c.k, .N = c.n, .topk = topk, .rows_max = c.rows, .limit = ex.swiglu_limit }, 6);
    e.setBuffer(r.weights.buf, r.weights.off, 7);
    e.setBuffer(scales.buf, scales.off, 8);
    e.dispatchIndirect(p.st.ind, 0, mtl.Size.of(256, 1, 1));

    const args = XmArgs{ .K = c.k, .N = c.n, .topk = topk, .pairs = c.rows * topk, .limit = ex.swiglu_limit };
    const pa = XuPatchArgs{ .patch_cap = patch_cap_up, .ug_cap = ug_cap_rows };
    e.setPipeline(p.up);
    bind(e, &.{ codes, scales, table, tile_table, r.pairs, act });
    e.setValue(args, 6);
    e.setBuffer(r.weights.buf, r.weights.off, 7);
    e.setBuffer(p.st.patch_meta, 0, 8);
    e.setBuffer(p.st.patch_vals, 0, 9);
    e.setBuffer(p.st.patch_cnt, 0, 10);
    e.setValue(pa, 11);
    e.dispatchIndirect(p.st.ind, 32, mtl.Size.of(128, 1, 1));
}

pub fn patchUpHybrid(p: *const Pipelines, e: mtl.ComputeEncoder, codes: Ref, scales: Ref, table: Ref, r: Route, tile_table: Ref, act: Ref, c: Call) void {
    const args = XmArgs{ .K = c.k, .N = c.n, .topk = topk, .pairs = c.rows * topk, .limit = ex.swiglu_limit };
    const pa = XuPatchArgs{ .patch_cap = patch_cap_up, .ug_cap = ug_cap_rows };
    e.setPipeline(p.patch_up);
    bind(e, &.{ codes, scales, table, r.pairs, r.weights, tile_table });
    e.setBuffer(p.st.patch_meta, 0, 6);
    e.setBuffer(p.st.patch_vals, 0, 7);
    e.setBuffer(p.st.patch_cnt, 0, 8);
    e.setBuffer(act.buf, act.off, 9);
    e.setValue(args, 10);
    e.setValue(pa, 11);
    e.setBuffer(p.st.unsafe_mask, 0, 12);
    e.dispatchIndirect(p.st.patch_cnt, 16, mtl.Size.of(128, 1, 1));
}

pub fn dprepHybrid(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, dq: Ref, c: Call) void {
    const pairs = c.rows * topk;
    const ds: Ref = .{ .buf = dq.buf, .off = dq.off + 2 * @as(usize, pairs + 32) * c.n };
    e.setPipeline(p.dprep_hybrid);
    bind(e, &.{ act, dq, ds });
    e.setValue([3]u32{ c.n, pairs, ug_cap_rows }, 3);
    e.setBuffer(p.st.unsafe_mask, 0, 4);
    e.setBuffer(p.st.multi_pairs, 0, 5);
    e.setBuffer(p.st.ind, 29 * 4, 6);
    e.dispatchIndirect(p.st.ind, 64, mtl.Size.of(256, 1, 1));
}

pub fn downOneHybrid(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, table: Ref, r: Route, out: Ref, c: Call) void {
    e.setPipeline(p.down_one);
    e.setBuffer(act.buf, act.off, 0);
    e.setBuffer(table.buf, table.off, 1);
    e.setBuffer(p.st.one_slots, 0, 2);
    e.setBuffer(p.st.ind, 28 * 4, 3);
    e.setBuffer(r.pairs.buf, r.pairs.off, 4);
    e.setBuffer(out.buf, out.off, 5);
    e.setValue(DsDownArgs{ .K = c.n, .N = c.k, .topk = topk, .rows_max = c.rows }, 6);
    e.dispatchIndirect(p.st.ind, 16, mtl.Size.of(256, 1, 1));
}

pub fn downXuHybrid(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, dq: Ref, table: Ref, tile_table: Ref, out: Ref, c: Call) void {
    const pairs = c.rows * topk;
    const ds: Ref = .{ .buf = dq.buf, .off = dq.off + 2 * @as(usize, pairs + 32) * c.n };
    const args = XmArgs{ .K = c.n, .N = c.k, .topk = topk, .pairs = pairs, .limit = 0 };
    const pa = XuPatchArgs{ .patch_cap = patch_cap_dn, .ug_cap = ug_cap_rows };
    e.setPipeline(p.down);
    bind(e, &.{ act, table, tile_table, out });
    e.setValue(args, 4);
    e.setBuffer(dq.buf, dq.off, 5);
    e.setBuffer(ds.buf, ds.off, 6);
    e.setBuffer(p.st.patch_meta, 0, 7);
    e.setBuffer(p.st.patch_vals, 0, 8);
    e.setBuffer(p.st.patch_cnt, 32, 9);
    e.setValue(pa, 10);
    e.dispatchIndirect(p.st.ind, 48, mtl.Size.of(64, 1, 1));
}

pub fn patchDownHybrid(p: *const Pipelines, e: mtl.ComputeEncoder, act: Ref, dq: Ref, table: Ref, tile_table: Ref, out: Ref, c: Call) void {
    const pairs = c.rows * topk;
    const ds: Ref = .{ .buf = dq.buf, .off = dq.off + 2 * @as(usize, pairs + 32) * c.n };
    const args = XmArgs{ .K = c.n, .N = c.k, .topk = topk, .pairs = pairs, .limit = 0 };
    const pa = XuPatchArgs{ .patch_cap = patch_cap_dn, .ug_cap = ug_cap_rows };
    e.setPipeline(p.patch_down);
    bind(e, &.{ act, table, tile_table, ds });
    e.setBuffer(p.st.patch_meta, 0, 4);
    e.setBuffer(p.st.patch_vals, 0, 5);
    e.setBuffer(p.st.patch_cnt, 32, 6);
    e.setBuffer(out.buf, out.off, 7);
    e.setValue(args, 8);
    e.setValue(pa, 9);
    e.setBuffer(p.st.unsafe_mask, 0, 10);
    e.dispatchIndirect(p.st.patch_cnt, 48, mtl.Size.of(128, 1, 1));
}
