//! Real Metal pipeline loading and encoder dispatch for Inkling 2 Mini attention.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
pub const planning = @import("attention_plan.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Cache = struct { k: Ref, v: Ref };
pub const RowMeta = struct { pos: Ref, seg_of: Ref, segs: Ref };
pub const Slot = struct { kv: Cache, ring: u32, chunk: u32 = 0, last_pos: u32 = 0 };

pub const Kernels = struct {
    scalar_lib: mtl.Library,
    prompt_lib: mtl.Library,
    prep: mtl.Pipeline,
    prep_slots: mtl.Pipeline,
    rel_table: mtl.Pipeline,
    decode: mtl.Pipeline,
    decode_gqa: mtl.Pipeline,
    decode_split: mtl.Pipeline,
    decode_slots: mtl.Pipeline,
    merge: mtl.Pipeline,
    prompt: mtl.Pipeline,
    prompt_sliding: mtl.Pipeline,
    verify_slide_nax: ?mtl.Pipeline = null,
    verify_global_nax: ?mtl.Pipeline = null,
    units: bool,

    pub fn init(a: std.mem.Allocator, device: mtl.Device) !Kernels {
        var scalar_lib = try mtl.Library.fromSource(device, src.attention, mtl.CompileOptions.mlx());
        errdefer scalar_lib.deinit();
        const units = device.tensorUnits();
        const prompt_src = try a.dupe(u8, if (units) src.prompt_attention else src.prompt_attention_pre_m5);
        defer a.free(prompt_src);
        var prompt_lib = try mtl.Library.fromSource(device, prompt_src, mtl.CompileOptions.mlx());
        errdefer prompt_lib.deinit();
        var pipes: [10]mtl.Pipeline = undefined;
        var made: usize = 0;
        errdefer for (pipes[0..made]) |pipe| pipe.deinit();
        const libs = [_]mtl.Library{ scalar_lib, scalar_lib, scalar_lib, scalar_lib, scalar_lib, scalar_lib, scalar_lib, scalar_lib, prompt_lib, prompt_lib };
        const names = [_][]const u8{
            "tf_ink2_prep",
            "tf_ink2_prep_slots",
            "tf_ink2_rel_table",
            "tf_ink2_attn_scalar",
            "tf_ink2_attn_gqa",
            "tf_ink2_attn_split",
            "tf_ink2_attn_slots",
            "tf_ink2_merge",
            "tf_ink2_attn_prompt",
            "tf_ink2_attn_prompt_sliding",
        };
        for (libs, names, &pipes) |lib, name, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
        }
        var verify_slide_nax: ?mtl.Pipeline = null;
        var verify_global_nax: ?mtl.Pipeline = null;
        if (units) {
            verify_slide_nax = try mtl.Pipeline.init(device, prompt_lib, "tf_ink2_attn_verify_slide_nax", false);
            errdefer if (verify_slide_nax) |p| p.deinit();
            verify_global_nax = try mtl.Pipeline.init(device, prompt_lib, "tf_ink2_attn_verify_global_nax", false);
        }
        return .{
            .scalar_lib = scalar_lib,
            .prompt_lib = prompt_lib,
            .prep = pipes[0],
            .prep_slots = pipes[1],
            .rel_table = pipes[2],
            .decode = pipes[3],
            .decode_gqa = pipes[4],
            .decode_split = pipes[5],
            .decode_slots = pipes[6],
            .merge = pipes[7],
            .prompt = pipes[8],
            .prompt_sliding = pipes[9],
            .verify_slide_nax = verify_slide_nax,
            .verify_global_nax = verify_global_nax,
            .units = units,
        };
    }
    pub fn deinit(k: *Kernels) void {
        if (k.verify_global_nax) |p| p.deinit();
        if (k.verify_slide_nax) |p| p.deinit();
        k.prompt_sliding.deinit();
        k.prompt.deinit();
        k.merge.deinit();
        k.decode_slots.deinit();
        k.decode_split.deinit();
        k.decode_gqa.deinit();
        k.decode.deinit();
        k.rel_table.deinit();
        k.prep_slots.deinit();
        k.prep.deinit();
        k.prompt_lib.deinit();
        k.scalar_lib.deinit();
        k.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref, first: usize) void {
    for (refs, first..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}

/// raw holds independent q/k/v rows followed by q-side relative rows; gated sliding visits never infer K or V from Q.
pub fn prep(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, c: planning.Prep, meta: RowMeta, raw: [4]Ref, scales: [2]Ref, q: Ref, rel: Ref, kv: Cache) void {
    e.setPipeline(k.prep);
    bind(e, &raw, 0);
    bind(e, &scales, 4);
    bind(e, &.{ meta.pos, meta.seg_of, meta.segs, q, rel, kv.k, kv.v }, 6);
    e.setValue(planning.PrepArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .global = @intFromBool(l.tau), .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor_pos = c.floor, .alpha = c.alpha, .eps = c.eps }, 13);
    e.dispatchGroups(mtl.Size.of(r.rows, r.heads + 2 * r.kv_heads, 1), mtl.Size.of(32, 1, 1));
}

/// Batched prep for up to 8 stream slots; unused slots bind safely to slots[0].
pub fn prepSlots(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, global: bool, c: planning.Prep, meta: RowMeta, raw: [4]Ref, scales: [2]Ref, q: Ref, rel: Ref, slots: []const Slot) void {
    std.debug.assert(slots.len > 0 and slots.len <= planning.max_slots);
    e.setPipeline(k.prep_slots);
    bind(e, &raw, 0);
    bind(e, &scales, 4);
    bind(e, &.{ meta.pos, meta.seg_of, q, rel }, 6);
    var rings: [planning.max_slots]u32 = undefined;
    for (0..planning.max_slots) |i| {
        const s = if (i < slots.len) slots[i] else slots[0];
        e.setBuffer(s.kv.k.buf, s.kv.k.off, 10 + i);
        e.setBuffer(s.kv.v.buf, s.kv.v.off, 18 + i);
        rings[i] = s.ring;
    }
    e.setValue(planning.PrepSlotsArgs{
        .heads = r.heads,
        .kv_heads = r.kv_heads,
        .global = @intFromBool(global),
        .n_slots = @intCast(slots.len),
        .rings = rings,
        .qk_clip = c.qk_clip,
        .rel_clip = c.rel_clip,
        .floor_pos = c.floor,
        .alpha = c.alpha,
        .eps = c.eps,
    }, 26);
    e.dispatchGroups(mtl.Size.of(r.rows, r.heads + 2 * r.kv_heads, 1), mtl.Size.of(32, 1, 1));
}

pub fn relTable(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, rel: Ref, proj: Ref, tab: Ref) void {
    e.setPipeline(k.rel_table);
    bind(e, &.{ rel, proj, tab }, 0);
    e.dispatchGroups(mtl.Size.of(r.rows * r.heads, 1, 1), mtl.Size.of(256, 1, 1));
}

fn promptSlidingLaunch(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, launch: planning.Launch, meta: RowMeta, q: Ref, kv: Cache, tab: Ref, sink: Ref, gate: Ref, out: Ref) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, planning.head_dim));
    e.setPipeline(k.prompt_sliding);
    bind(e, &.{ q, kv.k, kv.v, tab, meta.segs, sink, gate, out }, 0);
    e.setValue(planning.PromptArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = l.window, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale }, 8);
    e.dispatchGroups(mtl.Size.of(launch.grid[0], launch.grid[1], launch.grid[2]), mtl.Size.of(launch.threads, 1, 1));
}

fn partialsLaunch(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, launch: planning.Launch, meta: RowMeta, q: Ref, kv: Cache, tab: Ref, part_o: Ref, part_ml: Ref) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, planning.head_dim));
    if (launch.path == .decode) {
        if (k.units and r.streams == 1 and r.heads == 24 and r.kv_heads == 8 and r.rows >= 2 and r.last_pos >= 256) {
            const pair_tiles: usize = (r.rows * 3 + 15) / 16;
            if (l.window > 0) {
                const p0: u32 = r.last_pos + 1 - r.rows;
                const lo_min: u32 = if (p0 + 1 > l.window) p0 + 1 - l.window else 0;
                const c_min: u32 = lo_min / l.chunk;
                const c_max: u32 = r.last_pos / l.chunk;
                e.setPipeline(k.verify_slide_nax.?);
                bind(e, &.{ q, kv.k, kv.v, tab, meta.segs, part_o, part_ml }, 0);
                e.setValue(planning.PromptArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = l.window, .chunk = l.chunk, .c0 = c_min, .chunks = launch.partials_rh, .scale = scale }, 7);
                e.dispatchGroups(mtl.Size.of(pair_tiles, r.kv_heads, c_max - c_min + 1), mtl.Size.of(32, 1, 1));
                return;
            }
            e.setPipeline(k.verify_global_nax.?);
            bind(e, &.{ q, kv.k, kv.v, tab, meta.segs, part_o, part_ml }, 0);
            e.setValue(planning.PromptArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = 0, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale }, 7);
            e.dispatchGroups(mtl.Size.of(pair_tiles, r.kv_heads, launch.grid[1]), mtl.Size.of(256, 1, 1));
            return;
        }
        if (r.streams == 1 and r.heads == 24 and r.kv_heads == 8 and r.last_pos >= 256) {
            const pair_rows = l.window == 0 and r.rows > 1;
            const row_groups: usize = if (pair_rows) (r.rows + 1) / 2 else r.rows;
            e.setPipeline(if (l.window == 0) k.decode_split else k.decode_gqa);
            bind(e, &.{ q, kv.k, kv.v, tab, meta.segs, part_o, part_ml, meta.pos, meta.seg_of }, 0);
            e.setValue(planning.AttnArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = l.window, .chunk = l.chunk, .c0 = r.rows, .chunks = launch.partials_rh, .scale = scale }, 9);
            const tg_threads: usize = if (l.window == 0) 256 else launch.threads;
            e.dispatchGroups(mtl.Size.of(row_groups * r.kv_heads, launch.grid[1], launch.grid[2]), mtl.Size.of(tg_threads, 1, 1));
            return;
        }
        e.setPipeline(k.decode);
        bind(e, &.{ q, kv.k, kv.v, tab, meta.segs, part_o, part_ml, meta.pos, meta.seg_of }, 0);
        e.setValue(planning.AttnArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = l.window, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale }, 9);
    } else {
        e.setPipeline(k.prompt);
        bind(e, &.{ q, kv.k, kv.v, tab, meta.segs, part_o, part_ml }, 0);
        e.setValue(planning.PromptArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = l.window, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale }, 7);
    }
    e.dispatchGroups(mtl.Size.of(launch.grid[0], launch.grid[1], launch.grid[2]), mtl.Size.of(launch.threads, 1, 1));
}

/// Batched scalar attention partials for up to 8 stream slots; unused slots bind safely to slots[0].
pub fn partialsSlots(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, window: u32, stride: u32, meta: RowMeta, q: Ref, tab: Ref, part_o: Ref, part_ml: Ref, slots: []const Slot) !void {
    std.debug.assert(slots.len > 0 and slots.len <= planning.max_slots);
    var slot_meta: [planning.max_slots]planning.SlotMeta = undefined;
    for (0..planning.max_slots) |i| {
        const s = if (i < slots.len) slots[i] else slots[0];
        const layer = planning.Layer{ .window = window, .ring = s.ring, .chunk = s.chunk, .tau = window == 0 };
        if (i < slots.len and planning.slotsFor(layer, @as(usize, s.last_pos) + 1) > stride) return error.Inkling2Partials;
        slot_meta[i] = .{ .ring = s.ring, .chunk = s.chunk };
    }
    const scale: f32 = 1.0 / @sqrt(@as(f32, planning.head_dim));
    e.setPipeline(k.decode_slots);
    bind(e, &.{ q, tab, part_o, part_ml, meta.pos, meta.seg_of }, 0);
    for (0..planning.max_slots) |i| {
        const s = if (i < slots.len) slots[i] else slots[0];
        e.setBuffer(s.kv.k.buf, s.kv.k.off, 6 + i);
        e.setBuffer(s.kv.v.buf, s.kv.v.off, 14 + i);
    }
    e.setValue(planning.AttnSlotsArgs{
        .heads = r.heads,
        .kv_heads = r.kv_heads,
        .window = window,
        .c0 = 0,
        .chunks = stride,
        .n_slots = @intCast(slots.len),
        .scale = scale,
        .slots = slot_meta,
    }, 22);
    e.dispatchGroups(mtl.Size.of(r.rows * r.heads, stride, 1), mtl.Size.of(planning.decode_threads, 1, 1));
}

fn mergeLaunch(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, launch: planning.Launch, meta: RowMeta, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) void {
    const single_sliding = launch.path == .prompt and l.window > 0;
    e.setPipeline(k.merge);
    bind(e, &.{ part_o, part_ml, sink, gate, meta.pos, out }, 0);
    e.setValue(planning.MergeArgs{
        .heads = r.heads,
        .window = if (single_sliding) 0 else l.window,
        .chunk = if (single_sliding) r.last_pos + 1 else l.chunk,
        .c0 = 0,
        .chunks = launch.partials_rh,
        .subs = launch.subs,
    }, 6);
    e.dispatchGroups(mtl.Size.of(launch.merge_grid[0], 1, 1), mtl.Size.of(planning.merge_threads, 1, 1));
}

/// The partial sums of every row and head: `merge` reads them after a barrier.
pub fn partials(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, stride: u32, slots: usize, meta: RowMeta, q: Ref, kv: Cache, tab: Ref, part_o: Ref, part_ml: Ref) !void {
    const launch = try planning.attend(r, l, stride, slots);
    partialsLaunch(k, e, r, l, launch, meta, q, kv, tab, part_o, part_ml);
}

/// The partials of `partials` merged, with the sink and gate, into each row's attention output.
pub fn merge(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, stride: u32, slots: usize, meta: RowMeta, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) !void {
    const launch = try planning.attend(r, l, stride, slots);
    mergeLaunch(k, e, r, l, launch, meta, part_o, part_ml, sink, gate, out);
}

pub fn attend(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, stride: u32, slots: usize, meta: RowMeta, q: Ref, kv: Cache, tab: Ref, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) !void {
    const launch = try planning.attend(r, l, stride, slots);
    if (launch.path == .prompt and l.window > 0) {
        promptSlidingLaunch(k, e, r, l, launch, meta, q, kv, tab, sink, gate, out);
        return;
    }
    partialsLaunch(k, e, r, l, launch, meta, q, kv, tab, part_o, part_ml);
    e.barrier();
    mergeLaunch(k, e, r, l, launch, meta, part_o, part_ml, sink, gate, out);
}

test {
    _ = @import("attention_plan.zig");
}
