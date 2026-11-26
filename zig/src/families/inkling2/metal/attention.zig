//! Real Metal pipeline loading and encoder dispatch for Inkling 2 Mini attention.
const std = @import("std");
const mtl = @import("metal");
const src = @import("inkling2_metal");
pub const planning = @import("attention_plan.zig");

pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Cache = struct { k: Ref, v: Ref };
pub const RowMeta = struct { pos: Ref, seg_of: Ref, segs: Ref };
pub const Slot = struct { kv: Cache, ring: u32, chunk: u32 = 0, last_pos: u32 = 0, rows: u32 = 0 };

pub const Kernels = struct {
    scalar_lib: mtl.Library,
    prompt_lib: mtl.Library,
    prep: mtl.Pipeline,
    prep_tree: mtl.Pipeline,
    prep_slots: mtl.Pipeline,
    decode: mtl.Pipeline,
    decode_tree: mtl.Pipeline,
    decode_slots: mtl.Pipeline,
    merge: mtl.Pipeline,
    prompt: mtl.Pipeline,
    prompt_tree_prefix: mtl.Pipeline,
    prompt_tree_tail: mtl.Pipeline,
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
        const libs = [_]mtl.Library{ scalar_lib, scalar_lib, scalar_lib, scalar_lib, scalar_lib, scalar_lib, scalar_lib, prompt_lib, prompt_lib, prompt_lib };
        const names = [_][]const u8{
            "tf_ink2_prep",
            "tf_ink2_prep_tree",
            "tf_ink2_prep_slots",
            "tf_ink2_attn_scalar",
            "tf_ink2_attn_scalar_tree",
            "tf_ink2_attn_slots",
            "tf_ink2_merge",
            "tf_ink2_attn_prompt",
            "tf_ink2_attn_prompt_tree_prefix",
            "tf_ink2_attn_prompt_tree_tail",
        };
        const reqs = [_]usize{ 32, 32, 32, 192, 32, 192, 128, 256, 256, 256 };
        for (libs, names, reqs, &pipes) |lib, name, req_threads, *pipe| {
            pipe.* = try mtl.Pipeline.init(device, lib, name, false);
            made += 1;
            if (pipe.maxThreads() < req_threads) return error.AttentionThreads;
        }
        return .{
            .scalar_lib = scalar_lib,
            .prompt_lib = prompt_lib,
            .prep = pipes[0],
            .prep_tree = pipes[1],
            .prep_slots = pipes[2],
            .decode = pipes[3],
            .decode_tree = pipes[4],
            .decode_slots = pipes[5],
            .merge = pipes[6],
            .prompt = pipes[7],
            .prompt_tree_prefix = pipes[8],
            .prompt_tree_tail = pipes[9],
            .units = units,
        };
    }
    pub fn deinit(k: *Kernels) void {
        k.prompt_tree_tail.deinit();
        k.prompt_tree_prefix.deinit();
        k.prompt.deinit();
        k.merge.deinit();
        k.decode_slots.deinit();
        k.decode_tree.deinit();
        k.decode.deinit();
        k.prep_slots.deinit();
        k.prep_tree.deinit();
        k.prep.deinit();
        k.prompt_lib.deinit();
        k.scalar_lib.deinit();
        k.* = undefined;
    }
};

fn bind(e: mtl.ComputeEncoder, refs: []const Ref, first: usize) void {
    for (refs, first..) |r, slot| e.setBuffer(r.buf, r.off, slot);
}

fn prepLaunch(pipe: mtl.Pipeline, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, c: planning.Prep, meta: RowMeta, raw: [4]Ref, scales: [2]Ref, q: Ref, rel: Ref, kv: Cache) void {
    e.setPipeline(pipe);
    bind(e, &raw, 0);
    bind(e, &scales, 4);
    bind(e, &.{ meta.pos, meta.seg_of, meta.segs, q, rel, kv.k, kv.v }, 6);
    e.setValue(planning.PrepArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .global = @intFromBool(l.tau), .qk_clip = c.qk_clip, .rel_clip = c.rel_clip, .floor_pos = c.floor, .alpha = c.alpha, .eps = c.eps }, 13);
    e.dispatchGroups(mtl.Size.of(r.rows, r.heads + 2 * r.kv_heads, 1), mtl.Size.of(32, 1, 1));
}

/// raw holds independent q/k/v rows followed by q-side relative rows; gated sliding visits never infer K or V from Q.
pub fn prep(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, c: planning.Prep, meta: RowMeta, raw: [4]Ref, scales: [2]Ref, q: Ref, rel: Ref, kv: Cache) void {
    prepLaunch(k.prep, e, r, l, c, meta, raw, scales, q, rel, kv);
}

pub fn prepTree(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, c: planning.Prep, meta: RowMeta, raw: [4]Ref, scales: [2]Ref, q: Ref, rel: Ref, kv: Cache) void {
    prepLaunch(k.prep_tree, e, r, l, c, meta, raw, scales, q, rel, kv);
}

/// Batched non-tree prep for up to 8 stream slots; unused slots bind safely to slots[0].
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

fn partialsLaunch(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, launch: planning.Launch, meta: RowMeta, q: Ref, kv: Cache, rel: Ref, proj: Ref, part_o: Ref, part_ml: Ref) void {
    const scale: f32 = 1.0 / @sqrt(@as(f32, planning.head_dim));
    if (launch.path == .decode) {
        e.setPipeline(k.decode);
        bind(e, &.{ q, kv.k, kv.v, rel, proj, meta.segs, part_o, part_ml, meta.pos, meta.seg_of }, 0);
        e.setValue(planning.AttnArgs{ .rows = r.rows, .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = l.window, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale }, 10);
    } else {
        e.setPipeline(k.prompt);
        bind(e, &.{ q, kv.k, kv.v, rel, proj, meta.segs, part_o, part_ml }, 0);
        e.setValue(planning.PromptArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale }, 8);
    }
    e.dispatchGroups(mtl.Size.of(launch.grid[0], launch.grid[1], launch.grid[2]), mtl.Size.of(launch.threads, 1, 1));
}

/// Batched scalar attention partials for up to 8 non-tree stream slots, retaining main's fused relative projection.
pub fn partialsSlots(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, window: u32, stride: u32, meta: RowMeta, q: Ref, rel: Ref, proj: Ref, part_o: Ref, part_ml: Ref, slots: []const Slot) !void {
    std.debug.assert(slots.len > 0 and slots.len <= planning.max_slots);
    var slot_meta: [planning.max_slots]planning.SlotMeta = undefined;
    var max_slot_rows: u32 = 0;
    for (0..planning.max_slots) |i| {
        const s = if (i < slots.len) slots[i] else slots[0];
        const layer = planning.Layer{ .window = window, .ring = s.ring, .chunk = s.chunk, .tau = window == 0 };
        if (i < slots.len and planning.slotsFor(layer, @as(usize, s.last_pos) + 1) > stride) return error.Inkling2Partials;
        slot_meta[i] = .{ .ring = s.ring, .chunk = s.chunk };
        if (i < slots.len) max_slot_rows = @max(max_slot_rows, if (s.rows > 0) s.rows else r.rows);
    }
    const scale: f32 = 1.0 / @sqrt(@as(f32, planning.head_dim));
    const row_blocks = std.math.divCeil(u32, max_slot_rows, planning.decode_row_block) catch unreachable;
    e.setPipeline(k.decode_slots);
    bind(e, &.{ q, rel, proj, part_o, part_ml, meta.pos, meta.seg_of }, 0);
    for (0..planning.max_slots) |i| {
        const s = if (i < slots.len) slots[i] else slots[0];
        e.setBuffer(s.kv.k.buf, s.kv.k.off, 7 + i);
        e.setBuffer(s.kv.v.buf, s.kv.v.off, 15 + i);
    }
    e.setValue(planning.AttnSlotsArgs{
        .heads = r.heads,
        .kv_heads = r.kv_heads,
        .window = window,
        .c0 = 0,
        .chunks = stride,
        .n_slots = @intCast(slots.len),
        .scale = scale,
        .rows = r.rows,
        .slots = slot_meta,
    }, 23);
    e.dispatchGroups(mtl.Size.of(@as(u32, @intCast(slots.len)) * r.kv_heads, stride, row_blocks), mtl.Size.of(planning.decode_threads, 1, 1));
}

fn mergeLaunch(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, launch: planning.Launch, meta: RowMeta, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) void {
    e.setPipeline(k.merge);
    bind(e, &.{ part_o, part_ml, sink, gate, meta.pos, out }, 0);
    e.setValue(planning.MergeArgs{ .heads = r.heads, .window = l.window, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .subs = launch.subs }, 6);
    e.dispatchGroups(mtl.Size.of(launch.merge_grid[0], 1, 1), mtl.Size.of(planning.merge_threads, 1, 1));
}

pub fn partials(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, stride: u32, slots: usize, meta: RowMeta, q: Ref, kv: Cache, rel: Ref, proj: Ref, part_o: Ref, part_ml: Ref) !void {
    const launch = try planning.attend(r, l, stride, slots);
    partialsLaunch(k, e, r, l, launch, meta, q, kv, rel, proj, part_o, part_ml);
}

pub fn merge(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, stride: u32, slots: usize, meta: RowMeta, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) !void {
    const launch = try planning.attend(r, l, stride, slots);
    mergeLaunch(k, e, r, l, launch, meta, part_o, part_ml, sink, gate, out);
}

pub fn attend(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, stride: u32, slots: usize, meta: RowMeta, q: Ref, kv: Cache, rel: Ref, proj: Ref, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) !void {
    const launch = try planning.attend(r, l, stride, slots);
    partialsLaunch(k, e, r, l, launch, meta, q, kv, rel, proj, part_o, part_ml);
    e.barrier();
    mergeLaunch(k, e, r, l, launch, meta, part_o, part_ml, sink, gate, out);
}

pub fn attendTree(k: *const Kernels, e: mtl.ComputeEncoder, r: planning.Rows, l: planning.Layer, stride: u32, start: u32, meta: RowMeta, tree_paths: Ref, tail_o: Ref, tail_ml: Ref, q: Ref, kv: Cache, rel: Ref, proj: Ref, part_o: Ref, part_ml: Ref, sink: Ref, gate: Ref, out: Ref) !void {
    const launch = try planning.attendTree(r, l, stride, start);
    const scale: f32 = 1.0 / @sqrt(@as(f32, planning.head_dim));
    if (launch.global) {
        const args = planning.PromptTreeArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale, .p_tile = launch.p_tile, .c_off = launch.c_off, .max_pos = r.last_pos };
        if (launch.p_tile > 0) {
            e.setPipeline(k.prompt_tree_prefix);
            bind(e, &.{ q, kv.k, kv.v, rel, proj, meta.segs, part_o, part_ml, meta.pos, tail_o, tail_ml }, 0);
            e.setValue(args, 11);
            e.dispatchGroups(mtl.Size.of(launch.prefix_grid[0], launch.prefix_grid[1], launch.prefix_grid[2]), mtl.Size.of(launch.threads, 1, 1));
            e.barrier();
        }
        e.setPipeline(k.prompt_tree_tail);
        bind(e, &.{ q, kv.k, kv.v, rel, proj, meta.segs, part_o, part_ml, meta.pos, tree_paths, tail_o, tail_ml }, 0);
        e.setValue(args, 12);
        e.dispatchGroups(mtl.Size.of(launch.tail_grid[0], launch.tail_grid[1], launch.tail_grid[2]), mtl.Size.of(launch.threads, 1, 1));
    } else {
        e.setPipeline(k.decode_tree);
        bind(e, &.{ q, kv.k, kv.v, rel, proj, meta.segs, part_o, part_ml, meta.pos, meta.seg_of, tree_paths }, 0);
        e.setValue(planning.AttnArgs{ .heads = r.heads, .kv_heads = r.kv_heads, .ring = l.ring, .window = l.window, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .scale = scale }, 11);
        e.dispatchGroups(mtl.Size.of(launch.tail_grid[0], launch.tail_grid[1], launch.tail_grid[2]), mtl.Size.of(launch.threads, 1, 1));
    }
    e.barrier();
    e.setPipeline(k.merge);
    bind(e, &.{ part_o, part_ml, sink, gate, meta.pos, out }, 0);
    e.setValue(planning.MergeArgs{ .heads = r.heads, .window = l.window, .chunk = l.chunk, .c0 = 0, .chunks = launch.partials_rh, .subs = 1 }, 6);
    e.dispatchGroups(mtl.Size.of(launch.merge_grid[0], 1, 1), mtl.Size.of(planning.merge_threads, 1, 1));
}

test {
    _ = @import("attention_plan.zig");
}
