//! Forty-eight independent visit caches and the GPU buffers used by one whole-loop forward.
const std = @import("std");
const mtl = @import("metal");
const plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const experts_plan = @import("experts_plan.zig");
const tree = @import("tree.zig");

pub const window_rows = 16;
pub const batch_rows = 32;
pub const prompt_rows = 256;
pub const max_rows = 256;
pub const max_segments = batch_rows;
pub const max_attention_stride = 64;
pub const hidden = 2304;
pub const feature_taps = 4;
pub const feature_row_bytes = feature_taps * hidden * 2;

pub fn captureBytes(rows: usize) usize {
    return rows * feature_row_bytes;
}
pub const query_width = 3072;
pub const kv_width = 1024;
pub const rel_width = 384;
pub const experts_per_token = 8;
pub const intermediate = 1152;
pub const vocabulary = 200058;
pub const message_entries = 48;
pub const block_slots = 4;
pub const conv_states = 4;
pub const visit_slots = 48;
pub const sliding_ring = ap.ringFor(1024, max_rows);
pub const conv_widths = [conv_states]usize{ 1024, 1024, 2304, 2304 };
pub const conv_row_values = 1024 + 1024 + 2304 + 2304;
pub const conv_bytes = visit_slots * 3 * conv_row_values * 2;
pub const taps_bytes = visit_slots * (3 + max_rows) * conv_row_values * 2;

pub fn globalSlots(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, ap.tile_keys);
}

pub fn globalStride(capacity: usize) usize {
    return @max(@as(usize, ap.sliding_chunk / ap.tile_keys + 2), std.math.divCeil(usize, capacity, ap.global_chunk) catch unreachable);
}

pub fn kvBytes(capacity: usize) usize {
    const row_bytes = kv_width * 2;
    return 40 * sliding_ring * 2 * row_bytes + 8 * globalSlots(capacity) * 2 * row_bytes;
}

pub fn cacheBytes(capacity: usize) usize {
    return kvBytes(capacity) + conv_bytes + taps_bytes + conv_bytes + vocabulary * 4;
}

pub fn convWidth(state: usize) usize {
    std.debug.assert(state < conv_states);
    return conv_widths[state];
}

/// Values before a convolution state's rows in one slot's tightly sized layout.
pub fn convBase(state: usize, tap_rows: usize) usize {
    var out: usize = 0;
    for (0..state) |s| out += tap_rows * conv_widths[s];
    return out;
}

pub fn convByteOffset(slot: usize, state: usize, rows_before: usize, tap_rows: usize) usize {
    std.debug.assert(slot < visit_slots and rows_before <= tap_rows);
    return (slot * tap_rows * conv_row_values + convBase(state, tap_rows) + rows_before * convWidth(state)) * 2;
}

pub const Ref = @import("inkling2_metal_ref").Ref;

/// Tap rows cover [start,start+rows); scratch rows carry a generation, unlike the prompt's stable last row.
pub const Tap = struct { buf: ?mtl.Buffer = null, off: usize = 0, start: u64 = 0, rows: usize = 0, gen: ?u64 = null };

/// Byte offset of a requested held feature range, rejecting stale and rejected rows.
pub fn featureOffset(t: Tap, start: u64, count: u32, generation: u64) !usize {
    if (t.gen) |g| if (g != generation) return error.StaleInkling2Taps;
    if (count == 0 or start < t.start) return error.Inkling2TapsNotHeld;
    const delta = start - t.start;
    const held: u64 = @intCast(t.rows);
    if (delta >= held or count > held - delta) return error.Inkling2TapsNotHeld;
    return t.off + @as(usize, @intCast(delta)) * feature_row_bytes;
}

pub const Cache = struct {
    kv: mtl.Buffer,
    conv: mtl.Buffer,
    taps: mtl.Buffer,
    backup: mtl.Buffer,
    logits: mtl.Buffer,
    tap: Tap = .{},
    /// A stepped prompt chunk's rows, good until the next forward.
    fill_tap: Tap = .{},
    last_tap: ?mtl.Buffer = null,
    key_at: [visit_slots]usize,
    value_at: [visit_slots]usize,
    rings: [visit_slots]u32,
    globals: [visit_slots]bool,
    capacity: usize,
    len: usize = 0,
    pending: ?usize = null,
    pending_has_tree: bool = false,
    pending_parents: [window_rows]i32 = @splat(-1),
    logits_base: usize = 0,
    logits_gen: u64 = 0,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(device: mtl.Device, capacity: usize, schedule: *const plan.Plan) !Cache {
        if (capacity == 0 or capacity > 1048576 or schedule.n_visits != visit_slots) return error.Inkling2CacheShape;
        var out: Cache = .{
            .kv = undefined,
            .conv = undefined,
            .taps = undefined,
            .backup = undefined,
            .logits = undefined,
            .key_at = undefined,
            .value_at = undefined,
            .rings = undefined,
            .globals = undefined,
            .capacity = capacity,
        };
        var at: usize = 0;
        for (0..visit_slots) |slot| {
            const global = schedule.at(schedule.visit(@intCast(slot)).layer).kind == .global;
            const ring_size = if (global) globalSlots(capacity) else sliding_ring;
            out.globals[slot] = global;
            out.rings[slot] = @intCast(ring_size);
            out.key_at[slot] = at;
            at += ring_size * kv_width * 2;
            out.value_at[slot] = at;
            at += ring_size * kv_width * 2;
        }
        std.debug.assert(at == kvBytes(capacity));
        out.kv = try device.buffer(@max(at, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.conv = try device.buffer(conv_bytes, mtl.ResourceOptions.private);
        errdefer out.conv.deinit();
        out.taps = try device.buffer(taps_bytes, mtl.ResourceOptions.private);
        errdefer out.taps.deinit();
        out.backup = try device.buffer(conv_bytes, mtl.ResourceOptions.private);
        errdefer out.backup.deinit();
        out.logits = try device.buffer(vocabulary * 4, mtl.ResourceOptions.shared);
        errdefer out.logits.deinit();
        const queue = try device.queue();
        defer queue.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.blit();
        e.fill(out.conv, 0, conv_bytes, 0);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("inkling2_loop cache clear: {s}", .{msg});
            return error.GpuFailed;
        }
        return out;
    }

    pub fn setResident(c: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        const buffers = [_]mtl.Buffer{ c.kv, c.conv, c.taps, c.backup, c.logits };
        if (c.follow) |f| {
            for (buffers) |b| f.leave(f.ctx, b);
            if (c.last_tap) |b| f.leave(f.ctx, b);
        }
        c.follow = hook;
        if (hook) |f| {
            for (buffers) |b| f.join(f.ctx, b);
            if (c.last_tap) |b| f.join(f.ctx, b);
        }
    }

    pub fn ring(c: *const Cache, slot: usize) usize {
        return c.rings[slot];
    }

    pub fn keys(c: *const Cache, slot: usize) Ref {
        return .{ .buf = c.kv, .off = c.key_at[slot] };
    }

    pub fn values(c: *const Cache, slot: usize) Ref {
        return .{ .buf = c.kv, .off = c.value_at[slot] };
    }

    pub fn history(c: *const Cache, slot: usize, state: usize) Ref {
        return .{ .buf = c.conv, .off = convByteOffset(slot, state, 0, 3) };
    }

    pub fn staged(c: *const Cache, slot: usize, state: usize, rows_before: usize) Ref {
        return .{ .buf = c.taps, .off = convByteOffset(slot, state, rows_before, 3 + max_rows) };
    }

    pub fn attentionLayer(c: *const Cache, schedule: *const plan.Plan, slot: usize, rows: usize) ap.Layer {
        std.debug.assert(rows <= max_rows);
        var layer = if (schedule.at(schedule.visit(@intCast(slot)).layer).kind == .global) ap.Layer.global(@intCast(c.capacity)) else ap.Layer.sliding(1024, @intCast(rows));
        layer.ring = @intCast(c.ring(slot));
        return layer;
    }

    /// Save all four pre-window three-tap histories before KV/conv speculation.
    pub fn begin(c: *Cache, queue: mtl.Queue, rows: usize) !void {
        return c.beginTree(queue, rows, null);
    }

    pub fn beginTree(c: *Cache, queue: mtl.Queue, rows: usize, parents: ?[]const i32) !void {
        if (c.pending != null or rows == 0 or rows > window_rows or c.len + rows > c.capacity) return error.Inkling2CacheTransaction;
        var has_tree = false;
        if (parents) |ps| {
            if (ps.len != rows) return error.BadParents;
            var depths: [window_rows]u32 = undefined;
            try validateParents(ps, depths[0..rows]);
            has_tree = !isChain(ps);
            @memcpy(c.pending_parents[0..rows], ps);
        }
        try copyGpu(queue, c.conv, 0, c.backup, 0, conv_bytes, "begin");
        c.pending = rows;
        c.pending_has_tree = has_tree;
    }

    /// Commit accepted rows: staged taps select the new state; rejected KV positions are overwritten on replay.
    pub fn commit(c: *Cache, queue: mtl.Queue, accepted: usize) !void {
        const rows = c.pending orelse return error.Inkling2NoTransaction;
        if (accepted > rows or c.pending_has_tree) return error.Inkling2CacheTransaction;
        if (accepted < rows) {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = queue.commandBuffer();
            const e = cb.blit();
            e.copy(c.backup, 0, c.conv, 0, conv_bytes);
            for (0..visit_slots) |slot| for (0..conv_states) |state| {
                const bytes = 3 * convWidth(state) * 2;
                e.copy(c.taps, convByteOffset(slot, state, accepted, 3 + max_rows), c.conv, convByteOffset(slot, state, 0, 3), bytes);
            };
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("inkling2_loop cache commit: {s}", .{msg});
                return error.GpuFailed;
            }
        }
        const start = c.len;
        c.len += accepted;
        if (c.tap.buf != null and c.tap.start == start) {
            c.tap.rows = accepted;
        } else c.tap = .{};
        c.pending = null;
        c.pending_has_tree = false;
    }

    pub fn commitPath(c: *Cache, queue: mtl.Queue, path: []const u32, ops: *const tree.Pipelines) !void {
        const rows = c.pending orelse return error.Inkling2NoTransaction;
        try validateKeepPath(rows, if (c.pending_has_tree) c.pending_parents[0..rows] else null, path);
        var is_prefix = true;
        for (path, 0..) |r, i| if (r != i) {
            is_prefix = false;
            break;
        };
        const full_prefix = is_prefix and path.len == rows;
        if (!full_prefix) {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = queue.commandBuffer();
            const e = cb.compute(.serial);
            if (!is_prefix) {
                var kv_args: tree.plan.KeepKvArgs = .{
                    .key_at = undefined,
                    .value_at = undefined,
                    .rings = c.rings,
                    .start = @intCast(c.len),
                    .path_len = @intCast(path.len),
                    .path = @splat(0),
                };
                for (0..visit_slots) |slot| {
                    kv_args.key_at[slot] = @intCast(c.key_at[slot]);
                    kv_args.value_at[slot] = @intCast(c.value_at[slot]);
                }
                @memcpy(kv_args.path[0..path.len], path);
                tree.keepKv(ops, e, .{ .buf = c.kv }, kv_args);
            }
            tree.keepConv(ops, e, .{ .buf = c.taps }, .{ .buf = c.conv }, .{ .src = convHistorySrc(path), .tap_rows = 3 + max_rows });
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("inkling2_loop cache commitPath: {s}", .{msg});
                return error.GpuFailed;
            }
        }
        const start = c.len;
        c.len += path.len;
        if (c.tap.buf != null and c.tap.start == start) {
            c.tap.rows = path[path.len - 1] + 1;
        } else c.tap = .{};
        c.pending = null;
        c.pending_has_tree = false;
    }

    pub fn rollback(c: *Cache, queue: mtl.Queue) !void {
        _ = c.pending orelse return error.Inkling2NoTransaction;
        try copyGpu(queue, c.backup, 0, c.conv, 0, conv_bytes, "rollback");
        c.pending = null;
        c.pending_has_tree = false;
        c.tap = .{};
    }

    pub fn deinit(c: *Cache) void {
        c.setResident(null);
        c.logits.deinit();
        if (c.last_tap) |b| b.deinit();
        c.backup.deinit();
        c.taps.deinit();
        c.conv.deinit();
        c.kv.deinit();
        c.* = undefined;
    }
};

fn copyGpu(queue: mtl.Queue, from: mtl.Buffer, from_off: usize, to: mtl.Buffer, to_off: usize, bytes: usize, comptime what: []const u8) !void {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const cb = queue.commandBuffer();
    const e = cb.blit();
    e.copy(from, from_off, to, to_off, bytes);
    e.end();
    cb.commit();
    cb.wait();
    if (cb.failure()) |msg| {
        std.log.err("inkling2_loop cache {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

pub const fields = [_][]const u8{
    "ids",      "pos",        "seg_of",       "segs",      "p",          "prelude",      "blocks",        "block_inv_rms", "bank0",       "bank1",
    "messages", "partial",    "ar",           "gated",     "mlp_ar",     "mlp_gated",    "q_norm",        "k_norm",        "v_norm",      "q_raw",
    "k_raw",    "v_raw",      "rel_raw",      "sink",      "gate",       "k_conv",       "v_conv",        "q_prep",        "rel_prep",    "part_o",
    "part_ml",  "att",        "attn_dense",   "attn_conv", "mlp_normed", "route_ids",    "route_weights", "gu",            "down",        "mlp_out",
    "mlp_conv", "logits_all", "route_scores", "gate_norm", "gate_zeta",  "expert_group", "conv_src",      "tree_paths",    "tree_tail_o", "tree_tail_ml",
};

fn sizes(rows: usize) [fields.len]usize {
    const r = rows;
    const tr: usize = @min(r, @as(usize, batch_rows));
    const h = hidden;
    const q = query_width;
    const kv = kv_width;
    const bf = 2;
    const fp = 4;
    return .{
        r * 4,                     r * 4,                                            r * 4,                                     max_segments * @sizeOf(ap.Seg),                r * h * bf,  r * h * bf,           r * block_slots * h * bf,
        r * block_slots * fp,      r * h * bf,                                       r * h * bf,                                r * message_entries * h * fp,                  r * h * fp,  3 * r * h * fp,       3 * r * h * fp,
        r * h * fp,                r * h * fp,                                       r * h * bf,                                r * h * bf,                                    r * h * bf,  r * q * bf,           r * kv * bf,
        r * kv * bf,               r * rel_width * bf,                               r * 24 * bf,                               r * q * bf,                                    r * kv * bf, r * kv * bf,          r * q * bf,
        r * rel_width * bf,        r * 24 * max_attention_stride * ap.head_dim * fp, r * 24 * max_attention_stride * 2 * fp,    r * q * bf,                                    r * h * bf,  r * h * bf,           r * h * bf,
        r * experts_per_token * 4, r * experts_per_token * fp,                       r * experts_per_token * intermediate * bf, r * experts_per_token * h * bf,                r * h * bf,  r * h * bf,           r * vocabulary * fp,
        r * 128 * fp,              r * 2 * h * bf,                                   r * 64 * fp,                               experts_plan.groupBytes(r, experts_per_token), tr * 4 * 4,  tr * window_rows * 4, tr * 2 * 24 * ap.head_dim * fp,
        tr * 24 * 16,
    };
}

pub fn scratchBytes(rows: usize) usize {
    var out: usize = 0;
    for (sizes(rows)) |n| out += n;
    return out;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    seg_of: mtl.Buffer,
    segs: mtl.Buffer,
    p: mtl.Buffer,
    prelude: mtl.Buffer,
    blocks: mtl.Buffer,
    block_inv_rms: mtl.Buffer,
    bank0: mtl.Buffer,
    bank1: mtl.Buffer,
    messages: mtl.Buffer,
    partial: mtl.Buffer,
    ar: mtl.Buffer,
    gated: mtl.Buffer,
    mlp_ar: mtl.Buffer,
    mlp_gated: mtl.Buffer,
    q_norm: mtl.Buffer,
    k_norm: mtl.Buffer,
    v_norm: mtl.Buffer,
    q_raw: mtl.Buffer,
    k_raw: mtl.Buffer,
    v_raw: mtl.Buffer,
    rel_raw: mtl.Buffer,
    sink: mtl.Buffer,
    gate: mtl.Buffer,
    k_conv: mtl.Buffer,
    v_conv: mtl.Buffer,
    q_prep: mtl.Buffer,
    rel_prep: mtl.Buffer,
    part_o: mtl.Buffer,
    part_ml: mtl.Buffer,
    att: mtl.Buffer,
    attn_dense: mtl.Buffer,
    attn_conv: mtl.Buffer,
    mlp_normed: mtl.Buffer,
    route_ids: mtl.Buffer,
    route_weights: mtl.Buffer,
    gu: mtl.Buffer,
    down: mtl.Buffer,
    mlp_out: mtl.Buffer,
    mlp_conv: mtl.Buffer,
    logits_all: mtl.Buffer,
    /// The split router's [row][128] scores and the split gates' [row][2H] BF16 norms and [row][64] zeta.
    route_scores: mtl.Buffer,
    gate_norm: mtl.Buffer,
    gate_zeta: mtl.Buffer,
    expert_group: mtl.Buffer,
    conv_src: mtl.Buffer,
    tree_paths: mtl.Buffer,
    tree_tail_o: mtl.Buffer,
    tree_tail_ml: mtl.Buffer,
    /// Opt-in DSpark features [row][tap][H], allocated by prepare_features.
    taps: ?mtl.Buffer = null,
    rows: usize,

    pub fn init(device: mtl.Device, rows: usize) !Scratch {
        @setEvalBranchQuota(10_000);
        if (rows == 0 or rows > max_rows) return error.Inkling2ScratchRows;
        const ns = sizes(rows);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        s.taps = null;
        s.rows = rows;
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        if (s.taps) |b| b.deinit();
        s.* = undefined;
    }
};

pub const validateParents = tree.plan.validateParents;
pub const isChain = tree.plan.isChain;
pub const validateKeepPath = tree.plan.validateKeepPath;
pub const buildConvSrc = tree.plan.buildConvSrc;
pub const buildTreePaths = tree.plan.buildTreePaths;
pub const convHistorySrc = tree.plan.convHistorySrc;

comptime {
    if (sliding_ring != 1280 or conv_bytes != 1916928) @compileError("Inkling cache geometry drift");
}
