//! Non-looped 48-layer Inkling 2 Small Metal caches and per-forward scratch buffers.
const std = @import("std");
const mtl = @import("metal");
const exl3_proj = @import("core").exl3_projection;
const plan = @import("../plan.zig");
const ap = @import("attention_plan.zig");
const small_plan = @import("inkling2_small_ops_plan");

pub const window_rows = 16;
pub const batch_rows = 32;
pub const prompt_rows = 256;
pub const max_rows = 256;
pub const max_segments = batch_rows;
pub const max_attention_stride = 64;
pub const hidden = small_plan.hidden;
pub const heads = 48;
pub const kv_heads = 8;
pub const query_width = heads * ap.head_dim;
pub const kv_width = kv_heads * ap.head_dim;
pub const rel_width = heads * ap.d_rel;
pub const experts_per_token = small_plan.topk;
pub const expert_count = small_plan.expert_count;
pub const intermediate = 2304;
pub const vocabulary = small_plan.vocabulary;
pub const block_slots = small_plan.block_slots;
pub const feature_taps = small_plan.tap_channels;
pub const feature_row_bytes = feature_taps * hidden * 2;
pub const conv_states = 4;
pub const layer_slots = 48;
pub const sliding_ring = ap.ringFor(1024, max_rows);
pub const conv_widths = [conv_states]usize{ kv_width, kv_width, hidden, hidden };
pub const conv_row_values = kv_width + kv_width + hidden + hidden;
pub const conv_bytes = layer_slots * 3 * conv_row_values * 2;

pub fn captureBytes(rows: usize) usize {
    return rows * feature_row_bytes;
}

pub fn globalSlots(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, ap.tile_keys);
}

pub fn convWidth(state: usize) usize {
    std.debug.assert(state < conv_states);
    return conv_widths[state];
}

pub fn convBase(state: usize, tap_rows: usize) usize {
    var out: usize = 0;
    for (0..state) |s| out += tap_rows * conv_widths[s];
    return out;
}

pub fn convByteOffset(slot: usize, state: usize, rows_before: usize, tap_rows: usize) usize {
    std.debug.assert(rows_before <= tap_rows);
    return (slot * tap_rows * conv_row_values + convBase(state, tap_rows) + rows_before * convWidth(state)) * 2;
}

pub fn kvBytes(capacity: usize) usize {
    const row_bytes = kv_width * 2;
    return 40 * sliding_ring * 2 * row_bytes + 8 * globalSlots(capacity) * 2 * row_bytes;
}

pub fn cacheBytes(capacity: usize) usize {
    const taps_b = layer_slots * (3 + max_rows) * conv_row_values * 2;
    return kvBytes(capacity) + conv_bytes + taps_b + conv_bytes + vocabulary * 4;
}

pub const Ref = @import("inkling2_metal_ref").Ref;
pub const Tap = struct { buf: ?mtl.Buffer = null, off: usize = 0, start: u64 = 0, rows: usize = 0, gen: ?u64 = null };

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
    last_tap: ?mtl.Buffer = null,
    key_at: [layer_slots]usize = @splat(0),
    value_at: [layer_slots]usize = @splat(0),
    rings: [layer_slots]u32 = @splat(0),
    globals: [layer_slots]bool = @splat(false),
    slots: usize,
    max_chunk_rows: usize,
    conv_total_bytes: usize,
    capacity: usize,
    len: usize = 0,
    pending: ?usize = null,
    follow: ?mtl.ResidencySet.Follow = null,

    pub fn init(device: mtl.Device, capacity: usize, schedule: *const plan.Plan) !Cache {
        return initFor(device, capacity, schedule, max_rows, vocabulary);
    }

    pub fn initFor(device: mtl.Device, capacity: usize, schedule: *const plan.Plan, chunk_rows: usize, vocab: usize) !Cache {
        if (capacity == 0 or capacity > 1048576 or schedule.n_visits == 0 or schedule.n_visits > layer_slots or chunk_rows == 0 or chunk_rows > max_rows) {
            return error.Inkling2CacheShape;
        }
        const slots = schedule.n_visits;
        const ring_local = ap.ringFor(1024, @intCast(chunk_rows));
        var out: Cache = .{
            .kv = undefined,
            .conv = undefined,
            .taps = undefined,
            .backup = undefined,
            .logits = undefined,
            .slots = slots,
            .max_chunk_rows = chunk_rows,
            .conv_total_bytes = slots * 3 * conv_row_values * 2,
            .capacity = capacity,
        };
        var at: usize = 0;
        for (0..slots) |slot| {
            const global = schedule.at(schedule.visit(@intCast(slot)).layer).kind == .global;
            const ring_size = if (global) globalSlots(capacity) else ring_local;
            out.globals[slot] = global;
            out.rings[slot] = @intCast(ring_size);
            out.key_at[slot] = at;
            at += ring_size * kv_width * 2;
            out.value_at[slot] = at;
            at += ring_size * kv_width * 2;
        }
        const taps_total_bytes = slots * (3 + chunk_rows) * conv_row_values * 2;
        out.kv = try device.buffer(@max(at, 16), mtl.ResourceOptions.private);
        errdefer out.kv.deinit();
        out.conv = try device.buffer(out.conv_total_bytes, mtl.ResourceOptions.private);
        errdefer out.conv.deinit();
        out.taps = try device.buffer(taps_total_bytes, mtl.ResourceOptions.private);
        errdefer out.taps.deinit();
        out.backup = try device.buffer(out.conv_total_bytes, mtl.ResourceOptions.private);
        errdefer out.backup.deinit();
        out.logits = try device.buffer(@max(vocab * 4, 16), mtl.ResourceOptions.shared);
        errdefer out.logits.deinit();

        const queue = try device.queue();
        defer queue.deinit();
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const cb = queue.commandBuffer();
        const e = cb.blit();
        e.fill(out.conv, 0, out.conv_total_bytes, 0);
        e.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("inkling2_small cache clear: {s}", .{msg});
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
        return .{ .buf = c.taps, .off = convByteOffset(slot, state, rows_before, 3 + c.max_chunk_rows) };
    }

    pub fn attentionLayer(c: *const Cache, schedule: *const plan.Plan, slot: usize, rows: usize) ap.Layer {
        std.debug.assert(rows <= c.max_chunk_rows);
        var layer = if (schedule.at(schedule.visit(@intCast(slot)).layer).kind == .global)
            ap.Layer.global(@intCast(c.capacity))
        else
            ap.Layer.sliding(1024, @intCast(rows));
        layer.ring = @intCast(c.ring(slot));
        return layer;
    }

    pub fn begin(c: *Cache, queue: mtl.Queue, rows: usize) !void {
        if (c.pending != null or rows == 0 or rows > window_rows or c.len + rows > c.capacity) return error.Inkling2CacheTransaction;
        try copyGpu(queue, c.conv, 0, c.backup, 0, c.conv_total_bytes, "begin");
        c.pending = rows;
    }

    pub fn commit(c: *Cache, queue: mtl.Queue, accepted: usize) !void {
        const rows = c.pending orelse return error.Inkling2NoTransaction;
        if (accepted > rows) return error.Inkling2CacheTransaction;
        if (accepted < rows) {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            const cb = queue.commandBuffer();
            const e = cb.blit();
            e.copy(c.backup, 0, c.conv, 0, c.conv_total_bytes);
            for (0..c.slots) |slot| for (0..conv_states) |state| {
                const bytes = 3 * convWidth(state) * 2;
                e.copy(c.taps, convByteOffset(slot, state, accepted, 3 + c.max_chunk_rows), c.conv, convByteOffset(slot, state, 0, 3), bytes);
            };
            e.end();
            cb.commit();
            cb.wait();
            if (cb.failure()) |msg| {
                std.log.err("inkling2_small cache commit: {s}", .{msg});
                return error.GpuFailed;
            }
        }
        const start = c.len;
        c.len += accepted;
        if (c.tap.buf != null and c.tap.start == start) {
            c.tap.rows = accepted;
        } else c.tap = .{};
        c.pending = null;
    }

    pub fn rollback(c: *Cache, queue: mtl.Queue) !void {
        _ = c.pending orelse return error.Inkling2NoTransaction;
        try copyGpu(queue, c.backup, 0, c.conv, 0, c.conv_total_bytes, "rollback");
        c.pending = null;
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
        std.log.err("inkling2_small cache {s}: {s}", .{ what, msg });
        return error.GpuFailed;
    }
}

pub const fields = [_][]const u8{
    "ids",          "pos",           "seg_of",  "segs",   "p",          "blocks",    "partial",
    "ar",           "mlp_ar",        "q_norm",  "k_norm", "v_norm",     "q_raw",     "k_raw",
    "v_raw",        "rel_raw",       "sink",    "gate",   "k_conv",     "v_conv",    "q_prep",
    "rel_prep",     "part_o",        "part_ml", "att",    "attn_dense", "attn_conv", "mlp_normed",
    "route_ids",    "route_weights", "gu_raw",  "gu",     "down",       "mlp_out",   "mlp_conv",
    "exl3_scratch", "logits_all",
};

fn sizes(rows: usize, vocab: usize) [fields.len]usize {
    const r = rows;
    const h = hidden;
    const q = query_width;
    const kv = kv_width;
    const bf = 2;
    const fp = 4;
    const exl3_b = exl3_proj.scratchBytes(h, h, r * experts_per_token) catch unreachable;
    return .{
        r * 4,                     r * 4,                                               r * 4,                                     max_segments * @sizeOf(ap.Seg),            r * h * bf,                     r * block_slots * h * bf, r * h * fp,
        3 * r * h * fp,            r * h * fp,                                          r * h * bf,                                r * h * bf,                                r * h * bf,                     r * q * bf,               r * kv * bf,
        r * kv * bf,               r * rel_width * bf,                                  r * heads * bf,                            r * q * bf,                                r * kv * bf,                    r * kv * bf,              r * q * bf,
        r * rel_width * bf,        r * heads * max_attention_stride * ap.head_dim * fp, r * heads * max_attention_stride * 2 * fp, r * q * bf,                                r * h * bf,                     r * h * bf,               r * h * bf,
        r * experts_per_token * 4, r * experts_per_token * fp,                          r * experts_per_token * h * bf,            r * experts_per_token * intermediate * bf, r * experts_per_token * h * bf, r * h * bf,               r * h * bf,
        exl3_b,                    r * vocab * fp,
    };
}

pub fn scratchBytes(rows: usize) usize {
    var out: usize = 0;
    for (sizes(rows, vocabulary)) |n| out += n;
    return out;
}

pub const Scratch = struct {
    ids: mtl.Buffer,
    pos: mtl.Buffer,
    seg_of: mtl.Buffer,
    segs: mtl.Buffer,
    p: mtl.Buffer,
    blocks: mtl.Buffer,
    partial: mtl.Buffer,
    ar: mtl.Buffer,
    mlp_ar: mtl.Buffer,
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
    gu_raw: mtl.Buffer,
    gu: mtl.Buffer,
    down: mtl.Buffer,
    mlp_out: mtl.Buffer,
    mlp_conv: mtl.Buffer,
    exl3_scratch: mtl.Buffer,
    logits_all: mtl.Buffer,
    taps: ?mtl.Buffer = null,
    rows: usize,
    vocab: usize,

    pub fn init(device: mtl.Device, rows: usize) !Scratch {
        return initFor(device, rows, vocabulary);
    }

    pub fn initFor(device: mtl.Device, rows: usize, vocab: usize) !Scratch {
        @setEvalBranchQuota(10_000);
        if (rows == 0 or rows > max_rows or vocab == 0) return error.Inkling2ScratchRows;
        const ns = sizes(rows, vocab);
        var s: Scratch = undefined;
        var made: usize = 0;
        errdefer inline for (fields, 0..) |f, i| if (i < made) @field(s, f).deinit();
        inline for (fields, 0..) |f, i| {
            @field(s, f) = try device.buffer(@max(ns[i], 16), mtl.ResourceOptions.shared);
            made += 1;
        }
        s.taps = null;
        s.rows = rows;
        s.vocab = vocab;
        return s;
    }

    pub fn deinit(s: *Scratch) void {
        inline for (fields) |f| @field(s, f).deinit();
        if (s.taps) |b| b.deinit();
        s.* = undefined;
    }
};

comptime {
    if (sliding_ring != 1280 or conv_bytes != 3244032) @compileError("Inkling 2 Small cache geometry drift");
}
