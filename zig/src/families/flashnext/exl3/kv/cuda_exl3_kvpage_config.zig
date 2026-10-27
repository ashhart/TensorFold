//! Paged KV shape and sizing: the Config, the layout hash, admission room and the window and row plans.

const std = @import("std");
const wt = @import("../../cuda_prompt_wt.zig");
const parts = @import("../../cuda_batch_state.zig");
const shared_contract = @import("../offload/exl3_shared_kv.zig");
const shared_device = @import("../offload/cuda_exl3_shared_kv.zig");
const snaps = @import("cuda_exl3_kvpage_snapshot.zig");

const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const ratio = wt.index_ratio;
const top = wt.index_top;
const att_rows = parts.att_rows;
const attn_n = wt.attn_n;
const work_n: usize = att_rows * top;

const SnapshotLayout = snaps.SnapshotLayout;

/// One token's K (or V) in one attention layer: [2][256] bf16.
pub const row_bytes: usize = kv_heads * head_dim * 2;

pub fn cdiv(a: usize, b: usize) usize {
    return (a + b - 1) / b;
}

pub fn cint(v: usize) c_int {
    return @intCast(v);
}

/// Shape of a paged sequence's device side (TF_FLASHNEXT_KVPAGE=1 turns it on for every new sequence).
pub const Config = struct {
    /// Tokens kept dense on the card before a call's first row (>= 2051, so a dense row's keys are all in it).
    window: usize = 4096,
    /// 4-key block slots, one pool for the 12 layers (the prompt's layer-major order gives one layer all of it).
    slots: usize = 49152,
    /// The most rows one attention call writes (the engine's chunk / a deep prefill's sub-block).
    rows: usize = 1536,
    shared_lanes: usize = 0,

    pub fn forChunk(cfg: Config, chunk: usize) Config {
        var out = cfg;
        out.rows = @max(out.rows, chunk);
        return out;
    }

    pub fn validate(cfg: Config) !void {
        if (cfg.window < wt.dense_keys or cfg.slots == 0 or cfg.rows == 0) return error.KvPageConfig;
        if (cfg.shared_lanes > shared_device.max_lanes) return error.KvPageConfig;
    }
};

pub fn configIdentity(cfg: ?Config) u64 {
    const paging = cfg orelse return 0;
    const shape = [_]u64{ paging.window, paging.rows, paging.slots, paging.shared_lanes, row_bytes, attn_n };
    return std.hash.Wyhash.hash(0, std.mem.asBytes(&shape));
}

pub fn admissionRoom(local: u64, stream_bytes: usize, cfg: ?Config) !u64 {
    const paging = cfg orelse return local;
    try paging.validate();
    if (paging.shared_lanes == 0) return local;
    return @min(local, try std.math.mul(u64, stream_bytes, paging.shared_lanes));
}

pub fn pagingRows(cfg: Config, origin: usize, rows: usize) !usize {
    try cfg.validate();
    if (rows == 0 or rows > att_rows) return error.KvPageRows;
    if (origin % ratio != 0) return error.KvPageWindow;
    if (origin / ratio <= cfg.slots) return rows;
    const bounded = @min(rows, cfg.slots / top);
    if (bounded == 0) return error.KvPageCapacity;
    return bounded;
}

pub fn windowPlan(cfg: Config, wr: usize, w0: usize, pos0: usize, n: usize) !usize {
    try cfg.validate();
    if (n == 0 or n > cfg.rows) return error.KvPageRows;
    if (w0 % ratio != 0) return error.KvPageWindow;
    if (pos0 == 0) {
        _ = try pagingRows(cfg, 0, @min(n, att_rows));
        return 0;
    }
    if (pos0 < w0) return error.KvPageRewind;
    if (pos0 - w0 > wr) return error.KvPageRows;
    const end = try std.math.add(usize, pos0 - w0, n);
    const next = if (end > wr) (pos0 -| cfg.window) / ratio * ratio else w0;
    if (pos0 - next > wr or n > wr - (pos0 - next)) return error.KvPageRows;
    _ = try pagingRows(cfg, next, @min(n, att_rows));
    return next;
}

/// Device bytes of a paged sequence's K/V side for `cap` tokens under `cfg` (windows, slots, tables, work lists).
pub fn deviceBytesFor(cap: usize, cfg: Config) !usize {
    const layout = try SnapshotLayout.init(cap, cfg, 0);
    if (cfg.shared_lanes != 0) {
        _ = try sharedBytesFor(cap, cfg);
        return 0;
    }
    return std.math.add(usize, try std.math.add(usize, layout.device, layout.tables), work_n * 4 * 4);
}

pub fn sharedBytesFor(cap: usize, cfg: Config) !usize {
    try cfg.validate();
    if (cfg.shared_lanes == 0) return 0;
    return (try shared_device.Layout.init(.{ .cap = cap, .lanes = cfg.shared_lanes, .window = cfg.window, .rows = cfg.rows, .slots = cfg.slots })).common_bytes;
}

pub fn ownedSnapshotBytes(cfg: Config) !usize {
    if (cfg.shared_lanes == 0) return 0;
    try cfg.validate();
    return std.math.mul(usize, cfg.slots, 2 * ratio * row_bytes + @sizeOf(shared_contract.OwnedEntry));
}
