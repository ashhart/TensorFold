//! Paged KV snapshots and counters: the layout of a saved image, its validation and one layer's Stats.

const std = @import("std");
const wt = @import("../../cuda_prompt_wt.zig");
const shared_device = @import("../offload/cuda_exl3_shared_kv.zig");
const configs = @import("cuda_exl3_kvpage_config.zig");

const ratio = wt.index_ratio;
const attn_n = wt.attn_n;
const ctl_n: usize = 8;

const Config = configs.Config;
const row_bytes = configs.row_bytes;

pub const SnapshotLayout = struct {
    cap: usize,
    cfg: Config,
    at: usize,
    wr: usize,
    nblocks: usize,
    canonical: usize,
    device: usize,
    tables: usize,
    total: usize,

    pub fn init(cap: usize, cfg: Config, at: usize) !SnapshotLayout {
        try cfg.validate();
        if (cap == 0 or at > cap) return error.KvPageRows;
        _ = try std.math.mul(usize, cap, attn_n * 2 * row_bytes);
        if (cfg.shared_lanes != 0) _ = try shared_device.Layout.init(.{ .cap = cap, .lanes = cfg.shared_lanes, .window = cfg.window, .rows = cfg.rows, .slots = cfg.slots });
        const wr = try std.math.add(usize, try std.math.add(usize, cfg.window, cfg.rows), ratio);
        const nblocks = try std.math.add(usize, (try std.math.add(usize, cap, ratio - 1)) / ratio, 1);
        const canonical = try std.math.mul(usize, at, attn_n * 2 * row_bytes);
        const rows = try std.math.add(usize, try std.math.mul(usize, attn_n, wr), if (cfg.shared_lanes == 0) try std.math.mul(usize, ratio, cfg.slots) else 0);
        const device = try std.math.mul(usize, rows, 2 * row_bytes);
        const words = if (cfg.shared_lanes == 0) try std.math.add(usize, try std.math.mul(usize, attn_n, try std.math.add(usize, try std.math.mul(usize, 2, nblocks), ctl_n)), try std.math.mul(usize, 2, cfg.slots)) else try std.math.mul(usize, attn_n, try std.math.add(usize, nblocks, ctl_n));
        const tables = try std.math.mul(usize, words, 4);
        return .{ .cap = cap, .cfg = cfg, .at = at, .wr = wr, .nblocks = nblocks, .canonical = canonical, .device = device, .tables = tables, .total = try std.math.add(usize, canonical, try std.math.add(usize, device, tables)) };
    }
};

pub const Snapshot = struct {
    layout: SnapshotLayout,
    win0: [attn_n]usize,
    rebases: [attn_n]u64,
    written: [attn_n]u64,

    pub fn validate(image: Snapshot) !void {
        const expected = try SnapshotLayout.init(image.layout.cap, image.layout.cfg, image.layout.at);
        if (!std.meta.eql(expected, image.layout)) return error.KvPageSnapshotLayout;
        for (image.win0) |origin| {
            if (origin % ratio != 0 or origin > image.layout.at or image.layout.at - origin > image.layout.wr) return error.KvPageWindow;
        }
    }
};

/// Counters of one layer (kvp_plan's ctl) plus the host's window moves.
pub const Stats = struct {
    misses: u64 = 0,
    needed: u64 = 0,
    calls: u64 = 0,
    overflow: bool = false,
    rebases: u64 = 0,
    written_rows: u64 = 0,
    epoch: u32 = 0,

    pub fn add(a: *Stats, b: Stats) void {
        a.misses += b.misses;
        a.needed += b.needed;
        a.calls += b.calls;
        a.overflow = a.overflow or b.overflow;
        a.rebases += b.rebases;
        a.written_rows += b.written_rows;
        a.epoch = @max(a.epoch, b.epoch);
    }
};
