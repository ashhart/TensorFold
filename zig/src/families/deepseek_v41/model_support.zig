//! Plain V4.1 execution over resident EXL3 weights and source-owned packed caches.
const builtin = @import("builtin");
const std = @import("std");
const mtl = @import("metal");
const checkpoint = @import("core").checkpoint_metal;
const exl3 = @import("core").exl3_projection;
const config = @import("config.zig");
const load = @import("load.zig");
const weights = @import("weights.zig");
const work = @import("workspace.zig");
pub const attention = @import("attention.zig");
const attention_common = @import("attention_common.zig");
pub const kv = @import("kv.zig");
pub const ops = @import("ops.zig");
pub const mhc = @import("mhc.zig");
pub const csa2 = @import("csa2.zig");
pub const csa2_index = @import("csa2_index.zig");
pub const rope = @import("rope.zig");
const route = @import("route.zig");
const engram = @import("engram.zig");
const engram_stream = @import("engram_rows_safetensors.zig");
const View = @import("view.zig").View;
const gpu_counters = @import("profiler.zig");
const expert_groups = @import("core").expert_groups;
const scan_probe = @import("scan_probe.zig");

// Every shared Zig/Metal argument struct is checked against its Metal twin at comptime, from one registry.
comptime {
    @import("mirror_checked.zig").force();
}

/// Monotonic host nanoseconds (the family's own clock, as batch_prof.zig reads it).
pub fn nowNs() u64 {
    if (comptime builtin.os.tag == .windows) return 0;
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const prefill_step = 15872;
pub const Logits = enum { none, last, all };

/// One slot's rows of a batched forward: the slot's window, its ids, and the position its first row lands at.
pub const Seg = struct { slot: u32, ids: []const u32, p0: u32 };

/// The forward-stage hidden-state hash.
pub const HidHash = struct {
    const Kind = enum(u8) { embed, stream, attn_in, attn, ffn_in, moe, logits, a_qr, a_q, a_kv, a_vals, a_gates, a_latent, a_idxk, a_idxq, a_idxw, a_pos, a_grp, a_sco, a_pos1, a_chunk, a_main };
    const Event = struct { kind: Kind, layer: u32, off: usize, len: usize };
    stage: ?mtl.Buffer = null,
    base: usize = 0,
    bytes: usize = 0,
    sample: u32 = 16,
    full_layer: i32 = -1,
    ev: [3400]Event = undefined,
    nev: usize = 0,

    pub fn on() bool {
        return std.c.getenv("TF_DSV41_HID_HASH") != null;
    }

    pub fn init(rows: u32) HidHash {
        var h = HidHash{};
        if (std.c.getenv("TF_DSV41_HID_ROWS")) |v| h.sample = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 16;
        if (h.sample == 0) h.sample = 1;
        if (h.sample > 16) h.sample = 16;
        if (std.c.getenv("TF_DSV41_HID_FULL")) |v| h.full_layer = std.fmt.parseInt(i32, std.mem.span(v), 10) catch -1;
        const s = @min(rows, h.sample);
        // The stream row is 20,480 bytes, a branch/normalized row 10,240, and the logits row 129,280 f32.
        h.bytes = @as(usize, s) * 6 * 20480 + @as(usize, 40) * 4 * @as(usize, s) * 10240 +
            (if (h.full_layer >= 0) @as(usize, rows) * (5 * 20480 + 106000) + 16 * 8192 * 4 + 8192 * 4 + 512 * 4 + 4096 * 528 + 4096 * 288 else 0) + 129280 * 4;
        return h;
    }

    pub fn rowsFor(h: *const HidHash, rows: u32, layer: u32) u32 {
        if (h.full_layer >= 0 and layer == @as(u32, @intCast(h.full_layer))) return rows;
        return @min(rows, h.sample);
    }

    pub fn record(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, kind: Kind, layer: u32) void {
        const stg = h.stage orelse return;
        const width: u32 = switch (kind) {
            .embed, .stream => 10240,
            else => 5120,
        };
        const n = h.rowsFor(rows, layer);
        // A whole-stage dump is ONE contiguous copy and one event, whatever the chunk's row count.
        if (n == rows and rows != 0) {
            if (h.nev >= h.ev.len) return;
            const len: usize = @as(usize, rows) * width * 2;
            const off = h.base;
            h.base += std.mem.alignForward(usize, len, 256);
            if (h.base > h.bytes) return;
            h.ev[h.nev] = .{ .kind = kind, .layer = layer, .off = off, .len = len };
            h.nev += 1;
            ops.encodeCopy(enc, p, view, .{ .buffer = stg, .offset = off }, .{ .rows = rows, .width = width, .x_stride = width, .y_stride = width });
            return;
        }
        var k: u32 = 0;
        while (k < n) : (k += 1) {
            if (h.nev >= h.ev.len) return;
            const row: u32 = if (n <= 1) 0 else @intCast((@as(usize, k) * (rows - 1)) / (n - 1));
            const len: usize = @as(usize, width) * 2;
            const off = h.base;
            h.base += std.mem.alignForward(usize, len, 256);
            if (h.base > h.bytes) return;
            h.ev[h.nev] = .{ .kind = kind, .layer = layer, .off = off, .len = len };
            h.nev += 1;
            ops.encodeCopy(enc, p, view.shifted(@as(usize, row) * width * 2), .{ .buffer = stg, .offset = off }, .{ .rows = 1, .width = width, .x_stride = width, .y_stride = width });
        }
    }

    /// A region whose row width is a byte count.
    pub fn recordBytes(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, row_bytes: u32, kind: Kind, layer: u32) void {
        const stg = h.stage orelse return;
        if (rows == 0 or row_bytes == 0 or row_bytes % 2 != 0) return;
        if (h.nev >= h.ev.len) return;
        const width = row_bytes / 2;
        const len: usize = @as(usize, rows) * row_bytes;
        const off = h.base;
        h.base += std.mem.alignForward(usize, len, 256);
        if (h.base > h.bytes) return;
        h.ev[h.nev] = .{ .kind = kind, .layer = layer, .off = off, .len = len };
        h.nev += 1;
        ops.encodeCopy(enc, p, view, .{ .buffer = stg, .offset = off }, .{ .rows = rows, .width = width, .x_stride = width, .y_stride = width });
    }

    /// The same with an explicit source row stride, for the tile-major score array.
    pub fn recordStrided(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, x_stride_bytes: u32, width_bytes: u32, kind: Kind, layer: u32) void {
        const stg = h.stage orelse return;
        if (rows == 0 or width_bytes == 0 or width_bytes % 2 != 0 or x_stride_bytes % 2 != 0) return;
        if (h.nev >= h.ev.len) return;
        const len: usize = @as(usize, rows) * width_bytes;
        const off = h.base;
        h.base += std.mem.alignForward(usize, len, 256);
        if (h.base > h.bytes) return;
        h.ev[h.nev] = .{ .kind = kind, .layer = layer, .off = off, .len = len };
        h.nev += 1;
        ops.encodeCopy(enc, p, view, .{ .buffer = stg, .offset = off }, .{ .rows = rows, .width = width_bytes / 2, .x_stride = x_stride_bytes / 2, .y_stride = width_bytes / 2 });
    }

    pub fn recordLogits(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View) void {
        const stg = h.stage orelse return;
        if (h.nev >= h.ev.len) return;
        const len: usize = 129280 * 4;
        const off = h.base;
        h.base += std.mem.alignForward(usize, len, 256);
        if (h.base > h.bytes) return;
        h.ev[h.nev] = .{ .kind = .logits, .layer = 0, .off = off, .len = len };
        h.nev += 1;
        ops.encodeDump(enc, p, view, .{ .buffer = stg, .offset = off }, 129280, false);
    }

    pub fn fnvMore(h: u64, bytes: []const u8) u64 {
        var x = h;
        for (bytes) |b| {
            x ^= b;
            x *%= 0x100000001b3;
        }
        return x;
    }

    pub fn report(h: *const HidHash) void {
        const stg = h.stage orelse return;
        const raw: [*]const u8 = stg.contents();
        var i: usize = 0;
        while (i < h.nev) {
            const kind = h.ev[i].kind;
            const layer = h.ev[i].layer;
            const tag: []const u8 = switch (kind) {
                .embed => "E",
                .stream => "S",
                .attn_in => "N",
                .attn => "A",
                .ffn_in => "F",
                .moe => "M",
                .logits => "L",
                .a_qr => "qr",
                .a_q => "q",
                .a_kv => "kv",
                .a_vals => "vals",
                .a_gates => "gts",
                .a_latent => "lat",
                .a_idxk => "idxk",
                .a_idxq => "idxq",
                .a_idxw => "idxw",
                .a_pos => "pos",
                .a_grp => "grp",
                .a_sco => "sco",
                .a_pos1 => "poslast",
                .a_chunk => "chunk",
                .a_main => "main",
            };
            var x: u64 = 0xcbf29ce484222325;
            var n: usize = 0;
            const first_off = h.ev[i].off;
            const first_len = h.ev[i].len;
            while (i < h.nev and h.ev[i].kind == kind and h.ev[i].layer == layer) : (i += 1) {
                x = fnvMore(x, raw[h.ev[i].off..][0..h.ev[i].len]);
                n += 1;
            }
            std.log.info("hid {s}{d} {x:0>16} rows {d}", .{ tag, layer, x, n });
            if (kind == .a_pos1 and first_len == 512 * 4) {
                const p32: [*]const i32 = @ptrCast(@alignCast(raw + first_off));
                var b: [4096]u8 = undefined;
                var off: usize = 0;
                for (0..512) |q| {
                    if (off >= b.len) break;
                    const piece = std.fmt.bufPrint(b[off..], "{d} ", .{p32[q]}) catch break;
                    off += piece.len;
                }
                std.log.info("hidposlast {s}", .{b[0..off]});
            }
        }
    }
};

// The drafter's NaN filler rows: the dump's segments in f32 units, adding the draft's key, query, key table and output.
pub const mtp_dbg_raw = 4 * 5 * 5120;
pub const mtp_dbg_qr = mtp_dbg_raw + 5 * 512;
pub const mtp_dbg_qpre = mtp_dbg_qr + 5 * 1280;
pub const mtp_dbg_dense = mtp_dbg_qpre + 5 * 32768;
pub const mtp_dbg_qpost = mtp_dbg_dense + 133 * 512;
pub const mtp_dbg_rot = mtp_dbg_qpost + 5 * 32768;
pub const mtp_dbg_sums = mtp_dbg_rot + 5 * 5120;
pub const mtp_dbg_coll = mtp_dbg_sums + 5 * 512;
pub const mtp_dbg_pre = mtp_dbg_coll + 5 * 5120;
pub const mtp_dbg_pin = mtp_dbg_pre + 5 * 4;
pub const mtp_dbg_prot = mtp_dbg_pin + 5 * 5120;
pub const mtp_dbg_psum = mtp_dbg_prot + 5 * 5120;
pub const mtp_dbg_total = mtp_dbg_psum + 5 * 512;
pub const mtp_dbg_grouped = mtp_dbg_total;
pub const mtp_dbg_total2 = mtp_dbg_grouped + 5 * 8192;
pub const mtp_dbg_woasums = mtp_dbg_total2;
pub const mtp_dbg_total3 = mtp_dbg_woasums + 5 * 1024;
pub const Options = struct {
    dir: []const u8,
    table_dir: ?[]const u8 = null,
    /// Release shards read row by row through a bounded LRU; null keeps the tables resident.
    stream_dir: ?[]const u8 = null,
    /// Cached rows a layer under stream_dir (TF_DSV41_ENGRAM_CACHE overrides).
    stream_rows_cache: u32 = 64,
    token_map: []const u8,
    context: u32 = 0,
    chunk_rows: u32 = prefill_step,
    /// Lane slots, each with its own caches and Engram cursor; 1 keeps the single-lane layout.
    lanes: u32 = 1,
    /// Refuse the load past this many bytes; null admits to Metal's working set minus the margin.
    ram_limit_bytes: ?u64 = null,
};

pub const Pipelines = struct {
    projection: exl3.Pipelines,
    ops: ops.Pipelines,
    mhc: mhc.Pipelines,
    route: route.Pipelines,
    engram: engram.Pipelines,
    attention: attention.Pipelines,
    groups: expert_groups.Pipelines,
    /// The fused hyper-connection boundary (TF_DSV41_NO_HC_BOUNDARY): the rms, mix.
    hc_boundary: ?mhc.BoundaryPipelines = null,

    pub fn init(device: mtl.Device, gpa: std.mem.Allocator, capacity: u32, c: config.Config) !Pipelines {
        var ep = try exl3.Pipelines.loadWith(device, gpa);
        errdefer ep.deinit();
        var op = try ops.Pipelines.load(device);
        errdefer op.deinit();
        var hp = try mhc.Pipelines.load(device);
        errdefer hp.deinit();
        var rp = try route.Pipelines.load(device);
        errdefer rp.deinit();
        var gp = try engram.Pipelines.load(device);
        errdefer gp.deinit();
        var groups = try expert_groups.Pipelines.load(device);
        errdefer groups.deinit();
        const hc_boundary: ?mhc.BoundaryPipelines = if (std.c.getenv("TF_DSV41_NO_HC_BOUNDARY") == null and device.tensorUnits()) try mhc.BoundaryPipelines.load(device) else null;
        const ap = try attention.Pipelines.load(device, capacity, c);
        return .{ .projection = ep, .ops = op, .mhc = hp, .route = rp, .engram = gp, .attention = ap, .groups = groups, .hc_boundary = hc_boundary };
    }

    pub fn deinit(p: *Pipelines) void {
        if (p.hc_boundary) |*bp| bp.deinit();
        p.attention.deinit();
        p.engram.deinit();
        p.route.deinit();
        p.groups.deinit();
        p.mhc.deinit();
        p.ops.deinit();
        p.projection.deinit();
    }
};

pub var taps_on: ?bool = null;
/// TF_DSV41_TAPS: capture the drafter's three hidden-state taps (the draft forward reads them; nothing else does).
pub fn tapsOn() bool {
    if (taps_on) |v| return v;
    taps_on = std.c.getenv("TF_DSV41_TAPS") != null;
    return taps_on.?;
}

pub var markov_bias_off: ?bool = null;
/// TF_DSV41_NO_MARKOV=1 turns the Markov bigram bias OFF: the draft is the head-s own argmax over the logits row.
pub fn mtpStopPoint() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_STOP") orelse return 0;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
}

pub fn markovBiasOffOn() bool {
    if (markov_bias_off) |v| return v;
    const raw = std.c.getenv("TF_DSV41_NO_MARKOV") orelse {
        markov_bias_off = false;
        return false;
    };
    markov_bias_off = !std.mem.eql(u8, std.mem.span(raw), "0");
    return markov_bias_off.?;
}

pub var dspark_true_pos: ?bool = null;
/// The release's DSparkAttention writes each main row at its own ring slot.
pub fn dsparkTruePosOn() bool {
    if (dspark_true_pos) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DSPARK_TRUE_POS") orelse {
        dspark_true_pos = true;
        return true;
    };
    dspark_true_pos = !std.mem.eql(u8, std.mem.span(raw), "0");
    return dspark_true_pos.?;
}

pub var taps_shared_base: ?bool = null;
/// TF_DSV41_TAPS_SHARED_BASE=0 restores the earlier per-layer form.
pub fn tapsSharedBaseOn() bool {
    if (taps_shared_base) |v| return v;
    const raw = std.c.getenv("TF_DSV41_TAPS_SHARED_BASE") orelse {
        taps_shared_base = true;
        return true;
    };
    taps_shared_base = !std.mem.eql(u8, std.mem.span(raw), "0");
    return taps_shared_base.?;
}

pub var partial_txn_on: ?bool = null;
/// TF_DSV41_PARTIAL_TXN: the compressed path's partial-block scratch rides the slot transaction.
pub fn partialTxnOn() bool {
    if (partial_txn_on) |v| return v;
    const raw = std.c.getenv("TF_DSV41_PARTIAL_TXN") orelse return true;
    partial_txn_on = std.mem.eql(u8, std.mem.span(raw), "0") == false;
    return partial_txn_on.?;
}

/// The no-re-feed rollback also restores the window ring's slots that the verify's REJECTED rows overwrote.
pub var ring_keep_on: ?bool = null;
pub fn ringKeepOn() bool {
    if (ring_keep_on) |v| return v;
    const raw = std.c.getenv("TF_DSV41_RING_KEEP") orelse return true;
    ring_keep_on = !std.mem.eql(u8, std.mem.span(raw), "0");
    return ring_keep_on.?;
}
/// The most rows a verify window can carry: the MTP depth cap (3) plus the pending row.
pub const max_ring_rows = 8;

/// The release's draft attention ropes its queries and keys at their own positions.
pub fn mtpRopeAllOn() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_ROPE_ALL") orelse return 2;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2;
    return @min(n, 2);
}

/// TF_DSV41_COND_NOROPE=1 republishes the drafter's conditioning rows without their rope phase.
pub fn condRopeOn() bool {
    const v = std.c.getenv("TF_DSV41_COND_NOROPE") orelse return true;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

/// The release's DSparkAttention reads the ring its conditioning writes (`main_kv` from `main_x`).
pub fn mtpTrunkRing() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_TRUNK_RING") orelse return 0;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
    return @min(n, 2);
}

/// TF_DSV41_COND_NOALIGN=1 restores the earlier advancing cursor; the taps ring's slots are the token positions.
pub fn condAlignOn() bool {
    const v = std.c.getenv("TF_DSV41_COND_NOALIGN") orelse return true;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

pub var cond_rows_cache: ?u32 = null;
pub fn condRowsWanted() u32 {
    if (cond_rows_cache) |v| return v;
    var out: u32 = 128;
    if (std.c.getenv("TF_DSV41_COND_ONE")) |v| {
        if (!std.mem.eql(u8, std.mem.span(v), "0")) out = 0;
    }
    if (out != 0) if (std.c.getenv("TF_DSV41_COND_ROWS")) |v| {
        if (std.mem.eql(u8, std.mem.span(v), "1")) {
            out = 0;
        } else {
            const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 128;
            out = if (n <= 1) 0 else @min(n, 128);
        }
    };
    if (out == 0) {
        std.log.warn("dsv41: the drafter's conditioning is the one-row form (TF_DSV41_COND_ONE=1, or TF_DSV41_COND_ROWS=1): the first-draft acceptance and the novel rate fall. The standing is the 128-row window: leave TF_DSV41_COND_ROWS unset.", .{});
    } else if (out != 128) {
        std.log.warn("dsv41: the drafter's conditioning is narrowed to {d} rows; the standing is 128 and the novel rate falls with the window.", .{out});
    }
    cond_rows_cache = out;
    return out;
}

/// THE ENGINE'S COMPUTE ENCODER IN CONCURRENT DISPATCH MODE: TF_DSV41_CONCURRENT=1 arms it; unset is serial.
pub var concurrent_cache: ?bool = null;
pub fn concurrentOn() bool {
    if (concurrent_cache) |v| return v;
    const v = std.c.getenv("TF_DSV41_CONCURRENT");
    concurrent_cache = if (v) |s| !std.mem.eql(u8, std.mem.span(s), "0") else false;
    return concurrent_cache.?;
}

/// A query row's per-head RMS, for the instrument that compares the engine's query scale with the release's per-head.
pub fn headRms(f: [*]const f32, heads: u32, width: u32) struct { min: f32, max: f32, mean: f32 } {
    var mn: f32 = 1e30;
    var mx: f32 = -1e30;
    var sum: f64 = 0;
    var h: u32 = 0;
    while (h < heads) : (h += 1) {
        var acc: f64 = 0;
        var i: u32 = 0;
        while (i < width) : (i += 1) {
            const v = f[@as(usize, h) * width + i];
            acc += @as(f64, v) * @as(f64, v);
        }
        const r: f32 = @floatCast(@sqrt(acc / @as(f64, width)));
        mn = @min(mn, r);
        mx = @max(mx, r);
        sum += r;
    }
    return .{ .min = mn, .max = mx, .mean = @floatCast(sum / @as(f64, heads)) };
}

/// The top-k of a logits row (the instrument): the drafter's row 0 against the trunk's row 0 at the same position.
pub fn logitsTopK(f: [*]const f32, n: usize, out: *[5]u32, vals: *[5]f32) void {
    @memset(out, 0);
    @memset(vals, 0);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const v = f[i];
        if (!std.math.isFinite(v)) continue;
        var j: usize = 0;
        while (j < 5) : (j += 1) {
            if (v > vals[j]) {
                var t: usize = 4;
                while (t > j) : (t -= 1) {
                    vals[t] = vals[t - 1];
                    out[t] = out[t - 1];
                }
                vals[j] = v;
                out[j] = @intCast(i);
                break;
            }
        }
    }
}

extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "the DSpark defaults are the release form" {
    _ = unsetenv("TF_DSV41_DSPARK_TRUE_POS");
    _ = unsetenv("TF_DSV41_MTP_ROPE_ALL");
    _ = unsetenv("TF_DSV41_MTP_TRUNK_RING");
    dspark_true_pos = null;
    try std.testing.expect(dsparkTruePosOn());
    try std.testing.expectEqual(@as(u32, 2), mtpRopeAllOn());
    try std.testing.expectEqual(@as(u32, 0), mtpTrunkRing());
}
