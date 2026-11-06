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

// Every shared Zig/Metal argument struct is checked against its Metal twin at comptime, from one registry, so a
// struct no forward happens to touch is checked too. A drift is a build error.
comptime {
    @import("mirror_checked.zig").force();
}

/// Monotonic host nanoseconds (the family's own clock, as batch_prof.zig reads it).
fn nowNs() u64 {
    if (comptime builtin.os.tag == .windows) return 0;
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

pub const prefill_step = 15872;
pub const Logits = enum { none, last, all };

/// One slot's rows of a batched forward: the slot's window, its ids, and the position its first row lands at. The
/// batch rung is one-row segments (the shared round's lanes).
pub const Seg = struct { slot: u32, ids: []const u32, p0: u32 };

/// The forward-stage hidden-state hash. Repeated passes of one prompt in one process must read one hash a stage;
/// the first tag that moves names the first divergent forward stage. `TF_DSV41_HID_HASH=1` enables it,
/// `TF_DSV41_HID_ROWS=n` (default 16) samples n rows a stage, `TF_DSV41_HID_FULL=<layer>` copies every row of that
/// layer's stages.
const HidHash = struct {
    const Kind = enum(u8) { embed, stream, attn_in, attn, ffn_in, moe, logits, a_qr, a_q, a_kv, a_vals, a_gates, a_latent, a_idxk, a_idxq, a_idxw, a_pos, a_grp, a_sco, a_pos1, a_chunk, a_main };
    const Event = struct { kind: Kind, layer: u32, off: usize, len: usize };
    stage: ?mtl.Buffer = null,
    base: usize = 0,
    bytes: usize = 0,
    sample: u32 = 16,
    full_layer: i32 = -1,
    ev: [3400]Event = undefined,
    nev: usize = 0,

    fn on() bool {
        return std.c.getenv("TF_DSV41_HID_HASH") != null;
    }

    fn init(rows: u32) HidHash {
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

    fn rowsFor(h: *const HidHash, rows: u32, layer: u32) u32 {
        if (h.full_layer >= 0 and layer == @as(u32, @intCast(h.full_layer))) return rows;
        return @min(rows, h.sample);
    }

    fn record(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, kind: Kind, layer: u32) void {
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

    /// A region whose row width is a byte count (the attention workspace's fixed per-row regions, the
    /// selection's pick array). Whole layer only; one contiguous copy.
    fn recordBytes(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, row_bytes: u32, kind: Kind, layer: u32) void {
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

    /// The same with an explicit source row stride, for the tile-major score array: the whole visible
    /// prefix of every row in one copy (packed destination rows).
    fn recordStrided(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View, rows: u32, x_stride_bytes: u32, width_bytes: u32, kind: Kind, layer: u32) void {
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

    fn recordLogits(h: *HidHash, enc: mtl.ComputeEncoder, p: *const ops.Pipelines, view: View) void {
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

    fn fnvMore(h: u64, bytes: []const u8) u64 {
        var x = h;
        for (bytes) |b| {
            x ^= b;
            x *%= 0x100000001b3;
        }
        return x;
    }

    fn report(h: *const HidHash) void {
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

// The drafter's NaN filler rows. The dump's segments, in f32 units: the four the row dump already had (embed,
// attn_in, attn_out, moe_in), then the draft's own key rows, its query's two stages, the decoded key table the scan
// reads, and the attention's output.
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

const Pipelines = struct {
    projection: exl3.Pipelines,
    ops: ops.Pipelines,
    mhc: mhc.Pipelines,
    route: route.Pipelines,
    engram: engram.Pipelines,
    attention: attention.Pipelines,
    groups: expert_groups.Pipelines,
    /// The fused hyper-connection boundary (TF_DSV41_HC_BOUNDARY): the rms, mix, collapse and post of a
    /// boundary in two launches instead of five. Null keeps the shipped path.
    hc_boundary: ?mhc.BoundaryPipelines = null,

    fn init(device: mtl.Device, gpa: std.mem.Allocator, capacity: u32, c: config.Config) !Pipelines {
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

    fn deinit(p: *Pipelines) void {
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

var taps_on: ?bool = null;
/// TF_DSV41_TAPS: capture the drafter's three hidden-state taps (the draft forward reads them; nothing else does).
fn tapsOn() bool {
    if (taps_on) |v| return v;
    taps_on = std.c.getenv("TF_DSV41_TAPS") != null;
    return taps_on.?;
}

var markov_bias_off: ?bool = null;
/// TF_DSV41_NO_MARKOV=1 turns the Markov bigram bias OFF: the draft is the head-s own argmax over the logits row.
/// The arm separates 'the head is weak' from 'the bias poisons the pick'. The drafter's stop point (0 = off). See
/// `proposeTokens`: the arm ends the drafter's forward at a point and COMMITS what was encoded, so the arm's cost
/// is the drafter's prefix.
fn mtpStopPoint() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_STOP") orelse return 0;
    return std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
}

fn markovBiasOffOn() bool {
    if (markov_bias_off) |v| return v;
    const raw = std.c.getenv("TF_DSV41_NO_MARKOV") orelse {
        markov_bias_off = false;
        return false;
    };
    markov_bias_off = !std.mem.eql(u8, std.mem.span(raw), "0");
    return markov_bias_off.?;
}

var dspark_true_pos: ?bool = null;
/// TF_DSV41_DSPARK_TRUE_POS=1 publishes the drafter's context at the capture's own positions: each captured row
/// keeps its position and the drafter's own rows are roped at the next token's. The release's DSparkAttention
/// ropes the draft rows at the positions after the main sequence and writes each main row at its own ring slot
/// (`start_pos % win`), so this is the release's form; unset keeps the earlier prefix/shifted placement for a
/// same-binary A/B.
fn dsparkTruePosOn() bool {
    if (dspark_true_pos) |v| return v;
    const raw = std.c.getenv("TF_DSV41_DSPARK_TRUE_POS") orelse {
        dspark_true_pos = false;
        return false;
    };
    dspark_true_pos = !std.mem.eql(u8, std.mem.span(raw), "0");
    return dspark_true_pos.?;
}

var taps_shared_base: ?bool = null;
/// TF_DSV41_TAPS_SHARED_BASE=0 restores the port's form (the taps ring advances once a TARGET LAYER, so the three
/// layers' rows land in three different ranges and the conditioning mixes three positions). Default on: the shared
/// base is the correct form.
fn tapsSharedBaseOn() bool {
    if (taps_shared_base) |v| return v;
    const raw = std.c.getenv("TF_DSV41_TAPS_SHARED_BASE") orelse {
        taps_shared_base = true;
        return true;
    };
    taps_shared_base = !std.mem.eql(u8, std.mem.span(raw), "0");
    return taps_shared_base.?;
}

var partial_txn_on: ?bool = null;
/// TF_DSV41_PARTIAL_TXN: the compressed path's partial-block scratch rides the slot transaction. Default on; `=0`
/// restores the pre-fix behaviour for a same-binary A/B.
fn partialTxnOn() bool {
    if (partial_txn_on) |v| return v;
    const raw = std.c.getenv("TF_DSV41_PARTIAL_TXN") orelse return true;
    partial_txn_on = std.mem.eql(u8, std.mem.span(raw), "0") == false;
    return partial_txn_on.?;
}

/// The no-re-feed rollback also restores the window ring's slots that the verify's REJECTED rows overwrote. Past
/// position `window` those slots alias positions that are still visible, so without this a partial keep corrupts
/// the next forward's keys (depth 1 never does: its one rejected row's slot aliases a position outside the next
/// window). Default on; the same-binary opt-out `TF_DSV41_RING_KEEP=0` restores the pre-fix behaviour for an A/B.
var ring_keep_on: ?bool = null;
fn ringKeepOn() bool {
    if (ring_keep_on) |v| return v;
    const raw = std.c.getenv("TF_DSV41_RING_KEEP") orelse return true;
    ring_keep_on = !std.mem.eql(u8, std.mem.span(raw), "0");
    return ring_keep_on.?;
}
/// The most rows a verify window can carry: the MTP depth cap (3) plus the pending row, with slack
/// for TF_DSV41_MTP_DEPTH_MAX; the ring stash covers this many slots a layer.
const max_ring_rows = 8;

/// TF_DSV41_MTP_ROPE_ALL=1|2: publish the drafter's ring rows with their rope phase (0: the earlier
/// unroped form). A same-binary A/B for the MTP rewrite; the release's draft attention ropes its queries
/// and keys at their own positions.
fn mtpRopeAllOn() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_ROPE_ALL") orelse return 0;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
    return @min(n, 2);
}

/// TF_DSV41_COND_NOROPE=1 republishes the drafter's conditioning rows without their rope phase.
fn condRopeOn() bool {
    const v = std.c.getenv("TF_DSV41_COND_NOROPE") orelse return true;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_DSV41_MTP_TRUNK_RING: the drafter's attention reads the main model's key ring instead of its own
/// (1: the block's tap layer; 2: the trunk's last layer), which is the release's draft-attention input.
fn mtpTrunkRing() u32 {
    const v = std.c.getenv("TF_DSV41_MTP_TRUNK_RING") orelse return 0;
    const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 0;
    return @min(n, 2);
}

/// TF_DSV41_COND_NOALIGN=1 restores the earlier advancing cursor; the taps ring's slots are the token
/// positions (the capture writes at position % 128).
fn condAlignOn() bool {
    const v = std.c.getenv("TF_DSV41_COND_NOALIGN") orelse return true;
    return std.mem.eql(u8, std.mem.span(v), "0");
}

var cond_rows_cache: ?u32 = null;
fn condRowsWanted() u32 {
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

/// THE ENGINE'S COMPUTE ENCODER IN CONCURRENT DISPATCH MODE. Metal's concurrent dispatch type lets the GPU run
/// independent dispatches at once, and the tree's barrier is already written for that semantics (metal/command.zig:
/// "Later dispatches see every buffer write before this point (concurrent encoders)"). TF_DSV41_CONCURRENT=1 arms
/// it; unset or =0 is the shipped serial path, byte for byte.
var concurrent_cache: ?bool = null;
fn concurrentOn() bool {
    if (concurrent_cache) |v| return v;
    const v = std.c.getenv("TF_DSV41_CONCURRENT");
    concurrent_cache = if (v) |s| !std.mem.eql(u8, std.mem.span(s), "0") else false;
    return concurrent_cache.?;
}

/// A query row's per-head RMS, for the instrument that compares the engine's query scale with the release's
/// per-head query normalization (the release normalizes `qr` before `wq_b`). The query's rows are its heads, so
/// the flat [heads, head_dim] layout is the [rows, heads, head_dim] one.
fn headRms(f: [*]const f32, heads: u32, width: u32) struct { min: f32, max: f32, mean: f32 } {
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

/// Serialized unique owner; submissions finish before return/release, cancellation is between bounded chunks. The
/// top-k of a logits row (the instrument): the drafter's row 0 against the trunk's own row 0 at the same position.
/// The MTP module is trained to predict the trunk's next-token distribution, so a healthy drafter's row 0 must be
/// CLOSE to the trunk's -- a different argmax is fine, a different scale is not.
fn logitsTopK(f: [*]const f32, n: usize, out: *[5]u32, vals: *[5]f32) void {
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

pub const Model = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    device: mtl.Device,
    /// The gate GEMM's f16 activation copy when the pack's router gate is f16 (TF_DSV41_ROUTE_GEMM). One buffer
    /// sized to the chunk, reused a layer: bf16 -> f16 is exact, and the weight keeps its own 10 bits.
    gate_x: ?mtl.Buffer = null,
    /// The gate capture's staging buffers (TF_DSV41_GATE_DUMP): one pair, reused every layer and every token,
    /// because a buffer freed while a command buffer still names it is a GPU invalid resource.
    gate_dump_x: ?mtl.Buffer = null,
    gate_dump_l: ?mtl.Buffer = null,
    queue: mtl.Queue,
    residency: mtl.ResidencySet,
    checkpoint: checkpoint.Checkpoint,
    weights: weights.Model,
    cfg: config.Config,
    cfg_arena: std.heap.ArenaAllocator,
    lanes: u32,
    caches: []kv.cache.State,
    /// Slot 0's state; existing single-lane callers keep working unchanged.
    cache: *kv.cache.State,
    /// The drafter's three KV rings (TF_DSV41_MTP_DRAFTS), built on first use; null until then.
    drafter_state: ?*kv.cache.State = null,
    /// The drafter's one-token id (the embedding kernel reads an ids view); shared, four bytes, written a step.
    drafter_ids: ?mtl.Buffer = null,
    /// The Markov chain's picks (a u32 a step), kept apart from the one-token id buffer the embedding reads.
    markov_ids: ?mtl.Buffer = null,
    /// The Markov chain's per-threadgroup bests (ceil(vocab/256) float2), allocated once with the drafts.
    markov_part: ?mtl.Buffer = null,
    /// The last row the conditioning taps captured: the drafter reads that row's three layers.
    /// The tap ring's next write row and how many of its 128 rows are live (a rolling window across forwards).
    taps_row: u32 = 0,
    /// Rows the last tap capture kept (1..128): the draft conditioning's row count.
    taps_n: u32 = 0,
    /// The rows the LAST tap capture kept. They are the tail of the taps ring and they are consecutive positions,
    /// so the conditioning can publish the window's missing positions at once.
    taps_rows_last: u32 = 0,
    /// The rows of the LAST tap capture that are valid (0 = all of them). The no-re-feed port's rollback keeps a
    /// prefix of the verify's capture: the rejected rows' taps are not the committed history, so the conditioning
    /// publishes the prefix, not the capture's tail.
    taps_rows_valid: u32 = 0,
    /// The adaptive depth policy's signal -- the accepted drafts a round, smoothed. The per-lane signal (the first
    /// draft's acceptance) and its counters.
    mtp_accept_slot: [32]f64 = @splat(0.0),
    mtp_accept_seen_slot: [32]u32 = @splat(0),
    /// The round's per-draft acceptance (kept/offered), the hybrid policy's stay-deep signal.
    mtp_round_slot: [32]f64 = @splat(0.0),
    mtp_round_seen_slot: [32]u32 = @splat(0),
    /// The ring's write row for the CURRENT forward. The three target layers SHARE it: the taps buffer is
    /// interleaved by position (row p = the three layers' values), so the capture's ring advances ONCE a forward.
    /// The port advanced it once a TARGET LAYER, so the three layers' rows landed in three different ranges and the
    /// conditioning's row p mixed three positions (or stale rows).
    taps_base: u32 = 0,
    /// The 128-row conditioning scratch (rows x 5120 bf16) and its packed keys (rows x 528), allocated with the taps.
    cond: ?mtl.Buffer = null,
    cond_wkv: ?mtl.Buffer = null,
    mtp_drafts: [4]u32 = .{ 0, 0, 0, 0 },
    mtp_count: u32 = 0,
    /// The draft round's own logits scratch (eight rows), so its forward never writes the workspace's logits field.
    mtp_logits: ?mtl.Buffer = null,
    /// The drafter's router logits (its own 128 columns a row; the backbone's 384-wide rows would overlap the
    /// drafter's five).
    dspark_logits: ?mtl.Buffer = null,
    /// The trunk's own head input (the verify's row 0), for the drafter-vs-trunk hidden probe.
    mtp_trunk_hidden: ?mtl.Buffer = null,
    /// Each slot's Engram cursor as of its beginSlot, restored by rollbackSlot.
    engram_stash: []u8,
    /// Each slot's compressed-path partial scratch as of its beginSlot (TF_DSV41_PARTIAL_TXN), restored by
    /// rollbackSlot: a rejected window leaves the half-block accumulator holding its last draft row, and the
    /// re-feed of the block that straddles the rollback position would read it.
    partial_stash: []u8,
    /// Each slot's stash of the window ring's next `max_ring_rows` slots, taken before a verify and put back for
    /// the rows the round rejects. `lanes * layers * max_ring_rows * ring_stride` bytes.
    ring_stash: mtl.Buffer,
    /// A MERGED TAIL CHUNK'S STASH. When a pass runs its tail mark inside the chunk (no second forward), the chunk
    /// publishes its PREFIX -- the rows before the mark -- into this slot's copy of the window ring and the taps
    /// ring, then publishes the whole chunk into the live ones as usual; the state at the mark is then copied from
    /// the stash. `publish_limit` is the prefix row count while such a chunk runs (0: no merged chunk), and
    /// `snap_pos`/`snap_hist`/`snap_taps_*` are the mark's re-pointed metadata.
    snap_ring: mtl.Buffer,
    taps_stash: mtl.Buffer,
    publish_limit: u32 = 0,
    snap_pos: u32 = 0,
    snap_hist: [4]u32 = .{ 0, 0, 0, 0 },
    snap_taps_row: u32 = 0,
    snap_taps_n: u32 = 0,
    snap_taps_base: u32 = 0,
    snap_taps_last: u32 = 0,
    /// Rows read from the release's shards a window at a time; null when the tables are resident.
    stream: ?*engram_stream.Rows,
    /// The window's staged rows: two layers of [rows, gather_dim] bf16, CPU-filled.
    stream_stage: ?mtl.Buffer,
    workspace: work.Workspace,
    attention_workspace: attention.Workspace,
    /// The drafter's taps: three backbone layers' hidden states, a row's worth of positions at a time.
    taps: mtl.Buffer,
    /// Layer 0's wq_b input (the post-q_norm `qr`), for the GPTQ calibration dump. Null unless
    /// TF_DSV41_DUMP_WQB_X is set.
    wqb_x_debug: ?mtl.Buffer = null,
    /// Layer 0's MoE input (the experts' output-error dump).
    moe_x_debug: ?mtl.Buffer = null,
    moe_x_rows: u32 = 0,
    pipelines: Pipelines,
    admitted_bytes: usize,
    load_seconds: f64 = 0,
    last_gpu_seconds: f64 = 0,
    logits_rows: usize = 0,
    valid: bool = true,
    profile: ?*gpu_counters.Recorder = null,
    /// The fused boundary's stream parity and the pending expand (TF_DSV41_HC_BOUNDARY).
    stream_alt: bool = false,
    pending_expand: bool = false,
    host_prof: bool = false,
    host_hash_ns: u64 = 0,
    host_stage_ns: u64 = 0,
    host_stage_misses: u64 = 0,
    host_stage_faults: u64 = 0,
    host_stage_pf: u8 = 0,
    host_stage_thr: u16 = 0,
    /// The adaptive Engram gather's state. The decode-sized list starts single-threaded; two consecutive slow small
    /// gathers (cold pages) switch it to the mapped path's workers, and every 16th parallel call one
    /// single-threaded gather probes whether the stream has turned warm. So the warm band never leaves the old
    /// path. TF_DSV41_ENGRAM_ADAPT=0 is the same-binary control.
    engram_par: bool = false,
    /// The thread state machine's probe is not the arbiter: a prefetched window makes the next window's probe look
    /// warm. TF_DSV41_ENGRAM_PF=0 disables the whole arm.
    engram_pf_on: bool = false,
    engram_pf_off: u8 = 0,
    engram_probe_in: u8 = 0,
    engram_probe_every: u8 = 16,
    engram_slow_n: u8 = 0,
    /// The second Engram layer's prompt gather is deferred to its own layer, so the GPU runs the earlier layers
    /// while the read happens. Set by `stageRows` on a prompt-sized chunk.
    engram_pending: bool = false,
    profile_stage_dispatch: [gpu_counters.stage_count]u64 = @splat(0),
    profile_last_count: u64 = 0,
    profile_open_stage: ?gpu_counters.Stage = null,
    grouped_prefill: bool = true,
    observer: ?Observer = null,
    /// The attention's own per-stage observer (the `stagediff` harness). Null on every default path; installing it
    /// only adds captures, never arithmetic.
    attention_observer: ?attention.DiagnosticObserver = null,
    pub const Observer = struct {
        ctx: *anyopaque,
        encode: *const fn (*anyopaque, mtl.ComputeEncoder, u32, gpu_counters.Stage, *const Model) void,
    };

    pub fn open(gpa: std.mem.Allocator, io: std.Io, o: Options) !Model {
        var arena = std.heap.ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const c = try config.Config.read(arena.allocator(), io, o.dir);
        const position_limit: u32 = @intCast(c.rope.max_positions);
        if (o.context > position_limit) return error.ContextFull;
        const capacity = if (o.context == 0) position_limit else o.context;
        const from_config = c.engram_table_dir.items();
        const resident_dir: ?[]const u8 = o.table_dir orelse (if (from_config.len == 0) null else from_config);
        if (o.stream_dir == null and resident_dir == null) return error.MissingEngramTables;
        const plan_dir: ?[]const u8 = if (o.stream_dir == null) (resident_dir orelse return error.MissingEngramTables) else null;
        var plan = try load.Plan.open(gpa, io, o.dir, plan_dir);
        defer plan.deinit();
        const streaming = o.stream_dir != null;
        try weights.validate(&plan, streaming);
        const map_blob = try std.Io.Dir.cwd().readFileAlloc(io, o.token_map, arena.allocator(), .limited(2 << 20));
        const map = try engram.hash.TokenMap.parse(map_blob);
        const lanes = o.lanes;
        // A draft round runs five rows through the same workspace and attention scratch; at decode the plan is one
        // row, so both are widened to the round's width (TF_DSV41_MTP_DRAFTS).
        // A drafted round needs a plan wider than one row: `exact_width` is this plan's rows, and the lane core will
        // not build a window the plan cannot hold. So both drafting knobs widen it, not just the MTP proposer.
        const drafted = std.c.getenv("TF_DSV41_MTP_DRAFTS") != null or std.c.getenv("TF_DSV41_LOOKUP_DRAFTS") != null;
        const round_rows: u32 = if (drafted) @max(o.chunk_rows, 8) else o.chunk_rows;
        const wp = try work.Plan.initLanes(round_rows, lanes);
        const kp = kv.cache.Plan{ .capacity = capacity, .layers = 40 };
        const workspace_bytes = try std.math.add(usize, wp.bytes, attention.Workspace.bytes(o.chunk_rows));
        const cache_bytes = try std.math.add(usize, kp.residentBytes(), weights.sizeofDescriptors());
        const execution_bytes = try std.math.add(usize, workspace_bytes, try std.math.mul(usize, cache_bytes, lanes));
        const rotary_bytes = attention.Pipelines.residentBytes(capacity, c);
        const needed = try std.math.add(usize, plan.bytes, try std.math.add(usize, execution_bytes, rotary_bytes));
        // The int8 twin lives beside the trellis it supersedes at prompt rows; it is additive until the trellis
        // attention weights can be dropped, so the admission must see it.
        const int8_extra: usize = if (std.c.getenv("TF_DSV41_I8_PREFILL") != null) 5 << 30 else 0;
        const needed_all = try std.math.add(usize, needed, int8_extra);
        if (o.ram_limit_bytes) |limit| if (needed_all > limit) {
            std.log.err("the resident payload needs {d} bytes; the load limit is {d}", .{ needed_all, limit });
            return error.LoadLimitExceeded;
        };
        const load_started = std.Io.Timestamp.now(io, .awake);
        const device = try mtl.Device.init();
        errdefer device.deinit();
        const margin: u64 = 8 << 30;
        const available = device.maxWorkingSet() -| device.allocated() -| margin;
        if (needed_all > available) {
            std.log.err("full residency needs {d} bytes plus {d} margin; Metal admission allows {d}", .{ needed_all, margin, available });
            return error.InsufficientResidentMemory;
        }
        const max_buffer = mtl.objc.msg(u64, device.id, "maxBufferLength", .{});
        for (plan.shards.items) |shard| if (shard.end - shard.begin > max_buffer) return error.ShardExceedsMetalBufferLimit;
        std.log.info("full resident admission: weights={d}, total={d}, context={d}, headroom={d}", .{ plan.bytes, needed, capacity, available - needed });
        var pipelines = try Pipelines.init(device, gpa, capacity, c);
        errdefer pipelines.deinit();
        const queue = try device.queue();
        errdefer queue.deinit();
        const residency = try device.residencySet(plan.shards.items.len + 128);
        errdefer residency.deinit();
        var cp = try plan.materialize(gpa, device, residency, queue);
        errdefer cp.deinit();
        var bound = try weights.bind(device, &cp, streaming);
        errdefer bound.deinit();
        // The prompt-row int8 sidecar. Flag-gated; the trellis matrices stay bound for decode and for any chunk
        // under the rows gate.
        if (std.c.getenv("TF_DSV41_I8_PREFILL") != null) {
            const sidecar = std.c.getenv("TF_DSV41_I8_DIR") orelse return error.MissingInt8Sidecar;
            try weights.attachInt8(gpa, io, device, &bound, std.mem.span(sidecar));
        }
        const caches = try gpa.alloc(kv.cache.State, lanes);
        errdefer gpa.free(caches);
        const engram_stash = try gpa.alloc(u8, lanes * 16);
        errdefer gpa.free(engram_stash);
        @memset(engram_stash, 0);
        const partial_stash = try gpa.alloc(u8, lanes * kv.cache.Plan.partialBytes(kp));
        errdefer gpa.free(partial_stash);
        @memset(partial_stash, 0);
        const ring_stash_bytes = @as(usize, lanes) * kp.layers * max_ring_rows * kv.cache.ring_stride;
        const ring_stash = try device.buffer(ring_stash_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer ring_stash.deinit();
        @memset(ring_stash.contents()[0..ring_stash_bytes], 0);
        for (caches) |*state| {
            state.* = try kv.cache.State.init(device, kp);
            errdefer state.deinit();
        }
        var workspace = try work.Workspace.init(device, wp, map.bytes);
        errdefer workspace.deinit();
        var aw = try attention.Workspace.init(device, round_rows);
        errdefer aw.deinit();
        const taps = try device.buffer(3 * 128 * 5120 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer taps.deinit();
        const snap_ring = try device.buffer(@as(usize, lanes) * kp.layers * kv.cache.window * kv.cache.ring_stride, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer snap_ring.deinit();
        @memset(snap_ring.contents()[0..snap_ring.length()], 0);
        const taps_stash = try device.buffer(3 * 128 * 5120 * 2, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer taps_stash.deinit();
        @memset(taps_stash.contents()[0..taps_stash.length()], 0);
        // The conditioning scratch: 128 rows of 5120 bf16, then 128 rows of 512 bf16 keys, then 128 packed rows.
        // The draft round's own logits scratch: a five-row forward keeps its head rows and Markov reads here, not
        // in the workspace's logits field (one row at decode, which a draft forward overran -- round 212).
        const mtp_logits = try device.buffer(8 * 129280 * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        const cond_bytes = 128 * 5120 * 2 + 128 * 512 * 2 + 128 * 528;
        const cond = try device.buffer(cond_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        // The fused `wkv` dispatch's scratch. The input window is written three times (one per block), the output
        // comes back as three 128-row windows, and the ids are constant: routes name the block, the input and
        // output maps are the identity (the grouped tile's rows are the descriptors' own rows).
        const cond_wkv_bytes = 3 * 128 * 5120 * 2 + 3 * 128 * 512 * 2 + 3 * 3 * 128 * 4;
        const cond_wkv = try device.buffer(cond_wkv_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        @memset(cond_wkv.contents()[0..cond_wkv_bytes], 0);
        {
            const ids: [*]u32 = @ptrCast(@alignCast(cond_wkv.contents() + 3 * 128 * 5120 * 2 + 3 * 128 * 512 * 2));
            for (0..384) |r| {
                ids[r] = @intCast(r / 128);
                ids[384 + r] = @intCast(r);
                ids[768 + r] = @intCast(r);
            }
        }
        errdefer cond_wkv.deinit();
        errdefer taps.deinit();
        residency.add(taps);
        residency.add(cond_wkv);
        var stream_rows: ?*engram_stream.Rows = null;
        var stream_stage: ?mtl.Buffer = null;
        // Invariant: this block is the only place stream_rows and stream_stage are created, and weights.bind nulls
        // Engram.table/scales on the same condition. Keep the three in step: no live Model may have a null table
        // with a null stage, or the serve would gather from nothing.
        if (o.stream_dir) |dir| {
            const engram_layers = [_]u32{ engram.hash.layers[0].layer_id, engram.hash.layers[1].layer_id };
            const cap: u32 = if (std.c.getenv("TF_DSV41_ENGRAM_CACHE")) |v| std.fmt.parseInt(u32, std.mem.span(v), 10) catch o.stream_rows_cache else o.stream_rows_cache;
            stream_rows = try engram_stream.Rows.open(gpa, io, dir, 0, 1, &engram_layers, cap);
            errdefer stream_rows.?.close();
            const stage_bytes = try std.math.mul(usize, wp.rows, 2 * engram.gather_dim * @sizeOf(u16));
            stream_stage = try device.buffer(stage_bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            std.log.info("engram stream: rows from {s}, {d} cached a layer, {d} staging bytes a layer", .{ dir, cap, stage_bytes / 2 });
        }
        bound.registerResidency(residency);
        for (caches) |*state| state.registerResidency(residency);
        residency.add(workspace.buffer);
        residency.add(workspace.shared_maps);
        aw.registerResidency(residency);
        if (stream_stage) |buf| residency.add(buf);
        residency.add(ring_stash);
        pipelines.attention.registerResidency(residency);
        // Layer 0's MoE input, 2,048 rows x 5,120 f32.
        var moe_x_debug: ?mtl.Buffer = null;
        if (std.c.getenv("TF_DSV41_DUMP_MOE_X") != null) {
            const b = try device.buffer(2048 * 5120 * 4, mtl.ResourceOptions.shared);
            errdefer b.deinit();
            moe_x_debug = b;
            residency.add(b);
        }
        // Layer 0's wq_b input, 2,048 rows x 1,280 f32, for the GPTQ Hessian.
        var wqb_x_debug: ?mtl.Buffer = null;
        if (std.c.getenv("TF_DSV41_DUMP_WQB_X") != null) {
            const b = try device.buffer(2048 * 1280 * 4, mtl.ResourceOptions.shared);
            errdefer b.deinit();
            wqb_x_debug = b;
            residency.add(b);
            attention.wqb_x_debug = b;
        }
        residency.commit();
        residency.requestResidency();
        queue.addResidencySet(residency);
        return .{ .gpa = gpa, .io = io, .device = device, .queue = queue, .residency = residency, .checkpoint = cp, .weights = bound, .cfg = c, .cfg_arena = arena, .lanes = lanes, .caches = caches, .cache = &caches[0], .engram_stash = engram_stash, .partial_stash = partial_stash, .ring_stash = ring_stash, .snap_ring = snap_ring, .taps_stash = taps_stash, .stream = stream_rows, .stream_stage = stream_stage, .workspace = workspace, .attention_workspace = aw, .taps = taps, .cond = cond, .cond_wkv = cond_wkv, .wqb_x_debug = wqb_x_debug, .moe_x_debug = moe_x_debug, .mtp_logits = mtp_logits, .pipelines = pipelines, .admitted_bytes = needed, .load_seconds = @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds() - load_started.toNanoseconds())) / 1e9 };
    }

    pub fn deinit(m: *Model) void {
        m.queue.removeResidencySet(m.residency);
        m.residency.deinit();
        if (m.drafter_ids) |ids| ids.deinit();
        if (m.markov_part) |part| part.deinit();
        if (m.drafter_state) |state| {
            state.deinit();
            m.gpa.destroy(state);
        }
        if (m.stream) |rows| rows.close();
        if (m.stream_stage) |buf| buf.deinit();
        m.attention_workspace.deinit();
        m.workspace.deinit();
        for (m.caches) |*state| state.deinit();
        m.gpa.free(m.caches);
        m.gpa.free(m.engram_stash);
        m.gpa.free(m.partial_stash);
        m.ring_stash.deinit();
        m.snap_ring.deinit();
        m.taps_stash.deinit();
        if (m.gate_x) |b| b.deinit();
        if (m.dspark_logits) |b| b.deinit();
        if (m.cond_wkv) |b| b.deinit();
        if (m.wqb_x_debug) |b| {
            b.deinit();
            attention.wqb_x_debug = null;
        }
        if (m.moe_x_debug) |b| b.deinit();
        m.weights.deinit();
        m.checkpoint.deinit();
        m.pipelines.deinit();
        m.queue.deinit();
        m.device.deinit();
        m.cfg_arena.deinit();
    }

    pub fn reset(m: *Model) !void {
        for (0..m.lanes) |slot| try m.resetSlot(@intCast(slot));
    }

    /// One lane slot back to empty: its cursor, its mutable cache bytes and its Engram history.
    pub fn resetSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const state = &m.caches[slot];
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        m.valid = false;
        m.logits_rows = 0;
        state.position = 0;
        m.attention_workspace.selection_layer = null;
        m.attention_workspace.selection_rows = 0;
        m.attention_workspace.candidate_rows = 0;
        m.last_gpu_seconds = 0;
        state.saved_position = null;
        @memset(state.live.contents()[0..state.plan.mutableBytes()], 0);
        const cb = m.queue.commandBufferUnretained();
        const enc = cb.compute(.serial);
        engram.encodeReset(enc, &m.pipelines.engram, try m.workspace.historyView(slot));
        enc.end();
        cb.commit();
        cb.wait();
        try completed(cb);
        m.valid = true;
        m.logits_rows = 0;
    }

    /// A KEPT PROMPT STATE. One slot's whole cache -- the packed main/index tables and the mutable bytes (the
    /// window rings, the compressed path's partials and the raw-row captures) -- plus its cursor, the four-token
    /// Engram history the next chunk's hash pass reads, and the taps ring's rows and cursor (the drafter's
    /// conditioning). Every copied buffer is `shared` and every forward that wrote it committed and waited, so the
    /// copy is exactly the state a pass at `position` stands on: a restored state continues the prompt where the
    /// saved one stood, and `resumed` equals `fresh` token for token.
    pub const Snapshot = struct {
        storage: []u8,
        live: []u8,
        taps: []u8,
        position: u32,
        history: [4]u32,
        taps_row: u32,
        taps_n: u32,
        taps_rows_last: u32,
        taps_rows_valid: u32,
        taps_base: u32,
    };

    /// START A MERGED TAIL CHUNK. `hist` ends at `at` (the last prompt tokens, newest last), the chunk starts at
    /// `from` and runs past `at`. The stash is primed with the live ring and taps -- a prefix shorter than the
    /// window keeps the older slots -- the mark's metadata is recorded, and `publish_limit` makes the chunk's
    /// per-layer publish and tap capture also write its PREFIX into the stash. Returns false when the mark's
    /// history cannot be built; the caller then keeps the cut.
    pub fn beginSnapTail(m: *Model, slot: u32, hist: []const u32, at: u32, from: u32) bool {
        if (hist.len < 3) return false;
        const state = &m.caches[slot];
        const q = at - from;
        const n: u32 = @min(q, kv.cache.window);
        // THE RECORD IS THE HASH PASS'S, NOT RAW TOKENS: `int tail[3]` of COMPRESSED ids, newest first, then `uint
        // available` (saturates at 3). A pass standing at `at` holds the three tokens before it through the same
        // token map the hash pass reads (The first form wrote raw prompt tokens here: the Engram history at the
        // resumed position then read 77/304/223 instead of 2460/174/259 and the turn diverged).
        const tm = m.workspace.view(.token_map);
        const map = engram.hash.TokenMap{ .bytes = (tm.buffer.contents() + tm.offset)[0 .. engram.hash.raw_vocab * 4] };
        var tail: [3]u32 = .{ 0, 0, 0 };
        for (0..3) |k| tail[k] = map.get(hist[hist.len - 1 - k]) catch return false;
        m.publish_limit = q;
        m.snap_pos = at;
        for (&m.snap_hist, 0..) |*v, k| v.* = if (k < 3) tail[k] else 3;
        m.snap_taps_base = if (condAlignOn()) from % kv.cache.window else m.taps_row;
        m.snap_taps_row = (m.taps_row + n) % kv.cache.window;
        m.snap_taps_n = @min(m.taps_n + n, kv.cache.window);
        m.snap_taps_last = n;
        const ring_bytes = @as(usize, state.plan.layers) * kv.cache.window * kv.cache.ring_stride;
        @memcpy(m.snap_ring.contents()[@as(usize, slot) * ring_bytes ..][0..ring_bytes], state.live.contents()[0..ring_bytes]);
        @memcpy(m.taps_stash.contents()[0..m.taps.length()], m.taps.contents()[0..m.taps.length()]);
        m.attention_workspace.publish_limit = q;
        m.attention_workspace.publish_stash = .{ .buffer = m.snap_ring, .offset = @as(usize, slot) * ring_bytes };
        return true;
    }

    /// A merged tail chunk re-points the taps ring BY POSITION; the cursor arm (TF_DSV41_COND_NOALIGN=1) is not
    /// position-addressed, so the merge stays off there.
    pub fn snapTailAllowed(m: *const Model) bool {
        _ = m;
        return condAlignOn();
    }

    pub fn endSnapTail(m: *Model) void {
        m.publish_limit = 0;
        m.attention_workspace.publish_limit = 0;
        m.attention_workspace.publish_stash = null;
    }

    /// The bytes one kept state holds: fixed, whatever its position (the slot's whole cache).
    pub fn snapshotBytes(m: *const Model) u64 {
        const plan = m.caches[0].plan;
        return @as(u64, plan.packedBytes()) + plan.mutableBytes() + m.taps.length();
    }

    pub fn snapshotSave(m: *Model, slot: u32) !*Snapshot {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const state = &m.caches[slot];
        const snap = try m.gpa.create(Snapshot);
        errdefer m.gpa.destroy(snap);
        const storage = try m.gpa.dupe(u8, state.storage.contents()[0..state.plan.packedBytes()]);
        errdefer m.gpa.free(storage);
        const live = try m.gpa.dupe(u8, state.live.contents()[0..state.plan.mutableBytes()]);
        errdefer m.gpa.free(live);
        // A merged tail chunk's copy is THE STATE AT THE MARK, not at the chunk end: the taps ring and the window
        // ring come from the stash the chunk's prefix published, the four-token history is the caller's, and the
        // compressed partials at a tile mark are empty.
        const merging = m.publish_limit != 0;
        const taps_src = if (merging) m.taps_stash else m.taps;
        const taps = try m.gpa.dupe(u8, taps_src.contents()[0..m.taps.length()]);
        errdefer m.gpa.free(taps);
        const view = try m.workspace.historyView(slot);
        const head: [*]const u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        var history: [4]u32 = undefined;
        for (&history, 0..) |*v, i| v.* = std.mem.readInt(u32, head[i * 4 ..][0..4], .little);
        snap.* = .{
            .storage = storage,
            .live = live,
            .taps = taps,
            .position = if (merging) m.snap_pos else state.position,
            .history = if (merging) m.snap_hist else history,
            .taps_row = if (merging) m.snap_taps_row else m.taps_row,
            .taps_n = if (merging) m.snap_taps_n else m.taps_n,
            .taps_rows_last = if (merging) m.snap_taps_last else m.taps_rows_last,
            .taps_rows_valid = if (merging) 0 else m.taps_rows_valid,
            .taps_base = if (merging) m.snap_taps_base else m.taps_base,
        };
        if (merging) {
            const ring_bytes = @as(usize, state.plan.layers) * kv.cache.window * kv.cache.ring_stride;
            const from = @as(usize, slot) * ring_bytes;
            @memcpy(live[0..ring_bytes], m.snap_ring.contents()[from..][0..ring_bytes]);
            const off = kv.cache.Plan.partialOffset(state.plan);
            if (off < live.len) @memset(live[off..], 0);
        }
        if (std.c.getenv("TF_DSV41_SNAP_HASH") != null) {
            const ring_bytes = @as(usize, state.plan.layers) * kv.cache.window * kv.cache.ring_stride;
            var h = std.hash.Wyhash.init(0);
            h.update(live[0..ring_bytes]);
            var hl = std.hash.Wyhash.init(1);
            hl.update(live[0 .. kv.cache.window * kv.cache.ring_stride]);
            var hs = std.hash.Wyhash.init(2);
            hs.update(storage);
            var ht = std.hash.Wyhash.init(3);
            ht.update(taps);
            std.log.info("snap hash: pos {d} merging {} ring {x} layer0 {x} storage {x} taps {x} hist {d} {d} {d} {d} tapsrow {d} tapsn {d} last {d}", .{ snap.position, merging, h.final(), hl.final(), hs.final(), ht.final(), snap.history[0], snap.history[1], snap.history[2], snap.history[3], snap.taps_row, snap.taps_n, snap.taps_rows_last });
        }
        return snap;
    }

    pub fn snapshotDrop(m: *Model, snap: *Snapshot) void {
        m.gpa.free(snap.storage);
        m.gpa.free(snap.live);
        m.gpa.free(snap.taps);
        m.gpa.destroy(snap);
    }

    /// Make one slot's cache the saved state's: the next chunk starts at its position. The caller resets the
    /// slot first, so nothing a previous request left where the state does not cover it survives.
    pub fn snapshotRestore(m: *Model, slot: u32, snap: *const Snapshot) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const state = &m.caches[slot];
        if (snap.storage.len != state.plan.packedBytes() or snap.live.len != state.plan.mutableBytes() or snap.taps.len != m.taps.length()) return error.SnapshotLayout;
        if (snap.position > state.plan.capacity) return error.SnapshotOutOfRange;
        @memcpy(state.storage.contents()[0..snap.storage.len], snap.storage);
        @memcpy(state.live.contents()[0..snap.live.len], snap.live);
        @memcpy(m.taps.contents()[0..snap.taps.len], snap.taps);
        const view = try m.workspace.historyView(slot);
        const head: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        for (snap.history, 0..) |v, i| std.mem.writeInt(u32, head[i * 4 ..][0..4], v, .little);
        state.position = snap.position;
        state.saved_position = null;
        m.taps_row = snap.taps_row;
        m.taps_n = snap.taps_n;
        m.taps_rows_last = snap.taps_rows_last;
        m.taps_rows_valid = snap.taps_rows_valid;
        m.taps_base = snap.taps_base;
        m.attention_workspace.selection_layer = null;
        m.attention_workspace.selection_rows = 0;
        m.attention_workspace.candidate_rows = 0;
        m.valid = true;
    }

    /// A keep/rollback transaction over one slot's cache cursor and its 16-byte Engram history.
    pub fn beginSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const view = try m.workspace.historyView(slot);
        const history: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        @memcpy(m.engram_stash[slot * 16 ..][0..16], history[0..16]);
        try m.caches[slot].begin();
        // Stash the window ring's next slots before the verify writes them. The keep path puts back the rows the
        // round rejects; without it a wrapped ring's stale slots alias positions the next forward still reads.
        if (ringKeepOn()) {
            const state = &m.caches[slot];
            const stash_bytes = @as(usize, state.plan.layers) * max_ring_rows * kv.cache.ring_stride;
            const cb = m.queue.commandBufferUnretained();
            var enc = cb.compute(.serial);
            var enc_open = true;
            errdefer if (enc_open) enc.end();
            const ring = View{ .buffer = state.live, .offset = 0 };
            const stash = View{ .buffer = m.ring_stash, .offset = @as(usize, slot) * stash_bytes };
            try m.pipelines.attention.cache.encodeRingMove(enc, stash, ring, .{ .layers = state.plan.layers, .move_rows = max_ring_rows, .max_rows = max_ring_rows, .window = kv.cache.window, .position = state.position, .stride_words = kv.cache.ring_stride / 4, .first = 0, .to_stash = 1 });
            enc.end();
            enc_open = false;
            cb.commit();
            cb.wait();
            try completed(cb);
        }
        if (partialTxnOn()) {
            const state = &m.caches[slot];
            const len = kv.cache.Plan.partialBytes(state.plan);
            const base = kv.cache.Plan.partialOffset(state.plan);
            @memcpy(m.partial_stash[@as(usize, slot) * len ..][0..len], state.live.contents()[base .. base + len]);
        }
    }

    pub fn commitSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        try m.caches[slot].commit();
    }

    pub fn rollbackSlot(m: *Model, slot: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        const view = try m.workspace.historyView(slot);
        const history: [*]u8 = @ptrCast(@alignCast(view.buffer.contents() + view.offset));
        try m.caches[slot].rollback();
        if (partialTxnOn()) {
            const state = &m.caches[slot];
            const len = kv.cache.Plan.partialBytes(state.plan);
            const base = kv.cache.Plan.partialOffset(state.plan);
            @memcpy(state.live.contents()[base .. base + len], m.partial_stash[@as(usize, slot) * len ..][0..len]);
        }
        @memcpy(history[0..16], m.engram_stash[slot * 16 ..][0..16]);
    }

    /// The no-re-feed port. It restores the compressed path's open row from the capture (the group that straddles
    /// the new position), advances the Engram's cursor over the kept rows, and marks the taps' valid prefix (the
    /// rejected rows' taps are not the committed history).
    pub fn portKeep(m: *Model, slot: u32, kept: u32) !void {
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        // The adaptive depth policy's signal. The verify's kept rows are the accepted drafts plus the pending
        // token, so `kept - 1` is the accepted drafts this round. The policy is a POLICY (which depth to run), not
        // a numerics change: it picks between two already-characterised windows (depth 1 is token-for-token
        // identical to the plain path; depth 3 is the standing one).
        const state = &m.caches[slot];
        // The rollback put the cursor back before the window; the re-feed's forward is what advanced it
        // over the kept rows, so the port advances it the same way.
        const pos = state.position + kept;
        // The pool's groups are cut at even positions, so a new position that is odd leaves the group
        // {pos-1, pos} open: the open row (pos-1) must be the row the capture holds.
        if (kept > 0 and pos % 2 == 1) {
            const e = m.queue.commandBufferUnretained();
            var enc = e.compute(.serial);
            for ([_]u32{ 2, 8, 14 }) |owner| {
                const source = try state.source(owner, 2);
                const at = source.capture.offset + @as(usize, (pos - 1) % kv.cache.capture_rows) * 2 * kv.cache.capture_width * 4;
                const values = View{ .buffer = source.capture.buffer, .offset = at };
                const gates = View{ .buffer = source.capture.buffer, .offset = at + kv.cache.capture_width * 4 };
                try m.pipelines.attention.cache.restorePool(enc, values, gates, source.partial, .{ .rows = 1, .width = kv.cache.capture_width, .ratio = 2, .position = pos - 1, .stride = kv.cache.capture_width, .input_f32 = 1, .eps = m.cfg.eps });
            }
            enc.end();
            e.commit();
            e.wait();
            try completed(e);
        }
        try state.append(kept);
        // The Engram's cursor: the re-feed's hash pass advanced it over the kept rows, and the port does
        // the same, on the window's own ids (the first `kept` of them).
        if (m.stream != null and std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null) try m.hashPass(slot, kept);
        // The taps: the conditioning publishes the capture's valid prefix (the kept rows).
        m.taps_rows_valid = kept;
    }

    /// Put back the window ring slots that the verify's rejected rows wrote. Called after `rollbackSlot` (so
    /// `position` is the window's start) and before the next forward. The stash was taken by `beginSlot`; rows
    /// `kept..rows-1` are the rejected ones.
    pub fn restoreRing(m: *Model, slot: u32, kept: u32, rows: u32) !void {
        if (!ringKeepOn()) return;
        if (slot >= m.lanes or rows > max_ring_rows or kept > rows) return;
        const state = &m.caches[slot];
        const move = rows - kept;
        if (move == 0) return;
        const stash_bytes = @as(usize, state.plan.layers) * max_ring_rows * kv.cache.ring_stride;
        const cb = m.queue.commandBufferUnretained();
        var enc = cb.compute(.serial);
        var enc_open = true;
        errdefer if (enc_open) enc.end();
        const ring = View{ .buffer = state.live, .offset = 0 };
        const stash = View{ .buffer = m.ring_stash, .offset = @as(usize, slot) * stash_bytes };
        try m.pipelines.attention.cache.encodeRingMove(enc, ring, stash, .{ .layers = state.plan.layers, .move_rows = move, .max_rows = max_ring_rows, .window = kv.cache.window, .position = state.position, .stride_words = kv.cache.ring_stride / 4, .first = kept, .to_stash = 0 });
        enc.end();
        enc_open = false;
        cb.commit();
        cb.wait();
        try completed(cb);
    }

    /// One command buffer per chunk; no operator waits or reads routes back to the CPU.
    pub fn forward(m: *Model, ids: []const u32, logits: Logits) ![]const f32 {
        return m.forwardOn(0, ids, logits);
    }

    /// A chunk against lane slot `slot`'s caches and Engram cursor: rows land at that slot's positions.
    pub fn forwardOn(m: *Model, slot: u32, ids: []const u32, logits: Logits) ![]const f32 {
        if (!m.valid) return error.InvalidModelState;
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        return m.forwardSegs(&.{.{ .slot = slot, .ids = ids, .p0 = m.caches[slot].position }}, logits);
    }

    /// One forward over several slots' one-row windows (a shared round's lanes). The row-wise layers run once over
    /// every segment's rows; the attention core runs once a segment against its own slot state. One-row segments
    /// only, and no instruments: the rung exists to amortise the dense attention weights, not to batch prefill or
    /// to run a profiler.
    pub fn forwardSegs(m: *Model, segs: []const Seg, logits: Logits) ![]const f32 {
        if (!m.valid) return error.InvalidModelState;
        if (segs.len == 0 or segs.len > 4) return error.InvalidLaneSlot;
        const first_seg = segs[0];
        const slot = first_seg.slot;
        if (slot >= m.lanes) return error.InvalidLaneSlot;
        var total_rows: u32 = 0;
        for (segs) |sg| {
            if (sg.slot >= m.lanes) return error.InvalidLaneSlot;
            if (sg.ids.len == 0) return error.InvalidPrefillChunk;
            if (sg.p0 != m.caches[sg.slot].position) return error.PositionMismatch;
            if (m.caches[sg.slot].position > m.caches[sg.slot].plan.capacity or
                sg.ids.len > m.caches[sg.slot].plan.capacity - m.caches[sg.slot].position) return error.ContextFull;
            if (total_rows + sg.ids.len > m.workspace.plan.rows) return error.InvalidPrefillChunk;
            total_rows += @intCast(sg.ids.len);
        }
        if (segs.len > 1 and (m.profile != null or m.observer != null or m.stream == null or
            std.c.getenv("TF_DSV41_STAGE_CB") != null or std.c.getenv("TF_DSV41_LAYER_CB") != null or
            std.c.getenv("TF_DSV41_ATTN_PROF") != null or std.c.getenv("TF_DSV41_HID_HASH") != null))
            return error.BatchUnsupported;
        const state = &m.caches[slot];
        const rows: u32 = total_rows;
        if (segs.len > 1 and std.c.getenv("TF_DSV41_BATCH_DEBUG") != null) std.log.info("forwardSegs: segs={d} rows={d}", .{ segs.len, rows });
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) {
            const hv = try m.workspace.historyView(slot);
            const head: [*]const u32 = @ptrCast(@alignCast(hv.buffer.contents() + hv.offset));
            std.log.info("fwd: slot {d} segs {d} rows {d} pos {d} taps {d}/{d} engram {d} {d} {d} {d}", .{ slot, segs.len, rows, first_seg.p0, m.taps_n, m.taps_row, head[0], head[1], head[2], head[3] });
        }
        // A forward starts from the embedding: its stream lives in `.stream` and no sublayer branch is pending.
        // A draft round in between leaves both at whatever its own forward ended on, so they are set here.
        m.stream_alt = false;
        m.pending_expand = false;
        m.engram_pending = false;
        for (segs) |sg| for (sg.ids, 0..) |id, at| if (id >= 129280) {
            std.log.err("forwardOn: unsupported token id {d} at row {d} of {d} (position {d})", .{ id, at, sg.ids.len, sg.p0 });
            return error.UnsupportedToken;
        };
        m.logits_rows = 0;
        {
            const dst = m.workspace.tokenIds(rows);
            var at: usize = 0;
            for (segs) |sg| {
                @memcpy(dst[at..][0..sg.ids.len], sg.ids);
                at += sg.ids.len;
            }
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        // The forward-stage hash's staging buffer, one forward's life.
        var hid = HidHash{};
        var hid_stage: ?mtl.Buffer = null;
        defer if (hid_stage) |b| b.deinit();
        if (HidHash.on()) {
            hid = HidHash.init(rows);
            hid_stage = try m.device.buffer(hid.bytes, mtl.ResourceOptions.shared);
            hid.stage = hid_stage.?;
        }
        // Probe: the sparse scan's own input and output, one layer, one forward's staging.
        var sp_stage: ?mtl.Buffer = null;
        defer if (sp_stage) |b| {
            b.deinit();
            scan_probe.stage = null;
        };
        if (scan_probe.on()) {
            scan_probe.init(rows, @intCast(m.cfg.heads * m.cfg.head_dim));
            sp_stage = try m.device.buffer(scan_probe.cap, mtl.ResourceOptions.shared);
            scan_probe.stage = sp_stage.?;
            scan_probe.reset();
        }
        if (m.stream != null and std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null) {
            const hash_start: u64 = if (m.host_prof) nowNs() else 0;
            m.host_stage_ns = 0;
            m.host_stage_misses = 0;
            m.host_stage_faults = 0;
            if (segs.len == 1) try m.hashPass(slot, rows) else try m.hashPassSegs(segs, rows);
            if (m.host_prof) m.host_hash_ns = nowNs() - hash_start;
        }
        const host_enc_start: u64 = nowNs();
        const dispatches_before = mtl.command.dispatch_count.load(.monotonic);
        // TF_DSV41_LAYER_CB: one command buffer a layer, with the GPU's own seconds for each (the counter recorder
        // is broken on this build).
        const layer_cb = std.c.getenv("TF_DSV41_LAYER_CB") != null;
        // THE PIPELINED FORWARD. When TF_DSV41_PIPE_LAYERS=n (default 2; 0 = the shipped single buffer) the first n
        // layers are committed as their own command buffer WITHOUT a wait, so the GPU starts on them while the host
        // encodes the rest. The landed 2 is the A/B's best. The dispatch order, every dispatch's arguments and
        // every kernel's arithmetic are unchanged; only the command buffer boundary moves, and command buffers on
        // one queue run in commit order. Instruments that own the command buffer (the layer/stage split, the
        // attention profiler, the GPU recorder, a diagnostic observer) keep the shipped single-buffer form.
        const pipe_layers: u32 = if (rows <= 8 and m.profile == null and m.observer == null and
            std.c.getenv("TF_DSV41_LAYER_CB") == null and std.c.getenv("TF_DSV41_STAGE_CB") == null and
            std.c.getenv("TF_DSV41_ATTN_PROF") == null)
            (if (std.c.getenv("TF_DSV41_PIPE_LAYERS")) |v|
                (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2)
            else
                2)
        else
            0;
        var pipe_prev: mtl.CommandBuffer = undefined;
        var pipe_prev_set = false;
        var pipe_gpu_extra: f64 = 0;
        // The command buffers committed before the second Engram layer's staged read.
        var engram_prev: [4]mtl.CommandBuffer = undefined;
        var engram_prev_n: usize = 0;
        var e = m.queue.commandBufferUnretained();
        var enc = e.compute(if (concurrentOn()) .concurrent else .serial);
        for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
        var encoding = true;
        var record_started = false;
        errdefer {
            if (m.profile) |recorder| {
                if (record_started) {
                    if (encoding) recorder.cancelEncoding(&enc) catch {} else recorder.discardAfterCompletion() catch {};
                } else if (encoding) enc.end();
            } else if (encoding) enc.end();
        }
        if (m.profile) |recorder| {
            try recorder.beginCommand(e);
            record_started = true;
        }
        try m.profileSpan(&enc, .other);
        const p = &m.pipelines;
        const ws = &m.workspace;
        if (m.stream == null) {
            for (engram.hash.layers, 0..) |meta, index| {
                try engram.encodeHash(enc, &p.engram, meta, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), ws.view(if (index == 0) .engram_ids_0 else .engram_ids_1), rows);
            }
            try engram.encodeCommit(enc, &p.engram, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), rows);
        }
        ops.encodeEmbedding(enc, &p.ops, ws.view(.ids), tensor(m.weights.embed), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .table_stride = 5120, .y_stride = 5120, .copies = 1 });
        enc.barrier();
        mhc.encodeExpand(enc, &p.mhc, ws.view(.collapsed), ws.view(.stream), ws.view(.previous_pre), .{ .rows = rows, .width = 5120, .x_stride = 5120, .stream_stride = 20480 });
        enc.barrier();
        hid.record(enc, &p.ops, m.streamView(), rows, .embed, 0);
        var asegs: [8]attention.Seg = undefined;
        {
            var at: u32 = 0;
            for (segs, 0..) |sg, si| {
                asegs[si] = .{ .state = &m.caches[sg.slot], .p0 = sg.p0, .r0 = at, .n = @intCast(sg.ids.len) };
                at += @intCast(sg.ids.len);
            }
        }
        for (&m.weights.layers, 0..) |*layer, index| {
            if (pipe_layers != 0 and index == pipe_layers) {
                enc.end();
                e.commit();
                pipe_prev = e;
                pipe_prev_set = true;
                e = m.queue.commandBufferUnretained();
                enc = e.compute(if (concurrentOn()) .concurrent else .serial);
                for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
            }
            // THE ENGRAM GATE READS THE EXPANDED STREAM. The fused boundary defers the previous stage's branch
            // expand to this layer's boundary (below), but the Engram gate runs first and is a consumer of the
            // EXPANDED stream: the reference's order is expand -> gate -> collapse. Flush the pending expand here,
            // before the gate, so the gate sees the post-branch hidden and the expand's residual is the pre-Engram
            // stream; the boundary below then runs expand=false and computes the same mixes over the post-Engram
            // stream. encodePost is the same arithmetic as expand_mix's expand (post[k]*branch + sum_j
            // comb[j*4+k]*old[j]).
            if (layer.engram != null and m.pending_expand and
                std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null and
                std.c.getenv("TF_DSV41_NO_ENGRAM_FLUSH") == null)
            {
                const cur = m.streamView();
                const nxt = if (m.stream_alt) ws.view(.stream) else ws.view(.stream_next);
                mhc.encodePost(enc, &p.mhc, ws.view(.branch), cur, ws.view(.post), ws.view(.comb), nxt, .{ .rows = rows, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
                m.stream_alt = !m.stream_alt;
                m.pending_expand = false;
                enc.barrier();
            }
            hid.record(enc, &p.ops, m.streamView(), rows, .stream, @intCast(index));
            // TF_DSV41_SKIP_ENGRAM: a pricing arm for the Engram term (its gate kernels and the host hash pass);
            // the reply degrades, the rate prices the stage.
            if (layer.engram != null and std.c.getenv("TF_DSV41_SKIP_ENGRAM") == null) {
                const g = &layer.engram.?;
                if (m.engram_pending and @as(u32, @intCast(index)) == engram.hash.layers[1].layer_id and engramPipeRows(m, rows)) {
                    // Commit the layers encoded so far, so the GPU is running them while the second layer's rows
                    // are read. The queue is in order and a command-buffer boundary is a full barrier, so the next
                    // buffer sees everything this one wrote.
                    enc.end();
                    e.commit();
                    if (engram_prev_n < engram_prev.len) {
                        engram_prev[engram_prev_n] = e;
                        engram_prev_n += 1;
                    }
                    e = m.queue.commandBufferUnretained();
                    enc = e.compute(if (concurrentOn()) .concurrent else .serial);
                    for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
                    m.engram_pending = false;
                    const stage_t0: u64 = if (m.host_prof) nowNs() else 0;
                    try m.stageLayer(rows, 1);
                    if (m.host_prof) m.host_stage_ns += nowNs() - stage_t0;
                    m.stageHash(rows, "l1");
                }
                try m.profileSpan(&enc, .engram);
                const engram_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_ENGRAM") != null) 2 else 1;
                for (0..engram_passes) |_| if (m.stream_stage) |staged| {
                    const at = @as(usize, if (index == 1) 0 else 1) * @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
                    try m.project(enc, g.wkv, View{ .buffer = staged, .offset = at }, ws.view(.engram_kv), rows, engram.gather_dim, 25600, .bf16);
                } else {
                    try engram.encodeGather(enc, &p.engram, .{ .weight = tensor(g.table.?), .scale = tensor(g.scales.?), .rows = @intCast(g.table.?.shape[0]) }, ws.view(if (index == 1) .engram_ids_0 else .engram_ids_1), ws.view(.engram_gather), rows, 6144);
                    try m.project(enc, g.wkv, ws.view(.engram_gather), ws.view(.engram_kv), rows, 6144, 25600, .bf16);
                };
                try engram.encodeGate(enc, &p.engram, .{ .q_weight = tensor(g.q_weight), .k_weight = tensor(g.k_weight), .dim = 5120, .copies = 4, .weight_dtype = switch (g.q_weight.dtype) {
                    .bf16 => .bf16,
                    else => .f32,
                }, .hidden_stride = 20480, .kv_stride = 25600, .output_stride = 20480, .eps = m.cfg.eps }, m.streamView(), ws.view(.engram_kv), null, m.streamView(), rows);
            }
            // The drafter reads the mean of this layer's input streams at its three target layers (round 127).
            // The taps buffer holds sixteen rows a layer: only a decode step or a draft round fits, and a prompt's
            // rows would run past it (the original tap guard).
            if (segs.len == 1 and tapsOn() and index >= 37 and index <= 39) {
                // The last rows of this pass, a ring of 128: the drafter's conditioning is the window's rows, not
                // one summary (TF_DSV41_COND_ROWS).
                const n: u32 = @min(rows, 128);
                const skip = @as(usize, rows - n) * 20480 * 2;
                // THE CAPTURED ROWS' OWN FIRST POSITION. Every shipped chunk skips a multiple of 128 rows, so the
                // forward's start position was the same slot; a merged tail chunk's length is not a multiple of
                // 128, and its live capture must still land on the rows it holds.
                const base = (state.position + (rows - n)) % 128;
                // A MERGED TAIL CHUNK ALSO STASHES THE PREFIX -- the taps ring as it stands at the mark.
                if (m.publish_limit != 0) {
                    const q = @min(m.publish_limit, rows);
                    const sn: u32 = @min(q, 128);
                    ops.encodeStreamMean(enc, &p.ops, m.streamView().shifted(@as(usize, q - sn) * 20480 * 2), .{ .buffer = m.taps_stash }, .{ .rows = sn, .width = 5120, .stream_stride = 20480, .out_stride = 5120, .slot = @intCast(index - 37), .row_base = (state.position + q - sn) % 128 });
                }
                // THE THREE TARGET LAYERS SHARE ONE row_base. The taps buffer is interleaved by position (row p =
                // the three layers' values), so the ring advances ONCE a forward. The port advanced it once a
                // TARGET LAYER: layer 37 wrote rows [B, B+n), layer 38 rows [B+n, B+2n) and layer 39 rows [B+2n,
                // B+3n), so the conditioning's row p mixed three positions (and, at the first capture, two stale
                // slices).
                const shared = tapsSharedBaseOn();
                if (shared and index == 37) {
                    // THE TAPS RING'S SLOT IS THE TOKEN POSITION. The cursor advanced by the CAPTURED rows (the
                    // verify's window, four a round) while the position advanced by the KEPT rows (about 1.3 a
                    // round), so the ring's rows drifted out of alignment with the positions -- 2.7 rows a round.
                    // The stream-mean kernel writes modulo the ring, so the aligned base is the forward's start
                    // position.
                    m.taps_base = if (condAlignOn()) state.position % 128 else m.taps_row;
                    // A fresh capture's rows are all valid; the port's rollback sets the prefix.
                    m.taps_rows_valid = 0;
                    m.taps_row = (m.taps_row + n) % 128;
                    m.taps_n = @min(m.taps_n + n, 128);
                }
                ops.encodeStreamMean(enc, &p.ops, m.streamView().shifted(skip), .{ .buffer = m.taps }, .{ .rows = n, .width = 5120, .stream_stride = 20480, .out_stride = 5120, .slot = @intCast(index - 37), .row_base = if (shared) base else m.taps_row });
                // The capture's row count, so the conditioning knows how many tail rows of the taps ring are
                // consecutive positions (the last capture is the verify's or the re-feed's, and its rows are the
                // positions the window is missing).
                m.taps_rows_last = n;
                if (!shared) {
                    // A fresh capture's rows are all valid; the port's rollback sets the prefix.
                    m.taps_rows_valid = 0;
                    m.taps_row = (m.taps_row + n) % 128;
                    m.taps_n = @min(m.taps_n + n, 128);
                }
                enc.barrier();
            }
            try splitStage(m, &enc, &e, "engram", index);
            try m.profileSpan(&enc, .mhc);
            if (p.hc_boundary) |*bp| {
                m.boundary(enc, bp, layer.hc_attn, ws.view(.previous_pre), ws.view(.attention_pre), layer.attn_norm, rows, true);
                enc.barrier();
            } else {
                m.mixes(enc, layer.hc_attn, ws.view(.attention_pre), rows);
                mhc.encodeCollapse(enc, &p.mhc, m.streamView(), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
                enc.barrier();
                m.normalize(enc, ws.view(.collapsed), layer.attn_norm, ws.view(.normalized), rows);
            }
            hid.record(enc, &p.ops, ws.view(.normalized), rows, .attn_in, @intCast(index));
            try splitStage(m, &enc, &e, "mhc_attn", index);
            try m.profileSpan(&enc, .attention);
            if (std.c.getenv("TF_DSV41_SKIP_ATTN") == null) {
                if (attention.profLayerOn(@intCast(index))) {
                    var aprof = attention.Prof{ .queue = m.queue, .cb = e, .enc = enc, .on = true };
                    try attention.encodeProfiled(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg, &aprof);
                    enc = aprof.enc;
                    e = aprof.cb;
                } else if (m.attention_observer) |attention_observer|
                    try attention.encodeDiagnostic(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg, true, attention_observer, null, &.{})
                else if (segs.len > 1)
                    try attention.encodeDiagnostic(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg, false, null, null, asegs[0..segs.len])
                else
                    try attention.encode(&p.attention, enc, &layer.attn, state, &m.attention_workspace, ws.view(.normalized), ws.view(.branch), &p.projection, ws.projection(), @intCast(index), rows, m.cfg);
            }
            enc.barrier();
            hid.record(enc, &p.ops, ws.view(.branch), rows, .attn, @intCast(index));
            if (hid.full_layer >= 0 and index == @as(usize, @intCast(hid.full_layer))) {
                // The attention block's own regions, in execution order, so the first one that moves names the
                // kernel. The row widths are attention.Workspace's fixed per-row region sizes.
                const aw = &m.attention_workspace;
                const li: u32 = @intCast(index);
                hid.recordBytes(enc, &p.ops, aw.view(.qr), rows, 1280 * 2, .a_qr, li);
                hid.recordBytes(enc, &p.ops, aw.view(.query), rows, 32768 * 2, .a_q, li);
                hid.recordBytes(enc, &p.ops, aw.view(.raw_kv), rows, 512 * 2, .a_kv, li);
                hid.recordBytes(enc, &p.ops, aw.view(.values), rows, 512 * 4, .a_vals, li);
                hid.recordBytes(enc, &p.ops, aw.view(.gates), rows, 512 * 4, .a_gates, li);
                // A compressed region is written for this chunk's `complete` new rows only; the tail of the
                // scratch is the previous request's bytes and is not a reading of this forward.
                const aratio = m.cfg.compressRatio(@intCast(index));
                const complete: u32 = if (aratio == 0) 0 else @intCast((state.position + rows) / aratio - state.position / aratio);
                if (complete != 0) {
                    hid.recordBytes(enc, &p.ops, aw.view(.latent), complete, 512 * 2, .a_latent, li);
                    hid.recordBytes(enc, &p.ops, aw.view(.index_k), complete, 128 * 2, .a_idxk, li);
                }
                hid.recordBytes(enc, &p.ops, aw.view(.index_q), rows, 4096 * 2, .a_idxq, li);
                hid.recordBytes(enc, &p.ops, aw.view(.index_weights), rows, 32 * 2, .a_idxw, li);
                // The picks in 64-row bands, each its own line, so the first band that moves names the row range.
                var band: u32 = 0;
                while (@as(usize, band) * 64 < rows) : (band += 1) {
                    const r0: u32 = band * 64;
                    const nb: u32 = @min(64, rows - r0);
                    hid.recordBytes(enc, &p.ops, aw.selection.positions().shifted(@as(usize, r0) * 512 * 4), nb, 512 * 4, .a_pos, 200000 + band);
                }
                hid.recordBytes(enc, &p.ops, aw.selection.positions().shifted(@as(usize, rows - 1) * 512 * 4), 1, 512 * 4, .a_pos1, li);
                hid.recordBytes(enc, &p.ops, aw.selection.windowChunk(), rows, 528, .a_chunk, li);
                if (aratio != 0) {
                    const srcv = try state.source(@intCast(index), aratio);
                    const compv: u32 = @intCast((state.position + rows) / aratio);
                    if (compv != 0) hid.recordBytes(enc, &p.ops, srcv.main, compv, 288, .a_main, li);
                }
                if (aratio != 0) {
                    const keys_all: u32 = @intCast((state.position + rows) / aratio);
                    if (keys_all != 0 and keys_all <= 4096)
                        hid.recordStrided(enc, &p.ops, aw.selection.scores(), rows, 4096 * 4, keys_all * 4, .a_sco, 999999);
                }
                if (aratio != 0 and rows > 1) {
                    var k: u32 = 0;
                    while (k < 16 and k < rows) : (k += 1) {
                        const row: u32 = (rows - 1) - @as(u32, @intCast((@as(usize, rows - 1) * k) / 15));
                        const vis: u32 = @intCast(@min((state.position + row + 1) / aratio, (state.position + rows) / aratio));
                        if (vis == 0) continue;
                        hid.recordBytes(enc, &p.ops, aw.selection.scores().shifted(@as(usize, row) * 4096 * 4), 1, vis * 4, .a_sco, 100000 + row);
                    }
                }
                hid.recordBytes(enc, &p.ops, aw.view(.grouped), rows, 8192 * 2, .a_grp, li);
            }
            try splitStage(m, &enc, &e, "attention", index);
            try m.profileSpan(&enc, .mhc);
            if (p.hc_boundary != null) m.pending_expand = true else m.post(enc, rows);
            if (m.observer) |observer| observer.encode(observer.ctx, enc, @intCast(index), .attention, m);
            if (p.hc_boundary) |*bp| {
                m.boundary(enc, bp, layer.hc_ffn, ws.view(.attention_pre), ws.view(.previous_pre), layer.ffn_norm, rows, true);
                enc.barrier();
            } else {
                m.mixes(enc, layer.hc_ffn, ws.view(.previous_pre), rows);
                mhc.encodeCollapse(enc, &p.mhc, m.streamView(), ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
                enc.barrier();
                m.normalize(enc, ws.view(.collapsed), layer.ffn_norm, ws.view(.normalized), rows);
            }
            hid.record(enc, &p.ops, ws.view(.normalized), rows, .ffn_in, @intCast(index));
            try splitStage(m, &enc, &e, "mhc_ffn", index);
            try m.profileSpan(&enc, .moe);
            if (m.moe_x_debug) |b| {
                if (index == 0 and rows >= 64 and rows <= 2048) {
                    ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 0 }, rows * 5120, true);
                    m.moe_x_rows = rows;
                }
            }
            if (std.c.getenv("TF_DSV41_SKIP_MOE") == null) try m.moe(&enc, &e, &layer.moe, rows, @intCast(index), m.cfg.topk, null);
            hid.record(enc, &p.ops, ws.view(.branch), rows, .moe, @intCast(index));
            try m.profileSpan(&enc, .mhc);
            if (p.hc_boundary != null) m.pending_expand = true else m.post(enc, rows);
            if (m.observer) |observer| observer.encode(observer.ctx, enc, @intCast(index), .moe, m);
            try splitStage(m, &enc, &e, "moe", index);
            if (layer_cb) {
                enc.end();
                e.commit();
                e.wait();
                try completed(e);
                const gpu: f64 = e.gpuSeconds() * 1e3;
                const host: f64 = @as(f64, @floatFromInt(nowNs() - host_enc_start)) / 1e6;
                std.log.info("layer {d:>2}: command gpu {d:7.2} ms, wall so far {d:8.2} ms, ratio {d}, engram {}", .{ index, gpu, host, m.cfg.compressRatio(@intCast(index)), layer.engram != null });
                e = m.queue.commandBufferUnretained();
                enc = e.compute(.serial);
                for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
            }
        }
        if (logits != .none) {
            try m.profileSpan(&enc, .head);
            const output_rows: u32 = if (logits == .all) rows else 1;
            const first: usize = if (logits == .all) 0 else rows - 1;
            if (p.hc_boundary) |*bp| {
                if (m.pending_expand) {
                    const last = &m.weights.layers[m.weights.layers.len - 1].hc_ffn;
                    const wsx = &m.workspace;
                    const cur = if (m.stream_alt) wsx.view(.stream_next) else wsx.view(.stream);
                    const next = if (m.stream_alt) wsx.view(.stream) else wsx.view(.stream_next);
                    enc.setPipeline(bp.expand_mix);
                    inline for (.{ cur, wsx.view(.branch), wsx.view(.post), wsx.view(.comb), tensor(last.projection), wsx.view(.hc_part), next }, 0..) |r, i| enc.setBuffer(r.buffer, r.offset, i);
                    enc.dispatchGroups(.{ .width = 20, .height = if (logits == .all) rows else 1 }, .{ .width = 256 });
                    enc.barrier();
                    m.stream_alt = !m.stream_alt;
                    m.pending_expand = false;
                }
            }
            mhc.encodeCollapse(enc, &p.mhc, m.streamView().shifted(first * 20480 * 2), ws.view(.previous_pre).shifted(first * 4 * 4), ws.view(.collapsed), .{ .rows = output_rows, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            m.normalize(enc, ws.view(.collapsed), m.weights.norm, ws.view(.normalized), output_rows);
            // The trunk's own head input (the row 0 of the window), so the drafter's head input can be compared
            // with it (the cosine): the MTP module's hidden must live in the trunk's space.
            if (std.c.getenv("TF_DSV41_MTP_DEBUG_TOP") != null) {
                if (m.mtp_trunk_hidden == null) {
                    m.mtp_trunk_hidden = try m.device.buffer(5120 * 4, mtl.ResourceOptions.shared);
                    m.residency.add(m.mtp_trunk_hidden.?);
                }
                ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = m.mtp_trunk_hidden.?, .offset = 0 }, 5120, false);
            }
            const head_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_HEAD") != null) 2 else 1;
            for (0..head_passes) |_| try m.project(enc, m.weights.head, ws.view(.normalized), ws.view(.logits), output_rows, 5120, 129280, .f32);
            hid.recordLogits(enc, &p.ops, ws.view(.logits).shifted(@as(usize, output_rows - 1) * 129280 * 4));
        }
        // TF_DSV41_ROUTE_DEBUG5: the head's own logits (first row), the number that decides the token.
        if (std.c.getenv("TF_DSV41_ROUTE_DEBUG5") != null and !gate_dump_done3 and logits != .none) {
            gate_dump_done3 = true;
            const n: u32 = 129280;
            const stg = try m.device.buffer(@as(usize, n) * 4, mtl.ResourceOptions.shared);
            ops.encodeDump(enc, &p.ops, ws.view(.logits), .{ .buffer = stg, .offset = 0 }, n, false);
            try splitStage(m, &enc, &e, "head", 0);
            const v: [*]const f32 = @ptrCast(@alignCast(stg.contents()));
            var best: f32 = -1e30;
            var best_id: u32 = 0;
            for (0..n) |i| if (v[i] > best) {
                best = v[i];
                best_id = @intCast(i);
            };
            std.log.info("head logits {d:.4} {d:.4} {d:.4} | argmax {d} @ {d:.4}", .{ v[0], v[1], v[2], best_id, best });
        }
        if (m.profile) |recorder| {
            if (recorder.activeSpan()) try recorder.endSpan(&enc);
            try recorder.finishEncoding(&enc);
        } else enc.end();
        encoding = false;
        m.valid = false;
        const host_commit: u64 = nowNs();
        e.commit();
        e.wait();
        const host_done: u64 = nowNs();
        try completed(e);
        hid.report();
        scan_probe.report();
        if (pipe_prev_set) {
            // The earlier chunks completed in queue order by the last wait; check them and count their GPU time.
            try completed(pipe_prev);
            pipe_gpu_extra = pipe_prev.gpuSeconds() * 1e3;
        }
        for (engram_prev[0..engram_prev_n]) |cb| {
            try completed(cb);
            pipe_gpu_extra += cb.gpuSeconds() * 1e3;
        }
        // The same prefill's layer-0 MoE input to /tmp/moe_x.f32.
        if (m.moe_x_debug) |b| {
            if (m.moe_x_rows > 0) {
                const n = @as(usize, m.moe_x_rows) * 5120 * 4;
                const out = try m.gpa.alloc(u8, 16 + n);
                defer m.gpa.free(out);
                std.mem.writeInt(u64, out[0..8], @intCast(m.moe_x_rows), .little);
                std.mem.writeInt(u64, out[8..16], 5120, .little);
                const src: [*]const u8 = @ptrCast(b.contents());
                @memcpy(out[16..], src[0..n]);
                std.Io.Dir.cwd().writeFile(m.io, .{ .sub_path = "/tmp/moe_x.f32", .data = out }) catch {};
                std.log.info("moe_x dump: {d} rows x 5120 f32 -> /tmp/moe_x.f32", .{m.moe_x_rows});
                m.moe_x_rows = 0;
            }
        }
        // One 2,048-row prefill's wq_b input to /tmp/wqb_x.f32 (a 16-byte K,R header, then f32).
        if (m.wqb_x_debug) |b| {
            if (attention.wqb_x_rows > 0) {
                const n = @as(usize, attention.wqb_x_rows) * @as(usize, m.cfg.q_lora) * 4;
                const out = try m.gpa.alloc(u8, 16 + n);
                defer m.gpa.free(out);
                std.mem.writeInt(u64, out[0..8], @intCast(attention.wqb_x_rows), .little);
                std.mem.writeInt(u64, out[8..16], @intCast(m.cfg.q_lora), .little);
                const src: [*]const u8 = @ptrCast(b.contents());
                @memcpy(out[16..], src[0..n]);
                std.Io.Dir.cwd().writeFile(m.io, .{ .sub_path = "/tmp/wqb_x.f32", .data = out }) catch {};
                std.log.info("wqb_x dump: {d} rows x {d} f32 -> /tmp/wqb_x.f32", .{ attention.wqb_x_rows, m.cfg.q_lora });
                attention.wqb_x_rows = 0;
            }
        }
        // The proposer runs from the backend's draft(), not here: its own forward writes the logits buffer from
        // row 1 on, which is exactly the rows a verify (logits .all) returns to the engine.
        if (false and std.c.getenv("TF_DSV41_MTP_DRAFTS") != null and logits != .none and m.weights.dspark != null) {
            const logit_view = m.workspace.view(.logits);
            const logit_bytes = logit_view.buffer.contents();
            var best: u32 = 0;
            var best_value: f32 = -std.math.inf(f32);
            for (0..129280) |t| {
                const v = @as(f32, @bitCast(std.mem.readInt(u32, logit_bytes[logit_view.offset + t * 4 ..][0..4], .little)));
                if (v > best_value) {
                    best_value = v;
                    best = @intCast(t);
                }
            }
            if (m.proposeTokens(0, best, 3)) |drafts| {
                m.mtp_drafts = drafts;
                m.mtp_count = 3;
                std.log.info("mtp drafts: pending={d} next={d} {d} {d}", .{ best, drafts[0], drafts[1], drafts[2] });
            } else |err| std.log.warn("mtp: drafts unavailable ({s})", .{@errorName(err)});
        }

        // TF_DSV41_MTP_DRAFTS: the drafter's proposal after this forward, from the token the head just chose.
        if (m.profile) |recorder| {
            _ = try recorder.readAfterCompletion();
            const agg = recorder.aggregates();
            var total: f64 = 0;
            for (agg) |a| total += a.gpu_seconds;
            for (agg, gpu_counters.stage_names) |a, name| {
                if (a.calls != 0) std.log.info("stage {s}: {d} calls, {d:.1} ms, {d:.1}% of recorded", .{ name, a.calls, a.gpu_seconds * 1e3, if (total > 0) a.gpu_seconds / total * 100 else 0 });
            }
            std.log.info("stage total: {d:.1} ms", .{total * 1e3});
            if (m.profile_open_stage) |prev_stage| {
                m.profile_stage_dispatch[@backingInt(prev_stage)] += mtl.command.dispatch_count.load(.monotonic) - m.profile_last_count;
                m.profile_open_stage = null;
            }
            for (gpu_counters.stage_names, m.profile_stage_dispatch) |name, d| {
                if (d != 0) std.log.info("stage dispatches {s}: {d}", .{ name, d });
            }
            var hist: [20]u64 = undefined;
            for (&hist, 0..) |*h, i| h.* = mtl.command.dispatch_hist[i].load(.monotonic);
            std.log.info("dispatch histogram (log2 threadgroups): {any}, threadgroups {d}, indirect {d}", .{ hist, mtl.command.dispatch_threadgroups.load(.monotonic), mtl.command.dispatch_indirect.load(.monotonic) });
        }
        // TF_DSV41_HOST_PROF without the counter recorder (which is broken on this build): the host's own clocks
        // and the command buffer's GPU seconds, so "host or GPU" stops being an inference.
        if (m.host_prof) {
            std.log.info("host: rows {d}, engram hash {d:.2} ms (of which stage {d:.2} ms, misses {d}, faults {d}, pf {d}, thr {d})", .{
                rows,
                @as(f64, @floatFromInt(m.host_hash_ns)) / 1e6,
                @as(f64, @floatFromInt(m.host_stage_ns)) / 1e6,
                m.host_stage_misses,
                m.host_stage_faults,
                m.host_stage_pf,
                m.host_stage_thr,
            });
            std.log.info("forward: rows {d}, encode {d:.2} ms, commit+wait {d:.2} ms, command gpu {d:.2} ms, wall {d:.2} ms, dispatches {d}, pipe {d}", .{
                rows,
                @as(f64, @floatFromInt(host_commit - host_enc_start)) / 1e6,
                @as(f64, @floatFromInt(host_done - host_commit)) / 1e6,
                e.gpuSeconds() * 1e3 + pipe_gpu_extra,
                @as(f64, @floatFromInt(host_done - host_enc_start)) / 1e6,
                mtl.command.dispatch_count.load(.monotonic) - dispatches_before,
                pipe_layers,
            });
        }
        for (segs) |sg| try m.caches[sg.slot].append(@intCast(sg.ids.len));
        m.last_gpu_seconds = e.gpuSeconds() + pipe_gpu_extra / 1e3;
        m.valid = true;
        m.logits_rows = if (logits == .all) rows else if (logits == .last) 1 else 0;
        return ws.logits(m.logits_rows);
    }

    /// The rows the host may hash. The decode/verify window (<= 8 rows) pays the GPU's commit+wait before its ids
    /// are readable on the host; its hash is 48 integer reductions a row, so the host computes it where the staging
    /// reads it and the round trip disappears. The prompt's thousands-row chunks are cheaper on the GPU.
    /// `TF_DSV41_CPU_HASH=0` is the shipped GPU pass.
    fn cpuHashRows() u32 {
        const S = struct {
            var cached: i64 = -1;
        };
        if (S.cached < 0) S.cached = if (std.c.getenv("TF_DSV41_CPU_HASH")) |e|
            (std.fmt.parseInt(i64, std.mem.span(e), 10) catch 8)
        else
            8;
        return @intCast(@max(0, S.cached));
    }

    /// The streamed path's own hash pass: the ids must be readable on the host, so the shipped form is a
    /// second command buffer; a decode-sized window takes the host's own copy of the same integer hash.
    fn hashPass(m: *Model, slot: u32, rows: u32) !void {
        const p = &m.pipelines;
        const ws = &m.workspace;
        if (rows <= cpuHashRows()) {
            try engram.encodeHashHost(try ws.historyView(slot), ws.view(.ids), ws.view(.token_map), ws.view(.engram_ids_0), ws.view(.engram_ids_1), rows);
            try m.stageRows(rows);
            return;
        }
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const e = m.queue.commandBufferUnretained();
        var enc = e.compute(.serial);
        for (engram.hash.layers, 0..) |meta, index| {
            try engram.encodeHash(enc, &p.engram, meta, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), ws.view(if (index == 0) .engram_ids_0 else .engram_ids_1), rows);
        }
        try engram.encodeCommit(enc, &p.engram, ws.view(.ids), ws.view(.token_map), null, try ws.historyView(slot), rows);
        enc.end();
        e.commit();
        e.wait();
        try completed(e);
        try m.stageRows(rows);
    }

    /// The batch's hash pass -- each segment's ids at its own rows, against its own slot's Engram history, then the
    /// one staging read over all rows.
    fn hashPassSegs(m: *Model, segs: []const Seg, rows: u32) !void {
        const ws = &m.workspace;
        var r0: usize = 0;
        for (segs) |sg| {
            const cols: usize = engram.hash.columns;
            try engram.encodeHashHost(try ws.historyView(sg.slot), ws.view(.ids).shifted(r0 * 4), ws.view(.token_map), ws.view(.engram_ids_0).shifted(r0 * cols * 4), ws.view(.engram_ids_1).shifted(r0 * cols * 4), @intCast(sg.ids.len));
            r0 += sg.ids.len;
        }
        try m.stageRows(rows);
    }

    /// Both Engram layers' rows for this window, read from the release's shards into the staging buffer.
    /// Threads the Engram gather splits its ids across (TF_DSV41_ENGRAM_THREADS, 1 keeps one thread).
    /// The default follows the read path: the mapped gather reads by address, so its workers scale; the
    /// positional reads share one I/O context and serialise on it, and one thread is their best case.
    fn engramThreads(m: *Model) usize {
        // The mapped default is sixteen for the large gather too.
        var v: usize = if (m.stream) |s| (if (s.mapped()) 16 else 1) else 1;
        if (std.c.getenv("TF_DSV41_ENGRAM_THREADS")) |e| {
            v = std.fmt.parseInt(usize, std.mem.span(e), 10) catch v;
            if (v == 0 or v > 64) v = 1;
        }
        return v;
    }

    /// The latch's coldness clock. Default: the whole stage (prefetch + gather). `=gather` restores The
    /// post-prefetch signal (the same-binary control).
    fn latchStage() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_LATCH")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "gather")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// The whole-window page-in (TF_DSV41_ENGRAM_PF). Default on; =0 is the same-binary control.
    fn engramPrefetch() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_PF")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// The decode-sized list's workers. Sixteen is the gather's own thread table's height.
    /// TF_DSV41_ENGRAM_SMALL_THREADS is the arm (1..16).
    fn engramSmallThreads() usize {
        const S = struct {
            var cached: usize = 0;
        };
        if (S.cached != 0) return S.cached;
        var v: usize = 16;
        if (std.c.getenv("TF_DSV41_ENGRAM_SMALL_THREADS")) |e| {
            v = std.fmt.parseInt(usize, std.mem.span(e), 10) catch v;
            if (v == 0 or v > 16) v = 16;
        }
        S.cached = v;
        return v;
    }

    fn engramAdapt() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_ADAPT")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// Warm the pending row's Engram pages while the drafter's own forward runs. The pending token is known before
    /// `proposeTokens` and is the verify window's FIRST row, so its ids can be hashed from the standing record (a
    /// peek that does not advance it) and its pages handed to madvise now; the next verify's `stageRows` recomputes
    /// the same ids and finds the pages resident. Only the page-in moves: no id, byte, reduction or reply changes.
    /// `TF_DSV41_ENGRAM_PEND=0` is the same-binary control.
    fn engramPendingOn() bool {
        const S = struct {
            var cached: i8 = 0;
        };
        if (S.cached == 0) S.cached = if (std.c.getenv("TF_DSV41_ENGRAM_PEND")) |e|
            (if (std.mem.eql(u8, std.mem.span(e), "0")) -1 else 1)
        else
            1;
        return S.cached > 0;
    }

    /// Stage the second Engram layer's rows at its own layer instead of before the forward. The ids are already
    /// known (the hash pass ran); only the read's clock moves. `=0` is the control.
    fn engramPipeOn() bool {
        // Read per forward: the same process can run the control and the arm.
        if (std.c.getenv("TF_DSV41_ENGRAM_PIPE")) |e| return !std.mem.eql(u8, std.mem.span(e), "0");
        return true;
    }

    fn engramPipeRows(m: *Model, rows: u32) bool {
        if (rows <= 8 or !engramPipeOn()) return false;
        if (m.stream == null or m.stream_stage == null) return false;
        if (engram.hash.layers.len != 2) return false;
        if (engram.hash.layers[1].layer_id <= engram.hash.layers[0].layer_id) return false;
        if (m.profile != null or m.observer != null) return false;
        return std.c.getenv("TF_DSV41_LAYER_CB") == null and
            std.c.getenv("TF_DSV41_STAGE_CB") == null and
            std.c.getenv("TF_DSV41_ATTN_PROF") == null;
    }

    fn prefetchPending(m: *Model, slot: u32, pending: u32) void {
        const src = m.stream orelse return;
        if (!src.mapped()) return;
        const ws = &m.workspace;
        const history = ws.historyView(slot) catch return;
        var ids: [2][engram.hash.columns]u32 = undefined;
        engram.peekHashHost(history, ws.view(.token_map), pending, &ids) catch return;
        for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |_, li| src.prefetch(li, &ids[li]);
    }

    fn stageFnv(bytes: []const u8) u64 {
        var h: u64 = 0xcbf29ce484222325;
        for (bytes) |b| {
            h ^= b;
            h *%= 0x100000001b3;
        }
        return h;
    }

    /// `TF_DSV41_ENGRAM_STAGE_HASH=1`; nothing in the default path reads this.
    fn stageHash(m: *Model, rows: u32, tag: []const u8) void {
        if (std.c.getenv("TF_DSV41_ENGRAM_STAGE_HASH") == null) return;
        const buf = m.stream_stage orelse return;
        const ws = &m.workspace;
        const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
        const base: [*]const u8 = buf.contents();
        const n_ids: usize = @as(usize, rows) * engram.hash.columns;
        const i0v = ws.view(.engram_ids_0);
        const i1v = ws.view(.engram_ids_1);
        const ids_a: [*]const u32 = @ptrCast(@alignCast(i0v.buffer.contents() + i0v.offset));
        const ids_b: [*]const u32 = @ptrCast(@alignCast(i1v.buffer.contents() + i1v.offset));
        std.log.info("engram stage hash {s} rows {d} ids0 {x} ids1 {x} staged0 {x} staged1 {x}", .{
            tag,                                             rows,
            stageFnv(std.mem.sliceAsBytes(ids_a[0..n_ids])), stageFnv(std.mem.sliceAsBytes(ids_b[0..n_ids])),
            stageFnv(base[0..layer_bytes]),                  stageFnv(base[layer_bytes..][0..layer_bytes]),
        });
    }

    fn stageLayer(m: *Model, rows: u32, li: usize) !void {
        const src = m.stream orelse return;
        const buf = m.stream_stage orelse return error.EngramNotOpen;
        const ws = &m.workspace;
        const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
        const base: [*]u8 = buf.contents();
        const n_ids: usize = @as(usize, rows) * engram.hash.columns;
        const v = ws.view(if (li == 0) work.Field.engram_ids_0 else work.Field.engram_ids_1);
        const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
        const out = std.mem.bytesAsSlice(u16, base[li * layer_bytes ..][0..layer_bytes]);
        try src.rowsU32Parallel(li, at[0..n_ids], engram.head_dim, out, engramThreads(m));
    }

    fn stageRows(m: *Model, rows: u32) !void {
        const stage_start: u64 = if (m.host_prof) nowNs() else 0;
        defer if (m.host_prof) {
            m.host_stage_ns += nowNs() - stage_start;
        };
        m.stageHash(rows, "pre");
        const src = m.stream orelse return;
        const buf = m.stream_stage orelse return error.EngramNotOpen;
        const ws = &m.workspace;
        const cols: usize = engram.hash.columns;
        const layer_bytes = @as(usize, rows) * engram.gather_dim * @sizeOf(u16);
        const base: [*]u8 = buf.contents();
        const n_ids: usize = @as(usize, rows) * cols;
        const small = n_ids < 512;
        // The latch's clock is taken HERE, before the page-in. This clock sees prefetch + gather.
        const w0: u64 = if (small and latchStage()) nowNs() else 0;
        const adapt = small and engramAdapt();
        var threads: usize = engramThreads(m);
        var probed = false;
        if (small) {
            if (!adapt) {
                threads = 1;
            } else if (m.engram_par and m.engram_probe_in > 0) {
                m.engram_probe_in -= 1;
                threads = engramSmallThreads();
            } else {
                threads = 1;
                probed = true;
            }
        }
        // The window's page-ins, only where they pay. The prompt's rows are contiguous and the kernel already
        // readahead-s them, so the large path is left alone; a single-threaded probe must not be prefetched (its
        // time is the page-cache test); the warm band never reaches here.
        const pf_start: u64 = if (small and engramPrefetch()) nowNs() else 0;
        // The stage's own major faults, for the HOST_PROF line; the latch below uses the gather's clock.
        const ru0 = std.posix.getrusage(std.posix.rusage.SELF);
        const pf_on = small and m.engram_pf_on and pf_start != 0;
        if (m.host_prof) {
            m.host_stage_pf = if (pf_on) 1 else 0;
            m.host_stage_thr = @intCast(threads);
        }
        if (pf_on) {
            for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |field, li| {
                const v = ws.view(field);
                const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                src.prefetch(li, at[0..n_ids]);
            }
        }
        const t0: u64 = if (adapt) nowNs() else 0;
        const g0: u64 = if (small) nowNs() else 0;
        const misses0: u64 = if (m.host_prof) src.stats.misses else 0;
        if (!small and engramPipeRows(m, rows)) {
            // The first layer now, the second at its own layer (the GPU runs the earlier layers while that read
            // happens). Same ids, same bytes, same decode -- only the clock moves.
            try m.stageLayer(rows, 0);
            m.engram_pending = true;
        } else {
            for ([_]work.Field{ .engram_ids_0, .engram_ids_1 }, 0..) |field, li| {
                const v = ws.view(field);
                const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                const ids = at[0..n_ids];
                const out = std.mem.bytesAsSlice(u16, base[li * layer_bytes ..][0..layer_bytes]);
                try src.rowsU32Parallel(li, ids, engram.head_dim, out, threads);
            }
        }
        m.stageHash(rows, "stage");
        if (adapt) {
            const dt = nowNs() - t0;
            if (probed) {
                if (dt > 1_500_000) {
                    m.engram_slow_n += 1;
                    if (m.engram_slow_n >= 2) {
                        m.engram_par = true;
                        m.engram_probe_in = m.engram_probe_every;
                        m.engram_probe_every = if (m.engram_probe_every >= 128) 128 else m.engram_probe_every * 2;
                        m.engram_slow_n = 0;
                    }
                } else {
                    m.engram_slow_n = 0;
                    m.engram_par = false;
                    m.engram_probe_in = 0;
                    m.engram_probe_every = 16;
                }
            }
        }
        if (g0 != 0) {
            // The latch's signal is the gather's own clock, taken AFTER the prefetch: the fault counters cannot
            // serve, because a prefetched window's I/O is the madvise call and its reads are then minor faults, so
            // the page-in hides the very coldness the latch must see. The off-side is delayed three windows:
            // consecutive decode windows share most of their n-grams, so one resident window between two cold ones
            // is the norm, not the stream turning warm.
            const gather = nowNs() - (if (w0 != 0) w0 else g0);
            if (gather > 400_000) {
                m.engram_pf_on = true;
                m.engram_pf_off = 0;
            } else if (gather < 200_000) {
                if (m.engram_pf_off < 3) m.engram_pf_off += 1;
                if (m.engram_pf_off >= 3) m.engram_pf_on = false;
            } else {
                m.engram_pf_off = 0;
            }
        }
        if (m.host_prof) {
            m.host_stage_misses +%= src.stats.misses -% misses0;
            const faults1: i64 = std.posix.getrusage(std.posix.rusage.SELF).majflt;
            if (faults1 > ru0.majflt) m.host_stage_faults +%= @intCast(faults1 - ru0.majflt);
        }
        src.recordWindow(rows);
    }

    /// Copy `rows` rows of the running hidden state into tap `slot` (0..2 for the layers 37, 38 and 39 the drafter's
    /// main_proj reads concatenated). A prompt chunk has thousands of rows and drafts nothing.
    /// The drafter's token id, written where the embedding kernel reads it (the buffer is CPU-shared).
    fn drafterId(m: *Model, token: u32) !View {
        if (m.drafter_ids == null) {
            m.drafter_ids = try m.device.buffer(4, mtl.ResourceOptions.shared);
            m.markov_ids = try m.device.buffer(8 * 4, mtl.ResourceOptions.shared); // five Markov steps, a u32 each
            m.residency.add(m.drafter_ids.?);
        }
        std.mem.writeInt(u32, m.drafter_ids.?.contents()[0..4], token, .little);
        return .{ .buffer = m.drafter_ids.? };
    }

    /// The drafter's round of ids: the pending token, then the port's noise filler (128799) a row.
    fn drafterRound(m: *Model, pending: u32, rows: u32) !View {
        if (m.drafter_ids == null) {
            if (m.drafter_ids) |old| old.deinit();
            // The buffer is sized for the widest form (the five-row arm), not for the rows the first call happened
            // to use, so the arm cannot overflow it.
            m.drafter_ids = try m.device.buffer(@max((rows + 1) * 4, 6 * 4), mtl.ResourceOptions.shared);
            m.residency.add(m.drafter_ids.?);
        }
        const bytes = m.drafter_ids.?.contents();
        var r: u32 = 0;
        while (r < rows) : (r += 1)
            std.mem.writeInt(u32, bytes[@as(usize, r) * 4 ..][0..4], if (r == 0) pending else 128799, .little);
        return .{ .buffer = m.drafter_ids.? };
    }

    /// The Markov chain's partial bests: ceil(vocab/256) float2 entries, allocated once.
    fn markovPart(m: *Model) !View {
        if (m.markov_part == null) {
            const groups: usize = (129280 + 255) / 256;
            m.markov_part = try m.device.buffer(groups * 8, mtl.ResourceOptions.shared);
            m.residency.add(m.markov_part.?);
        }
        return .{ .buffer = m.markov_part.? };
    }

    /// The drafter's state: `drafter_layers` rings, allocated once and kept for the model's life.
    fn drafterState(m: *Model) !*kv.cache.State {
        if (m.drafter_state) |state| return state;
        const state = try m.gpa.create(kv.cache.State);
        errdefer m.gpa.destroy(state);
        state.* = try kv.cache.State.init(m.device, .{ .capacity = m.cache.plan.capacity, .layers = kv.cache.drafter_layers });
        m.drafter_state = state;
        return state;
    }

    fn tap(m: *Model, enc: mtl.ComputeEncoder, x: View, rows: u32, slot: usize) !void {
        if (rows > 16 or !tapsOn()) return;
        ops.encodeCopy(enc, &m.pipelines.ops, x, m.tapView(slot), .{
            .rows = rows,
            .width = 5120,
            .x_stride = 5120,
            .y_stride = 5120,
        });
        enc.barrier();
    }
    pub fn tapView(m: *const Model, slot: usize) View {
        return .{ .buffer = m.taps, .offset = slot * 16 * 5120 * 2 };
    }
    fn profileSpan(m: *Model, encoder: *mtl.ComputeEncoder, stage: gpu_counters.Stage) !void {
        const recorder = m.profile orelse return;
        if (!recorder.enabled()) return;
        const counted = mtl.command.dispatch_count.load(.monotonic);
        if (m.profile_open_stage) |prev| m.profile_stage_dispatch[@backingInt(prev)] += counted - m.profile_last_count;
        if (recorder.activeSpan()) try recorder.endSpan(encoder);
        try recorder.beginSpan(encoder, stage);
        m.profile_open_stage = stage;
        m.profile_last_count = counted;
        for (m.checkpoint.shards.items) |shard| encoder.useResource(shard.buffer, mtl.ResourceUsage.read);
    }

    fn normalize(m: *Model, enc: mtl.ComputeEncoder, x: View, w: checkpoint.Tensor, y: View, rows: u32) void {
        ops.encodeRms(enc, &m.pipelines.ops, x, tensor(w), y, .{ .rows = rows, .width = 5120, .x_stride = 5120, .y_stride = 5120, .eps = m.cfg.eps });
        enc.barrier();
    }

    fn mixes(m: *Model, enc: mtl.ComputeEncoder, w: weights.Hc, pre: View, rows: u32) void {
        if (std.c.getenv("TF_DSV41_SKIP_MHC") != null) return;
        const ws = &m.workspace;
        mhc.encodeRms(enc, &m.pipelines.mhc, ws.view(.stream), ws.view(.inv), .{ .rows = rows, .width = 5120, .stream_stride = 20480, .eps = m.cfg.eps });
        mhc.encodeMixDense(enc, &m.pipelines.mhc, ws.view(.stream), tensor(w.projection), ws.view(.mixes), .{ .rows = rows, .k = 20480, .n = 24, .x_stride = 20480 });
        enc.barrier();
        mhc.encodeSplit(enc, &m.pipelines.mhc, ws.view(.mixes), ws.view(.inv), tensor(w.scale), tensor(w.base), pre, ws.view(.post), ws.view(.comb), .{ .rows = rows, .eps = m.cfg.hc_eps });
        enc.barrier();
    }

    fn post(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
        if (std.c.getenv("TF_DSV41_SKIP_MHC") != null) return;
        const ws = &m.workspace;
        mhc.encodePost(enc, &m.pipelines.mhc, ws.view(.branch), ws.view(.stream), ws.view(.post), ws.view(.comb), ws.view(.stream), .{ .rows = rows, .width = 5120, .branch_stride = 5120, .residual_stride = 20480, .y_stride = 20480 });
        enc.barrier();
    }

    fn project(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
        // The prompt-row int8 twin. Decode and every small chunk keep the trellis, so the gate is the call's own
        // row count; the sidecar is attached only for the attention's dense projections.
        if (matrix.int8) |side| {
            if (rows >= 1024 and std.c.getenv("TF_DSV41_I8_PREFILL") != null) {
                try exl3.encodePrefillInt8(&m.pipelines.projection, enc, matrix, side, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output }, matrix.int8_direct);
                enc.barrier();
                return;
            }
        }
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        enc.barrier();
    }

    /// The same projection with the CALLER's eight-row dense hint -- the tensor-unit dense tile from eight rows up
    /// instead of the shipped sixty-four.
    fn projectDense8(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output) !void {
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output, .dense8 = true });
        enc.barrier();
    }

    /// The projection's own three views -- the input it reads, the rotated input it leaves in the scratch, and the
    /// sums the GEMM wrote -- dumped where the call made them, so a row's fault is localised to the stage that
    /// produced it.
    fn projectDebug(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output, b: mtl.Buffer) !void {
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        enc.barrier();
        const sc = m.workspace.projection();
        ops.encodeDump(enc, &m.pipelines.ops, x, .{ .buffer = b, .offset = mtp_dbg_pin * 4 }, @intCast(rows * xs), true);
        const rot = View{ .buffer = sc.buffer, .offset = sc.offset };
        ops.encodeDump(enc, &m.pipelines.ops, rot, .{ .buffer = b, .offset = mtp_dbg_prot * 4 }, @intCast(rows * matrix.k), true);
        const sums_off = (@as(usize, rows) * matrix.k * 2 + 15) & ~@as(usize, 15);
        const sums = View{ .buffer = sc.buffer, .offset = sc.offset + sums_off };
        ops.encodeDump(enc, &m.pipelines.ops, sums, .{ .buffer = b, .offset = mtp_dbg_psum * 4 }, @intCast(rows * matrix.n), false);
        enc.barrier();
    }

    /// The wo_a group's own GEMM sums, before the inverse rotation, where the overflow is born.
    fn projectDebugSums(m: *Model, enc: mtl.ComputeEncoder, matrix: exl3.Matrix, x: View, y: View, rows: u32, xs: usize, ys: usize, output: exl3.Output, b: mtl.Buffer) !void {
        try exl3.encode(&m.pipelines.projection, enc, matrix, convert(x), convert(y), m.workspace.projection(), .{ .count = rows, .x_stride = xs, .y_stride = ys, .output = output });
        enc.barrier();
        const sc = m.workspace.projection();
        const sums_off = (@as(usize, rows) * matrix.k * 2 + 15) & ~@as(usize, 15);
        const sums = View{ .buffer = sc.buffer, .offset = sc.offset + sums_off };
        ops.encodeDump(enc, &m.pipelines.ops, sums, .{ .buffer = b, .offset = mtp_dbg_woasums * 4 }, @intCast(rows * matrix.n), false);
        enc.barrier();
    }

    /// Rows a routed expert's tensor-unit tile covers (TF_DSV41_MOE_TILE_ROWS, 0 = the kernel's 64). A taller tile
    /// decodes each weight once for more rows; it is the routed stage's one free shape.
    fn expertPasses() u32 {
        const v = std.c.getenv("TF_DSV41_DOUBLE_EXPERTS") orelse return 1;
        const n = std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2;
        return @max(1, @min(n, 8));
    }

    fn moeTileRows() u32 {
        // The grouped plan pads each expert's block to the tile's own width, so the plan's tile and the routed
        // GEMM's tile are one number (64 by default, 128 for the taller tile).
        return expert_groups.tileRows();
    }

    /// The drafter block's attention: the four projections with their two norms, the round's own keys packed into
    /// the chunk, the scan over the stage's ring plus that chunk with no compressor and no indexer, the eight `wo_a`
    /// slices, `wo_b`, and the publish. Its big scratch lives in `.logits` from row 1 on.
    fn draftAttention(m: *Model, enc0: mtl.ComputeEncoder, blk: *const weights.Layer, state: *kv.cache.State, slot: u32, layer: u32, rows: u32, position: u32, dbg: ?mtl.Buffer, prof: ?*CondProf) !void {
        // The local copy of the encoder -- the split replaces it, so the parts below encode into the fresh one.
        var enc = enc0;
        const p = &m.pipelines;
        const c = m.cfg;
        const ws = &m.workspace;
        const aw = &m.attention_workspace;
        const q_width = c.heads * c.head_dim;
        // The trunk's layer rope spectrum -- the tap layers (37..39) pool with ratio 2, so their keys carry the
        // COMPRESSED spectrum; the drafter's query and its own rows must match it.
        const tl0: u32 = if (37 + layer < m.weights.layers.len) 37 + layer else m.weights.layers.len - 1;
        const ring_spectrum: rope.Spectrum = if (mtpTrunkRing() == 0)
            .normal
        else if (m.cfg.compress_ratios.items()[tl0] == 0)
            .normal
        else
            .compressed;
        const qr = aw.view(.qr);
        const raw = aw.view(.raw_kv);
        const query = aw.view(.query);
        // The release's DSparkAttention ropes the draft queries and keys at their own positions, inverse-ropes the
        // output and scales the scores by 1/sqrt(head_dim). TF_DSV41_DSPARK_NO_ROPE / _SCALE1 restore the earlier
        // forms for a same-binary A/B.
        const dspark_rope = std.c.getenv("TF_DSV41_DSPARK_NO_ROPE") == null;
        // TF_DSV41_DSPARK_SCALE arms the correct scale on its own (the rope flag keeps its old coupling).
        const dspark_scale = std.c.getenv("TF_DSV41_DSPARK_SCALE1") == null;
        // The release gives every draft row its own consecutive position (its rope slice starts after the main
        // sequence) and its draft rows attend to all of the block's draft keys (the top-k index list carries every
        // draft key for every draft row). TF_DSV41_DSPARK_CHUNK_ALL shares one rope step and drops the draft mask,
        // the earlier one-position/whole-block form, for a same-binary A/B.
        const dspark_chunk_all = std.c.getenv("TF_DSV41_DSPARK_CHUNK_ALL") != null;
        // The draft rows' own mask: the release's draft attention is noncausal across the block, so the mask
        // (TF_DSV41_DSPARK_NO_CHUNK_MASK unset) is the earlier causal form; the arm drops it alone.
        const dspark_no_chunk_mask = std.c.getenv("TF_DSV41_DSPARK_NO_CHUNK_MASK") != null;
        // The drafter block's own matrix shapes and the config's strides.
        if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) std.log.info(
            "mtp shapes: wq_a k={d} n={d} halves={d} cb={d} | wkv k={d} n={d} halves={d} cb={d} | wq_b k={d} n={d} halves={d} cb={d} || hidden={d} head_dim={d} q_lora={d} heads={d} eps={d}",
            .{ blk.attn.wq_a.k, blk.attn.wq_a.n, blk.attn.wq_a.width.halves, @backingInt(blk.attn.wq_a.codebook), blk.attn.wkv.k, blk.attn.wkv.n, blk.attn.wkv.width.halves, @backingInt(blk.attn.wkv.codebook), blk.attn.wq_b.k, blk.attn.wq_b.n, blk.attn.wq_b.width.halves, @backingInt(blk.attn.wq_b.codebook), c.hidden, c.head_dim, c.q_lora, c.heads, c.eps },
        );
        try m.project(enc, blk.attn.wq_a, ws.view(.normalized), qr, rows, c.hidden, c.q_lora, .bf16);
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wq_a", layer, &enc);
        }
        if (dbg) |b| {
            if (layer == 0) try m.projectDebug(enc, blk.attn.wkv, ws.view(.normalized), raw, rows, c.hidden, c.head_dim, .bf16, b);
        } else {
            try m.project(enc, blk.attn.wkv, ws.view(.normalized), raw, rows, c.hidden, c.head_dim, .bf16);
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wkv", layer, &enc);
        }
        ops.encodeRms(enc, &p.ops, qr, tensor(blk.attn.q_norm), qr, .{ .rows = rows, .width = c.q_lora, .x_stride = c.q_lora, .y_stride = c.q_lora, .eps = c.eps });
        ops.encodeRms(enc, &p.ops, raw, tensor(blk.attn.kv_norm), raw, .{ .rows = rows, .width = c.head_dim, .x_stride = c.head_dim, .y_stride = c.head_dim, .eps = c.eps });
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/norms", layer, &enc);
        }
        if (dspark_rope) {
            try p.attention.rotary.encode(enc, raw, .{ .rows = rows, .heads = 1, .width = c.head_dim, .rotary = c.rope_dim, .stride = c.head_dim, .position = position, .step = if (dspark_chunk_all) 0 else 1, .inverse = @intFromBool(false), .spectrum = ring_spectrum });
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/rope_k", layer, &enc);
        }
        // The key rows the pack reads, and the query's low-rank stage.
        if (dbg) |b| {
            if (layer == 0) {
                ops.encodeDump(enc, &p.ops, raw, .{ .buffer = b, .offset = mtp_dbg_raw * 4 }, rows * c.head_dim, true);
                ops.encodeDump(enc, &p.ops, qr, .{ .buffer = b, .offset = mtp_dbg_qr * 4 }, rows * c.q_lora, true);
            }
        }
        try m.project(enc, blk.attn.wq_b, qr, query, rows, c.q_lora, q_width, .bf16);
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wq_b", layer, &enc);
        }
        if (dbg) |b| {
            if (layer == 0) ops.encodeDump(enc, &p.ops, query, .{ .buffer = b, .offset = mtp_dbg_qpre * 4 }, rows * q_width, true);
        }
        if (dspark_rope) {
            try p.attention.rotary.encode(enc, query, .{ .rows = rows, .heads = c.heads, .width = c.head_dim, .rotary = c.rope_dim, .stride = q_width, .position = position, .step = if (dspark_chunk_all) 0 else 1, .inverse = @intFromBool(false), .spectrum = ring_spectrum });
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/rope_q", layer, &enc);
        }
        const chunk = aw.selection.windowChunk();
        try p.attention.cache.encodePack(enc, raw, chunk, .{ .rows = rows, .width = c.head_dim, .stride = c.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/pack", layer, &enc);
        }
        // The release's DSparkAttention reads the main model's keys for the window (`wkv(main_x)`, roped at the
        // main positions, written one row a step into its ring). The drafter's own ring carries three projected
        // conditioning rows plus the drafter's own rows; TF_DSV41_MTP_TRUNK_RING selects the main ring instead.
        const ring_view: View = switch (mtpTrunkRing()) {
            1 => blk: {
                if (slot >= m.caches.len) return error.InvalidLaneSlot;
                break :blk try m.caches[slot].ring(tl0);
            },
            2 => blk: {
                if (slot >= m.caches.len) return error.InvalidLaneSlot;
                break :blk try m.caches[slot].ring(m.weights.layers.len - 1);
            },
            else => try state.ring(layer),
        };
        try p.attention.sparse.encode(enc, query, ring_view, chunk, chunk, aw.selection.positions(), tensor(blk.attn.sink), query, aw.dense(), aw.splitPartials(), .{ .rows = rows, .heads = c.heads, .width = c.head_dim, .q_stride = q_width, .position = position, .chunk_rows = rows, .ratio = 0, .compressed_rows = 0, .topk = 0, .splits = 1, .scale = if (dspark_scale) 0.04419417382415922 else 1.0, .chunk_all = @intFromBool(dspark_chunk_all or dspark_no_chunk_mask) });
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/sparse", layer, &enc);
        }
        if (dspark_rope) {
            try p.attention.rotary.encode(enc, query, .{ .rows = rows, .heads = c.heads, .width = c.head_dim, .rotary = c.rope_dim, .stride = q_width, .position = position, .step = if (dspark_chunk_all) 0 else 1, .inverse = @intFromBool(true), .spectrum = ring_spectrum });
        }
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/invrope", layer, &enc);
        }
        // The decoded key table the scan read (the ring's 128 rows, then the chunk's five) and the attention's own
        // output.
        if (dbg) |b| {
            if (layer == 0) {
                ops.encodeDump(enc, &p.ops, aw.dense(), .{ .buffer = b, .offset = mtp_dbg_dense * 4 }, 133 * c.head_dim, true);
                ops.encodeDump(enc, &p.ops, query, .{ .buffer = b, .offset = mtp_dbg_qpost * 4 }, rows * q_width, true);
            }
        }
        const grouped = ws.view(.logits).shifted(129280 * 4);
        const per_group = q_width / blk.attn.wo_a.len;
        // THE EIGHT OUTPUT GROUPS IN ONE INDEXED DISPATCH -- the trunk's own form. The drafter's layout is the
        // trunk's: q is row-major, so `rows * o_groups` rows of `per_group` are exactly the (row, group) slices,
        // and `grouped` is the same row-major layout at `o_lora`. The drafter's eight slices are the same shape as
        // the trunk's (k 4096, n 1024). TF_DSV41_NO_FUSE_DRAFT_WO_A=1 restores the eight separate projections for a
        // same-binary A/B.
        const fuse_wo_a = blk.attn.wo_a_descriptors != null and rows <= 8 and dbg == null and
            std.c.getenv("TF_DSV41_NO_FUSE_DRAFT_WO_A") == null;
        if (fuse_wo_a) {
            for (&blk.attn.wo_a) |*wm| exl3.useMatrix(enc, wm.*);
            try exl3.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = blk.attn.wo_a_descriptors.?, .offset = 0 }, .count = c.o_groups, .k = per_group, .n = c.o_lora }, convert(query), convert(grouped), m.workspace.projection(), .{
                .count = rows * c.o_groups,
                .x_stride = per_group,
                .y_stride = c.o_lora,
                .output = .bf16,
                .output_rows = rows * c.o_groups,
                .routes = .{ .ids = convert(aw.idMap()), .repeat = 1 },
            });
        } else for (blk.attn.wo_a, 0..) |matrix, group| {
            // The grouped output is row-major at group_output (eight o_lora groups a row), as the backbone's call
            // has it; the row stride was c.o_lora here, so the groups overlapped and the wo_b read stale rows --
            // the drafter's filler rows went NaN. Fixed.
            if (dbg) |b| {
                if (layer == 0 and group == 0)
                    try m.projectDebugSums(enc, matrix, query.shifted(group * per_group * 2), grouped.shifted(group * c.o_lora * 2), rows, q_width, c.o_groups * c.o_lora, .bf16, b)
                else
                    try m.project(enc, matrix, query.shifted(group * per_group * 2), grouped.shifted(group * c.o_lora * 2), rows, q_width, c.o_groups * c.o_lora, .bf16);
            } else {
                try m.project(enc, matrix, query.shifted(group * per_group * 2), grouped.shifted(group * c.o_lora * 2), rows, q_width, c.o_groups * c.o_lora, .bf16);
            }
        }
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wo_a", layer, &enc);
        }
        // The wo_a output (the wo_b input) where the NaN is born.
        if (dbg) |b| {
            if (layer == 0) ops.encodeDump(enc, &p.ops, grouped, .{ .buffer = b, .offset = mtp_dbg_grouped * 4 }, rows * c.o_groups * c.o_lora, true);
        }
        try m.project(enc, blk.attn.wo_b, grouped, ws.view(.branch), rows, c.o_groups * c.o_lora, c.hidden, .bf16);
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/wo_b", layer, &enc);
        }
        // The release publishes only the main rows into the draft ring: each step writes `main_kv` at
        // `start_pos % win`, never the draft rows. Ours also publishes the draft's own rows into the main ring at
        // the positions they would occupy; TF_DSV41_NO_DRAFT_PUBLISH=1 skips that publish (the arm).
        if (std.c.getenv("TF_DSV41_NO_DRAFT_PUBLISH") == null)
            try p.attention.cache.publishWindow(enc, chunk, try state.ring(layer), rows, position);
        enc.barrier();
        if (prof) |pf| {
            if (pf.blk_on) pf.split("blk", "attn/publish", layer, &enc);
        }
    }

    /// The drafter's conditioning, split into its own command buffers so that each part's GPU window can be read on
    /// its own.
    const CondProf = struct {
        queue: mtl.Queue,
        cb: mtl.CommandBuffer,
        enc: mtl.ComputeEncoder,
        on: bool,
        /// The BLOCKS' split (TF_DSV41_BLOCK_PROF=1) -- the drafter's three blocks have never been split, and they
        /// are the larger half of the drafter's forward.
        blk_on: bool = false,
        /// The block the parts belong to (the MoE's own calls do not know it).
        block: u32 = 0,
        total_ms: f64 = 0,
        parts: u32 = 0,

        /// End the buffer, commit it, wait, print the part's GPU window, and start a fresh buffer. The encoder
        /// comes by POINTER, so the caller's own copy is updated too -- the blocks' parts are cut from inside the
        /// attention and the MoE, whose callers encode into the fresh encoder. The tag names the section ("cond" or
        /// "blk"), the label the part.
        fn split(self: *CondProf, tag: []const u8, label: []const u8, block: u32, enc: *mtl.ComputeEncoder) void {
            enc.end();
            self.cb.commit();
            self.cb.wait();
            const ms = self.cb.gpuSeconds() * 1e3;
            self.total_ms += ms;
            self.parts += 1;
            std.log.info("{s} part {s} b{d}: {d:.3} ms", .{ tag, label, block, ms });
            self.cb = self.queue.commandBuffer();
            const e = self.cb.compute(if (concurrentOn()) .concurrent else .serial);
            self.enc = e;
            enc.* = e;
        }
    };

    /// TF_DSV41_MTP_DRAFTS: five rows a round (the pending token and four fillers) through three blocks of the
    /// drafter's own weights, then the shared head; the head writes from `.logits` row 2 on. THE FIRST DRAFT's
    /// ACCEPTANCE, smoothed -- the adaptive depth policy's signal. It is the fraction of rounds whose kept count is
    /// at least two (the pending token plus the first draft), and it is the SAME at every depth (the yield is not:
    /// it scales with the depth, so the novel text yields 0.62 at depth 1 and 1.22 at depth 3 while the predictable
    /// text yields 1.07 and 3.02 -- one threshold cannot separate the texts). The policy's warm-up length -- a
    /// running mean over the first `k` samples, then the shipped EMA. `TF_DSV41_MTP_ADAPT_WARM=0` is the pure EMA
    /// (the same-binary control).
    var mtp_warm_cache: ?u32 = null;
    fn mtpWarmRounds() u32 {
        if (mtp_warm_cache) |v| return v;
        const raw = std.c.getenv("TF_DSV41_MTP_ADAPT_WARM") orelse {
            mtp_warm_cache = 4;
            return 4;
        };
        const n = std.fmt.parseInt(u32, std.mem.span(raw), 10) catch 0;
        mtp_warm_cache = @min(n, 16);
        return mtp_warm_cache.?;
    }

    /// The lane's first-draft acceptance (the adaptive depth policy's signal), smoothed.
    pub fn noteAcceptSlot(m: *Model, slot: u32, hit: bool) void {
        if (slot >= m.mtp_accept_slot.len) return;
        const v: f64 = if (hit) 1.0 else 0.0;
        // THE FIRST SAMPLE SETS THE SIGNAL (a request's first round is a valid observation of the first draft's
        // acceptance). The pure-EMA form (the signal starts at zero) costs the PREDICTABLE text ten rounds of the
        // cheap window before it passes UP -- a tenth of the warm probe's rounds at the cheap rate.
        if (m.mtp_accept_seen_slot[slot] == 0) {
            m.mtp_accept_slot[slot] = v;
        } else if (m.mtp_accept_seen_slot[slot] < mtpWarmRounds()) {
            const n: f64 = @floatFromInt(m.mtp_accept_seen_slot[slot] + 1);
            m.mtp_accept_slot[slot] = (m.mtp_accept_slot[slot] * (n - 1.0) + v) / n;
        } else {
            m.mtp_accept_slot[slot] = m.mtp_accept_slot[slot] * 0.85 + v * 0.15;
        }
        m.mtp_accept_seen_slot[slot] += 1;
    }

    /// THE ROUND'S OWN PER-DRAFT ACCEPTANCE as the policy's signal (the same EMA and the same first-sample rule as
    /// `noteAcceptSlot`). At depth 1 it IS the first-draft hit (kept/1); at depth 3 it falls to ~0.41 on novel text
    /// -- where the first-draft sample stays ~0.99 because a deep window's first draft is usually accepted while
    /// its second and third are not. That is the depth-awareness the policy needs to leave the deep window when the
    /// deep window stops paying.
    pub fn noteAcceptRatioSlot(m: *Model, slot: u32, ratio: f64) void {
        if (slot >= m.mtp_round_slot.len) return;
        const v = std.math.clamp(ratio, 0.0, 1.0);
        if (m.mtp_round_seen_slot[slot] == 0) {
            m.mtp_round_slot[slot] = v;
        } else {
            m.mtp_round_slot[slot] = m.mtp_round_slot[slot] * 0.85 + v * 0.15;
        }
        m.mtp_round_seen_slot[slot] += 1;
    }

    /// The round's per-draft acceptance (the hybrid policy's stay-deep signal).
    pub fn mtpRoundSlot(m: *Model, slot: u32) f64 {
        if (slot >= m.mtp_round_slot.len) return 0;
        return m.mtp_round_slot[slot];
    }

    /// The lane's first-draft acceptance (the adaptive depth policy's signal).
    pub fn mtpAcceptSlot(m: *Model, slot: u32) f64 {
        if (slot >= m.mtp_accept_slot.len) return 0;
        return m.mtp_accept_slot[slot];
    }

    /// The signal is per REQUEST -- a new request starts from a clean signal.
    pub fn resetAcceptSlot(m: *Model, slot: u32) void {
        if (slot >= m.mtp_accept_slot.len) return;
        m.mtp_accept_slot[slot] = 0;
        m.mtp_accept_seen_slot[slot] = 0;
        m.mtp_round_slot[slot] = 0;
        m.mtp_round_seen_slot[slot] = 0;
    }

    pub fn proposeTokens(m: *Model, slot: u32, pending: u32, count: u32) ![4]u32 {
        const dspark = m.weights.dspark orelse return error.NoDrafter;
        // Bisect arms: TF_DSV41_MTP_STUB returns before the draft forward, TF_DSV41_MTP_STUB_COND after the
        // conditioning. Both leave the engine's hand-off and verify in place, so they separate the machinery
        // from the draft round's own execution.
        if (std.c.getenv("TF_DSV41_MTP_STUB") != null) return .{ 0, 0, 0, 0 };
        // The pending row is this round's first verified row; start its page-in now, under the drafter's own GPU
        // window, instead of at the verify's stageRows.
        if (engramPendingOn()) prefetchPending(m, slot, pending);
        // The draft round runs its own boundaries through the shared coefficient views and the stream parity: a
        // main forward re-initialises them, but the *parity* chooses which stream the next boundary reads, so the
        // model fields are saved and put back around the round.
        const alt0 = m.stream_alt;
        const pend0 = m.pending_expand;
        defer {
            m.stream_alt = alt0;
            m.pending_expand = pend0;
        }
        const ws = &m.workspace;
        const p = &m.pipelines;
        const aw = &m.attention_workspace;
        const state = try m.drafterState();
        // ROUND 529: the conditioning's cursor is THIS slot's committed position, not lane 0's.
        const slot_pos = m.caches[slot].position;
        state.position = slot_pos;
        // THE DRAFTER RUNS `depth + 1` ROWS, NOT FIVE. The chain's step s reads the logits row s and the bias of
        // the token before it, and writes the pick into IDS[s+1]; the hand-off takes picks 0..count-1 (the lane's
        // depth, at most three). The five-row form computed two rows whose picks (the fourth and the fifth) the
        // hand-off never read, and every extra row the drafter runs is published into its own 128-row attention
        // window at its own position -- so the extra rows are not free: they displace the real history the
        // drafter's rows attend over. The rows are the ones the chain reads plus the one the chain's own step
        // needs, so `depth + 1` rows is the form. TF_DSV41_MTP_ROWS restores the five-row form for a same-binary
        // A/B.
        const n: u32 = if (std.c.getenv("TF_DSV41_MTP_ROWS")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 5)
        else
            @min(@max(count, 1) + 1, 5);
        var out: [4]u32 = @splat(0);
        var cb = m.queue.commandBuffer();
        var enc = cb.compute(if (concurrentOn()) .concurrent else .serial);
        // The conditioning's per-part GPU split, and the host's own encode time against the GPU's window. Neither
        // flag changes a byte of the shipped path.
        var cond_prof = CondProf{ .queue = m.queue, .cb = cb, .enc = enc, .on = std.c.getenv("TF_DSV41_COND_PROF") != null, .blk_on = std.c.getenv("TF_DSV41_BLOCK_PROF") != null };
        const host_prof = std.c.getenv("TF_DSV41_HOST_PROF") != null;
        const t_start: f64 = if (host_prof) mtl.clock.seconds() else 0;
        // The conditioning: the three target layers' stream means (the taps, one 15,360-wide row) through
        // main_proj and main_norm, then each stage's keys into its ring -- what the draft rows attend to.
        // The conditioning: the last `nc` rows of the rolling tap ring (the decode window's own rows, captured a
        // token a forward) through main_proj and main_norm, then each stage's keys into its ring. Round 320: the
        // ring used to hold only the current pass's row (`taps_n = rows = 1` at decode), so 127 of the 128 window
        // slots the draft rows read were uninitialised packed cache and the filler rows went NaN.
        const pos_now = slot_pos;
        // The captured rows are the last `taps_rows_last` rows of the last forward, at their own positions
        // [pos_now - cap_n, pos_now). The release writes each main row at its own position and ropes the draft
        // rows at the positions after the sequence; the earlier form published the valid prefix at the suffix's
        // positions, shifting the context by the rows the round kept.
        const cap_n: u32 = @min(m.taps_rows_last, 128);
        const cap_pos: u32 = pos_now -| cap_n;
        const true_pos = dsparkTruePosOn();
        // The capture's valid rows: the rollback's prefix (the rejected rows' taps are not the committed
        // history), or the whole capture when no rollback ran (the whole window landed).
        const cap_rows: u32 = @min(@min(m.taps_rows_last, m.taps_n), 128);
        // The conditioning publishes the capture's whole rows at their own positions, so the ring holds the main
        // model's rows everywhere; the earlier form published only the kept prefix, leaving a hole at every
        // rejected position. TF_DSV41_DSPARK_CAPTURE_ALL=0 restores the prefix form.
        const dspark_capture_all: u32 = if (std.c.getenv("TF_DSV41_DSPARK_CAPTURE_ALL")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 1)
        else
            1;
        const tail: u32 = if (std.c.getenv("TF_DSV41_DSPARK_COND_ONE") != null)
            1
        else if (m.taps_rows_valid != 0)
            @min(m.taps_rows_valid, cap_rows)
        else
            cap_rows;
        // The published rows' positions: the capture's own (the release's form) or the earlier shifted form.
        const pub_pos: u32 = if (true_pos) cap_pos else pos_now - tail;
        // The drafter's own rows' rope position: the next token's position (the release ropes every draft row at
        // start_pos + seqlen + i) or the earlier slot position.
        const draft_pos: u32 = if (true_pos) cap_pos + tail else slot_pos;
        // The windowed context's row count. With the aligned ring the rows are the positions' own -- the ring's
        // slots are the positions. The count is rounded down to a multiple of four so the dense multi-row body
        // takes it; 1 keeps the one-row body.
        var nc: u32 = 1;
        if (condRowsWanted() != 0) {
            const have: u32 = if (condAlignOn()) @min(pos_now, 128) else @min(m.taps_n, pos_now);
            const want: u32 = @min(condRowsWanted(), have);
            nc = if (want >= 4) (want / 4) * 4 else want;
        }
        if (nc > 1) {
            const base = m.cond.?;
            const cond = View{ .buffer = base, .offset = 0 };
            const raw = View{ .buffer = base, .offset = 128 * 5120 * 2 };
            const chunk = View{ .buffer = base, .offset = 128 * 5120 * 2 + 128 * 512 * 2 };
            // The ring is circular: the oldest live row is `first`, and the last `nA` rows are contiguous there.
            // With the aligned ring the window's first row is the slot of the position pos_now - nc.
            const first: u32 = if (condAlignOn()) (pos_now - nc) % 128 else (m.taps_row + 128 - nc) % 128;
            const n_a: u32 = @min(nc, 128 - first);
            // The eight-row dense hint -- this projection's window wraps, so the smaller of its two pieces runs on
            // 1..63 rows, and the tile's gate is 64.
            try m.projectDense8(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = @as(usize, first) * 3 * 5120 * 2 }, cond, n_a, 3 * 5120, 5120, .bf16);
            if (n_a < nc)
                try m.projectDense8(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = 0 }, View{ .buffer = cond.buffer, .offset = cond.offset + @as(usize, n_a) * 5120 * 2 }, nc - n_a, 3 * 5120, 5120, .bf16);
            m.normalize(enc, cond, dspark.main_norm, cond, nc);
            enc.barrier();
            // The conditioning's parts, one command buffer each.
            if (cond_prof.on) cond_prof.split("cond", "tapproj+norm", 0, &enc);
            cb = cond_prof.cb;
            // THE THREE PER-BLOCK `wkv` PROJECTIONS AS ONE GROUPED INDEXED DISPATCH. The grouped form lays the
            // window out three times (one expert a 128-row tile group) and takes one dispatch with three times the
            // threadgroups, so the same bytes decode in parallel. TF_DSV41_NO_COND_WKV_FUSE=1 restores the three
            // calls (the A/B).
            const hd = m.cfg.head_dim;
            const fuse_wkv = m.cond_wkv != null and dspark.cond_wkv_descriptors != null and
                std.c.getenv("TF_DSV41_NO_COND_WKV_FUSE") == null;
            var raw_kv = [3]View{ raw, raw, raw };
            if (fuse_wkv) {
                const cb3 = m.cond_wkv.?;
                const in3 = View{ .buffer = cb3, .offset = 0 };
                const out3 = View{ .buffer = cb3, .offset = 3 * 128 * 5120 * 2 };
                const ids_off: usize = 3 * 128 * 5120 * 2 + 3 * 128 * hd * 2;
                for (0..3) |g| ops.encodeCopy(enc, &p.ops, cond, .{ .buffer = cb3, .offset = @as(usize, g) * 128 * 5120 * 2 }, .{ .rows = nc, .width = 5120, .x_stride = 5120, .y_stride = 5120 });
                enc.barrier();
                try exl3.encodeIndexed(&p.projection, enc, .{ .descriptors = .{ .buffer = dspark.cond_wkv_descriptors.?, .offset = 0 }, .count = 3, .k = 5120, .n = hd }, convert(in3), convert(out3), m.workspace.projection(), .{
                    .count = 3 * 128,
                    .x_stride = 5120,
                    .y_stride = hd,
                    .output = .bf16,
                    .input_ids = .{ .buffer = cb3, .offset = ids_off + 384 * 4 },
                    .input_rows = 3 * 128,
                    .output_ids = .{ .buffer = cb3, .offset = ids_off + 768 * 4 },
                    .output_rows = 3 * 128,
                    .routes = .{ .ids = .{ .buffer = cb3, .offset = ids_off }, .repeat = 1 },
                    .grouped = true,
                    .tile_rows = 64,
                });
                enc.barrier();
                if (cond_prof.on) cond_prof.split("cond", "wkv3", 0, &enc);
                cb = cond_prof.cb;
                for (0..3) |g| raw_kv[g] = .{ .buffer = out3.buffer, .offset = out3.offset + @as(usize, g) * 128 * hd * 2 };
            }
            var s_i: u32 = 0;
            while (s_i < dspark.blocks.len) : (s_i += 1) {
                const blk = &dspark.blocks[s_i];
                if (!fuse_wkv) {
                    const cond_wkv_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_COND_WKV") != null) 2 else 1;
                    for (0..cond_wkv_passes) |_| try m.project(enc, blk.attn.wkv, cond, raw, nc, 5120, hd, .bf16);
                    // The conditioning's parts, one command buffer each.
                    if (cond_prof.on) cond_prof.split("cond", "wkv", s_i, &enc);
                    cb = cond_prof.cb;
                }
                const raw_g = if (fuse_wkv) raw_kv[@intCast(s_i)] else raw;
                ops.encodeRms(enc, &p.ops, raw_g, tensor(blk.attn.kv_norm), raw_g, .{ .rows = nc, .width = hd, .x_stride = hd, .y_stride = hd, .eps = m.cfg.eps });
                enc.barrier();
                if (cond_prof.on) cond_prof.split("cond", "rms", s_i, &enc);
                cb = cond_prof.cb;
                // The release roped the main rows' keys at their own positions before they enter the draft ring.
                if (condRopeOn()) {
                    try p.attention.rotary.encode(enc, raw_g, .{ .rows = nc, .heads = 1, .width = hd, .rotary = m.cfg.rope_dim, .stride = hd, .position = pos_now - nc, .step = 1, .inverse = 0, .spectrum = .normal });
                    enc.barrier();
                }
                // The conditioning's parts, one command buffer each.
                if (cond_prof.on) cond_prof.split("cond", "rope", s_i, &enc);
                cb = cond_prof.cb;
                try p.attention.cache.encodePack(enc, raw_g, chunk, .{ .rows = nc, .width = hd, .stride = hd, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
                enc.barrier();
                // The conditioning's parts, one command buffer each.
                if (cond_prof.on) cond_prof.split("cond", "pack", s_i, &enc);
                cb = cond_prof.cb;
                try p.attention.cache.publishWindow(enc, chunk, try state.ring(s_i), nc, pos_now - nc);
                enc.barrier();
                if (cond_prof.on) cond_prof.split("cond", "publish", s_i, &enc);
                cb = cond_prof.cb;
            }
        } else {
            // The release ropes the committed main rows' keys at their own position before they enter the stage's
            // ring (its DSparkAttention applies the rotary to `main_kv`); TF_DSV41_DSPARK_NO_RING_ROPE=1 restores
            // the earlier unroped publish.
            const dspark_ring_rope = std.c.getenv("TF_DSV41_DSPARK_NO_RING_ROPE") == null;
            if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) std.log.info(
                "mtp cond: taps_row={d} taps_n={d} pos_now={d} publish_pos={d} tap_row={d} ring_rope={} || cap_n={d} cap_pos={d} tail={d} valid={d} pub_pos={d} draft_pos={d} true_pos={}",
                .{ m.taps_row, m.taps_n, pos_now, pos_now -| 1, (m.taps_row + 127) % 128, dspark_ring_rope, cap_n, cap_pos, tail, m.taps_rows_valid, pub_pos, draft_pos, true_pos },
            );
            // The conditioning publishes the taps ring's whole tail -- the rows the last capture kept, which are
            // consecutive positions -- at their own positions. The port published ONE row a round, so a round that
            // landed two tokens advanced the position by two and left the position between them unwritten: the
            // drafter's 128-row window then held a row published 128 positions ago (with that old position's rope
            // phase). After a prompt the first round published one of the prompt's 128 tail rows, so the window was
            // uninitialised for the first 128 rounds. TF_DSV41_DSPARK_COND_ONE restores the port's one-row form.
            // The capture's rows are the ring's tail, but the no-re-feed port's rollback keeps only a PREFIX of the
            // capture: the rejected rows' taps are not the committed history, so the conditioning publishes the
            // prefix, not the tail. The prefix's rows are the capture's FIRST rows, so their positions are the
            // capture's own -- [cap_pos, cap_pos + tail) -- and the ring's slots follow from them. The port's
            // [pos_now - tail, pos_now) are the SUFFIX's positions. The published rows are the ring's [base,
            // base+tail); the capture's first row is its base, and the rows may wrap the ring's end -- then the
            // second piece starts at row zero. With the aligned ring the capture's first row is the slot of the
            // position pos_now - tail (the kept prefix's first position).
            const base: u32 = if (condAlignOn())
                (pos_now - tail) % 128
            else if (m.taps_rows_valid != 0)
                (m.taps_row + 128 - cap_rows) % 128
            else
                (m.taps_row + 128 - tail) % 128;
            // The published rows are the capture's whole (the main model's rows at their own positions, the
            // rejected ones included) or only the kept prefix (the earlier form). The publish's position is the
            // capture's first row's position either way, so the rows land where they belong.
            const pub_rows: u32 = if (dspark_capture_all != 0) cap_rows else tail;
            const n_a: u32 = @min(pub_rows, 128 - base);
            // The doubling arm for the conditioning's tap projection (idempotent: the same input gives the same
            // output, so the reply is unchanged and only the cost doubles).
            const cond_proj_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_COND_PROWS") != null) 2 else 1;
            for (0..cond_proj_passes) |_| try m.project(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = @as(u64, base) * 3 * 5120 * 2 }, ws.view(.shared_down), n_a, 3 * 5120, 5120, .bf16);
            if (n_a < pub_rows) {
                const dst = ws.view(.shared_down);
                try m.project(enc, dspark.main_proj, .{ .buffer = m.taps, .offset = 0 }, View{ .buffer = dst.buffer, .offset = dst.offset + @as(usize, n_a) * 5120 * 2 }, pub_rows - n_a, 3 * 5120, 5120, .bf16);
            }
            m.normalize(enc, ws.view(.shared_down), dspark.main_norm, ws.view(.shared_down), pub_rows);
            enc.barrier();
            const chunk = aw.selection.windowChunk();
            var s_i: u32 = 0;
            while (s_i < dspark.blocks.len) : (s_i += 1) {
                const blk = &dspark.blocks[s_i];
                const raw = aw.view(.raw_kv);
                try m.project(enc, blk.attn.wkv, ws.view(.shared_down), raw, pub_rows, 5120, m.cfg.head_dim, .bf16);
                ops.encodeRms(enc, &p.ops, raw, tensor(blk.attn.kv_norm), raw, .{ .rows = pub_rows, .width = m.cfg.head_dim, .x_stride = m.cfg.head_dim, .y_stride = m.cfg.head_dim, .eps = m.cfg.eps });
                enc.barrier();
                if (dspark_ring_rope) {
                    // Every published row carries its rope phase. The rope covered `tail` rows (the kept prefix)
                    // while the pack wrote `pub_rows`, so rows past the first were packed unroped and the drafter's
                    // attention read unroped keys against a roped query. =1 ropes at pub_pos (the earlier
                    // positions); =2 ropes at cap_pos (their own positions, the release's form).
                    const rope_mode = mtpRopeAllOn();
                    const rope_rows: u32 = if (rope_mode != 0) pub_rows else tail;
                    const rope_pos: u32 = if (rope_mode == 2) cap_pos else pub_pos;
                    try p.attention.rotary.encode(enc, raw, .{ .rows = rope_rows, .heads = 1, .width = m.cfg.head_dim, .rotary = m.cfg.rope_dim, .stride = m.cfg.head_dim, .position = rope_pos, .step = 1, .inverse = 0, .spectrum = .normal });
                    enc.barrier();
                }
                try p.attention.cache.encodePack(enc, raw, chunk, .{ .rows = pub_rows, .width = m.cfg.head_dim, .stride = m.cfg.head_dim, .packed_stride = kv.cache.ring_stride, .block = 32, .kind = .window_fp8 });
                enc.barrier();
                // The instrument arm TF_DSV41_DSPARK_NO_COND skips the publish, so the drafter's ring keeps
                // whatever it held (the prompt's tail) -- a STALE context. If the drafter's head input and the
                // trunk's barely move, the ring's contents never reach the drafter's output.
                if (std.c.getenv("TF_DSV41_DSPARK_NO_COND") == null)
                    // The value 2 computes the capture's whole rows but publishes only the kept prefix, so the arm
                    // separates the ring's CONTENT from the extra rows' COMPUTATION.
                    try p.attention.cache.publishWindow(enc, chunk, try state.ring(s_i), if (dspark_capture_all == 2) tail else pub_rows, pub_pos);
                enc.barrier();
            }
        }
        // THE DRAFTER'S STOP POINT 1 (after the conditioning). The arm commits what was encoded, so the command
        // buffer's completion fires.
        if (mtpStopPoint() == 1 or std.c.getenv("TF_DSV41_MTP_STUB_COND") != null) {
            enc.end();
            cb.commit();
            cb.wait();
            std.log.info("mtp stop 1: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
            try completed(cb);
            return .{ 0, 0, 0, 0 };
        }
        // THE DRAFTER'S OWN EARLY COMMIT. Same mechanism and same guarantees as: a command-buffer boundary is a
        // full barrier, the queue is in-order, and the final wait implies the earlier buffer. `TF_DSV41_MTP_PIPE=0`
        // is the same-binary control; an instrument that owns the buffer (`COND_PROF`, `BLOCK_PROF`) keeps the
        // shipped single-buffer form.
        var pipe_prev: mtl.CommandBuffer = undefined;
        var pipe_prev_set = false;
        const mtp_pipe: bool = blk: {
            if (cond_prof.on or cond_prof.blk_on) break :blk false;
            const raw = std.c.getenv("TF_DSV41_MTP_PIPE") orelse break :blk true;
            break :blk !std.mem.eql(u8, std.mem.span(raw), "0");
        };
        if (mtp_pipe) {
            enc.end();
            cb.commit();
            pipe_prev = cb;
            pipe_prev_set = true;
            cb = m.queue.commandBuffer();
            enc = cb.compute(if (concurrentOn()) .concurrent else .serial);
            cond_prof.cb = cb;
            cond_prof.enc = enc;
        }
        const ids = try m.drafterRound(pending, n);
        ops.encodeEmbedding(enc, &p.ops, ids, tensor(m.weights.embed), ws.view(.branch), .{ .rows = n, .width = 5120, .table_stride = 5120, .y_stride = 5120, .copies = 1 });
        enc.barrier();
        if (cond_prof.blk_on) cond_prof.split("blk", "prologue/embed", 0, &enc);
        const dbg: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_MTP_DEBUG_ROWS") != null) try m.device.buffer(mtp_dbg_total3 * 4, mtl.ResourceOptions.shared) else null;
        defer if (dbg) |b| b.deinit();
        if (dbg) |b| ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 0 }, 5 * 5120, true);
        // The mHC's expanded stream and its `pre` vector carry across the drafter's blocks, as the trunk's do (the
        // expand is once, before the loop, and the release's DSparkBlock threads `pre_mix` through each stage).
        // Expanding inside the loop reset the stream to the branch and the pre to one-hot at every block.
        mhc.encodeExpand(enc, &p.mhc, ws.view(.branch), ws.view(.stream), ws.view(.previous_pre), .{ .rows = n, .width = 5120, .x_stride = 5120, .stream_stride = 20480, .pre_stride = 4 });
        if (cond_prof.blk_on) cond_prof.split("blk", "prologue/expand", 0, &enc);
        var k: u32 = 0;
        while (k < dspark.blocks.len) : (k += 1) {
            const blk = &dspark.blocks[k];
            cond_prof.block = k;
            m.mixes(enc, blk.hc_attn, ws.view(.attention_pre), n);
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            if (cond_prof.blk_on) cond_prof.split("blk", "mhc_attn", k, &enc);
            // The mHC's collapsed rows (the norm's input) and their magnitudes.
            if (dbg) |b| {
                if (k == 0) {
                    ops.encodeDump(enc, &p.ops, ws.view(.collapsed), .{ .buffer = b, .offset = mtp_dbg_coll * 4 }, n * 5120, true);
                    ops.encodeDump(enc, &p.ops, ws.view(.previous_pre), .{ .buffer = b, .offset = mtp_dbg_pre * 4 }, n * 4, false);
                }
            }
            m.normalize(enc, ws.view(.collapsed), blk.attn_norm, ws.view(.normalized), n);
            if (cond_prof.blk_on) cond_prof.split("blk", "attn_norm", k, &enc);
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 1 * 5 * 5120 * 4 }, 5 * 5120, true);
            try m.draftAttention(enc, blk, state, slot, k, n, @intCast(if (std.c.getenv("TF_DSV41_DSPARK_POS1") != null) (draft_pos -| 1) else draft_pos), dbg, &cond_prof);
            enc = cond_prof.enc;
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 2 * 5 * 5120 * 4 }, 5 * 5120, true);
            m.post(enc, n);
            m.mixes(enc, blk.hc_ffn, ws.view(.previous_pre), n);
            mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.attention_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
            enc.barrier();
            if (cond_prof.blk_on) cond_prof.split("blk", "mhc_ffn", k, &enc);
            m.normalize(enc, ws.view(.collapsed), blk.ffn_norm, ws.view(.normalized), n);
            if (cond_prof.blk_on) cond_prof.split("blk", "ffn_norm", k, &enc);
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 3 * 5 * 5120 * 4 }, 5 * 5120, true);
            try m.moe(&enc, null, &blk.moe, n, 0, m.cfg.dspark_topk, &cond_prof);
            m.post(enc, n);
            // The branch after the MoE (the stage contributions: the attention's output against the MoE's), so the
            // row's composition is read off the magnitudes.
            if (dbg) |b| if (k == 0) ops.encodeDump(enc, &p.ops, ws.view(.branch), .{ .buffer = b, .offset = 4 * 5 * 5120 * 4 }, 5 * 5120, true);
            enc = cond_prof.enc;
            cb = cond_prof.cb;
        }
        // THE DRAFTER'S STOP POINT 2 (after the three blocks). The arm commits what was encoded, so the command
        // buffer's completion fires.
        if (mtpStopPoint() == 2 or std.c.getenv("TF_DSV41_MTP_STUB_BLOCKS") != null) {
            enc.end();
            cb.commit();
            cb.wait();
            std.log.info("mtp stop 2: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
            try completed(cb);
            return .{ 0, 0, 0, 0 };
        }
        mhc.encodeCollapse(enc, &p.mhc, ws.view(.stream), ws.view(.previous_pre), ws.view(.collapsed), .{ .rows = n, .width = 5120, .stream_stride = 20480, .y_stride = 5120 });
        enc.barrier();
        m.normalize(enc, ws.view(.collapsed), dspark.norm, ws.view(.normalized), n);
        const logits = View{ .buffer = m.mtp_logits.?, .offset = 2 * 129280 * 4 };
        try m.project(enc, m.weights.head, ws.view(.normalized), logits, n, 5120, 129280, .f32);
        enc.barrier();
        // THE DRAFTER'S STOP POINT 3 (after the head projection). The arm commits what was encoded, so the command
        // buffer's completion fires.
        if (mtpStopPoint() == 3 or std.c.getenv("TF_DSV41_MTP_STUB_HEAD") != null) {
            enc.end();
            cb.commit();
            cb.wait();
            std.log.info("mtp stop 3: rows {d} gpu {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 });
            try completed(cb);
            return .{ 0, 0, 0, 0 };
        }
        // The Markov-biased chain (the port's head chain): row k's logits plus the rank-256 bias of the token
        // before it, its argmax the next token; five steps from the pending token to four drafts.
        const emb = dspark.markov_embed orelse return error.NoMarkov;
        const head_embed = dspark.markov_head orelse return error.NoMarkov;
        if (m.markov_ids == null) {
            m.markov_ids = try m.device.buffer(8 * 4, mtl.ResourceOptions.shared);
            m.residency.add(m.markov_ids.?);
        }
        // The chain's step 0 reads the token before the first draft; its picks land in this buffer, seeded here.
        std.mem.writeInt(u32, m.markov_ids.?.contents()[0..4], pending, .little);
        const ids_view = View{ .buffer = m.markov_ids.?, .offset = 0 };
        const part = try m.markovPart();
        const groups: u32 = (129280 + 255) / 256;
        // The tree price needs the *biased* logits' top-2, and PART is the only place the biased logits survive
        // (the pick consumes them). PART is overwritten by the next step, so each step's 505 float2 bests are
        // copied into its own region before the barrier.
        const price_dbg: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_TREE_PRICE") != null)
            try m.device.buffer(5 * 505 * 8, mtl.ResourceOptions.shared)
        else
            null;
        defer if (price_dbg) |b| b.deinit();
        var step: u32 = 0;
        while (step < n) : (step += 1) {
            ops.encodeMarkov(enc, &p.ops, logits, tensor(emb), tensor(head_embed), ids_view, part, .{ .vocab = 129280, .rank = 256, .step = @intCast(step), .pad = @intFromBool(markovBiasOffOn()) });
            if (price_dbg) |b| ops.encodeDump(enc, &p.ops, part, .{ .buffer = b, .offset = @as(u64, step) * groups * 8 }, groups * 2, false);
            enc.barrier();
            ops.encodeMarkovPick(enc, &p.ops, part, ids_view, groups, step);
            enc.barrier();
        }
        // The drafter's logits row 0 against the trunk's own row-0 logits (the verify's first row -- the same
        // prediction the drafter's row 0 makes). The drafter's forward runs after the verify, so the workspace's
        // `.logits` row 0 still holds the trunk's row-0 logits (the drafter's own wo_a writes from row 1 on).
        const top_buf: ?mtl.Buffer = if (std.c.getenv("TF_DSV41_MTP_DEBUG_TOP") != null)
            try m.device.buffer(2 * 129280 * 4 + 5120 * 4, mtl.ResourceOptions.shared)
        else
            null;
        defer if (top_buf) |b| b.deinit();
        if (top_buf) |b| {
            ops.encodeDump(enc, &p.ops, logits, .{ .buffer = b, .offset = 0 }, 129280, false);
            ops.encodeDump(enc, &p.ops, ws.view(.logits), .{ .buffer = b, .offset = 129280 * 4 }, 129280, false);
            ops.encodeDump(enc, &p.ops, ws.view(.normalized), .{ .buffer = b, .offset = 2 * 129280 * 4 }, 5120, false);
            enc.barrier();
        }
        // The host's own encode time, the wait, and the GPU's window -- so the drafter's cost can be split into
        // "the GPU's work" and "the host's encoding" (and, under the conditioning's split, the gap between them).
        const t_encoded: f64 = if (host_prof) mtl.clock.seconds() else 0;
        enc.end();
        cb.commit();
        cb.wait();
        const t_done: f64 = if (host_prof) mtl.clock.seconds() else 0;
        try completed(cb);
        if (price_dbg) |b| {
            const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
            const g2: usize = @as(usize, groups) * 2;
            var ps: usize = 0;
            while (ps < n) : (ps += 1) {
                var v1: f32 = -std.math.inf(f32);
                var id1: u32 = 0;
                var v2: f32 = -std.math.inf(f32);
                var id2: u32 = 0;
                for (0..groups) |g| {
                    const v = f[ps * g2 + g * 2];
                    const id: u32 = @bitCast(f[ps * g2 + g * 2 + 1]);
                    if (v > v1 or (v == v1 and id < id1)) {
                        v2 = v1;
                        id2 = id1;
                        v1 = v;
                        id1 = id;
                    } else if (v > v2 or (v == v2 and id < id2)) {
                        v2 = v;
                        id2 = id;
                    }
                }
                std.log.info("treeprice propose round {d} step {d} top2 {d} {d}", .{ price_round, ps, id1, id2 });
            }
        }
        // The early buffer completed in queue order; check it and count its GPU time so the `mtp forward` line
        // keeps meaning the whole drafter forward.
        var gpu_extra_ms: f64 = 0;
        if (pipe_prev_set) {
            try completed(pipe_prev);
            gpu_extra_ms = pipe_prev.gpuSeconds() * 1e3;
        }
        // The drafter's own forward's GPU time and wall. The stage timer covers the trunk's layers, not this
        // forward -- and the conditioning's cost (the windowed context's projection, the keys, the rope and the
        // publish) lives here.
        if (host_prof) {
            const encode_ms = (t_encoded - t_start) * 1e3;
            const wait_ms = (t_done - t_encoded) * 1e3;
            if (cond_prof.on or cond_prof.blk_on) {
                cond_prof.total_ms += cb.gpuSeconds() * 1e3;
                std.log.info("mtp forward: rows {d} gpu {d:.3} ms (split sum, {d} parts) encode {d:.2} ms wait {d:.2} ms", .{ n, cond_prof.total_ms, cond_prof.parts + 1, encode_ms, wait_ms });
            } else {
                std.log.info("mtp forward: rows {d} gpu {d:.2} ms encode {d:.2} ms wait {d:.2} ms", .{ n, cb.gpuSeconds() * 1e3 + gpu_extra_ms, encode_ms, wait_ms });
            }
        }
        if (!host_prof and cond_prof.blk_on) {
            cond_prof.total_ms += cb.gpuSeconds() * 1e3;
            std.log.info("mtp forward: rows {d} gpu {d:.3} ms (split sum, {d} parts)", .{ n, cond_prof.total_ms, cond_prof.parts + 1 });
        }
        if (top_buf) |b| {
            const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
            var dout: [5]u32 = undefined;
            var dval: [5]f32 = undefined;
            var tout: [5]u32 = undefined;
            var tval: [5]f32 = undefined;
            logitsTopK(f, 129280, &dout, &dval);
            logitsTopK(f + 129280, 129280, &tout, &tval);
            std.log.info("mtp top: drafter argmax={d} vals={d:.4} {d:.4} {d:.4} ids={d} {d} {d} {d} {d}", .{ dout[0], dval[0], dval[1], dval[2], dout[0], dout[1], dout[2], dout[3], dout[4] });
            std.log.info("mtp top: trunk   argmax={d} vals={d:.4} {d:.4} {d:.4} ids={d} {d} {d} {d} {d}", .{ tout[0], tval[0], tval[1], tval[2], tout[0], tout[1], tout[2], tout[3], tout[4] });
            std.log.info("mtp top: agree={} drafter@trunk_argmax={d:.4} trunk@drafter_argmax={d:.4}", .{ dout[0] == tout[0], f[@intCast(tout[0])], f[129280 + @as(usize, @intCast(dout[0]))] });
            // The drafter's head input against the trunk's own head input (the verify's row 0): the cosine says
            // whether the MTP module's hidden lives in the trunk's hidden space at all.
            if (m.mtp_trunk_hidden) |tb| {
                const t: [*]const f32 = @ptrCast(@alignCast(tb.contents()));
                const dh = f + 2 * 129280;
                var dot: f64 = 0;
                var dn: f64 = 0;
                var tn: f64 = 0;
                var c: usize = 0;
                while (c < 5120) : (c += 1) {
                    dot += @as(f64, dh[c]) * @as(f64, t[c]);
                    dn += @as(f64, dh[c]) * @as(f64, dh[c]);
                    tn += @as(f64, t[c]) * @as(f64, t[c]);
                }
                std.log.info("mtp hidden: drafter rms={d:.5} trunk rms={d:.5} cosine={d:.5}", .{ @sqrt(dn / 5120.0), @sqrt(tn / 5120.0), dot / @sqrt(dn * tn) });
            }
        }
        if (std.c.getenv("TF_DSV41_MTP_DEBUG_LOGITS") != null) {
            {
                const ev = m.weights.embed;
                std.log.info("mtp embed buffer {d} bytes offset {d}", .{ ev.buffer.length(), ev.offset });
                const tp: [*]const u16 = @ptrCast(@alignCast(ev.buffer.contents() + ev.offset));
                var kk: usize = 0;
                while (kk < 3) : (kk += 1) {
                    const base = (128799 + kk) * 5120;
                    std.log.info("mtp embed row {d}: {d:.5} {d:.5} {d:.5}", .{ 128799 + kk, asF32(tp[base]), asF32(tp[base + 1]), asF32(tp[base + 2]) });
                }
            }
            if (m.drafter_ids) |idsbuf| {
                const ip: [*]const u32 = @ptrCast(@alignCast(idsbuf.contents()));
                std.log.info("mtp ids: {d} {d} {d} {d} {d} (buffer {d} bytes)", .{ ip[0], ip[1], ip[2], ip[3], ip[4], idsbuf.length() });
            }
            if (dbg) |b| {
                const f: [*]const f32 = @ptrCast(@alignCast(b.contents()));
                inline for (.{ "embed", "attn_in", "attn_out", "moe_in", "moe_out" }, 0..) |name, seg| {
                    var rr: usize = 0;
                    while (rr < 5) : (rr += 1) {
                        var bad: usize = 0;
                        var cc: usize = 0;
                        var acc: f64 = 0;
                        while (cc < 5120) : (cc += 1) {
                            if (!std.math.isFinite(f[(seg * 5 + rr) * 5120 + cc])) bad += 1;
                            acc += @as(f64, f[(seg * 5 + rr) * 5120 + cc]) * @as(f64, f[(seg * 5 + rr) * 5120 + cc]);
                        }
                        // The row's RMS: the attention's contribution (attn_out - embed) and the MoE's are read off
                        // the row magnitudes -- a drafter whose attention contributes nothing reads a flat logits
                        // row (the round's instrument).
                        std.log.info("mtp {s} row {d}: nonfinite={d} rms={d:.5} first={d:.5} {d:.5}", .{ name, rr, bad, @sqrt(acc / 5120.0), f[(seg * 5 + rr) * 5120], f[(seg * 5 + rr) * 5120 + 1] });
                    }
                }
                // The drafter's key rows (the pack's input), its query's two stages, the decoded key table the scan
                // reads, and the attention's output: a row's first non-finite column localises it.
                {
                    var rrk: usize = 0;
                    while (rrk < 5) : (rrk += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_raw + rrk * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp key row {d}: nonfinite={d} maxabs={d:.5}", .{ rrk, bad, mx });
                    }
                    var rrq: usize = 0;
                    while (rrq < 5) : (rrq += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 1280) : (cc += 1) {
                            const v = f[mtp_dbg_qr + rrq * 1280 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp qr row {d}: nonfinite={d} maxabs={d:.5}", .{ rrq, bad, mx });
                    }
                    var rrp: usize = 0;
                    while (rrp < 5) : (rrp += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 32768) : (cc += 1) {
                            const v = f[mtp_dbg_qpre + rrp * 32768 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp qpre row {d}: nonfinite={d} maxabs={d:.5}", .{ rrp, bad, mx });
                    }
                    // The drafter's query's per-head RMS -- the instrument's read of the scale the softmax saw
                    // (flat when every head is tiny, saturated when one is huge).
                    {
                        const r = headRms(f + mtp_dbg_qpre, 64, 512);
                        std.log.info("mtp qpre head rms: min={d:.5} max={d:.5} mean={d:.5}", .{ r.min, r.max, r.mean });
                    }
                    var rrd: usize = 0;
                    while (rrd < 133) : (rrd += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_dense + rrd * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        if (bad != 0 or rrd >= 126) std.log.info("mtp dense row {d}: nonfinite={d} maxabs={d:.5}", .{ rrd, bad, mx });
                    }
                    // THE RING'S DECODED KEYS: the RMS of a ring row and the cosine between two ADJACENT positions
                    // (the slots 127 and 0 are one position apart across the ring's wrap). If the ring's keys are
                    // one vector, the drafter's softmax cannot discriminate the history.
                    {
                        const ring0 = mtp_dbg_dense;
                        const ring1 = mtp_dbg_dense + 127 * 512;
                        const ring2 = mtp_dbg_dense + 126 * 512;
                        var a0: f64 = 0;
                        var a1: f64 = 0;
                        var a2: f64 = 0;
                        var d01: f64 = 0;
                        var c01: f64 = 0;
                        var c02: f64 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v0 = f[ring0 + cc];
                            const v1 = f[ring1 + cc];
                            const v2 = f[ring2 + cc];
                            a0 += @as(f64, v0) * @as(f64, v0);
                            a1 += @as(f64, v1) * @as(f64, v1);
                            a2 += @as(f64, v2) * @as(f64, v2);
                            d01 += (@as(f64, v1) - @as(f64, v0)) * (@as(f64, v1) - @as(f64, v0));
                            c01 += @as(f64, v0) * @as(f64, v1);
                            c02 += @as(f64, v0) * @as(f64, v2);
                        }
                        std.log.info("mtp ring keys: rms(slot0)={d:.5} rms(slot127)={d:.5} rms(slot126)={d:.5} cos(0,127)={d:.5} cos(0,126)={d:.5} rmsdiff(0,127)={d:.5}", .{ @sqrt(a0 / 512.0), @sqrt(a1 / 512.0), @sqrt(a2 / 512.0), c01 / @sqrt(a0 * a1), c02 / @sqrt(a0 * a2), @sqrt(d01 / 512.0) });
                    }
                    var rrz: usize = 0;
                    while (rrz < 5) : (rrz += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 32768) : (cc += 1) {
                            const v = f[mtp_dbg_qpost + rrz * 32768 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp qpost row {d}: nonfinite={d} maxabs={d:.5}", .{ rrz, bad, mx });
                    }
                    var rrc: usize = 0;
                    while (rrc < 5) : (rrc += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_coll + rrc * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp collapsed row {d}: nonfinite={d} maxabs={d:.5}", .{ rrc, bad, mx });
                    }
                    var rre: usize = 0;
                    while (rre < 5) : (rre += 1) {
                        var bad: usize = 0;
                        var cc: usize = 0;
                        while (cc < 4) : (cc += 1) {
                            if (!std.math.isFinite(f[mtp_dbg_pre + rre * 4 + cc])) bad += 1;
                        }
                        std.log.info("mtp pre row {d}: nonfinite={d} vals={d:.5} {d:.5} {d:.5} {d:.5}", .{ rre, bad, f[mtp_dbg_pre + rre * 4], f[mtp_dbg_pre + rre * 4 + 1], f[mtp_dbg_pre + rre * 4 + 2], f[mtp_dbg_pre + rre * 4 + 3] });
                    }
                    var rrn: usize = 0;
                    while (rrn < 5) : (rrn += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[(1 * 5 + rrn) * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp norm row {d}: nonfinite={d} maxabs={d:.5}", .{ rrn, bad, mx });
                    }
                    var rri: usize = 0;
                    while (rri < 5) : (rri += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_pin + rri * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp pin row {d}: nonfinite={d} maxabs={d:.5}", .{ rri, bad, mx });
                    }
                    var rro: usize = 0;
                    while (rro < 5) : (rro += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_prot + rro * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp prot row {d}: nonfinite={d} maxabs={d:.5}", .{ rro, bad, mx });
                    }
                    var rrs: usize = 0;
                    while (rrs < 5) : (rrs += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_psum + rrs * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp psum row {d}: nonfinite={d} maxabs={d:.5}", .{ rrs, bad, mx });
                    }
                    var rrt: usize = 0;
                    while (rrt < 5) : (rrt += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 5120) : (cc += 1) {
                            const v = f[mtp_dbg_rot + rrt * 5120 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp rot row {d}: nonfinite={d} maxabs={d:.5}", .{ rrt, bad, mx });
                    }
                    var rrg: usize = 0;
                    while (rrg < 5) : (rrg += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 8192) : (cc += 1) {
                            const v = f[mtp_dbg_grouped + rrg * 8192 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp grouped row {d}: nonfinite={d} maxabs={d:.5}", .{ rrg, bad, mx });
                    }
                    var rrw: usize = 0;
                    while (rrw < 5) : (rrw += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 1024) : (cc += 1) {
                            const v = f[mtp_dbg_woasums + rrw * 1024 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp woasums row {d}: nonfinite={d} maxabs={d:.5}", .{ rrw, bad, mx });
                    }
                    var rru: usize = 0;
                    while (rru < 5) : (rru += 1) {
                        var bad: usize = 0;
                        var mx: f32 = 0;
                        var cc: usize = 0;
                        while (cc < 512) : (cc += 1) {
                            const v = f[mtp_dbg_sums + rru * 512 + cc];
                            if (!std.math.isFinite(v)) bad += 1 else if (@abs(v) > mx) mx = @abs(v);
                        }
                        std.log.info("mtp sums row {d}: nonfinite={d} maxabs={d:.5}", .{ rru, bad, mx });
                    }
                }
            }
            inline for (.{ .{ "branch", work.Field.branch }, .{ "normalized", work.Field.normalized }, .{ "collapsed", work.Field.collapsed } }) |spec| {
                const v = ws.view(spec[1]);
                const b: [*]const u16 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
                var rrz: usize = 0;
                while (rrz < 5) : (rrz += 1) {
                    const w = b[rrz * 5120 ..];
                    const a0 = asF32(w[0]);
                    const a1 = asF32(w[1]);
                    const a2 = asF32(w[2]);
                    var bad: usize = 0;
                    var cc: usize = 0;
                    while (cc < 5120) : (cc += 1) {
                        if (!std.math.isFinite(asF32(w[cc]))) bad += 1;
                    }
                    std.log.info("mtp {s} row {d}: nonfinite={d} first={d:.5} {d:.5} {d:.5}", .{ spec[0], rrz, bad, a0, a1, a2 });
                }
            }
            const f: [*]const f32 = @ptrCast(@alignCast(m.mtp_logits.?.contents()));
            var row: usize = 0;
            while (row < 5) : (row += 1) {
                var bad: usize = 0;
                var best: f32 = -std.math.inf(f32);
                var best_col: usize = 0;
                var col: usize = 0;
                while (col < 129280) : (col += 1) {
                    const v = f[row * 129280 + col];
                    if (!std.math.isFinite(v)) bad += 1 else if (v > best) {
                        best = v;
                        best_col = col;
                    }
                }
                std.log.info("mtp logits row {d}: nonfinite={d} max={d:.5} at {d}", .{ row, bad, best, best_col });
            }
        }
        const idbytes = m.markov_ids.?.contents();
        if (std.c.getenv("TF_DSV41_DRAFT_DEBUG") != null) {
            const picks: [*]const u32 = @ptrCast(@alignCast(idbytes));
            std.log.info("mtp picks: {d} {d} {d} {d} {d} {d}", .{ picks[0], picks[1], picks[2], picks[3], picks[4], picks[5] });
        }
        const rows_out: u32 = @min(count, n);
        var r: u32 = 0;
        while (r < rows_out) : (r += 1) out[r] = std.mem.readInt(u32, idbytes[(r + 1) * 4 ..][0..4], .little);
        return out;
    }

    /// The stream the boundary state currently lives in (the fused path swaps two buffers).
    fn streamView(m: *const Model) View {
        return m.workspace.view(if (m.stream_alt) .stream_next else .stream);
    }
    fn streamNextView(m: *const Model) View {
        return m.workspace.view(if (m.stream_alt) .stream else .stream_next);
    }
    /// One fused boundary: the pending branch expanded into the other stream buffer, its mix partials, then the
    /// split: this boundary's pre/post/comb from the partials and the lagged `pre_in` collapse into `normed`.
    fn boundary(m: *Model, enc: mtl.ComputeEncoder, bp: *const mhc.BoundaryPipelines, w: weights.Hc, pre_in: View, pre_out: View, norm: checkpoint.Tensor, rows: u32, with_mixes: bool) void {
        if (std.c.getenv("TF_DSV41_SKIP_MHC") != null) return;
        const ws = &m.workspace;
        const cur = if (m.stream_alt) ws.view(.stream_next) else ws.view(.stream);
        const next = if (m.stream_alt) ws.view(.stream) else ws.view(.stream_next);
        const expand = m.pending_expand;
        mhc.encodeBoundary(enc, bp, .{ .rows = rows, .expand = expand, .mixes = with_mixes, .eps = m.cfg.eps, .hc_eps = m.cfg.hc_eps, .iters = m.cfg.hc_sinkhorn_iters }, .{
            .x_old = cur,
            .x_new = next,
            .branch = ws.view(.branch),
            .post = ws.view(.post),
            .comb = ws.view(.comb),
            .fn_ = tensor(w.projection),
            .part = ws.view(.hc_part),
            .base = tensor(w.base),
            .scale = tensor(w.scale),
            .pre_in = pre_in,
            .norm = tensor(norm),
            .normed = ws.view(.normalized),
            .pre_out = pre_out,
            .post_out = ws.view(.post),
            .comb_out = ws.view(.comb),
        });
        if (expand) m.stream_alt = !m.stream_alt;
        m.pending_expand = false;
    }

    /// TF_DSV41_GATE_TREE=0 forces the shipped tf-dense-col gate; the default is the bit-exact tiled tree.
    fn gateTreeOff() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var v = false;
        if (std.c.getenv("TF_DSV41_GATE_TREE")) |e| v = std.mem.eql(u8, std.mem.span(e), "0");
        S.cached = v;
        return v;
    }

    /// TF_DSV41_GATE_WIDE=1 takes the wide (staged) gate of, whose order and logits are the shipped ones.
    fn gateWideOn() bool {
        const S = struct {
            var cached: ?bool = null;
        };
        if (S.cached) |v| return v;
        var v = false;
        if (std.c.getenv("TF_DSV41_GATE_WIDE")) |e| v = !std.mem.eql(u8, std.mem.span(e), "0");
        S.cached = v;
        return v;
    }

    fn moe(m: *Model, enc: *mtl.ComputeEncoder, e: ?*mtl.CommandBuffer, w: *const weights.Moe, rows: u32, layer_index: u32, topk: u32, prof: ?*CondProf) !void {
        var encoder = enc.*;
        // A block's expert count is what its descriptor table was bound with: 384 for the backbone, 128 for the
        // drafter's blocks (the table is the authority, not a literal).
        const experts: u32 = @intCast(w.descriptors[0].length() / @sizeOf(exl3.ExpertDescriptor));
        const ws = &m.workspace;
        const p = &m.pipelines;
        // A block's own router geometry: the backbone's 384 experts and six picks, the drafter's 128 experts and
        // three picks, read from the pack (`dspark_n_routed_experts` / `dspark_num_experts_per_tok`) and the
        // config. Reading the backbone's geometry for both would read past the drafter's 128-row gate matrix and
        // let its picks name experts outside its descriptor table.
        const picks: u32 = if (topk == 0) m.cfg.topk else topk;
        const dspark_block: bool = experts != m.cfg.experts;
        // The drafter's logits live in their own buffer: its gate has 128 columns, so the backbone's 384-wide rows
        // would overlap the drafter's five.
        const logits_dest: View = if (dspark_block) blk: {
            if (m.dspark_logits == null) m.dspark_logits = try m.device.buffer(256 * @as(usize, experts) * 4, mtl.ResourceOptions.shared);
            break :blk View{ .buffer = m.dspark_logits.?, .offset = 0 };
        } else ws.view(.router_logits);
        // TF_DSV41_ROUTE_TENSOR: the tiled path.
        const route_tensor = std.c.getenv("TF_DSV41_ROUTE_TENSOR") != null;
        // TF_DSV41_ROUTE_GEMM: the gate on the tensor units (MLX's gemm_nax, bf16 logits the route kernel reads).
        // The bf16 twin is built a layer at load (weights.bind), not cached once for the model: the gate is per
        // layer, and the single model-level copy made every layer route with layer 0's gate.
        // TF_DSV41_ROUTE_GEMM_MINROWS: the row count the gate GEMM needs before it engages (64 = the shipped
        // condition).
        const gemm_min_rows: u32 = if (std.c.getenv("TF_DSV41_ROUTE_GEMM_MINROWS")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 64)
        else
            64;
        // The gate GEMM is the promoted default for prompt rows. TF_DSV41_ROUTE_GEMM=0 restores the shipped tree.
        const gemm_raw = std.c.getenv("TF_DSV41_ROUTE_GEMM");
        const gemm_on = if (gemm_raw) |r| !std.mem.eql(u8, std.mem.span(r), "0") else true;
        const gemm_env = gemm_on and rows >= gemm_min_rows;
        // The f16 entry is the one that ships for these packs: their router gate is f16, and the GEMM binds both
        // operands to one element type, so the *activation* is converted (bf16 -> f16, exact) and the weight keeps
        // its 10 mantissa bits. A pack whose gate is already bf16 takes the bf16 entry. Both write fp32 logits, so
        // route sees tf-dense-col's precision.
        const gate_f16_gemm = gemm_env and p.ops.gate_f16_f32 != null and w.gate.dtype == .f16;
        const gate_bf16_gemm = !gate_f16_gemm and gemm_env and p.ops.gate_f32 != null and w.gate.dtype == .bf16;
        const route_gemm = gate_f16_gemm or gate_bf16_gemm;
        if (std.c.getenv("TF_DSV41_ROUTE_DEBUG") != null)
            std.log.info("route gemm: flag {} f16 {} bf16f32 {} bf16out {} dtype {s} rows {d} -> {s}", .{
                std.c.getenv("TF_DSV41_ROUTE_GEMM") != null, p.ops.gate_f16_f32 != null, p.ops.gate_f32 != null,
                p.ops.gate != null,                          @tagName(w.gate.dtype),     rows,
                if (route_gemm) "gemm" else "dense-col",
            });
        // TF_DSV41_ROUTE_GEMM_SCRATCH: the GEMM runs but writes a scratch, and the shipped kernel still feeds
        // route — the test that separates "the GEMM's output is wrong" from "the GEMM's presence poisons".
        const gate_scratch = route_gemm and std.c.getenv("TF_DSV41_ROUTE_GEMM_SCRATCH") != null;
        // TF_DSV41_ROUTE_GEMM_PRIVATE: the GEMM writes its own buffer and route reads it there — the arm that keeps
        // the workspace's logits region untouched.
        const gate_private = route_gemm and std.c.getenv("TF_DSV41_ROUTE_GEMM_PRIVATE") != null;
        var gate_pdest: ?mtl.Buffer = null;
        if (route_gemm) {
            const dest = if (gate_scratch or gate_private) blk: {
                const b = try m.device.buffer(@as(usize, rows) * experts * 4, mtl.ResourceOptions.shared);
                gate_pdest = b;
                break :blk View{ .buffer = b, .offset = 0 };
            } else logits_dest;
            if (gate_f16_gemm) {
                if (m.gate_x == null) m.gate_x = try m.device.buffer(@as(usize, m.workspace.plan.rows) * 5120 * 2, mtl.ResourceOptions.shared);
                const xf = View{ .buffer = m.gate_x.?, .offset = 0 };
                ops.encodeBf16ToF16(encoder, &p.ops, ws.view(.normalized), xf, rows * 5120);
                encoder.barrier();
                ops.encodeGateGemmF16F32(encoder, &p.ops, xf, tensor(w.gate), dest, .{ .rows = rows, .k = 5120 });
            } else if (gate_bf16_gemm) {
                ops.encodeGateGemmF32(encoder, &p.ops, ws.view(.normalized), tensor(w.gate), dest, .{ .rows = rows, .k = 5120 });
            }
            encoder.barrier();
        }
        // Default on for prompt rows; TF_DSV41_GATE_TREE=0 forces the shipped tf-dense-col back.
        const gate_tree = rows >= 64 and !gateTreeOff();
        // TF_DSV41_GATE_WIDE: the wide gate. Same order, same logits; eight rows and four columns a threadgroup
        // with the x and w tiles staged in threadgroup memory.
        const gate_wide = gate_tree and gateWideOn();
        // The wide gate -- eight rows and four columns a threadgroup with the tiles staged in threadgroup memory.
        // The order is the shipped one's (the lane's sixteen consecutive partials, the chunk order, the xor tree),
        // so the logits are bit-identical and the router's picks cannot move.
        if (!route_gemm or gate_scratch) if (gate_wide and (w.gate.dtype == .bf16 or w.gate.dtype == .f16)) {
            mhc.encodeGateTree8(encoder, &p.mhc, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .bf16_w = @intFromBool(w.gate.dtype == .bf16) });
        } else if (!route_gemm or gate_scratch) if (gate_tree and (w.gate.dtype == .bf16 or w.gate.dtype == .f16)) {
            mhc.encodeGateTree4(encoder, &p.mhc, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .bf16_w = @intFromBool(w.gate.dtype == .bf16) });
        } else if (!route_gemm or gate_scratch) if (!route_tensor and (w.gate.dtype == .bf16 or w.gate.dtype == .f16)) {
            mhc.encodeColDense(encoder, &p.mhc, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .bf16_w = @intFromBool(w.gate.dtype == .bf16) });
        } else ops.encodeDense(encoder, &p.ops, ws.view(.normalized), tensor(w.gate), logits_dest, .{ .rows = rows, .k = 5120, .n = experts, .x_stride = 5120, .w_stride = 5120, .y_stride = experts, .w_dtype = if (w.gate.dtype == .bf16) .bf16 else .f16, .y_dtype = .f32 });
        encoder.barrier();
        // TF_DSV41_GATE_DUMP=<dir>: the gate stage's own input and the logits it produced, copied out of the
        // workspace before the route kernel reads them. The copy changes no value the engine computes.
        if (std.c.getenv("TF_DSV41_GATE_DUMP")) |dump_dir| {
            const want: u32 = if (std.c.getenv("TF_DSV41_GATE_DUMP_ROWS")) |v|
                (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 512)
            else
                512;
            const n = @min(rows, want);
            const x_elems: u32 = n * 5120;
            const l_elems: u32 = n * 384;
            if (m.gate_dump_x == null) {
                m.gate_dump_x = try m.device.buffer(@as(usize, want) * 5120 * 4, mtl.ResourceOptions.shared);
                m.gate_dump_l = try m.device.buffer(@as(usize, want) * 384 * 4, mtl.ResourceOptions.shared);
            }
            const stg_x = m.gate_dump_x.?;
            const stg_l = m.gate_dump_l.?;
            ops.encodeDump(encoder, &p.ops, ws.view(.normalized), .{ .buffer = stg_x, .offset = 0 }, x_elems, true);
            ops.encodeDump(encoder, &p.ops, ws.view(.router_logits), .{ .buffer = stg_l, .offset = 0 }, l_elems, false);
            try splitStage(m, enc, e, "gatedump", 0);
            encoder = enc.*;
            const x_path = try std.fmt.allocPrint(m.gpa, "{s}/L{d:0>2}-r{d}-x.bin", .{ dump_dir, layer_index, n });
            defer m.gpa.free(x_path);
            {
                var f = try std.Io.Dir.cwd().createFile(m.io, x_path, .{});
                defer f.close(m.io);
                try f.writeStreamingAll(m.io, stg_x.contents()[0 .. @as(usize, x_elems) * 4]);
            }
            const l_path = try std.fmt.allocPrint(m.gpa, "{s}/L{d:0>2}-r{d}-logits.bin", .{ dump_dir, layer_index, n });
            defer m.gpa.free(l_path);
            {
                var f = try std.Io.Dir.cwd().createFile(m.io, l_path, .{});
                defer f.close(m.io);
                try f.writeStreamingAll(m.io, stg_l.contents()[0 .. @as(usize, l_elems) * 4]);
            }
        }
        const skip_experts = std.c.getenv("TF_DSV41_SKIP_EXPERTS") != null;
        // TF_DSV41_GROUP_MIN_ROWS lowers the threshold below a prompt chunk's eight rows so windows can use it.
        const group_min: u32 = if (std.c.getenv("TF_DSV41_GROUP_MIN_ROWS")) |v|
            (std.fmt.parseInt(u32, std.mem.span(v), 10) catch 8)
        else
            8;
        const grouped: ?expert_groups.Scratch = if (m.grouped_prefill and rows >= group_min and rows >= 2) try ws.grouping(rows, picks, experts) else null;
        // The fused gate+up call: one indexed dispatch over a table of 2E descriptors -- a token's six gate picks
        // then its six up picks one table entry further -- so the layer's two expert dispatches become one and
        // every output row keeps the shipped per-row arithmetic, split order and rounding. The fused call composes
        // with the grouped plan: the fused pick table gets its own twelve-slot plan, so a window's gate+up dispatch
        // dedups experts AND pays one launch.
        const fuse = w.fused != null and rows < 8 and std.c.getenv("TF_DSV41_NO_GATE_UP_FUSION") == null;
        const fused_grouped: ?expert_groups.Scratch = if (fuse and grouped != null) try ws.groupingFused(rows, picks, experts) else null;
        // The merged MoE up dispatch is possible when the fused table carries the two shared descriptors and the
        // call is the ungrouped (window) path; the route kernel writes the tail of the pick maps itself, so no host
        // write touches a GPU-written buffer range.
        const moe_up_merged = fuse and fused_grouped == null and
            w.fused.?.length() / @sizeOf(exl3.ExpertDescriptor) == 2 * experts + 2 and
            std.c.getenv("TF_DSV41_NO_FUSE_MOE_UP") == null;
        // THE SHARED EXPERT'S SLOTS OUT OF THE FUSED INDEXED CALL. The fused gate/up dispatch's tail
        // (fids[2*r*picks .. +2*r]) names each shared matrix once a token row, so the indexed body decodes the same
        // shared matrix once a row. The split runs the routed-only indexed call and dispatches the two shared
        // matrices as dense multi-row calls, which decode each matrix ONCE for the window. TF_DSV41_SHARED_DENSE=1
        // arms it; unset is the shipped call.
        const shared_split = moe_up_merged and rows >= 2 and std.c.getenv("TF_DSV41_SHARED_DENSE") != null;
        // TF_DSV41_ROUTE_GEMM_F32: the GEMM's bf16 logits decoded into a private f32 buffer, so the route kernel
        // takes its shipped path — the test that separates a bad read from a bad write.
        var logits_view = logits_dest;
        var logit_dtype: u32 = 0;
        if (gate_private) {
            logits_view = .{ .buffer = gate_pdest.?, .offset = 0 };
        }
        if (route_gemm and false) {
            const n = @as(u32, rows) * 384;
            const f32buf = try m.device.buffer(@as(usize, n) * 4, mtl.ResourceOptions.shared);
            ops.encodeDump(encoder, &p.ops, ws.view(.router_logits), .{ .buffer = f32buf, .offset = 0 }, n, true);
            encoder.barrier();
            logits_view = .{ .buffer = f32buf, .offset = 0 };
            logit_dtype = 0;
        }
        // The MoE's parts -- the gate, the route, the gate/up, the SwiGLU, the down and the shared expert, each in
        // its own command buffer.
        if (prof) |pf| {
            if (pf.blk_on) {
                pf.split("blk", "moe/gate", pf.block, enc);
                encoder = enc.*;
            }
        }
        route.encode(encoder, &p.route, logits_view, tensor(w.bias), ws.view(.route_ids), ws.view(.route_weights), .{ .ids = ws.view(.fused_ids), .sources = ws.view(.fused_sources) }, .{ .logit_dtype = logit_dtype, .rows = rows, .bias_dtype = switch (w.bias.dtype) {
            .f32 => .f32,
            .f16 => .f16,
            .bf16 => .bf16,
            else => .f16,
        }, .fused = if (fuse) experts else 0, .logit_stride = experts, .output_stride = picks, .experts = experts, .topk = picks, .moe_up = @intFromBool(moe_up_merged and !shared_split) });
        try splitStage(m, enc, e, "route", 0);
        encoder = enc.*;
        if (prof) |pf| {
            if (pf.blk_on) {
                pf.split("blk", "moe/route", pf.block, enc);
                encoder = enc.*;
            }
        }

        encoder.barrier();
        // TF_DSV41_ROUTE_DEBUG3: the GPU-side read-back (the buffers are not host-readable): the gate copy's first
        // 40 elements and the logits' first 40, into shared staging, printed once at layer 0.
        if (std.c.getenv("TF_DSV41_ROUTE_DEBUG3") != null and !gate_dump_done) {
            gate_dump_done = true;
            const stg_gate = try m.device.buffer(40 * 4, mtl.ResourceOptions.shared);
            const stg_logits = try m.device.buffer(1200 * 4, mtl.ResourceOptions.shared);
            if (w.gate_bf16) |g| ops.encodeDump(encoder, &p.ops, .{ .buffer = g.buffer, .offset = g.offset }, .{ .buffer = stg_gate, .offset = 0 }, 40, true);
            // The last row's logits: the reply's first token comes from them, and row 0 alone cannot rule coverage out.
            const last_off: usize = @as(usize, rows - 1) * 384 * 4;
            ops.encodeDump(encoder, &p.ops, .{ .buffer = ws.view(.router_logits).buffer, .offset = ws.view(.router_logits).offset + last_off }, .{ .buffer = stg_logits, .offset = 0 }, 384, false);
            try splitStage(m, enc, e, "dump", 0);
            encoder = enc.*;
            const g40: [*]const f32 = @ptrCast(@alignCast(stg_gate.contents()));
            const l40: [*]const f32 = @ptrCast(@alignCast(stg_logits.contents()));
            std.log.info("dump gate[0..3] {d:.4} {d:.4} {d:.4} {d:.4}", .{ g40[0], g40[1], g40[2], g40[3] });
            var best: [3]f32 = .{ -1e30, -1e30, -1e30 };
            var best_id: [3]u32 = .{ 0, 0, 0 };
            for (0..384) |col| {
                const v = l40[col];
                if (v > best[0]) {
                    best[2] = best[1];
                    best_id[2] = best_id[1];
                    best[1] = best[0];
                    best_id[1] = best_id[0];
                    best[0] = v;
                    best_id[0] = @intCast(col);
                } else if (v > best[1]) {
                    best[2] = best[1];
                    best_id[2] = best_id[1];
                    best[1] = v;
                    best_id[1] = @intCast(col);
                } else if (v > best[2]) {
                    best[2] = v;
                    best_id[2] = @intCast(col);
                }
            }
            std.log.info("dump row0 top3: {d}@{d:.4} {d}@{d:.4} {d}@{d:.4}", .{ best_id[0], best[0], best_id[1], best[1], best_id[2], best[2] });
        }
        // The gate's first eight logits of row 0, read both ways: the one look that tells a layout fault from a
        // scale fault (TF_DSV41_ROUTE_DEBUG2).
        if (std.c.getenv("TF_DSV41_ROUTE_DEBUG2") != null and !gate_dump_done) {
            gate_dump_done = true;
            try splitStage(m, enc, e, "gatedump", 0);
            encoder = enc.*;
            const v = ws.view(.router_logits);
            const base = v.buffer.contents() + v.offset;
            const p32: [*]const f32 = @ptrCast(@alignCast(base));
            const p16: [*]const u16 = @ptrCast(@alignCast(base));
            var a32: [8]f32 = undefined;
            var a16: [8]f32 = undefined;
            for (0..8) |i| {
                a32[i] = p32[i];
                a16[i] = @bitCast(@as(u32, p16[i]) << 16);
            }
            std.log.info("gate logits row0 as f32: {d:.4} {d:.4} {d:.4} {d:.4}", .{ a32[0], a32[1], a32[2], a32[3] });
            std.log.info("gate logits row0 as bf16: {d:.4} {d:.4} {d:.4} {d:.4}", .{ a16[0], a16[1], a16[2], a16[3] });
        }
        // The last row's six router weights and six picks: the only router output never compared.
        if (std.c.getenv("TF_DSV41_ROUTE_DEBUG4") != null and !gate_dump_done2) {
            gate_dump_done2 = true;
            const stg_wt = try m.device.buffer(6 * 4, mtl.ResourceOptions.shared);
            const stg_id = try m.device.buffer(6 * 4, mtl.ResourceOptions.shared);
            const rw = ws.view(.route_weights);
            const ri = ws.view(.route_ids);
            const off_w: usize = @as(usize, rows - 1) * 6 * 4;
            ops.encodeDump(encoder, &p.ops, .{ .buffer = rw.buffer, .offset = rw.offset + off_w }, .{ .buffer = stg_wt, .offset = 0 }, 6, false);
            ops.encodeDump(encoder, &p.ops, .{ .buffer = ri.buffer, .offset = ri.offset + off_w }, .{ .buffer = stg_id, .offset = 0 }, 6, false);
            try splitStage(m, enc, e, "picks", 0);
            encoder = enc.*;
            const w6: [*]const f32 = @ptrCast(@alignCast(stg_wt.contents()));
            const id6: [*]const f32 = @ptrCast(@alignCast(stg_id.contents()));
            std.log.info("picks ids {d} {d} {d} {d} {d} {d}", .{ @as(u32, @bitCast(id6[0])), @as(u32, @bitCast(id6[1])), @as(u32, @bitCast(id6[2])), @as(u32, @bitCast(id6[3])), @as(u32, @bitCast(id6[4])), @as(u32, @bitCast(id6[5])) });
            std.log.info("picks weights {d:.4} {d:.4} {d:.4} {d:.4} {d:.4} {d:.4}", .{ w6[0], w6[1], w6[2], w6[3], w6[4], w6[5] });
        }
        // (inert unless TF_DSV41_ROUTE_IDS=1): the last rows' route ids, read back from the GPU. The drafted path's
        // second row picks its own six experts, so the overlap between a row's picks and the next row's picks is
        // the "localisation" the record wants: the two-row window's twelve slots against the distinct experts. The
        // record has it for the four-row window (seventeen distinct of twenty-four) and for five rows, never for
        // the standing two-row window the NOVEL rate runs on.
        const want_ids = std.c.getenv("TF_DSV41_ROUTE_IDS") != null and
            ((rows == 2 and !route_ids_done_pair) or (rows > 8 and !route_ids_done_prompt));
        if (want_ids) {
            if (rows == 2) route_ids_done_pair = true else route_ids_done_prompt = true;
            const nrows: usize = @min(rows, 2);
            const n: u32 = @intCast(nrows * picks);
            const stg_ids = try m.device.buffer(@as(usize, n) * 4, mtl.ResourceOptions.shared);
            const ri = ws.view(.route_ids);
            const off_i: usize = (@as(usize, rows) - nrows) * picks * 4;
            ops.encodeDump(encoder, &p.ops, .{ .buffer = ri.buffer, .offset = ri.offset + off_i }, .{ .buffer = stg_ids, .offset = 0 }, n, false);
            try splitStage(m, enc, e, "ids", 0);
            encoder = enc.*;
            const ip: [*]const u32 = @ptrCast(@alignCast(stg_ids.contents()));
            var distinct: u32 = 0;
            var buf: [256]u32 = undefined;
            for (0..n) |i| {
                const iu: usize = @intCast(i);
                buf[iu] = ip[i];
                var seen = false;
                for (0..iu) |j| {
                    if (buf[j] == ip[i]) seen = true;
                }
                if (!seen) distinct += 1;
            }
            std.log.info("route ids: rows {d} slots {d} distinct {d}", .{ nrows, n, distinct });
            var line: [2048]u8 = undefined;
            var at: usize = 0;
            for (0..n) |i| {
                const s = std.fmt.bufPrint(line[at..], "{d} ", .{ip[i]}) catch break;
                at += s.len;
            }
            std.log.info("route ids: {s}", .{line[0..at]});
        }
        var routed_rows = exl3.Rows{ .count = @as(usize, rows) * picks, .x_stride = 5120, .y_stride = 2304, .input_rows = rows, .output_rows = @as(usize, rows) * picks, .tile_rows = moeTileRows(), .routes = .{ .ids = convert(ws.view(.route_ids)), .repeat = picks } };
        if (grouped) |g| {
            try expert_groups.encode(encoder, &p.groups, convert(ws.view(.route_ids)), g, picks, 0);
            routed_rows.count = g.plan.padded;
            routed_rows.input_ids = g.view(.token_ids);
            routed_rows.output_ids = g.view(.slot_ids);
            routed_rows.routes = .{ .ids = g.view(.expert_ids) };
            routed_rows.grouped = true;
            // (inert unless TF_DSV41_ROUTE_HIST=1): the grouped plan's own geometry, read back from the GPU -- the
            // per-expert counts and the per-expert block offsets. The host's plan sizes the row buffer with a
            // formula (experts + (rows - experts) / tile tiles); the GPU's grouping pads EACH expert's block to a
            // multiple of the tile and lays the blocks out consecutively, so the tiles that do work are the sum of
            // the per-expert ceilings, and the plan's tail is unused. The weight decode is per tile, so the plan's
            // padding is a direct multiplier on the routed stage's biggest term.
            if (std.c.getenv("TF_DSV41_ROUTE_HIST") != null and !plan_hist_done) {
                plan_hist_done = true;
                const stg_cnt = try m.device.buffer(384 * 4, mtl.ResourceOptions.shared);
                const stg_off = try m.device.buffer(384 * 4, mtl.ResourceOptions.shared);
                ops.encodeDump(encoder, &p.ops, .{ .buffer = g.view(.counts).buffer, .offset = g.view(.counts).offset }, .{ .buffer = stg_cnt, .offset = 0 }, 384, false);
                ops.encodeDump(encoder, &p.ops, .{ .buffer = g.view(.offsets).buffer, .offset = g.view(.offsets).offset }, .{ .buffer = stg_off, .offset = 0 }, 384, false);
                try splitStage(m, enc, e, "plan", 0);
                encoder = enc.*;
                const cp: [*]const u32 = @ptrCast(@alignCast(stg_cnt.contents()));
                const op: [*]const u32 = @ptrCast(@alignCast(stg_off.contents()));
                var live: u64 = 0;
                var tiles: u64 = 0;
                var used: u64 = 0;
                var nz: u32 = 0;
                var mx: u32 = 0;
                var hist: [10]u32 = [_]u32{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
                for (0..384) |ei| {
                    const c = cp[ei];
                    live += @as(u64, c);
                    if (c > 0) {
                        nz += 1;
                        if (c > mx) mx = c;
                        const t = (c + 63) / 64;
                        tiles += @as(u64, t);
                        used += @as(u64, t) * 64;
                        hist[@min(t, 9)] += 1;
                    }
                }
                std.log.info("plan: rows {d} live {d} experts_used {d} max_count {d} tiles_used {d} used_rows {d} plan_padded {d}", .{ rows * picks, live, nz, mx, tiles, used, g.plan.padded });
                std.log.info("plan tiles a expert: 1:{d} 2:{d} 3:{d} 4:{d} 5:{d} 6:{d} 7:{d} 8:{d} 9+:{d}", .{ hist[1], hist[2], hist[3], hist[4], hist[5], hist[6], hist[7], hist[8], hist[9] });
                var line: [4096]u8 = undefined;
                var at: usize = 0;
                for (0..384) |ei| {
                    const s = std.fmt.bufPrint(line[at..], "{d},", .{cp[ei]}) catch break;
                    at += s.len;
                }
                std.log.info("plan counts: {s}", .{line[0..at]});
                var line2: [4096]u8 = undefined;
                var at2: usize = 0;
                for (0..384) |ei| {
                    const s = std.fmt.bufPrint(line2[at2..], "{d},", .{op[ei]}) catch break;
                    at2 += s.len;
                }
                std.log.info("plan offsets: {s}", .{line2[0..at2]});
            }
        }
        // The fused table is two blocks of six picks a token, so its sort reads a slot's block for the token.
        if (fused_grouped) |g| try expert_groups.encode(encoder, &p.groups, convert(ws.view(.fused_ids)), g, picks, picks * rows);
        if (fuse) {
            var both = routed_rows;
            if (fused_grouped) |g| {
                both.count = g.plan.padded;
                both.input_ids = g.view(.token_ids);
                both.output_ids = g.view(.slot_ids);
                both.routes = .{ .ids = g.view(.expert_ids) };
                both.grouped = true;
            } else {
                const r = @as(usize, rows);
                const routed_count = r * 2 * picks;
                const merge_up = moe_up_merged;
                both.count = routed_count + if (merge_up and !shared_split) 2 * r else 0;
                both.input_rows = r;
                both.input_ids = convert(ws.view(.fused_sources));
                both.routes = .{ .ids = convert(ws.view(.fused_ids)), .repeat = 1 };
                both.output_rows = both.count;
                // The shared expert's tail (fids[2*r*picks .. +2r], route.metal) names each of the two shared
                // matrices r times, so with two or more token rows the routes carry the duplicates the paired body
                // shares a decode by. One row has none (the plain path).
                both.pair = r >= 2;
            }
            if (fused_grouped != null) both.output_rows = @as(usize, rows) * 2 * picks;
            const fused_passes: u32 = expertPasses();
            const fused_experts: u32 = 2 * experts + 2 * @as(u32, @intFromBool(moe_up_merged));
            for (0..fused_passes) |_| if (!skip_experts) try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = w.fused.? }, .count = fused_experts, .k = 5120, .n = 2304 }, convert(ws.view(.normalized)), convert(ws.view(.gate)), ws.projection(), both);
            try splitStage(m, enc, e, "gateup", 0);
            encoder = enc.*;
            if (prof) |pf| {
                if (pf.blk_on) {
                    pf.split("blk", "moe/gateup", pf.block, enc);
                    encoder = enc.*;
                }
            }

            encoder.barrier();
        } else for (0..2) |i| {
            if (!skip_experts) try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = w.descriptors[i] }, .count = experts, .k = 5120, .n = 2304 }, convert(ws.view(.normalized)), convert(ws.view(if (i == 0) .gate else .up)), ws.projection(), routed_rows);
            encoder.barrier();
        }
        const mlp_passes: u32 = if (std.c.getenv("TF_DSV41_DOUBLE_MLP") != null) 2 else 1;
        // THE FUSED CALL'S UP HALF IS AT THE BLOCK'S OWN PICK COUNT. The fused call writes every token's gate picks
        // then every token's up picks, so the up half starts at row `rows * picks` -- the backbone's six or the
        // DRAFTER'S THREE. This landing keeps the fusion and fixes the offset; the backbone's six picks make the
        // offset identical, so the trunk's canon does not move.
        const fuse_up_picks: u32 = if (std.c.getenv("TF_DSV41_FUSE_UP_SIX") != null) 6 else picks;
        const up_view = if (fuse) View{ .buffer = ws.view(.gate).buffer, .offset = ws.view(.gate).offset + @as(usize, rows) * fuse_up_picks * 2304 * 2 } else ws.view(.up);
        // The routed SwiGLU writes the down projection's rotated input in the same pass (the same su scale and
        // rotation as the separate rotate_in), inside the grouped plan's destination slot, and the down call below
        // skips that pass. Exactness is the two-pass path's bits (GPU differential green, canons and band held);
        // TF_DSV41_NO_SWIGLU_ROT restores the two-pass arm for A/B.
        const swiglu_rot = std.c.getenv("TF_DSV41_NO_SWIGLU_ROT") == null and std.c.getenv("TF_DSV41_SKIP_SWIGLU") == null;
        if (swiglu_rot)
            exl3.encodeSwiGLUPreRotate(&p.projection, encoder, convert(ws.view(.gate)), convert(up_view), convert(ws.view(.route_weights)), convert(ws.view(.route_ids)), if (grouped) |g| g.view(.destination) else null, convert(.{ .buffer = w.descriptors[2] }), ws.projection(), .{ .rows = @as(usize, rows) * picks, .k = 2304, .input_stride = 2304, .output_stride = 2304, .experts = experts })
        else for (0..mlp_passes) |_| if (std.c.getenv("TF_DSV41_SKIP_SWIGLU") == null) ops.encodeSwiGLU(encoder, &p.ops, ws.view(.gate), up_view, ws.view(.route_weights), ws.view(.activation), .{ .rows = rows * picks, .width = 2304, .input_stride = 2304, .output_stride = 2304, .weighted = true });
        try splitStage(m, enc, e, "swiglu", 0);
        encoder = enc.*;
        if (prof) |pf| {
            if (pf.blk_on) {
                pf.split("blk", "moe/swiglu", pf.block, enc);
                encoder = enc.*;
            }
        }

        encoder.barrier();
        routed_rows.x_stride = 2304;
        routed_rows.y_stride = 5120;
        routed_rows.routes.?.repeat = 1;
        routed_rows.input_rows = @as(usize, rows) * picks; // The block's own pick count, not the backbone's six
        if (grouped) |g| routed_rows.input_ids = g.view(.slot_ids);
        const down_passes: u32 = expertPasses();
        routed_rows.prerotated = swiglu_rot;
        const down_input: exl3.View = if (swiglu_rot) ws.projection() else convert(ws.view(.activation));
        for (0..down_passes) |_| if (!skip_experts) try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = w.descriptors[2] }, .count = experts, .k = 2304, .n = 5120 }, down_input, convert(ws.view(.routed)), ws.projection(), routed_rows);
        try splitStage(m, enc, e, "down", 0);
        encoder = enc.*;
        if (prof) |pf| {
            if (pf.blk_on) {
                pf.split("blk", "moe/down", pf.block, enc);
                encoder = enc.*;
            }
        }

        encoder.barrier();
        const skip_shared = std.c.getenv("TF_DSV41_SKIP_SHARED") != null;
        if (!skip_shared) {
            // The shared expert's gate and up in one indexed dispatch. The two matrices share the input row, so the
            // gather map repeats it (`r % rows`) and the route map picks descriptor 0 (w1) for the first `rows`
            // output rows and 1 (w3) for the next; the output is gate rows then up rows in the `up` region, which
            // the routed SwiGLU has already consumed. The per-row body is the register-only static body the routed
            // decode runs, and the dense one-row path runs its Dense twin, so the arithmetic per row is the same.
            const shared_fuse = rows <= 8 and w.shared_fused != null and std.c.getenv("TF_DSV41_NO_FUSE_SHARED") == null;
            // The shared SwiGLU writes the down projection's ROTATED input in the same pass (the same arithmetic as
            // the routed path's encodeSwiGLUPreRotate, The GPU differential), so the shared down skips its own
            // rotate_in dispatch. The dense down keeps its body and k order -- same values, same bits.
            // TF_DSV41_NO_FUSE_SHARED_ROT=1 restores the two-pass arm.
            const shared_rot = moe_up_merged and w.shared_down != null and w.shared_down_ids != null and
                std.c.getenv("TF_DSV41_NO_FUSE_SHARED_ROT") == null;
            for (0..mlp_passes) |_| {
                if (moe_up_merged) {
                    // The shared gate/up already landed in the merged dispatch: rows [12r, 13r) are the shared
                    // gate and [13r, 14r) the shared up, right after the routed gate/up at [0, 12r).
                    const gv = ws.view(.gate);
                    const sg = View{ .buffer = gv.buffer, .offset = gv.offset + @as(usize, rows) * 2 * picks * 2304 * 2 };
                    const su = View{ .buffer = gv.buffer, .offset = sg.offset + @as(usize, rows) * 2304 * 2 };
                    if (shared_split) {
                        if (!shared_split_logged) {
                            shared_split_logged = true;
                            std.log.info("shared split: dense multi-row shared gate/up, rows {d}", .{rows});
                        }
                        try m.project(encoder, w.shared[0], ws.view(.normalized), sg, rows, 5120, 2304, .bf16);
                        try m.project(encoder, w.shared[1], ws.view(.normalized), su, rows, 5120, 2304, .bf16);
                        encoder.barrier();
                    }
                    if (shared_rot) {
                        exl3.encodeSwiGLUPreRotate(&p.projection, encoder, convert(sg), convert(su), convert(ws.view(.route_weights)), convert(View{ .buffer = w.shared_down_ids.?, .offset = 0 }), null, convert(View{ .buffer = w.shared_down.?, .offset = 0 }), ws.projection(), .{ .rows = rows, .k = 2304, .input_stride = 2304, .output_stride = 2304, .weighted = false, .experts = 1 });
                    } else ops.encodeSwiGLU(encoder, &p.ops, sg, su, ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
                } else if (shared_fuse) {
                    try exl3.encodeIndexed(&p.projection, encoder, .{ .descriptors = .{ .buffer = w.shared_fused.?, .offset = 0 }, .count = 2, .k = 5120, .n = 2304 }, convert(ws.view(.normalized)), convert(ws.view(.up)), ws.projection(), .{
                        .count = @as(usize, rows) * 2,
                        .x_stride = 5120,
                        .y_stride = 2304,
                        .output = .bf16,
                        .input_ids = convert(ws.sharedMapInputs(rows)),
                        .input_rows = rows,
                        .output_rows = @as(usize, rows) * 2,
                        .routes = .{ .ids = convert(ws.sharedMapRoutes(rows)) },
                    });
                    const fused = ws.view(.up);
                    const up_v = View{ .buffer = fused.buffer, .offset = fused.offset + @as(usize, rows) * 2304 * 2 };
                    ops.encodeSwiGLU(encoder, &p.ops, fused, up_v, ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
                } else {
                    try m.project(encoder, w.shared[0], ws.view(.normalized), ws.view(.shared_gate), rows, 5120, 2304, .bf16);
                    try m.project(encoder, w.shared[1], ws.view(.normalized), ws.view(.shared_up), rows, 5120, 2304, .bf16);
                    ops.encodeSwiGLU(encoder, &p.ops, ws.view(.shared_gate), ws.view(.shared_up), ws.view(.route_weights), ws.view(.shared_activation), .{ .rows = rows, .width = 2304, .input_stride = 2304, .output_stride = 2304 });
                }
                encoder.barrier();
                if (shared_rot) {
                    try exl3.encode(&p.projection, encoder, w.shared[2], ws.projection(), convert(ws.view(.shared_down)), ws.projection(), .{ .count = rows, .x_stride = 2304, .y_stride = 5120, .prerotated = true });
                } else try m.project(encoder, w.shared[2], ws.view(.shared_activation), ws.view(.shared_down), rows, 2304, 5120, .bf16);
            }
        }
        try splitStage(m, enc, e, "shared", 0);
        encoder = enc.*;
        if (prof) |pf| {
            if (pf.blk_on) {
                pf.split("blk", "moe/shared", pf.block, enc);
                encoder = enc.*;
            }
        }

        if (std.c.getenv("TF_DSV41_SKIP_COMBINE") == null) ops.encodeCombine(encoder, &p.ops, ws.view(.routed), ws.view(.shared_down), ws.view(.route_ids), ws.view(.branch), .{ .rows = rows, .width = 5120, .routed_row_stride = picks * 5120, .slot_stride = 5120, .shared_stride = 5120, .y_stride = 5120, .id_stride = picks });
        // A second identical pass leaves the value unchanged and prices the kernel in place.
        if (std.c.getenv("TF_DSV41_DOUBLE_COMBINE") != null) ops.encodeCombine(encoder, &p.ops, ws.view(.routed), ws.view(.shared_down), ws.view(.route_ids), ws.view(.branch), .{ .rows = rows, .width = 5120, .routed_row_stride = picks * 5120, .slot_stride = 5120, .shared_stride = 5120, .y_stride = 5120, .id_stride = picks });
        encoder.barrier();
        if (prof) |pf| {
            if (pf.blk_on) {
                pf.split("blk", "moe/combine", pf.block, enc);
                encoder = enc.*;
            }
        }
    }
};

fn tensor(t: checkpoint.Tensor) View {
    return .{ .buffer = t.buffer, .offset = t.offset };
}
fn asF32(x: u16) f32 {
    return @bitCast(@as(u32, x) << 16);
}

fn convert(v: View) exl3.View {
    return .{ .buffer = v.buffer, .offset = v.offset };
}

/// TF_DSV41_STAGE_CB: close the command buffer at a stage boundary, log its GPU seconds under `label`, and start
/// the next one.
var split_stage_dispatches: u64 = 0;
var split_stage_threadgroups: u64 = 0;
var gate_dump_done: bool = false;
var route_ids_done_pair: bool = false;
var route_ids_done_prompt: bool = false;
var plan_hist_done: bool = false;
var shared_split_logged: bool = false;
var gate_dump_done2: bool = false;
/// The tree-price instrument's round join (verify round R <-> proposal round R).
pub var price_round: u64 = 0;
var gate_dump_done3: bool = false;
fn splitStage(m: *Model, enc: *mtl.ComputeEncoder, e: ?*mtl.CommandBuffer, label: []const u8, index: usize) !void {
    if (std.c.getenv("TF_DSV41_STAGE_CB") == null) return;
    if (e == null) return;
    mtl.command.count_enabled = true;
    const dispatches = mtl.command.dispatch_count.load(.monotonic);
    enc.end();
    e.?.commit();
    e.?.wait();
    try completed(e.?.*);
    // The threadgroup count beside the dispatch count -- the parallelism ledger.
    const groups = mtl.command.dispatch_threadgroups.load(.monotonic);
    std.log.info("stage {s} layer {d}: gpu {d:7.2} ms, dispatches {d}, threadgroups {d}", .{ label, index, e.?.gpuSeconds() * 1e3, dispatches - split_stage_dispatches, groups - split_stage_threadgroups });
    split_stage_dispatches = dispatches;
    split_stage_threadgroups = groups;
    e.?.* = m.queue.commandBufferUnretained();
    enc.* = e.?.compute(.serial);
    for (m.checkpoint.shards.items) |shard| enc.useResource(shard.buffer, mtl.ResourceUsage.read);
}

fn completed(cb: mtl.CommandBuffer) !void {
    if (cb.failure()) |message| {
        std.log.err("DeepSeek V4.1 GPU execution failed: {s}", .{message});
        return error.GpuExecutionFailed;
    }
}
