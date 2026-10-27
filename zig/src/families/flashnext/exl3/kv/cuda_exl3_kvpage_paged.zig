//! One sequence's paged K/V: the host store, the device window and block cache, and the attention over them.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("../../cuda_prompt_wt.zig");
const parts = @import("../../cuda_batch_state.zig");
const telemetry = @import("../model/exl3_telemetry.zig");
const shared_contract = @import("../offload/exl3_shared_kv.zig");
const shared_device = @import("../offload/cuda_exl3_shared_kv.zig");
const hub = @import("cuda_exl3_kvpage.zig");
const configs = @import("cuda_exl3_kvpage_config.zig");
const snaps = @import("cuda_exl3_kvpage_snapshot.zig");

const head_dim = wt.head_dim;
const kv_heads = wt.kv_heads;
const ratio = wt.index_ratio;
const top = wt.index_top;
const idw = wt.index_idw;
const att_rows = parts.att_rows;
const attn_n = wt.attn_n;
const ctl_n: usize = 8;
const work_n: usize = att_rows * top;

const attnRows = hub.attnRows;
const select = hub.select;
const Config = configs.Config;
const cdiv = configs.cdiv;
const cint = configs.cint;
const pagingRows = configs.pagingRows;
const row_bytes = configs.row_bytes;
const windowPlan = configs.windowPlan;
const Snapshot = snaps.Snapshot;
const SnapshotLayout = snaps.SnapshotLayout;
const Stats = snaps.Stats;

const Kern = struct {
    module: cuda.Module,
    plan: cuda.Function,
    fetch: cuda.Function,
    remap: cuda.Function,

    fn load(d: *const cuda.Driver) !Kern {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        var k: Kern = undefined;
        k.module = try cuda.Module.load(d, cuda.kernels.flashnext_exl3_kvpage);
        errdefer k.module.unload();
        k.plan = try k.module.function("kvp_plan");
        k.fetch = try k.module.function("kvp_fetch");
        k.remap = try k.module.function("kvp_remap");
        return k;
    }
};

/// One sequence's K/V of the 12 attention layers: all of it in mapped pinned host memory, [layer][K|V][cap][2][256].
pub const Paged = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    k: Kern,
    cfg: Config,
    cap: usize,
    /// Window rows: the kept window, one call's rows, and a block's slack.
    wr: usize,
    nblocks: usize,
    host: cuda.HostBuffer,
    host_dev: u64,
    status_host: cuda.HostBuffer,
    dev: cuda.DeviceBuffer,
    /// blk2slot [12][nblocks] i32, mark [12][nblocks] u32, ctl [12][8] i32, slot2blk [slots] i32, stamp [slots] u32.
    tabs: cuda.DeviceBuffer,
    /// need, miss [att_rows * 512] i32 and fetch [2 * att_rows * 512] i32 (one call's, shared by the layers).
    work: cuda.DeviceBuffer,
    win0: [attn_n]usize = @splat(0),
    rebases: [attn_n]u64 = @splat(0),
    written: [attn_n]u64 = @splat(0),
    telemetry_id: u64 = 0,
    telemetry_revision: u64 = 0,
    shared_arena: ?*shared_device.Arena = null,
    shared_sequence: shared_contract.Sequence = undefined,
    shared_attached: bool = true,
    incarnation: u64 = 0,
    retired_for_reuse: bool = false,

    pub fn init(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, cfg: Config) !*Paged {
        return initStore(gpa, d, cap, cfg, null);
    }

    pub fn initShared(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, cfg: Config, arena: *shared_device.Arena) !*Paged {
        return initStore(gpa, d, cap, cfg, arena);
    }

    fn initStore(gpa: std.mem.Allocator, d: *const cuda.Driver, cap: usize, cfg: Config, shared: ?*shared_device.Arena) !*Paged {
        const layout = try SnapshotLayout.init(cap, cfg, 0);
        if ((cfg.shared_lanes != 0) != (shared != null)) return error.SharedKvArenaRequired;
        if (shared) |arena| {
            const options = arena.layout.options;
            if (cap != options.cap) return error.SharedKvCommonCap;
            if (d != arena.driver or cfg.shared_lanes != options.lanes or cfg.window != options.window or cfg.rows != options.rows or cfg.slots != options.slots) return error.SharedKvGeometry;
        }
        const pg = try gpa.create(Paged);
        errdefer gpa.destroy(pg);
        pg.* = .{ .gpa = gpa, .d = d, .k = undefined, .cfg = cfg, .cap = cap, .wr = layout.wr, .nblocks = layout.nblocks, .host = undefined, .host_dev = 0, .status_host = undefined, .dev = undefined, .tabs = undefined, .work = undefined };
        pg.telemetry_id = try telemetry.identity();
        pg.incarnation = try shared_contract.identity();
        pg.shared_arena = shared;
        if (shared) |arena| pg.shared_sequence = try arena.register(cap) else pg.k = try Kern.load(d);
        errdefer if (shared) |arena| arena.release(pg.shared_sequence, .{ .d = d, .handle = null }) catch unreachable else pg.k.module.unload();
        pg.host = try cuda.HostBuffer.allocMapped(d, attn_n * 2 * cap * row_bytes);
        errdefer pg.host.free();
        pg.host_dev = try pg.host.device();
        pg.status_host = try cuda.HostBuffer.alloc(d, attn_n * ctl_n * 4);
        errdefer pg.status_host.free();
        pg.dev = if (shared) |arena| arena.device else try cuda.DeviceBuffer.alloc(d, layout.device);
        if (shared != null) pg.dev.borrowed = true;
        errdefer pg.dev.free();
        pg.tabs = if (shared) |arena| arena.tables else try cuda.DeviceBuffer.alloc(d, layout.tables);
        if (shared != null) pg.tabs.borrowed = true;
        errdefer pg.tabs.free();
        pg.work = if (shared) |arena| arena.work else try cuda.DeviceBuffer.alloc(d, work_n * 4 * 4);
        if (shared != null) pg.work.borrowed = true;
        errdefer pg.work.free();
        if (shared == null) {
            try pg.dev.fill8(0, null);
            try pg.tabs.fill8(0, null);
            try pg.resetTables(null);
        }
        return pg;
    }

    pub fn destroy(pg: *Paged) void {
        if (pg.cfg.shared_lanes != 0 and pg.shared_attached) pg.shared_arena.?.release(pg.shared_sequence, .{ .d = pg.d, .handle = null }) catch @panic("shared KV lane retirement failed");
        pg.work.free();
        pg.tabs.free();
        pg.dev.free();
        pg.status_host.free();
        pg.host.free();
        if (pg.cfg.shared_lanes == 0) pg.k.module.unload();
        pg.gpa.destroy(pg);
    }

    pub fn retireForReuse(pg: *Paged) !void {
        if (pg.retired_for_reuse) return error.KvPageRetired;
        if (pg.shared_arena) |arena| if (arena.active != null) return error.SharedKvBusy;
        try pg.d.synchronize();
        if (pg.shared_arena) |arena| {
            try arena.release(pg.shared_sequence, .{ .d = pg.d, .handle = null });
            pg.shared_attached = false;
        }
        pg.retired_for_reuse = true;
    }

    pub fn resetForReuse(pg: *Paged, stream: cuda.Stream) !void {
        if (!pg.retired_for_reuse) return error.KvPageNotRetired;
        if (stream.d != pg.d) return error.SharedKvDriver;
        const identity = try telemetry.identity();
        const incarnation = try shared_contract.identity();
        if (pg.shared_arena) |arena| {
            if (pg.shared_attached) return error.SharedKvSequenceLive;
            pg.shared_sequence = try arena.register(pg.cap);
            pg.shared_attached = true;
        } else {
            try pg.dev.fill8(0, stream.handle);
            try pg.work.fill8(0, stream.handle);
            try pg.resetTables(stream);
        }
        @memset(pg.host.bytes, 0);
        @memset(pg.status_host.bytes, 0);
        pg.win0 = @splat(0);
        pg.rebases = @splat(0);
        pg.written = @splat(0);
        pg.telemetry_id = identity;
        pg.telemetry_revision = 0;
        pg.incarnation = incarnation;
        pg.retired_for_reuse = false;
    }

    /// Device bytes of the K/V side (windows and slots).
    pub fn devBytes(pg: *const Paged) usize {
        if (pg.cfg.shared_lanes != 0) return 0;
        return 2 * pg.rowsDev() * row_bytes;
    }

    /// Device bytes in all: K/V side, tables and the call's work lists (deviceBytesFor, a host test checks).
    pub fn deviceBytes(pg: *const Paged) usize {
        if (pg.cfg.shared_lanes != 0) return 0;
        return pg.devBytes() + pg.tabWords() * 4 + work_n * 4 * 4;
    }

    /// Rows of the K (or V) buffer: the 12 windows, then the slots.
    fn rowsDev(pg: *const Paged) usize {
        if (pg.cfg.shared_lanes != 0) return pg.shared_arena.?.layout.side_rows;
        return attn_n * pg.wr + ratio * pg.cfg.slots;
    }

    fn tabWords(pg: *const Paged) usize {
        return attn_n * (2 * pg.nblocks + ctl_n) + 2 * pg.cfg.slots;
    }

    /// Layer `ai`'s window base in the K buffer (its kernels index from here).
    pub fn devK(pg: *const Paged, ai: usize) u64 {
        if (pg.cfg.shared_lanes != 0) return pg.dev.ptr + (pg.shared_arena.?.layout.windowOffset(pg.shared_sequence.lane, ai) catch unreachable) * row_bytes;
        return pg.dev.ptr + ai * pg.wr * row_bytes;
    }

    pub fn devV(pg: *const Paged, ai: usize) u64 {
        if (pg.cfg.shared_lanes != 0) return pg.devK(ai) + pg.rowsDev() * row_bytes;
        return pg.dev.ptr + (pg.rowsDev() + ai * pg.wr) * row_bytes;
    }

    /// Rows from layer `ai`'s window base to the first slot.
    fn slotBase(pg: *const Paged, ai: usize) usize {
        if (pg.cfg.shared_lanes != 0) return pg.shared_arena.?.layout.slotBase(pg.shared_sequence.lane, ai) catch unreachable;
        return (attn_n - ai) * pg.wr;
    }

    fn hostK(pg: *const Paged, ai: usize) u64 {
        return pg.host_dev + (2 * ai) * pg.cap * row_bytes;
    }

    fn hostV(pg: *const Paged, ai: usize) u64 {
        return pg.host_dev + (2 * ai + 1) * pg.cap * row_bytes;
    }

    const Tab = struct { blk2slot: u64, mark: u64, ctl: u64, slot2blk: u64, stamp: u64, blk2slot_all: u64 };

    fn tab(pg: *const Paged, ai: usize) Tab {
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            const layout = arena.layout;
            const key = @as(usize, pg.shared_sequence.lane) * layout.lane_words + ai * layout.blocks;
            return .{ .blk2slot = arena.tables.ptr + key * 4, .mark = arena.tables.ptr + layout.marks + key * 4, .ctl = arena.tables.ptr + layout.controls + (@as(usize, pg.shared_sequence.lane) * attn_n + ai) * ctl_n * 4, .slot2blk = arena.tables.ptr + layout.owners, .stamp = arena.tables.ptr + layout.stamps, .blk2slot_all = arena.tables.ptr };
        }
        const base = pg.tabs.ptr;
        const nb = pg.nblocks * 4;
        const all = base;
        const marks = base + attn_n * nb;
        const ctls = marks + attn_n * nb;
        const s2b = ctls + attn_n * ctl_n * 4;
        return .{ .blk2slot = all + ai * nb, .mark = marks + ai * nb, .ctl = ctls + ai * ctl_n * 4, .slot2blk = s2b, .stamp = s2b + pg.cfg.slots * 4, .blk2slot_all = all };
    }

    fn memset32(pg: *const Paged, ptr: u64, value: u32, words: usize, stream: ?cuda.Stream) !void {
        if (stream) |s| return pg.d.check(pg.d.api.cuMemsetD32Async(ptr, value, words, s.handle), "cuMemsetD32Async");
        try pg.d.check(pg.d.api.cuMemsetD32_v2(ptr, value, words), "cuMemsetD32");
    }

    /// An empty cache with a fresh device clock and every window at 0.
    fn resetTables(pg: *Paged, stream: ?cuda.Stream) !void {
        pg.advanceTelemetry();
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            try arena.clear(try arena.leaseFor(pg.shared_sequence), pg.shared_sequence, stream orelse .{ .d = pg.d, .handle = null });
            pg.win0 = @splat(0);
            return;
        }
        const t = pg.tab(0);
        try pg.memset32(t.blk2slot_all, 0xFFFF_FFFF, attn_n * pg.nblocks, stream);
        try pg.memset32(t.slot2blk, 0xFFFF_FFFF, pg.cfg.slots, stream);
        try pg.memset32(t.stamp, 0, pg.cfg.slots, stream);
        try pg.memset32(pg.tabs.ptr + attn_n * pg.nblocks * 4, 0, attn_n * pg.nblocks, stream);
        try pg.memset32(t.ctl, 0, attn_n * ctl_n, stream);
        pg.win0 = @splat(0);
    }

    pub fn copy(pg: *const Paged, stream: cuda.Stream, dst: u64, src: u64, n: usize) !void {
        if (n == 0) return;
        try pg.d.check(pg.d.api.cuMemcpyDtoDAsync_v2(dst, src, n, stream.handle), "cuMemcpyDtoDAsync");
    }

    /// Moves the window so rows [pos0, pos0 + n) fit: the rows that leave it are written out to host memory.
    fn rebase(pg: *Paged, ai: usize, pos0: usize, stream: cuda.Stream) !void {
        const w0 = pg.win0[ai];
        const new0 = (pos0 -| pg.cfg.window) / ratio * ratio;
        if (new0 <= w0) return error.KvPageWindow;
        const shift = new0 - w0;
        if (pos0 < w0 or pos0 > w0 + pg.wr or shift > pg.wr) return error.KvPageWindow;
        for ([_][2]u64{ .{ pg.devK(ai), pg.hostK(ai) }, .{ pg.devV(ai), pg.hostV(ai) } }) |dh| {
            try pg.copy(stream, dh[1] + w0 * row_bytes, dh[0], shift * row_bytes);
            const len = pos0 - new0;
            var at: usize = 0;
            while (at < len) {
                const piece = @min(shift, len - at);
                try pg.copy(stream, dh[0] + at * row_bytes, dh[0] + (shift + at) * row_bytes, piece * row_bytes);
                at += piece;
            }
        }
        pg.win0[ai] = new0;
        pg.rebases[ai] += 1;
        pg.written[ai] += shift;
    }

    /// Every row the sequence holds written out to host memory (the window's rows too): host K/V then equal what an
    pub fn flush(pg: *Paged, stream: cuda.Stream, fed: usize) !void {
        _ = try pg.snapshot(fed);
        for (0..attn_n) |ai| try pg.flushLayer(stream, ai, fed);
    }

    pub fn snapshot(pg: *const Paged, at: usize) !Snapshot {
        try pg.validateSharedGeometry();
        const image: Snapshot = .{ .layout = try SnapshotLayout.init(pg.cap, pg.cfg, at), .win0 = pg.win0, .rebases = pg.rebases, .written = pg.written };
        try image.validate();
        if (pg.wr != image.layout.wr or pg.nblocks != image.layout.nblocks or (pg.cfg.shared_lanes == 0 and (pg.dev.len != image.layout.device or pg.tabs.len != image.layout.tables)) or pg.host.bytes.len != attn_n * 2 * pg.cap * row_bytes) return error.KvPageSnapshotLayout;
        return image;
    }

    pub fn validateSnapshot(pg: *const Paged, image: Snapshot) !void {
        try pg.validateSharedGeometry();
        try image.validate();
        const expected = try SnapshotLayout.init(pg.cap, pg.cfg, image.layout.at);
        if (!std.meta.eql(expected, image.layout) or pg.wr != expected.wr or pg.nblocks != expected.nblocks or (pg.cfg.shared_lanes == 0 and (pg.dev.len != expected.device or pg.tabs.len != expected.tables)) or pg.host.bytes.len != attn_n * 2 * pg.cap * row_bytes) return error.KvPageSnapshotLayout;
    }

    fn validateSharedGeometry(pg: *const Paged) !void {
        if (pg.cfg.shared_lanes == 0) return;
        if (!pg.shared_attached) return error.SharedKvSequence;
        const arena = pg.shared_arena orelse return error.KvPageSnapshotLayout;
        const expected = try shared_device.Layout.init(.{ .cap = pg.cap, .lanes = pg.cfg.shared_lanes, .window = pg.cfg.window, .rows = pg.cfg.rows, .slots = pg.cfg.slots });
        if (!std.meta.eql(expected, arena.layout) or pg.dev.ptr != arena.device.ptr or pg.dev.len != arena.device.len or pg.tabs.ptr != arena.tables.ptr or pg.tabs.len != arena.tables.len or pg.work.ptr != arena.work.ptr or pg.work.len != arena.work.len) return error.KvPageSnapshotLayout;
    }

    pub fn moveSnapshot(pg: *Paged, stream: cuda.Stream, base: u64, image: Snapshot, saving: bool) !void {
        var offset: usize = 0;
        const prefix = image.layout.at * row_bytes;
        for (0..attn_n) |ai| for ([_]u64{ pg.hostK(ai), pg.hostV(ai) }) |address| {
            if (saving) try pg.copy(stream, base + offset, address, prefix) else try pg.copy(stream, address, base + offset, prefix);
            offset += prefix;
        };
        if (pg.cfg.shared_lanes != 0) {
            for (0..attn_n) |ai| for ([_]u64{ pg.devK(ai), pg.devV(ai) }) |address| {
                const size = pg.wr * row_bytes;
                if (saving) try pg.copy(stream, base + offset, address, size) else try pg.copy(stream, address, base + offset, size);
                offset += size;
            };
            const tables = pg.tab(0);
            for ([_][2]u64{ .{ tables.mark, attn_n * pg.nblocks * 4 }, .{ tables.ctl, attn_n * ctl_n * 4 } }) |section| {
                if (saving) try pg.copy(stream, base + offset, section[0], @intCast(section[1])) else try pg.copy(stream, section[0], base + offset, @intCast(section[1]));
                offset += @intCast(section[1]);
            }
            std.debug.assert(offset == image.layout.total);
            return;
        }
        for ([_]cuda.DeviceBuffer{ pg.dev, pg.tabs }) |buffer| {
            if (saving) try pg.copy(stream, base + offset, buffer.ptr, buffer.len) else try pg.copy(stream, buffer.ptr, base + offset, buffer.len);
            offset += buffer.len;
        }
        std.debug.assert(offset == image.layout.total);
    }

    pub fn publishSnapshot(pg: *Paged, image: Snapshot) void {
        pg.advanceTelemetry();
        pg.win0 = image.win0;
        pg.rebases = image.rebases;
        pg.written = image.written;
    }

    fn advanceTelemetry(pg: *Paged) void {
        if (pg.telemetry_revision == std.math.maxInt(u64)) pg.telemetry_id = 0 else pg.telemetry_revision += 1;
    }

    pub fn flushLayer(pg: *Paged, stream: cuda.Stream, ai: usize, fed: usize) !void {
        if (fed > pg.cap) return error.KvPageRows;
        const w0 = pg.win0[ai];
        if (fed <= w0) return;
        if (fed - w0 > pg.wr) return error.KvPageWindow;
        try pg.copy(stream, pg.hostK(ai) + w0 * row_bytes, pg.devK(ai), (fed - w0) * row_bytes);
        try pg.copy(stream, pg.hostV(ai) + w0 * row_bytes, pg.devV(ai), (fed - w0) * row_bytes);
    }

    /// The host copy of layer `ai`'s K (v false) or V rows (after `flush`).
    pub fn hostRows(pg: *const Paged, ai: usize, v: bool) []const u8 {
        const off = (2 * ai + @intFromBool(v)) * pg.cap * row_bytes;
        return pg.host.bytes[off..][0 .. pg.cap * row_bytes];
    }

    pub fn stats(pg: *const Paged, ai: usize) !Stats {
        var c: [ctl_n]i32 = undefined;
        try pg.d.check(pg.d.api.cuMemcpyDtoH_v2(&c, pg.tab(ai).ctl, ctl_n * 4), "cuMemcpyDtoH");
        return .{ .misses = @as(u32, @bitCast(c[3])), .needed = @as(u32, @bitCast(c[4])), .calls = @as(u32, @bitCast(c[5])), .overflow = c[2] != 0, .rebases = pg.rebases[ai], .written_rows = pg.written[ai], .epoch = @bitCast(c[6]) };
    }

    pub fn sampleClockAfterConsumers(pg: *const Paged) !u32 {
        const address = if (pg.shared_arena) |arena| arena.tables.ptr + arena.layout.clock else pg.tab(0).ctl;
        var clock: u32 = undefined;
        try pg.d.check(pg.d.api.cuMemcpyDtoH_v2(&clock, address, @sizeOf(u32)), "cuMemcpyDtoH(KV telemetry clock)");
        return clock;
    }

    pub fn total(pg: *const Paged) !Stats {
        var t: Stats = .{};
        for (0..attn_n) |ai| t.add(try pg.stats(ai));
        return t;
    }

    pub fn stageStatus(pg: *Paged, stream: cuda.Stream) !void {
        const ctl = pg.tab(0).ctl;
        try pg.d.check(pg.d.api.cuMemcpyDtoHAsync_v2(pg.status_host.bytes.ptr, ctl, attn_n * ctl_n * 4, stream.handle), "cuMemcpyDtoHAsync");
    }

    pub fn checkStatus(pg: *const Paged) !void {
        const controls: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, pg.status_host.bytes[0 .. attn_n * ctl_n * 4]));
        for (0..attn_n) |ai| if (controls[ai * ctl_n + 2] != 0) return error.KvPageOverflow;
    }

    /// One call's paging after fn_qsa_select listed `m` rows' keys in qids / qnk / qspr: the plan (hits stamped, the
    pub fn page(pg: *Paged, stream: cuda.Stream, ai: usize, qids: u64, qnk: u64, qspr: u64, m: usize) !void {
        if (ai >= attn_n) return error.KvPageRows;
        if (try pagingRows(pg.cfg, pg.win0[ai], m) != m) return error.KvPageCapacity;
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            return arena.pageFor(try arena.leaseFor(pg.shared_sequence), pg.shared_sequence, stream, ai, pg.win0[ai], m, qids, qnk, qspr, pg.hostK(ai), pg.hostV(ai));
        }
        const t = pg.tab(ai);
        const w0 = pg.win0[ai];
        const need = pg.work.ptr;
        const miss = need + work_n * 4;
        const fetch = miss + work_n * 4;
        var a: cuda.Args = .{};
        for ([_]u64{ qids, qnk, qspr }) |v| a.add(v);
        for ([_]usize{ m, idw, top, ratio, w0 }) |v| a.add(cint(v));
        a.add(pg.tab(0).ctl);
        a.add(t.blk2slot);
        a.add(t.blk2slot_all);
        a.add(cint(ai * pg.nblocks));
        for ([_]u64{ t.slot2blk, t.stamp, t.mark }) |v| a.add(v);
        a.add(cint(pg.cfg.slots));
        for ([_]u64{ t.ctl, need, miss, fetch }) |v| a.add(v);
        try cuda.launch.launch(pg.k.plan, .{ .grid = .{ .x = 1 }, .block = .{ .x = 1024 } }, stream, &a);
        var f: cuda.Args = .{};
        for ([_]u64{ pg.hostK(ai), pg.hostV(ai), pg.devK(ai), pg.devV(ai), t.ctl, fetch }) |v| f.add(v);
        for ([_]usize{ pg.slotBase(ai), row_bytes / 16, ratio }) |v| f.add(cint(v));
        try cuda.launch.launch(pg.k.fetch, .{ .grid = .{ .x = 128 }, .block = .{ .x = 256 } }, stream, &f);
        var r: cuda.Args = .{};
        for ([_]u64{ qids, qnk, qspr }) |v| r.add(v);
        for ([_]usize{ idw, w0, pg.slotBase(ai) }) |v| r.add(cint(v));
        r.add(t.blk2slot);
        r.add(cint(ratio));
        r.add(t.ctl);
        try cuda.launch.launch(pg.k.remap, .{ .grid = .{ .x = @intCast(m) }, .block = .{ .x = 256 } }, stream, &r);
    }

    pub fn validateRange(pg: *const Paged, ai: usize, pos0: usize, n: usize) !void {
        if (ai >= attn_n or n == 0 or n > pg.cfg.rows or pos0 > pg.cap or n > pg.cap - pos0) return error.KvPageRows;
        _ = try windowPlan(pg.cfg, pg.wr, pg.win0[ai], pos0, n);
    }

    pub fn prepareLayer(pg: *Paged, stream: cuda.Stream, ai: usize, pos0: usize, n: usize) !void {
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            try arena.bindStream(try arena.leaseFor(pg.shared_sequence), stream);
        }
        try pg.validateRange(ai, pos0, n);
        if (pos0 == 0 and ai == 0) try pg.resetTables(stream);
        const w0 = try windowPlan(pg.cfg, pg.wr, pg.win0[ai], pos0, n);
        if (w0 != pg.win0[ai]) try pg.rebase(ai, pos0, stream);
    }

    pub fn validateSharedLease(pg: *const Paged) !void {
        if (pg.cfg.shared_lanes != 0) _ = try pg.shared_arena.?.leaseFor(pg.shared_sequence);
    }

    pub fn validateSharedLeaseOn(pg: *const Paged, stream: cuda.Stream) !void {
        if (pg.cfg.shared_lanes != 0) {
            const arena = pg.shared_arena.?;
            try arena.bindStream(try arena.leaseFor(pg.shared_sequence), stream);
        }
    }

    /// Test support: the window's start of layer `ai` set by hand (the rows before it then come from host memory).
    pub fn setWindow(pg: *Paged, ai: usize, w0: usize) void {
        pg.win0[ai] = w0;
    }

    /// Test support: the device K (v false) or V buffer from layer `ai`'s window base: rows [0, slotRows(ai)).
    pub fn devRows(pg: *const Paged, ai: usize, v: bool) u64 {
        return if (v) pg.devV(ai) else pg.devK(ai);
    }

    /// Test support: rows from layer `ai`'s window base to the end of the slots, and the first slot's row.
    pub fn slotRows(pg: *const Paged, ai: usize) [2]usize {
        return .{ pg.slotBase(ai) + ratio * pg.cfg.slots, pg.slotBase(ai) };
    }

    /// Test support: the host K/V of layer `ai` to write directly.
    pub fn hostMut(pg: *Paged, ai: usize, v: bool) []u8 {
        const off = (2 * ai + @intFromBool(v)) * pg.cap * row_bytes;
        return pg.host.bytes[off..][0 .. pg.cap * row_bytes];
    }

    /// attnRows over the paged cache: prep writes the rows into the window (its base offset by -win0 rows).
    pub fn attnRows(pg: *Paged, p: anytype, lw: anytype, ai: usize, pa: u64, n: usize, at: anytype) !void {
        try pg.validateSharedLeaseOn(p.stream);
        const s = p.s;
        const pos0 = p.pos0;
        try pg.prepareLayer(p.stream, ai, pos0, n);
        const w0 = pg.win0[ai];
        const kd = pg.devK(ai);
        const vd = pg.devV(ai);
        const ikc = p.seq.ikc[ai].ptr;
        const pooled = p.seq.pooled[ai].ptr;
        try p.tri.attnPrep(pa, at.pos, lw.q_scale.ptr, lw.k_scale.ptr, lw.i_scale.ptr, s.inv, at.q, kd -% w0 * row_bytes, vd -% w0 * row_bytes, at.iq, ikc, s.dummy, s.dummy, wt.eps, n, 0, .{});
        try p.tri.qsaPool(ikc, pooled, at.pos, lw.ik_scale.ptr, s.inv, wt.eps, n, .{});
        const nb = parts.scoreStride(p.seq.cap);
        var r0: usize = 0;
        var b: usize = 0;
        while (r0 < n) : ({
            r0 += att_rows;
            b += 1;
        }) {
            const m = @min(att_rows, n - r0);
            const ends = pos0 + r0 + m;
            const pos = at.pos + b * 16;
            try select(p, at.iq + r0 * wt.index_heads * wt.index_dim * 2, pooled, pos, nb, m, ends);
            if (hub.trace) |tr| try tr.record(p.stream, ai, pos0 + r0, m, p.lanes.len > 0, s.qids, s.qnk, s.qspr);
            const chunks = cdiv(@min(ends, wt.dense_keys), 512);
            const page_rows = try pagingRows(pg.cfg, w0, m);
            var first: usize = 0;
            while (first < m) {
                const count = @min(page_rows, m - first);
                const ids = s.qids + first * idw * 4;
                const nk = s.qnk + first * 4;
                const spr = s.qspr + first * 4;
                if (ends / ratio > top) {
                    try pg.page(p.stream, ai, ids, nk, spr, count);
                } else if (w0 != 0) return error.KvPageWindow;
                try p.tri.attnChunks(at.q + (r0 + first) * wt.q_heads * head_dim * 2, kd, vd, pos, s.po, s.pm, s.pl, ids, nk, spr, count, kv_heads, chunks, true, .{});
                try p.tri.attnMerge(s.po, s.pm, s.pl, pos, at.ao + (r0 + first) * wt.out_k * 2, nk, spr, count, kv_heads, true);
                first += count;
            }
        }
    }
};
