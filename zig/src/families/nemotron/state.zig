//! Streams' caches: Mamba states in a slot pool every stream shares, each stream's KV rows, and a forward's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const fwd = @import("forward.zig");
const lesson_route = @import("living").lesson_route;
const kv_pool = @import("../../core/kv_pool.zig");
const kvp = @import("../../core/metal_kv_pages.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// The rows a forward or a stream's window takes at most.
pub const max_rows = 128;

/// A stream's replayed rows at most: a window's rows whose Mamba inputs the next forward runs again first.
pub const replay_rows = 64;

/// Rows of a forward whose Mamba states a later tree row reloads (branch parents), at most.
pub const tree_saves = 64;

/// Head levels whose best tokens a stream keeps.
pub const max_levels = 72;

/// Mamba slots for streams: slot 0, two per stream (state and pending forward), a lone round's third, and a spare.
pub fn poolSlots(streams: usize) usize {
    return 2 * streams + 3;
}

/// Mamba states by slot (slot 0: the zero state, never written), and by replay id a stream's last rows' raw inputs.
pub const Pool = struct {
    conv: [cfg.max_layers]mtl.Buffer = undefined, // [slots, KC-1, CD] bf16, by Mamba index
    ssm: [cfg.max_layers]mtl.Buffer = undefined, // [slots, H, DH, DS] f32
    raw: [cfg.max_layers]mtl.Buffer = undefined, // [rids, 2, replay_rows, CD] bf16
    dtraw: [cfg.max_layers]mtl.Buffer = undefined, // [rids, 2, replay_rows, H] bf16
    tree: mtl.Buffer = undefined, // f32 [tree_saves, H, DH, DS]: a forward's branch parents' states (any layer, in turn)
    layers: usize,
    slots: usize,
    free: std.ArrayList(u32) = .empty,
    rids: std.ArrayList(i32) = .empty,
    gpa: std.mem.Allocator,
    residency: ?mtl.ResidencySet.Follow = null,

    pub fn setResident(self: *Pool, hook: ?mtl.ResidencySet.Follow) void {
        if (self.residency) |f| self.followBuffers(f, false);
        self.residency = hook;
        if (hook) |f| self.followBuffers(f, true);
    }

    fn followBuffers(self: *Pool, hook: mtl.ResidencySet.Follow, joining: bool) void {
        for (0..self.layers) |i| for ([_]mtl.Buffer{ self.conv[i], self.ssm[i], self.raw[i], self.dtraw[i] }) |b|
            mtl.resident.fields.one(b, hook, joining);
        mtl.resident.fields.one(self.tree, hook, joining);
    }

    pub fn appendBuffers(self: *const Pool, gpa: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        for (0..self.layers) |i| try list.appendSlice(gpa, &.{ self.conv[i], self.ssm[i], self.raw[i], self.dtraw[i] });
        try list.append(gpa, self.tree);
    }

    /// Bytes Pool.init's conv, ssm and replay buffers take at `slots` slots and `streams` replay ids (its tree aside).
    pub fn bytes(c: cfg.Config, slots: usize, streams: usize) u64 {
        const slot: u64 = (c.conv_kernel - 1) * c.convDim() * 2 + c.mamba_heads * c.mamba_head_dim * c.state * 4;
        const replay: u64 = 2 * replay_rows * (c.convDim() * 2 + c.mamba_heads * 2);
        return c.count(.mamba) * (slots * slot + streams * replay);
    }

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, c: cfg.Config, slots: usize, streams: usize) !Pool {
        var p = Pool{ .layers = c.count(.mamba), .slots = slots, .gpa = gpa };
        const conv_bytes = slots * (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm_bytes = slots * c.mamba_heads * c.mamba_head_dim * c.state * 4;
        const raw_bytes = streams * 2 * replay_rows * c.convDim() * 2;
        const dtraw_bytes = streams * 2 * replay_rows * c.mamba_heads * 2;
        for (0..p.layers) |m| {
            p.conv[m] = try device.buffer(conv_bytes, opts);
            p.ssm[m] = try device.buffer(ssm_bytes, opts);
            p.raw[m] = try device.buffer(raw_bytes, opts);
            p.dtraw[m] = try device.buffer(dtraw_bytes, opts);
            @memset(p.conv[m].contents()[0..conv_bytes], 0);
            @memset(p.ssm[m].contents()[0..ssm_bytes], 0);
            @memset(p.raw[m].contents()[0..raw_bytes], 0);
            @memset(p.dtraw[m].contents()[0..dtraw_bytes], 0);
        }
        p.tree = try device.buffer(tree_saves * c.mamba_heads * c.mamba_head_dim * c.state * 4, opts);
        var s = slots;
        while (s > 1) {
            s -= 1;
            try p.free.append(gpa, @intCast(s));
        }
        var r = streams;
        while (r > 0) {
            r -= 1;
            try p.rids.append(gpa, @intCast(r));
        }
        return p;
    }

    pub fn take(self: *Pool) !u32 {
        return self.free.pop() orelse error.StateSlotsFull;
    }

    /// Return a slot (never the zero state's).
    pub fn give(self: *Pool, slot: u32) void {
        if (slot != 0) self.free.appendAssumeCapacity(slot);
    }

    pub fn takeRid(self: *Pool) !i32 {
        return self.rids.pop() orelse error.ReplayIdsFull;
    }

    pub fn giveRid(self: *Pool, rid: i32) void {
        if (rid >= 0) self.rids.appendAssumeCapacity(rid);
    }

    pub fn deinit(self: *Pool) void {
        self.setResident(null);
        for (0..self.layers) |m| {
            self.conv[m].deinit();
            self.ssm[m].deinit();
            self.raw[m].deinit();
            self.dtraw[m].deinit();
        }
        self.tree.deinit();
        self.free.deinit(self.gpa);
        self.rids.deinit(self.gpa);
    }
};

/// TF_NEMOTRON_METAL_KV_PAGES=1: a page holds 256 rows of every attention layer's keys then values, the head's last.
pub fn pagedLayout(c: cfg.Config, head: bool) kv_pool.Layout {
    const caches = c.count(.attention) + @intFromBool(head);
    return .{ .fixed_bytes = Cache.bytes(c, 0, head) + Pool.bytes(c, 2, 1), .token_bytes = caches * 2 * c.kv_heads * c.head_dim * 2, .page_tokens = kvp.page_tokens, .snap_bytes = 0 };
}

/// A stream's lease of pages: the pool, and the lane whose table maps its rows.
pub const Lease = struct { pages: *kvp.Pages, lane: usize };

/// A paged cache's rows: cache `index`'s keys and values in every page of the arena, by the lane's table.
pub const Paged = struct {
    lease: Lease,
    k_off: usize, // cache `index`'s keys in a page
    v_off: usize,

    pub fn arena(p: Paged) mtl.Buffer {
        return p.lease.pages.arena.buf.?;
    }

    pub fn table(p: Paged) mtl.Buffer {
        return p.lease.pages.tables.buf.?;
    }

    pub fn tableOff(p: Paged) usize {
        return p.lease.pages.tables.offset(p.lease.lane);
    }

    pub fn pageBytes(p: Paged) u64 {
        return p.lease.pages.arena.page_bytes;
    }

    /// The stream's page ids in row order (all its rows' pages, covered when it opened).
    pub fn ids(p: Paged) []const u32 {
        return p.lease.pages.ids(p.lease.lane);
    }
};

/// One attention cache: [KVH, capacity, HD] bf16 keys and values, `len` rows written; or its rows in pages.
pub const Kv = struct {
    k: mtl.Buffer, // paged: the arena (`paged` says where its rows are)
    v: mtl.Buffer,
    capacity: usize,
    paged: ?Paged = null,

    /// Cache `index` of `caches` over the lease's pages, `capacity` rows covered.
    fn initPaged(c: cfg.Config, lease: Lease, index: usize, capacity: usize) Kv {
        const half = c.kv_heads * kvp.page_tokens * c.head_dim * 2;
        const arena = lease.pages.arena.buf.?;
        return .{ .k = arena, .v = arena, .capacity = capacity, .paged = .{ .lease = lease, .k_off = 2 * index * half, .v_off = (2 * index + 1) * half } };
    }

    fn init(device: mtl.Device, c: cfg.Config, capacity: usize) !Kv {
        const bytes = c.kv_heads * capacity * c.head_dim * 2;
        const k = try device.buffer(bytes, opts);
        errdefer k.deinit();
        return .{ .k = k, .v = try device.buffer(bytes, opts), .capacity = capacity };
    }

    fn deinit(self: Kv) void {
        if (self.paged != null) return; // the pool's arena
        self.k.deinit();
        self.v.deinit();
    }
};

/// A forward's stored state (after its replayed rows, or all rows when `full`) and its rows' input parity.
pub const Pending = struct { rows: usize, slot: ?u32, parity: u1, full: bool };

/// A kept tree path whose key and value rows still sit at the window's rows (len + path[j], moving to len + j).
pub const Compaction = struct { len: usize, n: usize, path: [replay_rows]u32 };

/// One stream's caches: its KV rows (backbone and draft head), its Mamba slot, its window ids on the GPU.
pub const Cache = struct {
    kv: [cfg.max_layers]Kv = undefined, // by attention index
    mtp: ?Kv = null,
    attentions: usize,
    len: usize = 0, // rows in the backbone caches
    mtp_len: usize = 0, // rows in the draft head's cache
    slot: u32 = 0, // the Mamba state before the `replay` rows
    replay: usize = 0, // rows of the last forward the next one runs first (their inputs at `parity`)
    replay_map: [replay_rows]u8 = std.simd.iota(u8, replay_rows), // each replayed row's saved input row (a tree's path)
    parity: u1 = 0,
    rid: i32 = -1, // replay id (-1: none, rows are never saved); also the stream's lesson latch slot
    decide: i64 = lesson_route.none, // the prompt row where the stream's lessons decide (lesson_route)
    untaught: bool = false, // reads the model as shipped: no learned neuron acts on its rows (a lesson writer's reads)
    pending: ?Pending = null,
    compact: ?Compaction = null, // a kept tree path's cache rows, moved by the stream's next command buffer
    windows: [2]mtl.Buffer, // u32 [max_rows]: a window's ids, the pending token then the held drafts
    wcur: u1 = 0,
    start: usize = 0, // this stream's first row in the last forward it took part in
    rows: usize = 0, // and its rows there
    follow: mtl.Buffer, // u32 [max_rows]: the tokens after a round's kept rows, for the head
    topk: mtl.Buffer, // the head's last drafts' best 4 by level: u32 ids [max_levels, 4], then f32 probabilities
    levels: usize = 0, // levels the head last drafted (a tree's trunk: its rank-0 path's steps)
    trunk: [max_levels]u8 = std.simd.iota(u8, max_levels), // each trunk level's top-k slot
    residency: ?mtl.ResidencySet.Follow = null,

    pub fn setResident(self: *Cache, hook: ?mtl.ResidencySet.Follow) void {
        if (self.residency) |f| self.followBuffers(f, false);
        self.residency = hook;
        if (hook) |f| self.followBuffers(f, true);
    }

    fn followBuffers(self: *Cache, hook: mtl.ResidencySet.Follow, joining: bool) void {
        mtl.resident.fields.apply(self, hook, joining);
        for (self.kv[0..self.attentions]) |*kv| if (kv.paged == null) mtl.resident.fields.apply(kv, hook, joining);
        if (self.mtp) |*kv| if (kv.paged == null) mtl.resident.fields.apply(kv, hook, joining);
    }

    pub fn appendBuffers(self: *const Cache, gpa: std.mem.Allocator, list: *std.ArrayList(mtl.Buffer)) !void {
        try mtl.resident.fields.append(gpa, list, self);
        for (self.kv[0..self.attentions]) |*kv| if (kv.paged == null) try mtl.resident.fields.append(gpa, list, kv);
        if (self.mtp) |*kv| if (kv.paged == null) try mtl.resident.fields.append(gpa, list, kv);
    }

    pub fn init(device: mtl.Device, c: cfg.Config, capacity: usize, head: bool) !Cache {
        return initPaged(device, c, capacity, head, null);
    }

    /// `lease`: every cache's rows in the lease's pages (covered for `capacity` rows), no buffers of its own.
    pub fn initPaged(device: mtl.Device, c: cfg.Config, capacity: usize, head: bool, lease: ?Lease) !Cache {
        const first = try device.buffer(max_rows * 4, opts);
        errdefer first.deinit();
        const second = try device.buffer(max_rows * 4, opts);
        errdefer second.deinit();
        const follow = try device.buffer(max_rows * 4, opts);
        errdefer follow.deinit();
        const topk = try device.buffer(2 * max_levels * 4 * 4, opts);
        errdefer topk.deinit();
        var s = Cache{
            .attentions = c.count(.attention),
            .windows = .{ first, second },
            .follow = follow,
            .topk = topk,
        };
        var made: usize = 0;
        errdefer for (s.kv[0..made]) |kv| kv.deinit();
        for (0..s.attentions) |a| {
            s.kv[a] = if (lease) |l| Kv.initPaged(c, l, a, capacity) else try Kv.init(device, c, capacity);
            made += 1;
        }
        if (head) s.mtp = if (lease) |l| Kv.initPaged(c, l, s.attentions, capacity) else try Kv.init(device, c, capacity);
        return s;
    }

    /// Bytes one stream's KV caches and token windows take on its first prompt.
    pub fn bytes(c: cfg.Config, capacity: usize, head: bool) u64 {
        const kv: u64 = (c.count(.attention) + @intFromBool(head)) * 2 * c.kv_heads * capacity * c.head_dim * 2;
        return kv + 3 * max_rows * 4 + 2 * max_levels * 4 * 4;
    }

    /// After a prompt chunk of `rows` rows whose last row's Mamba states went to `slot` (nothing left to replay).
    pub fn advance(self: *Cache, pool: *Pool, rows: usize, slot: u32) void {
        std.debug.assert(self.replay == 0);
        pool.give(self.slot);
        self.slot = slot;
        self.len += rows;
    }

    /// Keep `keep` of a forward's rows: they replay first in the stream's next forward (none after a full store).
    pub fn commit(self: *Cache, pool: *Pool, keep: usize) void {
        const p = self.pending orelse return;
        std.debug.assert(keep >= 1 and keep <= p.rows and (!p.full or keep == p.rows));
        if (p.slot) |s| {
            pool.give(self.slot);
            self.slot = s;
        }
        self.replay = if (p.full) 0 else keep;
        self.replay_map = std.simd.iota(u8, replay_rows);
        self.parity = p.parity;
        self.len += keep;
        self.pending = null;
    }

    pub fn deinit(self: *Cache, pool: *Pool) void {
        self.setResident(null);
        if (self.pending) |p| if (p.slot) |s| pool.give(s);
        pool.give(self.slot);
        pool.giveRid(self.rid);
        if (self.attentions > 0) if (self.kv[0].paged) |p| p.lease.pages.finish(p.lease.lane); // its promise and pages
        for (0..self.attentions) |a| self.kv[a].deinit();
        if (self.mtp) |m| m.deinit();
        for (self.windows) |w| w.deinit();
        self.follow.deinit();
        self.topk.deinit();
    }
};

/// Activations of one forward of up to `rows` rows (MP = rows padded to 16), and the head's logits.
pub const Scratch = struct {
    rows: usize,
    ids: mtl.Buffer, // u32 [rows]: a shared round's window ids
    h: mtl.Buffer, // residual [rows, D]
    x: mtl.Buffer, // normed [rows, D]
    xs: mtl.Buffer, // [D/64, MP]
    proj: mtl.Buffer, // [rows, PROJ]
    xbc: mtl.Buffer, // [replayed + rows, CD]
    y: mtl.Buffer, // [rows, XD]
    yn: mtl.Buffer, // [rows, XD]
    ys: mtl.Buffer, // [XD/64, MP]
    delta: mtl.Buffer, // [rows, D]
    logits_e: mtl.Buffer, // [rows, E] bf16
    idx: mtl.Buffer, // u32 [max(rows*K, 8)]
    wt: mtl.Buffer, // f32 [max(rows*K, 8)]
    uids: mtl.Buffer,
    start: mtl.Buffer,
    count: mtl.Buffer,
    members: mtl.Buffer,
    ucount: mtl.Buffer,
    act: mtl.Buffer, // [rows*K, W]
    ey: mtl.Buffer, // [rows*K, D]
    sh_act: mtl.Buffer, // [rows, SW]
    sh_xs: mtl.Buffer, // [SW/64, MP]
    sh: mtl.Buffer, // [rows, D]
    qkv: mtl.Buffer, // [rows, QKV]
    qp: mtl.Buffer, // [KVH, G*rows, HD]
    po: mtl.Buffer, // f32 [KVH * chunks * G*rows * HD]
    pm: mtl.Buffer,
    pl: mtl.Buffer,
    att: mtl.Buffer, // [H, rows, HD]
    ax: mtl.Buffer, // [rows, H*HD]
    axs: mtl.Buffer, // [H*HD/64, MP]
    hx: mtl.Buffer, // head input row sums [D/64, 16]
    logits: mtl.Buffer, // [rows, V] bf16
    part: mtl.Buffer, // f32 [SK, MP, N]: split-K partials of the narrow projections
    tpaths: mtl.Buffer, // i32 [max_rows, tree.max_depth]: a tree's rows' ancestors by depth
    tdepths: mtl.Buffer, // i32 [max_rows]
    tpo: mtl.Buffer, // f32 [KVH, 2, rows, 16, HD]: tree rows' tail partials
    tpm: mtl.Buffer,
    tpl: mtl.Buffer,
    kept: mtl.Buffer, // [rows, D]: a tree's kept rows gathered in path order, for the head
    route_log: ?mtl.Buffer = null, // a probe's copy of each MoE layer's expert picks (u32 [layers, rows * K])
    route_at: usize = 0,

    pub fn init(device: mtl.Device, c: cfg.Config, rows: usize, capacity: usize) !Scratch {
        const mp = 16 * ((rows + 15) / 16);
        const pairs = @max(rows * c.top_k, 8);
        const g = c.heads / c.kv_heads;
        const chunks = (capacity + 511) / 512;
        const B = struct {
            fn make(d: mtl.Device, bytes: usize) !mtl.Buffer {
                const n = @max(bytes, 64);
                const b = try d.buffer(n, opts);
                @memset(b.contents()[0..n], 0);
                return b;
            }
        };
        return .{
            .rows = rows,
            .ids = try B.make(device, @max(rows, 8) * 4),
            .h = try B.make(device, rows * c.hidden * 2),
            .x = try B.make(device, rows * c.hidden * 2),
            .xs = try B.make(device, c.hidden / 64 * mp * 4),
            .proj = try B.make(device, rows * c.projDim() * 2),
            .xbc = try B.make(device, 2 * rows * c.convDim() * 2), // replayed rows and the forward's
            .y = try B.make(device, rows * c.inner() * 2),
            .yn = try B.make(device, rows * c.inner() * 2),
            .ys = try B.make(device, c.inner() / 64 * mp * 4),
            .delta = try B.make(device, rows * c.hidden * 2),
            .logits_e = try B.make(device, rows * c.experts * 2),
            .idx = try B.make(device, pairs * 4),
            .wt = try B.make(device, pairs * 4),
            .uids = try B.make(device, pairs * 4),
            .start = try B.make(device, pairs * 4),
            .count = try B.make(device, pairs * 4),
            .members = try B.make(device, pairs * 4),
            .ucount = try B.make(device, 16),
            .act = try B.make(device, pairs * c.expert_width * 2),
            .ey = try B.make(device, pairs * c.hidden * 2),
            .sh_act = try B.make(device, rows * c.shared_width * 2),
            .sh_xs = try B.make(device, c.shared_width / 64 * mp * 4),
            .sh = try B.make(device, rows * c.hidden * 2),
            .qkv = try B.make(device, rows * c.qkvDim() * 2),
            .qp = try B.make(device, c.kv_heads * g * rows * c.head_dim * 2),
            .po = try B.make(device, c.kv_heads * chunks * g * mp * c.head_dim * 4),
            .pm = try B.make(device, c.kv_heads * chunks * g * mp * 4),
            .pl = try B.make(device, c.kv_heads * chunks * g * mp * 4),
            .att = try B.make(device, c.heads * rows * c.head_dim * 2),
            .ax = try B.make(device, rows * c.heads * c.head_dim * 2),
            .axs = try B.make(device, c.heads * c.head_dim / 64 * mp * 4),
            .hx = try B.make(device, c.hidden / 64 * 16 * 4),
            .logits = try B.make(device, rows * c.vocab * 2),
            .tpaths = try B.make(device, max_rows * 64 * 4),
            .tdepths = try B.make(device, max_rows * 4),
            .tpo = try B.make(device, c.kv_heads * 2 * mp * 16 * c.head_dim * 4),
            .tpm = try B.make(device, c.kv_heads * 2 * mp * 16 * 4),
            .tpl = try B.make(device, c.kv_heads * 2 * mp * 16 * 4),
            .kept = try B.make(device, rows * c.hidden * 2),
            .part = try B.make(device, partCols(c) * mp * 4),
        };
    }

    pub fn deinit(self: *Scratch) void {
        const info = @typeInfo(Scratch).@"struct";
        inline for (info.field_names, info.field_types) |name, T| {
            if (T == mtl.Buffer) @field(self, name).deinit();
        }
    }
};

/// The widest split-K partial row over the projections forward.coop splits (in, out, the head's eh).
fn partCols(c: cfg.Config) usize {
    const shapes = [_][2]usize{ .{ c.projDim(), c.hidden }, .{ c.hidden, c.inner() }, .{ c.hidden, 2 * c.hidden } };
    var most: usize = 0;
    for (shapes) |nk| most = @max(most, fwd.splitK(nk[0], nk[1]) * nk[0]);
    return most;
}

test "Cache.bytes matches the Metal buffers Cache.init allocates" {
    var why: cfg.Why = .{};
    const c = try cfg.parse(std.testing.allocator, cfg.lightning, &why);
    var device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var pool = try Pool.init(std.testing.allocator, device, c, 3, 2);
    defer pool.deinit();
    for ([_]bool{ false, true }) |head| {
        var cache = try Cache.init(device, c, 4096, head);
        defer cache.deinit(&pool);
        var actual: u64 = cache.windows[0].length() + cache.windows[1].length() + cache.follow.length() + cache.topk.length();
        for (cache.kv[0..cache.attentions]) |kv| actual += kv.k.length() + kv.v.length();
        if (cache.mtp) |m| actual += m.k.length() + m.v.length();
        try std.testing.expectEqual(actual, Cache.bytes(c, 4096, head));
    }
}

test "Pool.bytes matches the Metal buffers Pool.init allocates, its tree aside" {
    var why: cfg.Why = .{};
    const c = try cfg.parse(std.testing.allocator, cfg.lightning, &why);
    var device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    var pool = try Pool.init(std.testing.allocator, device, c, 7, 3);
    defer pool.deinit();
    var actual: u64 = 0;
    for (0..pool.layers) |m| actual += pool.conv[m].length() + pool.ssm[m].length() + pool.raw[m].length() + pool.dtraw[m].length();
    try std.testing.expectEqual(actual, Pool.bytes(c, 7, 3));
}

test "a pool of poolSlots holds every stream's state and pending forward, a lone round's third, and one spare" {
    var why: cfg.Why = .{};
    const c = try cfg.parse(std.testing.allocator, cfg.lightning, &why);
    var device = mtl.Device.init() catch return error.SkipZigTest;
    defer device.deinit();
    for ([_]usize{ 1, 2, 4, 8 }) |streams| {
        var pool = try Pool.init(std.testing.allocator, device, c, poolSlots(streams), streams + 2);
        defer pool.deinit();
        for (0..2 * streams + 1) |_| _ = try pool.take();
        _ = try pool.take();
        try std.testing.expectError(error.StateSlotsFull, pool.take());
    }
}
