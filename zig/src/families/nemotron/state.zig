//! Streams' caches: Mamba states in a slot pool every stream shares, each stream's KV rows, and a forward's scratch.
const std = @import("std");
const mtl = @import("metal");
const cfg = @import("config.zig");
const fwd = @import("forward.zig");

const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;

/// The rows a forward or a stream's window takes at most.
pub const max_rows = 128;

/// A stream's replayed rows at most: a window's rows whose Mamba inputs the next forward runs again first.
pub const replay_rows = 64;

/// Rows of a forward whose Mamba states a later tree row reloads (branch parents), at most.
pub const tree_saves = 64;

/// Head levels whose best tokens a stream keeps.
pub const max_levels = 72;

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

    pub fn init(gpa: std.mem.Allocator, device: mtl.Device, c: cfg.Config, slots: usize, streams: usize) !Pool {
        var p = Pool{ .layers = c.count(.mamba), .slots = slots, .gpa = gpa };
        const conv_bytes = slots * (c.conv_kernel - 1) * c.convDim() * 2;
        const ssm_bytes = slots * c.mamba_heads * c.mamba_head_dim * c.state * 4;
        for (0..p.layers) |m| {
            p.conv[m] = try device.buffer(conv_bytes, opts);
            p.ssm[m] = try device.buffer(ssm_bytes, opts);
            p.raw[m] = try device.buffer(streams * 2 * replay_rows * c.convDim() * 2, opts);
            p.dtraw[m] = try device.buffer(streams * 2 * replay_rows * c.mamba_heads * 2, opts);
            @memset(p.conv[m].contents()[0 .. conv_bytes / slots], 0);
            @memset(p.ssm[m].contents()[0 .. ssm_bytes / slots], 0);
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

/// One attention cache: [KVH, capacity, HD] bf16 keys and values, `len` rows written.
pub const Kv = struct {
    k: mtl.Buffer,
    v: mtl.Buffer,
    capacity: usize,

    fn init(device: mtl.Device, c: cfg.Config, capacity: usize) !Kv {
        const bytes = c.kv_heads * capacity * c.head_dim * 2;
        return .{ .k = try device.buffer(bytes, opts), .v = try device.buffer(bytes, opts), .capacity = capacity };
    }

    fn deinit(self: Kv) void {
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
    rid: i32 = -1, // replay id (-1: none, rows are never saved)
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

    pub fn init(device: mtl.Device, c: cfg.Config, capacity: usize, head: bool) !Cache {
        var s = Cache{
            .attentions = c.count(.attention),
            .windows = .{ try device.buffer(max_rows * 4, opts), try device.buffer(max_rows * 4, opts) },
            .follow = try device.buffer(max_rows * 4, opts),
            .topk = try device.buffer(2 * max_levels * 4 * 4, opts),
        };
        for (0..s.attentions) |a| s.kv[a] = try Kv.init(device, c, capacity);
        if (head) s.mtp = try Kv.init(device, c, capacity);
        return s;
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
        if (self.pending) |p| if (p.slot) |s| pool.give(s);
        pool.give(self.slot);
        pool.giveRid(self.rid);
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
                return d.buffer(@max(bytes, 64), opts);
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
