//! Split KV's row exchange: each rank sends its owned selected comp rows, one all-gather; bit-equal to replicated.

const std = @import("std");
const tp = @import("tp");
const DevicePtr = tp.collective.DevicePtr;
const Stream = tp.collective.Stream;

/// One selection's dense gather: [R, K] logical comp rows -> this rank's owned rows into send, and remapped tokens.
pub const DenseArgs = extern struct {
    /// int32 [R, K] logical rows (-1: none)
    sel: u64,
    rows: u32,
    k: u32,
    /// int32 split page tables (Slot.localTableAt): one [pages] table, or row mode's stacked [S, pts]
    table: u64,
    pts: u32,
    /// int32 [R]: each row's slot in the stacked tables (0: every row reads table row 0)
    rslot: u64,
    /// log2 of rows a page
    psh: u32,
    /// the family's local tensor and its row bytes
    base: u64,
    row_bytes: u32,
    world: u32,
    /// out: [R x K, row bytes] and int32 [R, K]
    send: u64,
    tok: u64,
};

/// One selection's packed gather (`Exchange.densePacked`): DenseArgs' fields at the same offsets, plus rank and lens.
pub const PackArgs = extern struct {
    /// int32 [R, K] logical rows (-1: none)
    sel: u64,
    rows: u32,
    k: u32,
    /// int32 split page tables, as DenseArgs
    table: u64,
    pts: u32,
    /// int32 [R] or 0, as DenseArgs
    rslot: u64,
    psh: u32,
    /// this rank (sends the entries whose owner it is)
    rank: u32,
    base: u64,
    row_bytes: u32,
    world: u32,
    /// out: [R x K, row bytes] (own rows first c_rank, list order), int32 [R, K] tokens, int32 [world] count x rb
    send: u64,
    tok: u64,
    lens: u64,

    pub fn of(a: DenseArgs, rank: u32, lens: u64) PackArgs {
        return .{ .sel = a.sel, .rows = a.rows, .k = a.k, .table = a.table, .pts = a.pts, .rslot = a.rslot, .psh = a.psh, .rank = rank, .base = a.base, .row_bytes = a.row_bytes, .world = a.world, .send = a.send, .tok = a.tok, .lens = lens };
    }

    pub fn dense(p: PackArgs) DenseArgs {
        return .{ .sel = p.sel, .rows = p.rows, .k = p.k, .table = p.table, .pts = p.pts, .rslot = p.rslot, .psh = p.psh, .base = p.base, .row_bytes = p.row_bytes, .world = p.world, .send = p.send, .tok = p.tok };
    }
};

/// TF_DSV41_KV_SPLIT_DEDUP's pack (kvsplit.cu dedup_*): a slot's row sent once, its entries read one place.
pub const DedupArgs = extern struct {
    p: PackArgs,
    /// int32 [map rows]: key slot x (pts << psh) + logical row
    map: u64,
    /// int32 [R x K] each, and uint32 [blocks of 256 entries, max_world]
    first: u64,
    place: u64,
    bcnt: u64,
};

/// The dedup knob's device scratch (Kv's), and what its map covers.
pub const Dedup = struct {
    map: u64,
    map_rows: u64,
    first: u64,
    place: u64,
    bcnt: u64,
    /// entries the scratch takes (R x K)
    entries: u64,
    /// slots the map holds (row mode's rslot values below it)
    slots: u32,

    /// A row-mode window (one stream's exchanges) whose keys fit the map and entries the scratch: alike on every rank.
    pub fn covers(d: Dedup, a: PackArgs) bool {
        const n = @as(u64, a.rows) * a.k;
        const keys = @as(u64, d.slots) * (@as(u64, a.pts) << @intCast(a.psh));
        return a.rslot != 0 and n > 0 and n <= d.entries and keys <= d.map_rows;
    }

    pub fn args(d: Dedup, a: PackArgs) DedupArgs {
        return .{ .p = a, .map = d.map, .first = d.first, .place = d.place, .bcnt = d.bcnt };
    }
};

/// TF_DSV41_KV_SPLIT_DEDUP=1: packed exchanges send each slot's unique rows (default off).
pub fn dedupOn() bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_KV_SPLIT_DEDUP") orelse return false);
    return std.mem.eql(u8, std.mem.trim(u8, v, " \t"), "1");
}

/// TF_DSV41_KV_SPLIT_EARLY=1: an index layer's exchange issued right after its selection (block.zig; default off).
pub fn earlyOn() bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_KV_SPLIT_EARLY") orelse return false);
    return std.mem.eql(u8, std.mem.trim(u8, v, " \t"), "1");
}

/// A prefill union by history (`Exchange.history`): owner o's rows of the mapped pages at o x M, in page order.
pub const HistArgs = extern struct {
    /// int32 [R, K] logical rows (-1: none)
    sel: u64,
    rows: u32,
    k: u32,
    /// int32 [pages] split page table (Slot.localTableAt)
    table: u64,
    /// the slot's mapped pages (Slot.len): every row a selection names is below pages << psh
    pages: u32,
    psh: u32,
    rank: u32,
    world: u32,
    /// rows an owner (historyRows): owner o's row list fills recv rows [o M, o M + M)
    m: u32,
    row_bytes: u32,
    base: u64,
    /// out: [M, row bytes] (this rank's rows in page order, the padding unspecified) and int32 [R, K] tokens
    send: u64,
    tok: u64,
};

/// Rows an owner in a history union of `pages` mapped pages: owner 0's pages (the most), whole.
pub fn historyRows(pages: u32, psh: u32, world: u32) u32 {
    return ((pages + world - 1) / world) << @intCast(psh);
}

/// A row's place in its owner's history list: its page's turn among the owner's pages, then its row in it.
pub fn historyIndex(t: u32, psh: u32, world: u32) u32 {
    const mask = (@as(u32, 1) << @intCast(psh)) - 1;
    return (((t >> @intCast(psh)) / world) << @intCast(psh)) | (t & mask);
}

/// TF_DSV41_KV_SPLIT_HISTORY=1: a prefill union under the cap sends the owned rows; no plan, no host sync.
pub fn historyOn() bool {
    const v = std.mem.span(std.c.getenv("TF_DSV41_KV_SPLIT_HISTORY") orelse return false);
    return std.mem.eql(u8, std.mem.trim(u8, v, " \t"), "1");
}

comptime {
    std.debug.assert(@sizeOf(DenseArgs) == 80 and @sizeOf(PackArgs) == 88 and @sizeOf(DedupArgs) == 120 and @sizeOf(HistArgs) == 72);
    std.debug.assert(@offsetOf(PackArgs, "rank") == 44 and @offsetOf(PackArgs, "base") == @offsetOf(DenseArgs, "base") and @offsetOf(PackArgs, "lens") == 80);
}

/// The device work around the all-gather: dense gather + remap and a gather by row list (HostKernels below).
pub const Kernels = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        dense: *const fn (ptr: *anyopaque, a: DenseArgs, stream: Stream) anyerror!void,
        /// Packed gather: tokens, owned rows into send[0:c_rank], lens (one launch, W <= 8). Null: densePacked refuses.
        pack: ?*const fn (ptr: *anyopaque, a: PackArgs, stream: Stream) anyerror!void = null,
        /// send[i] = base row phys[i] (uint32 [n])
        gather: *const fn (ptr: *anyopaque, base: u64, row_bytes: u32, phys: u64, n: u32, send: u64, stream: Stream) anyerror!void,
        /// The deduplicated pack (kvsplit.cu dedup_*: four launches); null: a Kernels without it (packs without dedup).
        dedup: ?*const fn (ptr: *anyopaque, a: DedupArgs, stream: Stream) anyerror!void = null,
        /// The history union's gather and tokens (kvsplit.cu hist_*: two launches); null: unions plan on the host.
        history: ?*const fn (ptr: *anyopaque, a: HistArgs, stream: Stream) anyerror!void = null,
    };
};

/// The kernels on host memory (host pointers): the reference the device kernels must equal, and the host ranks' path.
pub const HostKernels = struct {
    pub fn kernels(h: *HostKernels) Kernels {
        return .{ .ptr = h, .vtable = &.{ .dense = dense, .gather = gather, .pack = pack, .dedup = dedup, .history = history } };
    }

    /// hist_gather / hist_tok: row i of this rank's list is its page rank + W (i >> psh); the tokens by historyIndex.
    fn history(_: *anyopaque, a: HistArgs, _: Stream) anyerror!void {
        const sel: [*]const i32 = @ptrFromInt(a.sel);
        const table: [*]const i32 = @ptrFromInt(a.table);
        const base: [*]const u8 = @ptrFromInt(a.base);
        const send: [*]u8 = @ptrFromInt(a.send);
        const tok: [*]i32 = @ptrFromInt(a.tok);
        const mask = (@as(u32, 1) << @intCast(a.psh)) - 1;
        for (0..a.m) |i| {
            const lp = a.rank + a.world * (@as(u32, @intCast(i)) >> @intCast(a.psh));
            if (lp >= a.pages) continue;
            const phys = (@as(usize, @intCast(table[lp])) << @intCast(a.psh)) | (i & mask);
            @memcpy(send[i * a.row_bytes ..][0..a.row_bytes], base[phys * a.row_bytes ..][0..a.row_bytes]);
        }
        for (0..@as(usize, a.rows) * a.k) |i| {
            if (sel[i] < 0) {
                tok[i] = -1;
                continue;
            }
            const t: u32 = @intCast(sel[i]);
            if (t >> @intCast(a.psh) >= a.pages) return error.RowOutsideTable;
            tok[i] = @intCast(ownerOf(t, a.psh, a.world) * a.m + historyIndex(t, a.psh, a.world));
        }
    }

    /// dedup_*'s steps in order on host memory: the first entry of each key, places among each owner's first entries.
    fn dedup(_: *anyopaque, d: DedupArgs, _: Stream) anyerror!void {
        const a = d.p;
        if (a.world < 1 or a.world > max_world or a.rank >= a.world) return error.BadWorld;
        const n = @as(usize, a.rows) * a.k;
        const sel: [*]const i32 = @ptrFromInt(a.sel);
        const tab: [*]const i32 = @ptrFromInt(a.table);
        const rslot: ?[*]const i32 = if (a.rslot == 0) null else @ptrFromInt(a.rslot);
        const map: [*]i32 = @ptrFromInt(d.map);
        const first: [*]i32 = @ptrFromInt(d.first);
        const place: [*]i32 = @ptrFromInt(d.place);
        const base: [*]const u8 = @ptrFromInt(a.base);
        const send: [*]u8 = @ptrFromInt(a.send);
        const tok: [*]i32 = @ptrFromInt(a.tok);
        const lens: [*]i32 = @ptrFromInt(a.lens);
        const mask = (@as(u32, 1) << @intCast(a.psh)) - 1;
        const span = @as(usize, a.pts) << @intCast(a.psh);
        const key = struct {
            fn of(rs: ?[*]const i32, k: u32, sp: usize, i: usize, t: i32) usize {
                return (if (rs) |r| @as(usize, @intCast(r[i / k])) * sp else 0) + @as(usize, @intCast(t));
            }
        }.of;
        for (0..n) |i| if (sel[i] >= 0) {
            const at = &map[key(rslot, a.k, span, i, sel[i])];
            at.* = @min(at.*, @as(i32, @intCast(i)));
        };
        var count: [max_world]u32 = @splat(0);
        for (0..n) |i| {
            first[i] = if (sel[i] < 0) -1 else map[key(rslot, a.k, span, i, sel[i])];
            if (first[i] != @as(i32, @intCast(i))) continue;
            const t: u32 = @intCast(sel[i]);
            const o = ownerOf(t, a.psh, a.world);
            place[i] = @intCast(count[o]);
            if (o == a.rank) {
                const s: usize = if (rslot) |rs| @intCast(rs[i / a.k]) else 0;
                const local: u32 = @intCast(tab[s * a.pts + (t >> @intCast(a.psh))]);
                const phys = (@as(usize, local) << @intCast(a.psh)) | (t & mask);
                @memcpy(send[count[o] * a.row_bytes ..][0..a.row_bytes], base[phys * a.row_bytes ..][0..a.row_bytes]);
            }
            count[o] += 1;
        }
        for (0..n) |i| {
            if (first[i] < 0) {
                tok[i] = -1;
                continue;
            }
            const t: u32 = @intCast(sel[i]);
            tok[i] = @intCast(ownerOf(t, a.psh, a.world) * n + @as(usize, @intCast(place[@intCast(first[i])])));
        }
        for (0..n) |i| if (first[i] == @as(i32, @intCast(i))) {
            map[key(rslot, a.k, span, i, sel[i])] = std.math.maxInt(i32);
        };
        for (0..a.world) |o| lens[o] = @intCast(count[o] * a.row_bytes);
    }

    /// One pass in list order: an owner's running count is each entry's place (pack_kernel's prefix sums).
    fn pack(_: *anyopaque, a: PackArgs, _: Stream) anyerror!void {
        if (a.world < 1 or a.world > max_world or a.rank >= a.world) return error.BadWorld;
        const n = @as(usize, a.rows) * a.k;
        const sel: [*]const i32 = @ptrFromInt(a.sel);
        const tab: [*]const i32 = @ptrFromInt(a.table);
        const rslot: ?[*]const i32 = if (a.rslot == 0) null else @ptrFromInt(a.rslot);
        const base: [*]const u8 = @ptrFromInt(a.base);
        const send: [*]u8 = @ptrFromInt(a.send);
        const tok: [*]i32 = @ptrFromInt(a.tok);
        const lens: [*]i32 = @ptrFromInt(a.lens);
        const mask = (@as(u32, 1) << @intCast(a.psh)) - 1;
        var count: [max_world]u32 = @splat(0);
        for (0..n) |i| {
            if (sel[i] < 0) {
                tok[i] = -1;
                continue;
            }
            const t: u32 = @intCast(sel[i]);
            const o = ownerOf(t, a.psh, a.world);
            const at = count[o];
            count[o] += 1;
            tok[i] = @intCast(o * n + at);
            if (o != a.rank) continue;
            const s: usize = if (rslot) |rs| @intCast(rs[i / a.k]) else 0;
            const local: u32 = @intCast(tab[s * a.pts + (t >> @intCast(a.psh))]);
            const phys = (@as(usize, local) << @intCast(a.psh)) | (t & mask);
            @memcpy(send[at * a.row_bytes ..][0..a.row_bytes], base[phys * a.row_bytes ..][0..a.row_bytes]);
        }
        for (0..a.world) |o| lens[o] = @intCast(count[o] * a.row_bytes);
    }

    fn dense(_: *anyopaque, a: DenseArgs, _: Stream) anyerror!void {
        const n = @as(usize, a.rows) * a.k;
        const sel: [*]const i32 = @ptrFromInt(a.sel);
        const tab: [*]const i32 = @ptrFromInt(a.table);
        const rslot: ?[*]const i32 = if (a.rslot == 0) null else @ptrFromInt(a.rslot);
        const base: [*]const u8 = @ptrFromInt(a.base);
        const send: [*]u8 = @ptrFromInt(a.send);
        const tok: [*]i32 = @ptrFromInt(a.tok);
        const mask = (@as(u32, 1) << @intCast(a.psh)) - 1;
        for (0..n) |i| {
            const t: u32 = @intCast(@max(sel[i], 0));
            const row = i / a.k;
            const s: usize = if (rslot) |rs| @intCast(rs[row]) else 0;
            const local: u32 = @intCast(tab[s * a.pts + (t >> @intCast(a.psh))]);
            const phys = (@as(usize, local) << @intCast(a.psh)) | (t & mask);
            @memcpy(send[i * a.row_bytes ..][0..a.row_bytes], base[phys * a.row_bytes ..][0..a.row_bytes]);
            tok[i] = if (sel[i] < 0) -1 else @intCast(ownerOf(t, a.psh, a.world) * n + i);
        }
    }

    fn gather(_: *anyopaque, base: u64, rb: u32, phys: u64, n: u32, send: u64, _: Stream) anyerror!void {
        const ph: [*]const u32 = @ptrFromInt(phys);
        const b: [*]const u8 = @ptrFromInt(base);
        const s: [*]u8 = @ptrFromInt(send);
        for (0..n) |i| @memcpy(s[i * rb ..][0..rb], b[@as(usize, ph[i]) * rb ..][0..rb]);
    }
};

/// The rank that stores logical row t of a split family: its logical page's residue.
pub fn ownerOf(t: u32, psh: u32, world: u32) u32 {
    return (t >> @intCast(psh)) % world;
}

/// Send and receive bytes of a dense exchange of R x K entries.
pub fn denseBytes(rows: u32, k: u32, row_bytes: u32, world: u32) struct { send: u64, recv: u64 } {
    const s = @as(u64, rows) * k * row_bytes;
    return .{ .send = s, .recv = s * world };
}

/// A prefill union block: rows [a, b), m rows an owner (padded), this rank's local rows to send, the block's tokens.
pub const Block = struct {
    a: u32,
    b: u32,
    m: u32,
    phys: []u32,
    tokens: []i32,

    pub fn deinit(blk: *Block, gpa: std.mem.Allocator) void {
        gpa.free(blk.phys);
        gpa.free(blk.tokens);
    }
};

pub const UnionIn = struct {
    /// [rows, k] logical rows (-1: none), host
    sel: []const i32,
    k: u32,
    /// the slot's split table (local page by logical page), host
    table: []const u32,
    psh: u32,
    world: u32,
    rank: u32,
    row_bytes: u32,
    /// send + receive bytes a block may take (TF_DSV41_KV_SPLIT_UNION_MIB)
    cap: u64,
};

/// Ranks a split exchange plans for (the session pool's bound: sessions.pool.max_world).
pub const max_world = 8;

/// Plans a segment's union exchange: blocks of rows whose (W + 1) x M rows fit the cap; O(entries + rows), no sort.
pub fn planUnion(gpa: std.mem.Allocator, in: UnionIn, out: *std.ArrayList(Block)) !void {
    if (in.world < 1 or in.world > max_world or in.rank >= in.world) return error.BadWorld;
    const rows: u32 = @intCast(in.sel.len / in.k);
    // a row's mark, then its index + 1 in its owner's list (every row a selection names is in the table's pages)
    const seen = try gpa.alloc(u32, in.table.len << @intCast(in.psh));
    defer gpa.free(seen);
    try planRange(gpa, in, seen, 0, rows, out);
}

fn planRange(gpa: std.mem.Allocator, in: UnionIn, seen: []u32, a: u32, b: u32, out: *std.ArrayList(Block)) !void {
    const part = in.sel[@as(usize, a) * in.k .. @as(usize, b) * in.k];
    @memset(seen, 0);
    for (part) |t| if (t >= 0) {
        if (t >= seen.len) return error.RowOutsideTable;
        seen[@intCast(t)] = 1;
    };
    // each owner's rows in ascending order (the sorted unique union, partitioned by owner): a row's index is a count
    var sizes: [max_world]u32 = @splat(0);
    for (seen, 0..) |*x, t| if (x.* != 0) {
        const r = ownerOf(@intCast(t), in.psh, in.world);
        sizes[r] += 1;
        x.* = sizes[r];
    };
    var m: u32 = 1;
    for (sizes[0..in.world]) |sz| m = @max(m, sz);
    if (@as(u64, in.world + 1) * m * in.row_bytes > in.cap and b - a > 1) {
        const h = a + (b - a) / 2;
        try planRange(gpa, in, seen, a, h, out);
        return planRange(gpa, in, seen, h, b, out);
    }
    var blk: Block = .{ .a = a, .b = b, .m = m, .phys = try gpa.alloc(u32, m), .tokens = try gpa.alloc(i32, part.len) };
    errdefer blk.deinit(gpa);
    @memset(blk.phys, 0); // padding: any stored row
    const mask = (@as(u32, 1) << @intCast(in.psh)) - 1;
    for (seen, 0..) |x, t| if (x != 0 and ownerOf(@intCast(t), in.psh, in.world) == in.rank) {
        const u: u32 = @intCast(t);
        blk.phys[x - 1] = (in.table[u >> @intCast(in.psh)] << @intCast(in.psh)) | (u & mask);
    };
    for (part, blk.tokens) |t, *o| {
        if (t < 0) {
            o.* = -1;
            continue;
        }
        const u: u32 = @intCast(t);
        o.* = @intCast(ownerOf(u, in.psh, in.world) * m + seen[u] - 1);
    }
    try out.append(gpa, blk);
}

/// TF_DSV41_KV_SPLIT_COMPACT: off (0, dense), on (1: packed where all transports take lengths and it fits), force.
pub const Compact = enum { off, on, force };

pub const compact_env = "TF_DSV41_KV_SPLIT_COMPACT";

pub fn parseCompact(text: []const u8) error{BadCompactMode}!Compact {
    const v = std.mem.trim(u8, text, " \t");
    const eq = std.ascii.eqlIgnoreCase;
    if (v.len == 0 or eq(v, "0") or eq(v, "off") or eq(v, "no") or eq(v, "false")) return .off;
    if (eq(v, "1") or eq(v, "on") or eq(v, "yes") or eq(v, "true")) return .on;
    if (eq(v, "force")) return .force;
    return error.BadCompactMode;
}

/// The knob from the environment (unset: off).
pub fn compactMode() error{BadCompactMode}!Compact {
    const raw = std.mem.span(std.c.getenv(compact_env) orelse return .off);
    return parseCompact(raw) catch |e| {
        std.log.err("[tensorfold] {s}={s}: 0 / 1 / force", .{ compact_env, raw });
        return e;
    };
}

/// The exchange itself: kernels to fill the send buffer, the communicator's all-gather to fill the receive buffer.
pub const Exchange = struct {
    comm: tp.Collective,
    kernels: Kernels,
    compact: Compact = .off,
    /// Every rank's transport takes device lengths (`Collective.varAgreed`, asked once at `init` when compact is on).
    var_ok: bool = false,
    /// TF_DSV41_KV_SPLIT_DEDUP's scratch (null: off)
    dedup: ?Dedup = null,

    /// An exchange under `mode`; every rank calls it alike outside graphs (`on` asks the communicator once).
    pub fn init(comm: tp.Collective, kernels: Kernels, mode: Compact) !Exchange {
        return .{ .comm = comm, .kernels = kernels, .compact = mode, .var_ok = mode == .on and try comm.varAgreed() };
    }

    /// Whether a dense exchange of `nbytes`-byte shards (R x K x row bytes) goes packed: the same answer on every rank.
    pub fn packs(x: Exchange, nbytes: u64) bool {
        return x.compact == .force or (x.compact == .on and x.var_ok and x.comm.varFits(@intCast(nbytes)));
    }

    /// The window's exchange as the knob says: densePacked where `packs`, else dense (a.lens unused then).
    pub fn window(x: Exchange, a: PackArgs, recv: DevicePtr, stream: Stream) !void {
        if (x.packs(@as(u64, a.rows) * a.k * a.row_bytes)) {
            if (x.dedup) |d| if (x.kernels.vtable.dedup) |f| if (d.covers(a)) {
                try f(x.kernels.ptr, d.args(a), stream);
                return x.comm.allGatherV(a.send, recv, @as(usize, a.rows) * a.k * a.row_bytes, a.lens, .u8, stream);
            };
            return x.densePacked(a, recv, stream);
        }
        return x.dense(a.dense(), recv, stream);
    }

    /// Packed: send (own c_rank rows first), tokens, lens by one kernel, then recv [W, R x K, rb]; graph-capturable.
    pub fn densePacked(x: Exchange, a: PackArgs, recv: DevicePtr, stream: Stream) !void {
        const f = x.kernels.vtable.pack orelse return error.Unsupported;
        try f(x.kernels.ptr, a, stream);
        try x.comm.allGatherV(a.send, recv, @as(usize, a.rows) * a.k * a.row_bytes, a.lens, .u8, stream);
    }

    /// Dense: send [R x K, rb] gathered, tokens remapped, then recv [W, R x K, rb] (graph-capturable).
    pub fn dense(x: Exchange, a: DenseArgs, recv: DevicePtr, stream: Stream) !void {
        try x.kernels.vtable.dense(x.kernels.ptr, a, stream);
        try x.comm.allGather(a.send, recv, @as(usize, a.rows) * a.k * a.row_bytes, .u8, stream);
    }

    /// A history union: this rank's M rows and the tokens (two kernels), then recv [W, M, rb]; no host sync.
    pub fn history(x: Exchange, a: HistArgs, recv: DevicePtr, stream: Stream) !void {
        const f = x.kernels.vtable.history orelse return error.Unsupported;
        try f(x.kernels.ptr, a, stream);
        try x.comm.allGather(a.send, recv, @as(usize, a.m) * a.row_bytes, .u8, stream);
    }

    /// One union block: this rank's M rows (phys: the block's list on the device) to send, recv [W, M, rb].
    pub fn unionBlock(x: Exchange, base: u64, row_bytes: u32, phys: DevicePtr, m: u32, send: DevicePtr, recv: DevicePtr, stream: Stream) !void {
        try x.kernels.vtable.gather(x.kernels.ptr, base, row_bytes, phys, m, send, stream);
        try x.comm.allGather(send, recv, @as(usize, m) * row_bytes, .u8, stream);
    }
};
