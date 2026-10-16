//! A stream's device caches: its page table over the attention pools, and each linear layer's conv window and state.

const std = @import("std");
const hip = @import("hip");
const view = @import("../model/view.zig");
const pages = @import("pages.zig");
const Allocator = std.mem.Allocator;

pub const LayerCache = union(enum) {
    /// The keys and values are in the pool's pages.
    full,
    linear: struct { conv: hip.DeviceBuffer, state: hip.DeviceBuffer },
};

/// Words of a descriptor before its layers: the positions it holds and the last kept final row.
pub const header_words = 2;

pub const Caches = struct {
    layers: []LayerCache,
    /// Positions the page table can name.
    total: usize,
    /// Kernel view: positions, last kept final row (0 until `setHidden`), two addresses a layer, then the page table.
    desc: hip.DeviceBuffer,
    pool: *pages.Pool,
    /// The page table as the device holds it: the pages of the first positions, each held once by this stream.
    table: std.ArrayList(u32) = .empty,
    /// Pages promised to this stream, not yet taken (the pool's reservation).
    promised: usize = 0,

    /// Device bytes the caches of `total` positions hold beside their pages: linear state and the descriptor.
    pub fn deviceBytes(m: *const view.Model, total: usize) usize {
        const s = m.spec;
        var linear: usize = 0;
        for (0..s.n_layers) |i| linear += @intFromBool(!s.full(i));
        const conv = (s.conv - 1) * view.convChannels(s) * 4;
        const state = s.value_heads * s.value_dim * s.key_dim * 4;
        return linear * (conv + state) + 8 * (header_words + 2 * s.n_layers + std.mem.alignForward(usize, pages.pagesFor(total), 2) / 2);
    }

    /// Caches for up to `total` positions with no pages yet; every linear byte zeroed.
    pub fn init(gpa: Allocator, pool: *pages.Pool, m: *const view.Model, total: usize) !Caches {
        const d = pool.d;
        const s = m.spec;
        const layers = try gpa.alloc(LayerCache, s.n_layers);
        var made: usize = 0;
        errdefer {
            for (layers[0..made]) |*l| free(l);
            gpa.free(layers);
        }
        for (layers, 0..) |*l, i| {
            if (s.full(i)) {
                l.* = .full;
            } else {
                var conv = try hip.DeviceBuffer.alloc(d, (s.conv - 1) * view.convChannels(s) * 4);
                errdefer conv.free();
                var state = try hip.DeviceBuffer.alloc(d, s.value_heads * s.value_dim * s.key_dim * 4);
                errdefer state.free();
                try conv.fill8(0);
                try state.fill8(0);
                l.* = .{ .linear = .{ .conv = conv, .state = state } };
            }
            made += 1;
        }
        const slots = std.mem.alignForward(usize, pages.pagesFor(total), 2);
        const words = try gpa.alloc(u64, header_words + 2 * s.n_layers + slots / 2);
        defer gpa.free(words);
        @memset(words, 0);
        var desc = try hip.DeviceBuffer.alloc(d, 8 * words.len);
        errdefer desc.free();
        words[0] = total;
        for (layers, 0..) |l, i| switch (l) {
            .full => words[header_words + 2 * i ..][0..2].* = .{ pool.keys[i].base(), pool.values[i].base() },
            .linear => |x| words[header_words + 2 * i ..][0..2].* = .{ x.conv.base(), x.state.base() },
        };
        try desc.upload(0, std.mem.sliceAsBytes(words));
        return .{ .layers = layers, .total = total, .desc = desc, .pool = pool };
    }

    /// Caches for `total` positions with every page taken now (a check or a timing, with no reservation to keep).
    pub fn initFull(gpa: Allocator, pool: *pages.Pool, m: *const view.Model, total: usize) !Caches {
        var c = try init(gpa, pool, m, total);
        errdefer c.deinit(gpa);
        _ = try c.grow(gpa, total);
        return c;
    }

    /// The stream's last kept final row lives at `ptr`: a round's keep copies it there.
    pub fn setHidden(c: *const Caches, ptr: u64) !void {
        try c.desc.upload(8, std.mem.asBytes(&ptr));
    }

    fn free(l: *LayerCache) void {
        switch (l.*) {
            .full => {},
            .linear => |*x| {
                x.conv.free();
                x.state.free();
            },
        }
    }

    /// Hands every page back (to the tree and the other streams that hold them, or to the pool) and frees the buffers.
    pub fn deinit(c: *Caches, gpa: Allocator) void {
        for (c.table.items) |id| c.pool.release(id);
        c.pool.ids.reserved -|= c.promised;
        c.table.deinit(gpa);
        c.desc.free();
        for (c.layers) |*l| free(l);
        gpa.free(c.layers);
        c.* = undefined;
    }

    /// Positions the pages cover.
    pub fn covered(c: *const Caches) usize {
        return @min(c.table.items.len * pages.tokens, c.total);
    }

    /// Bytes this stream holds: its linear state and every page it names (shared ones counted whole).
    pub fn held(c: *const Caches) usize {
        var n: usize = c.table.items.len * c.pool.pageBytes();
        for (c.layers) |l| switch (l) {
            .full => {},
            .linear => |x| n += x.conv.len + x.state.len,
        };
        return n;
    }

    fn put(c: *const Caches, first: usize, ids: []const u32) !void {
        const base = 8 * (header_words + 2 * c.layers.len);
        try c.desc.upload(base + 4 * first, std.mem.sliceAsBytes(ids));
    }

    /// Takes pages until `n` positions are covered, counted against any promise; new ids valid until the next call.
    pub fn grow(c: *Caches, gpa: Allocator, n: usize) ![]const u32 {
        const want = pages.pagesFor(@min(n, c.total));
        const first = c.table.items.len;
        if (want <= first) return c.table.items[first..];
        try c.table.ensureTotalCapacity(gpa, want);
        errdefer c.table.shrinkRetainingCapacity(first);
        while (c.table.items.len < want) c.table.appendAssumeCapacity(c.pool.take() orelse return error.OutOfPages);
        const taken = want - first;
        const own = @min(c.promised, taken);
        c.promised -= own;
        c.pool.ids.reserved -|= own;
        try c.put(first, c.table.items[first..]);
        return c.table.items[first..];
    }

    /// Sets the table from page `at` to `ids`; the caller's references move into it and replaced pages are released.
    pub fn set(c: *Caches, gpa: Allocator, at: usize, ids: []const u32) !void {
        try c.table.ensureTotalCapacity(gpa, at + ids.len);
        for (ids, at..) |id, i| {
            if (i < c.table.items.len) {
                if (c.table.items[i] == id) {
                    c.pool.release(id);
                    continue;
                }
                c.pool.release(c.table.items[i]);
                c.table.items[i] = id;
            } else {
                std.debug.assert(i == c.table.items.len);
                c.table.appendAssumeCapacity(id);
            }
        }
        try c.put(at, c.table.items[at..][0..ids.len]);
    }

    /// Copies each shared page holding positions `from .. to`; `out` gets its table index, old and new page.
    pub fn writable(c: *Caches, out: *std.ArrayList([3]u32), gpa: Allocator, from: usize, to: usize, stream: hip.abi.Stream) !void {
        var i = from / pages.tokens;
        const last = @min(pages.pagesFor(to), c.table.items.len);
        while (i < last) : (i += 1) {
            const old = c.table.items[i];
            if (c.pool.ids.refs[old] < 2) continue;
            const fresh = c.pool.take() orelse return error.OutOfPages;
            errdefer c.pool.release(fresh);
            try c.pool.copyPage(old, fresh, stream);
            c.table.items[i] = fresh;
            c.pool.release(old);
            try c.put(i, c.table.items[i..][0..1]);
            try out.append(gpa, .{ @intCast(i), old, fresh });
        }
    }

    /// The attention kernels' view of a full layer's pages.
    pub fn paged(c: *const Caches, m: *const view.Model, index: usize) hip.ops.Ops.Paged {
        const base = c.desc.base() + 8 * (header_words + 2 * c.layers.len);
        return .{ .k = c.pool.keys[index].base(), .v = c.pool.values[index].base(), .table = base, .kind = m.act, .kv_heads = m.spec.kv_heads, .d = m.spec.head_dim, .count = c.pool.count };
    }

    /// Bytes of the linear layers' state.
    pub fn linearBytes(c: *const Caches) usize {
        var n: usize = 0;
        for (c.layers) |l| switch (l) {
            .full => {},
            .linear => |x| n += x.conv.len + x.state.len,
        };
        return n;
    }
};

/// A stream's linear state at a prompt's page edge: each linear layer's conv window and DeltaNet state, in tree slots.
pub const Snapshots = struct {
    gpa: Allocator,
    d: *const hip.Runtime,
    slots: std.ArrayList(?hip.DeviceBuffer) = .empty,

    pub fn init(gpa: Allocator, d: *const hip.Runtime) Snapshots {
        return .{ .gpa = gpa, .d = d };
    }

    pub fn deinit(s: *Snapshots) void {
        for (s.slots.items) |*b| if (b.*) |*x| x.free();
        s.slots.deinit(s.gpa);
    }

    /// Frees every slot's buffer.
    pub fn clear(s: *Snapshots) void {
        for (s.slots.items) |*b| if (b.*) |*x| x.free();
        s.slots.clearRetainingCapacity();
    }

    /// The caches' linear state copied into slot `slot` (its buffer made on first use).
    pub fn take(s: *Snapshots, slot: u32, c: *const Caches, stream: hip.abi.Stream) !void {
        while (s.slots.items.len <= slot) try s.slots.append(s.gpa, null);
        const entry = &s.slots.items[slot];
        if (entry.* == null) entry.* = try hip.DeviceBuffer.alloc(s.d, c.linearBytes());
        var at: usize = 0;
        for (c.layers) |l| switch (l) {
            .full => {},
            .linear => |x| for ([_]hip.DeviceBuffer{ x.conv, x.state }) |b| {
                try hip.raw.copy(entry.*.?, at, b.base(), b.len, stream);
                at += b.len;
            },
        };
    }

    /// Slot `slot`'s linear state copied back into the caches.
    pub fn put(s: *const Snapshots, slot: u32, c: *Caches, stream: hip.abi.Stream) !void {
        const buf = (if (slot < s.slots.items.len) s.slots.items[slot] else null) orelse return error.NoSnapshot;
        var at: usize = 0;
        for (c.layers) |l| switch (l) {
            .full => {},
            .linear => |x| for ([_]hip.DeviceBuffer{ x.conv, x.state }) |b| {
                try hip.raw.copy(b, 0, buf.base() + at, b.len, stream);
                at += b.len;
            },
        };
    }

    /// Bytes the slots hold.
    pub fn held(s: *const Snapshots) usize {
        var n: usize = 0;
        for (s.slots.items) |b| n += if (b) |x| x.len else 0;
        return n;
    }
};
