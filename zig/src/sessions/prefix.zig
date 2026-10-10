//! Session entries by prefix: a trie over chained page digests, one node per distinct run of full pages a tag.

const std = @import("std");

pub const Digest = [16]u8;
pub const none = std.math.maxInt(u32);
const Blake = std.crypto.hash.blake2.Blake2b128;

/// Chained digests of the full pages of ids under a tag: out[j] determines tag and ids[0 .. page (j + 1)].
pub fn chain(tag: u32, ids: []const i32, page: u32, out: []Digest) void {
    var prev = root(tag);
    for (out, 0..) |*o, j| {
        o.* = step(prev, ids[j * page ..][0..page]);
        prev = o.*;
    }
}

/// The digest of a tag's empty prefix.
pub fn root(tag: u32) Digest {
    var h = Blake.init(.{});
    h.update("tf-sess-root");
    h.update(std.mem.asBytes(&tag));
    return h.finalResult();
}

/// The next page's digest.
pub fn step(prev: Digest, page_ids: []const i32) Digest {
    var h = Blake.init(.{});
    h.update(&prev);
    h.update(std.mem.sliceAsBytes(page_ids));
    return h.finalResult();
}

/// An entry's identity: its node's digest (tag and full pages), its length and tail.
pub fn entryKey(node: Digest, pos: u64, tail: []const i32) Digest {
    var h = Blake.init(.{});
    h.update("tf-sess-entry");
    h.update(&node);
    h.update(std.mem.asBytes(&pos));
    h.update(std.mem.sliceAsBytes(tail));
    return h.finalResult();
}

pub const Tier = enum { ram, any };

pub fn Index(comptime Extra: type) type {
    return struct {
        const Self = @This();

        pub const Node = struct {
            digest: Digest,
            parent: u32,
            depth: u32,
            first_child: u32 = none,
            next_sib: u32 = none,
            prev_sib: u32 = none,
            first_entry: u32 = none,
            /// entries in the subtree (this node's included): all, and those in RAM
            sub_all: u32 = 0,
            sub_ram: u32 = 0,
            /// RAM entries at this node itself
            own_ram: u32 = 0,
            /// the tokens of the page leading into this node, while sub_ram > 0
            tokens: ?[]i32 = null,
        };

        pub const Entry = struct {
            key: Digest,
            tag: u32,
            pos: u64,
            node: u32,
            tail: []i32,
            ram: bool,
            next_at: u32 = none,
            prev_at: u32 = none,
            live: bool = true,
            extra: Extra,
        };

        /// An entry inserted, or the one already there with the same key.
        pub const Placed = struct { id: u32, new: bool };

        gpa: std.mem.Allocator,
        page: u32,
        nodes: std.ArrayList(Node) = .empty,
        free_nodes: std.ArrayList(u32) = .empty,
        by_digest: std.AutoHashMapUnmanaged(Digest, u32) = .empty,
        entries: std.ArrayList(Entry) = .empty,
        free_entries: std.ArrayList(u32) = .empty,
        by_key: std.AutoHashMapUnmanaged(Digest, u32) = .empty,
        /// page-sized token blocks, reused
        blocks: std.ArrayList([]i32) = .empty,

        pub fn init(gpa: std.mem.Allocator, page: u32) Self {
            return .{ .gpa = gpa, .page = page };
        }

        pub fn deinit(x: *Self) void {
            for (x.nodes.items) |n| if (n.tokens) |t| x.gpa.free(t);
            for (x.blocks.items) |b| x.gpa.free(b);
            for (x.entries.items) |e| if (e.live) x.gpa.free(e.tail);
            x.nodes.deinit(x.gpa);
            x.free_nodes.deinit(x.gpa);
            x.by_digest.deinit(x.gpa);
            x.entries.deinit(x.gpa);
            x.free_entries.deinit(x.gpa);
            x.by_key.deinit(x.gpa);
            x.blocks.deinit(x.gpa);
            x.* = undefined;
        }

        pub fn get(x: *Self, e: u32) *Entry {
            return &x.entries.items[e];
        }

        pub fn node(x: *Self, n: u32) *Node {
            return &x.nodes.items[n];
        }

        pub fn lookup(x: *const Self, key: Digest) ?u32 {
            return x.by_key.get(key);
        }

        fn newNode(x: *Self, digest: Digest, parent: u32, depth: u32) !u32 {
            const id: u32 = if (x.free_nodes.pop()) |i| i else blk: {
                try x.nodes.append(x.gpa, undefined);
                break :blk @intCast(x.nodes.items.len - 1);
            };
            x.nodes.items[id] = .{ .digest = digest, .parent = parent, .depth = depth };
            try x.by_digest.put(x.gpa, digest, id);
            if (parent != none) {
                const p = &x.nodes.items[parent];
                x.nodes.items[id].next_sib = p.first_child;
                if (p.first_child != none) x.nodes.items[p.first_child].prev_sib = id;
                p.first_child = id;
            }
            return id;
        }

        fn takeBlock(x: *Self) ![]i32 {
            if (x.blocks.pop()) |b| return b;
            return x.gpa.alloc(i32, x.page);
        }

        /// Adds an entry for ids (chain: its full pages' digests), or returns the one with the same key.
        pub fn insert(x: *Self, tag: u32, ids: []const i32, digests: []const Digest, ram: bool, extra: Extra) !Placed {
            const full: u32 = @intCast(ids.len / x.page);
            std.debug.assert(digests.len >= full);
            const r = root(tag);
            var n = x.by_digest.get(r) orelse try x.newNode(r, none, 0);
            for (0..full) |j| {
                const d = digests[j];
                n = x.by_digest.get(d) orelse try x.newNode(d, n, @intCast(j + 1));
            }
            const r2 = try x.place(tag, n, ids.len, ids[full * x.page ..], extra);
            if (r2.new and ram) try x.setRam(r2.id, true, ids);
            return r2;
        }

        /// Adds a disk entry continuing from node anchor (none: the tag's root); suffix: tokens [depth x page, pos).
        pub fn insertFrom(x: *Self, tag: u32, anchor: u32, suffix: []const i32, extra: Extra) !Placed {
            var n = if (anchor != none) anchor else blk: {
                const r = root(tag);
                break :blk x.by_digest.get(r) orelse try x.newNode(r, none, 0);
            };
            const d0 = x.nodes.items[n].depth;
            const full: u32 = @intCast(suffix.len / x.page);
            for (0..full) |j| {
                const d = step(x.nodes.items[n].digest, suffix[j * x.page ..][0..x.page]);
                n = x.by_digest.get(d) orelse try x.newNode(d, n, d0 + @as(u32, @intCast(j)) + 1);
            }
            return x.place(tag, n, @as(u64, d0) * x.page + suffix.len, suffix[full * x.page ..], extra);
        }

        /// The node at depth on an entry's path (its first `depth` full pages).
        pub fn nodeAt(x: *Self, id: u32, depth: u32) ?u32 {
            var n = x.entries.items[id].node;
            while (n != none and x.nodes.items[n].depth > depth) n = x.nodes.items[n].parent;
            return if (n != none and x.nodes.items[n].depth == depth) n else null;
        }

        fn place(x: *Self, tag: u32, n: u32, pos: u64, tail_ids: []const i32, extra: Extra) !Placed {
            const key = entryKey(x.nodes.items[n].digest, pos, tail_ids);
            if (x.by_key.get(key)) |e| return .{ .id = e, .new = false };
            const tail = try x.gpa.dupe(i32, tail_ids);
            errdefer x.gpa.free(tail);
            const id: u32 = if (x.free_entries.pop()) |i| i else blk: {
                try x.entries.append(x.gpa, undefined);
                break :blk @intCast(x.entries.items.len - 1);
            };
            try x.by_key.put(x.gpa, key, id);
            const nd = &x.nodes.items[n];
            x.entries.items[id] = .{ .key = key, .tag = tag, .pos = pos, .node = n, .tail = tail, .ram = false, .next_at = nd.first_entry, .extra = extra };
            if (nd.first_entry != none) x.entries.items[nd.first_entry].prev_at = id;
            nd.first_entry = id;
            x.addPath(n, 1, 0);
            return .{ .id = id, .new = true };
        }

        fn addPath(x: *Self, from: u32, d_all: i32, d_ram: i32) void {
            var n = from;
            while (n != none) {
                const nd = &x.nodes.items[n];
                nd.sub_all = @intCast(@as(i64, nd.sub_all) + d_all);
                nd.sub_ram = @intCast(@as(i64, nd.sub_ram) + d_ram);
                n = nd.parent;
            }
        }

        /// Moves an entry in or out of RAM; going in needs its ids (to keep its path's tokens).
        pub fn setRam(x: *Self, id: u32, ram: bool, ids: ?[]const i32) !void {
            const e = &x.entries.items[id];
            if (e.ram == ram) return;
            if (ram) {
                const all = ids orelse return error.IdsNeeded;
                var n = e.node;
                while (n != none and x.nodes.items[n].depth > 0) : (n = x.nodes.items[n].parent) {
                    const nd = &x.nodes.items[n];
                    if (nd.tokens != null) continue;
                    const b = try x.takeBlock();
                    @memcpy(b, all[(nd.depth - 1) * x.page ..][0..x.page]);
                    nd.tokens = b;
                }
            }
            e.ram = ram;
            x.nodes.items[e.node].own_ram = if (ram) x.nodes.items[e.node].own_ram + 1 else x.nodes.items[e.node].own_ram - 1;
            x.addPath(e.node, 0, if (ram) 1 else -1);
            if (!ram) x.dropTokens(e.node);
        }

        fn dropTokens(x: *Self, from: u32) void {
            var n = from;
            while (n != none) : (n = x.nodes.items[n].parent) {
                const nd = &x.nodes.items[n];
                if (nd.sub_ram > 0) return;
                if (nd.tokens) |t| {
                    x.blocks.append(x.gpa, t) catch x.gpa.free(t);
                    nd.tokens = null;
                }
            }
        }

        /// Removes an entry and the nodes no entry needs any more.
        pub fn remove(x: *Self, id: u32) void {
            x.setRam(id, false, null) catch unreachable;
            const e = &x.entries.items[id];
            const n = e.node;
            if (e.prev_at != none) x.entries.items[e.prev_at].next_at = e.next_at else x.nodes.items[n].first_entry = e.next_at;
            if (e.next_at != none) x.entries.items[e.next_at].prev_at = e.prev_at;
            _ = x.by_key.remove(e.key);
            x.gpa.free(e.tail);
            e.live = false;
            x.free_entries.append(x.gpa, id) catch {};
            x.addPath(n, -1, 0);
            var m = n;
            while (m != none and x.nodes.items[m].sub_all == 0) {
                const nd = x.nodes.items[m];
                if (nd.parent != none) {
                    if (nd.prev_sib != none) x.nodes.items[nd.prev_sib].next_sib = nd.next_sib else x.nodes.items[nd.parent].first_child = nd.next_sib;
                    if (nd.next_sib != none) x.nodes.items[nd.next_sib].prev_sib = nd.prev_sib;
                }
                _ = x.by_digest.remove(nd.digest);
                x.free_nodes.append(x.gpa, m) catch {};
                m = nd.parent;
            }
        }

        /// The entry's tail matches the prompt there and the entry is shorter (its full pages matched by digest).
        fn tailOf(e: *const Entry, prompt: []const i32) bool {
            const at: usize = @intCast(e.pos - e.tail.len);
            return e.pos < prompt.len and std.mem.eql(i32, e.tail, prompt[at..][0..e.tail.len]);
        }

        /// The longest entry of tag whose ids strictly prefix prompt (RAM first on a tie); digests: the prompt's chain.
        pub fn find(x: *Self, tag: u32, prompt: []const i32, digests: []const Digest, tier: Tier) ?u32 {
            const full: u32 = @intCast(prompt.len / x.page);
            const r = x.by_digest.get(root(tag)) orelse return null;
            // the deepest node on the prompt's path: each prefix of an entry's path has a node: binary search
            var lo: u32 = 0;
            var hi: u32 = full;
            while (lo < hi) {
                const mid = lo + (hi - lo + 1) / 2;
                if (x.by_digest.contains(digests[mid - 1])) lo = mid else hi = mid - 1;
            }
            var n: u32 = if (lo == 0) r else x.by_digest.get(digests[lo - 1]).?;
            while (n != none) : (n = x.nodes.items[n].parent) {
                const nd = &x.nodes.items[n];
                if (nd.sub_all == 0 or (tier == .ram and nd.own_ram == 0)) continue;
                var best: u32 = none;
                var e = nd.first_entry;
                while (e != none) : (e = x.entries.items[e].next_at) {
                    const en = &x.entries.items[e];
                    if (tier == .ram and !en.ram) continue;
                    if (!tailOf(en, prompt)) continue;
                    if (best == none or en.pos > x.entries.items[best].pos or (en.pos == x.entries.items[best].pos and en.ram)) best = e;
                }
                if (best != none) return best;
            }
            return null;
        }

        /// A newer entry of id's chain holds every position it does (tier .ram: in RAM), so id need not be written.
        pub fn covered(x: *Self, id: u32, tier: Tier) bool {
            const e = &x.entries.items[id];
            const nd = &x.nodes.items[e.node];
            var f = nd.first_entry;
            while (f != none) : (f = x.entries.items[f].next_at) {
                const o = &x.entries.items[f];
                if (f == id or (tier == .ram and !o.ram)) continue;
                if (o.pos > e.pos and std.mem.startsWith(i32, o.tail, e.tail)) return true;
            }
            var c = nd.first_child;
            while (c != none) : (c = x.nodes.items[c].next_sib) {
                const ch = &x.nodes.items[c];
                const n = if (tier == .ram) ch.sub_ram else ch.sub_all;
                if (n == 0) continue;
                if (e.tail.len == 0) return true;
                if (ch.tokens) |t| if (std.mem.startsWith(i32, t, e.tail)) return true;
            }
            return false;
        }

        /// How many RAM entries extend a RAM entry (longer, same tag, its ids their prefix): branch points have more.
        pub fn extensions(x: *Self, id: u32) u32 {
            const e = &x.entries.items[id];
            const nd = &x.nodes.items[e.node];
            var n: u32 = 0;
            var f = nd.first_entry;
            while (f != none) : (f = x.entries.items[f].next_at) {
                const o = &x.entries.items[f];
                if (f != id and o.ram and o.pos > e.pos and std.mem.startsWith(i32, o.tail, e.tail)) n += 1;
            }
            var c = nd.first_child;
            while (c != none) : (c = x.nodes.items[c].next_sib) {
                const ch = &x.nodes.items[c];
                if (ch.sub_ram == 0) continue;
                if (std.mem.startsWith(i32, ch.tokens.?, e.tail)) n += ch.sub_ram;
            }
            return n;
        }

        /// A RAM entry no other RAM entry extends: a chat's newest state.
        pub fn isLeaf(x: *Self, id: u32) bool {
            return x.entries.items[id].ram and !x.covered(id, .ram);
        }

        /// A RAM leaf's older turns (the RAM entries it extends) into out; returns the chat's recency (newest `used`).
        pub fn members(x: *Self, leaf: u32, out: *std.ArrayList(u32), used: *const fn (*const Extra) u64) !u64 {
            const l = &x.entries.items[leaf];
            var recency = used(&l.extra);
            var below: u32 = none; // the path's node under n: its tokens are the page an entry at n must prefix
            var n = l.node;
            while (n != none) : ({
                below = n;
                n = x.nodes.items[n].parent;
            }) {
                const nd = &x.nodes.items[n];
                if (nd.own_ram == 0) continue;
                const page_ids: []const i32 = if (below == none) l.tail else x.nodes.items[below].tokens.?;
                var f = nd.first_entry;
                while (f != none) : (f = x.entries.items[f].next_at) {
                    const o = &x.entries.items[f];
                    if (f == leaf or !o.ram or o.pos >= l.pos) continue;
                    if (!std.mem.startsWith(i32, page_ids, o.tail)) continue;
                    try out.append(x.gpa, f);
                    recency = @max(recency, used(&o.extra));
                }
            }
            return recency;
        }

        /// A RAM entry's ids, rebuilt from its path's pages and tail, into out (len = pos).
        pub fn idsOf(x: *Self, id: u32, out: []i32) !void {
            const e = &x.entries.items[id];
            if (out.len != e.pos) return error.BadLength;
            const at = e.pos - e.tail.len;
            @memcpy(out[at..], e.tail);
            var n = e.node;
            while (n != none and x.nodes.items[n].depth > 0) : (n = x.nodes.items[n].parent) {
                const nd = &x.nodes.items[n];
                const t = nd.tokens orelse return error.NotInRam;
                @memcpy(out[(nd.depth - 1) * x.page ..][0..x.page], t);
            }
        }
    };
}

const T = Index(u64);

fn usedOf(e: *const u64) u64 {
    return e.*;
}

fn seq(gpa: std.mem.Allocator, n: usize, salt: i32) ![]i32 {
    const v = try gpa.alloc(i32, n);
    for (v, 0..) |*t, i| t.* = @as(i32, @intCast(i % 997)) * 3 + salt;
    return v;
}

fn add(x: *T, tag: u32, ids: []const i32, ram: bool, used: u64) !u32 {
    var d: [64]Digest = undefined;
    chain(tag, ids, x.page, d[0 .. ids.len / x.page]);
    return (try x.insert(tag, ids, d[0 .. ids.len / x.page], ram, used)).id;
}

test "longest strict prefix, ties, tags and coverage" {
    const gpa = std.testing.allocator;
    var x = T.init(gpa, 16);
    defer x.deinit();
    const chat = try seq(gpa, 200, 0);
    defer gpa.free(chat);
    const t1 = try add(&x, 7, chat[0..40], true, 1); // a prompt's replay point
    const t2 = try add(&x, 7, chat[0..72], true, 2); // a turn's end
    const t3 = try add(&x, 7, chat[0..128], false, 3); // on disk only, page aligned
    _ = try add(&x, 8, chat[0..100], true, 4); // another tag
    var d: [16]Digest = undefined;
    chain(7, chat[0..40], 16, d[0..2]);
    const again = try x.insert(7, chat[0..40], d[0..2], true, 9);
    try std.testing.expect(again.id == t1 and !again.new);
    chain(7, chat[0..150], 16, d[0..9]);
    try std.testing.expectEqual(@as(?u32, t3), x.find(7, chat[0..150], d[0..9], .any));
    try std.testing.expectEqual(@as(?u32, t2), x.find(7, chat[0..150], d[0..9], .ram));
    try std.testing.expectEqual(@as(?u32, t2), x.find(7, chat[0..128], d[0..8], .any)); // strict: 128 is not a prefix of 128
    chain(7, chat[0..72], 16, d[0..4]);
    try std.testing.expectEqual(@as(?u32, t1), x.find(7, chat[0..72], d[0..4], .any));
    var other = try gpa.dupe(i32, chat[0..150]);
    defer gpa.free(other);
    other[50] = -1; // diverges inside t2's tail page: t1 still prefixes it
    chain(7, other, 16, d[0..9]);
    try std.testing.expectEqual(@as(?u32, t1), x.find(7, other, d[0..9], .any));
    // t3 extends t2 on disk only: its page's tokens are gone, so a tail there is undecided (the store marks covered_by)
    try std.testing.expect(x.covered(t1, .ram) and !x.covered(t2, .any) and !x.covered(t2, .ram) and !x.covered(t3, .any));
    const t4 = try add(&x, 7, chat[0..64], false, 5); // page aligned: any deeper entry covers it
    try std.testing.expect(x.covered(t4, .any) and x.covered(t4, .ram));
    x.remove(t4);
    try std.testing.expect(x.isLeaf(t2) and !x.isLeaf(t1));
    try std.testing.expectEqual(@as(u32, 1), x.extensions(t1));
    try std.testing.expectEqual(@as(u32, 0), x.extensions(t2));
    var m: std.ArrayList(u32) = .empty;
    defer m.deinit(gpa);
    try std.testing.expectEqual(@as(u64, 2), try x.members(t2, &m, usedOf));
    try std.testing.expectEqualSlices(u32, &.{t1}, m.items);
    const ids = try gpa.alloc(i32, 72);
    defer gpa.free(ids);
    try x.idsOf(t2, ids);
    try std.testing.expectEqualSlices(i32, chat[0..72], ids);
    // going to disk drops the path's tokens once no RAM entry needs them; removal prunes the nodes
    try x.setRam(t2, false, null);
    try x.setRam(t1, false, null);
    try std.testing.expectError(error.NotInRam, x.idsOf(t2, ids));
    x.remove(t3);
    x.remove(t2);
    x.remove(t1);
    chain(7, chat[0..150], 16, d[0..9]);
    try std.testing.expectEqual(@as(?u32, null), x.find(7, chat[0..150], d[0..9], .any));
    try std.testing.expectEqual(@as(u32, 1 + 6), x.by_digest.count()); // tag 8's root + its 6 full pages
}
