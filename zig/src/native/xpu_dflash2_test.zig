//! The XPU DFlash2 adapter through Drafted against the lane target's private drafting path, on recorded rounds.
const std = @import("std");
const lanes = @import("lanes");
const be = lanes.backend;
const Stream = lanes.Stream;
const Drafted = lanes.drafted.Drafted;
const ad = @import("xpu_dflash2.zig");

const gpa = std.testing.allocator;
pub const BASE: u64 = 0x100000; // the fake taps buffer's address: the drafter sees addresses only
pub const ROW: u64 = 10; // its bytes a row

/// A drafter that records every call that moves its contexts or its lattice, with its arguments.
pub const Fake = struct {
    log: std.ArrayList(u64) = .empty,
    ends: [4]u32 = @splat(0),
    cur: u32 = 0,
    cand: [16][16]u32 = undefined,
    multi: [16][16][16]u32 = undefined,

    fn note(f: *Fake, xs: []const u64) void {
        f.log.appendSlice(gpa, xs) catch @panic("oom");
    }
    pub fn use(f: *Fake, slot: u32) !void {
        f.cur = slot;
    }
    pub fn end(f: *Fake) u32 {
        return f.ends[f.cur];
    }
    pub fn skip(f: *Fake, n: u32) void {
        f.note(&.{ 1, f.cur, n });
        f.ends[f.cur] += n;
    }
    pub fn addTaps(f: *Fake, taps: *anyopaque, n: u32) !void {
        f.note(&.{ 2, f.cur, @intFromPtr(taps), n });
        f.ends[f.cur] += n;
    }
    pub fn absorbMulti(f: *Fake, items: []const M.Absorb) !void {
        if (items.len == 0) return;
        f.note(&.{ 3, items.len });
        for (items) |it| {
            f.note(&.{ it.slot, @intFromPtr(it.taps), it.n, if (it.rows) |r| r.len else 99 });
            if (it.rows) |r| for (r) |x| f.note(&.{x});
            f.ends[it.slot] += it.n;
        }
    }
    pub fn latticeMulti(f: *Fake, items: []const M.Block, depth: u32) !void {
        f.note(&.{ 4, items.len, depth });
        for (items, 0..) |it, b| {
            f.note(&.{ it.slot, it.pending });
            for (0..16) |d| for (0..16) |j| {
                f.multi[b][d][j] = @intCast((it.slot * 7919 + it.pending * 31 + d * 17 + j * 3) % 1000);
            };
        }
    }
    pub fn take(f: *Fake, i: usize) void {
        f.note(&.{ 5, i });
        f.cand = f.multi[i];
    }
    pub fn chain(f: *const Fake, pending: u32, count: u32, out: []u32) void {
        for (out[0..count], 0..) |*o, d| o.* = f.cand[d][0] + pending;
    }
    pub fn tree(f: *Fake, pending: u32, n: u32, noise: ?*const [16][16]f64, temp: f64, tokens: []u32, parents: []i32) u32 {
        var bits: u64 = @as(u64, @bitCast(temp)) +% pending;
        if (noise) |z| for (0..n) |d| for (0..16) |j| {
            bits +%= @bitCast(z[d % 5][j]); // rows past the lattice's depth are undefined
        };
        f.note(&.{ 6, pending, n, @intFromBool(noise != null), bits });
        for (tokens[0..n], parents[0..n], 0..) |*t, *p, i| {
            t.* = f.cand[i % 7][@intCast((bits >> @intCast(i % 32)) % 16)];
            p.* = @as(i32, @intCast(i)) - 1;
        }
        return n;
    }
};

pub const M = struct {
    pub const TOPK = 16;
    pub const WIN = 2047;
    pub const Block = struct { slot: u32, pending: u32 };
    pub const Absorb = struct { slot: u32, taps: *anyopaque, n: u32, rows: ?[]const u32 = null };
    pub const Drafter = Fake;
};

pub const Chunk = struct { start: u64, rows: u32 };
const Tail = struct { start: u64, rows: u32 };

/// The lane target as the adapter and Drafted see it: slots, the prompt pass's chunks, the last verify's windows.
pub const Host = struct {
    pub const max_slots = 4;
    pub const tap_row = ROW;
    pub const draft_step_ms = 0.6;
    pub const tap_layers = [_]u32{ 5, 19, 33, 47, 61 };
    block: u32 = 8,
    nodes: u32 = 15,
    owners: [4]?*Stream = @splat(null),
    chunks: []const Chunk = &.{},
    tail: ?Tail = null,
    wslots: [16]u32 = undefined,
    wfirst: [16]u32 = undefined,
    widths: [16]u32 = undefined,
    tapped: usize = 0,

    pub fn slotOf(h: *const Host, s: *const Stream) ?u32 {
        for (h.owners, 0..) |o, i| if (o == s) return @intCast(i);
        return null;
    }
    pub fn promptTail(h: *const Host, slot: u32) ?Tail {
        _ = slot;
        return h.tail;
    }
    fn cast(p: *anyopaque) *Host {
        return @ptrCast(@alignCast(p));
    }
    fn feat(offset: u64, rows: u32) be.Features {
        return .{ .buffer = BASE, .offset = offset, .rows = rows, .row_bytes = ROW, .space = .device, .dtype = .bf16 };
    }
    fn prefillFeatures(p: *anyopaque, s: *Stream, _: []const u32, sink: be.FeatureSink) anyerror!void {
        const h = cast(p);
        h.tapped = 0;
        for (h.chunks) |c| {
            h.tail = .{ .start = c.start, .rows = c.rows };
            try sink.take(sink.ptr, s, feat(0, c.rows), c.start);
        }
    }
    fn features(p: *anyopaque, s: *Stream, _: []const u32, start: u64, count: u32) anyerror!be.Features {
        const h = cast(p);
        const slot = h.slotOf(s).?;
        for (h.wslots[0..h.tapped], 0..) |sl, i| if (sl == slot) return feat(h.wfirst[i] * ROW, count);
        const t = h.tail orelse return error.FeaturesGone;
        if (start < t.start or start + count > t.start + t.rows) return error.FeaturesGone;
        return feat((start - t.start) * ROW, count);
    }
    fn verify(p: *anyopaque, windows: []const be.Window, _: []be.Verified) anyerror!void {
        const h = cast(p);
        h.tail = null;
        var at: u32 = 0;
        for (windows, 0..) |w, i| {
            h.wslots[i] = h.slotOf(w.stream).?;
            h.wfirst[i] = at;
            h.widths[i] = @intCast(w.rows());
            at += h.widths[i];
        }
        h.tapped = windows.len;
    }
    fn unused(_: *anyopaque, _: *Stream) anyerror!void {
        return error.Unused;
    }
    fn first(_: *anyopaque, _: *Stream, _: u64) anyerror!u64 {
        return error.Unused;
    }
    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.Unused;
    }
    fn read(_: *anyopaque, _: u64) anyerror!u32 {
        return error.Unused;
    }
    fn keep(_: *anyopaque, _: []const be.Window, _: []const []const u32) anyerror!void {}
    fn draft(_: *anyopaque, _: []const be.DraftRequest) anyerror!void {}
    fn release(_: *anyopaque, _: *Stream) void {}
    fn backend(h: *Host) be.Backend {
        return .{ .ptr = h, .vtable = &.{ .prefill = unused, .prefill_features = prefillFeatures, .features = features, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }
};

/// main's private path (native/xpu_lanes.zig before the move), its drafter calls ported as they were.
pub const Private = struct {
    d: *Fake,
    h: *Host,
    absorbed: [16]bool = @splat(false),
    count: usize = 0,
    chain: [4][16]u32 = undefined,
    chain_par: [4][16]i32 = undefined,
    chain_n: [4]u32 = @splat(0),

    pub fn prefill(p: *Private, s: *Stream, slot: u32, at: u64) !void {
        var skip: u64 = (s.prompt_len -| M.WIN) -| at;
        for (p.h.chunks) |c| { // onTaps, a call per chunk
            try p.d.use(slot);
            const k = @min(c.rows, skip);
            skip -= k;
            p.d.skip(@intCast(k));
            if (k < c.rows) try p.d.addTaps(@ptrFromInt(BASE + k * ROW), @intCast(c.rows - k));
        }
        p.count = 0;
    }
    fn verify(p: *Private, n: usize) void {
        p.count = n;
        @memset(&p.absorbed, false);
    }
    fn absorb(p: *Private, which: []const usize, paths: []const []const u32) !void {
        var items: [16]M.Absorb = undefined;
        for (which, paths, items[0..which.len]) |i, path, *it| {
            var straight = true;
            for (path, 0..) |row, j| straight = straight and row == j;
            it.* = .{ .slot = p.h.wslots[i], .taps = @ptrFromInt(BASE + p.h.wfirst[i] * ROW), .n = @intCast(path.len), .rows = if (straight) null else path };
            p.absorbed[i] = true;
        }
        try p.d.absorbMulti(items[0..which.len]);
    }
    fn keep(p: *Private, paths: []const []const u32) !void {
        var which: [16]usize = undefined;
        var ps: [16][]const u32 = undefined;
        var n: usize = 0;
        for (paths, 0..) |path, i| if (!p.absorbed[i]) {
            which[n] = i;
            ps[n] = path;
            n += 1;
        };
        try p.absorb(which[0..n], ps[0..n]);
        @memset(&p.absorbed, false);
        p.count = 0;
    }
    pub fn draft(p: *Private, requests: []const be.DraftRequest) !void {
        const d = p.d;
        var which: [16]usize = undefined;
        var ps: [16][]const u32 = undefined;
        var na: usize = 0;
        var items: [16]M.Block = undefined;
        var asked: [16]u32 = undefined;
        var reqs: [16]*const be.DraftRequest = undefined;
        var ni: usize = 0;
        var depth: u32 = 0;
        for (requests) |*q| {
            const slot = p.h.slotOf(q.stream).?;
            p.chain_n[slot] = 0;
            if (q.rows) |rows| if (p.count > 0) {
                const i = std.mem.indexOfScalar(u32, p.h.wslots[0..p.count], slot).?;
                if (!p.absorbed[i]) {
                    which[na] = i;
                    ps[na] = rows;
                    na += 1;
                }
            };
            if (q.depth == 0 or ni == items.len) continue;
            items[ni] = .{ .slot = slot, .pending = if (q.rows != null) q.follow[q.follow.len - 1] else q.first.?.value };
            asked[ni] = @min(q.depth, p.h.nodes);
            reqs[ni] = q;
            depth = @max(depth, @min(asked[ni], p.h.block - 1));
            ni += 1;
        }
        try p.absorb(which[0..na], ps[0..na]);
        if (ni == 0) return;
        for (requests) |q| if (q.depth > 0) {
            try d.use(p.h.slotOf(q.stream).?);
            if (d.end() + 1 != q.position) return error.BadDraftPosition;
        };
        try d.latticeMulti(items[0..ni], depth);
        for (items[0..ni], asked[0..ni], reqs[0..ni], 0..) |it, n, q, i| {
            d.take(i);
            var noise: [16][M.TOPK]f64 = undefined;
            const sp = q.stream.sampling;
            if (sp) |x| for (0..depth) |dd| for (0..M.TOPK) |j| {
                noise[dd][j] = -@log(-@log(lanes.sampling.uniform(x.seed, q.position + dd, d.cand[dd][j])));
            };
            if (n < ad.TREE_MIN) {
                d.chain(it.pending, n, p.chain[it.slot][0..n]);
                for (p.chain_par[it.slot][0..n], 0..) |*x, j| x.* = @as(i32, @intCast(j)) - 1;
                p.chain_n[it.slot] = n;
                continue;
            }
            p.chain_n[it.slot] = d.tree(it.pending, n, if (sp != null) &noise else null, if (sp) |x| x.temperature else 1.0, p.chain[it.slot][0..n], p.chain_par[it.slot][0..n]);
        }
    }
    fn tree(p: *Private, slot: u32, out: *std.ArrayList(u64)) !void {
        for (p.chain[slot][0..p.chain_n[slot]], p.chain_par[slot][0..p.chain_n[slot]]) |t, par| try out.appendSlice(gpa, &.{ t, @bitCast(@as(i64, par)) });
        p.chain_n[slot] = 0;
    }
};

/// One recorded run: the same prompts, verifies, kept paths and draft requests through both paths.
const Run = struct {
    fake: Fake = .{},
    host: Host = .{},
    trees: std.ArrayList(u64) = .empty,

    fn deinit(r: *Run) void {
        r.fake.log.deinit(gpa);
        r.trees.deinit(gpa);
    }
};

const Step = union(enum) {
    prefill: struct { s: usize, slot: u32, chunks: []const Chunk },
    verify: []const struct { s: usize, rows: u32 },
    keep: []const []const u32,
    draft: []const struct { s: usize, rows: ?[]const u32, depth: u32, start: u64, first: u32 = 0 },
    tree: usize,
};

pub fn request(ss: []const *Stream, q: anytype) be.DraftRequest {
    const s = ss[q.s];
    const follow: []const u32 = if (q.rows) |r| s.prompt()[0..r.len] else &.{};
    const cache = q.start + if (q.rows) |r| r.len else 0;
    return .{ .stream = s, .follow = follow, .first = if (q.rows == null) .{ .value = q.first } else null, .rows = q.rows, .start = q.start, .position = cache + 1, .depth = q.depth };
}

fn windowsOf(ss: []const *Stream, list: anytype, out: []be.Window, tokens: []const u32, pos: []const u64) []be.Window {
    for (list, out[0..list.len]) |w, *o| o.* = .{ .stream = ss[w.s], .pending = 3, .held = 0, .tokens = tokens[0 .. w.rows - 1], .parents = null, .positions = pos[0..w.rows] };
    return out[0..list.len];
}

fn private(r: *Run, ss: []const *Stream, steps: []const Step) !void {
    var p: Private = .{ .d = &r.fake, .h = &r.host };
    var wins: [16]be.Window = undefined;
    const toks: [16]u32 = @splat(4);
    const pos: [16]u64 = @splat(0);
    for (steps) |st| switch (st) {
        .prefill => |x| {
            r.host.owners[x.slot] = ss[x.s];
            r.host.chunks = x.chunks;
            r.host.tail = .{ .start = x.chunks[x.chunks.len - 1].start, .rows = x.chunks[x.chunks.len - 1].rows };
            try p.prefill(ss[x.s], x.slot, 0);
        },
        .verify => |list| {
            try Host.verify(&r.host, windowsOf(ss, list, &wins, &toks, &pos), &.{});
            p.verify(list.len);
        },
        .keep => |paths| try p.keep(paths),
        .draft => |list| {
            var reqs: [16]be.DraftRequest = undefined;
            for (list, reqs[0..list.len]) |q, *o| o.* = request(ss, q);
            try p.draft(reqs[0..list.len]);
        },
        .tree => |i| try p.tree(r.host.slotOf(ss[i]).?, &r.trees),
    };
}

fn adapted(r: *Run, ss: []const *Stream, steps: []const Step) !void {
    var a: ad.Adapter(M, Host) = .{ .d = &r.fake, .host = &r.host };
    var x = try Drafted.init(gpa, r.host.backend(), a.drafter());
    defer x.deinit();
    const b = x.backend();
    var wins: [16]be.Window = undefined;
    const toks: [16]u32 = @splat(4);
    const pos: [16]u64 = @splat(0);
    var outs: [16]be.Verified = undefined;
    for (steps) |st| switch (st) {
        .prefill => |p| {
            r.host.owners[p.slot] = ss[p.s];
            r.host.chunks = p.chunks;
            try b.prefill(ss[p.s]);
        },
        .verify => |list| try b.vtable.verify(b.ptr, windowsOf(ss, list, &wins, &toks, &pos), outs[0..list.len]),
        .keep => |paths| try b.vtable.keep(b.ptr, wins[0..paths.len], paths),
        .draft => |list| {
            var reqs: [16]be.DraftRequest = undefined;
            for (list, reqs[0..list.len]) |q, *o| o.* = request(ss, q);
            try b.vtable.draft(b.ptr, reqs[0..list.len]);
        },
        .tree => |i| {
            const held = (try b.vtable.tree.?(b.ptr, ss[i], gpa)).?;
            defer gpa.free(held.tokens.?);
            defer gpa.free(held.parents.?);
            for (held.tokens.?, held.parents.?) |t, par| try r.trees.appendSlice(gpa, &.{ t, @bitCast(@as(i64, par)) });
        },
    };
}

test "draft sources: XPU DFlash2 through Drafted makes main's drafter calls, rows and drafts" {
    var prompt: [2300]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i % 97);
    var a = try Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 64 });
    defer a.deinit(gpa);
    var b = try Stream.init(gpa, .{ .id = "b", .prompt = prompt[0..40], .max_new = 64, .sampling = .{ .seed = 7, .temperature = 0.8 } });
    defer b.deinit(gpa);
    var c = try Stream.init(gpa, .{ .id = "c", .prompt = prompt[0..300], .max_new = 64 });
    defer c.deinit(gpa);
    const ss = [_]*Stream{ &a, &b, &c };
    const tree_path = [_]u32{ 0, 1, 3 };
    const all = [_]u32{ 0, 1, 2, 3, 4, 5, 6, 7 };
    const steps = [_]Step{
        // a long prompt (rows further than WIN skipped), its last row inside its last chunk
        .{ .prefill = .{ .s = 0, .slot = 3, .chunks = &.{ .{ .start = 0, .rows = 1024 }, .{ .start = 1024, .rows = 1024 }, .{ .start = 2048, .rows = 252 } } } },
        .{ .draft = &.{.{ .s = 0, .rows = null, .depth = 7, .start = 2300, .first = 11 }} },
        .{ .tree = 0 },
        .{ .prefill = .{ .s = 1, .slot = 2, .chunks = &.{.{ .start = 0, .rows = 40 }} } },
        .{ .draft = &.{.{ .s = 1, .rows = null, .depth = 3, .start = 40, .first = 12 }} },
        .{ .tree = 1 },
        // the prompt's last row alone in its chunk
        .{ .prefill = .{ .s = 2, .slot = 1, .chunks = &.{ .{ .start = 0, .rows = 299 }, .{ .start = 299, .rows = 1 } } } },
        .{ .draft = &.{.{ .s = 2, .rows = null, .depth = 7, .start = 300, .first = 13 }} },
        .{ .tree = 2 },
        // a shared round: drafts before keep; a tree path, an absorb-only stream, a whole window
        .{ .verify = &.{ .{ .s = 0, .rows = 8 }, .{ .s = 1, .rows = 4 }, .{ .s = 2, .rows = 8 } } },
        .{ .draft = &.{ .{ .s = 0, .rows = &tree_path, .depth = 7, .start = 2300 }, .{ .s = 1, .rows = all[0..2], .depth = 0, .start = 40 }, .{ .s = 2, .rows = &all, .depth = 7, .start = 300 } } },
        .{ .keep = &.{ &tree_path, all[0..2], &all } },
        .{ .tree = 0 },
        .{ .tree = 2 },
        // a lone round: keep first, then a chain
        .{ .verify = &.{.{ .s = 0, .rows = 8 }} },
        .{ .keep = &.{all[0..3]} },
        .{ .draft = &.{.{ .s = 0, .rows = all[0..3], .depth = 4, .start = 2303 }} },
        .{ .tree = 0 },
        // all rows kept (no keep call), then a sampled and a greedy stream in one batch, each keyed as its own
        .{ .verify = &.{ .{ .s = 1, .rows = 1 }, .{ .s = 0, .rows = 5 } } },
        .{ .draft = &.{ .{ .s = 1, .rows = all[0..1], .depth = 6, .start = 42 }, .{ .s = 0, .rows = all[0..5], .depth = 7, .start = 2306 } } },
        .{ .tree = 1 },
        .{ .tree = 0 },
        // exactly TREE_MIN drafts: the chain/tree switch at its boundary
        .{ .verify = &.{.{ .s = 2, .rows = 3 }} },
        .{ .draft = &.{.{ .s = 2, .rows = all[0..3], .depth = ad.TREE_MIN, .start = 308 }} },
        .{ .tree = 2 },
    };
    var old: Run = .{};
    defer old.deinit();
    try private(&old, &ss, &steps);
    var new: Run = .{};
    defer new.deinit();
    try adapted(&new, &ss, &steps);
    try std.testing.expect(old.fake.log.items.len > 60);
    try std.testing.expectEqualSlices(u64, old.fake.log.items, new.fake.log.items);
    try std.testing.expect(old.trees.items.len > 40);
    try std.testing.expectEqualSlices(u64, old.trees.items, new.trees.items);
    try std.testing.expectEqualSlices(u32, &old.fake.ends, &new.fake.ends);
}

test "draft sources: XPU DFlash2 through Drafted keeps the lane target's round facts" {
    const L = @import("xpu_lanes.zig").Lanes;
    const xq = @import("xpu_qwen27");
    var b: L = undefined;
    b.drafter = @import("draft_sources_check.zig").present(@TypeOf(b.drafter.?));
    b.block = 8;
    b.nodes = 15;
    b.timed = 0;
    b.slots = 4;
    var a: ad.Adapter(xq.drafter, L) = .{ .d = b.drafter.?, .host = &b };
    var x = try Drafted.init(gpa, b.backend(), a.drafter());
    defer x.deinit();
    try std.testing.expectEqualDeep(b.facts(), ad.facts(&x, b.facts()));
    _ = &Drafted.backend; // and the adapter's operations compile on the real drafter
    _ = &@TypeOf(a).drafter;
}

test "draft sources: XPU DFlash2 refuses a gap in a prompt's rows and more held drafts than it grew" {
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i);
    var a = try Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 8 });
    defer a.deinit(gpa);
    var c = try Stream.init(gpa, .{ .id = "c", .prompt = &prompt, .max_new = 8 });
    defer c.deinit(gpa);
    var r: Run = .{};
    defer r.deinit();
    var dflash: ad.Adapter(M, Host) = .{ .d = &r.fake, .host = &r.host };
    var x = try Drafted.init(gpa, r.host.backend(), dflash.drafter());
    defer x.deinit();
    const b = x.backend();
    r.host.owners[0] = &a;
    r.host.chunks = &.{ .{ .start = 0, .rows = 10 }, .{ .start = 20, .rows = 20 } }; // rows 10..19 never given
    try std.testing.expectError(error.DraftContextGap, b.prefill(&a));
    r.host.owners[1] = &c;
    r.host.chunks = &.{ .{ .start = 0, .rows = 30 }, .{ .start = 30, .rows = 10 } };
    try b.prefill(&c);
    const ss = [_]*Stream{&c};
    try b.vtable.draft(b.ptr, &.{request(&ss, .{ .s = 0, .rows = @as(?[]const u32, null), .depth = 3, .start = 40, .first = 12 })});
    var out: [4]u32 = undefined;
    try b.vtable.held.?(b.ptr, &c, out[0..3]); // a chain of the 3 asked
    try std.testing.expectError(error.DraftHeldFewer, b.vtable.held.?(b.ptr, &c, out[0..4]));
}

test "draft sources: XPU lane target refuses another slot's prompt tail and drafts without Drafted" {
    const L = @import("xpu_lanes.zig").Lanes;
    const xq = @import("xpu_qwen27");
    var prompt: [40]u32 = undefined;
    for (&prompt, 0..) |*t, i| t.* = @intCast(i);
    var a = try Stream.init(gpa, .{ .id = "a", .prompt = &prompt, .max_new = 8 });
    defer a.deinit(gpa);
    var c = try Stream.init(gpa, .{ .id = "c", .prompt = &prompt, .max_new = 8 });
    defer c.deinit(gpa);
    var runner: xq.Runner = undefined;
    runner.acts.taps = @ptrFromInt(0x10000);
    var b: L = undefined;
    b.runner = &runner;
    b.drafter = @import("draft_sources_check.zig").present(@TypeOf(b.drafter.?));
    b.slots = 4;
    b.owners = @splat(null);
    b.owners[0] = &a;
    b.owners[1] = &c;
    b.tapped = 0;
    b.tail = .{ .slot = 1, .start = 0, .rows = 40 }; // c's prompt pass ran last
    const t = b.backend();
    const got = try t.vtable.features.?(t.ptr, &c, &.{}, 39, 1);
    try std.testing.expectEqual(@as(u64, 39 * L.tap_row), got.offset);
    try std.testing.expectError(error.FeaturesGone, t.vtable.features.?(t.ptr, &a, &.{}, 39, 1));
    // through Drafted: a's first draft after c's prompt pass refuses rather than read c's rows
    var dflash: ad.Adapter(xq.drafter, L) = .{ .d = b.drafter.?, .host = &b };
    var x = try Drafted.init(gpa, t, dflash.drafter());
    defer x.deinit();
    try x.lanes.put(gpa, &a, .{ .opened = true }); // context exists; only the target's feature ownership is invalid
    const w = x.backend();
    const first: be.DraftRequest = .{ .stream = &a, .follow = &.{}, .first = .{ .value = 9 }, .rows = null, .start = 40, .position = 41, .depth = 3 };
    try std.testing.expectError(error.FeaturesGone, w.vtable.draft(w.ptr, &.{first}));
    // the lane target alone holds no drafts: asked for any, it refuses; absorb-only requests pass
    var none = first;
    none.depth = 0;
    try t.vtable.draft(t.ptr, &.{none});
    var one = first;
    one.depth = 1; // the least depth that asks for drafts
    try std.testing.expectError(error.DraftNotAttached, t.vtable.draft(t.ptr, &.{ none, one }));
}
