//! The driver over a fake engine on loopback rings: 2 and 3 stages give one engine's bytes; planted faults are caught.
const std = @import("std");
const driver = @import("driver.zig");
const boot = @import("boot.zig");
const plan = @import("plan.zig");
const payload = @import("payload.zig");
const ring = @import("ring.zig");
const net = @import("net.zig");
const t = std.testing;

const layers = 6;
const hidden = 4;
const planes = 2;
const most_rows = 16;
const most_window = 8;
const parallel = 2;
const context = 256;
const most = 3;
const pipe_streams = 4;
const pipe_steps = 6;

/// A planted fault on one stage: a layer short of its range, a rollback of the position only, or no rollback.
const Plant = enum { none, short_range, no_restore, no_rollback };

/// An engine whose layer `l` mixes `l` and a per-stream running state into every row, so order and state both show.
const Fake = struct {
    pub const Engine = struct {
        gpa: std.mem.Allocator,
        first: u32,
        end: u32,
        plant: Plant = .none,
        /// Rank 0's head: every window row, and a prompt part's last row, `hidden` values each.
        out: std.ArrayList(u32) = .empty,
        h: [most_rows * hidden]u32 = undefined,
        r: [most_rows * hidden]u32 = undefined,
        io: [planes * most_rows * hidden * 4]u8 = undefined,
    };

    /// A stream's state at each layer (its own layers only change), and a window's states row by row.
    pub const Cache = struct {
        state: [layers]u32 = @splat(0),
        len: u32 = 0,
        last: ?driver.Window = null,
        before: [layers]u32 = @splat(0),
        rows: [most_window][layers]u32 = undefined,
    };

    /// One stream's rows in a round's forward, as the driver builds it.
    pub const Part = struct { cache: *Cache, rows: u32 };

    /// Every stage of one placement reports this version, on this backend.
    pub const version = "tensorfold fake stage";
    pub const backend: payload.Backend = .metal;
    /// The most streams a round carries.
    pub const most_streams = pipe_streams;

    /// The ring's boundary: two planes of `hidden` 4-byte values a row.
    pub fn boundary(_: *const Engine) boot.Boundary {
        return .{ .hidden = hidden, .row_dtype = .f32, .planes = planes, .draft_width = 0 };
    }

    /// The shapes every stage runs.
    pub fn shapes(p: u32, c: u32) boot.Shapes {
        return .{ .prompt_chunk = most_rows, .max_rows = most_rows, .parallel = p, .context = c };
    }

    pub fn kvDtype(_: *const Engine) payload.Dtype {
        return .bf16;
    }

    /// A slot's cache, made once at boot and closed at the end.
    pub fn open(_: std.mem.Allocator, _: *Engine, _: u32) !Cache {
        return .{};
    }

    pub fn close(_: std.mem.Allocator, _: *Cache) void {}

    /// A stream opened on the slot: its state zeroed.
    pub fn reset(_: *Engine, c: *Cache) !void {
        c.* = .{};
    }

    /// The rows the stream holds.
    pub fn position(c: *const Cache) u32 {
        return c.len;
    }

    /// The stream's last window, while no plan has settled it.
    pub fn window(c: *const Cache) ?driver.Window {
        return c.last;
    }

    /// The last window kept to its first `accepted` rows: position and state both.
    pub fn rollback(e: *Engine, c: *Cache, accepted: u32) !void {
        if (e.plant == .no_rollback) return;
        const w = c.last.?;
        if (e.plant != .no_restore) c.state = if (accepted == 0) c.before else c.rows[accepted - 1];
        c.len = w.start + accepted;
        c.last = null;
    }

    /// Rank 0's state between a round's layers and its head: nothing here, the rows come back by the hop.
    pub const Held = struct {};

    pub fn hold(_: std.mem.Allocator, _: *Engine) !Held {
        return .{};
    }

    pub fn unhold(_: std.mem.Allocator, _: *Held) void {}

    /// Rank 0 with rounds in flight: the embedding and its layers, and the rows on to the next stage.
    pub fn issue(e: *Engine, parts: []const Part, ids: []const u32, mode: plan.Mode, hop: driver.Hop, _: *Held) !void {
        const bytes = try runLayers(e, parts, ids, mode, null);
        pack(e, bytes);
        try hop.give(hop.ctx, bytes.len / (planes * hidden * 4), bytes);
    }

    /// Rank 0: an issued round's rows back from the last stage, then the head over them.
    pub fn complete(e: *Engine, parts: []const Part, mode: plan.Mode, hop: driver.Hop, _: *Held) !void {
        var n: usize = 0;
        for (parts) |p| n += p.rows;
        const bytes = e.io[0 .. planes * n * hidden * 4];
        try hop.take(hop.ctx, n, bytes);
        unpack(e, bytes);
        try head(e, parts, mode);
    }

    /// This stage's layers over the round's rows; rank 0 embeds and runs the head, the hop carries the rows between.
    pub fn forward(e: *Engine, parts: []const Part, ids: []const u32, mode: plan.Mode, hop: ?driver.Hop) !void {
        const bytes = try runLayers(e, parts, ids, mode, hop);
        if (hop) |x| {
            pack(e, bytes);
            try x.give(x.ctx, bytes.len / (planes * hidden * 4), bytes);
            if (e.first != 0) return;
            try x.take(x.ctx, bytes.len / (planes * hidden * 4), bytes);
            unpack(e, bytes);
        }
        if (e.first == 0) try head(e, parts, mode);
    }

    /// The rows in (embedded on rank 0, else by the hop) and this stage's layers over them; the boundary's bytes.
    fn runLayers(e: *Engine, parts: []const Part, ids: []const u32, mode: plan.Mode, hop: ?driver.Hop) ![]u8 {
        var n: usize = 0;
        for (parts) |p| n += p.rows;
        if (n > most_rows or n != ids.len) return error.BadRows;
        const bytes = e.io[0 .. planes * n * hidden * 4];
        if (e.first == 0) {
            embed(e, parts, ids);
        } else {
            try hop.?.take(hop.?.ctx, n, bytes);
            unpack(e, bytes);
        }
        const end = if (e.plant == .short_range) e.end - 1 else e.end;
        var at: usize = 0;
        for (parts) |p| {
            if (mode == .window and p.rows > most_window) return error.BadRows;
            if (mode == .window) p.cache.before = p.cache.state;
            for (e.first..end) |l| {
                for (0..p.rows) |k| {
                    layer(e, p.cache, @intCast(l), at + k);
                    if (mode == .window) p.cache.rows[k][l] = p.cache.state[l];
                }
            }
            at += p.rows;
            p.cache.last = if (mode == .window) .{ .start = p.cache.len, .rows = p.rows } else null;
            p.cache.len += p.rows;
        }
        return bytes;
    }

    fn embed(e: *Engine, parts: []const Part, ids: []const u32) void {
        var row: usize = 0;
        for (parts) |p| {
            for (0..p.rows) |k| {
                const pos: u32 = p.cache.len + @as(u32, @intCast(k));
                for (0..hidden) |j| {
                    e.h[row * hidden + j] = ids[row] *% 2654435761 +% pos *% 40503 +% @as(u32, @intCast(j));
                    e.r[row * hidden + j] = 0;
                }
                row += 1;
            }
        }
    }

    fn layer(e: *Engine, c: *Cache, l: u32, row: usize) void {
        const h = e.h[row * hidden ..][0..hidden];
        c.state[l] = c.state[l] *% 31 +% h[0] +% 1;
        for (h, e.r[row * hidden ..][0..hidden], 0..) |*v, *r, j| {
            v.* = v.* *% 0x9E3779B1 +% (l + 1) *% 0x85EBCA6B +% c.state[l];
            r.* +%= v.* ^ @as(u32, @intCast(j));
        }
    }

    fn head(e: *Engine, parts: []const Part, mode: plan.Mode) !void {
        var row: usize = 0;
        for (parts) |p| {
            for (0..p.rows) |k| {
                if (mode != .window and k + 1 != p.rows) continue;
                for (0..hidden) |j| try e.out.append(e.gpa, e.h[(row + k) * hidden + j] ^ e.r[(row + k) * hidden + j]);
            }
            row += p.rows;
        }
    }

    fn pack(e: *Engine, bytes: []u8) void {
        const vals = bytes.len / 4 / planes;
        for (0..vals) |i| {
            std.mem.writeInt(u32, bytes[4 * i ..][0..4], e.h[i], .little);
            std.mem.writeInt(u32, bytes[4 * (vals + i) ..][0..4], e.r[i], .little);
        }
    }

    fn unpack(e: *Engine, bytes: []const u8) void {
        const vals = bytes.len / 4 / planes;
        for (0..vals) |i| {
            e.h[i] = std.mem.readInt(u32, bytes[4 * i ..][0..4], .little);
            e.r[i] = std.mem.readInt(u32, bytes[4 * (vals + i) ..][0..4], .little);
        }
    }
};

const D = driver.Driver(Fake);

/// One round of the scenario: entries, then the round's mode; ids come from a fixed sequence.
const Round = struct { entries: []const driver.Entry, mode: plan.Mode };

/// Two prompt chunks, steps, a second stream beside, a keep-alive, windows kept 3, 0, 5 and 2 of 5, then closes.
const scenario = [_]Round{
    .{ .entries = &.{.{ .id = 1, .op = .open, .rows = 10 }}, .mode = .prompt },
    .{ .entries = &.{.{ .id = 1, .op = .@"continue", .rows = 6 }}, .mode = .prompt },
    .{ .entries = &.{.{ .id = 1, .op = .@"continue", .rows = 1 }}, .mode = .window },
    .{ .entries = &.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = 1 }}, .mode = .window },
    .{ .entries = &.{.{ .id = 2, .op = .open, .rows = 5 }}, .mode = .prompt },
    .{ .entries = &.{ .{ .id = 1, .op = .@"continue", .rows = 4, .accepted = 1 }, .{ .id = 2, .op = .@"continue", .rows = 4 } }, .mode = .window },
    .{ .entries = &.{}, .mode = .window },
    .{ .entries = &.{ .{ .id = 1, .op = .@"continue", .rows = 5, .accepted = 2 }, .{ .id = 2, .op = .@"continue", .rows = 1, .accepted = 4 } }, .mode = .window },
    .{ .entries = &.{.{ .id = 1, .op = .@"continue", .rows = 5, .accepted = 3 }}, .mode = .window },
    .{ .entries = &.{.{ .id = 1, .op = .@"continue", .rows = 5, .accepted = 0 }}, .mode = .window },
    .{ .entries = &.{.{ .id = 1, .op = .@"continue", .rows = 5, .accepted = 5 }}, .mode = .window },
    .{ .entries = &.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = 2 }}, .mode = .window },
    .{ .entries = &.{ .{ .id = 1, .op = .close }, .{ .id = 2, .op = .close } }, .mode = .window },
};

fn run(lead: *D.Lead) !void {
    var ids: [most_rows]u32 = undefined;
    for (&ids, 0..) |*x, i| x.* = @intCast(1000 + 37 * i);
    for (scenario) |r| {
        var n: u32 = 0;
        for (r.entries) |en| n += en.rows;
        if (r.entries.len == 0) try lead.beat() else try lead.step(r.entries, ids[0..n], r.mode);
    }
}

/// One engine, no ring: the reference bytes.
fn whole(gpa: std.mem.Allocator) ![]u32 {
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, parallel, context, 1);
    defer lead.deinit(gpa);
    try run(&lead);
    return e.out.toOwnedSlice(gpa);
}

/// A ring's layout, a fault planted per rank, and one rank that describes another range than the layout's.
const Case = struct {
    layout: []const boot.Range,
    plants: [most]Plant = @splat(.none),
    misdescribe: ?struct { rank: u32, range: boot.Range } = null,
    /// Pipelined rounds: the streams in `k` groups, `k` rounds in flight (0: the scenario, one round at a time).
    k: usize = 0,
    /// The last stage echoes a round's plan short or empty; rank 0 pipes one round without rows.
    echo: ?Echo = null,
};

const Echo = enum { short, empty };

const Result = struct { err: ?anyerror = null, out: []u32 = &.{} };

fn rankMain(gpa: std.mem.Allocator, o: ring.Options, l: net.Fd, c: Case, res: *Result) void {
    rankRun(gpa, o, l, c, res) catch |e| {
        res.err = e;
    };
}

fn rankRun(gpa: std.mem.Allocator, o: ring.Options, l: net.Fd, c: Case, res: *Result) !void {
    var r = try ring.Ring.join(o, l);
    defer r.close();
    defer r.stop(); // as the tools do: a clean end passes a stop round the ring
    const mine = c.layout[o.rank];
    var e: Fake.Engine = .{ .gpa = gpa, .first = mine.first, .end = mine.end, .plant = c.plants[o.rank] };
    defer e.out.deinit(gpa);
    var said = mine;
    if (c.misdescribe) |m| if (m.rank == o.rank) {
        said = m.range;
    };
    const own = try D.describe(&e, o.rank, o.world, said, if (c.k > 0) pipe_streams else parallel, context);
    const b = Fake.boundary(&e);
    const model: boot.Model = .{ .version = Fake.version, .kv_dtype = .bf16, .layers = layers, .boundary = b, .shapes = own.shapes, .taps = &.{} };
    const width: u32 = if (c.k > 0) pipe_streams else parallel;
    const next = try driver.bootRing(gpa, &r, &own, &model, c.layout, Fake.backend);
    if (c.echo != null and o.rank + 1 == o.world) return badEcho(&r, next, c.echo.?);
    if (o.rank != 0) return D.serve(gpa, &e, &r, width, context, next);
    var lead = try D.Lead.init(gpa, &e, &r, width, context, next);
    defer lead.deinit(gpa);
    if (c.echo != null) {
        try lead.pipe(t.io, 2);
        try lead.issue(&.{}, &.{}, .window);
        return lead.complete();
    }
    if (c.k > 0) {
        res.out = try piped(gpa, &lead, &e, c.k);
        return;
    }
    try run(&lead);
    res.out = try e.out.toOwnedSlice(gpa);
}

/// A last stage that echoes each plan cut short (one byte less) or empty, until rank 0 stops the ring.
fn badEcho(r: *ring.Ring, first_round: u64, echo: Echo) !void {
    var buf: [4096]u8 = undefined;
    var round = first_round;
    while (true) : (round += 1) {
        const h = r.recv(.plan, round, &buf) catch |err| return if (err == error.Stopped) {} else err;
        const n: usize = @intCast(h.bytes);
        try r.send(.plan, round, 0, &.{buf[0..if (echo == .short) n - 1 else 0]});
    }
}

/// Every rank of `c` on its own loopback thread; each result holds rank 0's bytes or the rank's error.
fn ringRun(gpa: std.mem.Allocator, c: Case, res: *[most]Result) !void {
    const world: u32 = @intCast(c.layout.len);
    var peers: [most]ring.Peer = undefined;
    var listeners: [most]net.Fd = undefined;
    for (0..world) |i| {
        listeners[i] = try net.listen("127.0.0.1", 0);
        peers[i] = .{ .host = "127.0.0.1", .port = try net.port(listeners[i]) };
    }
    const fp = driver.fingerprint(Fake.version, "{}", c.layout);
    var th: [most]std.Thread = undefined;
    for (0..world) |i| {
        const o: ring.Options = .{ .rank = @intCast(i), .world = world, .peers = peers[0..world], .fingerprint = fp, .key = @splat(0x3C), .timeout_ns = 5 * std.time.ns_per_s };
        th[i] = try std.Thread.spawn(.{}, rankMain, .{ gpa, o, listeners[i], c, &res[i] });
    }
    for (th[0..world]) |x| x.join();
}

fn expectRing(c: Case, same: bool) !void {
    const gpa = t.allocator;
    const ref = try whole(gpa);
    defer gpa.free(ref);
    var res: [most]Result = @splat(.{});
    try ringRun(gpa, c, &res);
    defer gpa.free(res[0].out);
    for (res[0..c.layout.len]) |x| try t.expectEqual(@as(?anyerror, null), x.err);
    try t.expectEqual(same, std.mem.eql(u32, ref, res[0].out));
}

const two = [_]boot.Range{ .{ .first = 0, .end = 3 }, .{ .first = 3, .end = 6 } };
const three = [_]boot.Range{ .{ .first = 0, .end = 2 }, .{ .first = 2, .end = 5 }, .{ .first = 5, .end = 6 } };

test "stage driver: one engine's head bytes are a function of every layer and every rollback" {
    const gpa = t.allocator;
    const ref = try whole(gpa);
    defer gpa.free(ref);
    try t.expectEqual(@as(usize, 35 * hidden), ref.len);
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = layers, .plant = .no_restore };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, parallel, context, 1);
    defer lead.deinit(gpa);
    try run(&lead);
    try t.expect(!std.mem.eql(u32, ref, e.out.items));
}

test "stage driver: 2 and 3 stages on a loopback ring give one engine's bytes" {
    try expectRing(.{ .layout = &two }, true);
    try expectRing(.{ .layout = &three }, true);
}

test "stage driver: a stage that runs one layer short of its range changes the bytes" {
    try expectRing(.{ .layout = &three, .plants = .{ .none, .short_range, .none } }, false);
}

test "stage driver: a follower that rolls back its position but not its state changes the bytes" {
    try expectRing(.{ .layout = &two, .plants = .{ .none, .no_restore, .none } }, false);
    try expectRing(.{ .layout = &three, .plants = .{ .none, .none, .no_restore } }, false);
}

test "stage driver: a follower that drops a rollback is refused by name, and rank 0 ends" {
    var res: [most]Result = @splat(.{});
    try ringRun(t.allocator, .{ .layout = &three, .plants = .{ .none, .no_rollback, .none } }, &res);
    defer t.allocator.free(res[0].out);
    try t.expectEqual(@as(?anyerror, error.OtherPosition), res[1].err);
    try t.expect(res[0].err != null);
}

test "stage driver: a stage that describes another range than the layout's is refused at boot" {
    var res: [most]Result = @splat(.{});
    try ringRun(t.allocator, .{ .layout = &three, .misdescribe = .{ .rank = 2, .range = .{ .first = 4, .end = 6 } } }, &res);
    defer t.allocator.free(res[0].out);
    try t.expectEqual(@as(?anyerror, error.OtherRange), res[0].err);
}

/// Each stream's head bytes in stream order, for `k` groups pipelined: prompts one at a time, then drafted windows.
fn piped(gpa: std.mem.Allocator, lead: *D.Lead, e: *Fake.Engine, k: usize) ![]u32 {
    try lead.pipe(t.io, k);
    var hist: [pipe_streams]std.ArrayList(u32) = @splat(.empty);
    defer for (&hist) |*h| h.deinit(gpa);
    var ids: [most_rows]u32 = undefined;
    for (0..pipe_streams) |s| {
        const n = 3 + s;
        for (ids[0..n], 0..) |*x, i| x.* = @intCast(500 + 11 * s + i);
        const before = e.out.items.len;
        try lead.step(&.{.{ .id = @intCast(s + 1), .op = .open, .rows = @intCast(n) }}, ids[0..n], .prompt);
        try hist[s].appendSlice(gpa, e.out.items[before..]);
    }
    var rows: [pipe_streams]u32 = @splat(0);
    var steps: [most]usize = @splat(0);
    var order: [most]usize = undefined; // the groups' rounds in flight, oldest first
    for (0..k) |g| {
        try issueGroup(lead, &hist, &rows, k, g, 0);
        order[g] = g;
    }
    var n_order = k;
    while (n_order > 0) {
        const g = order[0];
        std.mem.copyForwards(usize, order[0 .. n_order - 1], order[1..n_order]);
        n_order -= 1;
        const before = e.out.items.len;
        try lead.complete();
        var at = before;
        for (group(k, g)) |s| {
            try hist[s].appendSlice(gpa, e.out.items[at..][0 .. rows[s] * hidden]);
            at += rows[s] * hidden;
        }
        steps[g] += 1;
        if (steps[g] == pipe_steps) continue;
        try issueGroup(lead, &hist, &rows, k, g, steps[g]);
        order[n_order] = g;
        n_order += 1;
    }
    var closes: [pipe_streams]driver.Entry = undefined;
    for (&closes, 1..) |*c, id| c.* = .{ .id = @intCast(id), .op = .close };
    try lead.step(&closes, &.{}, .window);
    var all: std.ArrayList(u32) = .empty;
    for (hist) |h| try all.appendSlice(gpa, h.items);
    return all.toOwnedSlice(gpa);
}

const all_streams = [pipe_streams]usize{ 0, 1, 2, 3 };

/// Group `g` of `k`: contiguous streams, the first groups one longer.
fn group(k: usize, g: usize) []const usize {
    const base = pipe_streams / k;
    const extra = pipe_streams % k;
    const first = g * base + @min(g, extra);
    return all_streams[first..][0 .. base + @intFromBool(g < extra)];
}

/// Group `g`'s round `j`: per stream 1 to 3 rows from its own history, its last window kept as its history says.
fn issueGroup(lead: *D.Lead, hist: *[pipe_streams]std.ArrayList(u32), rows: *[pipe_streams]u32, k: usize, g: usize, j: usize) !void {
    var entries: [pipe_streams]driver.Entry = undefined;
    var ids: [most_rows]u32 = undefined;
    var n: usize = 0;
    for (group(k, g), entries[0..group(k, g).len]) |s, *en| {
        const last = hist[s].items[hist[s].items.len - 1];
        const accepted: u32 = if (j == 0) 0 else last % (rows[s] + 1);
        rows[s] = @intCast(1 + (j + s) % 3);
        en.* = .{ .id = @intCast(s + 1), .op = .@"continue", .rows = rows[s], .accepted = accepted };
        for (0..rows[s]) |i| ids[n + i] = 1000 + (last +% @as(u32, @intCast(31 * i + j))) % 50000;
        n += rows[s];
    }
    try lead.issue(entries[0..group(k, g).len], ids[0..n], .window);
}

/// One engine, no ring, pipelined alike: the reference.
fn wholePiped(gpa: std.mem.Allocator, k: usize) ![]u32 {
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, pipe_streams, context, 1);
    defer lead.deinit(gpa);
    return piped(gpa, &lead, &e, k);
}

test "pipelined rounds: one engine's streams are the same in 1, 2 and 3 groups" {
    const gpa = t.allocator;
    const ref = try wholePiped(gpa, 1);
    defer gpa.free(ref);
    try t.expect(ref.len > pipe_streams * pipe_steps * hidden);
    for ([_]usize{ 2, 3 }) |k| {
        const got = try wholePiped(gpa, k);
        defer gpa.free(got);
        try t.expectEqualSlices(u32, ref, got);
    }
}

test "pipelined rounds: 2 and 3 stages with 1, 2 and 3 rounds in flight give one engine's bytes" {
    const gpa = t.allocator;
    const ref = try wholePiped(gpa, 1);
    defer gpa.free(ref);
    for ([_][]const boot.Range{ &two, &three }) |layout| for ([_]usize{ 1, 2, 3 }) |k| {
        var res: [most]Result = @splat(.{});
        try ringRun(gpa, .{ .layout = layout, .k = k }, &res);
        defer gpa.free(res[0].out);
        for (res[0..layout.len]) |x| try t.expectEqual(@as(?anyerror, null), x.err);
        try t.expectEqualSlices(u32, ref, res[0].out);
    };
}

test "pipelined rounds: a planted fault on a follower still changes the bytes with rounds in flight" {
    const gpa = t.allocator;
    const ref = try wholePiped(gpa, 1);
    defer gpa.free(ref);
    var res: [most]Result = @splat(.{});
    try ringRun(gpa, .{ .layout = &three, .k = 2, .plants = .{ .none, .no_restore, .none } }, &res);
    defer gpa.free(res[0].out);
    try t.expect(!std.mem.eql(u32, ref, res[0].out));
}

test "pipelined rounds: a stream in flight, a step beside rounds in flight and one round too many are refused" {
    const gpa = t.allocator;
    var e: Fake.Engine = .{ .gpa = gpa, .first = 0, .end = layers };
    defer e.out.deinit(gpa);
    var lead = try D.Lead.init(gpa, &e, null, pipe_streams, context, 1);
    defer lead.deinit(gpa);
    try lead.pipe(t.io, 2);
    const ids = [_]u32{ 5, 6, 7 };
    try lead.step(&.{.{ .id = 1, .op = .open, .rows = 3 }}, &ids, .prompt);
    try lead.step(&.{.{ .id = 2, .op = .open, .rows = 3 }}, &ids, .prompt);
    try lead.issue(&.{.{ .id = 1, .op = .@"continue", .rows = 1 }}, ids[0..1], .window);
    try t.expectError(error.StreamInFlight, lead.issue(&.{.{ .id = 1, .op = .@"continue", .rows = 1, .accepted = 1 }}, ids[0..1], .window));
    try t.expectError(error.RoundsInFlight, lead.step(&.{.{ .id = 2, .op = .@"continue", .rows = 1 }}, ids[0..1], .window));
    try lead.issue(&.{.{ .id = 2, .op = .@"continue", .rows = 1 }}, ids[0..1], .window);
    try t.expectError(error.RoundsInFlight, lead.beat());
    try t.expectError(error.TooManyInFlight, lead.issue(&.{}, &.{}, .window));
    try lead.complete();
    try lead.complete();
    try t.expectError(error.NothingInFlight, lead.complete());
    try t.expectEqual(@as(usize, 0), lead.inFlight());
}

test "pipelined rounds: a short or empty echo of a round without rows is refused" {
    for ([_]Echo{ .short, .empty }) |echo| {
        var res: [most]Result = @splat(.{});
        try ringRun(t.allocator, .{ .layout = &two, .echo = echo }, &res);
        try t.expectEqual(@as(?anyerror, error.OtherEcho), res[0].err);
    }
}
