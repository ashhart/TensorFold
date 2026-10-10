//! The prefix tree's checks of `check`: shared prefixes equal fresh runs, copy on write, and no page leaks.

const std = @import("std");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const check_ctx = @import("check_ctx.zig");
const ids_file = @import("ids_file.zig");
const lanes_session = @import("lanes_session.zig");

const Ctx = check_ctx.Ctx;
const Prompt = ids_file.Prompt;
const Session = lanes_session.Session;
const pages = qwen35.pages;

/// Tokens of the system prompt the requests share, and how many questions follow it.
const system_len = 1100;
const questions = 8;
/// Of those, the ones whose conversation goes on a turn.
const turns = 4;
/// Pages the tree checks leave room for beside the streams'.
pub const tree_pages = 600;
/// Snapshots the stress keeps, and the streams its last pass runs at once: every question, turn and branch.
pub const slots = 8;
pub const most_streams = questions + turns + 2;

/// Tokens of the question of each request of the memory line, and how far apart they start.
const long_tail = 320;
const tail_step = 16;

/// Positions the longest request of the tree checks takes, replies of `tokens` tokens included.
pub fn capacity(tokens: u32) usize {
    return system_len + long_tail + 2 * tokens + 64;
}

/// The requests the tree is built from: every one a shared system prompt and its own question.
fn singles(c: *Ctx) ![]const Prompt {
    const sys = c.ids[0..system_len];
    const shared = try c.arena.dupe(u32, &.{system_len});
    const step = (c.ids.len - system_len) / questions;
    const out = try c.arena.alloc(Prompt, questions);
    for (out, 0..) |*p, j| {
        const tail = c.ids[system_len + j * step ..][0 .. 24 + 8 * (j % 4)];
        p.* = .{ .name = try std.fmt.allocPrint(c.arena, "a{d}", .{j}), .ids = try std.mem.concat(c.arena, u32, &.{ sys, tail }), .shared = shared };
    }
    return out;
}

/// Requests that branch off the others: one inside the system prompt's last page, one in the middle of its pages.
fn branches(c: *Ctx, singles_set: []const Prompt) ![]const Prompt {
    const sys = c.ids[0..system_len];
    const shared = try c.arena.dupe(u32, &.{system_len});
    const first = singles_set[0].ids[system_len..][0..20];
    const other = singles_set[questions - 1].ids[system_len..][0..30];
    const out = try c.arena.alloc(Prompt, 2);
    out[0] = .{ .name = "c0", .ids = try std.mem.concat(c.arena, u32, &.{ sys, first, other }), .shared = shared };
    out[1] = .{ .name = "c1", .ids = try std.mem.concat(c.arena, u32, &.{ sys[0..640], c.ids[100..160] }) };
    return out;
}

fn sameReplies(a: Session.Done, b: Session.Done, skip: usize, buf: []u8) ?[]const u8 {
    for (a.replies[skip..], b.replies[skip..]) |x, y| {
        const at = std.mem.indexOfDiff(u32, x.tokens, y.tokens) orelse continue;
        return std.fmt.bufPrint(buf, "{s} differs at token {d}", .{ x.name, at }) catch "differs";
    }
    return null;
}

/// Bytes a copy of the caches at each kept mark would hold: marks past where each request resumed, repeats once.
fn copiedBytes(c: *Ctx, set: []const Prompt, replies: []const Session.Reply) usize {
    const spec = c.e.model().spec;
    var kept: std.ArrayList([]const u32) = .empty;
    var bytes: usize = 0;
    for (set, replies) |p, r| {
        const marks = c.session.h.prefix.store.marks(c.arena, p.ids, r.cached, @intCast(p.ids.len / 2), p.shared, &.{}) catch return bytes;
        for (marks) |m| {
            const ids = p.ids[0..m];
            const known = for (kept.items) |k| {
                if (std.mem.eql(u32, k, ids)) break true;
            } else false;
            if (known) continue;
            kept.append(c.arena, ids) catch return bytes;
            bytes += qwen35.memory.stateBytes(spec, c.e.model().act.size(), m);
        }
    }
    return bytes;
}

fn mib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 20);
}

pub fn run(c: *Ctx) void {
    stress(c);
    memoryLine(c);
    if (c.tp()) {
        c.report.skip("radix: a resumed page equals a fresh one", "a prompt pass runs on one rank");
        c.report.skip("radix: a shared page is copied before a write", "the copy runs on one rank");
        return;
    }
    resumedPages(c);
    copyOnWrite(c);
}

/// Requests branching from shared prefixes, each against its own fresh run, alone and then all together.
fn stress(c: *Ctx) void {
    const name = "radix stress";
    if (c.ids.len < system_len + 8 * questions) return c.report.skip(name, "too few ids");
    const h = c.session.h;
    h.keepPrompts(0, 0);
    h.keepPrompts(slots, 4 << 30);
    defer h.keepPrompts(0, 0);
    const first = singles(c) catch |err| return c.report.broke(name, err);
    const alone = c.session.run(c.arena, .{ .prompts = first, .max_new = c.tokens, .solo = true }) catch |err| return c.report.broke(name, err);
    // the next turn of some conversations, and the branches
    const next = lanes_session.extend(c.arena, first[0..turns], alone.replies[0..turns]) catch |err| return c.report.broke(name, err);
    const branched = branches(c, first) catch |err| return c.report.broke(name, err);
    const second = std.mem.concat(c.arena, Prompt, &.{ next, branched }) catch |err| return c.report.broke(name, err);
    const more = c.session.run(c.arena, .{ .prompts = second, .max_new = c.tokens, .solo = true }) catch |err| return c.report.broke(name, err);
    const every = std.mem.concat(c.arena, Prompt, &.{ first, second }) catch |err| return c.report.broke(name, err);
    // all of them again at once, on the warm tree
    const together = c.session.run(c.arena, .{ .prompts = every, .max_new = c.tokens }) catch |err| return c.report.broke(name, err);
    const fresh = c.session.run(c.arena, .{ .prompts = every, .max_new = c.tokens, .solo = true, .drafts = false }) catch |err| return c.report.broke(name, err);
    var buf: [128]u8 = undefined;
    const label = std.fmt.allocPrint(c.arena, "radix: {d} requests, resumed == fresh", .{every.len}) catch return;
    if (sameReplies(alone, .{ .replies = fresh.replies[0..first.len], .seconds = 0, .rounds = 0, .replayed = 0, .submit_ns = 0, .spent = fresh.spent, .calls = fresh.calls }, 0, &buf)) |why| return c.report.fail(label, "{s}", .{why});
    if (sameReplies(more, .{ .replies = fresh.replies[first.len..], .seconds = 0, .rounds = 0, .replayed = 0, .submit_ns = 0, .spent = fresh.spent, .calls = fresh.calls }, 0, &buf)) |why| return c.report.fail(label, "{s}", .{why});
    var resumed: usize = 0;
    var edges = true;
    for ([_][]const Session.Reply{ alone.replies, more.replies, together.replies }) |set| for (set) |r| {
        edges = edges and r.cached % pages.tokens == 0;
        resumed += @intFromBool(r.cached > 0);
    };
    c.report.pass(label, "{d} of {d} resumed, {d} tokens each", .{ resumed, alone.replies.len + more.replies.len + together.replies.len, c.tokens });
    if (sameReplies(together, fresh, 0, &buf)) |why| return c.report.fail("radix: all together == fresh", "{s}", .{why});
    c.report.pass("radix: all together == fresh", "{d} streams, {d} drafts accepted", .{ together.replies.len, together.accepted() });
    if (!edges) c.report.fail("radix: spans start on page edges", "a request resumed off a page edge", .{}) else c.report.pass("radix: spans start on page edges", "every resume at a multiple of {d}", .{pages.tokens});
    // the requests after the first take the system prompt's whole pages from the tree
    const want: usize = system_len / pages.tokens * pages.tokens;
    var short: usize = 0;
    for (alone.replies[1..]) |r| short += @intFromBool(r.cached != want);
    if (short > 0) c.report.fail("radix: the system prompt is shared", "{d} of {d} requests after the first did not resume at {d}", .{ short, alone.replies.len - 1, want }) else c.report.pass("radix: the system prompt is shared", "{d} requests resumed at {d} of {d} tokens", .{ alone.replies.len - 1, want, system_len });
}

/// What `questions` requests sharing one system prompt leave in the tree, against per-cut copies of the caches.
fn memoryLine(c: *Ctx) void {
    const name = "radix memory";
    if (c.ids.len < system_len + tail_step * questions + long_tail) return c.report.skip(name, "too few ids");
    const h = c.session.h;
    h.keepPrompts(0, 0);
    h.keepPrompts(2 * questions, 2 << 30);
    defer h.keepPrompts(0, 0);
    const sys = c.ids[0..system_len];
    const shared = c.arena.dupe(u32, &.{system_len}) catch return;
    const set = c.arena.alloc(Prompt, questions) catch return;
    for (set, 0..) |*p, j| {
        const tail = c.ids[system_len + j * tail_step ..][0..long_tail];
        p.* = .{ .name = std.fmt.allocPrint(c.arena, "q{d}", .{j}) catch return, .ids = std.mem.concat(c.arena, u32, &.{ sys, tail }) catch return, .shared = shared };
    }
    const done = c.session.run(c.arena, .{ .prompts = set, .max_new = 1, .solo = true }) catch |err| return c.report.broke(name, err);
    const tree = &h.prefix.store.tree;
    const copied = copiedBytes(c, set, done.replies);
    const unique = pagesBytes(c, tree.held_pages);
    const states = tree.held() - unique;
    var cached: usize = 0;
    for (done.replies) |r| cached += r.cached;
    c.report.pass(name, "{d} requests sharing {d} system tokens, {d} tokens each: the tree holds {d:.1} MiB ({d} pages {d:.1} MiB, {d} snapshots {d:.1} MiB); copies of the caches at each cut held {d:.1} MiB; {d} prompt tokens came from the tree", .{ set.len, system_len, set[0].ids.len, mib(tree.held()), tree.held_pages, mib(unique), tree.snaps, mib(states), mib(copied), cached });
}

fn pagesBytes(c: *Ctx, n: usize) u64 {
    return n * c.e.pool.pageBytes();
}

/// The pages of a prompt resumed from the tree hold what a fresh prefill of it writes.
fn resumedPages(c: *Ctx) void {
    const name = "radix: a resumed page equals a fresh one";
    const h = c.session.h;
    const e = c.e;
    h.keepPrompts(0, 0);
    h.keepPrompts(8, 4 << 30);
    defer h.keepPrompts(0, 0);
    const set = singles(c) catch |err| return c.report.broke(name, err);
    // the first run keeps the system prompt, the second resumes from it
    _ = c.session.run(c.arena, .{ .prompts = set[0..1], .max_new = 1, .solo = true }) catch |err| return c.report.broke(name, err);
    const prompt = set[1].ids;
    var st = lanes.Stream.init(c.gpa, .{ .id = "resumed", .prompt = prompt, .max_new = 1, .history_len = @intCast(prompt.len / 2), .shared_prefixes = set[1].shared }) catch |err| return c.report.broke(name, err);
    defer st.deinit(c.gpa);
    const be = h.backend();
    be.prefill(&st) catch |err| return c.report.broke(name, err);
    defer be.release(&st);
    if (st.cached == 0) return c.report.fail(name, "the prompt resumed from nothing", .{});
    const lane = h.lanes.get(&st) orelse return c.report.fail(name, "no lane", .{});
    var fresh = e.newCaches(prompt.len + 8) catch |err| return c.report.broke(name, err);
    defer fresh.deinit(c.gpa);
    _ = e.prefill(&fresh, prompt, 0, null, .{ .sampling = null, .position = prompt.len }, null) catch |err| return c.report.broke(name, err);
    e.drain();
    const m = e.model();
    const layer_bytes = e.pool.layerBytes();
    const row = m.spec.head_dim * m.act.size();
    const mine = c.arena.alloc(u8, 2 * layer_bytes) catch return;
    const theirs = c.arena.alloc(u8, 2 * layer_bytes) catch return;
    var compared: usize = 0;
    for (0..m.spec.n_layers) |layer| {
        if (!m.spec.full(layer)) continue;
        for (0..pages.pagesFor(prompt.len)) |p| {
            e.pool.read(layer, lane.caches.table.items[p], mine) catch |err| return c.report.broke(name, err);
            e.pool.read(layer, fresh.table.items[p], theirs) catch |err| return c.report.broke(name, err);
            const used = @min(pages.tokens, prompt.len - p * pages.tokens);
            for ([_]usize{ 0, layer_bytes }) |half| for (0..m.spec.kv_heads) |head| {
                const at = half + head * pages.tokens * row;
                if (!std.mem.eql(u8, mine[at..][0 .. used * row], theirs[at..][0 .. used * row])) return c.report.fail(name, "layer {d}, page {d}, head {d} differs", .{ layer, p, head });
            };
            compared += 1;
        }
    }
    const per_layer = pages.pagesFor(prompt.len);
    c.report.pass(name, "{d} pages of {d} tokens in each of {d} attention layers, the first {d} from the tree, all equal", .{ per_layer, prompt.len, compared / per_layer, st.cached / pages.tokens });
}

/// A page two streams hold is copied before one writes it, and the copy holds the same bytes.
fn copyOnWrite(c: *Ctx) void {
    const name = "radix: a shared page is copied before a write";
    const e = c.e;
    const m = e.model();
    const len = pages.tokens + 36;
    var first = e.newCaches(2 * pages.tokens) catch |err| return c.report.broke(name, err);
    defer first.deinit(c.gpa);
    _ = e.prefill(&first, c.ids[0..len], 0, null, .{ .sampling = null, .position = len }, null) catch |err| return c.report.broke(name, err);
    e.drain();
    var second = e.emptyCaches(2 * pages.tokens) catch |err| return c.report.broke(name, err);
    defer second.deinit(c.gpa);
    for (first.table.items[0..2]) |id| e.pool.retain(id);
    second.set(c.gpa, 0, first.table.items[0..2]) catch |err| return c.report.broke(name, err);
    var done: std.ArrayList([3]u32) = .empty;
    defer done.deinit(c.gpa);
    // a write at position 100 lands in the second page, which both hold
    second.writable(&done, c.gpa, 100, 101, e.stream.handle) catch |err| return c.report.broke(name, err);
    e.drain();
    if (done.items.len != 1 or done.items[0][0] != 1) return c.report.fail(name, "{d} pages copied", .{done.items.len});
    if (second.table.items[0] != first.table.items[0] or second.table.items[1] == first.table.items[1]) return c.report.fail(name, "the wrong pages were copied", .{});
    const layer_bytes = e.pool.layerBytes();
    const a = c.arena.alloc(u8, 2 * layer_bytes) catch return;
    const b = c.arena.alloc(u8, 2 * layer_bytes) catch return;
    for (0..m.spec.n_layers) |layer| {
        if (!m.spec.full(layer)) continue;
        e.pool.read(layer, first.table.items[1], a) catch |err| return c.report.broke(name, err);
        e.pool.read(layer, second.table.items[1], b) catch |err| return c.report.broke(name, err);
        if (!std.mem.eql(u8, a, b)) return c.report.fail(name, "layer {d}: the copy differs", .{layer});
    }
    if (e.pool.ids.refs[first.table.items[0]] != 2 or e.pool.ids.refs[first.table.items[1]] != 1) return c.report.fail(name, "the holders were not counted", .{});
    c.report.pass(name, "page 1 copied, page 0 still shared, the copy equal in every attention layer", .{});
}

/// After every run the pool holds the pages it started with: no stream and no tree entry leaked one.
pub fn leaked(c: *Ctx) void {
    const name = "radix: every page comes back";
    c.session.h.keepPrompts(0, 0);
    const used = c.e.pool.ids.used();
    if (used != c.pages_at_start or c.e.pool.ids.reserved != 0) return c.report.fail(name, "{d} pages held and {d} promised after the runs, {d} held before", .{ used, c.e.pool.ids.reserved, c.pages_at_start });
    c.report.pass(name, "{d} of {d} pages held after the runs, as before", .{ used, c.e.pool.count });
}
