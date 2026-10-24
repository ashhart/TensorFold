//! Bounded decode-time lookups inject high-scoring notes when a reply names an uncovered subject.
const std = @import("std");
const api = @import("engine_api");
const json = @import("json");
const errors = @import("errors.zig");
const clock = @import("clock.zig");
const chat = @import("chat.zig");
const prompt_mod = @import("prompt.zig");
const chunk_plan = @import("chunk_plan.zig");
const model_text = @import("model_text.zig");
const reply_text = @import("reply_text.zig");
const notes_route = @import("notes_route.zig");
const lookup = @import("notes/lookup.zig");
const retrieve = @import("notes/retrieve.zig");
const index = @import("notes/index.zig");
const Storage = @import("notes/storage.zig").Storage;
const Mailbox = @import("mailbox.zig").Mailbox;
const fields = @import("notes_decode_fields.zig");
pub const Place = fields.Place;
pub const Ask = fields.Ask;
pub const asked = fields.asked;
pub const shapeAsked = fields.shapeAsked;
pub const topWeight = fields.topWeight;
pub const Injection = fields.Injection;
pub const Check = fields.Check;
pub const Hit = fields.Hit;
const echo_top = fields.echo_top;
const Server = @import("server.zig").Server;
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

/// The most lookups one reply makes while it decodes, those that find nothing counted too ("notes_decode_checks").
pub const default_checks: u32 = 3;
/// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 21.
pub const default_budget_ms: f64 = 50;
/// The fewest reply tokens from one lookup to the next ("notes_decode_gap"); a name written sooner waits for them.
pub const default_gap: u32 = 4;
/// The most injections one reply takes.
pub const most: usize = 2;
/// A new note qualifies when the query names its subject and its score, that boost in it, is this share of topWeight.
pub const min_share: f64 = 1.0;

/// One reply's decode-time lookup: what is in its notes block so far, and what it put in.
pub const Live = struct {
    a: Allocator,
    on: bool,
    store: *Storage,
    mutex: ?*std.Io.Mutex = null, // the server's notes_mutex; null in tests that do not share a writer
    io: std.Io = undefined,
    ver: u64, // Storage.gen the first block was read from; another one stops the checks
    question: []const u8,
    past: bool,
    cap: u32,
    exclude: []const u32, // the client's notes_exclude
    place: Place,
    echo: bool,
    dry: bool = false,
    names: bool = false, // a name the reply just wrote, of an entity with notes outside the block, is a check point (the only one)
    max_checks: u32 = default_checks,
    budget_ns: i96 = @intFromFloat(default_budget_ms * 1e6),
    gap: u32 = default_gap,
    attempts: u32 = 0, // lookups started, whatever they found
    triggers: usize = 0, // names that asked for a lookup
    trigger_ns: i96 = 0, // the per-token name scans, in all
    scans: usize = 0, // the per-token name scans, counted
    pending: bool = false, // a name asked for a lookup the gap still holds back
    covered: std.ArrayList(u32) = .empty, // entities whose name already asked for a lookup in this reply: never again
    shape: notes_route.Shape = .{},
    markers: reply_text.Markers = reply_text.think_markers,
    held: std.ArrayList(u32) = .empty, // every note the model sees: the first block's, then the injected
    late: std.ArrayList(u32) = .empty, // (b): the late block's notes
    block: []const u8, // the first block as the prompt has it now
    prepared_block: []const u8 = "", // the first block in the prepared messages, which every continuation starts from
    block_tokens: usize,
    late_block: []const u8 = "",
    late_tokens: usize = 0,
    injections: std.ArrayList(Injection) = .empty,
    checks: std.ArrayList(Check) = .empty,
    n_checks: usize = 0,
    capped: usize = 0, // checks whose qualifying notes the cap left out
    best: Check = .{ .at = 0 },
    check_ns: i96 = 0,
    since: u32 = 0, // reply tokens since the last lookup
    stopped: ?[]const u8 = null,
    prepared: ?*const chat.Prepared = null,
    base: ?*const api.Request = null,
    first: ?*Mailbox = null,
    boxes: std.ArrayList(*Mailbox) = .empty, // the continuations' mailboxes
    carried: api.Stats = .{}, // the counters of the segments before the current one
    pause_from: ?i96 = null,
    submitted: i96 = 0, // when the last continuation was submitted

    fn stop(l: *Live, why: []const u8) void {
        if (l.stopped == null) l.stopped = why;
    }

    /// No checks for this reply, and why.
    pub fn off(l: *Live, why: []const u8) void {
        l.on = false;
        l.stop(why);
    }

    /// Binds the reply about to be submitted; null when its engine state could not be restarted (then nothing runs).
    pub fn bind(l: *Live, srv: *const Server, prepared: *const chat.Prepared, base: *const api.Request, box: *Mailbox) ?*Live {
        if (!l.on) return null;
        l.prepared = prepared;
        l.base = base;
        l.first = box;
        l.markers = srv.markers;
        const why: ?[]const u8 = if (base.structure != null) "structured output" else if (base.call != null) "a required tool call" else if (base.logprobs != null or base.prompt_logprobs != null) "logprobs" else if (base.think_budget > 0) "a thinking budget" else if (prepared.input.prompt != null) "not a chat" else null;
        if (why) |w| {
            l.stop(w);
            return null;
        }
        return l;
    }

    /// Whether the cap or the budget is spent; then the checks stop for the rest of the reply, the name scans too.
    pub fn exhausted(l: *Live) bool {
        if (l.stopped != null) return true;
        if (l.attempts >= l.max_checks) {
            l.stop("the check cap ran out");
            return true;
        }
        if (l.check_ns + l.trigger_ns >= l.budget_ns) {
            l.stop("the check time budget ran out");
            return true;
        }
        return false;
    }

    /// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 22.
    pub fn due(l: *Live, found: ?u32) Allocator.Error!bool {
        if (l.stopped != null) return false;
        l.since +|= 1;
        if (found) |e| {
            try l.covered.append(l.a, e);
            l.triggers += 1;
            l.pending = true;
        }
        if (!l.pending) return false;
        if (l.exhausted()) return false;
        if (l.attempts > 0 and l.since < l.gap) return false;
        l.pending = false;
        l.since = 0;
        l.attempts += 1;
        return true;
    }

    /// After a reply token (`collected` the reply so far): the continuation to switch to when a check put notes in.
    pub fn atToken(l: *Live, srv: *Server, a: Allocator, collected: []const u32) chat.Failure!?*api.Request {
        if (l.stopped != null) return null;
        if (l.exhausted()) return null; // spent: no more name scans either
        var found: ?u32 = null;
        if (l.names) {
            l.scans += 1;
            const s0 = clock.nowNs(srv.io);
            found = l.nameWritten(collected);
            l.trigger_ns += clock.nowNs(srv.io) - s0;
        }
        if (!try l.due(found)) return null;
        // a thinking reply is read whole: its reasoning is where it names who the question is about (measured)
        const hit = blk: {
            const t0 = clock.nowNs(srv.io);
            defer l.check_ns += clock.nowNs(srv.io) - t0;
            break :blk (try l.check(srv.text, a, try srv.text.decode(a, collected), collected.len)) orelse return null;
        };
        const t1 = clock.nowNs(srv.io);
        const next = (try l.continuation(srv, a, collected, hit)) orelse return null;
        try l.took(hit, collected.len, next.prompt.len, clock.nowNs(srv.io) - t1);
        return next;
    }

    /// Whether the reply's last tokens end a name of an entity with a current note outside the block.
    pub fn nameJustWritten(l: *Live, collected: []const u32) bool {
        return l.nameWritten(collected) != null;
    }

    fn lock(l: *Live) void {
        if (l.mutex) |m| m.lockUncancelable(l.io);
    }

    fn unlock(l: *Live) void {
        if (l.mutex) |m| m.unlock(l.io);
    }

    /// Detailed invariants: docs/NOTES-CODE-COMMENTS.md, section 23.
    pub fn nameWritten(l: *Live, collected: []const u32) ?u32 {
        l.lock();
        defer l.unlock();
        const idx = l.store.idx orelse {
            l.stop("the notes changed during the reply");
            return null;
        };
        if (l.store.gen != l.ver) {
            l.stop("the notes changed during the reply");
            return null;
        }
        var keys: [index.max_phrase]u32 = undefined;
        var n: usize = 1;
        const maximum = if (l.shape.resolve_subjects) @max(idx.phraseMax(), idx.aliasMax()) else idx.phraseMax();
        while (n <= @min(collected.len, @max(maximum, 1))) : (n += 1) {
            for (collected[collected.len - n ..], keys[0..n]) |t, *k| k.* = idx.keyOf(t);
            const ordinary = if (n == 1) idx.nameOf(keys[0]) else idx.phraseOf(index.packPhrase(keys[0..n]));
            const alias = if (l.shape.resolve_subjects) idx.aliasOf(index.packPhrase(keys[0..n])) else null;
            const candidates = if (l.shape.resolve_subjects and ordinary == null) idx.aliasCandidates(index.packPhrase(keys[0..n])) else null;
            // An unresolved common name is not sufficient live-fetch evidence.

            // Expanding hundreds of entities also makes fitting unbounded work.
            if (candidates) |cs| if (cs.len > 1) continue;
            const single = [1]u32{ordinary orelse alias orelse index.no_note};
            const entities = if (candidates) |cs| cs else &single;
            for (entities) |e| {
                if (e == index.no_note) continue;
                if (l.shape.resolve_subjects and idx.self_entity == e) continue; // the assistant's "my" is not the user's
                if (std.mem.indexOfScalar(u32, l.covered.items, e) != null) continue; // looked up already: found or not
                var cur = idx.entCur(e);
                while (cur.next()) |p| {
                    const id = idx.idAt(p);
                    if (std.mem.indexOfScalar(u32, l.held.items, id) == null and std.mem.indexOfScalar(u32, l.exclude, id) == null) return e;
                }
                if (l.past) {
                    var pairs = idx.entKeys(e);
                    while (pairs.next()) |pair| {
                        var history = idx.pairNotes(pair);
                        while (history.next()) |p| {
                            const id = idx.idAt(p);
                            if (std.mem.indexOfScalar(u32, l.held.items, id) == null and std.mem.indexOfScalar(u32, l.exclude, id) == null) return e;
                        }
                    }
                }
            }
        }
        return null;
    }

    /// The lookup over the question and the reply so far; the new notes that score at least min_share and fit the cap.
    pub fn check(l: *Live, text: model_text.Text, a: Allocator, reply: []const u8, at: usize) Allocator.Error!?Hit {
        const s = l.store;
        const t0 = clock.nowNs(l.io);
        l.lock();
        defer l.unlock();
        const current = s.idx orelse {
            l.stop("the notes changed during the reply");
            return null;
        };
        if (s.gen != l.ver) {
            l.stop("the notes changed during the reply");
            return null;
        }
        var view = current.*;
        view.settings.resolve_subjects = l.shape.resolve_subjects;
        // Until relation disambiguation is available, live checks consume only

        // unique aliases. The unique map marks collisions with no_note.
        view.alias_candidates_off = true;
        if (l.shape.resolve_subjects) view.settings.resolve_self = false; // first-person aliases belong to the question-only prefetch
        const idx = &view;
        const query = try std.mem.concat(a, u8, &.{ l.question, "\n", try plain(a, reply, .{ l.markers.open, l.markers.close }) });
        const qids = text.encode(a, query, false) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Template => {
                l.stop("the tokenizer cannot encode the reply");
                return null;
            },
        };
        const sc = try retrieve.scratchFor(s, idx);
        const got = try lookup.lookup(idx, sc, qids, l.past, l.cap, try std.mem.concat(a, u32, &.{ l.exclude, l.held.items }), .{});
        const top = topWeight(idx, l.past);
        if (top == 0) return null;
        const names = try named(a, idx, sc);
        // the best of the lookup's own candidates, named or not: what the threshold is weighed against (diagnostics)
        var first: Check = if (got.chosen.len > 0) .{ .at = at, .best = idx.idAt(got.chosen[0]), .share = scoreOf(sc, got.chosen[0]) / top } else .{ .at = at };
        if (first.share > l.best.share) l.best = first;
        l.n_checks += 1;
        if (l.echo) {
            const n = @min(got.chosen.len, echo_top);
            const ids = try l.a.alloc(u32, n);
            const sh = try l.a.alloc(f64, n);
            const ln = try l.a.alloc(bool, n);
            for (got.chosen[0..n], ids, sh, ln) |p, *id, *x, *y| {
                id.* = idx.idAt(p);
                x.* = scoreOf(sc, p) / top;
                y.* = std.mem.indexOfScalar(u32, names.all, idx.entityAt(p)) != null;
            }
            first.ids = ids;
            first.shares = sh;
            first.named = ln;
            try l.checks.append(l.a, first);
        }
        if (l.dry) return null;
        // the named entities' notes, not in the block nor excluded, by the lookup's order: score, then told
        const boost = top * idx.settings.subject;
        var keys: std.ArrayList(u128) = .empty;
        var score: std.AutoHashMapUnmanaged(u32, f64) = .empty;
        for (names.all, 0..) |e, i| {
            const extra: f64 = if (i >= names.boosted) boost else 0; // lookup() boosted the first ones itself
            try addNotes(a, &score, sc, idx.entCur(e), extra);
            if (l.past) {
                var ks = idx.entKeys(e);
                while (ks.next()) |pk| try addNotes(a, &score, sc, idx.pairNotes(pk), extra);
            }
        }
        var it = score.iterator();
        while (it.next()) |kv| {
            const id = idx.idAt(kv.key_ptr.*);
            if (std.mem.indexOfScalar(u32, l.exclude, id) != null or std.mem.indexOfScalar(u32, l.held.items, id) != null) continue;
            if (!l.past and idx.stateAt(kv.key_ptr.*) != .current) continue;
            if (kv.value_ptr.* < min_share * top) continue;
            try keys.append(a, lookup.orderKey(kv.value_ptr.*, idx.rankAt(kv.key_ptr.*)));
        }
        std.sort.pdq(u128, keys.items, {}, std.sort.asc(u128));
        const fresh = try a.alloc(u32, keys.items.len); // places, best first
        const shares = try a.alloc(f64, keys.items.len);
        for (keys.items, fresh, shares) |key, *p, *x| {
            p.* = lookup.placeOf(idx, key);
            x.* = score.get(p.*).? / top;
        }
        if (fresh.len == 0) return null;
        // the block that would hold them, counted by the model's tokenizer: the lowest goes while it is over the cap
        const kept = try placesOf(a, idx, if (l.place == .first) l.held.items else l.late.items);
        var n = fresh.len;
        // Failed prefixes shrink geometrically: total rendered note volume is

        // below twice the initial pool, rather than every descending prefix.

        // This may leave spare room; the exact tokenizer still enforces cap.
        while (n > 0) : (n = nextFitPrefix(n)) {
            if (l.check_ns + l.trigger_ns + clock.nowNs(l.io) - t0 >= l.budget_ns) {
                l.stop("the check time budget ran out");
                return null;
            }
            // Failed fits must not accumulate in the request arena. This

            // bounds temporary memory, independently of how many fits fail.
            var fit_arena = std.heap.ArenaAllocator.init(s.zeroing.allocator());
            defer fit_arena.deinit();
            const fit_a = fit_arena.allocator();
            const shown = try lookup.render(idx, sc, try std.mem.concat(fit_a, u32, &.{ kept, fresh[0..n] }));
            // (b) after a first block says what notes are: its head is not said again, the titles and lines follow it
            const late = l.place != .first and l.block.len > 0 and std.mem.startsWith(u8, shown.block, idx.head) and shown.block.len > idx.head.len;
            const block = if (late) shown.block[idx.head.len + 1 ..] else shown.block;
            const counted = text.encode(fit_a, block, false) catch |e| switch (e) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Template => return null,
            };
            const tokens = counted.len;
            if ((if (l.place == .first) tokens else l.block_tokens + tokens) > l.cap) continue;
            const ids = try a.alloc(u32, n);
            for (fresh[0..n], ids) |p, *id| id.* = idx.idAt(p);
            return .{ .ids = ids, .shares = shares[0..n], .block = try l.a.dupe(u8, block), .tokens = tokens, .lookup_ns = clock.nowNs(l.io) - t0 };
        }
        l.capped += 1;
        return null;
    }

    /// The continuation request: the prompt rendered with the new notes, the reply so far after it; null: none can be.
    pub fn continuation(l: *Live, srv: *Server, a: Allocator, collected: []const u32, hit: Hit) Allocator.Error!?*api.Request {
        if (l.place == .user) return l.spliced(srv, a, collected, hit);
        const p = l.prepared.?;
        const base = l.base.?;
        var cx: Cx = .{ .a = a };
        var input = p.input;
        input.messages = (try l.withNotes(srv, a, p.input.messages, hit.block)) orelse return null;
        const rendered = prompt_mod.prepare(srv, &cx, input, p.thinking, p.effort) catch |e| switch (e) {
            error.OutOfMemory => return error.OutOfMemory,
            error.Refused => {
                l.stop("the prompt with the new notes cannot be rendered");
                return null;
            },
        };
        const ids = try std.mem.concat(a, u32, &.{ rendered.ids, collected });
        var left = base.max_tokens -| @as(u32, @intCast(collected.len));
        const window = srv.info.context_window;
        if (window > 0) left = @min(left, @as(u32, window) -| @as(u32, @intCast(ids.len)));
        if (left == 0) return null;
        const req = try a.create(api.Request);
        req.* = base.*;
        req.prompt = ids;
        req.max_tokens = left;
        req.history_len = @intCast(rendered.history_len);
        req.rewind_len = @intCast(rendered.rewind_len);
        // the shared cuts and chunk starts as chat.prepare makes them, for the new prompt
        const system_len = prompt_mod.systemPrefixLen(srv, &cx, input.messages, input.tools, rendered.ids, p.thinking, p.effort);
        var shared: std.ArrayList(u32) = .empty;
        if (system_len > 0) for ([_]i64{ @as(i64, @intCast(system_len)) - 2048, @as(i64, @intCast(system_len)) - 512, @intCast(system_len) }) |cut| {
            if (cut >= 512) try shared.append(a, @intCast(cut));
        };
        req.shared_prefixes = shared.items;
        req.chunks = try chunk_plan.withCut(a, try srv.chunks.starts(a, ids), if (srv.chunks.step > 0) @intCast(@max(system_len, 1) - 1) else 0, ids.len, srv.chunks.min_chunk);
        try @import("notes_decode_retention.zig").continuation(srv, a, base, req);
        req.decode_spans = prompt_mod.replySpans(srv, &cx, ids) catch &.{};
        if (base.decide != null) req.decide = prompt_mod.decideOf(srv, rendered.ids);
        return req;
    }

    /// The prepared messages with `block` in: (a) in place of the first block, or (b) a system message after the last.
    fn withNotes(l: *Live, srv: *const Server, a: Allocator, messages: Value, block: []const u8) Allocator.Error!?Value {
        const list = messages.items() orelse return null;
        if (l.place == .turn) {
            const out = try a.alloc(Value, list.len + 1);
            @memcpy(out[0..list.len], list);
            out[list.len] = try message(a, srv.late_system, block);
            return .{ .array = out };
        }
        const was = l.prepared_block;
        if (was.len == 0) { // no first block: the notes are one now, first, as the first lookup puts one
            const lead = list.len > 0 and std.mem.eql(u8, list[0].strField("role") orelse "", "system");
            const out = try a.alloc(Value, list.len + @intFromBool(!lead));
            if (lead) {
                @memcpy(out, list);
                const first = try json.copyObject(a, list[0].object);
                try first.put(a, "content", .{ .string = try std.mem.concat(a, u8, &.{ block, "\n\n", list[0].strField("content") orelse "" }) });
                out[0] = .{ .object = first };
            } else {
                out[0] = try message(a, "system", block);
                @memcpy(out[1..], list);
            }
            return .{ .array = out };
        }
        // the system message that holds the block, first or before the question (notes_position)
        for (list, 0..) |m, i| {
            if (!std.mem.eql(u8, m.strField("role") orelse "", "system")) continue;
            const content = m.strField("content") orelse "";
            const at = std.mem.indexOf(u8, content, was) orelse continue;
            const o = try json.copyObject(a, m.object);
            try o.put(a, "content", .{ .string = try std.mem.concat(a, u8, &.{ content[0..at], block, content[at + was.len ..] }) });
            const out = try a.dupe(Value, list);
            out[i] = .{ .object = o };
            return .{ .array = out };
        }
        l.stop("no system message holds the notes block");
        return null;
    }

    /// (c): the new notes as a system message spliced in right before the question, where a state is kept.
    fn spliced(l: *Live, srv: *Server, a: Allocator, collected: []const u32, hit: Hit) Allocator.Error!?*api.Request {
        const p = l.prepared.?;
        const base = l.base.?;
        var cx: Cx = .{ .a = a };
        const list = p.input.messages.items() orelse return null;
        var last = list.len;
        while (last > 0) : (last -= 1) if (std.mem.eql(u8, list[last - 1].strField("role") orelse "", "user")) break;
        if (last < 2) {
            l.stop("no message before the question to splice the notes after");
            return null;
        }
        last -= 1;
        const before = prompt_mod.renderIds(srv, &cx, .{ .array = list[0..last] }, p.input.tools, p.thinking, p.effort, false) catch null;
        const upto = prompt_mod.renderIds(srv, &cx, .{ .array = list[0 .. last + 1] }, p.input.tools, p.thinking, p.effort, false) catch null;
        const with_list = try a.alloc(Value, last + 2);
        @memcpy(with_list[0 .. last + 1], list[0 .. last + 1]);
        with_list[last + 1] = try message(a, srv.late_system, hit.block);
        const with = prompt_mod.renderIds(srv, &cx, .{ .array = with_list }, p.input.tools, p.thinking, p.effort, false) catch null;
        const prompt = base.prompt;
        if (before == null or upto == null or with == null or !std.mem.startsWith(u32, prompt, before.?) or !std.mem.startsWith(u32, with.?, upto.?)) {
            l.stop("the template does not render the notes as a message of their own before the question");
            return null;
        }
        const u = before.?.len;
        const t = with.?[upto.?.len..];
        const head = try std.mem.concat(a, u32, &.{ prompt[0..u], t, prompt[u..] });
        const ids = try std.mem.concat(a, u32, &.{ head, collected });
        var left = base.max_tokens -| @as(u32, @intCast(collected.len));
        const window = srv.info.context_window;
        if (window > 0) left = @min(left, @as(u32, window) -| @as(u32, @intCast(ids.len)));
        if (left == 0) return null;
        const n: u32 = @intCast(t.len);
        const req = try a.create(api.Request);
        req.* = base.*;
        req.prompt = ids;
        req.max_tokens = left;
        req.history_len = if (base.history_len >= u) base.history_len + n else base.history_len;
        req.rewind_len = if (base.rewind_len >= u) base.rewind_len + n else base.rewind_len;
        var shared: std.ArrayList(u32) = .empty;
        for (base.shared_prefixes) |c| if (c <= u) try shared.append(a, c);
        req.shared_prefixes = shared.items;
        // the first request's chunk starts before the splice (a state kept at one is resumed), the question's, the rest
        var starts: std.ArrayList(u32) = .empty;
        if (srv.chunks.step > 0) {
            for (base.chunks) |c| if (c < u) try starts.append(a, c);
            try starts.append(a, @intCast(u));
            for (try srv.chunks.starts(a, ids)) |c| if (c >= u + srv.chunks.min_chunk) try starts.append(a, c);
        }
        req.chunks = starts.items;
        req.decode_spans = prompt_mod.replySpans(srv, &cx, ids) catch &.{};
        if (base.decide != null) req.decide = prompt_mod.decideOf(srv, head);
        return req;
    }

    /// Records an injection the reply is switching to.
    fn took(l: *Live, hit: Hit, at: usize, prompt: usize, build_ns: i96) Allocator.Error!void {
        try l.held.appendSlice(l.a, hit.ids);
        if (l.place != .first) {
            try l.late.appendSlice(l.a, hit.ids);
            l.late_block = hit.block;
            l.late_tokens = hit.tokens;
        } else {
            l.block = hit.block;
            l.block_tokens = hit.tokens;
        }
        const ids = try l.a.dupe(u32, hit.ids);
        const shares = try l.a.dupe(f64, hit.shares);
        try l.injections.append(l.a, .{ .ids = ids, .shares = shares, .at = at, .place = l.place, .block_tokens = l.block_tokens + l.late_tokens, .lookup_ns = hit.lookup_ns, .build_ns = build_ns, .prompt = prompt });
        if (l.injections.items.len >= most) l.stop("two injections");
    }

    /// The mailbox of the next segment (its request cancelled at `cancelled`); the old one's counters are carried.
    pub fn segment(l: *Live, srv: *const Server, old: *Mailbox, cancelled: i96) Allocator.Error!*Mailbox {
        l.submitted = clock.nowNs(srv.io);
        if (l.injections.items.len > 0) l.injections.items[l.injections.items.len - 1].drain_ns = l.submitted - cancelled;
        l.carried.rounds += old.stats.rounds;
        l.carried.drafted += old.stats.drafted;
        l.carried.accepted += old.stats.accepted;
        if (old.stats.min_rows > 0 and (l.carried.min_rows == 0 or old.stats.min_rows < l.carried.min_rows)) l.carried.min_rows = old.stats.min_rows;
        try l.boxes.ensureUnusedCapacity(l.a, 1);
        const box = try l.a.create(Mailbox);
        box.* = .{ .io = srv.io, .gpa = srv.gpa };
        l.boxes.appendAssumeCapacity(box);
        return box;
    }

    /// Tokens landed at `arrived` from `box`: the pause before them ends; `switching`: a new one starts after them.
    pub fn landed(l: *Live, id: []const u8, arrived: i96, box: *const Mailbox, switching: bool) void {
        if (l.pause_from) |from| if (box != l.first) {
            const inj = &l.injections.items[l.injections.items.len - 1];
            inj.pause_ns = arrived - from;
            inj.wait_ns = arrived - l.submitted;
            inj.cached = box.cached;
            l.pause_from = null;
            fields.logLine(id, inj);
        };
        if (switching) l.pause_from = arrived;
    }

    /// The last segment's mailbox as the reply's: the earlier segments' counters, the first one's prompt pass.
    pub fn carry(l: *const Live, m: *Mailbox) void {
        const f = l.first orelse return;
        if (m == f) return;
        m.stats.rounds += l.carried.rounds;
        m.stats.drafted += l.carried.drafted;
        m.stats.accepted += l.carried.accepted;
        if (l.carried.min_rows > 0 and (m.stats.min_rows == 0 or l.carried.min_rows < m.stats.min_rows)) m.stats.min_rows = l.carried.min_rows;
        m.stats.prefill_seconds = f.stats.prefill_seconds;
        m.cached = f.cached;
        m.prefilled_ns = f.prefilled_ns;
        std.mem.swap([]u32, &m.widths, &f.widths);
        std.mem.swap([]bool, &m.raised, &f.raised);
    }

    pub fn free(l: *Live) void {
        for (l.boxes.items) |b| b.deinit();
        l.boxes.clearRetainingCapacity();
    }

    /// The reply's notes.decode field.
    pub fn value(l: *const Live, a: Allocator) Allocator.Error!Value {
        return fields.value(l, a);
    }
};

const nextFitPrefix = @import("notes_decode_helpers.zig").nextFitPrefix;

/// Sets up the decode-time lookup of a chat request that asked for notes (`used`); null when it asked for none.
pub fn setup(srv: *Server, cx: *Cx, body: Value, used: ?*const notes_route.Used, messages: Value, shape: notes_route.Shape) errors.Refused!?*Live {
    const store = srv.notes orelse return null;
    const ask = try asked(cx, body, used != null);
    const u = used orelse return null;
    const notes_ask = (try notes_route.asked(cx, body)) orelse notes_route.Ask{};
    const l = try cx.a.create(Live);
    // (b) needs a template that keeps a later system message in place; a template that makes it a user turn gets (a)
    const place: Place = if (ask.place != .first and !std.mem.eql(u8, srv.late_system, "system")) .first else ask.place;
    l.* = .{ .a = cx.a, .on = ask.on, .store = store, .mutex = &srv.notes_mutex, .io = srv.io, .ver = readFrom(u), .question = notes_route.lastUser(messages), .past = u.past, .cap = u.cap, .exclude = notes_ask.exclude, .place = place, .echo = u.echo, .dry = ask.dry, .names = ask.names, .max_checks = ask.checks, .budget_ns = @intFromFloat(ask.budget_ms * 1e6), .gap = ask.gap, .shape = shape, .block = u.found.block, .prepared_block = u.found.block, .block_tokens = if (u.found.block.len > 0) u.found.tokens else 0 };
    try l.held.appendSlice(cx.a, u.found.ids);
    return l;
}

const readFrom = @import("notes_decode_helpers.zig").readFrom;
const named = @import("notes_decode_helpers.zig").named;
const scoreOf = @import("notes_decode_helpers.zig").scoreOf;
const addNotes = @import("notes_decode_helpers.zig").addNotes;
const message = @import("notes_decode_helpers.zig").message;
const plain = @import("notes_decode_helpers.zig").plain;
const placesOf = @import("notes_decode_helpers.zig").placesOf;
