//! A smaller Llama drafting for the served one as the lane core's priced proposer; the target's windows verify it.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const budget = @import("budget.zig");
const backend = @import("backend.zig");
const Model = @import("model.zig").Model;
const Proposer = lanes.proposer.Proposer;
const pricing = @import("pricing.zig");
const Fit = pricing.Fit;
const depth = pricing.depth;
const depthShared = pricing.depthShared;
const ema = pricing.ema;

/// Up to `capacity` tokens for each of `streams` caches (one a request, kept for its prefix), in the target's ledger.
pub const Options = struct { capacity: usize, streams: u32, ledger: *budget.Ledger };

/// Token slots the chained steps draw into; a round takes at most window_rows of them.
const ring = 4096;
/// Drafts a round until the rates are measured.
const first_depth: usize = 6;
/// The acceptance prior (two drafts of three) and how much of the history a round keeps.
const prior_kept = 2.0;
const prior_tried = 3.0;
const decay = 0.9;

/// A cache a stream synced to its context by longest common prefix; greedy steps chained on the GPU; a measured depth.
pub const Drafter = struct {
    gpa: std.mem.Allocator,
    model: *Model, // owned: freed with the drafter
    options: Options,
    scratch: st.Scratch, // chained steps and short deltas
    prompt: st.Scratch, // prompt-sized deltas
    tokens: mtl.Buffer, // u32 [ring]
    next: u64 = 0, // the next free token slot
    slots: []Slot,
    follow: ?mtl.ResidencySet.Follow = null,
    step_ms: ?f64 = null, // one chained step, wall (null: not measured yet)
    rounds: Fit = .{}, // the target's round by its window's rows, from a proposal's return to its outcome

    /// A stream's cache and what it holds; a freed slot keeps both for the next request sharing its prefix.
    const Slot = struct {
        cache: ?st.Cache = null, // made or grown when its stream is admitted, to that stream's capacity
        seen: std.ArrayList(u32) = .empty, // the tokens whose keys and values the cache holds, in order
        next: ?u32 = null, // the token drawn after `seen` (the last fed row's argmax)
        taken: bool = false,
    };

    pub fn init(gpa: std.mem.Allocator, model: *Model, options: Options) !*Drafter {
        const d = try gpa.create(Drafter);
        errdefer gpa.destroy(d);
        var scratch = try st.Scratch.init(model.device, &model.config, st.batch_rows);
        errdefer scratch.deinit();
        var prompt = try st.Scratch.init(model.device, &model.config, st.prompt_rows);
        errdefer prompt.deinit();
        const tokens = try model.device.buffer(ring * 4, mtl.ResourceOptions.shared);
        errdefer tokens.deinit();
        const slots = try gpa.alloc(Slot, @max(options.streams, 1));
        for (slots) |*s| s.* = .{};
        d.* = .{ .gpa = gpa, .model = model, .options = options, .scratch = scratch, .prompt = prompt, .tokens = tokens, .slots = slots };
        return d;
    }

    /// A drafter for `model`, warmed: it owns the model from the call on, freed with it if the warm-up fails.
    pub fn start(gpa: std.mem.Allocator, model: *Model, options: Options) !*Drafter {
        const d = init(gpa, model, options) catch |err| {
            model.deinit();
            return err;
        };
        return started(Drafter, d);
    }

    pub fn deinit(d: *Drafter) void {
        for (d.slots) |*s| {
            d.drop(s);
            s.seen.deinit(d.gpa);
        }
        d.gpa.free(d.slots);
        d.tokens.deinit();
        d.scratch.deinit();
        d.prompt.deinit();
        d.model.deinit();
        d.gpa.destroy(d);
    }

    /// Bytes the drafter holds beside its weights and caches: its scratches and token slots.
    pub fn fixedBytes(c: *const cfg.Config) u64 {
        return st.scratchBytes(c, st.batch_rows) + st.scratchBytes(c, st.prompt_rows) + ring * 4;
    }

    pub fn setResident(d: *Drafter, hook: ?mtl.ResidencySet.Follow) void {
        if (d.follow) |f| {
            mtl.resident.fields.apply(&d.scratch, f, false);
            mtl.resident.fields.apply(&d.prompt, f, false);
        }
        d.follow = hook;
        if (hook) |f| {
            mtl.resident.fields.apply(&d.scratch, f, true);
            mtl.resident.fields.apply(&d.prompt, f, true);
        }
        for (d.slots) |*s| if (s.cache) |*c| c.setResident(hook);
    }

    /// A prompt chunk and a chained step once, so the first request skips the GPU's first touch.
    pub fn warm(d: *Drafter) !void {
        var cache = try st.Cache.init(d.model.device, &d.model.config, st.prompt_rows + st.window_rows);
        defer cache.deinit();
        var ids: [st.prompt_rows]u32 = undefined;
        for (&ids, 0..) |*t, i| t.* = @intCast((1000 + i * 37) % d.model.config.vocab);
        try d.feed(&cache, &ids, null);
        var flights: [1]mtl.CommandBuffer = undefined;
        try d.step(&cache, ids[0], &flights[0]);
        try land(&flights);
    }

    /// The lane host's proposer factory: a request takes the free slot sharing the longest prefix with its prompt.
    pub fn make(ctx: *anyopaque, gpa: std.mem.Allocator, prompt: []const u32, eos: []const u32) anyerror!Proposer {
        const d: *Drafter = @ptrCast(@alignCast(ctx));
        var best: ?usize = null;
        var best_len: usize = 0;
        for (d.slots, 0..) |*s, i| {
            if (s.taken) continue;
            const shared = commonPrefix(s.seen.items, prompt);
            if (best == null or shared > best_len) {
                best = i;
                best_len = shared;
            }
        }
        const i = best orelse return error.LlamaDrafterBusy;
        const slot = &d.slots[i];
        if (prompt.len + 1 > d.options.capacity) return error.LlamaDrafterPromptTooLong;
        const r = try gpa.create(Request);
        r.* = .{ .d = d, .gpa = gpa, .slot = i, .eos = eos };
        slot.taken = true;
        return r.proposer();
    }

    pub fn free(ctx: *anyopaque, p: Proposer) void {
        const d: *Drafter = @ptrCast(@alignCast(ctx));
        const r: *Request = @ptrCast(@alignCast(p.ptr));
        d.slots[r.slot].taken = false;
        if (r.rounds > 0) std.log.debug("llama drafter: {d} rounds, {d} drafts offered, {d} kept, {d:.2} tokens a round", .{ r.rounds, r.offered, r.landed, @as(f64, @floatFromInt(r.rounds + r.landed)) / @as(f64, @floatFromInt(r.rounds)) });
        r.out.deinit(r.gpa);
        r.gpa.destroy(r);
    }

    /// The streams holding a slot: more than one shares each round, every stream's step waiting on each one's drafts.
    fn sharers(d: *const Drafter) usize {
        var n: usize = 0;
        for (d.slots) |s| n += @intFromBool(s.taken);
        return n;
    }

    /// The target's admission hooks: a stream's slot grows with it, in its ledger.
    pub fn slotsHook(d: *Drafter) backend.Slots {
        return .{ .ctx = d, .growth = growthFn, .prepare = prepareFn, .shed = shedFn };
    }

    /// The slot `proposer` drafts from (null: something else's proposer, or none).
    fn slotFor(d: *Drafter, proposer: ?Proposer) ?*Slot {
        const p = proposer orelse return null;
        if (p.vtable != &Request.vtable) return null;
        const r: *Request = @ptrCast(@alignCast(p.ptr));
        return if (r.d == d) &d.slots[r.slot] else null;
    }

    /// Rows a stream of `capacity` tokens drafts within: its context, a window of drafts and the draw after them.
    fn rowsFor(d: *const Drafter, capacity: usize) usize {
        return @min(capacity + st.window_rows, d.options.capacity);
    }

    fn growthFn(ctx: *anyopaque, stream: *lanes.Stream, capacity: usize) u64 {
        const d: *Drafter = @ptrCast(@alignCast(ctx));
        const slot = d.slotFor(stream.proposer) orelse return 0;
        return growth(&d.model.config, if (slot.cache) |c| .{ .capacity = c.capacity, .len = c.len } else null, d.rowsFor(capacity));
    }

    fn prepareFn(ctx: *anyopaque, stream: *lanes.Stream, capacity: usize) anyerror!void {
        const d: *Drafter = @ptrCast(@alignCast(ctx));
        try d.prepare(stream.proposer, capacity);
    }

    /// The proposer's slot grown for a request of `capacity` tokens, as the target's admission grows a stream's.
    pub fn prepare(d: *Drafter, proposer: ?Proposer, capacity: usize) !void {
        const slot = d.slotFor(proposer) orelse return;
        try d.grow(slot, d.rowsFor(capacity));
    }

    fn shedFn(ctx: *anyopaque, bytes: u64) void {
        const d: *Drafter = @ptrCast(@alignCast(ctx));
        for (d.slots) |*s| {
            if (d.options.ledger.fits(bytes)) return;
            if (!s.taken) d.drop(s);
        }
    }

    /// A slot's cache and what it knew, freed and given back to the ledger.
    fn drop(d: *Drafter, slot: *Slot) void {
        if (slot.cache) |*c| {
            d.options.ledger.give(st.cacheBytes(&d.model.config, c.capacity));
            c.deinit();
        }
        slot.cache = null;
        slot.seen.clearRetainingCapacity();
        slot.next = null;
    }

    /// The slot's cache made or grown to `rows` (its state copied), every byte taken from the ledger as it is held.
    fn grow(d: *Drafter, slot: *Slot, rows: usize) !void {
        if (slot.cache) |c| if (c.capacity >= rows) return;
        const c = &d.model.config;
        const ledger = d.options.ledger;
        const made = st.cacheBytes(c, rows);
        ledger.take(made);
        errdefer ledger.give(made);
        var bigger = try st.Cache.init(d.model.device, c, rows);
        errdefer bigger.deinit();
        if (slot.cache) |*old| {
            const copy = snapshot.bytes(c, old.len);
            ledger.take(copy);
            defer ledger.give(copy);
            var snap = try snapshot.save(d.model.device, d.model.queue, c, old);
            defer snap.free();
            try snapshot.restore(d.model.queue, &snap, &bigger);
            ledger.give(st.cacheBytes(c, old.capacity));
            old.deinit();
        }
        bigger.setResident(d.follow);
        slot.cache = bigger;
    }

    /// `ids` at the cache's end in prompt chunks, waited on, the last chunk's row drawn into `draw` when given.
    fn feed(d: *Drafter, cache: *st.Cache, ids: []const u32, draw: ?*mtl.CommandBuffer) !void {
        var at: usize = 0;
        while (at < ids.len) {
            const end = @min(ids.len, at + st.prompt_rows);
            const s = if (end - at > st.batch_rows) &d.prompt else &d.scratch;
            const last = end == ids.len;
            if (last and draw != null) {
                const out = d.take();
                try fwd.run(d.model, s, cache, ids[at..end], .{ .draw = .{ .how = .greedy, .out = d.slotOf(out) }, .flight = draw });
            } else try fwd.run(d.model, s, cache, ids[at..end], .{});
            cache.len += end - at;
            at = end;
        }
    }

    /// One row from the host into a fresh slot, its argmax drawn into the slot after it; committed into `flight`.
    fn step(d: *Drafter, cache: *st.Cache, row: u32, flight: *mtl.CommandBuffer) !void {
        const in = d.take();
        d.tokens.slice(u32, ring)[in % ring] = row;
        const out = d.take();
        try fwd.run(d.model, &d.scratch, cache, &.{0}, .{ .ids = d.slotOf(in), .draw = .{ .how = .greedy, .out = d.slotOf(out) }, .flight = flight });
        cache.len += 1;
    }

    /// A step whose row is a drawn slot (the previous step's), the next drawn into a fresh slot.
    fn chain(d: *Drafter, cache: *st.Cache, in: u64, flight: *mtl.CommandBuffer) !void {
        const out = d.take();
        try fwd.run(d.model, &d.scratch, cache, &.{0}, .{ .ids = d.slotOf(in), .draw = .{ .how = .greedy, .out = d.slotOf(out) }, .flight = flight });
        cache.len += 1;
    }

    fn take(d: *Drafter) u64 {
        const h = d.next;
        d.next += 1;
        return h;
    }

    fn slotOf(d: *Drafter, h: u64) fwd.B {
        return .{ .buf = d.tokens, .off = @intCast(h % ring * 4) };
    }

    fn token(d: *Drafter, h: u64) u32 {
        return d.tokens.slice(u32, ring)[h % ring];
    }
};

/// What a slot's cache holds now: its capacity and the rows written.
pub const Held = struct { capacity: usize, len: usize };

/// Bytes a slot holding `have` takes to grow to `rows`: the new cache, and the old state's copy while it moves.
pub fn growth(c: *const cfg.Config, have: ?Held, rows: usize) u64 {
    if (have) |h| if (h.capacity >= rows) return 0;
    return st.cacheBytes(c, rows) + if (have) |h| snapshot.bytes(c, h.len) else 0;
}

/// `d` warmed, or freed when its warm-up fails: nothing made before the warm-up outlives a failed start.
fn started(comptime D: type, d: *D) !*D {
    errdefer d.deinit();
    try d.warm();
    return d;
}

/// Wait for every step in flight, oldest first, and release them.
fn land(flights: []mtl.CommandBuffer) !void {
    var failed = false;
    for (flights) |cb| {
        cb.wait();
        if (cb.failure()) |msg| {
            std.log.err("llama drafter step: {s}", .{msg});
            failed = true;
        }
        cb.release();
    }
    if (failed) return error.GpuFailed;
}

fn commonPrefix(a: []const u32, b: []const u32) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) i += 1;
    return i;
}

fn now() f64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(f64, @floatFromInt(ts.sec)) * 1e3 + @as(f64, @floatFromInt(ts.nsec)) * 1e-6;
}

/// One stream's proposer: its slot, its acceptance and what the last round asked.
pub const Request = struct {
    d: *Drafter,
    gpa: std.mem.Allocator,
    slot: usize,
    eos: []const u32,
    out: std.ArrayList(u32) = .empty,
    asked: ?struct { len: usize, max_draft: i64 } = null,
    returned: ?f64 = null, // when the last proposal returned (ms), for the round time
    kept: f64 = prior_kept, // decayed drafts landed
    tried: f64 = prior_tried, // decayed drafts judged: the landed ones and the first to break
    rounds: u64 = 0,
    offered: u64 = 0,
    landed: u64 = 0,
    unfit: bool = false, // a context token past the drafter's vocabulary: no proposals for this request
    alone: bool = true, // the last proposal's round had this stream alone: its time fits the rounds' line

    pub fn proposer(r: *Request) Proposer {
        return .{ .ptr = r, .vtable = &vtable };
    }

    const vtable: Proposer.VTable = .{ .propose = proposeFn, .last_match = lastMatch, .priced = true, .round = roundFn };

    fn self(p: *anyopaque) *Request {
        return @ptrCast(@alignCast(p));
    }

    fn proposeFn(p: *anyopaque, context: []const u32, max_draft: i64) anyerror![]const u32 {
        return self(p).propose(context, max_draft);
    }

    fn lastMatch(_: *anyopaque) i64 {
        return 0;
    }

    fn roundFn(p: *anyopaque, rows: u32, kept: u32) void {
        self(p).round(rows, kept);
    }

    /// Drafts a round at the measured rates (first_depth before), within the ask and window; shared rounds priced.
    pub fn want(r: *const Request, max_draft: i64) usize {
        const most: usize = @intCast(@min(max_draft, @as(i64, st.window_rows - 1)));
        const step = r.d.step_ms orelse return @min(first_depth, most);
        if (r.d.rounds.samples == 0) return @min(first_depth, most);
        const streams = r.d.sharers();
        if (streams > 1) return depthShared(r.kept / r.tried, r.d.rounds, step, most, streams);
        return depth(r.kept / r.tried, r.d.rounds, step, most);
    }

    /// The drafter's continuation of `context`, up to `max_draft` tokens, cut after an eos; borrowed until the next.
    pub fn propose(r: *Request, context: []const u32, max_draft: i64) ![]const u32 {
        if (r.asked) |a| if (a.len == context.len and a.max_draft == max_draft) return r.out.items;
        r.out.clearRetainingCapacity();
        r.asked = .{ .len = context.len, .max_draft = max_draft };
        r.returned = null; // an empty answer times no round
        if (max_draft <= 0 or r.unfit or context.len == 0) return r.out.items;
        const d = r.d;
        const slot = &d.slots[r.slot];
        const n = r.want(max_draft);
        if (n == 0) return r.out.items; // the shared round's drafts cost more than they land
        r.alone = d.sharers() <= 1;
        // the cache keeps the context's prefix it agrees with (an earlier round's drafts past the context included)
        const shared = commonPrefix(slot.seen.items, context);
        if (shared < context.len and shared < slot.seen.items.len) {
            slot.seen.shrinkRetainingCapacity(shared);
            slot.cache.?.len = shared;
            slot.next = null;
        }
        // room for the context, the drafts and the draw after them: the slot grew to its stream's capacity at admission
        const cache = if (slot.cache) |*c| c else return r.out.items;
        if (@max(context.len, slot.seen.items.len) + n + 1 > cache.capacity) return r.out.items;
        var flights: [st.window_rows]mtl.CommandBuffer = undefined;
        var draws: [st.window_rows]u64 = undefined; // each pass's drawn slot, in order
        var count: usize = 0;
        if (slot.seen.items.len < context.len) {
            // the rest of the context, its last row drawn; waited on, so the chain's time is its own
            const delta = context[slot.seen.items.len..];
            for (delta) |t| if (t >= d.model.config.vocab) {
                r.unfit = true;
                return r.out.items;
            };
            if (slot.seen.items.len + delta.len + n > cache.capacity) return r.out.items;
            try d.feed(cache, delta, &flights[0]);
            draws[0] = d.next - 1;
            count = 1;
            try slot.seen.appendSlice(d.gpa, delta);
            try land(flights[0..1]);
        } else if (slot.next == null) return error.LlamaDrafterUndrawn;
        const fed = slot.seen.items.len - context.len; // an earlier round's drafts the cache holds past the context
        var more = n -| (fed + 1); // chained steps past the drawn token
        if (slot.seen.items.len + more > cache.capacity) more = 0;
        if (more > 0) {
            const t0 = now();
            const first = count;
            if (count == 0) {
                // the drawn token is the host's copy: its step starts the chain
                try d.step(cache, slot.next.?, &flights[0]);
                draws[0] = d.next - 1;
                try slot.seen.append(d.gpa, slot.next.?);
                count = 1;
                more -= 1;
            }
            for (0..more) |_| {
                try d.chain(cache, draws[count - 1], &flights[count]);
                draws[count] = d.next - 1;
                count += 1;
            }
            try land(flights[first..count]);
            d.step_ms = ema(d.step_ms, (now() - t0) / @as(f64, @floatFromInt(count - first)));
        }
        if (count > 0) {
            // every pass but the last fed its draw to the next; the last draw is the token after `seen`
            for (draws[0 .. count - 1]) |h| try slot.seen.append(d.gpa, d.token(h));
            slot.next = d.token(draws[count - 1]);
        }
        const available = slot.seen.items.len - context.len;
        try r.out.appendSlice(r.gpa, slot.seen.items[context.len..][0..@min(available, n)]);
        if (r.out.items.len < n) try r.out.append(r.gpa, slot.next.?);
        for (r.out.items, 0..) |t, i| if (std.mem.indexOfScalar(u32, r.eos, t) != null) {
            r.out.shrinkRetainingCapacity(i + 1);
            break;
        };
        r.returned = now();
        return r.out.items;
    }

    /// The round's outcome: `rows` the window's rows, `kept` its drafts landed (no ask this round: a plain step).
    pub fn round(r: *Request, rows: u32, kept: u32) void {
        r.rounds += 1;
        defer r.asked = null;
        if (r.asked == null or rows < 2) return;
        if (r.returned) |t| if (r.alone) r.d.rounds.add(@floatFromInt(rows), now() - t);
        const offered = rows - 1;
        r.offered += offered;
        r.landed += kept;
        const k: f64 = @floatFromInt(kept);
        r.kept = decay * r.kept + k;
        r.tried = decay * r.tried + k + @as(f64, if (kept < offered) 1.0 else 0.0);
    }
};

test "llama drafter: a warm-up that fails frees the drafter it was made for" {
    const Fake = struct {
        freed: *bool,
        fn warm(_: *@This()) !void {
            return error.WarmUpFailed;
        }
        fn deinit(f: *@This()) void {
            f.freed.* = true;
        }
    };
    var freed = false;
    var f: Fake = .{ .freed = &freed };
    try std.testing.expectError(error.WarmUpFailed, started(Fake, &f));
    try std.testing.expect(freed);
}

test "llama drafter: a slot's growth counts the new cache and the old state's copy, nothing when it holds enough" {
    const c = try cfg.parse(std.testing.allocator, @embedFile("fixtures/drafter_config.json"));
    try std.testing.expectEqual(@as(u64, st.cacheBytes(&c, 9000)), growth(&c, null, 9000));
    try std.testing.expectEqual(@as(u64, st.cacheBytes(&c, 9000) + snapshot.bytes(&c, 3000)), growth(&c, .{ .capacity = 4096, .len = 3000 }, 9000));
    try std.testing.expectEqual(@as(u64, 0), growth(&c, .{ .capacity = 9000, .len = 3000 }, 9000));
}

test "llama drafter: the common prefix and the acceptance history" {
    try std.testing.expectEqual(@as(usize, 3), commonPrefix(&.{ 1, 2, 3, 4 }, &.{ 1, 2, 3, 9, 9 }));
    try std.testing.expectEqual(@as(usize, 2), commonPrefix(&.{ 1, 2 }, &.{ 1, 2, 3 }));
    try std.testing.expectEqual(@as(usize, 0), commonPrefix(&.{}, &.{1}));
    var d: Drafter = undefined;
    var r: Request = .{ .d = &d, .gpa = std.testing.allocator, .slot = 0, .eos = &.{} };
    r.asked = .{ .len = 10, .max_draft = 15 };
    r.round(7, 6); // six drafts, all landed: no break counted
    try std.testing.expectApproxEqAbs(decay * prior_kept + 6, r.kept, 1e-9);
    try std.testing.expectApproxEqAbs(decay * prior_tried + 6, r.tried, 1e-9);
    try std.testing.expect(r.asked == null);
    r.asked = .{ .len = 17, .max_draft = 15 };
    r.round(7, 2); // two of six: the third broke
    try std.testing.expectEqual(@as(u64, 12), r.offered);
    try std.testing.expectEqual(@as(u64, 8), r.landed);
    r.round(1, 0); // a plain step: no history
    try std.testing.expectEqual(@as(u64, 3), r.rounds);
    try std.testing.expectEqual(@as(u64, 12), r.offered);
}
