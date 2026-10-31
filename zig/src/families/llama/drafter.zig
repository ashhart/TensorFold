//! A smaller Llama drafting for the served one as the lane core's priced proposer; the target's windows verify it.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const cfg = @import("config.zig");
const st = @import("state.zig");
const fwd = @import("forward.zig");
const snapshot = @import("snapshot.zig");
const Model = @import("model.zig").Model;
const Proposer = lanes.proposer.Proposer;

/// Caches of up to `capacity` tokens for `streams` streams (one a request, kept between requests for its prefix).
pub const Options = struct { capacity: usize, streams: u32 };

/// Rows a slot's cache is made or grown in, so a short chat holds a short cache (the target sizes its own by request).
const grow = 4096;

/// Token slots the chained steps draw into; a round takes at most window_rows of them.
const ring = 4096;
/// Drafts a round until the rates are measured.
const first_depth: usize = 6;
/// The newest sample's weight in the running times.
const rate = 0.1;
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
        cache: ?st.Cache = null, // made and grown by `grow` rows as the context needs, up to the capacity
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

    pub fn deinit(d: *Drafter) void {
        for (d.slots) |*s| {
            if (s.cache) |*c| c.deinit();
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
        if (prompt.len + 1 > d.options.capacity or !try d.room(slot, @min(prompt.len + grow, d.options.capacity))) return error.LlamaDrafterPromptTooLong;
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

    /// More than one stream holds a slot: their rounds are shared, and every stream's step waits on each one's drafts.
    fn sharing(d: *const Drafter) bool {
        var n: usize = 0;
        for (d.slots) |s| n += @intFromBool(s.taken);
        return n > 1;
    }

    /// The slot's cache holds `needed` rows, made or grown (its state copied) in `grow` rows; false past the capacity.
    fn room(d: *Drafter, slot: *Slot, needed: usize) !bool {
        if (slot.cache) |c| if (c.capacity >= needed) return true;
        if (needed > d.options.capacity) return false;
        const doubled = if (slot.cache) |c| 2 * c.capacity else 0;
        const capacity = @min(d.options.capacity, std.mem.alignForward(usize, @max(needed, doubled), grow));
        var bigger = try st.Cache.init(d.model.device, &d.model.config, capacity);
        errdefer bigger.deinit();
        if (slot.cache) |*c| {
            var snap = try snapshot.save(d.model.device, d.model.queue, &d.model.config, c);
            defer snap.free();
            try snapshot.restore(d.model.queue, &snap, &bigger);
            c.deinit();
        }
        bigger.setResident(d.follow);
        slot.cache = bigger;
        return true;
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

fn ema(kept: ?f64, sample: f64) f64 {
    return if (kept) |k| k + rate * (sample - k) else sample;
}

/// A round's milliseconds by its window's rows, a line fitted to the recent rounds (a window costs more with rows).
pub const Fit = struct {
    samples: u32 = 0,
    x: f64 = 0, // rows
    y: f64 = 0, // ms
    xx: f64 = 0,
    xy: f64 = 0,

    pub fn add(f: *Fit, rows: f64, ms: f64) void {
        if (f.samples == 0) {
            f.* = .{ .samples = 1, .x = rows, .y = ms, .xx = rows * rows, .xy = rows * ms };
            return;
        }
        f.samples += 1;
        f.x = ema(f.x, rows);
        f.y = ema(f.y, ms);
        f.xx = ema(f.xx, rows * rows);
        f.xy = ema(f.xy, rows * ms);
    }

    /// Milliseconds a window of `rows` takes: the fitted line, or the mean while the rows seen hardly vary.
    pub fn at(f: Fit, rows: f64) f64 {
        const spread = f.xx - f.x * f.x;
        if (spread < 0.25) return f.y;
        const slope = @max(0, (f.xy - f.x * f.y) / spread);
        return @max(f.y + slope * (rows - f.x), 0.5 * f.y);
    }
};

/// The depth with the most tokens a millisecond: a round lands 1 + p + ... + p^k tokens for k + 1 rows and k steps.
pub fn depth(p: f64, rounds: Fit, step_ms: f64, most: usize) usize {
    var best: usize = 1;
    var best_rate: f64 = 0;
    var expect: f64 = 1;
    var pk: f64 = 1;
    for (1..most + 1) |k| {
        pk *= p;
        expect += pk;
        const kf: f64 = @floatFromInt(k);
        const r = expect / (rounds.at(kf + 1) + kf * step_ms);
        if (r > best_rate) {
            best_rate = r;
            best = k;
        }
    }
    return best;
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

    /// Drafts a round at the measured rates (first_depth before they are), within the ask and the window.
    pub fn want(r: *const Request, max_draft: i64) usize {
        const most: usize = @intCast(@min(max_draft, @as(i64, st.window_rows - 1)));
        const step = r.d.step_ms orelse return @min(first_depth, most);
        if (r.d.rounds.samples == 0) return @min(first_depth, most);
        return depth(r.kept / r.tried, r.d.rounds, step, most);
    }

    /// The drafter's continuation of `context`, up to `max_draft` tokens, cut after an eos; borrowed until the next.
    pub fn propose(r: *Request, context: []const u32, max_draft: i64) ![]const u32 {
        if (r.asked) |a| if (a.len == context.len and a.max_draft == max_draft) return r.out.items;
        r.out.clearRetainingCapacity();
        r.asked = .{ .len = context.len, .max_draft = max_draft };
        r.returned = null; // an empty answer times no round
        // shared rounds go undrafted: two drafting streams decoded slower together than two plain ones
        if (max_draft <= 0 or r.unfit or context.len == 0 or r.d.sharing()) return r.out.items;
        const d = r.d;
        const slot = &d.slots[r.slot];
        const n = r.want(max_draft);
        // the cache keeps the context's prefix it agrees with (an earlier round's drafts past the context included)
        const shared = commonPrefix(slot.seen.items, context);
        if (shared < context.len and shared < slot.seen.items.len) {
            slot.seen.shrinkRetainingCapacity(shared);
            slot.cache.?.len = shared;
            slot.next = null;
        }
        // room for the context, the drafts and the draw after them
        if (!try d.room(slot, @max(context.len, slot.seen.items.len) + n + 1)) return r.out.items;
        const cache = &slot.cache.?;
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
        if (r.returned) |t| r.d.rounds.add(@floatFromInt(rows), now() - t);
        const offered = rows - 1;
        r.offered += offered;
        r.landed += kept;
        const k: f64 = @floatFromInt(kept);
        r.kept = decay * r.kept + k;
        r.tried = decay * r.tried + k + @as(f64, if (kept < offered) 1.0 else 0.0);
    }
};

test "llama drafter: the depth pays for its steps and rows, more at a higher acceptance, none past the ask" {
    // flat 60 ms rounds and 3 ms steps: at p 0.7 drafting about 5-6 pays best; at 0.95 the window's 15; at 0.1 one
    var flat: Fit = .{};
    flat.add(4, 60);
    try std.testing.expect(depth(0.7, flat, 3, 15) >= 4 and depth(0.7, flat, 3, 15) <= 7);
    try std.testing.expectEqual(@as(usize, 15), depth(0.95, flat, 3, 15));
    try std.testing.expectEqual(@as(usize, 1), depth(0.1, flat, 3, 15));
    try std.testing.expectEqual(@as(usize, 3), depth(0.95, flat, 3, 3));
    // dear steps pull the depth down
    try std.testing.expect(depth(0.7, flat, 30, 15) <= 2);
    // rounds that grow 5 ms a row (as the M5's windows do) pull it down too
    var sloped: Fit = .{};
    for (0..40) |i| sloped.add(@floatFromInt(2 + i % 8), 50 + 5 * @as(f64, @floatFromInt(2 + i % 8)));
    try std.testing.expectApproxEqAbs(@as(f64, 100), sloped.at(10), 3);
    try std.testing.expect(depth(0.7, sloped, 3, 15) < depth(0.7, flat, 3, 15));
    try std.testing.expect(depth(0.95, sloped, 3, 15) >= 8);
    // one row count seen: the mean stands in for the line
    var one: Fit = .{};
    one.add(4, 60);
    one.add(4, 70);
    try std.testing.expectApproxEqAbs(one.y, one.at(12), 1e-9);
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
