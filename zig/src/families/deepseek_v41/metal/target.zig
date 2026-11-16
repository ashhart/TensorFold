//! The lanes' verify target on Metal: a slot's prompt (prompt.zig), its windows of rows (forward.zig), keeps and reuse.
const std = @import("std");
const mtl = @import("metal");
const host = @import("dsv41");
const kv = @import("kv.zig");
const rope = @import("rope.zig");
const launch_ = @import("target_launch.zig");
const scratch = @import("scratch.zig");
const forward = @import("forward.zig");
const prompt = @import("prompt.zig");
const pipelines = @import("pipelines.zig");
const snapshots = @import("snapshots.zig");
const tree_ = @import("tree.zig");
const steps = @import("prompt_steps.zig");
const picks = @import("picks.zig");
const vision = @import("vision.zig");
const vw = @import("vision_weights.zig");
const vt = @import("vision_tower.zig");
const vrows = @import("vision_rows.zig");
const prep = @import("vision_prep.zig");
const pick = @import("pick.zig");
const iface = host.iface;
const lanes = host.lanes;
const Config = host.config.Config;
const Allocator = std.mem.Allocator;
const R = pipelines.max_rows;
/// The shortest system part kept on its own for later conversations.
const min_shared: usize = 256;
/// What the target reads and borrows: the device, kernels, RoPE tables, layer scratch, weights and Engram rows.
pub const Env = struct {
    device: mtl.Device,
    queue: mtl.Queue,
    c: *const Config,
    p: *const pipelines.Pipelines,
    tables: *const rope.Tables,
    s: *const scratch.Scratch,
    model: forward.Model,
    lookup: ?forward.Lookup,
    /// the vision tower's kernels and weights (null: text only; a prompt with images is refused)
    vision: ?VisionRefs = null,
};
pub const VisionRefs = struct { kit: *const vision.Kit, w: *const vw.Weights };
pub const Options = struct {
    slots: u32,
    /// positions a slot holds (the context limit)
    limit: u32,
    prompt: prompt.Opts = .{},
    /// bytes of kept prompt states (0: no prompt cache)
    cache_bytes: u64 = 0,
    /// the weights' residency owner, which the target's own buffers join inside its load limit (null: none join)
    resident: ?*mtl.resident.Resident = null,
};
/// The drafter's work behind a window: `queue` right after its command buffers, `drain` before the target writes.
pub const Behind = struct {
    ctx: *anyopaque,
    queue: *const fn (ctx: *anyopaque, group: []const iface.Segment, logits: mtl.Buffer, off: usize, taps: mtl.Buffer) void,
    drain: *const fn (ctx: *anyopaque) anyerror!void,
};
/// What the target did, for the tests and the logs; `most_between`: the most launches between two of a prompt's pieces.
pub const Stats = struct { prompts: u64 = 0, prompt_rows: u64 = 0, hits: u64 = 0, cached_rows: u64 = 0, windows: u64 = 0, rows: u64 = 0, rewinds: u64 = 0, launches: u64 = 0, history_rows: u64 = 0, pieces: u64 = 0, abandoned: u64 = 0, most_between: u64 = 0 };

pub const Slot = @import("target_slot.zig").Slot;
pub const Target = struct {
    a: Allocator,
    env: Env,
    opt: Options,
    b: forward.Bufs,
    pr: prompt.Prompt,
    slots: []Slot,
    cache: snapshots.Store,
    stats: Stats = .{},
    /// a command buffer failed: the device state is no longer trusted (the health hook)
    failed: bool = false,
    /// the target's own buffers joined the weights' residency owner (they leave it before they are freed)
    joined: bool = false,
    /// draft trees: the branches' shadow slots and the open tree window (tree.zig)
    tree: tree_.Tree = .{},
    /// the ViT and aligner, when the environment has a tower
    tower: ?vt.Tower = null,
    /// sampled picks' float64 logits (picks.rows rows) and the ids 0 .. vocab - 1, kept between picks
    pick_vals: []f64 = &.{},
    pick_ids: []u64 = &.{},
    /// the drafter's work behind a window (null: none)
    behind: ?Behind = null,
    pub fn init(a: Allocator, env: Env, opt: Options) !*Target {
        const c = env.c;
        if (opt.slots == 0 or opt.limit == 0 or opt.limit > env.tables.positions) return error.BadOptions;
        _ = try prompt.check(c);
        const t = try a.create(Target);
        errdefer a.destroy(t);
        t.* = .{ .a = a, .env = env, .opt = opt, .b = undefined, .pr = undefined, .slots = &.{}, .cache = .{ .a = a, .cap = opt.cache_bytes } };
        t.b = try forward.Bufs.init(env.device, c);
        errdefer t.b.deinit();
        t.pr = try prompt.Prompt.init(env.device, c, opt.prompt, opt.limit);
        errdefer t.pr.deinit();
        t.pick_vals = try a.alloc(f64, picks.rows * @as(usize, c.vocab));
        errdefer a.free(t.pick_vals);
        t.pick_ids = try a.alloc(u64, c.vocab);
        errdefer a.free(t.pick_ids);
        for (t.pick_ids, 0..) |*id, i| id.* = i;
        const slots = try a.alloc(Slot, opt.slots);
        errdefer a.free(slots);
        var made: usize = 0;
        errdefer for (slots[0..made]) |*s| freeSlot(a, s);
        const cap = try prompt.tailRows(c, opt.prompt);
        const ptb = @max(4, @as(usize, ringRows(c)) * tapWidth(c) * 2);
        for (slots) |*s| {
            var st = try kv.Slot.init(a, env.device, c, opt.limit);
            errdefer st.deinit();
            var tail = try prompt.Tail.init(env.device, c, cap);
            errdefer tail.deinit();
            const ptaps = try env.device.buffer(ptb, mtl.ResourceOptions.shared);
            errdefer ptaps.deinit();
            const ring = try a.alloc(u16, @as(usize, ringRows(c)) * tapWidth(c));
            @memset(ring, 0);
            s.* = .{ .kv = st, .tail = tail, .ring = ring, .ptaps = ptaps };
            made += 1;
        }
        t.slots = slots;
        if (env.vision) |v| t.tower = vt.Tower.init(a, env.device, env.queue, v.kit, v.w);
        t.joinResident() catch {};
        return t;
    }
    /// Every buffer the target binds besides the weights joins their residency owner, so none waits to be mapped.
    fn joinResident(t: *Target) !void {
        const r = t.opt.resident orelse return;
        var bufs: std.ArrayList(mtl.Buffer) = .empty;
        defer bufs.deinit(t.a);
        try t.ownBuffers(&bufs);
        var bytes: u64 = 0;
        for (bufs.items) |b| bytes += b.length();
        // the owner refuses (and logs) buffers that would take it past its load limit
        t.joined = r.tryAddNow(bufs.items);
        if (t.joined) std.log.info("dsv41 target: {d:.1} GiB of buffers joined the weights' residency set", .{mtl.ResidencySet.gibs(bytes)});
    }
    fn ownBuffers(t: *const Target, out: *std.ArrayList(mtl.Buffer)) !void {
        try out.appendSlice(t.a, &.{ t.b.buf, t.pr.s.buf, t.pr.b.buf, t.env.s.buf, t.env.tables.window, t.env.tables.compressed });
        for (t.slots) |*s| {
            for (s.kv.regions) |r| {
                try out.append(t.a, r.window);
                inline for (.{ "pool", "keys", "carry", "inputs" }) |f| if (@field(r, f)) |b| try out.append(t.a, b);
            }
            if (s.kv.taps) |b| try out.append(t.a, b);
            try out.appendSlice(t.a, &.{ s.tail.x, s.tail.pre, s.ptaps });
        }
    }
    fn freeSlot(a: Allocator, s: *Slot) void {
        s.kv.deinit();
        s.tail.deinit();
        s.ptaps.deinit();
        a.free(s.ring);
        s.hist.deinit(a);
        s.hit.deinit(a);
        a.free(s.images);
        if (s.image_rows) |*r| r.deinit();
    }
    pub fn deinit(t: *Target) void {
        t.tree.deinit(t);
        if (t.joined) {
            var bufs: std.ArrayList(mtl.Buffer) = .empty;
            defer bufs.deinit(t.a);
            t.ownBuffers(&bufs) catch {};
            for (bufs.items) |b| t.opt.resident.?.remove(b);
        }
        if (t.tower) |*w| w.deinit();
        for (t.slots) |*s| freeSlot(t.a, s);
        t.a.free(t.slots);
        t.cache.deinit();
        t.pr.deinit();
        t.b.deinit();
        t.a.free(t.pick_vals);
        t.a.free(t.pick_ids);
        t.a.destroy(t);
    }

    /// The taps ring's rows: the drafter's context window.
    fn ringRows(c: *const Config) u32 {
        return c.window;
    }

    /// Values in a taps row: every target layer's hidden.
    fn tapWidth(c: *const Config) usize {
        return c.dspark_targets.len * c.hidden;
    }

    pub fn target(t: *Target) iface.Target {
        return .{ .ptr = t, .vtable = &.{ .prefill = prefill, .window = window, .keep = keep, .taps = taps, .release = release, .resumed = resumedFn, .history = history, .begin = steps.begin, .piece = steps.piece, .finish = steps.finish, .images = imagesFn } };
    }

    pub fn self(ptr: *anyopaque) *Target {
        return @ptrCast(@alignCast(ptr));
    }

    /// The drafter's queued work done before the target writes what it reads.
    fn drainBehind(t: *Target) !void {
        if (t.behind) |b| try b.drain(b.ctx);
    }

    /// The device buffer and offset of an address a `taps` named (the pass's ingest reads it there).
    pub fn bufferOf(t: *const Target, address: u64) ?struct { buf: mtl.Buffer, off: usize } {
        for (t.slots) |s| {
            const bufs = [_]?mtl.Buffer{ s.ptaps, s.kv.taps };
            for (bufs) |ob| if (ob) |b| {
                const lo = @intFromPtr(b.contents());
                if (address >= lo and address < lo + b.length()) return .{ .buf = b, .off = @intCast(address - lo) };
            };
        }
        return null;
    }

    // prompts ------------------------------------------------------------------------------------------------------

    /// The images of the slot's next prompt, their pixels borrowed until `prefill` returns.
    pub fn setImages(t: *Target, slot: u32, images: []const vrows.Image) !void {
        if (slot >= t.slots.len) return error.NoSlot;
        const s = &t.slots[slot];
        if (images.len > 0 and t.tower == null) return error.NoVision;
        const copy = try t.a.dupe(vrows.Image, images);
        t.a.free(s.images);
        s.images = copy;
    }

    fn imagesFn(ptr: *anyopaque, slot: u32, images: []const vrows.Image) anyerror!void {
        return self(ptr).setImages(slot, images);
    }

    fn resumedFn(ptr: *anyopaque, slot: u32) u64 {
        const t = self(ptr);
        return if (slot < t.slots.len) t.slots[slot].resumed else 0;
    }

    /// `ids` with each image position replaced by its virtual id (the prompt cache's and the Engram hash's view of it).
    fn withVids(t: *Target, ids: []const u32, images: []const vrows.Image) ![]u32 {
        const out = try t.a.dupe(u32, ids);
        errdefer t.a.free(out);
        for (images) |im| {
            if (@as(u64, im.at) + im.tokens > ids.len) return error.BadImageSpan;
            for (0..im.tokens) |p| out[im.at + p] = prep.vid(im.hash, @intCast(p));
        }
        return out;
    }

    fn prefill(ptr: *anyopaque, slot: u32, ids: []const u32, smp: ?lanes.Sampling, draw: u64) anyerror!u32 {
        const t = self(ptr);
        const st = try t.startPrompt(slot, ids);
        const s = st.s;
        const prompt_ids = s.hist.items; // image positions carry their virtual ids
        if (st.hit) |e| if (e.ids.len == prompt_ids.len) return t.endPrompt(s, e.logits.?, smp, draw);
        // states later prompts can resume from that share less than all of this one (kept as the prompt passes them)
        for (t.keepPoints(prompt_ids, s.resumed)) |end| if (end > 0) try t.keepAt(s, end);
        t.pr.run(t.promptEnv(), &s.kv, &s.tail, prompt_ids) catch |e| return t.fail(e);
        return t.endPrompt(s, null, smp, draw);
    }

    /// A prompt's start on a free slot: a kept state that starts it (restored), else the slot cleared; ids recorded.
    pub fn startPrompt(t: *Target, slot: u32, ids_in: []const u32) !struct { s: *Slot, hit: ?*snapshots.Entry } {
        if (slot >= t.slots.len) return error.NoSlot;
        try t.drainBehind();
        const s = &t.slots[slot];
        const c = t.env.c;
        if (s.live or s.filling) return error.SlotLive;
        if (ids_in.len == 0) return error.EmptyPrompt;
        if (ids_in.len >= t.opt.limit) return error.PastLimit;
        const images = s.images;
        s.images = &.{};
        defer t.a.free(images);
        var vids: ?[]u32 = null;
        defer if (vids) |v| t.a.free(v);
        if (images.len > 0) {
            if (t.tower == null) return error.NoVision;
            vids = try t.withVids(ids_in, images);
        }
        const ids: []const u32 = vids orelse ids_in;
        for (ids) |id| if (id >= c.vocab and !prep.isVid(id)) return error.BadToken;
        t.tree.forget(slot);
        s.whole = false;
        s.open = false;
        s.rows = 0;
        s.hit_on = false;
        t.pr.forget(&s.tail);
        t.stats.prompts += 1;
        // a kept state that starts this prompt (not one that ends inside an image's span), else a cleared slot
        var hit = if (t.cache.on()) t.cache.best(ids, t.replayMode()) else null;
        if (hit) |e| for (images) |im| if (e.ids.len > im.at and e.ids.len < im.at + im.tokens) {
            hit = null;
        };
        const at: u32 = if (hit) |e| @intCast(e.ids.len) else 0;
        if (hit) |e| {
            t.cache.restore(c, e, &s.kv, s.high, s.ring);
            t.stats.hits += 1;
            t.stats.cached_rows += at;
        } else snapshots.clear(c, &s.kv, s.high);
        if (images.len > 0) {
            s.image_rows = try vrows.Rows.build(t.a, t.env.device, &t.tower.?, images, @intCast(ids.len), t.opt.prompt.chunk, at);
            t.pr.images = if (s.image_rows) |*r| r else null;
        }
        s.resumed = at;
        s.tail.end = at;
        s.tail.done = at;
        s.hist.clearRetainingCapacity();
        try s.hist.appendSlice(t.a, ids);
        s.high = @max(at, ids.len);
        return .{ .s = s, .hit = hit };
    }

    /// The prompt's rows to `end`, short of its end, with the decoder's cone there: its taps and that state kept.
    pub fn keepAt(t: *Target, s: *Slot, end: u32) !void {
        t.pr.run(t.promptEnv(), &s.kv, &s.tail, s.hist.items[0..end]) catch |e| return t.fail(e);
        t.ringTaps(s, end);
        t.cache.save(t.env.c, &s.kv, s.hist.items[0..end], s.ring, null, t.replayMode()) catch {};
    }

    /// The taps rows the last prompt run wrote, its rows before `end`, into the ring.
    fn ringTaps(t: *const Target, s: *Slot, end: u32) void {
        const c = t.env.c;
        const tr = t.pr.tapRows(c);
        for (0..t.pr.taps) |i| t.ringPut(s, end - t.pr.taps + i, tr[i * tapWidth(c) ..][0..tapWidth(c)]);
    }

    /// The prompt's rows are in (`hit`: a kept state's logits, else the prompt head's): taps, choice, the slot live.
    pub fn endPrompt(t: *Target, s: *Slot, hit: ?[]const f32, smp: ?lanes.Sampling, draw: u64) !u32 {
        const c = t.env.c;
        t.pr.images = null;
        if (s.image_rows) |*r| r.deinit();
        s.image_rows = null;
        const n: u32 = @intCast(s.hist.items.len);
        t.stats.prompt_rows += n - s.resumed;
        const V: usize = c.vocab;
        var logits: []const f32 = undefined;
        if (hit) |l| logits = l else {
            const all = t.pr.logits(c);
            logits = all[(t.pr.head - 1) * V ..][0..V];
            t.ringTaps(s, n);
        }
        t.promptTaps(s, n);
        const choice = try pick.choose(t, logits, draw, smp);
        s.base = n;
        s.live = true;
        s.whole = true;
        // a thinking prompt is kept short of its <think> instead (keepPoints)
        if (t.cache.on() and s.resumed < n and t.keepPoints(s.hist.items, s.resumed)[1] == 0) t.cache.save(c, &s.kv, s.hist.items, s.ring, logits, t.replayMode()) catch {};
        return choice;
    }

    fn replayMode(t: *const Target) bool {
        return t.opt.prompt.replay and !t.opt.prompt.full;
    }

    /// Extra kept states (0: none): before the first user turn, and before a thinking prompt's closing <think>.
    pub fn keepPoints(t: *const Target, ids: []const u32, at: u32) [2]u32 {
        const c = t.env.c;
        const n = ids.len;
        var out: [2]u32 = .{ 0, 0 };
        if (!t.cache.on()) return out;
        // a fresh conversation shares the system text and tools, which end where the first user turn starts
        if (std.mem.indexOfScalar(u32, ids, c.user)) |u| {
            if (u >= min_shared and u > at) out[0] = @intCast(u);
        }
        // the next turn renders this turn's <think> as </think>, so it shares the prompt but that tag
        if (!t.replayMode() and n >= 2 and ids[n - 1] == c.think_open and ids[n - 2] == c.assistant and n - 1 > @max(at, out[0])) out[1] = @intCast(n - 1);
        return out;
    }

    pub fn promptEnv(t: *const Target) prompt.Env {
        const e = t.env;
        return .{ .c = e.c, .p = e.p, .tables = e.tables, .model = e.model, .lookup = e.lookup, .queue = e.queue };
    }

    fn ringPut(t: *const Target, s: *Slot, pos: u64, row: []const u16) void {
        const w = tapWidth(t.env.c);
        if (w == 0) return;
        const at: usize = @intCast(pos % ringRows(t.env.c));
        @memcpy(s.ring[at * w ..][0..w], row);
    }

    /// The taps of the prompt's last rows (the ring's, up to `ring rows`) in order, for `taps` after a prefill.
    fn promptTaps(t: *const Target, s: *Slot, n: u32) void {
        const c = t.env.c;
        const w = tapWidth(c);
        s.after_prompt = true;
        s.ptaps_rows = if (w == 0) 0 else @min(n, ringRows(c));
        const out = s.ptaps.slice(u16, @as(usize, s.ptaps_rows) * w);
        for (0..s.ptaps_rows) |i| {
            const pos = n - s.ptaps_rows + i;
            @memcpy(out[i * w ..][0..w], s.ring[(pos % ringRows(c)) * w ..][0..w]);
        }
    }

    pub fn fail(t: *Target, e: anyerror) anyerror {
        if (e == error.CommandBufferFailed) t.failed = true;
        return e;
    }

    // windows ------------------------------------------------------------------------------------------------------

    /// A round's windows: slots' rows share a launch (up to R rows, a slot once), each launch one forward.
    fn window(ptr: *anyopaque, segments: []const iface.Segment, choices: [][]u32) anyerror!void {
        const t = self(ptr);
        try t.drainBehind();
        // the tree windows (parents not one chain) run first, together: their branches on shadow slots
        try t.tree.window(t, segments, choices);
        var from: usize = 0;
        while (from < segments.len) {
            if (tree_.branched(segments[from])) {
                from += 1;
                continue;
            }
            var rows: usize = 0;
            var to = from;
            while (to < segments.len) : (to += 1) {
                const g = segments[to];
                if (g.slot >= t.slots.len) return error.NoSlot;
                if (tree_.branched(g)) break;
                if (to > from and (rows + g.tokens.len > R or dup(segments[from..to], g.slot))) break;
                rows += g.tokens.len;
            }
            try t.windowGroup(segments[from..to], choices[from..to]);
            from = to;
        }
    }

    fn dup(done: []const iface.Segment, slot: u32) bool {
        for (done) |o| if (o.slot == slot) return true;
        return false;
    }

    /// One launch over the rows of these segments (distinct slots, at most R rows in all).
    fn windowGroup(t: *Target, group: []const iface.Segment, choices: [][]u32) !void {
        const c = t.env.c;
        const V: usize = c.vocab;
        var total: u32 = 0;
        for (group, choices) |g, out| {
            const s = &t.slots[g.slot];
            const n = g.tokens.len;
            if (!s.live) return error.SlotNotLive;
            if (g.start != s.hist.items.len) return error.PositionMismatch;
            if (n == 0 or total + n > R or g.draws.len != n or out.len < n) return error.BadRows;
            if (g.start + n > t.opt.limit) return error.PastLimit;
            // the decode forward runs consecutive positions: a tree's rows must be one chain
            if (g.parents) |ps| for (ps, 0..) |q, r| if (q != @as(i32, @intCast(r)) - 1) return error.TreeWindow;
            for (g.tokens) |id| if (id >= c.vocab) return error.BadToken;
            total += @intCast(n);
        }
        for (group) |g| {
            const s = &t.slots[g.slot];
            t.tree.close(g.slot);
            if (s.open) t.commitTaps(s, s.rows); // a window never kept keeps all its rows
            s.whole = false;
            try s.hist.appendSlice(t.a, g.tokens);
            s.high = @max(s.high, s.hist.items.len);
        }
        t.launch(group) catch |e| return t.fail(e);
        const lg = t.b.logitsOf(c, total);
        var r0: usize = 0;
        for (group, choices) |g, out| {
            const s = &t.slots[g.slot];
            const n = g.tokens.len;
            const keyed = if (g.sampling) |smp| smp.temperature > 0 and n > 1 else false;
            if (keyed) try picks.keyed(lg[r0 * V ..][0 .. n * V], g.draws, g.sampling.?, t.pick_vals, t.pick_ids, out[0..n]) else {
                for (0..n) |r| out[r] = try pick.choose(t, lg[(r0 + r) * V ..][0..V], g.draws[r], g.sampling);
            }
            s.base = g.start;
            s.rows = @intCast(n);
            s.open = true;
            s.after_prompt = false;
            s.tail.end = @intCast(s.hist.items.len);
            s.tail.done = s.tail.end;
            s.whole = true;
            t.stats.windows += 1;
            t.stats.rows += n;
            r0 += n;
        }
        t.stats.launches += 1;
        // each slot's next window starts with the pick after the drafts its picks accept: its Engram reads start now
        if (t.env.lookup) |l| for (group, choices) |g, out| t.prefetchNext(l, g, out);
    }

    /// Starts reading the Engram rows of the slot's next pending row (a guess: a cut commit only wastes the reads).
    fn prefetchNext(t: *const Target, l: forward.Lookup, g: iface.Segment, out: []const u32) void {
        if (t.env.c.engram_layers.len == 0) return;
        var acc: usize = 0;
        while (acc + 1 < g.tokens.len and g.tokens[acc + 1] == out[acc]) acc += 1;
        const hist = t.slots[g.slot].hist.items;
        const at: usize = @intCast(g.start + acc + 1);
        const look = t.env.c.engram_max_ngram - 1;
        forward.prefetchRows(&.{.{ .ids = out[acc..][0..1], .lookback = hist[at -| look..at] }}, l);
    }

    /// One launch of the decode forward over the segments' rows, each slot's ids at its positions from `start`.
    fn launch(t: *Target, group: []const iface.Segment) !void {
        const look = t.env.c.engram_max_ngram - 1;
        var runs: [R]Run = undefined;
        for (group, runs[0..group.len]) |g, *r| {
            const s = &t.slots[g.slot];
            r.* = .{ .kv = &s.kv, .ids = g.tokens, .start = @intCast(g.start), .lookback = s.hist.items[@as(usize, @intCast(g.start)) -| look..@intCast(g.start)] };
        }
        return launch_.run(t, runs[0..group.len], group);
    }

    pub const Run = launch_.Run;

    pub fn launchRuns(t: *Target, runs: []const Run) !void {
        return launch_.run(t, runs, &.{});
    }

    /// Keep the window's first rows (a chain from row 0): later rows' ids go, the compressor's carry rewinds.
    fn keep(ptr: *anyopaque, slot: u32, path: []const u32) anyerror!void {
        const t = self(ptr);
        if (slot >= t.slots.len) return error.NoSlot;
        const s = &t.slots[slot];
        if (!s.open) return error.NoWindow;
        // a tree window's path: its branch's chain, taken from its shadow slot first when not the main one
        if (t.tree.pending(slot)) try t.tree.keep(t, slot, path) else for (path, 0..) |r, i| if (r != i) return error.NotAPath;
        if (path.len == 0 or path.len > s.rows) return error.NotAPath;
        s.open = false;
        const m: u32 = @intCast(path.len);
        t.commitTaps(s, m);
        if (m == s.rows) return;
        const len = s.base + m;
        s.hist.shrinkRetainingCapacity(@intCast(len));
        for (s.kv.regions) |*r| kv.rewindCarry(r, t.env.c, s.base, m);
        s.tail.end = @intCast(len);
        s.tail.done = s.tail.end;
        t.stats.rewinds += 1;
    }

    /// The ring takes the first `m` rows of the slot's last window (its kept rows); rows not kept never land in it.
    pub fn commitTaps(t: *const Target, s: *Slot, m: u32) void {
        const tb = s.kv.taps orelse return;
        const w = tapWidth(t.env.c);
        for (0..m) |r| t.ringPut(s, s.base + r, tb.slice(u16, @as(usize, s.rows) * w)[r * w ..][0..w]);
    }

    /// Committed rows' taps [start, start + rows) from the ring, in order, in the prompt-taps buffer.
    fn history(ptr: *anyopaque, slot: u32, start: u64, rows: u32) ?iface.Taps {
        const t = self(ptr);
        if (slot >= t.slots.len) return null;
        // the rows land in the prompt-taps buffer a queued ingest may still read
        t.drainBehind() catch return null;
        const s = &t.slots[slot];
        const c = t.env.c;
        const w = tapWidth(c);
        const end: u64 = if (s.open) s.base else s.hist.items.len;
        if (w == 0 or s.after_prompt or rows == 0 or start + rows > end or start + ringRows(c) < end) return null;
        const out = s.ptaps.slice(u16, @as(usize, rows) * w);
        for (0..rows) |i| @memcpy(out[i * w ..][0..w], s.ring[((start + i) % ringRows(c)) * w ..][0..w]);
        t.stats.history_rows += rows;
        return .{ .address = @intFromPtr(s.ptaps.contents()), .rows = rows, .stride = @intCast(w) };
    }

    fn taps(ptr: *anyopaque, slot: u32) iface.Taps {
        const t = self(ptr);
        const s = &t.slots[slot];
        const w: u32 = @intCast(tapWidth(t.env.c));
        if (s.after_prompt) return .{ .address = @intFromPtr(s.ptaps.contents()), .rows = s.ptaps_rows, .stride = w };
        const tb = s.kv.taps orelse return .{ .address = 0, .rows = 0, .stride = w };
        return .{ .address = @intFromPtr(tb.contents()), .rows = s.rows, .stride = w, .map = t.tree.map(slot) };
    }

    /// The request ended: its state is kept (the prompt cache) when every launch of it completed.
    fn release(ptr: *anyopaque, slot: u32) void {
        const t = self(ptr);
        if (slot >= t.slots.len) return;
        const s = &t.slots[slot];
        if (s.filling) steps.abandon(t, s);
        if (!s.live) return;
        t.tree.forget(slot);
        s.live = false;
        if (s.open) t.commitTaps(s, s.rows);
        s.open = false;
        if (s.whole and t.cache.on()) t.cache.save(t.env.c, &s.kv, s.hist.items, s.ring, null, t.replayMode()) catch {};
    }

    /// The serial reference without lanes: the prompt, then one row a step, each drawn at its own position.
    pub fn serial(t: *Target, slot: u32, ids: []const u32, n: usize, smp: ?lanes.Sampling) ![]u32 {
        var out: std.ArrayList(u32) = .empty;
        errdefer out.deinit(t.a);
        var tok = try prefill(t, slot, ids, smp, ids.len);
        while (out.items.len < n) {
            try out.append(t.a, tok);
            const at: u64 = ids.len + out.items.len - 1;
            var ch: [1]u32 = undefined;
            var cs = [_][]u32{&ch};
            try window(t, &.{.{ .slot = slot, .start = at, .tokens = &.{tok}, .parents = null, .draws = &.{at + 1}, .sampling = smp }}, &cs);
            tok = ch[0];
        }
        release(t, slot);
        return out.toOwnedSlice(t.a);
    }
};
