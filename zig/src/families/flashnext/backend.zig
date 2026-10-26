//! Isolated Flash Next sessions sharing target rounds and MTP command buffers.
const std = @import("std");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const Engine = @import("engine.zig").Engine;
const Session = @import("session.zig").Session;
const meta = @import("batch_meta.zig");
const forward = @import("batch_forward.zig");
const snap = @import("snapshot.zig");
const be = lanes.backend;
const Stream = lanes.Stream;
const Buf = fz.Buf;
const mtl = @import("metal");
const segments = @import("../../core/segments.zig");
const slots = @import("slots.zig");
const sample = @import("batch_sample.zig");
const think = @import("think.zig");
const batch_costs = @import("batch_costs.zig");
const grammar = @import("grammar.zig");
const fill = @import("prefill.zig");
const prompt_gpu = @import("prefill_metal.zig");
const RoundTimes = @import("round_times.zig").RoundTimes;
const batch_settle = @import("batch_settle.zig");
const batch_draft = @import("batch_draft.zig");

/// A stream's images through the host's tower: its prompt pass's rows and frame (in `a`), then their release.
pub const Images = struct { ctx: *anyopaque, build: *const fn (ctx: *anyopaque, a: std.mem.Allocator, images: []const lanes.stream.Image, prompt_len: usize, from: usize) anyerror!*fz.vis_prompt.VisPrompt, release: *const fn (ctx: *anyopaque, p: *fz.vis_prompt.VisPrompt) void };

pub const Backend = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    primary: *fz.Model,
    sessions: []Session,
    by: std.AutoHashMapUnmanaged(*Stream, usize) = .empty,
    filling: std.AutoHashMapUnmanaged(*Stream, fill.State) = .empty,
    hidden: Buf,
    window_count: u64 = 0,
    shared_count: u64 = 0,
    chain_drafts: bool = false,
    draft_calls: u64 = 0,
    draft_waits: u64 = 0,
    draft_seconds: f64 = 0,
    lone_handoffs: u64 = 0, // lone streams the engine's rounds handed to the shared ones
    sized: bool = false, // FLASHNEXT_SLOT_CONTEXTS set: sessions of several sizes; unset, every stream takes the first free one and none grows
    grow: usize = std.math.maxInt(usize), // reply tokens a session must hold at admission past the prompt (the rest: move)
    paused: std.ArrayList(*Stream) = .empty, // streams that outgrew their session and wait for a larger one to free
    moves: u64 = 0,
    serial_drafts: bool = false, // FZ_BATCH_MTP_SERIAL=1: each stream's drafts in their own command buffers
    overlap: bool = true, // FZ_VERIFY_OVERLAP=0: verify on a serial encoder; else a concurrent one, the streams' recurrences without barriers between them
    costs: ?batch_costs.Costs = null, // timeCosts: the lane allocator's prices
    images: ?Images = null, // --vision: the tower's rows for a stream's images; null: an image stream fails
    grammar: grammar.Masks, // structured output: the rounds' mask rows and first rows
    times: RoundTimes = .{}, // FZ_ROUND_TIMES=1: a shared round's phases
    fence: ?mtl.counters.Fence = null, // orders a round's state copies before its forward in one command buffer
    packed_drafts: batch_draft.State = .{}, // every stream's draft steps packed into one pass a step (FZ_BATCH_MTP_ROWS=0: off)

    /// Sessions of the given capacities (the first, the engine's own, holds its whole window whatever is given).
    pub fn init(gpa: std.mem.Allocator, e: *Engine, capacities: []const usize) !Backend {
        if (capacities.len < 2 or capacities.len > fz.MAXR or e.r.tp != null) return error.BadBatchSlots;
        const sessions = try gpa.alloc(Session, capacities.len);
        errdefer gpa.free(sessions);
        var made: usize = 0;
        errdefer for (sessions[0..made]) |*s| s.deinit(gpa);
        for (sessions, capacities, 0..) |*s, capacity, i| {
            s.* = try Session.init(gpa, e, capacity, i == 0);
            made += 1;
        }
        var masks = try grammar.Masks.init(e.r.device);
        errdefer masks.deinit(gpa);
        const hidden = try e.r.device.buffer(fz.MAXR * fz.WIDE * 2, fz.opts);
        const env = struct {
            fn on(name: [*:0]const u8) bool {
                return if (std.c.getenv(name)) |v| v[0] == '1' else false;
            }
        };
        return .{ .gpa = gpa, .e = e, .primary = e.m, .sessions = sessions, .hidden = .{ .b = hidden }, .grammar = masks, .chain_drafts = env.on("FZ_BATCH_MTP_CHAIN"), .serial_drafts = env.on("FZ_BATCH_MTP_SERIAL"), .overlap = if (std.c.getenv("FZ_VERIFY_OVERLAP")) |v| v[0] != '0' else true, .times = RoundTimes.init(), .fence = if (std.c.getenv("FZ_SETTLE_SYNC") == null) mtl.counters.Fence.init(e.r.device) catch null else null };
    }

    /// Times shared-round costs on the last free session, or leaves them unpriced when it cannot fit.
    pub fn timeCosts(b: *Backend) !void {
        return batch_costs.timeCosts(b);
    }

    pub fn deinit(b: *Backend) void {
        if (b.fence) |f| f.deinit();
        while (b.filling.count() > 0) {
            var it = b.filling.keyIterator();
            b.abortPrompt(it.next().?.*);
        }
        b.filling.deinit(b.gpa);
        b.e.m = b.primary;
        b.paused.deinit(b.gpa);
        b.by.deinit(b.gpa);
        for (b.sessions) |*s| s.deinit(b.gpa);
        b.gpa.free(b.sessions);
        b.hidden.b.deinit();
        b.grammar.deinit(b.gpa);
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = fill.Callbacks(Backend).whole, .prefill_step = fill.Callbacks(Backend).one, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .first_masked = grammar.firstMasked, .held = grammar.held } };
    }

    pub const facts = batch_costs.facts;

    pub fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }
    fn session(b: *Backend, s: *Stream) !*Session {
        return &b.sessions[b.by.get(s) orelse return error.UnknownStream];
    }
    pub fn settle(b: *Backend) !void {
        return batch_settle.now(b);
    }

    const Binding = @import("session_binding.zig").Binding;

    pub fn bind(b: *Backend, session_: *Session) !Binding {
        return @import("session_binding.zig").bind(b.e, session_);
    }

    /// Chooses a slot; LaneMemory means wait, ContextFull means no session can hold the reply.
    pub fn slotFor(b: *Backend, prompt: usize, max_new: usize) !usize {
        var caps: [fz.MAXR]usize = undefined;
        var used: [fz.MAXR]bool = undefined;
        var most: usize = 0;
        for (b.sessions, 0..) |s, j| {
            caps[j] = s.m.cap;
            used[j] = s.used;
            most = @max(most, s.m.cap);
        }
        const n = b.sessions.len;
        const whole = prompt + max_new + slots.WINDOW;
        const i = slots.pick(caps[0..n], used[0..n], b.sized, prompt, max_new, b.grow) orelse
            return if (!b.sized) error.NoFreeSlot else if (whole > most) error.ContextFull else error.LaneMemory;
        if (whole > (if (b.sized) most else caps[i])) return error.ContextFull; // sized, a smaller one grows into the largest
        return i;
    }

    /// Moves or pauses a sized stream when its remaining reply needs a larger session.
    fn makeRoom(b: *Backend, stream: *Stream, rows: usize) !void {
        if (!b.sized) return;
        const i = b.by.get(stream) orelse return error.UnknownStream;
        const s = &b.sessions[i];
        var most: usize = 0;
        var larger = false;
        for (b.sessions) |o| {
            most = @max(most, o.m.cap);
            larger = larger or (!o.used and o.m.cap > s.m.cap);
        }
        switch (slots.room(s.m.cap, s.m.pos, rows, stream.prompt().len + stream.max_new + slots.WINDOW, most, larger)) {
            .ok => {},
            .move => try b.move(stream, i),
            .pause => {
                if (!stream.paused) {
                    try b.paused.append(b.gpa, stream);
                    std.log.info("flash next: a stream at {d} tokens waits for a session larger than {d} tokens to free", .{ s.m.pos, s.m.cap });
                }
                stream.paused = true;
            },
            .full => return error.ContextFull,
        }
    }

    /// Moves to the smallest free larger session holding the next grow tokens, or the largest available.
    fn move(b: *Backend, stream: *Stream, i: usize) !void {
        const from = &b.sessions[i];
        const left: usize = @intCast(@max(stream.budgetLeft(), 0));
        const need = from.m.pos + @min(left, b.grow) + 2 * slots.WINDOW;
        var best: ?usize = null; // the smallest free larger session that holds `need`
        var most: ?usize = null; // else the largest free larger one
        for (b.sessions, 0..) |o, j| if (!o.used and o.m.cap > from.m.cap) {
            if (o.m.cap >= need and (best == null or o.m.cap < b.sessions[best.?].m.cap)) best = j;
            if (most == null or o.m.cap > b.sessions[most.?].m.cap) most = j;
        };
        const j = best orelse most orelse return error.LaneMemory;
        try b.settle();
        try from.moveTo(&b.sessions[j], b.e);
        try b.by.put(b.gpa, stream, j);
        from.used = false;
        from.round = .{};
        from.held_n = 0;
        b.moves += 1;
        std.log.info("flash next: a stream at {d} tokens moved from a {d}-token session to a {d}-token one ({d} moves)", .{ from.m.pos, from.m.cap, b.sessions[j].m.cap, b.moves });
    }

    /// Resumes paused streams; errors other than LaneMemory terminate the affected stream.
    fn resumePaused(b: *Backend) void {
        slots.resumePaused(&b.paused, b);
    }

    /// slots.resumePaused's move: the stream into a free larger session, LaneMemory while none is.
    pub fn moveOut(b: *Backend, stream: *Stream) !void {
        try b.move(stream, b.by.get(stream) orelse return error.UnknownStream);
    }

    /// slots.resumePaused's failure: the stream ends with an error the host reports, and its session frees now.
    pub fn failed(b: *Backend, stream: *Stream, err: anyerror) void {
        std.log.err("flash next: a paused stream at {d} tokens could not move to a larger session ({s}); it ends with an error", .{ stream.context.items.len, @errorName(err) });
        slots.endPaused(stream);
        const kv = b.by.fetchRemove(stream) orelse return;
        b.sessions[kv.value].used = false;
        b.sessions[kv.value].round = .{};
        b.sessions[kv.value].held_n = 0;
    }

    /// LaneHost's admission check: a free session holds the request now.
    pub fn fits(ptr: *anyopaque, prompt: u32, max_new: u32) bool {
        _ = self(ptr).slotFor(prompt, max_new) catch return false;
        return true;
    }

    pub fn promptPool(_: *Backend) @TypeOf(mtl.objc.Pool.push()) {
        return mtl.objc.Pool.push();
    }
    pub fn beginPrompt(b: *Backend, stream: *Stream, slot: ?usize) !fill.State {
        return prompt_gpu.begin(b, stream, slot);
    }
    pub fn settlePrompt(b: *Backend, _: *Stream) !void {
        try b.settle();
    }
    pub fn finishChunk(b: *Backend, _: *Stream) !void {
        try b.settle();
    }
    pub fn promptRows(b: *Backend, at: u32, end: u32) u32 {
        return prompt_gpu.rows(b, at, end);
    }
    pub fn chunkPrompt(b: *Backend, stream: *Stream, slot: usize, at: u32, rows: u32) !void {
        try prompt_gpu.chunk(b, stream, slot, at, rows);
    }
    pub fn markPrompt(b: *Backend, stream: *Stream, at: u32) !void {
        try prompt_gpu.mark(b, stream, at);
    }
    pub fn finishPrompt(b: *Backend, stream: *Stream) !void {
        try prompt_gpu.finish(b, stream);
    }
    pub fn abortPrompt(b: *Backend, stream: *Stream) void {
        release(b, stream);
    }
    pub fn decoding(b: *const Backend) bool {
        var it = b.by.keyIterator();
        while (it.next()) |key| {
            const s = key.*;
            if (!b.filling.contains(s) and !s.paused and !s.finished) return true;
        }
        return false;
    }

    /// Runs a lone request in session 0 until it finishes or hands over to shared rounds.
    pub fn lone(b: *Backend, stream: *Stream, hooks: anytype) !bool {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const s = &b.sessions[0]; // the GPU-side rounds run on the engine's own buffers
        if (s.used) return error.NoFreeSlot;
        if (stream.prompt().len + stream.max_new + @import("engine.zig").MARGIN > s.m.cap) return error.ContextFull;
        var handed = false;
        defer if (!handed) release(b, stream);
        if (!try fill.lone(b, stream, hooks)) {
            handed = true;
            return true;
        }
        const prompt = try @import("lone_stream.zig").commitFirst(b.gpa, stream, s.first); // decode copies it to the rounds' history before any round commits
        hooks.committed(hooks.ctx);
        if (stream.finished) return false;
        const Hooks = @TypeOf(hooks);
        const Ctx = struct {
            gpa: std.mem.Allocator,
            stream: *Stream,
            hooks: Hooks,
            failed: ?anyerror = null,
            fn prefilled(_: *anyopaque) void {}
            fn tokens(p: *anyopaque, toks: []const u32) bool {
                const c: *@This() = @ptrCast(@alignCast(p));
                _ = c.stream.commit(c.gpa, toks, &.{}) catch |err| {
                    c.failed = err;
                    return true;
                };
                c.hooks.committed(c.hooks.ctx);
                return c.stream.finished;
            }
            fn cancelled(p: *anyopaque) bool { // a request waits, or one is cancelled: the lane core takes over
                const c: *@This() = @ptrCast(@alignCast(p));
                return c.hooks.yield(c.hooks.ctx);
            }
        };
        var c: Ctx = .{ .gpa = b.gpa, .stream = stream, .hooks = hooks };
        var ho: Engine.Handoff = .{};
        {
            const old = try b.bind(s);
            defer old.restore(b.e);
            outputs(b, s, 0);
            var watch = try think.Watch.init(.{}); // no thinking budget: lone_host_lone.fits keeps those in the lane core
            const res = try b.e.decode(prompt, s.first, s.last, stream.max_new, stream.eos, null, &watch, .{ .ctx = &c, .prefilled = Ctx.prefilled, .tokens = Ctx.tokens, .cancelled = Ctx.cancelled }, &ho);
            if (c.failed) |err| return err;
            stream.rounds += res.rounds;
            stream.drafted += res.drafted;
            stream.accepted += res.accepted;
            if (res.rounds > 0) stream.min_rows = if (stream.min_rows == 0) res.min_rows else @min(stream.min_rows, res.min_rows);
        }
        if (!ho.yielded) {
            if (!stream.finished) { // the rounds ended where the stream did not (eos or length by the engine's count)
                stream.finished = true;
                stream.reason = .stop;
            }
            return false;
        }
        const ctx = stream.context.items;
        const at = ho.cache; // the cache's rows: every committed token but the last, the next window's first
        if (at + 1 != ctx.len or at < 2) return error.HandoffOutOfStep;
        s.m.pos = at;
        s.m.state = 0;
        s.m.state_row = 0;
        s.m.ple.hist = .{ ctx[at - 2], ctx[at - 1] };
        s.m.mtp.pos = at;
        s.m.mtp.drafted = 0;
        s.m.mtp.pooled_n = @min(s.m.mtp.pooled_n, at / 4); // blocks the rounds pooled are pooled again here
        for (&s.m.layers) |*l| if (!l.linear) {
            l.pooled_n = @min(l.pooled_n, at / 4);
        };
        s.held_n = 0;
        s.round = .{};
        stream.cache_len = at;
        stream.pending = ctx[at];
        stream.next = null;
        handed = true;
        b.lone_handoffs += 1;
        std.log.info("flash next: a lone stream at {d} tokens handed to the shared rounds ({d} handoffs)", .{ at, b.lone_handoffs });
        return true;
    }

    fn first(ptr: *anyopaque, stream: *Stream, position: u64) anyerror!u64 {
        if (self(ptr).filling.contains(stream)) return error.PromptIncomplete;
        const s = try self(ptr).session(stream);
        if (position != s.m.pos) return error.PositionMismatch;
        return s.first;
    }
    fn queue(_: *anyopaque, _: *Stream, _: be.Feed, _: u64) anyerror!u64 {
        return error.Unsupported;
    }
    fn read(_: *anyopaque, handle: u64) anyerror!u32 {
        return @intCast(handle);
    }

    pub fn outputs(b: *Backend, s: *Session, row: usize) void {
        var li: usize = 0;
        for (&s.m.layers) |*l| if (l.linear) {
            l.cs[1] = .{ .b = b.e.o_cs, .off = (li * fz.MAXR + row) * fz.CS_ROW };
            l.so[1] = .{ .b = b.e.o_so, .off = (li * fz.MAXR + row) * fz.SO_ROW };
            li += 1;
        };
    }

    fn verify(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const b = self(ptr);
        b.times.begin(false);
        const fold = if (b.fence != null and !b.sized) try batch_settle.keep(b) else blk: {
            try b.settle();
            break :blk false;
        };
        b.times.settled();
        if (windows.len == 0 or windows.len > fz.MAXR or out.len != windows.len) return error.BadBatch;
        for (windows) |w| try b.makeRoom(w.stream, w.rows());
        var desc: [fz.MAXR]meta.Window = undefined;
        var segs: [fz.MAXR]forward.Seg = undefined;
        for (windows, out, 0..) |w, o, i| {
            const s = try b.session(w.stream);
            if (w.parents != null or w.early or w.held > s.held_n) return error.WindowOutOfStep;
            if (w.rows() > fz.MAXR or w.rows() > s.m.cap - s.m.pos or
                w.positions.len != w.rows() or o.sampled.len != w.rows() or o.drafts.len != w.rows() - 1)
                return error.WindowOutOfStep;
            for (w.positions, 0..) |p, j| if (p != s.m.pos + j + 1) return error.PositionMismatch;
            desc[i] = .{ .slot = @intCast(b.by.get(w.stream).?), .pos = @intCast(s.m.pos), .rows = @intCast(w.rows()) };
        }
        const plan = try meta.forBackend(b, desc[0..windows.len]);
        const t = &b.primary.t;
        for (windows, out, 0..) |w, o, i| {
            const s = try b.session(w.stream);
            const row = plan.starts[i];
            s.row0 = row;
            try s.round.begin(s.m.pos, w.rows());
            s.tokens[0] = w.pending;
            @memcpy(s.tokens[1..][0..w.held], s.held[0..w.held]);
            @memcpy(s.tokens[1 + w.held ..][0..w.tokens.len], w.tokens);
            s.m.windowMeta(s.round.rows);
            s.m.pleIds(s.tokens[0..s.round.rows]);
            @memcpy(t.ids8.b.slice(u32, fz.MAXR)[row..][0..s.round.rows], s.tokens[0..s.round.rows]);
            @memcpy(t.ple_ids.b.slice(u32, fz.MAXR * 16)[row * 16 ..][0 .. s.round.rows * 16], s.m.t.ple_ids.b.slice(u32, s.round.rows * 16));
            segs[i] = .{ .s = s, .row0 = row, .rows = s.round.rows };
            @memcpy(o.drafts, s.tokens[1..s.round.rows]);
        }
        t.rows.b.slice(i32, 1)[0] = @intCast(plan.total);
        t.mdims.b.slice(i32, 2)[0] = @intCast(plan.total);
        const pos8 = t.pos8.b.slice(u32, fz.MAXR);
        @memcpy(pos8[0..plan.total], plan.positions[0..plan.total]);
        for (windows, 0..) |w, i| { // an image stream's rows past its last image rotate at their slot plus its delta
            const d = (try b.session(w.stream)).m.delta;
            fz.vis_prompt.decodePositions(pos8[plan.starts[i]..][0..w.rows()], d);
        }
        const cb = b.e.r.queue.commandBuffer();
        // The copies read the kept round's rows: encode them before outputs() repoints cs[1] at this round's.
        if (fold) batch_settle.encode(b, cb, b.fence.?);
        for (segs[0..windows.len], 0..) |g, i| outputs(b, g.s, plan.starts[i]);
        b.e.r.begin(cb, if (b.overlap) .concurrent else .serial);
        errdefer if (b.e.r.fsr != null) b.e.r.abort();
        if (fold) b.fence.?.waitCompute(b.e.r.enc);
        const masked = b.grammar.fill(windows, plan.starts[0..windows.len]);
        const was = b.e.r.serial;
        if (b.overlap) b.e.r.serial = false; // every launch then ends in a barrier, as the drafts' encoders do
        const encoded = forward.encode(b.e, segs[0..windows.len], plan.total, t.ids8, if (masked) &b.grammar else null);
        b.e.r.serial = was;
        try encoded;
        @memcpy(b.e.draw_at.b.slice(u32, fz.MAXR)[0..plan.total], plan.keys[0..plan.total]);
        var widths: [fz.MAXR]usize = undefined;
        var ps: [fz.MAXR]?lanes.Sampling = undefined;
        for (windows, 0..) |w, i| {
            widths[i] = w.rows();
            ps[i] = w.stream.sampling;
        }
        sample.encodeAll(b.e.r.enc, b.e.r.draw_pipe, t.logits, b.e.draw_at, t.picks, plan.starts[0..windows.len], widths[0..windows.len], ps[0..windows.len]);
        b.times.commit();
        try b.primary.finish(cb);
        b.times.end(cb, false, plan.total, windows.len);
        Session.copy(b.primary.last, b.hidden, plan.total * fz.WIDE * 2);
        for (out, 0..) |o, i| @memcpy(o.sampled, t.picks.b.slice(u32, plan.total)[plan.starts[i]..][0..o.sampled.len]);
        b.window_count += 1;
        if (windows.len > 1) b.shared_count += 1;
        if (std.c.getenv("FZ_BATCH_LOG") != null)
            std.log.info("flashnext batch: streams {d}, rows {d}, shared rounds {d}", .{ windows.len, plan.total, b.shared_count });
    }

    fn keep(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const b = self(ptr);
        for (windows, paths) |w, path| {
            for (path, 0..) |r, j| if (r != j) return error.TreeWindowsUnsupported;
            try (try b.session(w.stream)).keep(path.len);
        }
    }

    fn draft(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const b = self(ptr);
        if (!b.serial_drafts) return b.draftTogether(requests);
        for (requests) |r| {
            const pool = mtl.objc.Pool.push();
            defer pool.pop();
            if (r.early or r.lanes != null or r.ranks or r.depth >= fz.MAXR) return error.DraftOutOfStep;
            const s = try b.session(r.stream);
            var one: [1]u32 = undefined;
            const follow = if (r.first) |f| blk: {
                one[0] = switch (f) {
                    .value => |v| v,
                    .handle => |h| @intCast(h),
                };
                break :blk one[0..];
            } else r.follow;
            const hidden = if (r.rows) |path| blk: {
                try s.round.draft(r.start, path, follow.len);
                break :blk b.hidden.at(s.row0 * fz.WIDE * 2);
            } else s.last;
            const old = try b.bind(s);
            defer old.restore(b.e);
            s.held_n = r.depth;
            if (follow.len == 0) return error.DraftOutOfStep;
            const start = mtl.clock.seconds();
            if (b.chain_drafts and r.depth > 0 and follow.len <= fz.MAXR) {
                try s.draftChain(follow, hidden, r.depth);
                b.draft_waits += 1;
            } else {
                const token = try s.m.mtpAbsorb(follow, hidden);
                if (r.depth > 0) s.held[0] = token;
                if (r.depth > 1) for (1..r.depth) |i| {
                    s.held[i] = try s.m.mtpChain(s.held[i - 1]);
                };
                b.draft_waits += (follow.len + fz.MAXR - 1) / fz.MAXR + (r.depth -| 1);
            }
            b.draft_calls += 1;
            b.draft_seconds += mtl.clock.seconds() - start;
        }
    }

    /// Every request's absorb and chained drafts in one command buffer, one wait (a long follow absorbs alone first).
    fn draftTogether(b: *Backend, requests: []const be.DraftRequest) !void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const start = mtl.clock.seconds();
        b.times.begin(true);
        var follows: [fz.MAXR][]const u32 = undefined;
        var hiddens: [fz.MAXR]Buf = undefined;
        var ones: [fz.MAXR][1]u32 = undefined;
        var ss: [fz.MAXR]*Session = undefined;
        var depths: [fz.MAXR]usize = undefined;
        var draws: [fz.MAXR]?lanes.Sampling = undefined;
        var lands: [fz.MAXR]u64 = undefined;
        if (requests.len > fz.MAXR) return error.DraftOutOfStep;
        for (requests, 0..) |r, k| {
            if (r.early or r.lanes != null or r.ranks or r.depth >= fz.MAXR) return error.DraftOutOfStep;
            const s = try b.session(r.stream);
            ss[k], depths[k], draws[k], lands[k] = .{ s, r.depth, r.stream.sampling, r.position };
            follows[k] = if (r.first) |f| blk: {
                ones[k][0] = switch (f) {
                    .value => |v| v,
                    .handle => |h| @intCast(h),
                };
                break :blk ones[k][0..];
            } else r.follow;
            if (follows[k].len == 0) return error.DraftOutOfStep;
            hiddens[k] = if (r.rows) |path| blk: {
                try s.round.draft(r.start, path, follows[k].len);
                break :blk b.hidden.at(s.row0 * fz.WIDE * 2);
            } else s.last;
            if (follows[k].len > fz.MAXR) { // all but the last MAXR rows absorbed first, on their own
                const cut = follows[k].len - fz.MAXR;
                const old = try b.bind(s);
                defer old.restore(b.e);
                _ = try s.m.mtpAbsorb(follows[k][0..cut], hiddens[k]);
                hiddens[k] = hiddens[k].at(cut * fz.WIDE * 2);
                follows[k] = follows[k][cut..];
                b.draft_waits += (cut + fz.MAXR - 1) / fz.MAXR;
            }
        }
        const cb = b.e.r.queue.commandBuffer();
        b.e.r.begin(cb, if (b.e.r.serial) .serial else .concurrent);
        errdefer if (b.e.r.fsr != null) b.e.r.abort();
        const n = requests.len;
        const packed_ = batch_draft.encode(b, ss[0..n], follows[0..n], hiddens[0..n], depths[0..n], draws[0..n], lands[0..n]) catch |err| {
            if (b.e.r.fsr == null) b.e.r.enc.end();
            return err;
        };
        if (!packed_) for (requests, 0..) |r, k| {
            const s = try b.session(r.stream);
            const old = try b.bind(s);
            defer old.restore(b.e);
            s.encodeDrafts(follows[k], hiddens[k], r.depth) catch |err| {
                if (b.e.r.fsr == null) b.e.r.enc.end();
                return err;
            };
        };
        b.times.commit();
        try b.primary.finish(cb);
        b.times.end(cb, true, 0, requests.len);
        if (packed_) batch_draft.read(&b.packed_drafts);
        for (requests) |r| (try b.session(r.stream)).heldDrafts(r.depth);
        b.draft_waits += 1;
        b.draft_calls += requests.len;
        b.draft_seconds += mtl.clock.seconds() - start;
    }

    fn release(ptr: *anyopaque, stream: *Stream) void {
        const b = self(ptr);
        prompt_gpu.clear(b, stream);
        if (std.mem.indexOfScalar(*Stream, b.paused.items, stream)) |k| _ = b.paused.orderedRemove(k);
        const kv = b.by.fetchRemove(stream) orelse return;
        b.sessions[kv.value].used = false;
        b.sessions[kv.value].round = .{};
        b.sessions[kv.value].held_n = 0;
        if (b.paused.items.len > 0) b.resumePaused();
    }

    pub fn snapBytes(ptr: *anyopaque, at: u32) u64 {
        return self(ptr).e.snap_pool.size(snap.bytes(at));
    }
    pub fn snapCharged(_: *anyopaque, saved: *anyopaque) u64 {
        const state: *snap.State = @ptrCast(@alignCast(saved));
        return state.cap;
    }
    pub fn snapSpare(ptr: *anyopaque) u64 {
        return self(ptr).e.snap_pool.spare();
    }
    pub fn snapTrim(ptr: *anyopaque, room: u64) void {
        self(ptr).e.snap_pool.trim(room);
    }
    pub fn snapReuses(ptr: *anyopaque, at: u32) bool {
        return self(ptr).e.snap_pool.fits(snap.bytes(at));
    }
    pub fn snapSave(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
        const b = self(ptr);
        const s = try b.session(@ptrCast(@alignCast(owner orelse return error.NoStream)));
        const old = try b.bind(s);
        defer old.restore(b.e);
        return try snap.save(b.e, b.gpa, at);
    }
    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores;
    }
    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        snap.drop(self(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};

test "batch widths: shared verifier matches the legacy forward through lane rounds" {
    if (comptime @hasDecl(mtl, "flashnext_recording")) {
        try @import("batch_forward_test.zig").check(Backend.verify, Backend.keep);
    } else return error.SkipZigTest;
}
