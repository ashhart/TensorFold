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

/// A stream's images through the host's tower: its prompt pass's rows and frame (in `a`), then their release.
pub const Images = struct { ctx: *anyopaque, build: *const fn (ctx: *anyopaque, a: std.mem.Allocator, images: []const lanes.stream.Image, prompt_len: usize, from: usize) anyerror!*fz.vis_prompt.VisPrompt, release: *const fn (ctx: *anyopaque, p: *fz.vis_prompt.VisPrompt) void };

pub const Backend = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    primary: *fz.Model,
    sessions: []Session,
    by: std.AutoHashMapUnmanaged(*Stream, usize) = .empty,
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
    costs: ?batch_costs.Costs = null, // timeCosts: the lane allocator's prices
    images: ?Images = null, // --vision: the tower's rows for a stream's images; null: an image stream fails
    grammar: grammar.Masks, // structured output: the rounds' mask rows and first rows

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
        return .{ .gpa = gpa, .e = e, .primary = e.m, .sessions = sessions, .hidden = .{ .b = hidden }, .grammar = masks, .chain_drafts = env.on("FZ_BATCH_MTP_CHAIN"), .serial_drafts = env.on("FZ_BATCH_MTP_SERIAL") };
    }

    /// Times shared-round costs on the last free session, or leaves them unpriced when it cannot fit.
    pub fn timeCosts(b: *Backend) !void {
        return batch_costs.timeCosts(b);
    }

    pub fn deinit(b: *Backend) void {
        b.e.m = b.primary;
        b.paused.deinit(b.gpa);
        b.by.deinit(b.gpa);
        for (b.sessions) |*s| s.deinit(b.gpa);
        b.gpa.free(b.sessions);
        b.hidden.b.deinit();
        b.grammar.deinit(b.gpa);
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release, .first_masked = grammar.firstMasked, .held = grammar.held } };
    }

    pub const facts = batch_costs.facts;

    fn self(ptr: *anyopaque) *Backend {
        return @ptrCast(@alignCast(ptr));
    }
    fn session(b: *Backend, s: *Stream) !*Session {
        return &b.sessions[b.by.get(s) orelse return error.UnknownStream];
    }
    fn settle(b: *Backend) !void {
        var needed = false;
        for (b.sessions) |*s| if (s.used) {
            if (s.round.pending) try s.keep(s.round.rows);
            needed = needed or s.m.state != 0 or s.m.state_row != 0;
        };
        if (!needed) return;
        const cb = b.e.r.queue.commandBuffer();
        const blit = cb.blit();
        for (b.sessions) |*s| if (s.used) s.normalizeGpu(blit);
        blit.end();
        cb.commit();
        cb.wait();
        if (cb.failure()) |message| {
            std.log.err("session state copy failed: {s}", .{message});
            return error.GpuFailed;
        }
        for (b.sessions) |*s| if (s.used) {
            s.m.state = 0;
            s.m.state_row = 0;
        };
    }

    const Binding = @import("session_binding.zig").Binding;

    pub fn bind(b: *Backend, session_: *Session) !Binding {
        return @import("session_binding.zig").bind(b.e, session_);
    }

    /// Chooses a slot; LaneMemory means wait, ContextFull means no session can hold the reply.
    fn slotFor(b: *Backend, prompt: usize, max_new: usize) !usize {
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

    fn prefill(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        try self(ptr).fill(stream, null);
    }

    /// A stream's whole prompt pass, in session `slot` when given (free), else the first free one.
    fn fill(b: *Backend, stream: *Stream, slot: ?usize) !void {
        const prompt = stream.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        try b.settle();
        const i = slot orelse try b.slotFor(prompt.len, stream.max_new);
        if (b.sessions[i].used) return error.NoFreeSlot;
        if (slot != null and prompt.len + stream.max_new + slots.WINDOW > b.sessions[i].m.cap) return error.ContextFull;
        const s = &b.sessions[i];
        try b.by.put(b.gpa, stream, i);
        errdefer {
            _ = b.by.remove(stream);
            s.used = false;
        }
        s.used = true;
        s.held_n = 0;
        s.round = .{};
        s.row0 = 0;
        outputs(b, s, 0);
        const old = try b.bind(s);
        defer old.restore(b.e);
        s.m.reset();
        stream.cached = 0;
        if (stream.reuse.saved) |saved| {
            const state: *snap.State = @ptrCast(@alignCast(saved));
            if (state.at < prompt.len and state.at < s.m.cap) {
                try snap.restore(b.e, state);
                stream.cached = @intCast(state.at);
            } else stream.reuse_failed = true;
        }
        var arena: std.heap.ArenaAllocator = .init(b.gpa);
        defer arena.deinit();
        const vis: ?*fz.vis_prompt.VisPrompt = if (stream.images.len > 0) try (b.images orelse return error.NoImageTower).build(b.images.?.ctx, arena.allocator(), stream.images, prompt.len, s.m.pos) else null;
        s.m.vis = vis;
        defer s.m.vis = null;
        defer if (vis) |p| b.images.?.release(b.images.?.ctx, p);
        var at: usize = s.m.pos;
        while (at < prompt.len) {
            if (stream.isCancelled()) return error.Cancelled;
            var end = prompt.len;
            for (stream.chunks) |mark| if (mark > at) {
                end = @min(end, mark);
                break;
            };
            for (stream.reuse.marks) |mark| if (mark > at) {
                end = @min(end, mark);
                break;
            };
            const call = if (b.e.segments) segments.next(end - at, b.e.pr.step, @import("engine.zig").SEG_MIN) else segments.Call{ .rows = @min(end - at, b.e.pr.step), .parts = 1 };
            end = at + call.rows;
            const ps = [2]*fz.Prompt{ b.e.pr, b.e.pr2 };
            s.first = try fz.Prompt.chunkN(ps[0..call.parts], &s.m, b.gpa, prompt[at..end]);
            if (end == prompt.len and stream.grammar != null) try b.grammar.keepFirst(b.gpa, i, b.sessions.len, s.m.t.logits);
            // Sample before the next forward replaces the last prompt row's logits.
            if (end == prompt.len) if (sample.active(stream.sampling)) |p| {
                s.first = try sample.first(b.e, &s.m, p, @intCast(prompt.len));
            };
            for (ps[0..call.parts], 0..) |p, k| {
                const start = at + segments.start(call.rows, call.parts, k);
                const rows = segments.rows(call.rows, call.parts, k);
                const count = if (start + rows < prompt.len) rows else rows - 1;
                try p.mtpKeys(&s.m, start, prompt[start + 1 .. start + 1 + count], p.last);
                if (start + rows == prompt.len) Session.copy(p.last.at((rows - 1) * fz.WIDE * 2), s.last, fz.WIDE * 2);
            }
            s.normalize();
            at = end;
            if (std.mem.indexOfScalar(u32, stream.reuse.marks, @intCast(at)) != null)
                if (stream.reuse.hook) |hook| hook.at(hook.ptr, stream, @intCast(at));
        }
        s.m.mtp.pos = prompt.len - 1;
        s.m.mtp.drafted = 0;
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
        try b.fill(stream, 0);
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
        try b.settle();
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
        const plan = try meta.Plan.init(desc[0..windows.len], fz.CAP);
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
            outputs(b, s, row);
            segs[i] = .{ .s = s, .row0 = row, .rows = s.round.rows };
            @memcpy(o.drafts, s.tokens[1..s.round.rows]);
        }
        t.rows.b.slice(i32, 1)[0] = @intCast(plan.total);
        t.mdims.b.slice(i32, 2)[0] = @intCast(plan.total);
        @memcpy(t.pos8.b.slice(u32, fz.MAXR)[0..plan.total], plan.positions[0..plan.total]);
        const cb = b.e.r.queue.commandBuffer();
        b.e.r.enc = cb.compute(.serial);
        const masked = b.grammar.fill(windows, plan.starts[0..windows.len]);
        try forward.encode(b.e, segs[0..windows.len], plan.total, t.ids8, if (masked) &b.grammar else null);
        @memcpy(b.e.draw_at.b.slice(u32, fz.MAXR)[0..plan.total], plan.keys[0..plan.total]);
        for (windows, 0..) |w, i| if (sample.active(w.stream.sampling)) |p| {
            sample.encode(b.e.r.enc, b.e.r.draw_pipe, t.logits, b.e.draw_at, t.picks, plan.starts[i], w.rows(), p);
        };
        try b.primary.finish(cb);
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
        var follows: [fz.MAXR][]const u32 = undefined;
        var hiddens: [fz.MAXR]Buf = undefined;
        var ones: [fz.MAXR][1]u32 = undefined;
        if (requests.len > fz.MAXR) return error.DraftOutOfStep;
        for (requests, 0..) |r, k| {
            if (r.early or r.lanes != null or r.ranks or r.depth >= fz.MAXR) return error.DraftOutOfStep;
            const s = try b.session(r.stream);
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
        b.e.r.enc = cb.compute(if (b.e.r.serial) .serial else .concurrent);
        for (requests, 0..) |r, k| {
            const s = try b.session(r.stream);
            const old = try b.bind(s);
            defer old.restore(b.e);
            s.encodeDrafts(follows[k], hiddens[k], r.depth) catch |err| {
                b.e.r.enc.end();
                return err;
            };
        }
        try b.primary.finish(cb);
        for (requests) |r| (try b.session(r.stream)).heldDrafts(r.depth);
        b.draft_waits += 1;
        b.draft_calls += requests.len;
        b.draft_seconds += mtl.clock.seconds() - start;
    }

    fn release(ptr: *anyopaque, stream: *Stream) void {
        const b = self(ptr);
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
