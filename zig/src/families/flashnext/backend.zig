//! Flash Next's isolated sessions behind the shared lane scheduler; target rounds batch, MTP drafts are serial.
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

pub const Backend = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    primary: *fz.Model,
    sessions: []Session,
    by: std.AutoHashMapUnmanaged(*Stream, usize) = .empty,
    hidden: Buf,
    window_count: u64 = 0,
    shared_count: u64 = 0,

    pub fn init(gpa: std.mem.Allocator, e: *Engine, n: u32, capacity: usize) !Backend {
        if (n < 2 or n > fz.MAXR or e.r.tp != null) return error.BadBatchSlots;
        const sessions = try gpa.alloc(Session, n);
        errdefer gpa.free(sessions);
        var made: usize = 0;
        errdefer for (sessions[0..made]) |*s| s.deinit(gpa);
        for (sessions, 0..) |*s, i| {
            s.* = try Session.init(gpa, e, capacity, i == 0);
            made += 1;
        }
        const hidden = try e.r.device.buffer(fz.MAXR * fz.WIDE * 2, fz.opts);
        return .{ .gpa = gpa, .e = e, .primary = e.m, .sessions = sessions, .hidden = .{ .b = hidden } };
    }

    pub fn deinit(b: *Backend) void {
        b.e.m = b.primary;
        b.by.deinit(b.gpa);
        for (b.sessions) |*s| s.deinit(b.gpa);
        b.gpa.free(b.sessions);
        b.hidden.b.deinit();
    }

    pub fn backend(b: *Backend) be.Backend {
        return .{ .ptr = b, .vtable = &.{ .prefill = prefill, .first = first, .queue = queue, .read = read, .verify = verify, .keep = keep, .draft = draft, .release = release } };
    }

    pub fn facts(b: *Backend) lanes.Model {
        return .{ .exact_width = fz.MAXR, .first_copy_rows = 4, .mtp = true, .speculate = true, .speculate_early = false, .drafts = 8, .streams_exact = true, .hidden_rows = true, .batch_rows = fz.MAXR, .max_streams = @intCast(b.sessions.len), .draft_streams = false };
    }

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

    const Binding = struct {
        model: *fz.Model,
        host: @FieldType(Engine, "host"),
        cins: [2]Buf,
        select: ?fz.Select,
        shape: @TypeOf(@as(fz.Select, undefined).q_shape),
        passed: @FieldType(Engine, "passed"),
        fn restore(old: Binding, b: *Backend) void {
            b.e.m = old.model;
            b.e.host = old.host;
            b.e.cins = old.cins;
            b.e.r.sel = old.select;
            b.e.r.shapes.put(b.e.r.arena, "Kc_shape", old.shape) catch unreachable;
            b.e.passed = old.passed;
        }
    };

    fn bind(b: *Backend, s: *Session) !Binding {
        const e = b.e;
        const old: Binding = .{ .model = e.m, .host = e.host, .cins = e.cins, .select = e.r.sel, .shape = e.r.shapes.get("Kc_shape").?, .passed = e.passed };
        try e.r.shapes.put(e.r.arena, "Kc_shape", s.shape);
        e.m = &s.m;
        e.host = .{ .rows = s.m.t.rows, .mdims = s.m.t.mdims, .pos8 = s.m.t.pos8, .nk8 = s.m.t.nk8, .kvmeta = s.m.t.kvmeta, .slots = s.m.mtp.slots };
        e.cins[0] = s.m.ple.cin;
        e.r.sel = s.select[fz.CATCH - 1];
        e.passed = null;
        e.hostMode();
        return old;
    }

    fn prefill(ptr: *anyopaque, stream: *Stream) anyerror!void {
        const pool = mtl.objc.Pool.push();
        defer pool.pop();
        const b = self(ptr);
        if (stream.sampling) |p| if (p.temperature > 0) return error.GreedyOnly;
        const prompt = stream.prompt();
        if (prompt.len == 0) return error.EmptyPrompt;
        try b.settle();
        const i = for (b.sessions, 0..) |s, j| {
            if (!s.used) break j;
        } else return error.NoFreeSlot;
        const s = &b.sessions[i];
        if (prompt.len + stream.max_new + fz.MAXR + 1 > s.m.cap) return error.ContextFull;
        try b.by.put(b.gpa, stream, i);
        s.used = true;
        s.held_n = 0;
        s.round = .{};
        s.row0 = 0;
        outputs(b, s, 0);
        const old = try b.bind(s);
        defer old.restore(b);
        s.m.reset();
        stream.cached = 0;
        if (stream.reuse.saved) |saved| {
            const state: *snap.State = @ptrCast(@alignCast(saved));
            if (state.at < prompt.len and state.at < s.m.cap) {
                try snap.restore(b.e, state);
                stream.cached = @intCast(state.at);
            } else stream.reuse_failed = true;
        }
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

    fn outputs(b: *Backend, s: *Session, row: usize) void {
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
        try forward.encode(b.e, segs[0..windows.len], plan.total, t.ids8);
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
            defer old.restore(b);
            s.held_n = r.depth;
            if (follow.len == 0) return error.DraftOutOfStep;
            const token = try s.m.mtpAbsorb(follow, hidden);
            if (r.depth > 0) s.held[0] = token;
            if (r.depth > 1) for (1..r.depth) |i| {
                s.held[i] = try s.m.mtpChain(s.held[i - 1]);
            };
        }
    }

    fn release(ptr: *anyopaque, stream: *Stream) void {
        const b = self(ptr);
        const kv = b.by.fetchRemove(stream) orelse return;
        b.sessions[kv.value].used = false;
        b.sessions[kv.value].round = .{};
        b.sessions[kv.value].held_n = 0;
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
        defer old.restore(b);
        return try snap.save(b.e, b.gpa, at);
    }
    pub fn snapRestore(_: *anyopaque, _: ?*anyopaque, _: *anyopaque) anyerror!void {
        return error.BackendRestores;
    }
    pub fn snapDrop(ptr: *anyopaque, saved: *anyopaque) void {
        snap.drop(self(ptr).gpa, @ptrCast(@alignCast(saved)));
    }
};
