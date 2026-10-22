//! Session-owned prompt resources and completed Metal calls behind the shared stepped driver.
const std = @import("std");
const lanes = @import("lanes");
const fz = @import("replay.zig");
const engine = @import("engine.zig");
const Session = @import("session.zig").Session;
const fill = @import("prefill.zig");
const segments = @import("../../core/segments.zig");
const slots = @import("slots.zig");
const snap = @import("snapshot.zig");
const sample = @import("batch_sample.zig");

const Images = struct {
    arena: std.heap.ArenaAllocator,
    vis: *fz.vis_prompt.VisPrompt,
};

pub fn begin(b: anytype, stream: *lanes.Stream, preferred: ?usize) !fill.State {
    const prompt = stream.prompt();
    if (prompt.len == 0) return error.EmptyPrompt;
    if (b.by.contains(stream)) return error.PromptAlreadyStarted;
    const i = preferred orelse try b.slotFor(prompt.len, stream.max_new);
    if (b.sessions[i].used) return error.NoFreeSlot;
    if (preferred != null and prompt.len + stream.max_new + slots.WINDOW > b.sessions[i].m.cap)
        return error.ContextFull;
    const s = &b.sessions[i];
    try b.by.put(b.gpa, stream, i);
    errdefer b.abortPrompt(stream);
    s.used = true;
    s.held_n = 0;
    s.round = .{};
    s.row0 = 0;
    b.outputs(s, 0);
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
    var owner: ?*Images = null;
    if (stream.images.len > 0) {
        const images = b.images orelse return error.NoImageTower;
        const owned = try b.gpa.create(Images);
        errdefer b.gpa.destroy(owned);
        owned.arena = .init(b.gpa);
        errdefer owned.arena.deinit();
        owned.vis = try images.build(images.ctx, owned.arena.allocator(), stream.images, prompt.len, s.m.pos);
        owner = owned;
    }
    const at: u32 = @intCast(s.m.pos);
    stream.fill_at.store(at, .monotonic);
    return .{ .stream = stream, .slot = i, .at = at, .owner = owner };
}

pub fn rows(b: anytype, at: u32, end: u32) u32 {
    const step = fz.chunkRows(at, b.e.pr.step);
    const call = if (b.e.segments) segments.next(end - at, step, engine.SEG_MIN) else segments.Call{ .rows = @min(end - at, step), .parts = 1 };
    return @intCast(call.rows);
}

pub fn chunk(b: anytype, stream: *lanes.Stream, i: usize, at: u32, n: u32) !void {
    const s = &b.sessions[i];
    if (s.m.pos != at) return error.PositionMismatch;
    const old = try b.bind(s);
    defer old.restore(b.e);
    b.outputs(s, 0);
    const state = b.filling.get(stream) orelse return error.PromptOutOfStep;
    const owner: ?*Images = if (state.owner) |p| @ptrCast(@alignCast(p)) else null;
    s.m.vis = if (owner) |o| o.vis else null;
    defer s.m.vis = null;
    const prompt = stream.prompt();
    const end = @as(usize, at) + n;
    const step = fz.chunkRows(at, b.e.pr.step);
    const call = if (b.e.segments) segments.next(n, step, engine.SEG_MIN) else segments.Call{ .rows = n, .parts = 1 };
    if (call.rows != n) return error.ChunkOutOfStep;
    const ps = [2]*fz.Prompt{ b.e.pr, b.e.pr2 };
    s.first = try fz.Prompt.chunkN(ps[0..call.parts], &s.m, b.gpa, prompt[at..end]);
    if (end == prompt.len and stream.grammar != null) try b.grammar.keepFirst(b.gpa, i, b.sessions.len, s.m.t.logits);
    // Draw before head work can replace the final target row's logits.
    if (end == prompt.len) if (sample.active(stream.sampling)) |p| {
        s.first = try sample.first(b.e, &s.m, p, @intCast(prompt.len));
    };
    for (ps[0..call.parts], 0..) |p, k| {
        const span = try fill.piece(at, n, call.parts, k, prompt.len);
        try p.mtpKeys(&s.m, span.start, prompt[span.start + 1 ..][0..span.head_rows], p.last);
        if (span.final) Session.copy(p.last.at((span.rows - 1) * fz.WIDE * 2), s.last, fz.WIDE * 2);
    }
}

pub fn mark(b: anytype, stream: *lanes.Stream, at: u32) !void {
    const i = b.by.get(stream) orelse return error.UnknownStream;
    const s = &b.sessions[i];
    const old = try b.bind(s);
    defer old.restore(b.e);
    b.outputs(s, 0);
    if (stream.reuse.hook) |hook| hook.at(hook.ptr, stream, at);
}

fn dropImages(b: anytype, resource: ?*anyopaque) void {
    const owner: *Images = @ptrCast(@alignCast(resource orelse return));
    const images = b.images.?;
    images.release(images.ctx, owner.vis);
    owner.arena.deinit();
    b.gpa.destroy(owner);
}

pub fn finish(b: anytype, stream: *lanes.Stream) !void {
    const state = b.filling.getPtr(stream) orelse return error.PromptOutOfStep;
    const s = &b.sessions[state.slot];
    if (s.m.pos != stream.prompt().len) return error.PromptIncomplete;
    if (s.m.state != 0 or s.m.state_row != 0) return error.PromptStateNotNormalized;
    s.m.mtp.pos = stream.prompt().len - 1;
    s.m.mtp.drafted = 0;
    const owner = state.owner;
    state.owner = null;
    dropImages(b, owner);
}

pub fn clear(b: anytype, stream: *lanes.Stream) void {
    if (b.filling.fetchRemove(stream)) |entry| dropImages(b, entry.value.owner);
}
