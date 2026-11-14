//! The next prompt chunk's Engram rows, read on a worker while the GPU runs the current chunk; taken only on equal ids.
const std = @import("std");
const engram = @import("engram.zig");
const hash = @import("engram_hash.zig");
const model = @import("model.zig");
const Model = model.Model;
const nowNs = model.nowNs;

/// Lane slots that keep a hint.
pub const slots = 8;

/// Chunks below this many rows stage inline: their gather is cheap and a hint would only add a thread.
pub const min_rows: usize = 512;

pub const State = struct {
    gpa: std.mem.Allocator,
    /// The hinted tokens (the chunk after the one being encoded) and the slot whose history they continue.
    tokens: std.ArrayList(u32) = .empty,
    slot: u32 = 0,
    armed: bool = false,
    /// Per layer: the hashed ids of the hinted rows, then both layers' staged rows (layer-major, `ready` rows each).
    ids: [2]std.ArrayList(u32) = .{ .empty, .empty },
    rows: std.ArrayList(u16) = .empty,
    /// Rows the worker finished (0: none, or a failed read).
    ready: usize = 0,
    thread: ?std.Thread = null,
    worker_ns: u64 = 0,

    pub fn deinit(s: *State) void {
        s.join();
        s.tokens.deinit(s.gpa);
        for (&s.ids) |*l| l.deinit(s.gpa);
        s.rows.deinit(s.gpa);
    }

    pub fn join(s: *State) void {
        if (s.thread) |t| t.join();
        s.thread = null;
    }
};

/// TF_DSV41_ENGRAM_LOOKAHEAD=0 stages every chunk's rows before its GPU work, as before.
fn enabled() bool {
    const v = std.c.getenv("TF_DSV41_ENGRAM_LOOKAHEAD") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

fn state(m: *Model, slot: u32) !*State {
    if (slot >= m.look.len) return error.InvalidLaneSlot;
    if (m.look[slot]) |s| return s;
    const s = try m.gpa.create(State);
    s.* = .{ .gpa = m.gpa };
    m.look[slot] = s;
    return s;
}

pub fn deinit(m: *Model) void {
    for (&m.look) |*slot| if (slot.*) |s| {
        s.deinit();
        m.gpa.destroy(s);
        slot.* = null;
    };
}

/// The ids and rows of the hinted chunk, from the history as the current chunk left it.
fn work(m: *Model, s: *State) void {
    workInner(m, s) catch |err| {
        std.log.warn("engram lookahead: {s}", .{@errorName(err)});
        s.ready = 0;
    };
}

fn workInner(m: *Model, s: *State) !void {
    const t0 = nowNs();
    const src = m.stream orelse return error.EngramNotOpen;
    const ws = &m.workspace;
    const n = s.tokens.items.len;
    const history = try ws.historyView(s.slot);
    const token_map = ws.view(.token_map);
    const cols: usize = hash.columns;
    for (&s.ids) |*l| try l.resize(s.gpa, n * cols);
    try s.rows.resize(s.gpa, 2 * n * engram.gather_dim);
    try engram.peekHashChunk(history, token_map, s.tokens.items, s.ids[0].items, s.ids[1].items);
    for (0..2) |li| {
        const out = s.rows.items[li * n * engram.gather_dim ..][0 .. n * engram.gather_dim];
        try src.rowsU32Parallel(li, s.ids[li].items, engram.head_dim, out, m.engramThreads());
    }
    s.ready = n;
    s.worker_ns = nowNs() - t0;
}

/// The chunk after the next forward's, so that forward can read its rows while its GPU work runs.
pub fn hint(m: *Model, slot: u32, tokens: []const u32) void {
    if (m.stream == null or tokens.len < min_rows or !enabled()) return;
    const s = state(m, slot) catch return;
    s.join();
    s.tokens.clearRetainingCapacity();
    s.tokens.appendSlice(s.gpa, tokens) catch return;
    s.slot = slot;
    s.ready = 0;
    s.armed = true;
}

/// Start the worker once this forward's own staging is done (the Engram source is not shared between threads).
pub fn start(m: *Model, slot: u32) void {
    if (slot >= m.look.len) return;
    const s = m.look[slot] orelse return;
    if (!s.armed) return;
    s.armed = false;
    if (s.tokens.items.len == 0) return;
    s.ready = 0;
    s.thread = std.Thread.spawn(.{}, work, .{ m, s }) catch null;
}

pub fn join(m: *Model, slot: u32) void {
    if (slot < m.look.len) if (m.look[slot]) |s| s.join();
}

/// Copy the first `rows` rows into the stage when the hinted chunk is this one; false: stage them normally.
pub fn take(m: *Model, slot: u32, rows: u32) bool {
    if (slot >= m.look.len) return false;
    const s = m.look[slot] orelse return false;
    s.join();
    const n = s.ready;
    if (n < rows or rows < min_rows) return false;
    const buf = m.stream_stage orelse return false;
    const ws = &m.workspace;
    const cols: usize = hash.columns;
    const want = @as(usize, rows) * cols;
    for ([_]@TypeOf(ws.view(.engram_ids_0)){ ws.view(.engram_ids_0), ws.view(.engram_ids_1) }, 0..) |v, li| {
        const at: [*]const u32 = @ptrCast(@alignCast(v.buffer.contents() + v.offset));
        if (!std.mem.eql(u32, at[0..want], s.ids[li].items[0..want])) return false;
    }
    const layer = @as(usize, rows) * engram.gather_dim;
    const base: [*]u16 = @ptrCast(@alignCast(buf.contents()));
    for (0..2) |li| @memcpy(base[li * layer ..][0..layer], s.rows.items[li * n * engram.gather_dim ..][0..layer]);
    s.ready = 0;
    s.rows.clearAndFree(s.gpa);
    return true;
}
