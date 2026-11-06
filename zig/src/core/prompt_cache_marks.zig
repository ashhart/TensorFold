//! The marks a prompt pass keeps: history, tail/end states, shared blocks and the fork point.
const std = @import("std");
const Allocator = std.mem.Allocator;
const pc = @import("prompt_cache.zig");
const Store = pc.Store;
const Entry = pc.Entry;
const modes = @import("cache_modes.zig");
const cuts = @import("prompt_cuts.zig");
const forks = @import("prompt_cache_fork.zig");
const chunkClass = pc.chunkClass;
const freshLast = pc.freshLast;

/// `marks` with the active rewind second: kept however near the other marks, never at or before the resume point.
/// `tail_at_out` is the tail mark it planned (0: none), which a backend may snapshot inside a chunk (tail_snap).
pub fn marksRewind(s: *const Store, a: Allocator, prompt: []const u32, from: u32, history_len: u32, rewind: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, previous: []const u32, tail_at_out: *u32) ![]const u32 {
    tail_at_out.* = 0;
    const short = !s.rules.planned and prompt.len < s.rules.min_prompt; // the history's mark only, where the pass cuts a chunk anyway
    if (short and s.rules.chunk == 0) return &.{};
    var out: std.ArrayList(u32) = .empty;
    var want: std.ArrayList(u32) = .empty;
    defer want.deinit(a);
    // a warm pass (its history the whole prompt) keeps where a state can be keyed, its lookahead before the end
    const warm_pass = s.rules.warm != .off and history_len == prompt.len and s.rules.lookahead > 0;
    const repeat_at = s.repeat_log.mark(s.rules.repeat_min, prompt, from, s.rules.lookahead, s.rules.planned);
    const history: u32 = if (warm_pass) @intCast(prompt.len -| s.rules.lookahead) else if (history_len == 0) repeat_at else history_len;
    try want.append(a, if (short) cuts.chunkEnd(s.rules.chunk, from, history, prompt.len, starts, spans) else history);
    // The tail mark: one off-grid state per conversation at the pass's last whole tile, so the next turn
    // resumes at its own boundary and prefills only what it added. Tile-aligned (a resumed suffix must sit in
    // the lanes a fresh pass gives it) and used only while its suffix is one chunk. It is planned whenever the
    // bulk piece from `from` is a tensor-tile chunk (>= 64 rows), so the pass splits where its own grid split
    // would put a chunk end anyway and the next turn resumes near its end. On a pass from 0 the mark is a cut;
    // on a resumed pass it is a snapshot instead: the stash captures it inside the chunk that runs past it, so
    // one forward keeps both the tile state and the pass's own end. It needs a tensor-tile suffix (>= 64 rows)
    // for `snap_at` to fire; a shorter one would make the mark a barrier again.
    const resumed_tail = s.rules.resumed_tail;
    const tail_at: u32 = if (s.rules.tail and s.rules.tail_align > 1 and !short and prompt.len > s.rules.min_prompt and
        (!s.rules.end_mark or from == 0 or (s.rules.tail_snap and resumed_tail)))
    blk: {
        const tile: usize = @max(1, s.rules.tail_align);
        // The cut's last chunk keeps the reference's kernel class. A pass with no marks ends on `prompt.len %
        // grid` rows (the grid's whole chunks, then the remainder); the cut ends on `prompt.len % tile`. A
        // chunk's class changes at 8 rows (decode) and 64 (shared body vs the tensor tile), so a remainder in
        // the other class runs different arithmetic and can move the reply. Back the cut off whole tiles until
        // the class matches.
        const fresh_last: usize = if (s.rules.grid > 0 and prompt.len % s.rules.grid != 0)
            prompt.len % s.rules.grid
        else
            @min(prompt.len, @as(usize, @max(1, s.rules.grid)));
        const class_want = chunkClass(fresh_last);
        var f: usize = prompt.len % tile;
        if (f == 0) f = tile;
        var guard: u32 = 0;
        while (chunkClass(f) != class_want and guard < 4) : (guard += 1) f += tile;
        if (chunkClass(f) != class_want or f > prompt.len) break :blk 0;
        const at: u32 = @intCast(prompt.len - f);
        // A resumed pass advances only when its bulk piece is a tensor-tile chunk (the reference's own
        // intermediate chunks are) and the new state is past the point this pass resumed at.
        if (from > 0 and (at <= from or at - from < 64)) break :blk 0;
        if (from > 0 and prompt.len - at < 64) break :blk 0; // a resumed tail must be snapshottable, never a cut
        break :blk at;
    } else 0;
    tail_at_out.* = tail_at;
    if (tail_at != 0 and std.mem.indexOfScalar(u32, want.items, tail_at) == null) try want.append(a, tail_at);
    // THE PASS'S OWN END IS A STATE TOO, CLASS-CHECKED. The tile tail serves a turn that adds only a few rows;
    // a turn that adds a tensor-tile chunk does not need the 128 rows behind it re-prefilled (one whole chunk's
    // all-expert read, the cache turn's floor). A state at the very end is the state a fresh pass stands on
    // when this pass's last chunk kept the fresh remainder's class: rows are boundary- and lane-invariant
    // inside a class. `find` re-checks the class against the next turn's own remainder, so a small turn still
    // takes the tail.
    const end_at: u32 = if (s.rules.end_mark and !short and prompt.len > s.rules.min_prompt) @intCast(prompt.len) else 0;
    if (end_at != 0 and std.mem.indexOfScalar(u32, want.items, end_at) == null) try want.append(a, end_at);
    if (rewind > 0 and s.active_rewind_at == rewind) try want.append(a, rewind);
    // the prompt this one parts from: the one it resumed, or (fork rule) on a miss the kept one sharing the most
    const kept = if (previous.len > 0 or !s.rules.fork) previous else forks.longestShared(Entry, s.entries.items, prompt);
    const part: u32 = if (kept.len == 0 or short) 0 else @intCast(std.mem.indexOfDiff(u32, kept, prompt) orelse @min(kept.len, prompt.len));
    // the fork rule keeps one before the part, however near: a state at `at` is keyed with the token at `at`
    const stable: u32 = if (part == 0 or part >= history_len or part < history_len / 2) 0 else if (s.rules.fork) part -| s.rules.lookahead else part;
    if (stable > 0) try want.append(a, stable);
    if (!short) try want.appendSlice(a, shared);
    // tail: a prompt that names no history keeps its last chunk too (a chat keeps its history's mark alone)
    if (!short and s.rules.tail and history_len == 0) try want.append(a, @intCast(prompt.len -| @max(1, s.rules.lookahead)));
    const fork_at = if (s.rules.fork and s.rules.planned) s.startAt(starts, stable) else stable; // where the loop snaps it
    for (want.items, 0..) |w, k| {
        const tail = tail_at != 0 and w == tail_at;
        const end = end_at != 0 and w == end_at;
        const at = if (s.rules.planned and !tail and !end) s.startAt(starts, w) else w;
        if (at <= from or at + s.rules.lookahead > prompt.len or at > prompt.len or (at >= prompt.len and !tail and !end) or !s.usable(at, starts)) continue;
        const priority = tail or end or (rewind > 0 and k == 1 and at == rewind) or (s.rules.fork and at == fork_at);
        if (!priority and !(k == 0 and warm_pass) and at - from < s.rules.min_gap and (k > 0 or (from > 0 and s.rules.warm == .on))) continue; // near the resume point
        if (k > 0 and !priority) { // the history's mark always (warm families: away from the resume point); the others away from it and each other
            const near = for (out.items) |o| {
                if (@max(o, at) - @min(o, at) < s.rules.min_gap) break true;
            } else false;
            if (near) continue;
        }
        if (std.mem.indexOfScalar(u32, out.items, at) == null) try out.append(a, at);
    }
    std.mem.sort(u32, out.items, {}, std.sort.asc(u32));
    if (end_at != 0) { // the last chunk this pass actually runs must keep the fresh remainder's class
        var last_from: u32 = from;
        for (out.items) |o| {
            if (o > last_from and o < end_at) last_from = o;
        }
        const span: usize = end_at - last_from;
        const fresh: usize = freshLast(s.rules.grid, end_at);
        if (s.rules.grid < 64 or span < 64 or chunkClass(span) != chunkClass(fresh)) {
            if (std.mem.indexOfScalar(u32, out.items, end_at)) |i| {
                _ = out.orderedRemove(i);
            }
        }
    }
    return out.toOwnedSlice(a);
}
