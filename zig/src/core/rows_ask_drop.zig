//! POST /v1/imprint/forget on the engine thread: a conversation's own kept states leave RAM and disk; shared ones stay.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = pc.rows;
const ask = @import("rows_ask.zig");

/// How long saves the I/O thread is writing may take to land; past it the store's fence cancels them.
const settle_ns: i128 = 10 * std.time.ns_per_s;

/// The named conversation: its cache words as the host's lookups key them (a framed image request's first word apart).
const Named = struct {
    key: []const u32,
    frame: ?u32 = null, // a framed family's first word for this image request, which its states carry for key[0]

    /// On the conversation's path: a prefix of its prompt, or past it (a later turn's).
    fn related(n: Named, t: []const u32) bool {
        return n.with(t, n.key[0], false) or (if (n.frame) |w| n.with(t, w, false) else false);
    }

    /// At or below the named prompt: a state that prompt itself resumes from.
    fn below(n: Named, t: []const u32) bool {
        return n.with(t, n.key[0], true) or (if (n.frame) |w| n.with(t, w, true) else false);
    }

    fn with(n: Named, t: []const u32, first: u32, inside: bool) bool {
        const m = @min(t.len, n.key.len);
        if (m == 0 or (inside and t.len > n.key.len) or t[0] != first) return false;
        return std.mem.eql(u32, t[1..m], n.key[1..m]);
    }
};

/// The conversation keyed by `a.prompt` dropped: each of its states no other conversation builds on, then its answer.
pub fn drop(a: *ask.Ask, s: *pc.Store, keys: ask.Keys) void {
    const frame: u32 = if (keys.frame) |f| (if (a.images.len > 0) f(a.images) else 0) else 0;
    defer resumes(a, s, keys, frame);
    if (a.prompt.len == 0) return;
    const n: Named = .{ .key = a.prompt, .frame = if (frame != 0) pc.framed(a.prompt[0], frame) else null };
    const r = s.disk;
    if (r) |d| if (!settle(d)) rows.write.fence(d); // a save of it still in flight never publishes
    for (s.entries.items) |e| if (!e.expired and n.related(e.tokens)) {
        if (stays(s, n, e.tokens, e.shared)) {
            a.shared += 1;
            continue;
        }
        e.expired = true; // expired first: nothing is staged for disk on its way out, and a queued save of it goes
        a.dropped += 1;
        a.ram_freed += e.bytes;
    };
    s.collectExpired(); // no request is in flight, so nothing borrows them
    const d = r orelse return;
    const held = d.held();
    var i = d.states.items.len;
    while (i > 0) {
        i -= 1;
        const st = d.states.items[i];
        if (!n.related(st.tokens)) continue;
        if (stays(s, n, st.tokens, st.shared)) {
            a.shared += 1;
            continue;
        }
        if (d.fetch) |f| f.forget(st.name); // its read ahead goes with it
        d.remove(i); // the index's own name in the store's folder, and the blocks no other state names
        a.disk_dropped += 1;
    }
    a.freed = held -| d.held();
}

/// Whether a state of the conversation stays: past the named prompt only a shared cut; below it, one another uses.
fn stays(s: *const pc.Store, n: Named, t: []const u32, flagged: bool) bool {
    if (flagged) return true;
    if (!n.below(t)) return false;
    for (s.entries.items) |e| if (!e.expired and prefixOf(t, e.tokens) and others(n, e)) return true;
    const d = s.disk orelse return false;
    for (d.states.items) |st| if (prefixOf(t, st.tokens) and (st.shared or !n.related(st.tokens))) return true;
    return false;
}

/// A RAM state other conversations use: theirs, a shared cut, or below the prompt a branch point (policy.own's rule).
fn others(n: Named, e: *const pc.Entry) bool {
    return e.shared or !n.related(e.tokens) or (n.below(e.tokens) and e.branched);
}

fn prefixOf(p: []const u32, t: []const u32) bool {
    return p.len <= t.len and std.mem.eql(u32, t[0..p.len], p);
}

/// The writes the I/O thread holds land and are indexed (false: some still go after `settle_ns`).
fn settle(d: *rows.Rows) bool {
    const until = rows.write.nowNs() + settle_ns;
    while (d.worker) |w| {
        rows.write.reap(d);
        if (w.pending() == 0) return true;
        if (rows.write.nowNs() >= until) return false;
        rows.write.io.nap();
    }
    return true;
}

/// Where the conversation's next request would now resume, keyed as the host's lookups key it.
fn resumes(a: *ask.Ask, s: *pc.Store, keys: ask.Keys, frame: u32) void {
    const starts: []const u32 = if (keys.starts) a.starts else &.{};
    const spans: []const [2]u32 = if (keys.spans) a.spans else &.{};
    const own: ?[]u32 = if (frame != 0 and a.prompt.len > 0) s.gpa.dupe(u32, a.prompt) catch null else null;
    defer if (own) |k| s.gpa.free(k);
    if (own) |k| k[0] = pc.framed(k[0], frame);
    s.image_query = a.images;
    s.image_frame = frame;
    defer {
        s.image_query = null;
        s.image_frame = 0;
    }
    a.out = rows.read.resumable(s, own orelse a.prompt, starts, spans);
}
