//! Cone prefill (TF_DSV41_CONE): every prompt row through the encoder layers, the cone's last rows through the decoder.
const std = @import("std");
const mtl = @import("metal");
const ced = @import("ced.zig");
const ft = @import("forward_types.zig");
const model = @import("model.zig");
const ops = @import("ops.zig");
const view = @import("view.zig");
const impl = @import("forward_impl.zig");
const Model = model.Model;
const Seg = model.Seg;
const Logits = model.Logits;
const View = view.View;

/// Stash ring rows (position p at row p % cap); at least the cone's rows.
pub const cap: u32 = 4096;
/// A chunk shorter than this is not worth a cone (decode rows, tails).
const min_rows: u32 = 17;
/// The kept rows: the window ring's last rows.
const kept: u32 = ft.ring;
const max_chunks = 64;

pub const Phase = enum { off, encoder, decoder };

/// Bytes a row occupies in each stash section: stream, branch, pre, post, comb.
const sect = [5]usize{ 20480 * 2, 5120 * 2, 4 * 4, 4 * 4, 16 * 4 };

const Chunk = struct { pos: u32, rows: u32 };

/// Other prompts' runs that wait behind the active one (a prompt's chunks may alternate with another prompt's).
pub const parked_max = 3;

/// A prompt's run set aside while another prompt's chunks use the stash: the active run's fields, same names.
const Parked = struct {
    used: bool = false,
    slot: u32 = 0,
    stash: ?mtl.Buffer = null,
    ids: [cap]u32 = undefined,
    chunks: [max_chunks]Chunk = undefined,
    count: usize = 0,
    first: u32 = 0,
    next: u32 = 0,
};

pub const Cone = struct {
    on: ?bool = null,
    phase: Phase = .off,
    /// the decoder's first layer, and the layers [lo, hi) of the replay program in flight
    d: u32 = 20,
    lo: u32 = 20,
    hi: u32 = 40,
    /// the replay runs in programs of a few layers, each over fewer rows
    grouped: bool = false,
    /// the rows the decoder replays at most
    rows: u32 = 0,
    stash: ?mtl.Buffer = null,
    ids: [cap]u32 = undefined,
    chunks: [max_chunks]Chunk = undefined,
    count: usize = 0,
    pending: bool = false,
    slot: u32 = 0,
    /// the pending run's first position and the next one it continues at
    first: u32 = 0,
    next: u32 = 0,
    /// the decoder chunk's first position (loadStash)
    at: u32 = 0,
    /// other prompts' runs, set aside while the active one's chunks run; `park` places may be used (TF_DSV41_CONE_PARK)
    parked: [parked_max]Parked = @splat(.{}),
    park: u32 = parked_max,

    fn sectionOffset(i: usize) usize {
        var off: usize = 0;
        for (sect[0..i]) |b| off += b * cap;
        return off;
    }

    fn totalBytes() usize {
        return sectionOffset(5);
    }
};

/// Whether the cone serves this model's prompts.
pub fn active(m: *Model) bool {
    return enabled(m);
}

fn enabled(m: *Model) bool {
    if (m.cone.on) |v| return v;
    return configure(m, @import("sys").environment.read(u32, "TF_DSV41_CONE", 2) catch 2);
}

/// Set the cone mode (0 off, 1 one replay program, 2 layer groups); on only where the cone rule covers the config.
fn configure(m: *Model, mode: u32) bool {
    const c = &m.cone;
    c.on = false;
    c.grouped = mode >= 2;
    c.park = @min(parked_max, @import("sys").environment.read(u32, "TF_DSV41_CONE_PARK", parked_max) catch parked_max);
    if (mode == 0) return false;
    const d = ced.decoderStart(&m.cfg) catch return false;
    const rows = ced.suffixRows(&m.cfg, kept) catch return false;
    if (rows > cap) return false;
    // the decoder layers read the first one's pool: no other kv source among them
    for (d + 1..m.cfg.layers) |l| if (m.cfg.isKvSource(@intCast(l))) return false;
    c.d = d;
    c.rows = rows;
    c.lo = d;
    c.hi = m.cfg.layers;
    c.on = true;
    return true;
}

/// The check harness's switch between the modes (no run pending).
pub fn force(m: *Model, mode: u32) void {
    _ = configure(m, mode);
}

fn eligible(m: *Model, segs: []const Seg, logits: Logits) bool {
    return segs.len == 1 and logits != .all and m.profile == null and m.observer == null and
        m.attention_observer == null and m.pipelines.hc_boundary != null and m.publish_limit == 0 and
        std.c.getenv("TF_DSV41_LAYER_CB") == null and std.c.getenv("TF_DSV41_HID_HASH") == null and
        std.c.getenv("TF_DSV41_STAGE_CB") == null and std.c.getenv("TF_DSV41_ATTN_PROF") == null;
}

/// The cone path of one forward, or null when the call takes the full path.
pub fn route(m: *Model, segs: []const Seg, logits: Logits) !?[]const f32 {
    if (!enabled(m)) return null;
    const c = &m.cone;
    if (segs.len == 0 or segs[0].slot >= m.lanes) return null;
    const sg = segs[0];
    if (segs.len != 1) {
        // a shared round over a slot with a run: its state is replaced, nothing to replay
        for (segs) |s| drop(m, s.slot);
        return null;
    }
    const ok = eligible(m, segs, logits);
    const rows: u32 = @intCast(sg.ids.len);
    const final = logits != .none;
    const starts = ok and rows >= min_rows and !(final and rows <= c.rows);
    const has = hasRun(c, sg.slot);
    // this prompt's own run becomes the active one, the active run of another prompt parking behind it
    if (has or (c.pending and starts)) {
        if (!activate(c, sg.slot)) return null; // every parking place holds a prompt: this call takes the full path
    }
    if (c.pending and c.slot == sg.slot) {
        // the slot's state was replaced under the pending run: nothing to replay
        if (sg.p0 != c.next) {
            c.pending = false;
            return null;
        }
        if (!ok or c.count == max_chunks) _ = try replay(m, .none);
    }
    if (!ok) return null;
    if (!c.pending) {
        if (!starts) return null;
        c.pending = true;
        c.slot = sg.slot;
        c.first = sg.p0;
        c.next = sg.p0;
        c.count = 0;
    } else if (c.slot != sg.slot) return null;
    try ensureStash(m);
    c.phase = .encoder;
    defer c.phase = .off;
    const out = try impl.forwardCore(m, segs, .none);
    for (sg.ids, 0..) |id, i| c.ids[(sg.p0 + i) % cap] = id;
    c.chunks[c.count] = .{ .pos = sg.p0, .rows = rows };
    c.count += 1;
    c.next = sg.p0 + rows;
    if (!final) return out;
    return try replay(m, logits);
}

/// The decoder layers over the stashed rows, in programs of consecutive layers, ending the pending run.
pub fn replay(m: *Model, logits: Logits) ![]const f32 {
    const c = &m.cone;
    const end = c.next;
    const state = &m.caches[c.slot];
    c.pending = false;
    c.phase = .decoder;
    defer c.phase = .off;
    defer c.count = 0;
    var out: []const f32 = m.workspace.logits(0);
    var lo = c.d;
    while (lo < m.cfg.layers) {
        c.lo = lo;
        c.hi = programEnd(m, lo, end);
        defer lo = c.hi;
        const start = replayStart(end, c.first, rowsIn(m, lo));
        const final = c.hi == m.cfg.layers;
        for (c.chunks[0..c.count], 0..) |ch, i| {
            const a = @max(ch.pos, start);
            const b = ch.pos + ch.rows;
            const row0 = m.taps_row;
            // the drafter's tap ring advances as the full pass advances it, over rows the replay skips too
            if (final and model.tapsOn()) {
                const extra = @min(ch.rows, 128) - (if (a < b) @min(b - a, 128) else 0);
                m.taps_row = (m.taps_row + extra) % 128;
                m.taps_n = @min(m.taps_n + extra, 128);
            }
            if (a >= b) continue;
            state.position = a;
            c.at = a;
            var ids: [cap]u32 = undefined;
            for (0..b - a) |k| ids[k] = c.ids[(a + k) % cap];
            const last = final and i + 1 == c.count;
            out = try impl.forwardCore(m, &.{.{ .slot = c.slot, .ids = ids[0 .. b - a], .p0 = a }}, if (last) logits else .none);
            // the ring's base is the full pass's chunk start, not the replay's
            if (final and model.tapsOn() and model.tapsSharedBaseOn()) m.taps_base = if (model.condAlignOn()) ch.pos % 128 else row0;
        }
    }
    return out;
}

/// Where the decoder program that starts at layer `lo` ends: the next index source below the drafter's tap layers.
fn programEnd(m: *const Model, lo: u32, end: u32) u32 {
    var hi = m.cfg.layers;
    // the candidate block filter (long contexts) carries layer d's picks across layers: one program
    if (!m.cone.grouped or end >= 131072 or std.c.getenv("TF_DSV41_BLOCK_FILTER") != null) return hi;
    for (m.cfg.index_sources.items()) |l| if (l > lo and l < hi and l + 4 <= m.cfg.layers) {
        hi = l;
    };
    return hi;
}

/// The rows layer `lo`'s program starts from: what the kept rows depend on through the layers from `lo` on.
fn rowsIn(m: *const Model, lo: u32) u32 {
    return kept + (m.cfg.layers - lo) * (m.cfg.window - 1);
}

/// Whether `slot` has a run: the active one's or a parked one.
fn hasRun(c: *const Cone, slot: u32) bool {
    if (c.pending and c.slot == slot) return true;
    for (c.parked) |p| if (p.used and p.slot == slot) return true;
    return false;
}

/// The active run and a parked one change places.
fn swapRun(c: *Cone, p: *Parked) void {
    std.mem.swap(?mtl.Buffer, &c.stash, &p.stash);
    std.mem.swap([cap]u32, &c.ids, &p.ids);
    std.mem.swap([max_chunks]Chunk, &c.chunks, &p.chunks);
    std.mem.swap(usize, &c.count, &p.count);
    std.mem.swap(u32, &c.slot, &p.slot);
    std.mem.swap(u32, &c.first, &p.first);
    std.mem.swap(u32, &c.next, &p.next);
}

/// Makes `slot`'s run the active one (or none: a new run may start); another prompt's run parks; false: no place.
fn activate(c: *Cone, slot: u32) bool {
    if (c.pending and c.slot == slot) return true;
    var mine: ?*Parked = null;
    var free: ?*Parked = null;
    for (&c.parked, 0..) |*p, i| {
        if (p.used and p.slot == slot) mine = p else if (i < c.park and !p.used and free == null) free = p;
    }
    if (mine) |p| {
        // both runs pending: they trade places; an idle active run leaves its spare stash in the vacated place
        swapRun(c, p);
        if (!c.pending) {
            c.pending = true;
            p.used = false;
        }
        return true;
    }
    if (!c.pending) return true;
    const p = free orelse return false;
    swapRun(c, p);
    p.used = true;
    c.pending = false;
    return true;
}

/// Replay a pending run on `slot` so its state is exact at the position it stands at.
pub fn flush(m: *Model, slot: u32) !void {
    const c = &m.cone;
    if (!hasRun(c, slot)) return;
    _ = activate(c, slot); // its own run is active or parked: no parking place is needed
    _ = try replay(m, .none);
}

/// Forget a pending run on `slot` whose state is being replaced.
pub fn drop(m: *Model, slot: u32) void {
    dropRuns(&m.cone, slot);
}

fn dropRuns(c: *Cone, slot: u32) void {
    if (c.pending and c.slot == slot) c.pending = false;
    for (&c.parked) |*p| if (p.used and p.slot == slot) {
        p.used = false;
    };
}

pub fn release(m: *Model) void {
    const c = &m.cone;
    if (c.stash) |b| b.deinit();
    c.stash = null;
    for (&c.parked) |*p| {
        if (p.stash) |b| b.deinit();
        p.stash = null;
        p.used = false;
    }
}

fn ensureStash(m: *Model) !void {
    const c = &m.cone;
    if (c.stash != null) return;
    const buf = try m.device.buffer(Cone.totalBytes(), mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
    errdefer buf.deinit();
    m.residency.add(buf);
    m.residency.commit();
    m.residency.requestResidency();
    c.stash = buf;
}

fn sources(m: *const Model) [5]View {
    const ws = &m.workspace;
    return .{ m.streamView(), ws.view(.branch), ws.view(.previous_pre), ws.view(.post), ws.view(.comb) };
}

/// Copy the state entering the decoder (the chunk's last rows, at most the cone's) into the stash ring.
pub fn stashRows(m: *Model, enc: mtl.ComputeEncoder, rows: u32, p0: u32) void {
    const c = &m.cone;
    const take = @min(rows, c.rows);
    const q0 = p0 + rows - take;
    const split = ringRuns(q0, take);
    const src = sources(m);
    for (src, 0..) |s, i| {
        const row = sect[i];
        const base = Cone.sectionOffset(i);
        const dst0: View = .{ .buffer = c.stash.?, .offset = base + @as(usize, q0 % cap) * row };
        ops.encodeDump(enc, &m.pipelines.ops, s.shifted(@as(usize, rows - take) * row), dst0, @intCast(split[0] * row / 4), false);
        if (split[1] != 0) {
            const dst1: View = .{ .buffer = c.stash.?, .offset = base };
            ops.encodeDump(enc, &m.pipelines.ops, s.shifted(@as(usize, rows - take + split[0]) * row), dst1, @intCast(split[1] * row / 4), false);
        }
    }
    enc.barrier();
}

/// Load the stashed rows [at, at + rows) as the pending state entering the decoder layers.
pub fn loadStash(m: *Model, enc: mtl.ComputeEncoder, rows: u32) void {
    const c = &m.cone;
    const ws = &m.workspace;
    const dst = [5]View{ ws.view(.stream), ws.view(.branch), ws.view(.previous_pre), ws.view(.post), ws.view(.comb) };
    const split = ringRuns(c.at, rows);
    for (dst, 0..) |d, i| {
        const row = sect[i];
        const base = Cone.sectionOffset(i);
        const src0: View = .{ .buffer = c.stash.?, .offset = base + @as(usize, c.at % cap) * row };
        ops.encodeDump(enc, &m.pipelines.ops, src0, d, @intCast(split[0] * row / 4), false);
        if (split[1] != 0) {
            const src1: View = .{ .buffer = c.stash.?, .offset = base };
            ops.encodeDump(enc, &m.pipelines.ops, src1, d.shifted(@as(usize, split[0]) * row), @intCast(split[1] * row / 4), false);
        }
    }
    enc.barrier();
    m.stream_alt = false;
    m.pending_expand = true;
}

/// `n` stash rows from position `at` as up to two ring runs: to the ring's end, then from its start.
fn ringRuns(at: u32, n: u32) [2]u32 {
    const first = @min(n, cap - at % cap);
    return .{ first, n - first };
}

/// Where the decoder replay of a run that ends at `end` begins: the cone's rows back, but not before the run's start.
fn replayStart(end: u32, first: u32, rows: u32) u32 {
    return @max(end -| rows, first);
}

test "stash rows wrap at the ring end in two runs" {
    try std.testing.expectEqual([2]u32{ 100, 0 }, ringRuns(0, 100));
    try std.testing.expectEqual([2]u32{ 100, 0 }, ringRuns(cap - 100, 100));
    try std.testing.expectEqual([2]u32{ 100, 2568 }, ringRuns(cap - 100, 2668));
    try std.testing.expectEqual([2]u32{ 100, 2568 }, ringRuns(3 * cap - 100, 2668));
}

test "the replay starts a cone back, or at the run's first position" {
    try std.testing.expectEqual(@as(u32, 5332), replayStart(8000, 0, 2668));
    try std.testing.expectEqual(@as(u32, 4096), replayStart(8000, 4096, 4000));
    try std.testing.expectEqual(@as(u32, 0), replayStart(2000, 0, 2668));
    try std.testing.expectEqual(@as(u32, 1000), replayStart(2000, 1000, 2668));
}

test "the stash holds every section's rows for the whole ring" {
    var row: usize = 0;
    for (sect) |b| row += b;
    try std.testing.expectEqual(@as(usize, 51296), row);
    try std.testing.expectEqual(row * cap, Cone.totalBytes());
}

test "a prompt's run parks behind another's and comes back with its own chunks" {
    var c: Cone = .{};
    // prompt 1 starts a run and takes two chunks
    try std.testing.expect(activate(&c, 1));
    c.pending = true;
    c.slot = 1;
    c.count = 2;
    c.next = 8192;
    try std.testing.expect(hasRun(&c, 1));
    // prompt 2's chunk parks prompt 1's run
    try std.testing.expect(activate(&c, 2));
    try std.testing.expect(!c.pending);
    try std.testing.expect(hasRun(&c, 1) and !hasRun(&c, 2));
    c.pending = true;
    c.slot = 2;
    c.count = 1;
    c.next = 4096;
    // prompt 1 returns with its chunks; prompt 2 now waits parked
    try std.testing.expect(activate(&c, 1));
    try std.testing.expect(c.pending and c.slot == 1 and c.count == 2 and c.next == 8192);
    try std.testing.expect(hasRun(&c, 2));
    dropRuns(&c, 2);
    try std.testing.expect(!hasRun(&c, 2));
    // the parking places are bounded: a fourth prompt finds none
    var other: u32 = 10;
    while (other < 10 + parked_max) : (other += 1) {
        try std.testing.expect(activate(&c, other));
        c.pending = true;
        c.slot = other;
    }
    try std.testing.expect(!activate(&c, 99));
}
