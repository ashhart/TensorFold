//! Disk states read back: each block and the state's own file checked, then copied into a new kept state of its family.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const image = @import("rows_image.zig");
const chain = @import("rows_chain.zig");
const file = @import("rows_file.zig");
const rows_store = @import("rows_store.zig");
const write = @import("rows_write.zig");
const select = @import("rows_select.zig");
const Rows = rows_store.Rows;
const Name = chain.Name;

/// Disk state `i` read into a new kept state of its family; a file that fails a check goes, with every state naming it.
pub fn read(r: *Rows, i: usize) !pc.Saved {
    const f = r.family.vtable;
    const s = r.states.items[i];
    const t0 = write.nowNs();
    const parts = try r.parts(r.gpa, s.at);
    defer r.gpa.free(parts);
    var wants: std.ArrayList(write.Want) = .empty;
    defer wants.deinit(r.gpa);
    try r.blockNames(r.gpa, parts, s.tokens, s.spans, s.starts, s.at, &wants);
    const saved = try r.img.blank(r.family.ptr, s.at);
    errdefer f.drop(r.family.ptr, saved);
    var ranges: std.ArrayList(image.Range) = .empty;
    defer ranges.deinit(r.gpa);
    var bytes: u64 = 0;
    for (wants.items) |w| {
        ranges.clearRetainingCapacity();
        try image.blockRanges(parts, r.keying.block, w.window, w.k, &ranges, r.gpa);
        bytes += try one(r, w.name, .block, ".blk", 0, ranges.items, saved);
    }
    ranges.clearRetainingCapacity();
    try image.ownRanges(parts, s.at, r.keying.block, &ranges, r.gpa);
    bytes += try one(r, s.name, .state, ".st", s.at, ranges.items, saved);
    const ns = write.nowNs() - t0;
    if (ns > 0 and bytes > 64 << 20) r.read_rate = 0.5 * r.read_rate + 0.5 * @as(f64, @floatFromInt(bytes)) / (@as(f64, @floatFromInt(ns)) / 1e9);
    r.counts.read += bytes;
    r.counts.reads += 1;
    r.touch(i);
    if (!@import("builtin").is_test) std.log.info("imprint: read {d} tokens from disk ({d} MiB in {d:.0} ms)", .{ s.at, bytes >> 20, @as(f64, @floatFromInt(ns)) / 1e6 });
    return saved;
}

/// One file's payload read past the page cache, checked, and copied into `ranges` of `saved`; its bytes on disk.
fn one(r: *Rows, name: Name, kind: file.Kind, ext: []const u8, at: u32, ranges: []const image.Range, saved: pc.Saved) !u64 {
    const n = image.total(ranges);
    var buf: [1200]u8 = undefined;
    const p = try file.path(&buf, r.dir, &chain.hex(name), ext, false);
    var o = file.open(p, kind, name) catch |err| return bad(r, name, ext, err);
    defer o.close();
    if (o.trailer.payload != n or o.trailer.block != r.keying.block or o.trailer.at != at) return bad(r, name, ext, error.RowsRead);
    const stage = try r.staged(n);
    const got = file.payload(&o, stage) catch |err| return bad(r, name, ext, err);
    try r.img.scatter(r.family.ptr, saved, ranges, got);
    return o.trailer.payload + o.trailer.meta + @sizeOf(file.Trailer);
}

/// A file that failed its checks deleted, with every state that names it; the error passed on.
fn bad(r: *Rows, name: Name, ext: []const u8, err: anyerror) anyerror {
    r.counts.failed += 1;
    pc.note("imprint: a disk {s} failed its check ({s}); it and the states naming it are forgotten", .{ if (std.mem.eql(u8, ext, ".blk")) "block" else "state", @errorName(err) });
    rows_store.unlinkName(r.dir, name, ext);
    if (std.mem.eql(u8, ext, ".blk")) _ = r.blocks.remove(chain.stem(name));
    var i: usize = 0;
    var wants: std.ArrayList(write.Want) = .empty;
    defer wants.deinit(r.gpa);
    while (i < r.states.items.len) {
        const s = &r.states.items[i];
        var names = std.mem.eql(u8, &s.name, &name);
        if (!names) {
            wants.clearRetainingCapacity();
            r.wantsOf(r.gpa, s, &wants) catch {};
            for (wants.items) |w| names = names or std.mem.eql(u8, &w.name, &name);
        }
        if (names) r.remove(i) else i += 1;
    }
    return err;
}

/// A disk state reaching further than `have`, read back into a new RAM entry where that costs less than prefilling.
pub fn recall(s: *pc.Store, prompt: []const u32, born: u32, starts: []const u32, spans: []const modes.Span, have: ?*pc.Entry) ?*pc.Entry {
    const r = s.disk orelse return have;
    const i = select.choose(s, prompt, starts, spans, have).restore orelse return have;
    const st = r.states.items[i];
    const bytes = s.family.vtable.bytes(s.family.ptr, st.at);
    if (bytes > s.budget or s.most == 0) return have;
    while (!ready(s, bytes)) {
        if (s.family.vtable.trim) |trim| {
            const spare = s.spare();
            trim(s.family.ptr, s.budget -| s.held -| bytes);
            if (s.spare() < spare) continue;
        }
        s.remove(s.victimBut(have, prompt, born) orelse return have);
        s.counts.evicted += 1;
    }
    const e = s.gpa.create(pc.Entry) catch return have;
    e.* = .{ .tokens = &.{}, .at = st.at, .saved = undefined, .bytes = bytes, .born = born, .used = s.clock, .last = &.{}, .shared = st.shared };
    var ok = false;
    defer if (!ok) {
        s.gpa.free(e.tokens);
        s.gpa.free(e.decode_spans);
        s.gpa.free(e.last);
        s.gpa.destroy(e);
    };
    e.tokens = s.gpa.dupe(u32, st.tokens) catch return have;
    e.decode_spans = s.gpa.dupe(modes.Span, st.spans) catch return have;
    e.last = s.gpa.dupe(u32, prompt) catch return have;
    e.saved = read(r, i) catch return have;
    if (s.family.vtable.charged) |f| e.bytes = f(s.family.ptr, e.saved);
    s.entries.append(s.gpa, e) catch {
        s.family.vtable.drop(s.family.ptr, e.saved);
        return have;
    };
    ok = true;
    s.held += e.bytes;
    return e;
}

fn ready(s: *const pc.Store, bytes: u64) bool {
    if (s.held > s.budget or s.entries.items.len >= s.most) return false;
    const room = s.budget - s.held;
    const spare = s.spare();
    return spare <= room and bytes <= room - spare;
}

/// Whether reading disk state `i` costs less than prefilling the tokens it reaches past `from` (measured rates).
pub fn worth(r: *const Rows, i: usize, from: u32) bool {
    return if (select.savedSeconds(r, i, from)) |seconds| seconds > 0 else true;
}

/// How much of a prompt the server resumes, and from where, with the estimated wait for the rest.
pub const Resume = struct {
    tokens: u32 = 0, // the prompt's tokens
    ram: u32 = 0, // tokens a kept state in RAM covers
    disk: u32 = 0, // tokens a state on disk covers
    disk_bytes: u64 = 0,
    from: enum { ram, disk, none } = .none,
    seconds: f64 = 0, // the estimated wait: the disk read, then prefilling what is left
};

/// Where `prompt`'s next request would resume, read on the engine thread.
pub fn resumable(s: *pc.Store, prompt: []const u32, starts: []const u32, spans: []const modes.Span) Resume {
    var out: Resume = .{ .tokens = @intCast(prompt.len) };
    const have = s.find(prompt, starts, spans);
    if (have) |e| out.ram = e.at;
    out.from = if (out.ram > 0) .ram else .none;
    const r = s.disk orelse return finishResume(s, out);
    const chosen = select.choose(s, prompt, starts, spans, have);
    if (chosen.disk) |i| {
        const st = r.states.items[i];
        out.disk = st.at;
        out.disk_bytes = st.own +| st.rows;
    }
    if (chosen.restore != null) {
        out.from = .disk;
        out.seconds = chosen.read_seconds;
    }
    return finishResume(s, out);
}

fn finishResume(s: *const pc.Store, x: Resume) Resume {
    var out = x;
    const start = if (out.from == .disk) out.disk else out.ram;
    const rate = if (s.disk) |r| r.prefill_rate else 0;
    if (select.knownRate(rate)) {
        const total = out.seconds + @as(f64, @floatFromInt(out.tokens -| start)) / rate;
        if (std.math.isFinite(total)) out.seconds = total;
    }
    return out;
}
