//! --learn: a shared cut's kept state written to disk above the free-disk floor and within the cap, on every rank.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const imprint = @import("prompt_imprint.zig");
const Store = pc.Store;
const Entry = pc.Entry;
const note = pc.note;

/// A shared cut's state written to disk once (--learn): later sessions read it back, after a restart too.
pub fn learn(s: *Store, e: *const Entry, starts: []const u32) void {
    const im = s.imprint orelse return;
    if (e.decode_spans.len != 0) return; // learned states are prompt arithmetic only: their key holds no span map
    const write = s.family.vtable.write orelse return;
    const key = imprint.Imprint.keyOf(e.tokens);
    if (im.admission.waiting(im.dir)) return;
    finishPending(s, im) catch {
        im.admission.refuse(im.dir);
        return;
    };
    if (im.has(key) or e.bytes > im.cap) return;
    const payload_bytes = std.math.add(u64, e.bytes, 256 + @as(u64, e.tokens.len + starts.len) * 4) catch return;
    const disk_bytes = std.math.add(u64, payload_bytes, im.indexScratchBytes() catch return) catch return;
    if (im.admission.waiting(im.dir)) return;
    const local_need = im.admission.shortfall(im.dir, disk_bytes) orelse {
        im.admission.refuse(im.dir);
        return;
    };
    const extra = disk_bytes - e.bytes;
    const peer_need = if (s.family.vtable.peer_need) |need| need(s.family.ptr, e.saved, extra) catch {
        im.admission.refuse(im.dir);
        return;
    } else 0;
    if (peer_need == std.math.maxInt(u64)) {
        im.admission.refuse(im.dir);
        return;
    }
    const dirs = @import("lanes").learned_dirs;
    const others = otherCandidates(s, im) catch {
        im.admission.refuse(im.dir);
        return;
    };
    defer s.gpa.free(others);
    const used = std.math.add(u64, @max(im.total(), dirs.bytes(im.dir) catch return), dirs.totalOthers(im.root, im.dir) catch return) catch return;
    const cap_need = (std.math.add(u64, used, payload_bytes) catch return) -| im.cap;
    const planner = @import("learned_plan.zig");
    const candidates = s.gpa.alloc(planner.Candidate, im.metas.items.len + others.len) catch return;
    defer s.gpa.free(candidates);
    for (im.metas.items, candidates[0..im.metas.items.len], 0..) |meta, *candidate, i| {
        const peer = if (s.family.vtable.peer_reclaim) |reclaim| reclaim(s.family.ptr, meta.key) catch {
            im.admission.refuse(im.dir);
            return;
        } else 0;
        const local = if (s.family.vtable.reclaim) |f| f(s.family.ptr, im.dir, meta.key) catch return else 0;
        candidate.* = .{ .key = i, .local = local, .peer = peer, .cap = meta.bytes, .used = meta.used +| (1 << 63) };
    }
    for (others, candidates[im.metas.items.len..], im.metas.items.len..) |other, *candidate, i| {
        candidate.* = .{ .key = i, .local = other.local, .peer = other.peer, .cap = other.local, .used = other.used };
    }
    const victims = (planner.choose(s.gpa, candidates, .{ .local = local_need, .peer = peer_need, .cap = cap_need }) catch return) orelse {
        im.admission.refuse(im.dir);
        return learningPaused(im);
    };
    defer s.gpa.free(victims);
    const keys = s.gpa.alloc(u64, victims.len) catch return;
    defer s.gpa.free(keys);
    const current_count = im.metas.items.len;
    for (victims, keys) |victim, *k| k.* = if (victim < current_count) im.metas.items[victim].key else others[victim - current_count].id;
    for (victims, keys) |victim, victim_key| {
        if (victim < current_count) {
            unlearnChecked(s, im, victim_key) catch {
                im.admission.refuse(im.dir);
                return;
            };
        } else {
            if (s.family.vtable.peer_other_remove) |remove_other| remove_other(s.family.ptr, victim_key) catch {
                im.admission.refuse(im.dir);
                return;
            };
            dirs.removeOther(im.root, im.dir, victim_key) catch {
                im.admission.refuse(im.dir);
                return;
            };
        }
    }
    im.others = dirs.totalOthers(im.root, im.dir) catch return;
    switch (im.admission.reserve(im.dir, disk_bytes)) {
        .quiet => return,
        .refused => return learningPaused(im),
        .ready => {},
    }
    var success = false;
    defer im.admission.finish(im.dir, disk_bytes, success);
    defer if (s.family.vtable.peer_finish) |finish| finish(s.family.ptr, key, success) catch |err| note("finishing peer disk admission failed ({s})", .{@errorName(err)});
    if (s.family.vtable.peer_reserve) |peer_reservation| peer_reservation(s.family.ptr, e.saved, key, extra) catch return;
    im.beginWrite(key) catch return;
    write(s.family.ptr, e.saved, im.dir, key) catch |err| {
        finishPending(s, im) catch |cleanup_err| note("learned cleanup remains pending ({s})", .{@errorName(cleanup_err)});
        return note("learning {d} tokens failed ({s}); retry delayed", .{ e.at, @errorName(err) });
    };
    im.add(key, e.at, e.tokens, starts, e.bytes) catch |err| {
        finishPending(s, im) catch |cleanup_err| note("learned cleanup remains pending ({s})", .{@errorName(cleanup_err)});
        return note("indexing {d} learned tokens failed ({s}); retry delayed", .{ e.at, @errorName(err) });
    };
    im.clearPending() catch return;
    success = true;
    if (!@import("builtin").is_test) std.log.info("prompt cache: learned {d} tokens to disk", .{e.at});
}

/// Learned state `key` forgotten: its files (the family's), then its index record.
fn unlearnChecked(s: *Store, im: *imprint.Imprint, key: u64) !void {
    if (s.family.vtable.forget_checked) |f| try f(s.family.ptr, im.dir, key) else if (s.family.vtable.forget) |f| f(s.family.ptr, im.dir, key);
    try im.remove(key);
}
pub fn unlearn(s: *Store, im: *imprint.Imprint, key: u64) void {
    unlearnChecked(s, im, key) catch |err| note("forgetting a learned state failed ({s})", .{@errorName(err)});
}
fn finishPending(s: *Store, im: *imprint.Imprint) !void {
    const key = (try im.pendingWrite()) orelse return;
    if (!im.has(key)) {
        const drop_pending = s.family.vtable.forget_checked orelse return error.UncheckedLearnedCleanup;
        try drop_pending(s.family.ptr, im.dir, key);
    }
    try im.clearPending();
}
const Other = struct { id: u64, local: u64, peer: u64, used: u64 };
fn otherCandidates(s: *Store, im: *imprint.Imprint) ![]Other {
    const dirs = @import("lanes").learned_dirs;
    var out: std.ArrayList(Other) = .empty;
    errdefer out.deinit(s.gpa);
    var cursor: ?u64 = null;
    while (true) {
        const local = try dirs.next(im.root, im.dir, cursor);
        const peer = if (s.family.vtable.peer_other_next) |f| try f(s.family.ptr, cursor) else null;
        const id = if (local) |l| if (peer) |p| @min(l, p) else l else peer orelse break;
        const own_bytes = try dirs.otherBytes(im.root, im.dir, id);
        const peer_bytes = if (s.family.vtable.peer_other_bytes) |f| try f(s.family.ptr, id) else 0;
        const peer_used = if (s.family.vtable.peer_other_used) |f| try f(s.family.ptr, id) else 0;
        try out.append(s.gpa, .{ .id = id, .local = own_bytes, .peer = peer_bytes, .used = @max(try dirs.used(im.root, id), peer_used) });
        cursor = id;
    }
    return out.toOwnedSlice(s.gpa);
}

fn learningPaused(im: *const imprint.Imprint) void {
    note("learning paused: free disk cannot preserve the {d} MiB floor", .{im.admission.floor >> 20});
}
