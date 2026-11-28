//! Reservation-time protection for queued and admitted prompt resumptions, scoped to the host's serial lookup.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

const reserve = @import("prompt_cache_reserve.zig");
const Snapshot = struct {
    entries: std.ArrayList(*const pc.Entry) = .empty,
    promised: reserve.Room = .{},
    replacement: ?u32 = null,

    fn init(h: *LaneHost, current: *Job, replacement: ?u32) !Snapshot {
        var out: Snapshot = .{ .replacement = replacement };
        errdefer out.entries.deinit(h.gpa);
        h.lock();
        defer h.unlock();
        for (h.queued.items) |job| try out.protect(h, current, job);
        for (h.admitted.items) |job| {
            try out.protect(h, current, job);
            if (job != current) {
                out.promised.bytes +|= job.cache_promise.bytes;
                out.promised.slots +|= job.cache_promise.slots;
            }
        }
        return out;
    }

    fn protect(out: *Snapshot, h: *LaneHost, current: *Job, job: *Job) !void {
        if (job.lent) |serial| if (job != current) { // the very state it resumes at its first call, however long another
            for (h.cache.?.entries.items) |e| if (e.serial == serial and !e.expired) {
                if (std.mem.indexOfScalar(*const pc.Entry, out.entries.items, e) == null) try out.entries.append(h.gpa, e);
                return;
            };
        };
        for (h.cache.?.by_position.items) |e| if (!e.expired and
            h.cache.?.usable(e.at, job.request.chunks) and extends(current, job, e))
        {
            if (out.replacement) |at| if (at > e.at and takesPrefix(h, current, job, at)) continue;
            if (std.mem.indexOfScalar(*const pc.Entry, out.entries.items, e) == null) try out.entries.append(h.gpa, e);
            break;
        };
    }

    fn needed(ptr: *anyopaque, e: *const pc.Entry) bool {
        const snapshot: *const Snapshot = @ptrCast(@alignCast(ptr));
        return std.mem.indexOfScalar(*const pc.Entry, snapshot.entries.items, e) != null;
    }

    fn guard(snapshot: *Snapshot) reserve.Guard {
        return .{ .ctx = snapshot, .needed = needed, .exact = true, .promised = snapshot.promised };
    }
};

pub fn lookup(h: *LaneHost, job: *Job) !pc.Lookup {
    const store = h.cache.?;
    var snapshot = try Snapshot.init(h, job, null);
    defer snapshot.entries.deinit(h.gpa);
    const before = store.reservation_guard;
    store.reservation_guard = snapshot.guard();
    store.reservation_guard.?.promise = &job.cache_promise;
    defer store.reservation_guard = before;
    errdefer job.cache_promise = .{};
    const r = job.request;
    const previous_salt = store.salt;
    store.salt = r.cacheSalt();
    defer store.salt = previous_salt;
    const previous_messages = store.message_starts;
    store.message_starts = r.message_starts;
    defer store.message_starts = previous_messages;
    return store.lookupUserFrame(h.gpa, job.cacheKey(), r.bornLen(), r.history_len, r.rewind_len, r.user_end, r.shared_prefixes, r.chunks, h.spans(r), r.images, job.frame);
}

/// Complete a promised keep without spending room another admitted pass has already claimed.
pub fn keep(h: *LaneHost, job: *Job, stream: *anyopaque, at: u32) void {
    keepAt(h, job, stream, at, false, false);
}

/// A cancelled pass releases its unused promise; a warm pass can also hand its prefix to a queued follow-up.
pub fn interrupted(h: *LaneHost, job: *Job, stream: *anyopaque, at: u32) void {
    var handoff = false;
    if (h.info_.warm_turns and job.request.background and job.request.max_tokens == 0) {
        h.lock();
        for (h.queued.items) |next| if (takesPrefix(h, job, next, at)) {
            handoff = true;
            break;
        };
        h.unlock();
    }
    job.cache_promise = .{};
    keepAt(h, job, stream, at, true, handoff);
}

fn takesPrefix(h: *const LaneHost, current: *const Job, next: *const Job, at: u32) bool {
    const store = h.cache orelse return false;
    const r = next.request;
    if (current.request.cacheSalt() != r.cacheSalt()) return false;
    const previous_salt = store.salt;
    store.salt = r.cacheSalt();
    defer store.salt = previous_salt;
    if (next.started or r.background or r.promptOnly() or r.prompt_logprobs != null or r.untaught or
        r.images.len != 0 or current.request.images.len != 0 or !store.usable(at, r.chunks)) return false;
    if (store.find(next.cacheKey(), r.chunks, h.spans(r))) |e| if (e.at >= at) return false;
    return store.wouldServe(next.cacheKey(), r.chunks, h.spans(r), current.cacheKey(), h.spans(current.request), at);
}

fn keepAt(h: *LaneHost, job: *Job, stream: *anyopaque, at: u32, cancelled: bool, handoff: bool) void {
    const store = h.cache orelse return;
    const previous_salt = store.salt;
    store.salt = job.request.cacheSalt();
    defer store.salt = previous_salt;
    var snapshot = Snapshot.init(h, job, if (handoff) at else null) catch {
        job.cache_promise = .{};
        store.counts.failed += 1;
        pc.note("keeping {d} tokens failed (OutOfMemory); queue protection could not be copied", .{at});
        return;
    };
    defer snapshot.entries.deinit(h.gpa);
    const before = store.reservation_guard;
    store.reservation_guard = snapshot.guard();
    defer store.reservation_guard = before;
    const keep_others = store.keep_others;
    store.keep_others = keep_others or handoff;
    defer store.keep_others = keep_others;
    const planned = cancelled or handoff or std.mem.indexOfScalar(u32, job.marks, at) != null;
    if (cancelled) {
        const bytes = store.family.vtable.bytes(store.family.ptr, at);
        const can = reserve.evictable(store, null, job.cacheKey());
        const held = store.held +| if (store.family.vtable.trim == null) store.spare() else 0;
        if (bytes > store.budget -| snapshot.promised.bytes or
            held -| can.bytes +| bytes > store.budget -| snapshot.promised.bytes or
            store.entries.items.len -| can.slots >= store.most -| snapshot.promised.slots) return;
    }
    if (!planned) {
        const bytes = store.family.vtable.bytes(store.family.ptr, at);
        const promised = snapshot.promised;
        // A fallback checkpoint may use free room only; it neither evicts states nor spends this job's promise.
        if (store.held +| (if (store.family.vtable.trim == null) store.spare() else 0) +| bytes +| job.cache_promise.bytes > store.budget -| promised.bytes or
            store.entries.items.len +| job.cache_promise.slots >= store.most -| promised.slots) return;
    }
    const held = store.held;
    const slots = store.entries.items.len;
    _ = store.keepBornFrame(job.cacheKey(), job.request.bornLen(), at, stream, job.request.chunks, h.spans(job.request), job.request.images, job.frame);
    if (!planned) return;
    job.cache_promise.bytes -|= store.held -| held;
    job.cache_promise.slots -|= store.entries.items.len -| slots;
    if (job.marks.len == 0 or at >= job.marks[job.marks.len - 1]) job.cache_promise = .{};
}

fn extends(current: *const Job, job: *const Job, e: *const pc.Entry) bool {
    const r = job.request;
    if (e.salt != r.cacheSalt()) return false;
    if (job == current or r.promptOnly() or r.prompt_logprobs != null or r.untaught) return false;
    if (job.started and job.entry == null and job.lent == null) return false; // a lent state stays protected
    const key = job.cacheKey();
    if (e.at >= key.len or e.tokens.len == 0 or e.tokens.len > key.len) return false;
    if (!@import("cache_images.zig").equal(e.images, r.images, @intCast(e.tokens.len))) return false;
    if (!@import("cache_modes.zig").equal(e.decode_spans, job.host.spans(r), e.at)) return false;
    // A queued image job has not prepared its frame key yet; compare the same first word without allocating it.
    const frame = if (r.images.len > 0) if (job.host.frame_key) |f| f(r.images) else 0 else 0;
    if (e.image_frame != frame) return false;
    if (frame != 0 and job.key.len == 0) {
        const first = 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(frame, std.mem.asBytes(&key[0])) & 0x7fff_ffff));
        return e.tokens[0] == first and std.mem.eql(u32, e.tokens[1..], key[1..e.tokens.len]);
    }
    return std.mem.eql(u32, e.tokens, key[0..e.tokens.len]);
}

test "queued cache: an image protects only its own prefix and exact frame before admission" {
    const api = @import("engine_api.zig");
    const image: @import("lanes").stream.Image = .{ .pixels = &.{}, .gh = 1, .gw = 1, .at = 2, .tokens = 1, .hash = 17 };
    const words = [_]u32{ 1, 2, 3, 4, 5, 6 };
    const request: api.Request = .{ .prompt = &words, .cache_key = &words, .max_tokens = 1, .images = &.{image} };
    var h: LaneHost = undefined;
    h.frame_key = null;
    h.decoded_rows = false;
    var current: Job = .{ .host = &h, .id = 1, .request = &request, .sink = undefined };
    var queued: Job = .{ .host = &h, .id = 2, .request = &request, .sink = undefined };
    var prefix = [_]u32{ 1, 2, 3 };
    var ids = [_]@import("cache_images.zig").Identity{.{ .at = 2, .tokens = 1, .gh = 1, .gw = 1, .hash = 17 }};
    var e: pc.Entry = .{ .tokens = &prefix, .images = &ids, .at = 3, .saved = undefined, .bytes = 1, .born = 3, .used = 1, .last = &prefix };
    try std.testing.expect(extends(&current, &queued, &e));
    prefix[1] = 99;
    try std.testing.expect(!extends(&current, &queued, &e));
    prefix[1] = 2;
    ids[0].hash += 1;
    try std.testing.expect(!extends(&current, &queued, &e));
    ids[0].hash = 17;
    const Frame = struct {
        fn of(_: []const @import("lanes").stream.Image) u32 {
            return 7;
        }
    };
    h.frame_key = Frame.of;
    e.image_frame = 7;
    prefix[0] = 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(7, std.mem.asBytes(&words[0])) & 0x7fff_ffff));
    try std.testing.expect(extends(&current, &queued, &e));
    prefix[1] = 99;
    try std.testing.expect(!extends(&current, &queued, &e));
}

test "queued cache: one snapshot protects each job's deepest usable prefix including a shorter fork" {
    const Fake = @import("prompt_cache_fake.zig").Fake;
    const a = std.testing.allocator;
    var f: Fake = .{ .gpa = a };
    var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    for ([_]u32{ 8, 12 }) |at| {
        _ = f.pass(&store, words[0..at], .{});
        try std.testing.expect(store.keepBorn(&words, at, at, null, &.{}, &.{}));
    }
    var h: LaneHost = undefined;
    h.cache = &store;
    h.frame_key = null;
    h.decoded_rows = false;
    h.gpa = a;
    const request: @import("engine_api.zig").Request = .{ .prompt = &words, .max_tokens = 1 };
    const short: @import("engine_api.zig").Request = .{ .prompt = words[0..10], .max_tokens = 1 };
    var current: Job = .{ .host = &h, .id = 1, .request = &request, .sink = undefined };
    var queued: Job = .{ .host = &h, .id = 2, .request = &request, .sink = undefined };
    var fork: Job = .{ .host = &h, .id = 3, .request = &short, .sink = undefined };
    var snapshot: Snapshot = .{};
    defer snapshot.entries.deinit(a);
    try snapshot.protect(&h, &current, &queued);
    try std.testing.expectEqual(@as(usize, 1), snapshot.entries.items.len);
    try std.testing.expectEqual(@as(u32, 12), snapshot.entries.items[0].at);
    try snapshot.protect(&h, &current, &fork);
    try std.testing.expectEqual(@as(usize, 2), snapshot.entries.items.len);
    try std.testing.expectEqual(@as(u32, 8), snapshot.entries.items[1].at);
}

test {
    _ = @import("lane_cache_promise_test.zig");
}

test "queued cache: a lent job's own state stays protected when a longer prefix is kept after it lent" {
    const Fake = @import("prompt_cache_fake.zig").Fake;
    const a = std.testing.allocator;
    var f: Fake = .{ .gpa = a };
    var store = pc.Store.init(a, f.snapshots(), .{ .min_prompt = 0 }, 1000);
    defer store.deinit();
    const words = [_]u32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
    _ = f.pass(&store, words[0..8], .{});
    try std.testing.expect(store.keepBorn(&words, 8, 8, null, &.{}, &.{}));
    const lent = store.entries.items[0];
    var h: LaneHost = undefined;
    h.cache = &store;
    h.frame_key = null;
    h.decoded_rows = false;
    h.gpa = a;
    const request: @import("engine_api.zig").Request = .{ .prompt = &words, .max_tokens = 1 };
    var current: Job = .{ .host = &h, .id = 1, .request = &request, .sink = undefined };
    var waiting: Job = .{ .host = &h, .id = 2, .request = &request, .sink = undefined, .started = true, .lent = lent.serial };
    // another pass keeps a longer prefix of the same prompt before the lent job's first call
    _ = f.pass(&store, words[0..12], .{});
    try std.testing.expect(store.keepBorn(&words, 12, 12, null, &.{}, &.{}));
    var snapshot: Snapshot = .{};
    defer snapshot.entries.deinit(a);
    try snapshot.protect(&h, &current, &waiting);
    try std.testing.expectEqual(@as(usize, 1), snapshot.entries.items.len);
    try std.testing.expectEqual(lent, @constCast(snapshot.entries.items[0])); // the state it resumes, not the longer one
}
