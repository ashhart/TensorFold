//! Reservation-time protection for queued and admitted prompt resumptions, scoped to the host's serial lookup.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;

pub fn lookup(h: *LaneHost, job: *Job) !pc.Lookup {
    const store = h.cache.?;
    const before = store.reservation_guard;
    store.reservation_guard = .{ .ctx = job, .needed = needed };
    defer store.reservation_guard = before;
    const r = job.request;
    return store.lookupUserFrame(h.gpa, job.cacheKey(), r.bornLen(), r.history_len, r.rewind_len, r.user_end, r.shared_prefixes, r.chunks, h.spans(r), r.images, job.frame);
}

fn needed(ptr: *anyopaque, e: *const pc.Entry) bool {
    const current: *Job = @ptrCast(@alignCast(ptr));
    const h = current.host;
    h.lock();
    defer h.unlock();
    for (h.queued.items) |job| if (extends(current, job, e)) return true;
    for (h.admitted.items) |job| if (extends(current, job, e)) return true;
    return false;
}

fn extends(current: *const Job, job: *const Job, e: *const pc.Entry) bool {
    const r = job.request;
    if (job == current or r.promptOnly() or r.prompt_logprobs != null or r.untaught) return false;
    if (job.started and job.entry == null) return false;
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
