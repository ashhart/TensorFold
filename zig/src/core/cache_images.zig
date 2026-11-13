//! Exact image identities beside cache words; one image row cannot hold a 62-bit identity in a u32.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const modes = @import("cache_modes.zig");
const Image = @import("lanes").stream.Image;
const Allocator = std.mem.Allocator;

pub const Identity = struct { at: u32, tokens: u32, gh: u32, gw: u32, hash: u64 };

fn identity(i: Image) Identity {
    return .{ .at = i.at, .tokens = i.tokens, .gh = i.gh, .gw = i.gw, .hash = i.hash };
}

/// Include the image when its first row enters the keyed prefix, including any required lookahead.
pub fn prefix(a: Allocator, images: []const Image, at: u32) ![]Identity {
    var n: usize = 0;
    for (images) |i| n += @intFromBool(i.at < at);
    const out = try a.alloc(Identity, n);
    n = 0;
    for (images) |i| if (i.at < at) {
        out[n] = identity(i);
        n += 1;
    };
    return out;
}

pub fn equal(kept: []const Identity, images: []const Image, at: u32) bool {
    var n: usize = 0;
    for (images) |i| if (i.at < at) {
        if (n == kept.len or !std.meta.eql(kept[n], identity(i))) return false;
        n += 1;
    };
    return n == kept.len;
}

pub fn matches(s: *const pc.Store, e: *const pc.Entry) bool {
    return e.image_frame == s.image_frame and equal(e.images, s.image_query orelse &.{}, @intCast(e.tokens.len));
}

/// Store operations run on one engine thread. A scoped query must not nest another image operation.
pub fn lookupUser(s: *pc.Store, a: Allocator, prompt: []const u32, born: u32, history: u32, rewind: u32, user: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, images: []const Image) !pc.Lookup {
    return lookupUserFrame(s, a, prompt, born, history, rewind, user, shared, starts, spans, images, 0);
}

pub fn lookupUserFrame(s: *pc.Store, a: Allocator, prompt: []const u32, born: u32, history: u32, rewind: u32, user: u32, shared: []const u32, starts: []const u32, spans: []const modes.Span, images: []const Image, frame: u32) !pc.Lookup {
    std.debug.assert(s.image_query == null);
    s.image_query = images;
    s.image_frame = frame;
    defer {
        s.image_query = null;
        s.image_frame = 0;
    }
    return s.lookupUser(a, prompt, born, history, rewind, user, shared, starts, spans);
}

pub fn keepBorn(s: *pc.Store, prompt: []const u32, born: u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span, images: []const Image) bool {
    return keepBornFrame(s, prompt, born, at, owner, starts, spans, images, 0);
}

pub fn keepBornFrame(s: *pc.Store, prompt: []const u32, born: u32, at: u32, owner: ?*anyopaque, starts: []const u32, spans: []const modes.Span, images: []const Image, frame: u32) bool {
    std.debug.assert(s.image_query == null);
    s.image_query = images;
    s.image_frame = frame;
    defer {
        s.image_query = null;
        s.image_frame = 0;
    }
    return s.keepBorn(prompt, born, at, owner, starts, spans);
}

/// An image request's first cache word on a framed family: its frame mixed in, so a state fits that frame alone.
pub fn framed(first: u32, frame: u32) u32 {
    return 0x8000_0000 | @as(u32, @truncate(std.hash.Wyhash.hash(frame, std.mem.asBytes(&first)) & 0x7fff_ffff));
}

/// Current disk formats have no image metadata. Their reserved image words cannot be read or published safely.
pub fn diskSafe(tokens: []const u32) bool {
    for (tokens) |word| if (word & 0x8000_0000 != 0) return false;
    return true;
}

test "image prefix identity includes geometry, keeps text prefixes and ignores future images" {
    const t = std.testing;
    const original: Image = .{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 4, .tokens = 1, .hash = 7 };
    const kept = try prefix(t.allocator, &.{original}, 5);
    defer t.allocator.free(kept);
    try t.expect(equal(kept, &.{original}, 5));
    try t.expect(equal(&.{}, &.{original}, 4));
    try t.expect(!equal(kept, &.{}, 5));
    var changed = original;
    changed.hash += 1;
    try t.expect(!equal(kept, &.{changed}, 5));
    changed = original;
    changed.tokens += 1;
    try t.expect(!equal(kept, &.{changed}, 5));
    changed = original;
    changed.gh += 1;
    try t.expect(!equal(kept, &.{changed}, 5));
    changed = original;
    changed.at += 2;
    try t.expect(equal(kept, &.{ original, changed }, 5));
    try t.expect(!diskSafe(&.{ 1, 0x8000_0000, 2 }));
    try t.expect(diskSafe(&.{ 1, 2, 3 }));
}

test "image collision lookup checks rewind, reservation and interior cuts with the exact image prefix" {
    const t = std.testing;
    const Fake = struct {
        var saved: u8 = 0;
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + @as(u64, at);
        }
        fn save(_: *anyopaque, _: ?*anyopaque, _: u32) anyerror!pc.Saved {
            return &saved;
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) anyerror!void {}
        fn drop(_: *anyopaque, _: pc.Saved) void {}
    };
    var s = pc.Store.init(t.allocator, .{ .ptr = &Fake.saved, .vtable = &.{ .bytes = Fake.bytes, .save = Fake.save, .restore = Fake.restore, .drop = Fake.drop } }, .{ .chunk = 128, .min_gap = 256 }, 1 << 20);
    defer s.deinit();
    var arena = std.heap.ArenaAllocator.init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const words = try a.alloc(u32, 12000);
    @memset(words, 11); // omitted row keys reproduce any compact-key collision
    const first: Image = .{ .pixels = &.{}, .gh = 2, .gw = 2, .at = 2000, .tokens = 1, .hash = 7 };
    var other = first;
    other.hash = 8;
    try t.expect(s.keepBornImages(words, 12000, 1024, null, &.{}, &.{}, &.{first}));
    try t.expect(s.keepBornImages(words, 12000, 8960, null, &.{}, &.{}, &.{first}));
    const l = try s.lookupUserImages(a, words, 12000, 11008, 1024, 11008, &.{}, &.{}, &.{}, &.{other});
    try t.expectEqual(@as(u32, 1024), l.entry.?.at);
    s.release(l.entry.?);
    try t.expectEqual(@as(u32, 1024), s.active_rewind_at);
    try t.expect(std.mem.indexOfScalar(u32, l.marks, 11008) != null);
    try t.expect(std.mem.indexOfScalar(u32, l.marks, 8960) != null); // independent interior cut
    const near = try s.lookupUserImages(a, words, 12000, 11008, 9088, 0, &.{}, &.{}, &.{}, &.{other});
    try t.expectEqual(@as(u32, 1024), near.entry.?.at);
    s.release(near.entry.?);
    try t.expectEqual(@as(u32, 9088), s.active_rewind_at); // a nearby colliding image state is not the rewind state
    try t.expect(s.keepBornImages(words, 12000, 8960, null, &.{}, &.{}, &.{other}));
    try t.expectEqual(@as(usize, 3), s.entries.items.len);
    const hit = try s.lookupUserImages(a, words, 12000, 11008, 1024, 11008, &.{}, &.{}, &.{}, &.{first});
    try t.expectEqual(@as(u32, 8960), hit.entry.?.at);
    s.release(hit.entry.?);
    try t.expect(s.image_query == null);
    s.clear();
    s.rules.lookahead = 1;
    var boundary = first;
    boundary.at = 2;
    var replacement = boundary;
    replacement.hash += 1;
    try t.expect(s.keepBornImages(words, 12000, 2, null, &.{}, &.{}, &.{boundary}));
    const next = try s.lookupUserImages(a, words, 12000, 0, 0, 0, &.{}, &.{}, &.{}, &.{replacement});
    try t.expect(next.entry == null); // the next image row is part of a lookahead-dependent state
    s.clear();
    s.rules.lookahead = 0;
    try t.expect(s.keepBornFrame(words, 12000, 1, null, &.{}, &.{}, &.{boundary}, 42));
    try t.expectEqual(@as(usize, 0), s.entries.items[0].images.len);
    const frame_miss = try s.lookupUserFrame(a, words, 12000, 0, 0, 0, &.{}, &.{}, &.{}, &.{boundary}, 43);
    try t.expect(frame_miss.entry == null); // equal compact words cannot hide a different pre-image frame
    const frame_hit = try s.lookupUserFrame(a, words, 12000, 0, 0, 0, &.{}, &.{}, &.{}, &.{boundary}, 42);
    try t.expectEqual(@as(u32, 1), frame_hit.entry.?.at);
    s.release(frame_hit.entry.?);
    try t.expect(s.image_query == null and s.image_frame == 0);
}
