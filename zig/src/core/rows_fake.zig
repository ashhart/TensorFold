//! A host-memory family for the rows store's tests: its image has keys, a pooled index, MTP rows, a ring and own words.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = @import("rows_store.zig");
const image = rows.image;
const Allocator = std.mem.Allocator;
const Span = [2]u32;

pub const window = 600; // the ring's positions: wider than a block, so a ring has blocks of its own
const key_row = 8;
const pool_row = 6;
const mtp_row = 4;
const ring_row = 4;

/// A kept state: its position and image.
pub const Kept = struct { at: u32, img: []u8 };

/// The error every copy fails with once `after` of them have succeeded.
pub const Fault = struct { after: usize, err: anyerror };

/// The family: the request in its pass (prompt, spans, starts), its position and running sum.
pub const Fake = struct {
    gpa: Allocator,
    prompt: []const u32 = &.{},
    spans: []const Span = &.{},
    starts: []const u32 = &.{},
    planned: bool = false,
    at: u32 = 0,
    sum: u64 = 0,
    restored_ok: bool = true, // every restore so far equalled the image a fresh pass gives
    live: usize = 0,
    blanks: usize = 0,
    slots: ?usize = null, // a family whose states each take one of so many slots (Nemotron CUDA's): past them, none
    gathers: ?usize = null, // copies out of kept states that succeed before every later one fails (null: all succeed)
    scatters: ?Fault = null, // copies into new states that succeed before every later one fails (null: all succeed)
    gathered: usize = 0, // copies out of kept states made so far

    pub fn snapshots(f: *Fake) pc.Snapshots {
        return .{ .ptr = f, .vtable = &.{ .bytes = bytesFn, .save = saveFn, .restore = restoreFn, .drop = dropFn, .rows = &.{ .parts = partsFn, .gather = gatherFn, .blank = blankFn, .scatter = scatterFn } } };
    }

    fn of(ptr: *anyopaque) *Fake {
        return @ptrCast(@alignCast(ptr));
    }

    /// The image's parts at `at`.
    pub fn layout(at: u32, out: *std.ArrayList(image.Part), a: Allocator) !void {
        try out.append(a, .{ .bytes = 16 });
        try out.append(a, .{ .bytes = @as(u64, at) * key_row, .row = key_row });
        try out.append(a, .{ .bytes = @as(u64, at / 4 + 1) * pool_row, .row = pool_row, .per = 4 });
        try out.append(a, .{ .bytes = @as(u64, at) * mtp_row, .row = mtp_row });
        try out.append(a, .{ .bytes = @as(u64, @min(at, window)) * ring_row, .row = ring_row, .window = window });
        try out.append(a, .{ .bytes = 12 });
    }

    fn imageBytes(at: u32) usize {
        return 16 + @as(usize, at) * key_row + @as(usize, at / 4 + 1) * pool_row + @as(usize, at) * mtp_row + @as(usize, @min(at, window)) * ring_row + 12;
    }

    /// Whether row `p` was decoded, and (planned) the rows of the chunk holding it: both change its bits.
    fn arithmetic(f: *const Fake, p: u32) u64 {
        var x: u64 = 0;
        for (f.spans) |s| if (p >= s[0] and p < s[1]) {
            x = 0xdec0;
        };
        if (f.planned) {
            var lo: u32 = 0;
            var hi: u32 = @intCast(f.prompt.len);
            for (f.starts) |s| {
                if (s <= p and s > lo) lo = s;
                if (s > p and s < hi) hi = s;
            }
            x +%= @as(u64, hi - lo) << 20;
        }
        return x;
    }

    /// The image a fresh pass leaves at `at` (the MTP rows read one token on).
    pub fn build(f: *const Fake, at: u32, out: []u8) void {
        @memset(out, 0);
        const p_ = f.prompt;
        var sum: u64 = 0;
        for (p_[0..at]) |t| sum = sum *% 31 +% t;
        std.mem.writeInt(u64, out[0..8], at, .little);
        std.mem.writeInt(u64, out[8..16], sum, .little);
        var off: usize = 16;
        for (0..at) |p| std.mem.writeInt(u64, out[off + p * key_row ..][0..8], std.hash.Wyhash.hash(1, std.mem.sliceAsBytes(p_[0 .. p + 1])) ^ f.arithmetic(@intCast(p)), .little);
        off += at * key_row;
        for (0..at / 4 + 1) |u| {
            const end = @min((u + 1) * 4, at);
            const h = std.hash.Wyhash.hash(2, std.mem.sliceAsBytes(p_[0..end])) ^ f.arithmetic(@intCast(end -| 1));
            @memcpy(out[off + u * pool_row ..][0..pool_row], std.mem.asBytes(&h)[0..pool_row]);
        }
        off += (at / 4 + 1) * pool_row;
        for (0..at) |p| std.mem.writeInt(u32, out[off + p * mtp_row ..][0..4], @truncate(std.hash.Wyhash.hash(3, std.mem.sliceAsBytes(p_[0 .. p + 2]))), .little);
        off += at * mtp_row;
        for (at -| window..at) |p| std.mem.writeInt(u32, out[off + (p % window) * ring_row ..][0..4], p_[p] *% 7 +% @as(u32, @intCast(p)), .little);
        off += @as(usize, @min(at, window)) * ring_row;
        @memcpy(out[off..][0..12], "family words");
    }

    fn bytesFn(_: *anyopaque, at: u32) u64 {
        return imageBytes(at);
    }
    fn saveFn(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        if (at != f.at) return error.NotAtMark;
        if (f.slots) |n| if (f.live >= n) return error.NoSnapshotSlot;
        const k = try f.gpa.create(Kept);
        errdefer f.gpa.destroy(k);
        k.* = .{ .at = at, .img = try f.gpa.alloc(u8, imageBytes(at)) };
        f.build(at, k.img);
        f.live += 1;
        return k;
    }
    fn restoreFn(ptr: *anyopaque, _: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const f = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        const want = try f.gpa.alloc(u8, imageBytes(k.at));
        defer f.gpa.free(want);
        f.build(k.at, want);
        f.restored_ok = f.restored_ok and std.mem.eql(u8, want, k.img);
        f.at = k.at;
        f.sum = std.mem.readInt(u64, k.img[8..16], .little);
    }
    fn dropFn(ptr: *anyopaque, saved: pc.Saved) void {
        const f = of(ptr);
        const k: *Kept = @ptrCast(@alignCast(saved));
        f.gpa.free(k.img);
        f.gpa.destroy(k);
        f.live -= 1;
    }
    fn partsFn(_: *anyopaque, at: u32, out: *std.ArrayList(image.Part), a: Allocator) anyerror!void {
        try layout(at, out, a);
    }
    fn gatherFn(ptr: *anyopaque, saved: pc.Saved, ranges: []const image.Range, out: []u8) anyerror!void {
        const f = of(ptr);
        f.gathered += 1;
        if (f.gathers) |*n| if (n.* == 0) return error.GatherFailed else {
            n.* -= 1;
        };
        const k: *Kept = @ptrCast(@alignCast(saved));
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(out[o..][0..@intCast(r.len)], k.img[@intCast(r.off)..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }
    fn blankFn(ptr: *anyopaque, at: u32) anyerror!pc.Saved {
        const f = of(ptr);
        if (f.slots) |n| if (f.live >= n) return error.NoSnapshotSlot;
        const k = try f.gpa.create(Kept);
        errdefer f.gpa.destroy(k);
        k.* = .{ .at = at, .img = try f.gpa.alloc(u8, imageBytes(at)) };
        @memset(k.img, 0xaa);
        f.live += 1;
        f.blanks += 1;
        return k;
    }
    fn scatterFn(ptr: *anyopaque, saved: pc.Saved, ranges: []const image.Range, in: []const u8) anyerror!void {
        if (of(ptr).scatters) |*n| if (n.after == 0) return n.err else {
            n.after -= 1;
        };
        const k: *Kept = @ptrCast(@alignCast(saved));
        var o: usize = 0;
        for (ranges) |r| {
            @memcpy(k.img[@intCast(r.off)..][0..@intCast(r.len)], in[o..][0..@intCast(r.len)]);
            o += @intCast(r.len);
        }
    }

    /// A prompt pass of `prompt` from `plan.from`, keeping at its marks; the running sum a fresh pass would give.
    pub fn pass(f: *Fake, s: *pc.Store, plan: pc.Plan) u64 {
        if (plan.from == 0) {
            f.at = 0;
            f.sum = 0;
        }
        var mi: usize = 0;
        for (f.prompt[plan.from..]) |t| {
            f.sum = f.sum *% 31 +% t;
            f.at += 1;
            while (mi < plan.marks.len and plan.marks[mi] < f.at) mi += 1;
            if (mi < plan.marks.len and plan.marks[mi] == f.at) _ = s.keep(f.prompt, f.at, null, f.starts, f.spans);
        }
        return f.sum;
    }
};

pub fn fresh(prompt: []const u32) u64 {
    var sum: u64 = 0;
    for (prompt) |t| sum = sum *% 31 +% t;
    return sum;
}
