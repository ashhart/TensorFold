//! Mini DSpark's backend-neutral host: lane geometry and bytes, each stream's committed rows and held drafts.
const std = @import("std");
const lanes = @import("lanes");

const dr = lanes.drafter;
const Stream = lanes.Stream;
const Features = lanes.backend.Features;

pub const hidden = 2304;
pub const kv = 1024;
pub const heads = 24;
pub const block = 8;
pub const depth = block - 1;
pub const absorb_rows = 64;
pub const vocabulary = 200058;
pub const max_context = 1 << 20;
pub const taps = [_]u32{ 5, 23, 41, 48 };
pub const feature_row_bytes = taps.len * hidden * 2;
/// A global ring grows this many rows at a time, so it rounds a window up by at most `ring_slack` tokens.
pub const ring_step = 128;
pub const ring_slack = ring_step - 1;
/// A sliding layer's ring: its 1024-token window and a block, on the ring step.
pub const local_ring = std.mem.alignForward(usize, 1024 + block, ring_step);

/// What one token of the window costs: each stream's global K and V rows, and its row of the shared score scratch.
pub const TokenBytes = struct { lane: u64, shared: u64 };

pub fn tokenBytes() TokenBytes {
    return .{ .lane = 2 * kv * 2, .shared = block * heads * 4 };
}

pub fn globalRing(capacity: usize) usize {
    return std.mem.alignForward(usize, capacity, ring_step);
}

/// Layers 0 and 2 slide over their window; layer 1 attends to the whole committed context.
pub fn rings(capacity: usize) [3]usize {
    return .{ local_ring, globalRing(capacity), local_ring };
}

/// The two sliding layers' K and V rings and the newest target row: a lane's bytes at any window.
pub fn laneFixedBytes() u64 {
    return 2 * 2 * local_ring * kv * 2 + hidden * 2;
}

pub fn laneBytes(capacity: usize) u64 {
    return laneFixedBytes() + globalRing(capacity) * tokenBytes().lane;
}

/// The attention scores the streams' holds share: a block's rows by heads, for each key of the window and the block.
pub fn scoreBytes(capacity: usize) u64 {
    return tokenBytes().shared * (capacity + block);
}

/// The opener's window fit: a stream costs at most `fixed + window * lane`, the shared scores `shared` a token more.
pub const Fit = struct { fixed: u64, lane: u64, shared: u64 };

pub fn fit() Fit {
    const t = tokenBytes();
    return .{ .fixed = laneBytes(0) + ring_slack * t.lane, .lane = t.lane, .shared = t.shared };
}

/// One stream's committed context, target rows [from, end) in its rings, and the drafts its last hold left.
pub const Book = struct {
    rings: [3]usize,
    from: u64 = 0,
    end: u64 = 0,
    held: [depth]u32 = @splat(0),
    held_len: u32 = 0,

    pub fn init(capacity: usize) !Book {
        if (capacity == 0 or capacity > max_context) return error.DSparkLaneGeometry;
        return .{ .rings = rings(capacity) };
    }

    /// Commits rows [start, start + n): a gap past `end` starts the context anew there, a step back is refused.
    pub fn absorb(b: *Book, start: u64, n: u64) !void {
        if (n == 0 or n > max_context) return error.DraftFeaturesUnsupported;
        if (start < b.end or start + n > max_context) return error.DraftContextGap;
        if (start > b.end) b.from = start;
        b.end = start + n;
        b.held_len = 0;
    }

    /// The drafts a hold at `position` makes: at most `depth`, and only once every row before its anchor is committed.
    pub fn hold(b: *const Book, position: u64, want: u32) !u32 {
        if (want == 0 or position == 0) return error.DraftDepth;
        if (b.end != position - 1) return error.DraftContextGap;
        return @min(want, depth);
    }

    /// Keeps a hold's drafts for `take`; a draft past the vocabulary means its logits were not finite.
    pub fn keep(b: *Book, drafts: []const u32) !void {
        for (drafts) |t| if (t >= vocabulary) return error.DraftNonfinite;
        @memcpy(b.held[0..drafts.len], drafts);
        b.held_len = @intCast(drafts.len);
    }

    pub fn take(b: *const Book, out: []u32) !void {
        if (out.len > b.held_len) return error.DraftHeldFewer;
        @memcpy(out, b.held[0..out.len]);
    }
};

/// A backend's piece of an absorb: `rows` target rows at `offset` of `features`, into a lane's rings at `position`.
pub fn Absorb(comptime Lane: type) type {
    return struct { lane: *Lane, rings: [3]usize, features: Features, offset: u64, position: u64, rows: u32 };
}

/// A backend's piece of a hold: `depth` drafts after `pending`, its block anchored after the rows [from, anchor).
pub fn Hold(comptime Lane: type) type {
    return struct { lane: *Lane, rings: [3]usize, from: u64, anchor: u64, pending: u32, position: u64, depth: u32, sampling: ?lanes.Sampling };
}

/// What the round loop's depth rule takes from the backend: a draft's cost, plain rounds, one batch a round.
pub const Policy = struct { step_ms: f64 = 0, plain_guard: bool = true, batched: bool = false };

/// The drafter over `B`: Lane, openLane, closeLane, setScores, absorb and hold (all streams a call), policy.
pub fn Drafter(comptime B: type) type {
    return struct {
        const Self = @This();
        const Entry = struct { book: Book, lane: B.Lane };

        gpa: std.mem.Allocator,
        backend: *B,
        capacity: usize = 1,
        streams: std.AutoHashMapUnmanaged(*Stream, Entry) = .empty,
        absorbs: std.ArrayList(Absorb(B.Lane)) = .empty,
        holds: std.ArrayList(Hold(B.Lane)) = .empty,
        books: std.ArrayList(*Book) = .empty,
        drafts: std.ArrayList([depth]u32) = .empty,

        pub fn init(gpa: std.mem.Allocator, backend: *B) Self {
            return .{ .gpa = gpa, .backend = backend };
        }

        pub fn deinit(d: *Self) void {
            var it = d.streams.valueIterator();
            while (it.next()) |e| d.backend.closeLane(&e.lane);
            d.streams.deinit(d.gpa);
            d.absorbs.deinit(d.gpa);
            d.holds.deinit(d.gpa);
            d.books.deinit(d.gpa);
            d.drafts.deinit(d.gpa);
            d.* = undefined;
        }

        /// The selected window fixes every new lane's global ring and the shared score scratch's size.
        pub fn setContext(d: *Self, window: u32) !void {
            const capacity: usize = window;
            if (capacity == 0 or capacity > max_context) return error.DSparkLaneGeometry;
            if (d.streams.count() != 0) return error.DSparkContextActive;
            try d.backend.setScores(scoreBytes(capacity));
            d.capacity = capacity;
        }

        pub fn drafter(d: *Self) dr.Drafter {
            return .{ .ptr = d, .vtable = &.{ .taps = tapsFn, .facts = factsFn, .open = openFn, .absorb = absorbFn, .hold = holdFn, .held = heldFn, .release = releaseFn } };
        }

        fn cast(ptr: *anyopaque) *Self {
            return @ptrCast(@alignCast(ptr));
        }

        fn tapsFn(_: *anyopaque) []const u32 {
            return &taps;
        }

        fn factsFn(ptr: *anyopaque) dr.Facts {
            const d = cast(ptr);
            const p = d.backend.policy();
            return .{ .depth = depth, .step_ms = p.step_ms, .plain_guard = p.plain_guard, .batched = p.batched, .stream_bytes = @intCast(laneBytes(d.capacity)) };
        }

        fn openFn(ptr: *anyopaque, stream: *Stream) anyerror!void {
            const d = cast(ptr);
            if (d.streams.contains(stream)) return error.DraftStreamOpen;
            const book = try Book.init(d.capacity);
            var lane = try d.backend.openLane(book.rings);
            errdefer d.backend.closeLane(&lane);
            try d.streams.put(d.gpa, stream, .{ .book = book, .lane = lane });
        }

        fn releaseFn(ptr: *anyopaque, stream: *Stream) void {
            const d = cast(ptr);
            if (d.streams.fetchRemove(stream)) |entry| {
                var lane = entry.value.lane;
                d.backend.closeLane(&lane);
            }
        }

        fn heldFn(ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void {
            const d = cast(ptr);
            for (streams, out) |stream, tokens| {
                const e = d.streams.getPtr(stream) orelse return error.DraftStreamUnknown;
                try e.book.take(tokens);
            }
        }

        /// Commits each stream's rows, then hands the backend every stream's pieces of `absorb_rows` rows at once.
        fn absorbFn(ptr: *anyopaque, items: []const dr.Absorb) anyerror!void {
            const d = cast(ptr);
            d.absorbs.clearRetainingCapacity();
            for (items) |item| {
                const e = d.streams.getPtr(item.stream) orelse return error.DraftStreamUnknown;
                const f = item.features;
                if (f.space != .device or f.dtype != .bf16 or f.row_bytes != feature_row_bytes) return error.DraftFeaturesUnsupported;
                if (item.rows != null) return error.TreesNotBuilt;
                try e.book.absorb(item.start, f.rows);
                var position = item.start;
                while (position < e.book.end) {
                    const rows: u32 = @intCast(@min(absorb_rows, e.book.end - position));
                    const offset = f.offset + (position - item.start) * feature_row_bytes;
                    try d.absorbs.append(d.gpa, .{ .lane = &e.lane, .rings = e.book.rings, .features = f, .offset = offset, .position = position, .rows = rows });
                    position += rows;
                }
            }
            if (d.absorbs.items.len > 0) try d.backend.absorb(d.absorbs.items);
        }

        /// Checks every stream's hold first, then one backend call drafts them all; each stream keeps its own drafts.
        fn holdFn(ptr: *anyopaque, items: []const dr.Hold) anyerror!void {
            const d = cast(ptr);
            d.holds.clearRetainingCapacity();
            d.books.clearRetainingCapacity();
            for (items) |item| {
                const e = d.streams.getPtr(item.stream) orelse return error.DraftStreamUnknown;
                if (item.lanes != null) return error.TreesNotBuilt;
                const n = try e.book.hold(item.position, item.depth);
                const b = &e.book;
                try d.holds.append(d.gpa, .{ .lane = &e.lane, .rings = b.rings, .from = b.from, .anchor = b.end, .pending = item.pending, .position = item.position, .depth = n, .sampling = item.sampling });
                try d.books.append(d.gpa, b);
            }
            if (d.holds.items.len == 0) return;
            try d.drafts.resize(d.gpa, d.holds.items.len);
            try d.backend.hold(d.holds.items, d.drafts.items);
            for (d.holds.items, d.books.items, d.drafts.items) |h, b, *drafts| try b.keep(drafts[0..h.depth]);
        }
    };
}
