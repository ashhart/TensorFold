//! A drafter outside the target family reading its tapped states; each call takes a round's streams together.
const Stream = @import("stream.zig").Stream;
const Features = @import("backend.zig").Features;

/// What the round loop's depth rule needs from the drafter itself (not from the target, whose own head may not exist).
pub const Facts = struct {
    depth: u32, // drafts a round at most
    step_ms: f64, // one more draft's cost (timed at load); the rule prices a round as forward(rows) + step_ms x drafts
    prior: []const f64 = &.{}, // acceptance by depth until a stream has its own
    plain_guard: bool = true, // plain rounds compete with drafted depths (the rule may choose no drafts)
    batched: bool = false, // hold() runs a shared round's streams as one batch
};

/// Target rows [start, start + features.rows) and the token after each; rows at or past `start` replace what was held.
pub const Absorb = struct { stream: *Stream, features: Features, start: u64, follow: []const u32 };

/// Draft `depth` tokens after `pending`, the first at `position`; asked only once row `position - 2` was absorbed.
pub const Hold = struct { stream: *Stream, pending: u32, position: u64, depth: u32 };

pub const Drafter = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The target's layers the drafter reads, in the order its features concatenate them.
        taps: *const fn (ptr: *anyopaque) []const u32,
        facts: *const fn (ptr: *anyopaque) Facts,
        /// A new stream: its own caches, empty.
        open: *const fn (ptr: *anyopaque, s: *Stream) anyerror!void,
        absorb: *const fn (ptr: *anyopaque, items: []const Absorb) anyerror!void,
        hold: *const fn (ptr: *anyopaque, items: []const Hold) anyerror!void,
        /// Each stream's held drafts on the host: out[i] gets streams[i]'s first out[i].len (error if it holds fewer).
        held: *const fn (ptr: *anyopaque, streams: []const *Stream, out: []const []u32) anyerror!void,
        /// The stream left the rounds (only called for an opened stream).
        release: *const fn (ptr: *anyopaque, s: *Stream) void,
    };

    pub fn taps(d: Drafter) []const u32 {
        return d.vtable.taps(d.ptr);
    }
    pub fn facts(d: Drafter) Facts {
        return d.vtable.facts(d.ptr);
    }
    pub fn open(d: Drafter, s: *Stream) !void {
        return d.vtable.open(d.ptr, s);
    }
    pub fn absorb(d: Drafter, items: []const Absorb) !void {
        return d.vtable.absorb(d.ptr, items);
    }
    pub fn hold(d: Drafter, items: []const Hold) !void {
        return d.vtable.hold(d.ptr, items);
    }
    pub fn held(d: Drafter, streams: []const *Stream, out: []const []u32) !void {
        return d.vtable.held(d.ptr, streams, out);
    }
    pub fn release(d: Drafter, s: *Stream) void {
        d.vtable.release(d.ptr, s);
    }
};
