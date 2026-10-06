//! A drafter outside the target family (EAGLE-3, a DFlash block, a family's MTP head loaded on its own): it reads the
//! target's tapped states (backend.Features) and holds drafts for a stream's next round. drafted.zig puts one in front of
//! any target backend that exposes `features`; the target verifies every draft, so the drafter decides speed, never output.
const Stream = @import("stream.zig").Stream;
const Features = @import("backend.zig").Features;

pub const Drafter = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        /// The target's layers the drafter reads, in the order its features concatenate them.
        taps: *const fn (ptr: *anyopaque) []const u32,
        /// Drafts a round at most.
        depth: *const fn (ptr: *anyopaque) u32,
        /// A new stream: its own caches, empty.
        open: *const fn (ptr: *anyopaque, s: *Stream) anyerror!void,
        /// The states of cache rows [start, start + f.rows) and the token after each; rows at or past `start` the drafter
        /// held before are replaced (a rollback is the next absorb's `start`).
        absorb: *const fn (ptr: *anyopaque, s: *Stream, f: Features, start: u64, follow: []const u32) anyerror!void,
        /// Draft `depth` tokens after the last absorbed row's follow token, the first landing at `position`.
        hold: *const fn (ptr: *anyopaque, s: *Stream, position: u64, depth: u32) anyerror!void,
        /// The held drafts on the host; how many.
        held: *const fn (ptr: *anyopaque, s: *Stream, out: []u32) anyerror!usize,
        /// The stream left the rounds.
        release: *const fn (ptr: *anyopaque, s: *Stream) void,
    };

    pub fn taps(d: Drafter) []const u32 {
        return d.vtable.taps(d.ptr);
    }
    pub fn depth(d: Drafter) u32 {
        return d.vtable.depth(d.ptr);
    }
    pub fn open(d: Drafter, s: *Stream) !void {
        return d.vtable.open(d.ptr, s);
    }
    pub fn absorb(d: Drafter, s: *Stream, f: Features, start: u64, follow: []const u32) !void {
        return d.vtable.absorb(d.ptr, s, f, start, follow);
    }
    pub fn hold(d: Drafter, s: *Stream, position: u64, n: u32) !void {
        return d.vtable.hold(d.ptr, s, position, n);
    }
    pub fn held(d: Drafter, s: *Stream, out: []u32) !usize {
        return d.vtable.held(d.ptr, s, out);
    }
    pub fn release(d: Drafter, s: *Stream) void {
        d.vtable.release(d.ptr, s);
    }
};
