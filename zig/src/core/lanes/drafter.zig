//! A drafter outside the target family reading its tapped states; each call takes a round's streams together.
const Stream = @import("stream.zig").Stream;
const Features = @import("backend.zig").Features;
const Cost = @import("config.zig").Cost;
const Shape = @import("shape.zig").Shape;
const Held = @import("stream.zig").Held;
const Allocator = @import("std").mem.Allocator;

/// What the round loop's depth rule needs from the drafter itself (not from the target, whose own head may not exist).
pub const Facts = struct {
    depth: u32, // drafts a round at most
    step_ms: f64, // one more draft's cost (timed at load); the rule prices a round as forward(rows) + step_ms x drafts
    prior: []const f64 = &.{}, // acceptance by depth until a stream has its own
    plain_guard: bool = true, // plain rounds compete with drafted depths (the rule may choose no drafts)
    batched: bool = false, // hold() runs a shared round's streams as one batch
    stream_bytes: usize = 0, // device bytes each opened stream holds (its caches), for memory admission
    trees: bool = false, // hold() takes a tree of lanes (Hold.lanes), each lane its parent's rank-r token
    window_costs: []const Cost = &.{}, // the target's window ms by rows (timed at load), for the depth rule
    lane_costs: []const Cost = &.{}, // the same for the lane planner (Model.lane_costs)
    tree_nodes: u32 = 0, // hold() without lanes grows its own tree of up to `depth` nodes, at most this many (tree, chances)
};

/// Target rows [start, start + features.rows) and the token after each; `rows`, when given, picks which to absorb.
pub const Absorb = struct {
    stream: *Stream,
    features: Features,
    start: u64,
    follow: []const u32,
    rows: ?[]const u32 = null,
    prompt: bool = true, // prompt ingestion keeps its historical arithmetic; accepted live rows set false
    offered: u32 = 0, // the drafter's grown tree's first nodes the verified window carried (0: none of its own)
};

/// `depth` drafts after `pending` at `position`; with `lanes` a tree, lane i the rank-r choice after its parent.
pub const Hold = struct { stream: *Stream, pending: u32, position: u64, depth: u32, lanes: ?*const Shape = null };

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
        /// A grown tree's drafts and parents (gpa-owned; -1: under the pending token); null: no tree held.
        tree: ?*const fn (ptr: *anyopaque, s: *Stream, gpa: Allocator) anyerror!?Held = null,
        /// Each held node's chance of landing, its parents' included; false: none to give.
        chances: ?*const fn (ptr: *anyopaque, s: *Stream, out: []f64) anyerror!bool = null,
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
    pub fn tree(d: Drafter, s: *Stream, gpa: Allocator) !?Held {
        const f = d.vtable.tree orelse return null;
        return f(d.ptr, s, gpa);
    }
    pub fn chances(d: Drafter, s: *Stream, out: []f64) !bool {
        const f = d.vtable.chances orelse return false;
        return f(d.ptr, s, out);
    }
};
