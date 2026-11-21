//! A ZONOS3 request's spans that follow the reading position: its follower and, without LACI, the reader it needs.
const std = @import("std");
const api = @import("engine_api");
const z3 = @import("zonos3");

const sf = z3.span_follow;

/// Whether a request's spans that follow the reading name positions of its text: BOS, its tokens, EOS.
pub fn fits(r: *const api.SpeechRequest, text_len: usize) bool {
    const st = r.steering orelse return true;
    const f = st.follow orelse return true;
    for (f.ranges) |x| if (x[1] > text_len) return false;
    return true;
}

pub const State = struct {
    follower: sf.Follower,
    /// Without LACI, the readings of the request's own passes; under LACI the session hands the follower its own.
    reader: ?sf.Reader = null,
    heads: []z3.laci.Head,
    weights: []f32,

    pub fn init(gpa: std.mem.Allocator, f: api.SpeechSteering.Follow) !State {
        const heads = try gpa.alloc(z3.laci.Head, f.heads.len);
        errdefer gpa.free(heads);
        for (heads, f.heads) |*to, from| to.* = .{ .layer = from.layer, .head = from.head, .radius = from.radius };
        const weights = try gpa.alloc(f32, f.ranges.len);
        errdefer gpa.free(weights);
        const follower = try sf.Follower.init(gpa, f.ranges, .{ .lead = f.lead, .lag = f.lag });
        return .{ .follower = follower, .heads = heads, .weights = weights };
    }

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        if (s.reader) |*r| r.deinit();
        s.follower.deinit();
        gpa.free(s.heads);
        gpa.free(s.weights);
    }

    /// After the prompt ran on the request's lane: the text rows' keys, for readings of its own passes.
    pub fn read(s: *State, gpa: std.mem.Allocator, engine: anytype, text_start: usize, text_len: usize) !void {
        for (s.follower.ranges) |r| if (r[1] > text_len) return error.FollowRangeOutsideTheText;
        s.reader = try sf.Reader.init(gpa, engine, .{ .heads = s.heads, .frames_per_token = 1 }, text_start, text_len);
    }

    /// The reading of the pass that took generated row `frame - 1`, once there was one.
    pub fn observeLast(s: *State, frame: usize) !void {
        const r = &(s.reader orelse return);
        if (frame >= 2) try s.follower.observe(frame - 1, r.position());
    }

    /// What a pass copies a probed row's queries into; null under LACI, whose own probe does it.
    pub fn probe(s: *State) ?*z3.laci_probe.Probed {
        return if (s.reader) |*r| &r.probed else null;
    }

    /// The spans' weights for the forward that takes generated row `frame`.
    pub fn at(s: *State, frame: usize) ![]const f32 {
        try s.follower.weights(frame, s.weights);
        return s.weights;
    }
};
