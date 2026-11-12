//! A readout stream's state on the device: its memory rows, span sums, last row and ints, opened from its spans.
const std = @import("std");
const DeviceBuffer = @import("cuda").DeviceBuffer;
const Head = @import("head.zig").Head;
const put = @import("weights.zig").put;
const Stream = @import("cuda").Stream;

/// One readout's spans, as the lane core validated them (core/lanes/readout.zig's Spec) with its prompt.
pub const Request = struct {
    question_spans: []const [2]u32,
    question_types: []const u8,
    option_spans: []const [2]u32,
    option_question: []const u16,
    prompt: []const u32,
};

pub const State = struct {
    buf: DeviceBuffer,
    rows: usize, // the prompt's
    consumed: usize = 0,
    questions: usize,
    options: usize,
    types: []u8,
    first: []u32, // each question's first option, then the option count

    pub fn memory(st: *const State, h: *const Head) u64 {
        return st.buf.ptr + h.lane.memory;
    }
    pub fn sums(st: *const State, h: *const Head) u64 {
        return st.buf.ptr + h.lane.sums;
    }
    pub fn g(st: *const State, h: *const Head) u64 {
        return st.buf.ptr + h.lane.g;
    }
    pub fn spans(st: *const State, h: *const Head) u64 {
        return st.buf.ptr + h.lane.spans;
    }
    pub fn ids(st: *const State, h: *const Head) u64 {
        return st.buf.ptr + h.lane.ids;
    }
    pub fn owner(st: *const State, h: *const Head) u64 {
        return st.buf.ptr + h.lane.owner;
    }
    pub fn firsts(st: *const State, h: *const Head) u64 {
        return st.buf.ptr + h.lane.first;
    }
};

/// The request's state, sums zeroed on `stream`; refused past the bounds, with an empty question or a token past vocab.
pub fn open(h: *Head, req: Request, stream: Stream) !*State {
    const b = h.bounds;
    const q = req.question_spans.len;
    const t = req.option_spans.len;
    if (q == 0 or q > b.questions or t == 0 or t > b.options or req.prompt.len == 0 or req.prompt.len > b.rows) return error.ReadoutTooLarge;
    if (req.question_types.len != q or req.option_question.len != t) return error.InvalidReadout;
    for (req.prompt) |id| if (id >= h.vocab) return error.InvalidReadoutToken;
    const a = h.gpa;
    const st = try a.create(State);
    errdefer a.destroy(st);
    const first = try a.alloc(u32, q + 1);
    errdefer a.free(first);
    @memset(first, @intCast(t));
    var at: usize = 0;
    for (0..q) |k| {
        first[k] = @intCast(at);
        while (at < t and req.option_question[at] == k) at += 1;
        if (at == first[k]) return error.InvalidReadoutGroups;
    }
    if (at != t) return error.InvalidReadoutGroups;
    const types = try a.dupe(u8, req.question_types);
    errdefer a.free(types);
    var buf = try DeviceBuffer.alloc(h.d, h.lane.total);
    errdefer buf.free();
    st.* = .{ .buf = buf, .rows = req.prompt.len, .questions = q, .options = t, .types = types, .first = first };
    // the ints go up once; the sums start at zero, filled on the pieces' stream last (nothing after it can fail)
    const ints = try a.alloc(i32, 2 * (q + t) + req.prompt.len + t + q + 1);
    defer a.free(ints);
    var n: usize = 0;
    for ([_][]const [2]u32{ req.question_spans, req.option_spans }) |list| for (list) |s| {
        ints[n] = @intCast(s[0]);
        ints[n + 1] = @intCast(s[1]);
        n += 2;
    };
    try put(buf, h.lane.spans, std.mem.sliceAsBytes(ints[0..n]));
    for (req.prompt, ints[n..][0..req.prompt.len]) |id, *x| x.* = @intCast(id);
    try put(buf, h.lane.ids, std.mem.sliceAsBytes(ints[n..][0..req.prompt.len]));
    n += req.prompt.len;
    for (req.option_question, ints[n..][0..t]) |k, *x| x.* = k;
    try put(buf, h.lane.owner, std.mem.sliceAsBytes(ints[n..][0..t]));
    n += t;
    for (first, ints[n..][0 .. q + 1]) |f, *x| x.* = @intCast(f);
    try put(buf, h.lane.first, std.mem.sliceAsBytes(ints[n..][0 .. q + 1]));
    const zero: DeviceBuffer = .{ .d = h.d, .ptr = st.sums(h), .len = (q + t) * h.cfg.hidden_size * 4, .borrowed = true };
    try zero.fill8(0, stream.handle);
    return st;
}

pub fn close(h: *Head, st: *State) void {
    st.buf.free();
    h.gpa.free(st.types);
    h.gpa.free(st.first);
    h.gpa.destroy(st);
}
