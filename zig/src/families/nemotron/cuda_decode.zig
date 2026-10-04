//! decode.prefill + serial_decode on the Zig engine: the same rounds, the same tokens, timed the same way.

const std = @import("std");
const Engine = @import("cuda_engine.zig").Engine;
const seconds = @import("cuda_engine.zig").seconds;
const lookahead = @import("cuda_engine.zig").lookahead;
const state = @import("cuda_state.zig");
const drafts = @import("cuda_drafts.zig");

pub const Options = struct { stop_eos: bool = true };

pub const Result = struct {
    tokens: []u32,
    prefill_seconds: f64,
    decode_seconds: f64,
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
};

/// `count` tokens (the first from the prompt's last row), stopping after an end token when `stop_eos`.
pub fn generate(gpa: std.mem.Allocator, io: std.Io, e: *Engine, drafter: ?*drafts.Drafter, prompt: []const u32, count: usize, o: Options) !Result {
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(gpa);
    const head = if (drafter) |d| d.head else null;
    const t0 = std.Io.Clock.awake.now(io);
    try out.append(gpa, try e.prefill(prompt, null, head));
    const prefill_s = seconds(io, t0);
    try e.stream.synchronize();
    const t1 = std.Io.Clock.awake.now(io);
    var st: drafts.Stats = .{};
    if (out.items.len < count and !(o.stop_eos and e.c.isEos(out.items[0]))) {
        if (drafter) |d| {
            const last = (prompt.len - 1) % state.prefill_rows;
            st = try drafts.decode(gpa, e, d.head, &d.rule, prompt, &out, count, o.stop_eos, e.b.p_hidden + last * @as(u64, e.c.hidden) * 2);
        } else if (e.serial != null) {
            st.rounds = try serialGraphs(gpa, e, &out, count, o.stop_eos);
        } else while (out.items.len < count and !(o.stop_eos and e.c.isEos(out.items[out.items.len - 1]))) {
            try out.append(gpa, try e.step(out.items[out.items.len - 1], null));
            st.rounds += 1;
        }
    }
    const decode_s = seconds(io, t1);
    try e.stream.synchronize();
    return .{ .tokens = try out.toOwnedSlice(gpa), .prefill_seconds = prefill_s, .decode_seconds = decode_s, .rounds = st.rounds, .drafted = st.drafted, .accepted = st.accepted };
}

/// serial_decode as one graph replay a round, queued `lookahead` deep: the device feeds each token to the next round.
fn serialGraphs(gpa: std.mem.Allocator, e: *Engine, out: *std.ArrayList(u32), count: usize, stop_eos: bool) !usize {
    const g = e.serial.?;
    try e.upload(out.items[out.items.len - 1]);
    const base = e.pos;
    const total = count - out.items.len;
    var launched: usize = 0;
    var read: usize = 0;
    while (read < total) {
        while (launched < total and launched - read < lookahead) : (launched += 1) {
            try g.launchOn(e.stream);
            try e.done[launched % lookahead].record(e.stream);
        }
        try e.done[read % lookahead].synchronize();
        const tok = e.written(base + read + 1);
        try out.append(gpa, tok);
        read += 1;
        if (stop_eos and e.c.isEos(tok)) break;
    }
    e.pos = base + launched;
    e.parity ^= launched & 1;
    if (launched > 0) e.prev_keep = 1;
    return read;
}
