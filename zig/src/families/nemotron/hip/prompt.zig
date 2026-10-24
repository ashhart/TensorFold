//! One Nemotron prompt chunk, shared by serving and follower adapters.
const hip = @import("hip");
const forward = @import("forward.zig");
const Engine = @import("engine.zig").Engine;
const Lane = @import("lanes.zig").Lane;
const Tensor = hip.ops.Tensor;

pub fn chunk(h: anytype, l: *Lane, prompt: []const u32, from: usize, to: usize, cancel: ?Engine.Cancel) !?Tensor {
    const e = h.e;
    const m = e.model_();
    const n = to - from;
    try e.fresh();
    const o = e.ops(&e.arena);
    if (l.resumed) {
        l.resumed = false;
        if (h.head) |hd| _ = try hd.absorb(
            o,
            &l.s,
            .{ .ptr = l.hidden.base(), .kind = e.act },
            prompt[from..][0..1],
            true,
        );
    }
    const dev = try e.ids(o, prompt[from..to]);
    var poll: Engine.Poll = undefined;
    const hidden = try h.runPrompt(o, .{ .pool = &e.pool, .stream = &l.s, .trace = e.poll(cancel, n, &poll) }, dev, n, from);
    l.s.len = to;
    // the head absorbs each row with the token after it; the last row waits for the first drawn token
    const last = to == prompt.len;
    const pairs = if (last) n - 1 else n;
    if (h.head) |hd| if (pairs > 0) {
        _ = try hd.absorb(o, &l.s, hidden, prompt[from + 1 ..][0..pairs], true);
    };
    // the last row is kept for the head: the first token's pair, or a kept state's resume
    const width = m.dims.hidden * e.act.size();
    try hip.raw.copy(l.hidden, 0, hidden.ptr + (n - 1) * width, width, o.stream);
    if (!last) return null;
    return try e.logits(o, forward.at(hidden, (n - 1) * m.dims.hidden), 1);
}
