//! Native image-prefix preparation and cancellable prompt chunks driven by the shared lane host.
const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const c = @import("config.zig");
const backend = @import("backend.zig");
const decoder = @import("decoder.zig");
const Compute = @import("compute.zig").Compute;
const Tensor = hip.ops.Tensor;

fn cancelled(ptr: *anyopaque) bool {
    const s: *lanes.Stream = @ptrCast(@alignCast(ptr));
    return s.isCancelled();
}

fn compute(h: *backend.Backend, s: *lanes.Stream) Compute {
    var out = h.e.compute();
    out.cancel = .{ .ptr = s, .check = cancelled };
    return out;
}

fn begin(h: *backend.Backend, s: *lanes.Stream) !*backend.Lane {
    if (s.images.len != 1) return error.ParseNeedsOneImage;
    if (h.streams.map.count() >= h.e.streams) return error.LaneMemory;
    const image = s.images[0];
    const g = try @import("media.zig").geometry(image);
    if (image.tokens >= s.prompt_len or s.prompt_len > h.e.window) return error.ContextExceeded;
    for (s.prompt()) |id| if (id >= c.vocab) return error.BadInputToken;
    try h.e.laneRoom();
    const l = try h.a.create(backend.Lane);
    errdefer h.a.destroy(l);
    l.* = .{ .state = try decoder.State.init(&h.e.r, h.e.window), .prefix = undefined, .image_rows = image.tokens };
    errdefer l.state.deinit();
    l.prefix = try hip.DeviceBuffer.alloc(&h.e.r, image.tokens * c.hidden * 2);
    errdefer l.prefix.free();
    try h.streams.map.put(h.a, s, l);
    errdefer _ = h.streams.map.remove(s);
    errdefer h.e.stream.synchronize() catch {};
    if (s.reuse.saved) |saved| {
        try @import("cache.zig").restore(h, s, saved);
        if (l.fill != s.reuse.at) return error.BadSnapshot;
        s.cached = s.reuse.at;
        if (l.fill >= l.image_rows) return l;
    }
    h.e.scratch.reset();
    const comp = compute(h, s);
    var raw = try hip.DeviceBuffer.fromHost(&h.e.r, std.mem.sliceAsBytes(image.pixels));
    defer raw.free();
    const input = try comp.alloc(image.pixels.len, .bf16);
    try comp.ops.cast(.{ .ptr = raw.base(), .kind = .f32 }, input, image.pixels.len);
    try @import("tower.zig").encode(comp, input, g, .{ .ptr = l.prefix.base(), .kind = .bf16 });
    try h.e.stream.synchronize();
    return l;
}

pub fn step(h: *backend.Backend, s: *lanes.Stream, max_rows: u32) !bool {
    if (s.isCancelled()) return error.Cancelled;
    const l = h.streams.map.get(s) orelse try begin(h, s);
    const from = l.fill;
    if (from == s.prompt_len) {
        h.e.scratch.reset();
        try finish(h, s, l, compute(h, s));
        s.fill_at.store(@intCast(from), .release);
        return true;
    }
    var to = @min(s.prompt_len, from + @min(if (max_rows == 0) 512 else max_rows, 512));
    for (s.reuse.marks) |at| if (at > from and at < to) {
        to = at;
    };
    for (s.chunks) |at| if (at > from and at < to) {
        to = at;
    };
    if (to <= from) return error.EmptyPromptChunk;
    h.e.scratch.reset();
    const comp = compute(h, s);
    var ids = try hip.DeviceBuffer.fromHost(&h.e.r, std.mem.sliceAsBytes(s.prompt()[from..to]));
    defer ids.free();
    const embedded = try decoder.embed(comp, ids.base(), to - from);
    if (from < l.image_rows) {
        const image_end = @min(to, l.image_rows);
        const src: Tensor = .{ .ptr = l.prefix.base() + from * c.hidden * 2, .kind = .bf16 };
        try comp.ops.copyCols(src, c.hidden, 0, embedded.ptr, image_end - from, c.hidden);
    }
    const hidden = try decoder.forward(comp, &l.state, embedded, to - from);
    _ = hidden;
    if (to == s.prompt_len) try finish(h, s, l, comp);
    try h.e.stream.synchronize();
    l.fill = to;
    s.fill_at.store(@intCast(to), .release);
    if (s.reuse.hook) |hook| for (s.reuse.marks) |at| if (at == to) {
        hook.at(hook.ptr, s, @intCast(to));
        break;
    };
    return to == s.prompt_len;
}

fn finish(h: *backend.Backend, s: *lanes.Stream, l: *backend.Lane, comp: Compute) !void {
    const logits = try decoder.logits(comp, try l.state.hiddenAt(s.prompt_len - 1), 1);
    var r = try h.request(h.a, s, l.image_rows, s.prompt_len, &.{}, s.prompt_len);
    defer h.a.free(r.history);
    if (s.grammar != null and l.first_row == null) l.first_row = try h.a.alloc(u16, c.vocab);
    r.keep = l.first_row;
    var choice: [1]u32 = undefined;
    try h.sampler.draw(comp.ops, h.e.stream, logits, &.{r}, &choice, false);
    l.first = h.streams.drawn.take(choice[0]);
}
