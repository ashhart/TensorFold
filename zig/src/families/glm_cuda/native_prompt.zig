//! The GLM backend's prompt pass: a stream's caches (kept or new), its images through the tower, its first token.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const forward = @import("forward.zig");
const decode = @import("decode.zig");
const proto = @import("proto.zig");
const copy = @import("copy.zig");
const be = lanes.backend;
const Writer = proto.Writer;
const native = @import("native.zig");
const max_rows = native.max_rows;
const copy_index = native.copy_index;
const Owned = native.Owned;
const fits = native.fits;
const of = native.of;

/// Cache slots a kept state is allocated beyond its own request (TF_GLM_KEEP_HEADROOM, 8192; 0: none). Read once.
pub var keep_headroom: ?i64 = null;

pub fn keepHeadroom() i64 {
    if (keep_headroom == null) keep_headroom = if (std.c.getenv("TF_GLM_KEEP_HEADROOM")) |v| @max(0, std.fmt.parseInt(i64, std.mem.span(v), 10) catch 8192) else 8192;
    return keep_headroom.?;
}

/// The stream's prompt into its caches (a kept state it extends, or new ones from the pool); draws the first token.
pub fn prefillFn(p: *anyopaque, s: *lanes.Stream) anyerror!void {
    const self = of(p);
    const ids = s.prompt();
    const cap: i64 = @intCast(ids.len + s.max_new + max_rows);
    if (ids.len == 0 or cap > self.e.capacity) return error.PromptTooLong;
    const id = self.next_id;
    self.next_id += 1;
    // images: encoded, their placeholder runs found, rank 1 sent both, before the prompt (a refusal stays here)
    var spans: std.ArrayList([2]i64) = .empty;
    defer spans.deinit(self.gpa);
    var feats: ?cuda.DeviceBuffer = null;
    defer if (feats) |*f| f.free();
    if (s.images.len > 0) feats = try encodeImages(self, ids, s.images, id, &spans);
    const media = s.images.len > 0;
    // the longest kept state this prompt extends (not with images: their features are this prompt's alone)
    var best: ?usize = null;
    if (!media) for (self.kept.items, 0..) |k, i| {
        if (k.ids.len >= ids.len or k.s.capacity < cap or !std.mem.eql(u32, k.ids, ids[0..k.ids.len])) continue;
        if (best == null or k.ids.len > self.kept.items[best.?].ids.len) best = i;
    };
    var seq: *forward.Seq = undefined;
    var start: usize = 0;
    var fresh: i64 = 0;
    const keep_at: usize = if (s.drafts and !media) @max(1, ids.len - 1) else 0;
    if (best) |i| {
        const k = self.kept.orderedRemove(i);
        seq = k.s;
        start = k.ids.len;
        self.gpa.free(k.ids);
    } else {
        // a state to be kept gets headroom when the pool has it, so the conversation's next turn resumes it
        const head: i64 = if (keep_at > 0) keepHeadroom() else 0;
        var alloc = cap;
        // (kept states give way, as in fits(): a pool full of them is not a full pool)
        if (head > 0 and cap + head <= self.e.capacity and self.used -| self.keptBytes() + self.bytesOf(cap + head) <= self.room) alloc = cap + head;
        while (self.used + self.bytesOf(alloc) > self.room) if (!try self.evictOldest()) return error.NoRoom;
        fresh = alloc;
        seq = try self.newSeq(self.next_uid, alloc);
        self.next_uid += 1;
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(seq.uid);
    try w.int(fresh);
    try w.tokens(ids);
    try proto.writeSampling(&w, s.sampling);
    try w.int(start);
    try w.int(keep_at);
    try w.int(if (media) id else 0); // the image frame this prefill takes (rank 1 drops one sent for another request)
    try self.send(.prefill, &w);
    seq.sampling = s.sampling;
    s.cached = @intCast(start);
    if (feats) |f| forward.attach(seq, .{ .spans = spans.items, .feats = f.ptr });
    const first = try decode.prefill(self.e, seq, ids, start, keep_at);
    var history: ?copy.Copy = if (self.copy_on and s.drafts) try copy.Copy.init(self.gpa, ids, copy_index, self.copy_min, self.copy_min_reply) else null;
    errdefer if (history) |*h| h.deinit(self.gpa);
    try self.lanes_by.put(s, .{ .id = id, .s = seq, .first = self.take(first), .media = media, .prompt = try self.gpa.dupe(u32, ids), .copy = history });
}

/// The prompt's images through the tower in order, and their placeholder runs; rank 1 gets both in an image frame.
pub fn encodeImages(self: *Owned, ids: []const u32, images: []const lanes.Image, id: u64, spans: *std.ArrayList([2]i64)) !cuda.DeviceBuffer {
    const tower = &(self.tower orelse return error.NoVision);
    var i: usize = 0;
    while (i < ids.len) : (i += 1) {
        if (ids[i] != self.image_token) continue;
        var j = i;
        while (j < ids.len and ids[j] == self.image_token) j += 1;
        try spans.append(self.gpa, .{ @intCast(i), @intCast(j) });
        i = j;
    }
    if (spans.items.len != images.len) return error.ImageCountMismatch;
    var rows: usize = 0;
    for (images, spans.items) |im, sp| {
        const n: usize = @intCast(@divExact(im.grid[0] * im.grid[1] * im.grid[2], 4));
        if (n != sp[1] - sp[0]) return error.ImageTokensMismatch;
        rows += n;
    }
    var feats = try cuda.DeviceBuffer.alloc(self.ctx.d, rows * 4096 * 2);
    errdefer feats.free();
    var at: usize = 0;
    for (images, spans.items) |im, sp| {
        try tower.encode(&self.e.k, im.patches, im.grid, feats.ptr + at * 4096 * 2);
        at += @intCast(sp[1] - sp[0]);
    }
    var w: Writer = .{ .gpa = self.gpa };
    defer w.buf.deinit(self.gpa);
    try w.int(id);
    try w.int(spans.items.len);
    for (spans.items) |sp| {
        try w.int(sp[0]);
        try w.int(sp[1]);
    }
    try w.int(rows);
    const fat = w.buf.items.len;
    try w.buf.resize(self.gpa, fat + rows * 4096 * 2);
    try self.ctx.d.check(self.ctx.d.api.cuMemcpyDtoH_v2(w.buf.items[fat..].ptr, feats.ptr, rows * 4096 * 2), "image features");
    try self.send(.image, &w);
    return feats;
}

pub fn imageToken(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !u32 {
    const path = try std.fs.path.join(gpa, &.{ dir, "config.json" });
    defer gpa.free(path);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 20));
    defer gpa.free(text);
    const cfg = try std.json.parseFromSlice(std.json.Value, gpa, text, .{});
    defer cfg.deinit();
    const v = cfg.value.object.get("image_token_id") orelse return error.NoImageToken;
    return @intCast(v.integer);
}
