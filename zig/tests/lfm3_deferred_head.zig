//! Qualification of unobserved intermediate prompt-head omission.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm3;
const c = q.config;
const st = q.state;
fn eq(a: []const u8, b: []const u8) !void {
    if (!std.mem.eql(u8, a, b)) return error.DeferredHeadByteMismatch;
}
fn state(a: *const st.Cache, b: *const st.Cache) !void {
    if (a.len != b.len) return error.CacheLengthMismatch;
    for (a.blocks, b.blocks) |x, y| switch (x) {
        .conv => |s| try eq(s.contents()[0..st.conv_bytes], y.conv.contents()[0..st.conv_bytes]),
        .attention => |k| {
            for ([_]mtl.Buffer{ k.keys, k.values }, [_]mtl.Buffer{ y.attention.keys, y.attention.values }) |ab, bb| for (0..c.kv_heads) |h| {
                const n = a.len * c.head_dim * 2;
                try eq((ab.contents() + h * a.capacity * c.head_dim * 2)[0..n], (bb.contents() + h * b.capacity * c.head_dim * 2)[0..n]);
            };
        },
    };
}
fn logits(a: *const st.Cache, b: *const st.Cache) !void {
    try eq(a.logits.contents()[0 .. c.vocab * 2], b.logits.contents()[0 .. c.vocab * 2]);
}
fn token(cache: *const st.Cache) u32 {
    const v = cache.logits.slice(u16, c.vocab);
    var best: usize = 0;
    for (v[1..], 1..) |x, i| if (@as(f32, @bitCast(@as(u32, x) << 16)) > @as(f32, @bitCast(@as(u32, v[best]) << 16))) {
        best = i;
    };
    return @intCast(best);
}
pub fn main(init: std.process.Init) !void {
    const raw_args = try init.minimal.args.toSlice(init.arena.allocator());
    const bad_final = raw_args.len > 1 and std.mem.eql(u8, raw_args[raw_args.len - 1], "--bad-final");
    const args = raw_args[0 .. raw_args.len - @intFromBool(bad_final)];
    if (args.len != 4 and args.len != 6) return error.ExpectedModelTokensOutputOptionalImageRowsStart;
    std.debug.print("DEFERRED_HEAD_BUILD default={any}\n", .{(q.backend.Options{ .capacity = 1 }).defer_prompt_head});
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const raw = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.gpa, .limited(4 << 20));
    defer init.gpa.free(raw);
    const np = try tf.npy.parse(raw);
    if (!std.mem.eql(u8, np.descr, "<u4") or np.rank != 1 or np.count() <= 257 or np.count() > 8192) return error.TokenShape;
    const ids = try init.gpa.alloc(u32, np.count());
    defer init.gpa.free(ids);
    @memcpy(std.mem.sliceAsBytes(ids), np.data);
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    var ib: ?mtl.Buffer = null;
    defer if (ib) |b| b.deinit();
    var ip: ?mtl.Pipeline = null;
    defer if (ip) |p| p.deinit();
    var span: q.image_rows.Span = undefined;
    var images: ?q.image_rows.Images = null;
    if (args.len == 6) {
        if (m.config.kind != .vision_language) return error.ExpectedVisionModel;
        const start = try std.fmt.parseInt(u32, args[5], 10);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[4], init.gpa, .limited(64 << 20));
        defer init.gpa.free(bytes);
        const rb = c.hidden * 2;
        if (bytes.len == 0 or bytes.len % rb != 0 or start > ids.len or bytes.len / rb > ids.len - start) return error.ImageShape;
        for (ids, 0..) |id, i| if ((id == 125030) != (i >= start and i < start + bytes.len / rb)) return error.ImagePlaceholderMismatch;
        ib = try m.device.buffer(bytes.len, mtl.ResourceOptions.shared);
        @memcpy(ib.?.contents()[0..bytes.len], bytes);
        ip = try q.image_rows.load(m.device);
        span = .{ .at = start, .rows = @intCast(bytes.len / rb), .source = 0 };
        images = .{ .buffer = ib.?, .pipeline = ip.?, .spans = (&span)[0..1] };
    }
    var a = try st.Cache.init(init.gpa, m.device, &m.config.layer_types, ids.len + 32);
    defer a.deinit();
    var b = try st.Cache.init(init.gpa, m.device, &m.config.layer_types, ids.len + 32);
    defer b.deinit();
    var scratch = try st.Scratch.init(init.gpa, m.device, &m.config.layer_types, 128);
    defer scratch.deinit();
    var saved: ?*q.snapshot.Snap = null;
    defer if (saved) |s| q.snapshot.drop(init.gpa, s);
    var at: usize = 0;
    var skipped: usize = 0;
    var marks: usize = 0;
    var resumed_at: usize = 0;
    const old = try init.gpa.alloc(u8, c.vocab * 2);
    defer init.gpa.free(old);
    while (at < ids.len) {
        var end = @min(ids.len, at + 128);
        for ([_]usize{ 129, 257 }) |mark| {
            if (mark > at) end = @min(end, mark);
        }
        const marked = end == 129 or end == 257;
        const head: q.forward.Head = if (bad_final and end == ids.len) .none else if (end == ids.len or marked) .last else .none;
        @memcpy(old, b.logits.contents()[0 .. c.vocab * 2]);
        try q.forward.runWithImages(m, &scratch, &.{.{ .cache = &a, .rows = end - at }}, ids[at..end], false, .last, images);
        try q.forward.runWithImages(m, &scratch, &.{.{ .cache = &b, .rows = end - at }}, ids[at..end], false, head, images);
        try state(&a, &b);
        if (head == .none) {
            try eq(old, b.logits.contents()[0 .. c.vocab * 2]);
            skipped += 1;
        } else try logits(&a, &b);
        if (marked) {
            const sa = try q.snapshot.save(init.gpa, &m.config.layer_types, &a, @intCast(end));
            defer q.snapshot.drop(init.gpa, sa);
            const sb = try q.snapshot.save(init.gpa, &m.config.layer_types, &b, @intCast(end));
            try eq(sa.bytes, sb.bytes);
            marks += 1;
            if (end == 257) {
                saved = sb;
                resumed_at = end;
            } else q.snapshot.drop(init.gpa, sb);
        }
        at = end;
    }
    try logits(&a, &b);
    // A cancelled/yielded pass keeps a completed prefix; restore and continue exactly.
    var resumed = try st.Cache.init(init.gpa, m.device, &m.config.layer_types, ids.len + 32);
    defer resumed.deinit();
    try q.snapshot.restore(&resumed, saved.?);
    at = resumed_at;
    while (at < ids.len) {
        const end = @min(ids.len, at + 128);
        try q.forward.runWithImages(m, &scratch, &.{.{ .cache = &resumed, .rows = end - at }}, ids[at..end], false, if (end == ids.len) .last else .none, images);
        at = end;
    }
    try state(&a, &resumed);
    try logits(&a, &resumed);
    var tokens: [16]u32 = undefined;
    for (&tokens) |*t| {
        t.* = token(&a);
        if (t.* != token(&b)) return error.TokenMismatch;
        try q.forward.run(m, &scratch, &.{.{ .cache = &a, .rows = 1 }}, &.{t.*}, false, .all);
        try q.forward.run(m, &scratch, &.{.{ .cache = &b, .rows = 1 }}, &.{t.*}, false, .all);
        try state(&a, &b);
        try logits(&a, &b);
    }
    if (skipped == 0) return error.NoDeferredHeads;
    if (images == null) try @import("lfm3_deferred_callbacks.zig").check(init.gpa, m, ids[0..@min(ids.len, 512)]);
    const final_snapshot = try q.snapshot.save(init.gpa, &m.config.layer_types, &a, @intCast(a.len));
    defer q.snapshot.drop(init.gpa, final_snapshot);
    var state_hash: [32]u8 = undefined;
    var logits_hash: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(final_snapshot.bytes, &state_hash, .{});
    std.crypto.hash.sha2.Sha256.hash(a.logits.contents()[0 .. c.vocab * 2], &logits_hash, .{});
    const state_hex = std.fmt.bytesToHex(state_hash, .lower);
    const logits_hex = std.fmt.bytesToHex(logits_hash, .lower);
    const out = try std.json.Stringify.valueAlloc(init.gpa, .{ .final_state_sha256 = state_hex[0..], .final_logits_sha256 = logits_hex[0..], .prompt_tokens = ids.len, .skipped_heads = skipped, .mark_snapshots_equal = marks, .resumed_prefix = resumed_at, .resume_state_logits_equal = true, .full_active_kv_conv_equal = true, .full_logits_equal = true, .decode_tokens = tokens, .vision_rows = images != null, .scope = "real checkpoint differential correctness; backend callbacks exercised separately for text fixtures; not served timing" }, .{ .whitespace = .indent_2 });
    defer init.gpa.free(out);
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[3], .data = out });
    std.debug.print("DEFERRED_HEAD_PASS {s}\n", .{out});
}
