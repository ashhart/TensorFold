//! #1594 step 2b's token check: a restored state's greedy tokens against the Mac alone, with top-2 margins.
const std = @import("std");
const q = @import("tensorfold").qwen27;
const hs = @import("fabric").hybrid_state;
const imp = @import("import.zig");

/// One step's two best raw logits (bf16 widened, before any sampling or temperature) and their token ids.
pub const Top2 = struct { id1: u32, logit1: f32, id2: u32, logit2: f32 };

fn top2(xs: []const f32) Top2 {
    var t: Top2 = .{ .id1 = 0, .logit1 = -std.math.inf(f32), .id2 = 0, .logit2 = -std.math.inf(f32) };
    for (xs, 0..) |x, i| {
        if (x > t.logit1) {
            t.id2 = t.id1;
            t.logit2 = t.logit1;
            t.id1 = @intCast(i);
            t.logit1 = x;
        } else if (x > t.logit2) {
            t.id2 = @intCast(i);
            t.logit2 = x;
        }
    }
    return t;
}

/// `n` steps: free-running greedy (forced null) or fed `forced`; each step's top two and the taken token's logit.
const Run = struct { tokens: []u32, top: []Top2, taken: []f32 };

fn run(a: std.mem.Allocator, r: *q.decode_round.Runner, n: usize, forced: ?[]const u32) !Run {
    const out: Run = .{ .tokens = try a.alloc(u32, n), .top = try a.alloc(Top2, n), .taken = try a.alloc(f32, n) };
    for (0..n) |i| {
        const l = try imp.logits(a, r);
        defer a.free(l);
        out.top[i] = top2(l);
        out.tokens[i] = if (forced) |f| f[i] else out.top[i].id1;
        out.taken[i] = l[out.tokens[i]];
        try (q.session.Session{ .runner = r }).promptChunk(&.{out.tokens[i]}, true);
    }
    return out;
}

/// Where a restored state comes from: a file in hybrid_state's order, or the GB10's export over the real link.
const Source = union(enum) { file: []const u8, remote: struct { url: []const u8, link: []const u8, dir: []const u8 } };

/// The restored state over slot 0 with the prompt's tail prefilled, and the sha256 of its bytes as restored.
fn restore(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, src: Source) !struct { at: u32, sha: [32]u8 } {
    var st: *q.snapshot.State = undefined;
    var at: u32 = 0;
    switch (src) {
        .file => |path| {
            const file = try imp.readState(a, io, path);
            defer a.free(file);
            if (file.len < @sizeOf(hs.Header)) return error.NotAHybridState;
            var header: hs.Header = undefined;
            @memcpy(std.mem.asBytes(&header), file[0..@sizeOf(hs.Header)]);
            try header.check(hs.qwen27, ids);
            const body = file[@sizeOf(hs.Header)..];
            if (body.len != header.total) return error.HybridStateLength;
            try imp.sized(r, header.at);
            st = try q.snapshot.blank(a, r, null, header.at);
            @memcpy(st.buf.contents()[0..body.len], body);
            at = header.at;
        },
        .remote => |x| {
            const p = try @import("import_link.zig").pulledRemote(a, io, r, ids, x.url, x.link, x.dir);
            st = p.st;
            at = p.at;
            try imp.sized(r, at);
        },
    }
    defer q.snapshot.drop(a, st);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(st.buf.contents()[0..@intCast(hs.qwen27.bytes(at))], &sha, .{});
    try r.reset(0);
    try q.snapshot.restoreSlot(a, r, 0, null, st);
    try imp.feed(r, ids[at..], true);
    return .{ .at = at, .sha = sha };
}

/// `agree SOURCE N OUT.json`: the Mac's N greedy tokens, a restored state fed them, then two free-running restores.
pub fn agree(a: std.mem.Allocator, io: std.Io, r: *q.decode_round.Runner, ids: []const u32, spec: []const u8, dir: []const u8, n: usize, out_path: []const u8) !void {
    const src: Source = if (std.mem.startsWith(u8, spec, "file:")) .{ .file = spec[5..] } else if (std.mem.startsWith(u8, spec, "remote:")) blk: {
        const target = spec[7..];
        const comma = std.mem.indexOfScalar(u8, target, ',') orelse return error.Usage;
        break :blk .{ .remote = .{ .url = target[0..comma], .link = target[comma + 1 ..], .dir = dir } };
    } else return error.Usage;
    try r.reset(0);
    try imp.feed(r, ids, true);
    const mac = try run(a, r, n, null);
    const first = try restore(a, io, r, ids, src);
    const forced = try run(a, r, n, mac.tokens);
    const again = try restore(a, io, r, ids, src);
    const free1 = try run(a, r, n, null);
    const third = try restore(a, io, r, ids, src);
    const free2 = try run(a, r, n, null);

    var w: std.Io.Writer.Allocating = .init(a);
    var j: std.json.Stringify = .{ .writer = &w.writer, .options = .{} };
    try j.beginObject();
    try j.objectField("prompt_tokens");
    try j.write(ids.len);
    try j.objectField("restored_at");
    try j.write(first.at);
    try j.objectField("source");
    try j.write(spec);
    try j.objectField("steps");
    try j.write(n);
    try j.objectField("position_note");
    try j.write("step i is the reply's i-th token (0-based), at absolute position prompt_tokens + i; logits are raw, before sampling");
    try j.objectField("restore_sha256");
    try j.write(&[_][]const u8{ &std.fmt.bytesToHex(first.sha, .lower), &std.fmt.bytesToHex(again.sha, .lower), &std.fmt.bytesToHex(third.sha, .lower) });
    try j.objectField("mac_alone");
    try writeRun(&j, mac);
    try j.objectField("restored_forced");
    try writeRun(&j, forced);
    try j.objectField("forced_disagreements");
    try j.beginArray();
    var differ: usize = 0;
    for (0..n) |i| if (forced.top[i].id1 != mac.tokens[i]) {
        differ += 1;
        try j.beginObject();
        try j.objectField("step");
        try j.write(i);
        try j.objectField("mac_token");
        try j.write(mac.tokens[i]);
        try j.objectField("mac_margin");
        try j.write(mac.top[i].logit1 - mac.top[i].logit2);
        try j.objectField("restored_token");
        try j.write(forced.top[i].id1);
        try j.objectField("restored_margin_over_mac_token");
        try j.write(forced.top[i].logit1 - forced.taken[i]);
        try j.endObject();
    };
    try j.endArray();
    try j.objectField("free_runs");
    try j.beginArray();
    try writeRun(&j, free1);
    try writeRun(&j, free2);
    try j.endArray();
    const div1 = firstDiff(mac.tokens, free1.tokens);
    const repeat = firstDiff(free1.tokens, free2.tokens);
    try j.objectField("free_first_divergence_from_mac");
    try j.write(div1);
    try j.objectField("free_runs_first_divergence");
    try j.write(repeat);
    try j.endObject();
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out_path, .data = w.written() });
    const same_bytes = std.mem.eql(u8, &first.sha, &again.sha) and std.mem.eql(u8, &again.sha, &third.sha);
    std.debug.print("RESULT agree: restored {d} of {d} tokens; forced over {d} steps: {d} positions differ; free run from {s}; repeat restores {s}, their tokens {s}; to {s}\n", .{
        first.at,                                                                               ids.len,                                           n,                                                differ,
        if (div1) |d| try std.fmt.allocPrint(a, "step {d}", .{d}) else "nowhere (same tokens)", if (same_bytes) "byte-identical" else "DIFFERENT", if (repeat == null) "identical" else "DIFFERENT", out_path,
    });
}

fn firstDiff(x: []const u32, y: []const u32) ?usize {
    for (x, y, 0..) |u, v, i| if (u != v) return i;
    return null;
}

fn writeRun(j: *std.json.Stringify, x: Run) !void {
    try j.beginObject();
    try j.objectField("tokens");
    try j.write(x.tokens);
    try j.objectField("top2");
    try j.beginArray();
    for (x.top) |t| {
        try j.beginArray();
        try j.write(t.id1);
        try j.write(t.logit1);
        try j.write(t.id2);
        try j.write(t.logit2);
        try j.endArray();
    }
    try j.endArray();
    try j.endObject();
}

test "top2 takes the two best logits in order, the first of equals first" {
    const t = top2(&.{ 0.5, 3.0, -1.0, 3.0, 2.5 });
    try std.testing.expectEqual(@as(u32, 1), t.id1);
    try std.testing.expectEqual(@as(u32, 3), t.id2);
    try std.testing.expectEqual(@as(f32, 3.0), t.logit2);
    try std.testing.expectEqual(@as(?usize, 2), firstDiff(&.{ 1, 2, 3 }, &.{ 1, 2, 4 }));
    try std.testing.expectEqual(@as(?usize, null), firstDiff(&.{ 1, 2 }, &.{ 1, 2 }));
}
