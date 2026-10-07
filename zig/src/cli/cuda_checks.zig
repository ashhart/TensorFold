//! Oracle checks against the Python engine's capture: weight digests, teacher-forced windows, prompt chunks.

const std = @import("std");
const nemotron = @import("nemotron");

fn readJson(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(std.json.Value) {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(text);
    return std.json.parseFromSlice(std.json.Value, gpa, text, .{});
}

fn sha(bytes: []const u8) [64]u8 {
    var d: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &d, .{});
    return std.fmt.bytesToHex(d, .lower);
}

fn download(gpa: std.mem.Allocator, e: *nemotron.Engine, ptr: u64, len: usize) ![]u8 {
    const host = try gpa.alloc(u8, len);
    errdefer gpa.free(host);
    try e.ops().download(host, ptr);
    try e.stream.synchronize();
    return host;
}

/// Every loaded tensor's sha256 against the oracle's weights.json (the device layouts, byte for byte).
pub fn weights(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, path: []const u8) !u8 {
    const want = try readJson(gpa, io, path);
    defer want.deinit();
    var same: usize = 0;
    var bad: usize = 0;
    for (e.w.named.items) |n| {
        const host = try download(gpa, e, n.ptr, n.len);
        defer gpa.free(host);
        const got = sha(host);
        const expect = want.value.object.get(n.name) orelse {
            std.debug.print("MISSING {s}: the oracle has no tensor of this name\n", .{n.name});
            bad += 1;
            continue;
        };
        if (std.mem.eql(u8, &got, expect.string)) same += 1 else {
            std.debug.print("DIFFER {s}: {d} bytes\n", .{ n.name, n.len });
            bad += 1;
        }
    }
    var names: usize = 0;
    var it = want.value.object.iterator();
    while (it.next()) |kv| names += @intFromBool(!std.mem.endsWith(u8, kv.key_ptr.*, ".shape"));
    std.debug.print("{s} weights: {d} equal, {d} wrong, {d} loaded, {d} in the oracle\n", .{ if (bad == 0 and same == names) "PASS" else "FAIL", same, bad, e.w.named.items.len, names });
    return if (bad == 0 and same == names) 0 else 1;
}

fn dumpDir(gpa: std.mem.Allocator, io: std.Io, root: ?[]const u8, sub: []const u8) !?[]u8 {
    const r = root orelse return null;
    const dir = try std.fs.path.join(gpa, &.{ r, sub });
    try std.Io.Dir.cwd().createDirPath(io, dir);
    return dir;
}

/// Teacher-forced one-row windows from an empty state: each step's token and logits against the oracle's.
pub fn teacher(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, path: []const u8, dump: ?[]const u8) !u8 {
    const t = try readJson(gpa, io, path);
    defer t.deinit();
    const o = t.value.object;
    const tokens = o.get("tokens").?.array.items;
    const sampled = o.get("sampled").?.array.items;
    const logits = o.get("logits_sha256").?.array.items;
    const dumps = o.get("dump_steps").?.array.items;
    try e.reset();
    var bad: usize = 0;
    for (tokens, 0..) |tok, i| {
        var dir: ?[]u8 = null;
        for (dumps) |s| if (s.integer == i) {
            var buf: [16]u8 = undefined;
            dir = try dumpDir(gpa, io, dump, try std.fmt.bufPrint(&buf, "step{d:0>3}", .{i}));
        };
        defer if (dir) |d| gpa.free(d);
        var d: nemotron.Dump = .{ .gpa = gpa, .io = io, .dir = dir orelse "" };
        const got = try e.step(@intCast(tok.integer), if (dir != null) &d else null);
        const row = try download(gpa, e, e.b.logits, e.c.vocab * 2);
        defer gpa.free(row);
        const digest = sha(row);
        const ok = got == sampled[i].integer and std.mem.eql(u8, &digest, logits[i].string);
        if (!ok) {
            std.debug.print("step {d}: token {d} (oracle {d}), logits {s}\n", .{ i, got, sampled[i].integer, if (std.mem.eql(u8, &digest, logits[i].string)) "equal" else "differ" });
            bad += 1;
        }
    }
    std.debug.print("{s} teacher-forced windows: {d} of {d} steps bit-equal\n", .{ if (bad == 0) "PASS" else "FAIL", tokens.len - bad, tokens.len });
    return if (bad == 0) 0 else 1;
}

/// One named prompt's prefill (dumped per block when asked): its first token.
pub fn prefill(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, path: []const u8, name: []const u8, dump: ?[]const u8) !u8 {
    const p = try readJson(gpa, io, path);
    defer p.deinit();
    const items = (p.value.object.get(name) orelse return error.NoSuchPrompt).array.items;
    const ids = try gpa.alloc(u32, items.len);
    defer gpa.free(ids);
    for (items, ids) |v, *x| x.* = @intCast(v.integer);
    const dir = try dumpDir(gpa, io, dump, name);
    defer if (dir) |d| gpa.free(d);
    var d: nemotron.Dump = .{ .gpa = gpa, .io = io, .dir = dir orelse "" };
    const tok = try e.prefill(ids, if (dir != null) &d else null, null);
    std.debug.print("RESULT prefill {s}: {d} tokens, first token {d}\n", .{ name, ids.len, tok });
    return 0;
}

/// GPU ms a round of the serial graph and of the one-row window graph, each replayed `n` times back to back.
pub fn rounds(e: *nemotron.Engine, prompt: []const u32, n: usize) !u8 {
    const cuda = @import("cuda");
    var a = try cuda.Event.init(e.ctx.d, true);
    defer a.deinit();
    var b = try cuda.Event.init(e.ctx.d, true);
    defer b.deinit();
    for ([_]bool{ true, false, true, false }) |serial| {
        const first = try e.prefill(prompt, null, null);
        try e.upload(first);
        const g = if (serial) e.serial.? else e.windows[@intFromBool(e.sampling != null)][1].?;
        try a.record(e.stream);
        for (0..n) |_| try g.launchOn(e.stream);
        try b.record(e.stream);
        try b.synchronize();
        const ms = try cuda.Event.elapsedMs(a, b);
        std.debug.print("RESULT {s}: {d:.4} ms a round over {d} rounds\n", .{ if (serial) "serial graph" else "window graph", ms / @as(f32, @floatFromInt(n)), n });
    }
    return 0;
}

/// The raw data of a .npy file (little-endian, C order); the caller frees it.
fn npy(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]u8 {
    const path = try std.fs.path.join(gpa, &.{ dir, name });
    defer gpa.free(path);
    const all = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 28));
    defer gpa.free(all);
    if (all.len < 10 or !std.mem.eql(u8, all[1..6], "NUMPY")) return error.NotNpy;
    const v1 = all[6] == 1;
    const hlen: usize = if (v1) std.mem.readInt(u16, all[8..10], .little) else std.mem.readInt(u32, all[8..12], .little);
    const start = (if (v1) @as(usize, 10) else 12) + hlen;
    return gpa.dupe(u8, all[start..]);
}

/// Compare MTP topk and keyed draws with recorded drafts and confidence.
pub fn draws(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, dir: []const u8) !u8 {
    const cases = [_]struct { logits: []const u8, offset: u32, token: []const u8, prob: []const u8 }{
        .{ .logits = "lane/040_dense_out.npy", .offset = 0, .token = "lane/041_sample_after_a3.npy", .prob = "lane/041_sample_after_kprob.npy" },
        .{ .logits = "lane/056_dense_out.npy", .offset = 1, .token = "lane/057_sample_after_a3.npy", .prob = "lane/057_sample_after_kprob.npy" },
    };
    const unit: [17]f64 = @splat(1);
    var d = try nemotron.Drafter.init(gpa, io, e, "", false, &unit);
    defer d.deinit();
    var bad: usize = 0;
    for (cases) |c| {
        const logits = try npy(gpa, io, dir, c.logits);
        defer gpa.free(logits);
        const want_tok = try npy(gpa, io, dir, c.token);
        defer gpa.free(want_tok);
        const want_prob = try npy(gpa, io, dir, c.prob);
        defer gpa.free(want_prob);
        const got = try d.head.draw(logits, c.offset);
        const ok = std.mem.eql(u8, std.mem.asBytes(&got.token), want_tok[0..4]) and std.mem.eql(u8, std.mem.asBytes(&got.prob), want_prob[0..4]);
        bad += @intFromBool(!ok);
        std.debug.print("{s} draw at offset {d}: token {d} (oracle {d}), confidence bits {x} (oracle {x})\n", .{ if (ok) "BITEXACT" else "DIFFER", c.offset, got.token, std.mem.readInt(u32, want_tok[0..4], .little), @as(u32, @bitCast(got.prob)), std.mem.readInt(u32, want_prob[0..4], .little) });
    }
    return if (bad == 0) 0 else 1;
}

/// Replay serial tokens as drafts at widths 1-16, injecting a wrong token every third window.
pub fn widths(gpa: std.mem.Allocator, e: *nemotron.Engine, prompt: []const u32, count: usize) !u8 {
    const rows_max = nemotron.state.max_rows;
    var ref: std.ArrayList(u32) = .empty;
    defer ref.deinit(gpa);
    try ref.append(gpa, try e.prefill(prompt, null, null));
    while (ref.items.len < count) try ref.append(gpa, try e.step(ref.items[ref.items.len - 1], null));
    if (try e.prefill(prompt, null, null) != ref.items[0]) return error.FirstTokenDiffers;
    var rows_seen: [rows_max + 1]usize = @splat(0);
    var rows_bad: [rows_max + 1]usize = @splat(0);
    var at: usize = 0;
    var width: usize = 1;
    var window: usize = 0;
    while (at + 1 < ref.items.len) : (window += 1) {
        const rows = @min(width, ref.items.len - 1 - at); // every row's draw has serial's token to meet
        var ids: [rows_max]u32 = undefined;
        @memcpy(ids[0..rows], ref.items[at..][0..rows]);
        var keep = rows;
        if (window % 3 == 2 and rows > 2) { // a wrong draft: the rows from it on read a token serial never fed
            keep = rows / 2;
            ids[keep] = @intCast((ids[keep] + 1) % e.c.vocab);
        }
        try e.verify(ids[0..rows], rows, null);
        const sampled = try e.tokens();
        for (0..keep) |r| {
            rows_seen[rows] += 1;
            if (sampled[r] != ref.items[at + r + 1]) rows_bad[rows] += 1;
        }
        try e.commit(keep);
        at += keep;
        width = width % rows_max + 1;
    }
    var bad: usize = 0;
    var seen: usize = 0;
    for (1..rows_max + 1) |w| {
        bad += rows_bad[w];
        seen += rows_seen[w];
        std.debug.print("width {d}: {d} rows, {d} differ\n", .{ w, rows_seen[w], rows_bad[w] });
    }
    std.debug.print("{s} widths 1-{d}: {d} of {d} kept rows equal serial ({d} tokens, {d} windows)\n", .{ if (bad == 0) "PASS" else "FAIL", rows_max, seen - bad, seen, ref.items.len, window });
    return if (bad == 0) 0 else 1;
}
