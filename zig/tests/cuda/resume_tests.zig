//! Prompt states: one kept inside a chunk equals one kept where a chunk ends, and a pass resumed from it a fresh pass.

const std = @import("std");
const cuda = @import("cuda");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const Sha = std.crypto.hash.sha2.Sha256;
const reuse = nemotron.reuse;

fn add(s: *Sha, gpa: std.mem.Allocator, e: *nemotron.Engine, ptr: u64, len: usize) !void {
    const host = try gpa.alloc(u8, len);
    defer gpa.free(host);
    try e.ops().download(host, ptr);
    try e.stream.synchronize();
    s.update(host);
}

/// What a prompt of `len` tokens leaves: each layer's and the head's KV rows, the Mamba states, last row and token.
fn digest(gpa: std.mem.Allocator, e: *nemotron.Engine, h: *nemotron.Head, len: usize, first: u32) ![32]u8 {
    var s = Sha.init(.{});
    const row = nemotron.state.kvRowBytes(e.c);
    for ([2]u64{ e.b.k_cache, e.b.v_cache }) |plane| for (0..e.c.count(.attention)) |l| try add(&s, gpa, e, plane + l * e.max_len * row, len * row);
    const m = nemotron.state.mambaPlanes(e.c);
    try add(&s, gpa, e, e.b.ssm, m[0]);
    try add(&s, gpa, e, e.b.conv_base, m[1]);
    for ([2]u64{ h.k_cache, h.v_cache }) |plane| try add(&s, gpa, e, plane, (len - 1) * row); // the head stops a row short
    try add(&s, gpa, e, e.b.p_hidden + (len - 1 - e.last_chunk) * e.c.hidden * 2, e.c.hidden * 2);
    try add(&s, gpa, e, e.b.p_logits, e.c.vocab * 2);
    s.update(std.mem.asBytes(&first));
    var out: [32]u8 = undefined;
    s.final(&out);
    return out;
}

/// The keep pass's saves, as the lane host makes them.
const Saver = struct {
    target: *reuse.Target,
    at: [32]u32 = undefined,
    saved: [32]*anyopaque = undefined,
    n: usize = 0,
    err: ?anyerror = null,

    fn call(ctx: *anyopaque, at: u32) void {
        const k: *Saver = @ptrCast(@alignCast(ctx));
        const s = reuse.save(k.target, null, at) catch |err| {
            k.err = err;
            return;
        };
        k.at[k.n] = at;
        k.saved[k.n] = s;
        k.n += 1;
    }
};

/// MODEL at a 16k window, the prompt in IDS_FILE (3,000 tokens or more), states kept at tiny, off-grid and grid rows.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const gpa = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try list.append(gpa, try std.fmt.parseInt(u32, t, 10));
    const ids = list.items;
    if (ids.len < 3000) return error.PromptTooShort;
    const other = try gpa.dupe(u32, ids);
    defer gpa.free(other);
    std.mem.reverse(u32, other);
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = false, .sampling = null, .segments = 1 });
    defer e.deinit();
    const h = try nemotron.Head.init(e);
    defer h.deinit();
    var target: reuse.Target = .{ .e = e, .head = h };
    _ = try reuse.ready(&target, 16 << 30); // spare Mamba buffers, as a served cache sets aside
    const len = ids.len;
    const fresh = try digest(gpa, e, h, len, try e.prefillFrom(ids, 0, null, h, null, null));
    const marks = [_]u32{ 1, 7, 1500, 2047, 2048, 2049, @intCast(len - 5), @intCast(len - 1) }; // inside chunks, and one at an end
    var saver: Saver = .{ .target = &target };
    defer for (saver.saved[0..saver.n]) |s| reuse.drop(&target, s);
    const first = try e.prefillFrom(ids, 0, null, h, null, .{ .at = &marks, .ctx = &saver, .call = Saver.call });
    if (saver.err) |err| return err;
    try check.expect(saver.n == marks.len, "{d} states kept for {d} marks", .{ saver.n, marks.len });
    const kept = try digest(gpa, e, h, len, first);
    try check.expect(std.mem.eql(u8, &fresh, &kept), "a pass that keeps states differs from one that keeps none", .{});
    for (saver.at[0..saver.n], saver.saved[0..saver.n]) |at, s| {
        try check.expect(reuse.held(s) == reuse.bytes(&target, at), "the state at {d} holds {d} bytes, the budget counts {d}", .{ at, reuse.held(s), reuse.bytes(&target, at) });
        _ = try e.prefillFrom(other, 0, null, h, null, null); // every row the prompt reads holds another prompt's
        try reuse.restore(&target, null, s);
        const got = try digest(gpa, e, h, len, try e.prefillFrom(ids, at, null, h, null, null));
        std.debug.print("RESULT resume from {d} of {d}: {d} bytes kept, {s}\n", .{ at, len, reuse.held(s), if (std.mem.eql(u8, &fresh, &got)) "equal" else "DIFFER" });
        try check.expect(std.mem.eql(u8, &fresh, &got), "the pass resumed at {d} differs from the fresh one", .{at});
    }
    const seq = try e.newSeq(); // another stream's sequence, as a later request restores into one
    defer e.freeSeq(seq);
    e.bind(seq);
    try reuse.restore(&target, null, saver.saved[2]);
    const got = try digest(gpa, e, h, len, try e.prefillFrom(ids, saver.at[2], null, h, null, null));
    try check.expect(std.mem.eql(u8, &fresh, &got), "the state at {d} restored into another sequence differs", .{saver.at[2]});
    std.debug.print("PASS resume: {d} marks, each resumed equal to the fresh pass, and in a second sequence\n", .{marks.len});
}

fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8, list: *std.ArrayList(u32)) !void {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try list.append(gpa, try std.fmt.parseInt(u32, t, 10));
}

/// The first N ids at each pattern (`;` between, `,` within): states kept inside chunks against cut ones, bytewise.
pub fn insideChunks(gpu: check.Gpu, model: []const u8, ids_path: []const u8, n_text: []const u8, patterns: []const u8) !void {
    const gpa = gpu.gpa;
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(gpa);
    try readIds(gpa, gpu.io, ids_path, &list);
    const n = try std.fmt.parseInt(usize, n_text, 10);
    if (list.items.len < n) return error.PromptTooShort;
    const ids = list.items[0..n];
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = false, .sampling = null, .segments = 1 });
    defer e.deinit();
    const h = try nemotron.Head.init(e);
    defer h.deinit();
    var target: reuse.Target = .{ .e = e, .head = h };
    _ = try reuse.ready(&target, 16 << 30); // the first marks of a chunk take spares, the rest allocate
    const fresh = try digest(gpa, e, h, n, try e.prefillFrom(ids, 0, null, h, null, null));
    var bad: usize = 0;
    var it = std.mem.splitScalar(u8, patterns, ';');
    while (it.next()) |pat| {
        var rows_buf: [32]u32 = undefined;
        var k: usize = 0;
        var ct = std.mem.tokenizeScalar(u8, pat, ',');
        while (ct.next()) |c| : (k += 1) rows_buf[k] = try std.fmt.parseInt(u32, c, 10);
        const rows = rows_buf[0..k];
        var a: Saver = .{ .target = &target };
        var b: Saver = .{ .target = &target };
        defer for ([2]*Saver{ &a, &b }) |sv| for (sv.saved[0..sv.n]) |x| reuse.drop(&target, x);
        const da = try digest(gpa, e, h, n, try e.prefillFrom(ids, 0, null, h, null, .{ .at = rows, .ctx = &a, .call = Saver.call }));
        const db = try digest(gpa, e, h, n, try e.prefillFrom(ids, 0, null, h, null, .{ .at = rows, .ctx = &b, .call = Saver.call, .cut = true }));
        for ([2]*const Saver{ &a, &b }) |sv| if (sv.err) |err| return err;
        try check.expect(a.n == k and b.n == k, "pattern {s}: {d} and {d} states kept for {d} rows", .{ pat, a.n, b.n, k });
        var inside: usize = 0;
        var differ: ?u32 = null;
        for (rows, a.saved[0..k], b.saved[0..k]) |at, sa, sb| {
            inside += @intFromBool(at % nemotron.state.prefill_rows != 0);
            const len = reuse.stateBytes(e.c, at, true);
            const ha = try gpa.alloc(u8, len);
            defer gpa.free(ha);
            const hb = try gpa.alloc(u8, len);
            defer gpa.free(hb);
            try reuse.download(&target, sa, ha);
            try reuse.download(&target, sb, hb);
            if (differ == null and !std.mem.eql(u8, ha, hb)) differ = at;
        }
        const same = differ == null and std.mem.eql(u8, &da, &fresh) and std.mem.eql(u8, &db, &fresh);
        std.debug.print("RESULT marks {d} tokens [{s}]: {d} states, {d} inside chunks, {s}\n", .{ n, pat, k, inside, if (same) "byte-equal to the cut pass, both passes equal to a plain one" else "DIFFER" });
        if (differ) |at| std.debug.print("  first state that differs: row {d}\n", .{at});
        bad += @intFromBool(!same);
    }
    try check.expect(bad == 0, "{d} patterns differ", .{bad});
    std.debug.print("PASS marks: every pattern's states byte-equal\n", .{});
}
