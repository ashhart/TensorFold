//! Prompt states kept on the prompt grid: a pass resumed from each equals a fresh pass, in every row the prompt leaves.

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
    try add(&s, gpa, e, e.b.p_hidden + (len - 1) % nemotron.state.prefill_rows * e.c.hidden * 2, e.c.hidden * 2);
    try add(&s, gpa, e, e.b.p_logits, e.c.vocab * 2);
    s.update(std.mem.asBytes(&first));
    var out: [32]u8 = undefined;
    s.final(&out);
    return out;
}

/// The keep pass's saves, as the lane host makes them.
const Saver = struct {
    target: *reuse.Target,
    at: [8]u32 = undefined,
    saved: [8]*anyopaque = undefined,
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

/// MODEL at a 16k window, the prompt in IDS_FILE (6,145 tokens or more), states kept at every grid row inside it.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const gpa = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try list.append(gpa, try std.fmt.parseInt(u32, t, 10));
    const ids = list.items;
    if (ids.len <= 3 * nemotron.state.prefill_rows) return error.PromptTooShort;
    const other = try gpa.dupe(u32, ids);
    defer gpa.free(other);
    std.mem.reverse(u32, other);
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = false, .sampling = null, .segments = 1 });
    defer e.deinit();
    const h = try nemotron.Head.init(e);
    defer h.deinit();
    var target: reuse.Target = .{ .e = e, .head = h };
    const len = ids.len;
    const fresh = try digest(gpa, e, h, len, try e.prefillFrom(ids, 0, null, h, null, null));
    var marks_buf: [8]u32 = undefined;
    var n_marks: usize = 0;
    while ((n_marks + 1) * nemotron.state.prefill_rows < len and n_marks < marks_buf.len) : (n_marks += 1) marks_buf[n_marks] = @intCast((n_marks + 1) * nemotron.state.prefill_rows);
    const marks = marks_buf[0..n_marks];
    var saver: Saver = .{ .target = &target };
    defer for (saver.saved[0..saver.n]) |s| reuse.drop(&target, s);
    const first = try e.prefillFrom(ids, 0, null, h, null, .{ .at = marks, .ctx = &saver, .call = Saver.call });
    if (saver.err) |err| return err;
    try check.expect(saver.n == marks.len, "{d} states kept for {d} marks", .{ saver.n, marks.len });
    const kept = try digest(gpa, e, h, len, first);
    try check.expect(std.mem.eql(u8, &fresh, &kept), "a pass that keeps states differs from one that keeps none", .{});
    for (saver.at[0..saver.n], saver.saved[0..saver.n]) |at, s| {
        const copy: *reuse.Copy = @ptrCast(@alignCast(s));
        try check.expect(copy.buf.len == reuse.bytes(&target, at), "the state at {d} allocates {d} bytes, the budget counts {d}", .{ at, copy.buf.len, reuse.bytes(&target, at) });
        _ = try e.prefillFrom(other, 0, null, h, null, null); // every row the prompt reads holds another prompt's
        try reuse.restore(&target, null, s);
        const got = try digest(gpa, e, h, len, try e.prefillFrom(ids, at, null, h, null, null));
        std.debug.print("RESULT resume from {d} of {d}: {d} bytes kept, {s}\n", .{ at, len, copy.buf.len, if (std.mem.eql(u8, &fresh, &got)) "equal" else "DIFFER" });
        try check.expect(std.mem.eql(u8, &fresh, &got), "the pass resumed at {d} differs from the fresh one", .{at});
    }
    const seq = try e.newSeq(); // another stream's sequence, as a later request restores into one
    defer e.freeSeq(seq);
    e.bind(seq);
    try reuse.restore(&target, null, saver.saved[1]);
    const got = try digest(gpa, e, h, len, try e.prefillFrom(ids, saver.at[1], null, h, null, null));
    try check.expect(std.mem.eql(u8, &fresh, &got), "the state at {d} restored into another sequence differs", .{saver.at[1]});
    std.debug.print("PASS resume: {d} marks, each resumed equal to the fresh pass, and in a second sequence\n", .{marks.len});
}
