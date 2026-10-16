//! A paged sequence against the engine's own: its prompt pass, decode, and states resumed across page edges, bytewise.
const std = @import("std");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const Sha = std.crypto.hash.sha2.Sha256;
const reuse = nemotron.reuse;
const state = nemotron.state;
const Pages = @import("nemotron").pages.Pages;

/// `len` rows of the bound sequence's plane at `base` (a layer's K or V) into `s`: contiguous, or page by page.
fn rows(s: *Sha, gpa: std.mem.Allocator, e: *nemotron.Engine, p: ?*const Pages, base: u64, len: usize) !void {
    const row = state.kvRowBytes(e.c);
    const host = try gpa.alloc(u8, len * row);
    defer gpa.free(host);
    if (e.b.table == 0) try e.ops().download(host, base) else {
        var done: usize = 0;
        for (e.bound.pages) |id| {
            if (done == len) break;
            const n = @min(state.chunk_keys, len - done);
            try e.ops().download(host[done * row ..][0 .. n * row], base + id * p.?.e.pageBytes());
            done += n;
        }
    }
    try e.stream.synchronize();
    s.update(host);
}

fn bytes(s: *Sha, gpa: std.mem.Allocator, e: *nemotron.Engine, ptr: u64, len: usize) !void {
    const host = try gpa.alloc(u8, len);
    defer gpa.free(host);
    try e.ops().download(host, ptr);
    try e.stream.synchronize();
    s.update(host);
}

/// What a prompt of `len` tokens leaves on the bound sequence: KV rows, head rows, Mamba states, last row and token.
fn digest(gpa: std.mem.Allocator, e: *nemotron.Engine, h: *nemotron.Head, p: ?*const Pages, len: usize, first: u32) ![32]u8 {
    var s = Sha.init(.{});
    const row = state.kvRowBytes(e.c);
    const stride = if (e.b.table != 0) 2 * state.chunk_keys * row else e.max_len * row;
    for ([2]u64{ e.b.k_cache, e.b.v_cache }) |plane| for (0..e.c.count(.attention)) |l| try rows(&s, gpa, e, p, plane + l * stride, len);
    const m = state.mambaPlanes(e.c);
    try bytes(&s, gpa, e, e.b.ssm, m[0]);
    try bytes(&s, gpa, e, e.b.conv_base, m[1]);
    for ([2]u64{ h.k_cache, h.v_cache }) |plane| try rows(&s, gpa, e, p, plane, len - 1); // the head stops a row short
    try bytes(&s, gpa, e, e.b.p_hidden + (len - 1 - e.last_chunk) * e.c.hidden * 2, e.c.hidden * 2);
    try bytes(&s, gpa, e, e.b.p_logits, e.c.vocab * 2);
    s.update(std.mem.asBytes(&first));
    var out: [32]u8 = undefined;
    s.final(&out);
    return out;
}

/// The keep pass's saves, as the lane host makes them.
const Saver = struct {
    target: *reuse.Target,
    at: [16]u32 = undefined,
    saved: [16]*anyopaque = undefined,
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

/// `n` greedy tokens after the prompt pass, a one-row window each (a graph on the own sequence, a capture on another).
fn decode(e: *nemotron.Engine, first: u32, out: []u32) !void {
    var t = first;
    for (out) |*o| {
        t = try e.step(t, null);
        o.* = t;
    }
}

/// MODEL at a 16k window, the first 3,100 tokens of IDS_FILE: a paged sequence (pages shuffled, NaN where unwritten).
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const gpa = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var list: std.ArrayList(u32) = .empty;
    defer list.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try list.append(gpa, try std.fmt.parseInt(u32, t, 10));
    if (list.items.len < 3100) return error.PromptTooShort;
    const ids = list.items[0..3100];
    const other = try gpa.dupe(u32, ids);
    defer gpa.free(other);
    std.mem.reverse(u32, other);
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = true, .sampling = null, .segments = 1 });
    defer e.deinit();
    const h = try nemotron.Head.init(e);
    defer h.deinit();
    var target: reuse.Target = .{ .e = e, .head = h };
    _ = try reuse.slots(&target, 32);
    const want = nemotron.pages.need(nemotron.pages.layout(e), ids.len, 64);
    const p = try Pages.init(gpa, e, 2, 2 * want + 3); // two sequences, a few pages spare: ids handed out unordered
    defer p.deinit();
    p.poison = true;
    var prng = std.Random.DefaultPrng.init(7);
    prng.random().shuffle(u32, p.pool.free[0..p.pool.nfree]); // pages handed out in no order
    const marks = [_]u32{ 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049, @intCast(ids.len - 5) };
    var own_saves: Saver = .{ .target = &target };
    defer for (own_saves.saved[0..own_saves.n]) |s| reuse.drop(&target, s);
    const first = try e.prefillFrom(ids, 0, null, h, null, .{ .at = &marks, .ctx = &own_saves, .call = Saver.call });
    if (own_saves.err) |err| return err;
    const fresh = try digest(gpa, e, h, null, ids.len, first);
    var plain: [24]u32 = undefined;
    try decode(e, first, &plain);
    const seq = try p.open(ids.len, 64, false);
    e.bind(seq);
    var paged_saves: Saver = .{ .target = &target };
    defer for (paged_saves.saved[0..paged_saves.n]) |s| reuse.drop(&target, s);
    const pf = try e.prefillFrom(ids, 0, null, h, null, .{ .at = &marks, .ctx = &paged_saves, .call = Saver.call });
    if (paged_saves.err) |err| return err;
    try check.expect(std.mem.eql(u8, &fresh, &(try digest(gpa, e, h, p, ids.len, pf))), "the paged prompt pass differs from the own sequence's", .{});
    var got: [24]u32 = undefined;
    try decode(e, pf, &got);
    try check.expect(std.mem.eql(u32, &plain, &got), "the paged sequence decodes {any}, the own one {any}", .{ got, plain });
    var bad: usize = 0;
    for (own_saves.at[0..own_saves.n], own_saves.saved[0..own_saves.n], paged_saves.saved[0..own_saves.n]) |at, from_own, from_paged| {
        for ([2]*anyopaque{ from_own, from_paged }, 0..) |saved, way| { // own into paged, then paged into own
            const into = if (way == 0) seq else &e.own;
            e.bind(into);
            _ = try e.prefillFrom(other, 0, null, h, null, null); // every row the prompt reads holds another prompt's
            try reuse.restore(&target, null, saved);
            const d = try digest(gpa, e, h, if (way == 0) p else null, ids.len, try e.prefillFrom(ids, at, null, h, null, null));
            const same = std.mem.eql(u8, &fresh, &d);
            std.debug.print("RESULT resume at {d} {s}: {s}\n", .{ at, if (way == 0) "from the own sequence into a paged one" else "from a paged sequence into the own", if (same) "equal" else "DIFFER" });
            bad += @intFromBool(!same);
        }
    }
    try check.expect(bad == 0, "{d} resumed passes differ from the fresh one", .{bad});
    try check.expect(p.close(seq), "the paged sequence is not the pool's", .{});
    try check.expect(p.pool.idle() and p.pool.nfree == p.pool.total(), "the pool keeps {d} of {d} pages after the stream ended", .{ p.pool.total() - p.pool.nfree, p.pool.total() });
    std.debug.print("PASS paged: prompt pass and {d} decoded tokens equal the own sequence's; {d} states resumed both ways across page edges\n", .{ plain.len, own_saves.n });
}
