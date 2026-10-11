//! A prompt pass keeping states inside chunks and where one ends, and passes resumed from each, against one uncut pass.

const std = @import("std");
const nemotron = @import("nemotron");
const check = @import("check.zig");

const R = nemotron.state.prefill_rows;

/// What a pass leaves: caches and states, the head's caches, the last logits and row's hidden state, the token.
const Digest = struct { state: [32]u8, head: [32]u8, logits: [32]u8, hidden: [32]u8, token: u32 };

const Kept = struct {
    target: *nemotron.reuse.Target,
    at: std.ArrayList(u32) = .empty,
    saved: std.ArrayList(*anyopaque) = .empty,
    gpa: std.mem.Allocator,
    failed: bool = false,

    fn call(ctx: *anyopaque, at: u32) void {
        const k: *Kept = @ptrCast(@alignCast(ctx));
        const s = nemotron.reuse.save(k.target, null, at) catch {
            k.failed = true;
            return;
        };
        k.saved.append(k.gpa, s) catch {
            nemotron.reuse.drop(k.target, s);
            k.failed = true;
            return;
        };
        k.at.append(k.gpa, at) catch {
            _ = k.saved.pop();
            nemotron.reuse.drop(k.target, s);
            k.failed = true;
        };
    }
};

/// At a 16k window: IDS_FILE uncut, then keeping states on and off chunk ends, then resumed from each keep.
pub fn run(gpu: check.Gpu, model: []const u8, ids_path: []const u8) !void {
    const gpa = gpu.gpa;
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpa, .limited(64 << 20));
    defer gpa.free(text);
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(gpa);
    var it = std.mem.tokenizeAny(u8, text, ", \n");
    while (it.next()) |t| try ids.append(gpa, try std.fmt.parseInt(u32, t, 10));
    const n = ids.items.len;
    try check.expect(n > R + 64, "prompt-cuts: the prompt needs more than {d} tokens, has {d}", .{ R + 64, n });
    const e = try nemotron.Engine.init(gpa, gpu.io, gpu.ctx, model, null, .{ .context = 16384, .mtp = true, .graphs = true, .sampling = null, .segments = 1 });
    defer e.deinit();
    const h = try nemotron.Head.init(e);
    defer h.deinit();
    try h.capture();
    const plain = try digest(gpa, e, h, try e.prefill(ids.items, null, h), n - 1 - e.last_chunk);

    const marks = [_]u32{ 1, 777, R - 1, R, R + 1, @intCast(n - 33), @intCast(n - 1) };
    var target: nemotron.reuse.Target = .{ .e = e, .head = h };
    var kept: Kept = .{ .target = &target, .gpa = gpa };
    defer {
        for (kept.saved.items) |s| nemotron.reuse.drop(&target, s);
        kept.saved.deinit(gpa);
        kept.at.deinit(gpa);
    }
    const keep: nemotron.engine.Keep = .{ .at = &marks, .ctx = &kept, .call = Kept.call };
    const cut = try digest(gpa, e, h, try e.prefillFrom(ids.items, 0, null, h, null, keep), n - 1 - e.last_chunk);
    try check.expect(!kept.failed and kept.at.items.len == marks.len, "prompt-cuts: {d} of {d} keeps saved", .{ kept.at.items.len, marks.len });
    try check.expect(std.meta.eql(cut, plain), "prompt-cuts: a pass keeping at {any} differs from the uncut pass", .{marks});
    check.pass("prompt-cuts: a {d}-token pass keeping at {any} equals the uncut pass", .{ n, marks });

    for (kept.at.items, kept.saved.items) |at, s| {
        try nemotron.reuse.restore(&target, null, s);
        const got = try digest(gpa, e, h, try e.prefillFrom(ids.items, at, null, h, null, null), n - 1 - e.last_chunk);
        try check.expect(std.meta.eql(got, plain), "prompt-cuts: the pass resumed at {d} differs from the uncut pass", .{at});
    }
    check.pass("prompt-cuts: passes resumed at each keep equal the uncut pass", .{});
}

fn digest(gpa: std.mem.Allocator, e: *nemotron.Engine, h: *nemotron.Head, token: u32, last: usize) !Digest {
    const Sha = std.crypto.hash.sha2.Sha256;
    var st = Sha.init(.{});
    for (e.b.planes(), e.b.state_bytes) |p, len| try hashDevice(&st, gpa, e, p, len);
    var hd = Sha.init(.{});
    const sizes = h.seqSizes();
    try hashDevice(&hd, gpa, e, h.k_cache, sizes[0]);
    try hashDevice(&hd, gpa, e, h.v_cache, sizes[1]);
    var lg = Sha.init(.{});
    try hashDevice(&lg, gpa, e, e.b.p_logits, e.c.vocab * 2);
    var hi = Sha.init(.{});
    try hashDevice(&hi, gpa, e, e.b.p_hidden + last * @as(u64, e.c.hidden) * 2, e.c.hidden * 2);
    return .{ .state = st.finalResult(), .head = hd.finalResult(), .logits = lg.finalResult(), .hidden = hi.finalResult(), .token = token };
}

fn hashDevice(h: *std.crypto.hash.sha2.Sha256, gpa: std.mem.Allocator, e: *nemotron.Engine, ptr: u64, len: usize) !void {
    const piece = 64 << 20;
    const host = try gpa.alloc(u8, @min(len, piece));
    defer gpa.free(host);
    var at: usize = 0;
    while (at < len) : (at += piece) {
        const m = @min(piece, len - at);
        try e.ops().download(host[0..m], ptr + at);
        try e.stream.synchronize();
        h.update(host[0..m]);
    }
}
