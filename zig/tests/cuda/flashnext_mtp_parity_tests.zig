//! A shared round's streams' MTP heads in one pass against each stream's own pass, bit for bit: greedy, sampled, mixed.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;
const mtp = flash.mtp;

/// The streams' rows in one shared pass: two one-row heads, four 16-row paths, and both together.
const shapes = [_][]const usize{ &.{ 1, 1 }, &.{ 16, 16, 16, 16 }, &.{ 1, 1, 16, 16, 16, 16 } };

/// How a shape's streams draw: every stream greedy, every stream sampled under its own seed, or alternate lanes.
const Mode = enum { greedy, sampled, mixed };

/// Stream i's rule: greedy (null), or a keyed draw under its own seed.
fn rule(mode: Mode, i: usize) ?mtp.Sampling {
    const sampled = switch (mode) {
        .greedy => false,
        .sampled => true,
        .mixed => i % 2 == 1,
    };
    return if (sampled) .{ .seed = 19 + @as(u64, i), .temperature = 1.0, .top_k = 20, .top_p = 0.95 } else null;
}

/// Stream i's first draft position: the greedy shapes keep position 0, the others each their own.
fn position(mode: Mode, i: usize) u64 {
    return if (mode == .greedy) 0 else 5 + 3 * @as(u64, i);
}

/// The streams one side holds: a sequence and a draft each.
const Side = struct {
    seqs: [6]prompt.Seq = undefined,
    drafts: [6]mtp.Draft = undefined,
    made: usize = 0,

    fn deinit(s: *Side, gpa: std.mem.Allocator) void {
        for (s.seqs[0..s.made], s.drafts[0..s.made]) |*q, *d| {
            d.deinit();
            q.deinit(gpa);
        }
    }
};

/// The head's logits of two drafts, byte for byte.
fn sameLogits(gpu: check.Gpu, a: *mtp.Draft, b: *mtp.Draft) !bool {
    const x = try check.download(gpu, a.dlogits);
    defer gpu.gpa.free(x);
    const y = try check.download(gpu, b.dlogits);
    defer gpu.gpa.free(y);
    return std.mem.eql(u8, x, y);
}

pub fn run(gpu: check.Gpu, dir: []const u8, ids_path: []const u8) !void {
    const source = try std.Io.Dir.cwd().readFileAlloc(gpu.io, ids_path, gpu.gpa, .limited(1 << 20));
    defer gpu.gpa.free(source);
    var base: std.ArrayList(u32) = .empty;
    defer base.deinit(gpu.gpa);
    var words = std.mem.tokenizeAny(u8, source, " \r\n\t");
    while (words.next()) |word| try base.append(gpu.gpa, try std.fmt.parseInt(u32, word, 10));
    if (base.items.len < 16) return error.BadPrompt;
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    var glue = try flash.Glue.load(gpu.d);
    defer glue.deinit();
    const tri: flash.Tri = .{ .s = stream, .own = &glue };
    const e = try prompt.Engine.init(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, dir);
    defer e.deinit();
    const h = try mtp.Head.init(gpu.gpa, gpu.io, @constCast(gpu.d), &stream, dir);
    defer h.deinit();
    const ops: mtp.Ops(flash.Tri) = .{ .e = e, .tri = tri, .h = h };
    const ids = try gpu.gpa.alloc(u32, @min(base.items.len, 64));
    defer gpu.gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = base.items[i % base.items.len];
    // no confidence stop (the whole chain), then a stop above every probability (each chain keeps one token)
    for ([_]f64{ 0, 2 }) |confidence| {
        for (shapes) |widths| try shape(gpu, e, ops, &stream, ids, widths, confidence, .greedy);
    }
    // sampled streams under their own seeds and positions, alone and beside greedy ones; a stop some draws fall under
    for ([_]f64{ 0, 0.5 }) |confidence| {
        for (shapes) |widths| {
            try shape(gpu, e, ops, &stream, ids, widths, confidence, .sampled);
            try shape(gpu, e, ops, &stream, ids, widths, confidence, .mixed);
        }
    }
}

/// One shape's streams drafted alone and in one shared pass: the logits, the proposals, the probabilities and the stop.
fn shape(gpu: check.Gpu, e: *prompt.Engine, ops: mtp.Ops(flash.Tri), stream: *cuda.Stream, ids: []const u32, widths: []const usize, confidence: f64, mode: Mode) !void {
    const gpa = gpu.gpa;
    const cap = ids.len + 64;
    const count: usize = if (mode == .greedy) 2 else 4;
    const n = widths.len;
    var alone: Side = .{};
    defer alone.deinit(gpa);
    var together: Side = .{};
    defer together.deinit(gpa);
    var nexts: [6][16]u32 = undefined;
    var rows: [6]u64 = undefined;
    // each stream its own prompt (so its own positions) and its own draft rows
    for (widths, 0..) |w, i| {
        const prompt_len = ids.len - 3 * i;
        for (nexts[i][0..w], 0..) |*t, r| t.* = @intCast((37 + 101 * i + 17 * r) % 50_000);
        inline for (.{ &alone, &together }) |side| {
            side.seqs[i] = try e.newSeq(cap);
            side.drafts[i] = mtp.Draft.init(gpa, e.driver, stream.*, cap, ops.h.ids.len, e.chunk) catch |err| {
                side.seqs[i].deinit(gpa);
                return err;
            };
            side.made += 1;
            try ops.prefill(&side.seqs[i], &side.drafts[i], ids[0..prompt_len]);
        }
        rows[i] = alone.drafts[i].prompt_last;
    }
    // each stream's head alone: its own pass, its own chain, its own rule and position
    var alone_chains: [6]mtp.Chain = undefined;
    for (widths, 0..) |w, i| {
        try ops.draft(&alone.drafts[i], nexts[i][0..w], rows[i], position(mode, i), count, rule(mode, i), confidence, &alone_chains[i]);
    }
    // the same heads in one shared pass: one lane a stream
    var ds: [6]*mtp.Draft = undefined;
    var nexts_s: [6][]const u32 = undefined;
    var rows_s: [6]u64 = undefined;
    var counts: [6]usize = undefined;
    var positions: [6]u64 = undefined;
    var sms: [6]?mtp.Sampling = undefined;
    var chains: [6]mtp.Chain = undefined;
    var outs: [6]*mtp.Chain = undefined;
    for (widths, 0..) |w, i| {
        ds[i] = &together.drafts[i];
        nexts_s[i] = nexts[i][0..w];
        rows_s[i] = rows[i];
        counts[i] = count;
        positions[i] = position(mode, i);
        sms[i] = rule(mode, i);
        outs[i] = &chains[i];
    }
    try ops.draftMany(ds[0..n], nexts_s[0..n], rows_s[0..n], counts[0..n], positions[0..n], sms[0..n], confidence, outs[0..n]);
    try stream.synchronize();
    for (widths, 0..) |_, i| {
        try check.expect(try sameLogits(gpu, &alone.drafts[i], &together.drafts[i]), "a shared head's logits equal its lone pass's", .{});
        try check.expect(alone_chains[i].n == chains[i].n, "a shared chain's length equals its lone one's", .{});
        try check.expect(std.mem.eql(u32, alone_chains[i].tokens[0..alone_chains[i].n], chains[i].tokens[0..chains[i].n]), "a shared chain's proposals equal its lone one's", .{});
        try check.expect(std.mem.eql(f64, alone_chains[i].probs[0..alone_chains[i].n], chains[i].probs[0..chains[i].n]), "a shared chain's probabilities equal its lone one's", .{});
    }
    // the planted control: one stream's gathered row lands on another's (what a broken gather leaves)
    if (n > 1) {
        const bytes = together.drafts[0].dlogits.ptr;
        const size = together.drafts[0].dlogits.len;
        try check.expect(!(try sameLogits(gpu, &together.drafts[0], &together.drafts[1])), "the streams' heads differ before the control", .{});
        try e.driver.check(e.driver.api.cuMemcpyDtoD_v2(together.drafts[1].dlogits.ptr, bytes, size), "cuMemcpyDtoD");
        try stream.synchronize();
        try check.expect(!(try sameLogits(gpu, &alone.drafts[1], &together.drafts[1])), "the planted broken gather is caught", .{});
    }
    check.pass("a shared MTP head over {d} {s} streams ({any} rows) equals each stream's own pass: logits, proposals, probabilities and the stop", .{ n, @tagName(mode), widths });
}
