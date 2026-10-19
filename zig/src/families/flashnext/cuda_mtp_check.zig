//! The drafted check (flashnext-prompt with TF_FLASHNEXT_DRAFTS): serial and drafted decoding, token for token.

const std = @import("std");
const cuda = @import("cuda");
const wt = @import("cuda_prompt_wt.zig");
const prompt = @import("cuda_prompt.zig");
const mtp = @import("cuda_mtp.zig");

const vocab = wt.vocab;
const max_chain = mtp.max_chain;
const max_rows = mtp.max_rows;
const default_confidence = mtp.default_confidence;
const Head = mtp.Head;
const Draft = mtp.Draft;
const Chain = mtp.Chain;
const Ops = mtp.Ops;
const Sampling = mtp.Sampling;
const drawRow = mtp.drawRow;

/// What a drafted decode did: rounds, drafts offered and kept, and its wall time.
pub const Stats = struct {
    rounds: usize = 0,
    drafted: usize = 0,
    accepted: usize = 0,
    ms: f64 = 0,
    /// Wall time in the windows (forward and draws) and in the head (rollback, absorb, chain).
    verify_ms: f64 = 0,
    draft_ms: f64 = 0,
};

/// Python mtp_decode over a prefilled sequence: `out` holds the first token; decode until `count` tokens.
pub fn decode(comptime Tri: type, o: Ops(Tri), seq: *prompt.Seq, d: *Draft, out: *std.ArrayList(u32), count: usize, depth: usize, confidence: f64, sm: ?Sampling, stop: []const u32) !Stats {
    const gpa = o.e.gpa;
    var st: Stats = .{};
    var ch: Chain = .{};
    var sampled: [max_rows]u32 = undefined;
    var ids: [max_rows]u32 = undefined;
    const t0 = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
    const isStop = struct {
        fn f(list: []const u32, t: u32) bool {
            return std.mem.indexOfScalar(u32, list, t) != null;
        }
    }.f;
    try o.draft(d, out.items[out.items.len - 1 ..], d.prompt_last, seq.fed + 1, @min(depth, count - out.items.len), sm, confidence, &ch);
    while (out.items.len < count and !isStop(stop, out.items[out.items.len - 1])) {
        ids[0] = out.items[out.items.len - 1];
        @memcpy(ids[1..][0..ch.n], ch.slice());
        const rows = 1 + ch.n;
        const tv = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
        try o.verify(seq, d, ids[0..rows]);
        try o.draws(d, sm, &.{}, sampled[0..rows]);
        const td = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
        st.verify_ms += @as(f64, @floatFromInt(td - tv)) / 1e6;
        defer st.draft_ms += @as(f64, @floatFromInt(std.Io.Clock.awake.now(o.e.io).toNanoseconds() - td)) / 1e6;
        var keep: usize = 1;
        for (ch.slice(), 0..) |t, i| {
            if (sampled[i] != t or isStop(stop, sampled[i])) break;
            keep += 1;
        }
        try o.rollback(seq, d, keep);
        st.rounds += 1;
        st.drafted += ch.n;
        st.accepted += keep - 1;
        const take = @min(keep, count - out.items.len);
        try out.appendSlice(gpa, sampled[0..take]);
        if (out.items.len >= count or isStop(stop, out.items[out.items.len - 1])) break;
        try o.draft(d, sampled[0..keep], d.streams.ptr, seq.fed + 1, @min(depth, count - out.items.len), sm, confidence, &ch);
    }
    st.ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(o.e.io).toNanoseconds() - t0)) / 1e6;
    return st;
}

/// Serial decoding through the same pass and draws: one row a forward (the reference drafted decoding must equal).
pub fn serial(comptime Tri: type, o: Ops(Tri), seq: *prompt.Seq, out: *std.ArrayList(u32), count: usize, sm: ?Sampling, stop: []const u32) !f64 {
    const gpa = o.e.gpa;
    const t0 = std.Io.Clock.awake.now(o.e.io).toNanoseconds();
    const row = try gpa.alloc(u16, vocab);
    defer gpa.free(row);
    while (out.items.len < count and std.mem.indexOfScalar(u32, stop, out.items[out.items.len - 1]) == null) {
        var one = [_]u32{out.items[out.items.len - 1]};
        try o.e.predict(Tri, o.tri, seq, &one);
        for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
        try out.append(gpa, try drawRow(gpa, row, null, seq.fed, sm));
    }
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(o.e.io).toNanoseconds() - t0)) / 1e6;
}

fn envUsize(name: [:0]const u8, default: usize) usize {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch default;
}

fn envF64(name: [:0]const u8, default: f64) f64 {
    const v = std.c.getenv(name) orelse return default;
    return std.fmt.parseFloat(f64, std.mem.span(v)) catch default;
}

/// The drafted check (TF_FLASHNEXT_DRAFTS): serial and drafted decoding of `count` tokens a prompt, token for token.
pub fn check(comptime Tri: type, gpa: std.mem.Allocator, io: std.Io, driver: *cuda.Driver, stream: *cuda.Stream, tri: Tri, model_dir: []const u8, prompts: []const []const u32, count: usize) !bool {
    var e = try prompt.Engine.init(gpa, io, driver, stream, model_dir);
    defer e.deinit();
    const h = try Head.init(gpa, io, driver, stream, model_dir);
    defer h.deinit();
    const o: Ops(Tri) = .{ .e = e, .tri = tri, .h = h };
    const depth = envUsize("TF_FLASHNEXT_DRAFTS", mtp.default_depth);
    const confidence = envF64("TF_FLASHNEXT_CONFIDENCE", default_confidence);
    const sm: ?Sampling = if (std.c.getenv("TF_FLASHNEXT_SEED")) |seed| .{
        .seed = std.fmt.parseInt(u64, std.mem.span(seed), 10) catch 0,
        .temperature = envF64("TF_FLASHNEXT_TEMPERATURE", 1.0),
        .top_p = envF64("TF_FLASHNEXT_TOP_P", 0.95),
        .top_k = @intCast(envUsize("TF_FLASHNEXT_TOP_K", 20)),
    } else null;
    const repeat = envUsize("TF_FLASHNEXT_REPEAT", 1);
    const stop: []const u32 = &.{};
    var same = true;
    for (prompts, 0..) |ids, pi| {
        const cap = ids.len + count + max_rows;
        var seq = try e.newSeq(cap);
        defer seq.deinit(gpa);
        var d = try Draft.init(gpa, driver, stream.*, cap, h.ids.len, e.chunk);
        defer d.deinit();
        for (0..repeat) |rep| {
            // serial: the plain prompt pass, then one row a forward
            seq.fed = 0;
            seq.history = .{ e.eos, e.eos };
            try resetSeq(&seq, stream.*);
            var ref: std.ArrayList(u32) = .empty;
            defer ref.deinit(gpa);
            try e.predict(Tri, tri, &seq, ids);
            const row = try gpa.alloc(u16, vocab);
            defer gpa.free(row);
            for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
            try ref.append(gpa, try drawRow(gpa, row, null, seq.fed, sm));
            const serial_ms = try serial(Tri, o, &seq, &ref, count, sm, stop);
            // drafted: the prompt pass with its streams and the head's cache, then windows
            seq.fed = 0;
            seq.history = .{ e.eos, e.eos };
            try resetSeq(&seq, stream.*);
            var got: std.ArrayList(u32) = .empty;
            defer got.deinit(gpa);
            try o.prefill(&seq, &d, ids);
            for (row, 0..) |*v, i| v.* = std.mem.readInt(u16, seq.logits[2 * i ..][0..2], .little);
            try got.append(gpa, try drawRow(gpa, row, null, seq.fed, sm));
            const st = try decode(Tri, o, &seq, &d, &got, count, depth, confidence, sm, stop);
            const equal = std.mem.eql(u32, ref.items, got.items);
            same = same and equal;
            const n_ref: f64 = @floatFromInt(ref.items.len - 1);
            const n_got: f64 = @floatFromInt(got.items.len - 1);
            std.debug.print("prompt {d} ({d} ids) run {d}: serial {d:.2} tok/s, drafted {d:.2} tok/s ({d:.2}x), {d} rounds, {d} drafts, {d} kept ({d:.1}%), {d:.2} tokens a round: {s}\n", .{
                pi,                                                                                                              ids.len,                                                                rep,
                n_ref / (serial_ms / 1e3),                                                                                       n_got / (st.ms / 1e3),                                                  (n_got / (st.ms / 1e3)) / (n_ref / (serial_ms / 1e3)),
                st.rounds,                                                                                                       st.drafted,                                                             st.accepted,
                if (st.drafted > 0) 100.0 * @as(f64, @floatFromInt(st.accepted)) / @as(f64, @floatFromInt(st.drafted)) else 0.0, if (st.rounds > 0) n_got / @as(f64, @floatFromInt(st.rounds)) else 0.0, if (equal) "same tokens" else "DIFFER",
            });
            std.debug.print("  drafted rounds: {d:.2} ms a window ({d:.2} rows), {d:.2} ms in the head a round\n", .{ st.verify_ms / @as(f64, @floatFromInt(@max(st.rounds, 1))), @as(f64, @floatFromInt(st.rounds + st.drafted)) / @as(f64, @floatFromInt(@max(st.rounds, 1))), st.draft_ms / @as(f64, @floatFromInt(@max(st.rounds, 1))) });
            std.debug.print("serial ", .{});
            for (ref.items) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\ndrafted", .{});
            for (got.items) |t| std.debug.print(" {d}", .{t});
            std.debug.print("\n", .{});
        }
    }
    return same;
}

/// A sequence's caches back to empty (the check reruns a prompt on one allocation).
fn resetSeq(seq: *prompt.Seq, stream: cuda.Stream) !void {
    for (seq.conv) |*b| try b.fill8(0, stream.handle);
    for (seq.state) |*b| try b.fill8(0, stream.handle);
    try seq.tail.fill8(0, stream.handle);
}
