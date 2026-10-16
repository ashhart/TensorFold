//! kl: one precision's greedy logits rows on disk, then another's teacher-forced rows against them (KL, top-1, NLL).

const std = @import("std");
const nemotron = @import("nemotron");

/// A file: u32 rows and u32 vocab, then a record a row: the token fed next (u32) and the row's bf16 logits.
const header_bytes = 8;

fn recordBytes(vocab: usize) usize {
    return 4 + vocab * 2;
}

fn logit(row: []const u8, i: usize) f64 {
    return @floatCast(@as(f32, @bitCast(@as(u32, std.mem.readInt(u16, row[2 * i ..][0..2], .little)) << 16)));
}

fn argmax(row: []const u8, vocab: usize) usize {
    var best: usize = 0;
    for (1..vocab) |t| if (logit(row, t) > logit(row, best)) {
        best = t;
    };
    return best;
}

/// log softmax of a bf16 row in f64 into `out`.
fn logSoftmax(row: []const u8, out: []f64) void {
    var top: f64 = -std.math.inf(f64);
    for (out, 0..) |*o, t| {
        o.* = logit(row, t);
        top = @max(top, o.*);
    }
    var sum: f64 = 0;
    for (out) |o| sum += @exp(o - top);
    const z = top + @log(sum);
    for (out) |*o| o.* -= z;
}

fn download(gpa: std.mem.Allocator, e: *nemotron.Engine, ptr: u64, len: usize) ![]u8 {
    const host = try gpa.alloc(u8, len);
    errdefer gpa.free(host);
    try e.ops().download(host, ptr);
    try e.stream.synchronize();
    return host;
}

/// `dump`: `n` rows after `prompt` into the file, fed `feed` then greedy. `against`: the file's tokens fed, row by row.
pub fn run(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, prompt: []const u32, feed: []const u32, n: usize, dump: ?[]const u8, against: ?[]const u8) !u8 {
    if ((dump == null) == (against == null) or prompt.len == 0) {
        std.debug.print("kl: give --tokens and one of --dump FILE or --against FILE\n", .{});
        return 2;
    }
    const v = e.c.vocab;
    const rec = recordBytes(v);
    _ = try e.prefill(prompt, null, null);
    if (dump) |path| {
        const buf = try gpa.alloc(u8, header_bytes + n * rec);
        defer gpa.free(buf);
        std.mem.writeInt(u32, buf[0..4], @intCast(n), .little);
        std.mem.writeInt(u32, buf[4..8], @intCast(v), .little);
        for (0..n) |i| {
            const row = try download(gpa, e, if (i == 0) e.b.p_logits else e.b.logits, v * 2);
            defer gpa.free(row);
            const t: u32 = if (i < feed.len) feed[i] else @intCast(argmax(row, v));
            const at = header_bytes + i * rec;
            std.mem.writeInt(u32, buf[at..][0..4], t, .little);
            @memcpy(buf[at + 4 ..][0 .. v * 2], row);
            if (i + 1 < n) _ = try e.step(t, null);
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = buf });
        std.debug.print("RESULT kl dump: {d} rows ({d} fed) after a {d}-token prompt to {s}\n", .{ n, @min(n, feed.len), prompt.len, path });
        return 0;
    }
    const ref = try std.Io.Dir.cwd().readFileAlloc(io, against.?, gpa, .limited(1 << 34));
    defer gpa.free(ref);
    if (ref.len < header_bytes or std.mem.readInt(u32, ref[4..8], .little) != v) return error.VocabMismatch;
    const rows = @min(std.mem.readInt(u32, ref[0..4], .little), n);
    if (ref.len < header_bytes + rows * rec) return error.ShortFile;
    const lp = try gpa.alloc(f64, v);
    defer gpa.free(lp);
    const lq = try gpa.alloc(f64, v);
    defer gpa.free(lq);
    var kl_sum: f64 = 0;
    var kl_max: f64 = 0;
    var kl_at: usize = 0;
    var top1: usize = 0;
    var nll = [2]f64{ 0, 0 };
    var first_diff: ?usize = null;
    for (0..rows) |i| {
        const at = header_bytes + i * rec;
        const t = std.mem.readInt(u32, ref[at..][0..4], .little);
        const want = ref[at + 4 ..][0 .. v * 2];
        const row = try download(gpa, e, if (i == 0) e.b.p_logits else e.b.logits, v * 2);
        defer gpa.free(row);
        logSoftmax(want, lp);
        logSoftmax(row, lq);
        var kl: f64 = 0;
        for (lp, lq) |p, q| kl += @exp(p) * (p - q);
        kl_sum += kl;
        if (kl > kl_max) {
            kl_max = kl;
            kl_at = i;
        }
        const same = argmax(row, v) == argmax(want, v);
        top1 += @intFromBool(same);
        if (!same and first_diff == null) first_diff = i;
        nll[0] -= lp[t];
        nll[1] -= lq[t];
        if (i + 1 < rows) _ = try e.step(t, null);
    }
    const r: f64 = @floatFromInt(rows);
    std.debug.print("RESULT kl: {d} rows, mean KL {e:.3} nats, max {e:.3} at row {d}; top-1 {d} of {d}; first differs at {?d}; mean NLL of the fed tokens {d:.4} (reference) {d:.4} (this run)\n", .{ rows, kl_sum / r, kl_max, kl_at, top1, rows, first_diff, nll[0] / r, nll[1] / r });
    return 0;
}
