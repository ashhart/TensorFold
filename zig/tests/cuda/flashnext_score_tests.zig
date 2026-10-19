//! Decisions' label scores against the prompt's own last row from a plain pass, byte for byte.
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("check.zig");
const prompt = flash.prompt;

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
    // a short prompt, one past a pass chunk's attention block, and one past 2,048 keys (sparse selection)
    for ([_]usize{ 16, 300, 2600 }) |length| {
        const n = @min(length, base.items.len);
        const ids = base.items[0..n];
        var plain = try e.newSeq(n);
        defer plain.deinit(gpu.gpa);
        try e.forward(flash.Tri, tri, &plain, ids, .{});
        const best = prompt.argmax(plain.logits);
        const wanted = [_]u32{ 13, best, 220, 1, @intCast(prompt.vocabulary - 1) };
        var got: [wanted.len]f64 = undefined;
        const lse = try flash.score.labels(flash.Tri, e, tri, n, ids, &wanted, &got, true);
        const row = std.mem.bytesAsSlice(u16, plain.logits);
        var top: f64 = -std.math.inf(f64);
        for (row) |b| top = @max(top, @as(f64, flash.mtp.promote(b)));
        var sum: f64 = 0;
        for (row) |b| sum += @exp(@as(f64, flash.mtp.promote(b)) - top);
        for (wanted, got) |t, v| {
            const want: f64 = flash.mtp.promote(row[t]);
            try check.expect(@as(u64, @bitCast(v)) == @as(u64, @bitCast(want)), "label {d}'s logit is the plain pass's row value ({d} vs {d})", .{ t, v, want });
        }
        try check.expect(lse == top + @log(sum), "the logsumexp is the plain row's ({d} vs {d})", .{ lse, top + @log(sum) });
        var best_label: usize = 0;
        for (got, 0..) |v, i| if (v > got[best_label]) {
            best_label = i;
        };
        try check.expect(wanted[best_label] == best, "the best label is the greedy token", .{});
        check.pass("decision scores at {d} prompt tokens: {d} label logits and the logsumexp equal the plain pass's last row", .{ n, wanted.len });
    }
}
