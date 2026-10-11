//! Shared rounds on real weights: each stream's logits in rounds with the others equal its rounds alone, byte for byte.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const Round = q.round_plan.Round;
const help = "tf-qwen27-streams --model DIR [--streams 4 --tokens 48 --context 1024]";

const Hash = u64;

fn rowHash(r: *const q.decode_round.Runner, row: usize) Hash {
    const vocab = r.model.config.vocab;
    const words = r.model.frame.get(.logits).buffer.slice(u16, (row + 1) * vocab);
    return std.hash.Wyhash.hash(0, std.mem.sliceAsBytes(words[row * vocab ..][0..vocab]));
}

fn rowArgmax(r: *const q.decode_round.Runner, row: usize) !u32 {
    const vocab = r.model.config.vocab;
    return q.session.argmax(r.model.frame.get(.logits).buffer.slice(u16, (row + 1) * vocab)[row * vocab ..][0..vocab]);
}

/// A stream alone: its greedy tokens and each position's logits hash (position 0 is the prompt's last row).
const Solo = struct { tokens: []u32, hashes: []Hash };

fn solo(a: std.mem.Allocator, r: *q.decode_round.Runner, slot: u32, prompt: []const u32, count: usize) !Solo {
    for (0..r.slots) |i| try r.reset(@intCast(i));
    const s = q.session.Session{ .runner = r, .slot = slot };
    try s.prefill(prompt, 128);
    const out = Solo{ .tokens = try a.alloc(u32, count), .hashes = try a.alloc(Hash, count) };
    for (0..count) |t| {
        out.hashes[t] = rowHash(r, 0);
        out.tokens[t] = try s.greedy();
        try s.step(out.tokens[t]);
    }
    return out;
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    var dir: ?[]const u8 = null;
    var streams: u32 = 4;
    var count: usize = 48;
    var context: u32 = 1024;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--help")) return std.debug.print("{s}\n", .{help});
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        if (std.mem.eql(u8, arg, "--model")) dir = args[i] else if (std.mem.eql(u8, arg, "--streams")) streams = try std.fmt.parseInt(u32, args[i], 10) else if (std.mem.eql(u8, arg, "--tokens")) count = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, arg, "--context")) context = try std.fmt.parseInt(u32, args[i], 10) else return error.UnknownFlag;
    }
    if (streams < 2 or streams > 8 or count < 8) return error.BadOptions;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const model = try q.model.Model.load(init.gpa, io, dir orelse return error.MissingModel, 128);
    defer model.deinit();
    const vocab: u32 = @intCast(model.config.vocab);

    // prompts of unequal lengths, past one 128-row chunk for most
    const prompts = try a.alloc([]u32, streams);
    for (prompts, 0..) |*p, k| {
        p.* = try a.alloc(u32, 61 + 47 * k);
        for (p.*, 0..) |*t, j| t.* = @intCast((j * 7919 + k * 104729 + 13) % 150000 + 1000);
    }

    // each stream alone, in a one-slot runner and in its own slot of the shared runner
    var one = try q.decode_round.Runner.init(init.gpa, model, 1, context);
    defer one.deinit();
    var shared = try q.decode_round.Runner.init(init.gpa, model, streams, context);
    defer shared.deinit();
    const refs = try a.alloc(Solo, streams);
    for (refs, prompts, 0..) |*ref, p, k| {
        ref.* = try solo(a, &one, 0, p, count);
        const again = try solo(a, &shared, @intCast(k), p, count);
        if (!std.mem.eql(Hash, ref.hashes, again.hashes)) return fail("stream {d}: its slot of a {d}-slot runner differs from a one-slot runner", .{ k, streams });
    }
    std.debug.print("alone: {d} streams equal in a one-slot and a {d}-slot runner over {d} tokens\n", .{ streams, streams, count });

    // shared rounds: every stream's window in one forward; window widths vary, odd rounds end in a wrong draft
    for (0..streams) |k| try shared.reset(@intCast(k));
    for (prompts, 0..) |p, k| try (q.session.Session{ .runner = &shared, .slot = @intCast(k) }).prefill(p, 128);
    const at = try a.alloc(usize, streams); // the position each stream's next window starts at (its pending token's)
    @memset(at, 0);
    var rounds: usize = 0;
    var compared: usize = 0;
    var ids: [8][16]u32 = undefined;
    var inputs: [8]q.round_plan.Input = undefined;
    while (true) : (rounds += 1) {
        var n: usize = 0;
        var slots: [8]u32 = undefined;
        var wrong: [8]bool = undefined;
        for (0..streams) |k| {
            const ref = refs[k];
            if (at[k] + 1 >= count) continue;
            const width = @min(1 + (rounds + k) % 5, count - at[k] - 1);
            for (0..width) |r| ids[n][r] = ref.tokens[at[k] + r];
            wrong[n] = width > 1 and (rounds + k) % 2 == 1;
            if (wrong[n]) ids[n][width - 1] = (ids[n][width - 1] + 1) % vocab;
            inputs[n] = .{ .slot = @intCast(k), .start = shared.offsets[k], .capacity = shared.capacity, .ids = ids[n][0..width] };
            slots[n] = @intCast(k);
            n += 1;
        }
        if (n == 0) break;
        var round = try Round.init(init.gpa, inputs[0..n], @intCast(model.config.conv_kernel), shared.slots, vocab);
        defer round.deinit();
        try shared.verifyHead(&round, .all);
        var paths: [8][]const u32 = undefined;
        var path_rows: [8][16]u32 = undefined;
        for (0..n) |w| {
            const k = slots[w];
            const width = inputs[w].ids.len;
            const kept = if (wrong[w]) width - 1 else width;
            // a row's logits follow its own token: rows past a wrong draft are not the stream's
            for (0..kept) |r| {
                const row = round.firsts[w] + r;
                if (rowHash(&shared, row) != refs[k].hashes[at[k] + r + 1]) return fail("stream {d}: round {d} row {d} (position {d}) differs from the stream alone", .{ k, rounds, r, at[k] + r + 1 });
                if (try rowArgmax(&shared, row) != refs[k].tokens[at[k] + r + 1]) return fail("stream {d}: round {d} row {d} draws another token", .{ k, rounds, r });
                compared += 1;
            }
            for (path_rows[w][0..kept], 0..) |*x, r| x.* = @intCast(r);
            paths[w] = path_rows[w][0..kept];
            at[k] += kept;
        }
        try shared.keep(&round, paths[0..n]);
        for (0..n) |w| if (shared.offsets[slots[w]] != prompts[slots[w]].len + at[slots[w]]) return error.OffsetDrift;
    }
    std.debug.print("shared: {d} rounds, {d} logits rows equal the streams alone (windows of 1 to 5 rows, wrong last drafts dropped)\n", .{ rounds, compared });
}

fn fail(comptime fmt: []const u8, args: anytype) error{Mismatch} {
    std.debug.print("MISMATCH: " ++ fmt ++ "\n", args);
    return error.Mismatch;
}
