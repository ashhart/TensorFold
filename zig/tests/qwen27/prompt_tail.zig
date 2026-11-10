//! Synthetic prompt tail check: the last layer over its last lane block keeps the kept row's logits, KV and state.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.qwen27;
const graph = @import("synthetic_graph.zig");

const Pass = struct { fingerprint: [32]u8, tokens: [4]u32, final: [32]u8, head_row: []u8, skipped_row: []u8 };

/// One prompt in chunks of `chunk` rows; `every_row` taps the last layer, a reader that refuses the tail.
fn pass(a: std.mem.Allocator, r: *q.decode_round.Runner, ids: []const u32, chunk: usize, every_row: bool) !Pass {
    const s = q.session.Session{ .runner = r };
    const saved = r.taps;
    defer {
        r.taps = saved;
        r.capture_taps = false;
    }
    if (every_row) {
        r.taps.ids[0] = r.model.config.layers - 1;
        r.taps.count = 1;
        r.capture_taps = true;
    }
    try r.reset(0);
    try s.prefill(ids, chunk);
    var out: Pass = undefined;
    out.fingerprint = try s.fingerprint();
    // the final norm's rows in the frame: the kept row heads the logits, row 0 is one the tail skips
    const h = r.model.config.hidden * 2;
    const last = (ids.len - 1) % r.promptRows(@intCast(chunk));
    const rows = r.model.frame.get(.input).buffer.contents();
    out.head_row = try a.dupe(u8, rows[last * h ..][0..h]);
    out.skipped_row = try a.dupe(u8, rows[0..h]);
    for (&out.tokens, 0..) |*token, i| {
        token.* = try s.greedy();
        if (i + 1 < out.tokens.len) try s.step(token.*);
    }
    out.final = try s.fingerprint();
    return out;
}

pub fn main(init: std.process.Init) !void {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try graph.createLayers(a, 4);
    defer m.deinit();
    // a frame past one lane block, so a chunk can take the tail
    m.frame.deinit();
    m.frame = try q.gpu_frame.Frame.init(m.device, m.config, 512);
    var runner = try q.decode_round.Runner.init(a, m, 1, 1024);
    defer runner.deinit();
    var prompt: [700]u32 = undefined;
    for (&prompt, 0..) |*id, i| id.* = @intCast((i * 17 + 7) % 96);

    var cases: usize = 0;
    var tails: usize = 0;
    var controls: usize = 0;
    for ([_]usize{ 129, 200, 256, 300, 511, 700 }) |length| for ([_]usize{ 512, 160, 129 }) |chunk| {
        const ids = prompt[0..length];
        const plain = try pass(a, &runner, ids, chunk, true);
        const tail = try pass(a, &runner, ids, chunk, false);
        // control: a row the tail skips holds other bits, so the equal kept row is the tail's own work
        const took = !std.mem.eql(u8, plain.skipped_row, tail.skipped_row);
        const last_rows = length - (length - 1) / runner.promptRows(@intCast(chunk)) * runner.promptRows(@intCast(chunk));
        if (took != (last_rows > q.prompt_tail.block)) {
            std.debug.print("length {d} chunk {d}: tail taken {} with a last chunk of {d} rows\n", .{ length, chunk, took, last_rows });
            return error.TailChoiceDiffers;
        }
        if (!std.mem.eql(u8, &plain.fingerprint, &tail.fingerprint) or !std.mem.eql(u8, plain.head_row, tail.head_row)) {
            std.debug.print("length {d} chunk {d}: prompt state, KV or last logits differ with the tail\n", .{ length, chunk });
            return error.PromptTailDiffers;
        }
        if (!std.mem.eql(u32, &plain.tokens, &tail.tokens) or !std.mem.eql(u8, &plain.final, &tail.final)) {
            std.debug.print("length {d} chunk {d}: decoded tokens or state differ after the tail\n", .{ length, chunk });
            return error.DecodeAfterTailDiffers;
        }
        if (took) {
            controls += 1;
            tails += 1;
        }
        cases += 1;
    };
    if (tails == 0) return error.NoTailTaken;
    std.debug.print("qwen27 prompt tail: {d} prompt cases byte-equal (state, KV, last logits, 4 greedy tokens), {d} took the tail, {d} controls moved\n", .{ cases, tails, controls });
}
