//! Synthetic served-check failures retain emitted tokens, committed position and BF16 head words.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.qwen27;
const host = @import("qwen27_served");

pub const Reference = struct {
    states: []const [32]u8,
    logits: []const u16,
    vocab: usize,

    pub fn sameState(r: Reference, h: *host.Host, prompt: usize, emitted: usize) !bool {
        const offset = h.runner.offsets[0];
        if (offset < prompt or emitted == 0) return false;
        const fed = offset - prompt;
        if (fed + 1 < emitted or fed > emitted or fed + 1 >= r.states.len) return false;
        return std.mem.eql(u8, &r.states[fed + 1], &try (q.session.Session{ .runner = &h.runner }).fingerprint());
    }

    pub fn report(r: Reference, h: *host.Host, request: u64, prompt: usize, expected: []const u32, actual: []const u32, log: *tf.lanes.events.Log) !void {
        const state = try (q.session.Session{ .runner = &h.runner }).fingerprint();
        var matched: ?usize = null;
        for (r.states[1..], 1..) |s, i| if (std.mem.eql(u8, &s, &state)) {
            matched = i;
            break;
        };
        var first: ?usize = if (expected.len == actual.len) null else @min(expected.len, actual.len);
        for (expected[0..@min(expected.len, actual.len)], actual[0..@min(expected.len, actual.len)], 0..) |want, got, i| if (want != got) {
            first = i;
            break;
        };
        std.debug.print("SERVED_DIFF request={d} expected_tokens={d} actual_tokens={d} first_token_diff={?d} prompt={d} offset={d} pending_reference_offset={d} matching_reference_state={?d}\n", .{ request, expected.len, actual.len, first, prompt, h.runner.offsets[0], prompt + expected.len - 1, matched });
        std.debug.print("SERVED_TOKENS expected={any} actual={any}\n", .{ expected, actual });
        const values = h.model.frame.get(.logits).buffer.slice(u16, r.vocab);
        const want = r.logits[(expected.len - 1) * r.vocab ..][0..r.vocab];
        const top_want = try q.session.topTwo(want);
        const top_got = try q.session.topTwo(values);
        std.debug.print("SERVED_HEAD expected_ids={any} expected_logits={any} actual_ids={any} actual_logits={any}\n", .{ top_want.ids, top_want.logits, top_got.ids, top_got.logits });
        for (want, values, 0..) |w, v, col| if (w != v) std.debug.print("SERVED_LOGIT row=0 column={d} expected=0x{x:0>4} actual=0x{x:0>4}\n", .{ col, w, v });
        const fed = h.runner.offsets[0] - prompt;
        if (fed < r.logits.len / r.vocab) {
            const aligned = r.logits[fed * r.vocab ..][0..r.vocab];
            const top = try q.session.topTwo(aligned);
            var different: usize = 0;
            for (aligned, values) |w, v| different += @intFromBool(w != v);
            std.debug.print("SERVED_ALIGNED fed={d} reference_ids={any} reference_logits={any} differing_logits={d}/{d}\n", .{ fed, top.ids, top.logits, different, r.vocab });
        }
        for (0..log.events.items.len) |i| {
            const line = try log.line(h.gpa, i);
            defer h.gpa.free(line);
            std.debug.print("SERVED_EVENT {s}\n", .{line});
        }
    }
};
