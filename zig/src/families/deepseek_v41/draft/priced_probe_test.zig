//! A stream that went plain comes back to drafting once its text is predictable again: the probe's pass sees it.
const std = @import("std");
const pt = @import("priced_test.zig");
const priced = @import("priced.zig");

const t_ = std.testing;

/// Unpredictable text until position 60 (the prompts are 20 tokens), predictable after it.
fn turns(_: usize, pos: u64) f64 {
    return if (pos < 60) 0.1 else 0.95;
}

test "(d) plain while the text is unpredictable, a probe, then drafts again; replies == serial, probes within 1%" {
    for ([_]usize{ 1, 2 }) |n| {
        const lens = [_]u32{ 200, 200 };
        const r = try pt.run(pt.four[0..n], lens[0..n], .{ .chance = turns });
        defer r.deinit();
        try pt.expectSerial(r, pt.four[0..n], lens[0..n], null);
        const s = r.stats;
        try t_.expect(s.plain >= 20); // it went plain
        try t_.expect(s.probes >= 1); // a probe asked again
        try t_.expect(s.passes >= 2 + s.probes + 5); // and drafting resumed, round after round
        for (r.drafted) |dr| try t_.expect(dr >= 50);
        try t_.expect(s.low > 1.0 and s.below == 0);
        try t_.expect(s.probe_ms <= priced.probe_share * s.plain_total_ms + 12.0);
    }
}
