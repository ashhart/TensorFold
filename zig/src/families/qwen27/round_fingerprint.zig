//! The byte identity of a queued Qwen27 round.
const std = @import("std");
const Round = @import("round_plan.zig").Round;

pub fn fingerprint(round: *const Round) [32]u8 {
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.sliceAsBytes(round.ids));
    hash.update(std.mem.sliceAsBytes(round.slots));
    hash.update(std.mem.sliceAsBytes(round.feed_positions));
    hash.update(std.mem.sliceAsBytes(round.draw_positions));
    hash.update(std.mem.sliceAsBytes(round.firsts));
    for (round.windows) |window| {
        hash.update(std.mem.asBytes(&window.start));
        hash.update(std.mem.asBytes(&window.taps));
        hash.update(std.mem.sliceAsBytes(window.parents));
        hash.update(std.mem.sliceAsBytes(window.conv));
    }
    var out: [32]u8 = undefined;
    hash.final(&out);
    return out;
}
