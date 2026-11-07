//! --mtp-drafts and --mtp-confidence on a Mac: Qwen3.8-27B's MTP head, opt-in, exclusive with DFlash2.
const std = @import("std");

/// Drafts a chain holds at most: a verify window's rows less the pending token.
pub const max_drafts = 15;

/// Why the flags are refused for this checkpoint (`target`: Qwen3.8-27B), or null.
pub fn refusal(drafts: ?u32, confidence: ?f64, drafter: bool, target: bool) ?[]const u8 {
    if (drafts == null and confidence == null) return null;
    if (!target) return "--mtp-drafts and --mtp-confidence draft with Qwen3.8-27B's MTP head on a Mac; serve this checkpoint without them";
    if (confidence) |c| if (!(c >= 0 and c <= 1)) return "--mtp-confidence must be from 0 to 1";
    const n = drafts orelse return "--mtp-confidence stops the MTP head's chains, which --mtp-drafts N turns on: serve with both or neither";
    if (n > max_drafts) return std.fmt.comptimePrint("--mtp-drafts: Qwen3.8-27B on a Mac drafts at most {d} a round", .{max_drafts});
    if (n > 0 and drafter) return "--mtp-drafts drafts with the MTP head, and --drafter with DFlash2: serve with one of them";
    return null;
}

test "qwen27 MTP flags: the 27B takes them, each misuse is refused by name, and no flag changes nothing" {
    try std.testing.expect(refusal(null, null, true, false) == null);
    try std.testing.expect(refusal(3, null, false, true) == null);
    try std.testing.expect(refusal(3, 0.4, false, true) == null);
    try std.testing.expect(refusal(0, null, true, true) == null); // 0 turns the head off, so DFlash2 may draft
    try std.testing.expect(refusal(max_drafts, 1, false, true) == null);
    try std.testing.expectStringStartsWith(refusal(3, null, false, false).?, "--mtp-drafts and --mtp-confidence draft with Qwen3.8-27B's");
    try std.testing.expectStringStartsWith(refusal(null, 0.5, false, true).?, "--mtp-confidence stops");
    try std.testing.expectStringStartsWith(refusal(3, 1.5, false, true).?, "--mtp-confidence must be");
    try std.testing.expectStringStartsWith(refusal(3, std.math.nan(f64), false, true).?, "--mtp-confidence must be");
    try std.testing.expectStringStartsWith(refusal(max_drafts + 1, null, false, true).?, "--mtp-drafts: Qwen3.8-27B on a Mac drafts at most 15");
    try std.testing.expectStringStartsWith(refusal(2, null, true, true).?, "--mtp-drafts drafts with the MTP head, and --drafter");
}
