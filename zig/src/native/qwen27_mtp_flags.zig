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

/// Why the MTP head is refused on this Mac: an M5's tensor units, or K/V pages its chained rows do not cover.
pub fn deviceRefusal(drafts: ?u32, tensor_units: bool, pages: bool) ?Refused {
    if ((drafts orelse 0) == 0) return null;
    if (tensor_units) return .{ .err = error.MtpOnTensorUnits, .why = "--mtp-drafts: the MTP head drafts on Macs without tensor units (M1 to M4); an M5 serves Qwen3.8-27B with DFlash2 or plain rounds" };
    if (pages) return .{ .err = error.MtpWithKvPages, .why = "--mtp-drafts: the MTP head's chained rows do not take K/V pages yet; unset TF_QWEN27_METAL_KV_PAGES or serve without --mtp-drafts" };
    return null;
}
pub const Refused = struct { err: anyerror, why: []const u8 };

/// Hashes the disk identity's mode words; the head's word joins only when it is on, as main keys the rest.
pub fn identityWords(w: *std.hash.Wyhash, words: []const u64, head: bool) void {
    w.update(std.mem.sliceAsBytes(words));
    if (head) w.update(std.mem.asBytes(&@as(u64, 1)));
}

test "qwen27 MTP flags: a server without the head keys disk states as main does, and one with it does not" {
    const words = [_]u64{ 0x5eed, 1, 0, 1, 0x9e37 };
    var main = std.hash.Wyhash.init(0x7132);
    main.update(std.mem.sliceAsBytes(&words));
    var off = std.hash.Wyhash.init(0x7132);
    identityWords(&off, &words, false);
    var on = std.hash.Wyhash.init(0x7132);
    identityWords(&on, &words, true);
    try std.testing.expectEqual(main.final(), off.final());
    try std.testing.expect(main.final() != on.final());
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

test "qwen27 MTP flags: the head is refused by name on an M5 and with K/V pages, and taken otherwise" {
    try std.testing.expect(deviceRefusal(null, true, true) == null);
    try std.testing.expect(deviceRefusal(0, true, true) == null); // 0 turns the head off
    try std.testing.expect(deviceRefusal(3, false, false) == null);
    const m5 = deviceRefusal(3, true, false).?;
    try std.testing.expectEqual(error.MtpOnTensorUnits, m5.err);
    try std.testing.expectStringStartsWith(m5.why, "--mtp-drafts: the MTP head drafts on Macs without tensor units");
    const paged = deviceRefusal(3, false, true).?;
    try std.testing.expectEqual(error.MtpWithKvPages, paged.err);
    try std.testing.expect(std.mem.indexOf(u8, paged.why, "TF_QWEN27_METAL_KV_PAGES") != null);
}
