//! Fixed-panel receipt schema and the audit's hard targets; token equality stays report-only.
pub const Output = struct { content: []const u8, sha256: []const u8 };
pub const Kind = enum { unrelated, control, neighbour, recall };
pub const Row = struct {
    kind: Kind,
    question: []const u8,
    fact: u32 = 0,
    thinking: bool = false,
    baseline: ?Output = null,
    replies: [3]Output,
    classes: [3]enum { equal, alike, same, changed, recalled, missed },
};
pub const Counts = struct {
    checked: u32 = 0,
    byte_equal: u32 = 0,
    alike: u32 = 0,
    judged_same: u32 = 0,
    meaning_changed: u32 = 0,
    invented: u32 = 0,
    leaks: u32 = 0,
    recall_asked: u32 = 0,
    recall_passed: u32 = 0,
    earlier_asked: u32 = 0,
    earlier_passed: u32 = 0,
};
pub const Baseline = struct { version: u32 = 1, panel_sha256: []const u8, profile: []const u8, replies: []const Output };
pub const Report = struct {
    version: u32 = 1,
    status: []const u8 = "measured",
    panel_sha256: []const u8 = "",
    lessons: u32 = 0,
    gate: bool = false,
    complete: bool = true,
    untaught_mismatches: u32 = 0,
    plain: [3]Counts = @splat(.{}),
    thinking: [3]Counts = @splat(.{}),
    rows: []const Row = &.{},
    refused: ?[]const u8 = null,
};

fn median(xs: [3]bool) bool {
    return @as(u32, @intFromBool(xs[0])) + @intFromBool(xs[1]) + @intFromBool(xs[2]) >= 2;
}

pub fn refusal(r: Report) ?[]const u8 {
    if (!r.complete) return "fixed panel incomplete or supplied recall is missing or trained";
    if (r.untaught_mismatches != 0) return "untaught panel differs from its recorded baseline";
    const meaning_limit: u32 = if (r.lessons >= 10) 5 else 2;
    for (r.plain) |c| {
        if (c.invented != 0) return "fixed panel invented a user fact";
        if (c.leaks != 0) return "fixed panel leaked a fact into an unrelated question";
        if (c.meaning_changed > meaning_limit) return "fixed panel exceeded the cumulative meaning-change target";
    }
    // Leaks and inventions are hard with either thinking mode; recall with thinking on is report-only.
    for (r.thinking) |c| if (c.invented != 0 or c.leaks != 0) return "fixed thinking panel leaked or invented a fact";
    var recall: [3]bool = undefined;
    var earlier: [3]bool = undefined;
    for (r.plain, 0..) |c, i| {
        recall[i] = c.recall_asked > 0 and @as(u64, c.recall_passed) * 10 >= @as(u64, c.recall_asked) * 9;
        earlier[i] = @as(u64, c.earlier_passed) * 10 >= @as(u64, c.earlier_asked) * 9;
    }
    if (!median(recall)) return "fixed supplied recall missed the median 90% target";
    if (r.lessons >= 10 and !median(earlier)) return "fixed earlier-fact retention missed the median 90% target";
    return null;
}
