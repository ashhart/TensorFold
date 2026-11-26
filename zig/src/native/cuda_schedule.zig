//! TF_CUDA_SCHED: how a serve's host thread waits for its GPU, set on the primary context before the context starts.
const std = @import("std");
const cuda = @import("cuda");
const environment = @import("sys").environment;

/// The setting's words in its registered order, each the CU_CTX_SCHED_* flag of the same name.
const schedules = [_]cuda.Context.Schedule{ .auto, .spin, .yield, .blocking };

/// The schedule `raw` names; null when unset or blank (the driver's own `auto`), refused by name otherwise.
pub fn parse(raw: ?[]const u8) environment.Error!?cuda.Context.Schedule {
    const i = (try environment.value(?u64, "TF_CUDA_SCHED", raw, null)) orelse return null;
    return schedules[@intCast(i)];
}

/// Sets TF_CUDA_SCHED on `ordinal`'s primary context; unset leaves the context as it is. Call before cuda.Context.init.
pub fn apply(d: *const cuda.Driver, ordinal: c_int) !void {
    const s = try parse(environment.rawText("TF_CUDA_SCHED")) orelse return;
    try set(d, ordinal, s);
}

/// TF_CUDA_SCHED for an engine that opens its own driver; close it once its context starts. Off: null, ordinal unread.
pub fn hold(ordinal: u32) !?cuda.Driver {
    return holdWith(environment.rawText("TF_CUDA_SCHED"), ordinal);
}

fn holdWith(raw: ?[]const u8, ordinal: u32) !?cuda.Driver {
    const s = try parse(raw) orelse return null;
    const device = std.math.cast(c_int, ordinal) orelse return error.NoSuchCudaDevice;
    var d = try cuda.Driver.open();
    errdefer d.close();
    try set(&d, device, s);
    return d;
}

fn set(d: *const cuda.Driver, ordinal: c_int, s: cuda.Context.Schedule) !void {
    try cuda.Context.setSchedule(d, ordinal, s);
    std.log.info("cuda schedule: {t}", .{s});
}

test "TF_CUDA_SCHED: unset and blank keep the driver's schedule, each word names its flag, others are refused" {
    try std.testing.expectEqual(@as(?cuda.Context.Schedule, null), try parse(null));
    try std.testing.expectEqual(@as(?cuda.Context.Schedule, null), try parse(""));
    for (environment.rule("TF_CUDA_SCHED").words, schedules) |word, s| {
        try std.testing.expectEqualStrings(word, @tagName(s));
        try std.testing.expectEqual(@as(?cuda.Context.Schedule, s), try parse(word));
    }
    for ([_][]const u8{ "block", "4", "on", "blocking,spin" }) |bad| try std.testing.expectError(error.InvalidEnvironment, parse(bad));
}

test "TF_CUDA_SCHED hold: off returns before the ordinal is read; set, an ordinal past c_int is refused" {
    const past = @as(u32, std.math.maxInt(c_int)) + 1;
    for ([_]?[]const u8{ null, "", " " }) |off| {
        try std.testing.expectEqual(@as(?cuda.Driver, null), try holdWith(off, std.math.maxInt(u32)));
        try std.testing.expectEqual(@as(?cuda.Driver, null), try holdWith(off, past));
    }
    for (environment.rule("TF_CUDA_SCHED").words) |word| {
        try std.testing.expectError(error.NoSuchCudaDevice, holdWith(word, std.math.maxInt(u32)));
        try std.testing.expectError(error.NoSuchCudaDevice, holdWith(word, past));
    }
}
