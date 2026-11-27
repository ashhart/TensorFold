//! The actual release admission refuses unavailable DeepSeek backends independently of the environment.
const std = @import("std");
const gate = @import("release_refusal");
const env = @import("policy_env");
const expected = "DeepSeek-V4.1 is not served by this release's CUDA server; its whole-model gate is #827";
const moss_refusal = "sm_110 product: MOSS-Image is not qualified; serve Qwen3.8-27B";
const multi_targets: gate.Targets = .{ .built_sm = null, .sms = &.{ 80, 110, 121 } };
fn check(value: ?[:0]const u8) !void {
    env.value = value;
    defer env.value = null;
    for ([_][]const u8{ "deepseek_v4", "deepseek_v41" }) |name| {
        try std.testing.expectEqualStrings(expected, gate.backend(.{ .dir = "not-opened", .model_type = name }) orelse "");
        try std.testing.expectEqualStrings(expected, gate.backend(.{ .dir = "another-unread-checkpoint", .model_type = name, .policy = "unsupported", .mtp_drafts = 32 }) orelse "");
    }
}
test "DeepSeek release: TF_EXPERIMENTAL unset is a constant refusal before device work" {
    try check(null);
}
test "DeepSeek release: TF_EXPERIMENTAL=1 is the same constant refusal" {
    try check("1");
}
test "DeepSeek release: TF_EXPERIMENTAL=deepseek is the same constant refusal" {
    try check("deepseek");
}
test "DeepSeek release: other CUDA admission stays available" {
    env.value = "deepseek";
    defer env.value = null;
    try std.testing.expect(gate.backend(.{ .dir = "not-opened", .model_type = "nemotron_h" }) == null);
}

test "SM110 product: a recognized MOSS-Image pipeline is refused by name" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    try std.testing.expect(try gate.sm110Product(a, std.testing.io, .{ .dir = "not-opened", .model_type = "unknown" }, true, &problem));
    defer a.free(problem);
    try std.testing.expectEqualStrings(moss_refusal, problem);
}

fn simulatedCapability(_: std.mem.Allocator, flag: ?u32, _: *[]const u8) !?u32 {
    return if ((flag orelse 1) == 0) 110 else 121;
}

test "SM110 product: a multi-target build refuses MOSS-Image before opening its checkpoint on Thor" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    try std.testing.expect(try gate.sm110Startup(a, std.testing.io, .{
        .dir = "not-opened",
        .model_type = "unknown",
    }, true, multi_targets, simulatedCapability, &problem));
    defer a.free(problem);
    try std.testing.expectEqualStrings(moss_refusal, problem);
}

fn simulatedGb10(_: std.mem.Allocator, flag: ?u32, _: *[]const u8) !?u32 {
    try std.testing.expectEqual(@as(?u32, 0), flag);
    return 121;
}

test "SM110 product: MOSS-Image on device zero of another SM keeps its existing admission" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    try std.testing.expect(!try gate.sm110Startup(a, std.testing.io, .{
        .dir = "not-opened",
        .model_type = "unknown",
    }, true, multi_targets, simulatedGb10, &problem));
    try std.testing.expectEqualStrings("", problem);
}

test "SM110 product: LFM2 is refused when its fixed device zero is Thor" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    try std.testing.expect(try gate.sm110Startup(a, std.testing.io, .{
        .dir = "not-opened",
        .model_type = "lfm2_vl",
    }, false, multi_targets, simulatedCapability, &problem));
    defer a.free(problem);
    const why = "sm_110 product: lfm2_vl is not qualified; serve Qwen3.8-27B";
    try std.testing.expectEqualStrings(why, problem);
}

test "SM110 product: a text model's explicit Thor device takes precedence over the default GPU" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    try std.testing.expect(!try gate.sm110Startup(a, std.testing.io, .{
        .dir = "not-opened",
        .model_type = "another-family",
    }, false, multi_targets, simulatedCapability, &problem));
    try std.testing.expect(try gate.sm110Startup(a, std.testing.io, .{
        .dir = "not-opened",
        .model_type = "another-family",
        .device = 0,
    }, false, multi_targets, simulatedCapability, &problem));
    defer a.free(problem);
    const why = "sm_110 product: another-family is not qualified; serve Qwen3.8-27B";
    try std.testing.expectEqualStrings(why, problem);
}

fn unexpectedProbe(_: std.mem.Allocator, _: ?u32, _: *[]const u8) !?u32 {
    return error.UnexpectedCapabilityProbe;
}

test "SM110 product: the single-target product still refuses MOSS-Image before probing CUDA" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    try std.testing.expect(try gate.sm110Startup(a, std.testing.io, .{
        .dir = "not-opened",
        .model_type = "unknown",
    }, true, .{ .built_sm = 110, .sms = &.{110} }, unexpectedProbe, &problem));
    defer a.free(problem);
    try std.testing.expectEqualStrings(moss_refusal, problem);
}

test "SM110 product: existing release and single-target development builds do not probe CUDA" {
    const a = std.testing.allocator;
    var problem: []const u8 = "";
    for ([_]gate.Targets{
        .{ .built_sm = 121, .sms = &.{121} },
        .{ .built_sm = null, .sms = &.{ 80, 86, 120, 121 } },
        .{ .built_sm = null, .sms = &.{110} },
    }) |targets| {
        try std.testing.expect(!try gate.sm110Startup(a, std.testing.io, .{
            .dir = "not-opened",
            .model_type = "unknown",
        }, true, targets, unexpectedProbe, &problem));
    }
    try std.testing.expectEqualStrings("", problem);
}

fn unavailableCapability(_: std.mem.Allocator, _: ?u32, problem: *[]const u8) !?u32 {
    problem.* = "CUDA capability unavailable";
    return null;
}

test "SM110 product: a multi-target build refuses startup when it cannot identify the selected GPU" {
    var problem: []const u8 = "";
    try std.testing.expect(try gate.sm110Startup(std.testing.allocator, std.testing.io, .{
        .dir = "not-opened",
        .model_type = "unknown",
    }, true, multi_targets, unavailableCapability, &problem));
    try std.testing.expectEqualStrings("CUDA capability unavailable", problem);
}
