//! tf-moss-image-generate's options: the parser the driver in zig/tests/moss_image_generate_run.zig reads.
const std = @import("std");
const config = @import("config.zig");

pub const usage =
    \\tf-moss-image-generate TRANSFORMER_DIR [OPTIONS] OUT.png
    \\--prompt TEXT and OUT.png are required. The card's defaults are 1024x1024, 50 steps, guidance 3.0, and the
    \\negative branch skipping block 22; --negative TEXT defaults to the empty negative prompt, --seed to 42.
    \\A side is at most 1536 and width times height at most 1572864; --steps is 1 to 200, --guidance at least 1,
    \\--prompt and --negative TEXT at most 16384 bytes.
    \\TRANSFORMER_DIR is the checkpoint's transformer/ folder: scheduler/scheduler_config.json beside it gives the
    \\schedule, and its vae/ folder decodes the latents into OUT.png. Skip-layer guidance (--slg-layers, --slg-scale,
    \\--slg-start, --slg-stop) is refused: this family does not implement it.
;

/// The card's: the negative branch skips block 22.
pub const negative_skip = 22;

/// The caps the API serves and this tool refuses past exactly: 1536 a side, 1572864 pixels, 200 steps.
const max_side: i64 = 1536;
const max_pixels: usize = 1572864;
const max_steps: i64 = 200;
const max_prompt_bytes = 16 * 1024;

pub const Args = struct {
    out: []const u8,
    prompt: []const u8,
    negative: []const u8 = "",
    width: usize = 1024,
    height: usize = 1024,
    steps: usize = 50,
    guidance: f32 = 3.0,
    seed: u64 = 42,
};

pub const Error = error{
    UnknownOption,
    MissingValue,
    MissingPrompt,
    EmptyPrompt,
    MissingOutput,
    BadValue,
    BadImageSize,
    ImageTooLarge,
    BadStepCount,
    TooManySteps,
    SkipLayerGuidanceRefused,
};

/// The options taken, matched on the token after `--`.
pub const Option = enum { prompt, negative, width, height, steps, seed, guidance };

/// Whether `token` names the pipeline's skip-layer guidance, dashed or bare: `--slg`, `--slg-layers`, `slg_scale`.
fn skipLayer(token: []const u8) bool {
    const name = std.mem.trimStart(u8, token, "-");
    if (!std.mem.startsWith(u8, name, "slg")) return false;
    const rest = name["slg".len..];
    return rest.len == 0 or rest[0] == '-' or rest[0] == '_';
}

/// A width or height: a non-negative number within the served side limit, the grid's alignment checked with it.
fn imageSize(text: []const u8) Error!usize {
    const value = std.fmt.parseInt(i64, text, 10) catch return error.BadValue;
    if (value < 0) return error.BadImageSize;
    if (value > max_side) return error.ImageTooLarge;
    return @intCast(value);
}

/// The bounds on a finished pair of sides: the grid's alignment and the served pixel limit.
fn fit(args: *const Args) Error!void {
    _ = try config.grid(args.width, args.height);
    if (args.width * args.height > max_pixels) return error.ImageTooLarge;
}

/// The number of steps: the served 1 to 200, each bound its own error as the server's two sentences are.
fn stepCount(text: []const u8) Error!usize {
    const value = std.fmt.parseInt(i64, text, 10) catch return error.BadValue;
    if (value < 1) return error.BadStepCount;
    if (value > max_steps) return error.TooManySteps;
    return @intCast(value);
}

/// The seed: the served non-negative int64, which the pipeline draws from as a u64.
fn seedValue(text: []const u8) Error!u64 {
    const value = std.fmt.parseInt(i64, text, 10) catch return error.BadValue;
    if (value < 0) return error.BadValue;
    return @intCast(value);
}

/// The guidance scale: a finite number from 1, as the server takes it.
fn guidanceScale(text: []const u8) Error!f32 {
    const value = std.fmt.parseFloat(f32, text) catch return error.BadValue;
    if (!std.math.isFinite(value) or value < 1) return error.BadValue;
    return value;
}

/// A prompt's text: blank unless `blank_ok` is refused, as is one past the byte cap; the trimmed text comes back.
fn promptText(text: []const u8, blank_ok: bool) Error![]const u8 {
    const kept = std.mem.trim(u8, text, " \t\r\n");
    if (kept.len == 0 and !blank_ok) return error.EmptyPrompt;
    if (kept.len > max_prompt_bytes) return error.BadValue;
    return kept;
}

/// `tokens` are the options in any order after TRANSFORMER_DIR plus the output path; `refused` names the bad token.
pub fn parse(tokens: []const []const u8, refused: *?[]const u8, option: *?Option) Error!Args {
    refused.* = null;
    option.* = null;
    var args: Args = .{ .out = "", .prompt = "" };
    var prompt: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var side: ?Option = null;
    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        option.* = null;
        const token = tokens[i];
        if (skipLayer(token)) {
            refused.* = token;
            return error.SkipLayerGuidanceRefused;
        }
        if (std.mem.startsWith(u8, token, "--")) {
            const given = std.meta.stringToEnum(Option, token[2..]) orelse {
                refused.* = token;
                return error.UnknownOption;
            };
            option.* = given;
            const value = if (i + 1 < tokens.len) tokens[i + 1] else return error.MissingValue;
            i += 1;
            if (std.mem.startsWith(u8, value, "--")) {
                // The refusal belongs to the token that should have been the value, not to the option before it.
                option.* = null;
                refused.* = value;
                if (!skipLayer(value)) return error.MissingValue;
                return error.SkipLayerGuidanceRefused;
            }
            switch (given) {
                .prompt => prompt = try promptText(value, false),
                .negative => args.negative = try promptText(value, true),
                .width => args.width = try imageSize(value),
                .height => args.height = try imageSize(value),
                .steps => args.steps = try stepCount(value),
                .seed => args.seed = try seedValue(value),
                .guidance => args.guidance = try guidanceScale(value),
            }
            // The side a refused pair names: the last one given.
            if (given == .width or given == .height) side = given;
            continue;
        }
        if (out != null) {
            refused.* = token;
            return error.UnknownOption;
        }
        out = token;
    }
    option.* = null;
    // The finished pair, checked once as the server checks it: a later --width may rescue an earlier side.
    if (fit(&args)) |_| {} else |e| {
        option.* = side;
        return e;
    }
    args.prompt = prompt orelse return error.MissingPrompt;
    args.out = out orelse return error.MissingOutput;
    return args;
}

/// The sentence a refused value missed its option's bound by.
fn valueReason(given: Option, e: Error) []const u8 {
    return switch (given) {
        .width, .height => switch (e) {
            error.BadImageSize => "must be a positive multiple of 16, at most 1536",
            error.ImageTooLarge => "is past 1536 a side or 1572864 pixels",
            else => "must be a whole number from 1 to 1536",
        },
        .steps => switch (e) {
            error.BadStepCount => "must be from 1 to 200",
            error.TooManySteps => "is past 200",
            else => "must be a whole number from 1 to 200",
        },
        .seed => "must be a whole number from 0",
        .guidance => "must be a finite number from 1",
        .prompt, .negative => switch (e) {
            error.EmptyPrompt => "is blank",
            else => "is longer than 16384 bytes",
        },
    };
}

/// The refusal's sentence: the option's bound when a value was being read, else the plain one.
fn reason(e: Error, option: ?Option) []const u8 {
    if (e == error.MissingValue) return "the option has no value";
    if (option) |given| return valueReason(given, e);
    return switch (e) {
        error.SkipLayerGuidanceRefused => "skip-layer guidance is not implemented",
        error.UnknownOption => "unknown option or extra argument",
        error.MissingPrompt => "--prompt is required",
        error.MissingOutput => "the output PNG path is required",
        else => "the value is not a number in range",
    };
}

/// The one refusal line: the refused token, else the option, then its sentence.
fn line(a: std.mem.Allocator, e: Error, option: ?Option, refused: ?[]const u8) ![]const u8 {
    const why = reason(e, option);
    if (refused) |token| return std.fmt.allocPrint(a, "tf-moss-image-generate: {s}: {s}", .{ token, why });
    if (option) |given| return std.fmt.allocPrint(a, "tf-moss-image-generate: --{s}: {s}", .{ @tagName(given), why });
    return std.fmt.allocPrint(a, "tf-moss-image-generate: {s}", .{why});
}

/// The refusal's line, then the usage.
pub fn complain(a: std.mem.Allocator, e: Error, option: ?Option, refused: ?[]const u8) void {
    const said = line(a, e, option, refused) catch return;
    std.debug.print("{s}\n{s}\n", .{ said, usage });
}

/// `parse` with scratch out-params, for the tests that want only its result.
fn check(tokens: []const []const u8) Error!Args {
    var refused: ?[]const u8 = null;
    var option: ?Option = null;
    return parse(tokens, &refused, &option);
}

/// The refusal line `tokens` produce, as it prints; "" when they parse.
fn refusal(a: std.mem.Allocator, tokens: []const []const u8) ![]const u8 {
    var refused: ?[]const u8 = null;
    var option: ?Option = null;
    return if (parse(tokens, &refused, &option)) |_| "" else |e| line(a, e, option, refused);
}

test "the card's defaults" {
    const tokens = [_][]const u8{ "picture.png", "--prompt", "a fox" };
    var refused: ?[]const u8 = null;
    var option: ?Option = null;
    const args = try parse(&tokens, &refused, &option);
    try std.testing.expectEqualStrings("picture.png", args.out);
    try std.testing.expectEqualStrings("a fox", args.prompt);
    try std.testing.expectEqualStrings("", args.negative);
    try std.testing.expectEqual(@as(usize, 1024), args.width);
    try std.testing.expectEqual(@as(usize, 1024), args.height);
    try std.testing.expectEqual(@as(usize, 50), args.steps);
    try std.testing.expectEqual(@as(f32, 3.0), args.guidance);
    try std.testing.expectEqual(@as(u64, 42), args.seed);
    try std.testing.expectEqual(@as(?[]const u8, null), refused);
}

test "every option" {
    const every = [_][]const u8{ "--negative", "blurry", "--steps", "20", "pic.png", "--seed", "42", "--guidance", "7.5", "--height", "880", "--width", "1200", "--prompt", "a fox" };
    const args = try check(&every);
    try std.testing.expectEqualStrings("pic.png", args.out);
    try std.testing.expectEqualStrings("a fox", args.prompt);
    try std.testing.expectEqualStrings("blurry", args.negative);
    try std.testing.expectEqual(@as(usize, 1200), args.width);
    try std.testing.expectEqual(@as(usize, 880), args.height);
    try std.testing.expectEqual(@as(usize, 20), args.steps);
    try std.testing.expectEqual(@as(f32, 7.5), args.guidance);
    try std.testing.expectEqual(@as(u64, 42), args.seed);
}

test "options in any order" {
    const one = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", "1200", "--seed", "7" };
    const other = [_][]const u8{ "--seed", "7", "--width", "1200", "pic.png", "--prompt", "a fox" };
    const a = try check(&one);
    const b = try check(&other);
    try std.testing.expectEqual(a.width, b.width);
    try std.testing.expectEqual(a.seed, b.seed);
    try std.testing.expectEqualStrings(a.out, b.out);
    try std.testing.expectEqualStrings(a.prompt, b.prompt);
}

test "sizes are positive multiples of 16 that the grid holds" {
    for ([_][]const u8{ "1000", "0", "-16", "15" }) |size| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", size };
        try std.testing.expectError(error.BadImageSize, check(&tokens));
        const tall = [_][]const u8{ "pic.png", "--prompt", "a fox", "--height", size };
        try std.testing.expectError(error.BadImageSize, check(&tall));
    }
    const huge = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", "4294967296" };
    try std.testing.expectError(error.ImageTooLarge, check(&huge));
    const loose = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", "wide" };
    try std.testing.expectError(error.BadValue, check(&loose));
}

test "a side and the pixel count keep to the served bounds" {
    const widest = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", "1536", "--height", "1024" };
    const args = try check(&widest);
    try std.testing.expectEqual(@as(usize, 1536), args.width);
    try std.testing.expectEqual(@as(usize, 1024), args.height);
    const past_side = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", "1552" };
    try std.testing.expectError(error.ImageTooLarge, check(&past_side));
    const past_pixels = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", "1536", "--height", "1536" };
    try std.testing.expectError(error.ImageTooLarge, check(&past_pixels));
    // The pair is checked once, on the finished values: the last --width is the one that counts.
    const rescued = [_][]const u8{ "pic.png", "--prompt", "a fox", "--width", "1536", "--height", "1536", "--width", "512" };
    const twice = try check(&rescued);
    try std.testing.expectEqual(@as(usize, 512), twice.width);
    try std.testing.expectEqual(@as(usize, 1536), twice.height);
}

test "steps are from 1 to 200" {
    for ([_][]const u8{ "0", "-1" }) |count| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--steps", count };
        try std.testing.expectError(error.BadStepCount, check(&tokens));
    }
    for ([_][]const u8{ "201", "4097" }) |count| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--steps", count };
        try std.testing.expectError(error.TooManySteps, check(&tokens));
    }
    const loose = [_][]const u8{ "pic.png", "--prompt", "a fox", "--steps", "many" };
    try std.testing.expectError(error.BadValue, check(&loose));
    const last = [_][]const u8{ "pic.png", "--prompt", "a fox", "--steps", "200" };
    try std.testing.expectEqual(@as(usize, 200), (try check(&last)).steps);
}

test "a seed and a guidance scale are numbers in range" {
    const most = [_][]const u8{ "pic.png", "--prompt", "a fox", "--seed", "9223372036854775807", "--guidance", "1.5" };
    const args = try check(&most);
    try std.testing.expectEqual(@as(u64, 9223372036854775807), args.seed);
    try std.testing.expectEqual(@as(f32, 1.5), args.guidance);
    for ([_][]const u8{ "-1", "9223372036854775808", "seed" }) |seed| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--seed", seed };
        try std.testing.expectError(error.BadValue, check(&tokens));
    }
    for ([_][]const u8{ "nan", "inf", "0.5", "heavy" }) |scale| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--guidance", scale };
        try std.testing.expectError(error.BadValue, check(&tokens));
    }
    const least = [_][]const u8{ "pic.png", "--prompt", "a fox", "--seed", "0", "--guidance", "1" };
    const first = try check(&least);
    try std.testing.expectEqual(@as(u64, 0), first.seed);
    try std.testing.expectEqual(@as(f32, 1.0), first.guidance);
}

test "a prompt keeps its trimmed text and keeps to the served byte cap" {
    const spaced = [_][]const u8{ "pic.png", "--prompt", "  a fox  ", "--negative", "  blurry " };
    const args = try check(&spaced);
    try std.testing.expectEqualStrings("a fox", args.prompt);
    try std.testing.expectEqualStrings("blurry", args.negative);
    for ([_][]const u8{ "", " ", "\t\n" }) |blank| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", blank };
        try std.testing.expectError(error.EmptyPrompt, check(&tokens));
    }
    var at_cap: [16384]u8 = undefined;
    @memset(&at_cap, 'a');
    const full = [_][]const u8{ "pic.png", "--prompt", &at_cap };
    try std.testing.expectEqual(@as(usize, 16384), (try check(&full)).prompt.len);
    var past_cap: [16385]u8 = undefined;
    @memset(&past_cap, 'a');
    const long = [_][]const u8{ "pic.png", "--prompt", &past_cap };
    try std.testing.expectError(error.BadValue, check(&long));
    const long_negative = [_][]const u8{ "pic.png", "--prompt", "a fox", "--negative", &past_cap };
    try std.testing.expectError(error.BadValue, check(&long_negative));
}

test "an unknown option or extra argument is refused and named" {
    var refused: ?[]const u8 = null;
    var option: ?Option = null;
    const unknown = [_][]const u8{ "pic.png", "--prompt", "a fox", "--cfg", "7" };
    try std.testing.expectError(error.UnknownOption, parse(&unknown, &refused, &option));
    try std.testing.expectEqualStrings("--cfg", refused.?);
    const extra = [_][]const u8{ "pic.png", "second.png", "--prompt", "a fox" };
    try std.testing.expectError(error.UnknownOption, parse(&extra, &refused, &option));
    try std.testing.expectEqualStrings("second.png", refused.?);
}

test "an option without a value is refused" {
    for ([_][]const u8{ "--prompt", "--width", "--seed" }) |option| {
        const tokens = [_][]const u8{ "pic.png", option };
        try std.testing.expectError(error.MissingValue, check(&tokens));
    }
    const guidance = [_][]const u8{ "pic.png", "--prompt", "a fox", "--guidance" };
    try std.testing.expectError(error.MissingValue, check(&guidance));
}

test "an option's value cannot be another option" {
    var refused: ?[]const u8 = null;
    var option: ?Option = null;
    const swallowed = [_][]const u8{ "pic.png", "--prompt", "a fox", "--negative", "--slg-layers", "0.1" };
    try std.testing.expectError(error.SkipLayerGuidanceRefused, parse(&swallowed, &refused, &option));
    try std.testing.expectEqualStrings("--slg-layers", refused.?);
    const missing = [_][4][]const u8{
        .{ "pic.png", "--prompt", "--width", "1024" },
        .{ "pic.png", "--prompt", "--cfg", "7" },
    };
    for (missing) |tokens| try std.testing.expectError(error.MissingValue, check(&tokens));
    // A value that is another option names that option as the one left without a value.
    const swallowed_line = try refusal(std.testing.allocator, &missing[0]);
    defer std.testing.allocator.free(swallowed_line);
    try std.testing.expectEqualStrings("tf-moss-image-generate: --width: the option has no value", swallowed_line);
}

test "the output path and the prompt are required" {
    const no_out = [_][]const u8{ "--prompt", "a fox" };
    try std.testing.expectError(error.MissingOutput, check(&no_out));
    const no_prompt = [_][]const u8{"pic.png"};
    try std.testing.expectError(error.MissingPrompt, check(&no_prompt));
    const nothing: [0][]const u8 = .{};
    try std.testing.expectError(error.MissingPrompt, check(&nothing));
}

test "an empty prompt is refused" {
    const empty = [_][]const u8{ "pic.png", "--prompt", "" };
    try std.testing.expectError(error.EmptyPrompt, check(&empty));
}

test "every skip-layer guidance spelling is refused and named" {
    var refused: ?[]const u8 = null;
    var option: ?Option = null;
    const spellings = [_][]const u8{ "--slg", "--slg-layers", "--slg_scale", "--slg-start", "--slg_stop", "slg", "slg_layers", "slg_scale", "slg_start", "slg_stop" };
    for (spellings) |token| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", token, "0.5" };
        try std.testing.expectError(error.SkipLayerGuidanceRefused, parse(&tokens, &refused, &option));
        try std.testing.expectEqualStrings(token, refused.?);
    }
}

test "a refused value names the option it belongs to" {
    const cases = [_]struct { tokens: [5][]const u8, want: []const u8 }{
        .{ .tokens = .{ "pic.png", "--prompt", "a fox", "--width", "1000" }, .want = "tf-moss-image-generate: --width: must be a positive multiple of 16, at most 1536" },
        .{ .tokens = .{ "pic.png", "--prompt", "a fox", "--height", "1000" }, .want = "tf-moss-image-generate: --height: must be a positive multiple of 16, at most 1536" },
        .{ .tokens = .{ "pic.png", "--prompt", "a fox", "--width", "1552" }, .want = "tf-moss-image-generate: --width: is past 1536 a side or 1572864 pixels" },
        .{ .tokens = .{ "pic.png", "--prompt", "a fox", "--steps", "0" }, .want = "tf-moss-image-generate: --steps: must be from 1 to 200" },
        .{ .tokens = .{ "pic.png", "--prompt", "a fox", "--steps", "201" }, .want = "tf-moss-image-generate: --steps: is past 200" },
        .{ .tokens = .{ "pic.png", "--prompt", "a fox", "--seed", "-1" }, .want = "tf-moss-image-generate: --seed: must be a whole number from 0" },
        .{ .tokens = .{ "pic.png", "--prompt", "a fox", "--guidance", "0.5" }, .want = "tf-moss-image-generate: --guidance: must be a finite number from 1" },
        .{ .tokens = .{ "pic.png", "--prompt", " ", "--seed", "7" }, .want = "tf-moss-image-generate: --prompt: is blank" },
    };
    for (cases) |case| {
        const got = try refusal(std.testing.allocator, &case.tokens);
        defer std.testing.allocator.free(got);
        try std.testing.expectEqualStrings(case.want, got);
    }
}

test "a refusal names the refused token, else the bare sentence" {
    const unknown = try refusal(std.testing.allocator, &.{ "pic.png", "--prompt", "a fox", "--cfg", "7" });
    defer std.testing.allocator.free(unknown);
    try std.testing.expectEqualStrings("tf-moss-image-generate: --cfg: unknown option or extra argument", unknown);
    const slg = try refusal(std.testing.allocator, &.{ "pic.png", "--prompt", "a fox", "--negative", "--slg-layers" });
    defer std.testing.allocator.free(slg);
    try std.testing.expectEqualStrings("tf-moss-image-generate: --slg-layers: skip-layer guidance is not implemented", slg);
    const no_value = try refusal(std.testing.allocator, &.{ "pic.png", "--prompt", "a fox", "--width" });
    defer std.testing.allocator.free(no_value);
    try std.testing.expectEqualStrings("tf-moss-image-generate: --width: the option has no value", no_value);
    const pair = try refusal(std.testing.allocator, &.{ "pic.png", "--prompt", "a fox", "--width", "1536", "--height", "1536" });
    defer std.testing.allocator.free(pair);
    try std.testing.expectEqualStrings("tf-moss-image-generate: --height: is past 1536 a side or 1572864 pixels", pair);
    const no_prompt = try refusal(std.testing.allocator, &.{ "--width", "1024", "pic.png" });
    defer std.testing.allocator.free(no_prompt);
    try std.testing.expectEqualStrings("tf-moss-image-generate: --prompt is required", no_prompt);
    const no_out = try refusal(std.testing.allocator, &.{ "--prompt", "a fox" });
    defer std.testing.allocator.free(no_out);
    try std.testing.expectEqualStrings("tf-moss-image-generate: the output PNG path is required", no_out);
}
