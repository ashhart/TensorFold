//! tf-moss-image-generate's options and its pipeline for any backend `B`: a picture from options alone.
const std = @import("std");
const config = @import("config.zig");
const noise = @import("noise.zig");
const schedule = @import("schedule.zig");

const usage =
    \\tf-moss-image-generate TRANSFORMER_DIR [OPTIONS] OUT.png
    \\--prompt TEXT and OUT.png are required. The card's defaults are 1024x1024, 50 steps, guidance 3.0, and the
    \\negative branch skipping block 22; --negative TEXT defaults to the empty negative prompt, --seed to 42.
    \\TRANSFORMER_DIR is the checkpoint's transformer/ folder: scheduler/scheduler_config.json beside it gives the
    \\schedule, and its vae/ folder decodes the latents into OUT.png. Skip-layer guidance (--slg-layers, --slg-scale,
    \\--slg-start, --slg-stop) is refused: this family does not implement it.
;

/// The card's: the negative branch skips block 22.
const negative_skip = 22;

/// 4096 is generous headroom over the recorded 2,000-step cases; a count past it is a typo, not a run.
const most_steps = 4096;

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
    SkipLayerGuidanceRefused,
};

/// The options taken, matched on the token after `--`.
const Option = enum { prompt, negative, width, height, steps, seed, guidance };

/// Whether `token` names the pipeline's skip-layer guidance, dashed or bare: `--slg`, `--slg-layers`, `slg_scale`.
fn skipLayer(token: []const u8) bool {
    const name = std.mem.trimStart(u8, token, "-");
    if (!std.mem.startsWith(u8, name, "slg")) return false;
    const rest = name["slg".len..];
    return rest.len == 0 or rest[0] == '-' or rest[0] == '_';
}

/// A width or height: negative is no size, and the grid refuses zero, unaligned and oversized ones.
fn imageSize(text: []const u8) Error!usize {
    const value = std.fmt.parseInt(i64, text, 10) catch return error.BadValue;
    if (value < 0) return error.BadImageSize;
    return std.math.cast(usize, value) orelse error.ImageTooLarge;
}

/// The number of steps: at least one, and few enough that the schedule's arrays cannot be a typo's allocation.
fn stepCount(text: []const u8) Error!usize {
    const value = std.fmt.parseInt(i64, text, 10) catch return error.BadValue;
    if (value < 1) return error.BadStepCount;
    if (value > most_steps) return error.BadValue;
    return @intCast(value);
}

/// The guidance scale: any finite float, so the card's 3.0 can be replaced.
fn guidanceScale(text: []const u8) Error!f32 {
    const value = std.fmt.parseFloat(f32, text) catch return error.BadValue;
    if (!std.math.isFinite(value)) return error.BadValue;
    return value;
}

/// `tokens` are the arguments after TRANSFORMER_DIR: the options in any order and one output path.
pub fn parse(tokens: []const []const u8, refused: *?[]const u8) Error!Args {
    refused.* = null;
    var args: Args = .{ .out = "", .prompt = "" };
    var prompt: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var i: usize = 0;
    while (i < tokens.len) : (i += 1) {
        const token = tokens[i];
        if (skipLayer(token)) {
            refused.* = token;
            return error.SkipLayerGuidanceRefused;
        }
        if (std.mem.startsWith(u8, token, "--")) {
            const option = std.meta.stringToEnum(Option, token[2..]) orelse {
                refused.* = token;
                return error.UnknownOption;
            };
            const value = if (i + 1 < tokens.len) tokens[i + 1] else return error.MissingValue;
            i += 1;
            if (std.mem.startsWith(u8, value, "--")) {
                refused.* = value;
                if (!skipLayer(value)) return error.MissingValue;
                return error.SkipLayerGuidanceRefused;
            }
            switch (option) {
                .prompt => prompt = value,
                .negative => args.negative = value,
                .width => args.width = try imageSize(value),
                .height => args.height = try imageSize(value),
                .steps => args.steps = try stepCount(value),
                .seed => args.seed = std.fmt.parseInt(u64, value, 10) catch return error.BadValue,
                .guidance => args.guidance = try guidanceScale(value),
            }
            continue;
        }
        if (out != null) {
            refused.* = token;
            return error.UnknownOption;
        }
        out = token;
    }
    _ = try config.grid(args.width, args.height);
    args.prompt = prompt orelse return error.MissingPrompt;
    if (args.prompt.len == 0) return error.EmptyPrompt;
    args.out = out orelse return error.MissingOutput;
    return args;
}

/// The failure's line, naming the refused token when there is one, then the usage.
fn complain(e: Error, refused: ?[]const u8) void {
    const why = switch (e) {
        error.SkipLayerGuidanceRefused => "skip-layer guidance is not implemented",
        error.UnknownOption => "unknown option or extra argument",
        error.MissingValue => "the option has no value",
        error.MissingPrompt => "--prompt is required",
        error.EmptyPrompt => "--prompt is empty",
        error.MissingOutput => "the output PNG path is required",
        error.BadValue => "the value is not a number in range",
        error.BadImageSize => "--width and --height must be positive multiples of 16",
        error.ImageTooLarge => "the image is too large for the family's buffers",
        error.BadStepCount => "--steps must be at least 1",
    };
    if (refused) |token| {
        std.debug.print("tf-moss-image-generate: {s}: {s}\n", .{ token, why });
    } else std.debug.print("tf-moss-image-generate: {s}\n", .{why});
    std.debug.print("{s}\n", .{usage});
}

pub fn run(comptime B: type, init: std.process.Init) !u8 {
    const mi = B.mi;
    const a = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len < 2 or std.mem.startsWith(u8, argv[1], "--")) {
        std.debug.print("tf-moss-image-generate: TRANSFORMER_DIR is required\n{s}\n", .{usage});
        return 2;
    }
    var refused: ?[]const u8 = null;
    const args = parse(argv[2..], &refused) catch |e| {
        complain(e, refused);
        return 2;
    };
    const transformer = argv[1];
    const checkpoint = std.fs.path.dirname(transformer) orelse ".";
    const grid = try config.grid(args.width, args.height);
    const rows = grid.rows * grid.columns;
    const count = rows * config.channels;
    const steps: u32 = @intCast(args.steps);

    const path = try std.fs.path.join(a, &.{ checkpoint, "scheduler", "scheduler_config.json" });
    const settings = try schedule.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20)));
    const sched = try schedule.timesteps(a, settings, steps);
    const start = try a.alloc(f32, count);
    try noise.latents(a, args.seed, grid.rows, grid.columns, start);
    std.debug.print("seed {d}: {d} starting latents at {d}x{d}, {d} steps of the {d} asked for\n", .{ args.seed, count, args.width, args.height, sched.timesteps.len, args.steps });

    var gpu: B.Device = undefined;
    try gpu.init();
    defer gpu.deinit();
    var both: [2][]u16 = undefined;
    if (comptime @hasDecl(B.Device, "encode")) {
        const texts = [2][]const u8{ args.prompt, args.negative };
        std.debug.print("both prompts encoded natively: {d:.3} s\n", .{try gpu.encode(a, io, checkpoint, texts, &both)});
    } else {
        std.debug.print("tf-moss-image-generate: this build has no prompt encoder (the Metal build does)\n", .{});
        return 2;
    }
    const text = both[0].len / config.context;
    const negative_rows = both[1].len / config.context;
    // The transformer is freed when its run ends, before the decoder loads.
    const x = denoise: {
        const loading = B.seconds();
        const model = try gpu.load(std.heap.page_allocator, io, transformer, @max(text, negative_rows), args.width, args.height);
        defer model.deinit();
        std.debug.print("{s}: {d} image rows, prompts of {d} and {d} rows, guidance {d:.2}; loaded in {d:.1} s\n", .{ gpu.name(), rows, text, negative_rows, args.guidance, B.seconds() - loading });
        const refining = B.seconds();
        var branches: [2]mi.Prompt = undefined;
        branches[0] = try model.prompt(both[0]);
        defer branches[0].deinit();
        branches[1] = try model.prompt(both[1]);
        defer branches[1].deinit();
        std.debug.print("both prompts refined: {d:.3} s\n", .{B.seconds() - refining});

        const x = try a.dupe(f32, start);
        const cond = try a.alloc(f32, count);
        const uncond = try a.alloc(f32, count);
        var total: f64 = 0;
        for (0..sched.timesteps.len) |s| {
            const c = try model.forward(&branches[0], x, sched.timesteps[s], null, cond);
            const u = try model.forward(&branches[1], x, sched.timesteps[s], negative_skip, uncond);
            const by = sched.sigmas[s + 1] - sched.sigmas[s];
            for (x, cond, uncond) |*o, pc, pu| o.* += (pu + args.guidance * (pc - pu)) * by;
            const wall = c.wall + u.wall;
            total += wall;
            std.debug.print("step {d}/{d}: {d:.3} s ({d:.3} s on the GPU)\n", .{ s + 1, sched.timesteps.len, wall, c.gpu + u.gpu });
        }
        std.debug.print("denoise: {d:.2} s for {d} steps ({d} forwards)\n", .{ total, sched.timesteps.len, 2 * sched.timesteps.len });
        break :denoise x;
    };
    if (comptime @hasDecl(B.Device, "decode")) {
        const picture = try gpu.decode(a, io, checkpoint, x, args.width, args.height);
        std.debug.print("decoded {d}x{d}: {d:.3} s ({d:.3} s on the GPU), the decoder loaded in {d:.2} s\n", .{ args.width, args.height, picture.wall, picture.gpu, picture.load_s });
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args.out, .data = picture.png });
        std.debug.print("wrote {s}\n", .{args.out});
        return 0;
    } else {
        std.debug.print("tf-moss-image-generate: this build has no image decoder (the Metal build does)\n", .{});
        return 2;
    }
}

/// `parse` with a scratch `refused`, for the tests that want only its result.
fn check(tokens: []const []const u8) Error!Args {
    var refused: ?[]const u8 = null;
    return parse(tokens, &refused);
}

test "the card's defaults" {
    const tokens = [_][]const u8{ "picture.png", "--prompt", "a fox" };
    var refused: ?[]const u8 = null;
    const args = try parse(&tokens, &refused);
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

test "steps are at least one, and not a typo's count" {
    for ([_][]const u8{ "0", "-1" }) |count| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--steps", count };
        try std.testing.expectError(error.BadStepCount, check(&tokens));
    }
    for ([_][]const u8{ "4097", "many" }) |count| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--steps", count };
        try std.testing.expectError(error.BadValue, check(&tokens));
    }
}

test "a seed and a guidance scale are numbers in range" {
    const most = [_][]const u8{ "pic.png", "--prompt", "a fox", "--seed", "18446744073709551615", "--guidance", "0.25" };
    const args = try check(&most);
    try std.testing.expectEqual(@as(u64, std.math.maxInt(u64)), args.seed);
    try std.testing.expectEqual(@as(f32, 0.25), args.guidance);
    for ([_][]const u8{ "-1", "18446744073709551616", "seed" }) |seed| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--seed", seed };
        try std.testing.expectError(error.BadValue, check(&tokens));
    }
    for ([_][]const u8{ "nan", "inf", "heavy" }) |scale| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", "--guidance", scale };
        try std.testing.expectError(error.BadValue, check(&tokens));
    }
}

test "an unknown option or extra argument is refused and named" {
    var refused: ?[]const u8 = null;
    const unknown = [_][]const u8{ "pic.png", "--prompt", "a fox", "--cfg", "7" };
    try std.testing.expectError(error.UnknownOption, parse(&unknown, &refused));
    try std.testing.expectEqualStrings("--cfg", refused.?);
    const extra = [_][]const u8{ "pic.png", "second.png", "--prompt", "a fox" };
    try std.testing.expectError(error.UnknownOption, parse(&extra, &refused));
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
    const swallowed = [_][]const u8{ "pic.png", "--prompt", "a fox", "--negative", "--slg-layers", "0.1" };
    try std.testing.expectError(error.SkipLayerGuidanceRefused, parse(&swallowed, &refused));
    try std.testing.expectEqualStrings("--slg-layers", refused.?);
    const missing = [_][4][]const u8{
        .{ "pic.png", "--prompt", "--width", "1024" },
        .{ "pic.png", "--prompt", "--cfg", "7" },
    };
    for (missing) |tokens| try std.testing.expectError(error.MissingValue, check(&tokens));
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
    const spellings = [_][]const u8{ "--slg", "--slg-layers", "--slg_scale", "--slg-start", "--slg_stop", "slg", "slg_layers", "slg_scale", "slg_start", "slg_stop" };
    for (spellings) |token| {
        const tokens = [_][]const u8{ "pic.png", "--prompt", "a fox", token, "0.5" };
        try std.testing.expectError(error.SkipLayerGuidanceRefused, parse(&tokens, &refused));
        try std.testing.expectEqualStrings(token, refused.?);
    }
}
