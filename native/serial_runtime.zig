const std = @import("std");
const mx = @import("mlx.zig");
const sampling = @import("sampling.zig");

pub fn run(comptime M: type, init: std.process.Init, args: []const []const u8) !void {
    const a = init.gpa;
    const io = init.io;
    var prompt: []const u8 = "Write a short Python function that computes the Fibonacci sequence.";
    var token_list: ?[]const u8 = null;
    var max_tokens: usize = 32;
    var drafts: usize = if (@hasDecl(M, "propose")) 3 else 0;
    var settings = sampling.Sampling{};
    var seed_set = false;
    var report: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var drafter: ?[]const u8 = null;
    var drafter_bits: i32 = 8;
    var exact = false;
    var long_cache = false;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const key = args[i];
        if (std.mem.eql(u8, key, "--no-drafts")) {
            drafts = 0;
            continue;
        }
        if (std.mem.eql(u8, key, "--metal-simd")) {
            mx.force_simd = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--metal-sampling")) {
            settings.metal = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--check-exact")) {
            exact = true;
            continue;
        }
        if (std.mem.eql(u8, key, "--check-long-cache")) {
            long_cache = true;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingArgument;
        const value = args[i + 1];
        if (std.mem.eql(u8, key, "--drafter-bits")) {
            drafter_bits = try std.fmt.parseInt(i32, value, 10);
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, key, "--drafter")) {
            drafter = value;
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, key, "--mtp-drafts")) {
            drafts = try std.fmt.parseInt(usize, value, 10);
            i += 1;
            continue;
        }
        if (std.mem.eql(u8, key, "--prompt")) prompt = value else if (std.mem.eql(u8, key, "--tokens")) token_list = value else if (std.mem.eql(u8, key, "--max-tokens")) max_tokens = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, key, "--temperature")) settings.temperature = try std.fmt.parseFloat(f64, value) else if (std.mem.eql(u8, key, "--top-k")) settings.top_k = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, key, "--top-p")) settings.top_p = try std.fmt.parseFloat(f64, value) else if (std.mem.eql(u8, key, "--seed")) {
            settings.seed = try std.fmt.parseInt(u64, value, 10);
            seed_set = true;
        } else if (std.mem.eql(u8, key, "--report")) report = value else if (std.mem.eql(u8, key, "--dump-logits")) dump = value else return error.UnsupportedArgument;
        i += 1;
    }
    try settings.validate();
    if (drafts > 15) return error.InvalidDraftBudget;
    if (@hasDecl(M, "prepareRuntime")) try M.prepareRuntime();
    try mx.init();
    defer mx.shutdown();
    var model = try M.init(io, args[2]);
    defer model.deinit();
    if (drafter) |dir| {
        if (@hasDecl(M, "loadDraftBits")) try model.loadDraftBits(io, dir, drafter_bits) else if (@hasDecl(M, "loadDraft")) try model.loadDraft(io, dir) else return error.UnsupportedDrafts;
    }
    if (@hasField(M, "has_mtp")) if (!model.has_mtp) {
        drafts = 0;
    };
    if (exact or long_cache) {
        try model.checkExact(33);
        if (long_cache) for ([_]usize{ 1022, 1150, 2302 }) |prefix| try model.checkExact(prefix);
        return;
    }
    const path = try std.Io.Dir.cwd().realPathFileAlloc(io, args[2], a);
    defer a.free(path);
    var tokenizer = try @import("vendor/tokenizer.zig").loadTokenizer(io, a, path);
    defer tokenizer.deinit();
    var tokens: std.ArrayList(i32) = .empty;
    defer tokens.deinit(a);
    if (token_list) |list| {
        var split = std.mem.splitScalar(u8, list, ',');
        while (split.next()) |value| try tokens.append(a, try std.fmt.parseInt(i32, value, 10));
    } else {
        const ids = try tokenizer.encode(a, prompt);
        defer a.free(ids);
        for (ids) |id| try tokens.append(a, @intCast(id));
    }
    if (tokens.items.len == 0) return error.EmptyPrompt;
    if (tokens.items.len > 262144 or max_tokens > 262144 - tokens.items.len) return error.ContextLimitExceeded;
    const vocab = if (@hasField(M, "vocab")) model.vocab else M.vocab;
    for (tokens.items) |id| if (id < 0 or id >= vocab) return error.InvalidToken;
    if (!seed_set) settings.seed = sampling.seedFor(tokens.items);
    var generated = try @import("serial_generation.zig").generate(&model, tokens.items, max_tokens, settings, drafts, dump);
    defer generated.deinit();
    const text = try tokenizer.decode(a, generated.tokens.items, false);
    defer a.free(text);
    var buffer: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &buffer);
    try out.interface.writeAll(text);
    try out.interface.writeAll("\n");
    try out.interface.flush();
    if (report) |file| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(generated.tokens.items), &digest, .{});
        const bytes = try std.json.Stringify.valueAlloc(a, .{ .prompt_tokens = tokens.items, .tokens = generated.tokens.items, .text = text, .seed = settings.seed, .temperature = settings.temperature, .top_k = settings.top_k, .top_p = settings.top_p, .metal_sampling = settings.metal, .rounds = generated.rounds, .drafted = generated.drafted, .accepted = generated.accepted, .token_sha256 = std.fmt.bytesToHex(digest, .lower) }, .{});
        defer a.free(bytes);
        const f = try std.Io.Dir.cwd().createFile(io, file, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, bytes);
    }
}
