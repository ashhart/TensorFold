const std = @import("std");
const mx = @import("mlx.zig");
const sampling = @import("sampling.zig");

pub fn run(comptime M: type, init: std.process.Init, args: []const []const u8) !void {
    const a = init.gpa;
    const io = init.io;
    var prompt: []const u8 = "Write a short Python function that computes the Fibonacci sequence.";
    var token_list: ?[]const u8 = null;
    var max_tokens: usize = 32;
    var settings = sampling.Sampling{};
    var seed_set = false;
    var report: ?[]const u8 = null;
    var dump: ?[]const u8 = null;
    var exact = false;
    var long_cache = false;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const key = args[i];
        if (std.mem.eql(u8, key, "--no-drafts")) continue;
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
        if (std.mem.eql(u8, key, "--prompt")) prompt = value else if (std.mem.eql(u8, key, "--tokens")) token_list = value else if (std.mem.eql(u8, key, "--max-tokens")) max_tokens = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, key, "--temperature")) settings.temperature = try std.fmt.parseFloat(f64, value) else if (std.mem.eql(u8, key, "--top-k")) settings.top_k = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, key, "--top-p")) settings.top_p = try std.fmt.parseFloat(f64, value) else if (std.mem.eql(u8, key, "--seed")) {
            settings.seed = try std.fmt.parseInt(u64, value, 10);
            seed_set = true;
        } else if (std.mem.eql(u8, key, "--report")) report = value else if (std.mem.eql(u8, key, "--dump-logits")) dump = value else return error.UnsupportedArgument;
        i += 1;
    }
    try settings.validate();
    if (@hasDecl(M, "prepareRuntime")) try M.prepareRuntime();
    try mx.init();
    defer mx.shutdown();
    var model = try M.init(io, args[2]);
    defer model.deinit();
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
    var pending: i32 = 0;
    var offset: usize = 0;
    while (offset < tokens.items.len) {
        const count = @min(16, tokens.items.len - offset);
        var pass = try model.forward(tokens.items[offset..][0..count]);
        defer pass.deinit();
        const ids = try sampling.rows(&model.kernels, &pass.scope, try pass.scope.slice(pass.logits, 0, @intCast(count - 1), @intCast(count)), &.{@intCast(offset + count)}, settings);
        defer mx.allocator.free(ids);
        pending = ids[0];
        if (dump) |file| if (offset + count == tokens.items.len) {
            const z = try a.dupeSentinel(u8, file, 0);
            defer a.free(z);
            const logits = try pass.scope.cast(pass.logits, mx.f32t);
            try mx.eval(logits);
            try mx.check(mx.c.mlx_save(z, logits));
        };
        try model.commit(&pass, count);
        offset += count;
    }
    var generated: std.ArrayList(u32) = .empty;
    defer generated.deinit(a);
    while (generated.items.len < max_tokens) {
        try generated.append(a, @intCast(pending));
        const eos = if (@hasDecl(M, "isEos")) model.isEos(pending) else M.eos(pending);
        if (eos or generated.items.len == max_tokens) break;
        var pass = try model.forward(&.{pending});
        defer pass.deinit();
        const ids = try sampling.rows(&model.kernels, &pass.scope, pass.logits, &.{model.position + 1}, settings);
        defer mx.allocator.free(ids);
        pending = ids[0];
        try model.commit(&pass, 1);
    }
    const text = try tokenizer.decode(a, generated.items, false);
    defer a.free(text);
    var buffer: [4096]u8 = undefined;
    var out = std.Io.File.stdout().writer(io, &buffer);
    try out.interface.writeAll(text);
    try out.interface.writeAll("\n");
    try out.interface.flush();
    if (report) |file| {
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(generated.items), &digest, .{});
        const bytes = try std.json.Stringify.valueAlloc(a, .{ .prompt_tokens = tokens.items, .tokens = generated.items, .text = text, .seed = settings.seed, .temperature = settings.temperature, .top_k = settings.top_k, .top_p = settings.top_p, .metal_sampling = settings.metal, .token_sha256 = std.fmt.bytesToHex(digest, .lower) }, .{});
        defer a.free(bytes);
        const f = try std.Io.Dir.cwd().createFile(io, file, .{});
        defer f.close(io);
        try f.writeStreamingAll(io, bytes);
    }
}
