//! GLM-5.3 on the pair without the server: rank 0 runs a prompt and its greedy reply, rank 1 follows.
const std = @import("std");
const platform = @import("sys");
const tf = @import("tensorfold");
const tok = @import("tokenizer");
const g = tf.glm_dsa;
const a = std.heap.page_allocator;

const usage =
    \\tf-glm-dsa-run MODEL_DIR SETTINGS_JSON PROMPT (a .txt file: a user message in GLM's chat format; else token ids)
    \\GLM_LAYERS (the first N layers), GLM_MAX (reply tokens, 64), GLM_CONTEXT (8192), GLM_DEPTH (MTP drafts a round, 0),
    \\GLM_LIMIT_GB (this Mac's load limit, 70% of RAM), GLM_REPEAT (runs of the prompt, 1), GLM_FORCED=1 (the prompt's
    \\next-token agreement, each row's pick against the token after it), GLM_CHUNK (prompt rows a window, 256)
;

fn env(name: [:0]const u8, default: u64) u64 {
    const v = platform.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u64, std.mem.span(v), 10) catch default;
}

fn seconds(io: std.Io, t0: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(t0.durationTo(std.Io.Timestamp.now(io, .awake)).nanoseconds)) / 1e9;
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    if (argv.len < 4) {
        std.debug.print("{s}\n", .{usage});
        return error.Usage;
    }
    var ram: u64 = 0;
    var len: usize = 8;
    if (platform.c.sysctlbyname("hw.memsize", &ram, &len, null, 0) != 0) return error.NoRamSize;
    const limit = env("GLM_LIMIT_GB", ram * 7 / 10 / 1_000_000_000) * 1_000_000_000;
    const layers: ?u32 = if (platform.c.getenv("GLM_LAYERS") != null) @intCast(env("GLM_LAYERS", 78)) else null;
    const depth: u32 = @intCast(env("GLM_DEPTH", 0));
    const e = try g.engine.Engine.load(a, io, argv[1], argv[2], layers, depth > 0, @intCast(env("GLM_CONTEXT", 8192)), 1, limit);
    defer e.deinit();
    std.debug.print("rank {d}: loaded in {d:.1} s, {d:.1} GB on the GPU ({d:.1} GB allocated)\n", .{ e.rank(), e.load_seconds, @as(f64, @floatFromInt(e.m.bytes)) / 1e9, @as(f64, @floatFromInt(e.device.allocated())) / 1e9 });
    if (e.rank() == 1) return g.follow.follow(e);
    const text = try std.Io.Dir.cwd().readFileAlloc(io, argv[3], a, .limited(16 << 20));
    var t = try tok.loadTokenizer(io, a, try std.fmt.allocPrint(a, "{s}/tokenizer.json", .{argv[1]}));
    var prompt: std.ArrayList(u32) = .empty;
    if (std.mem.endsWith(u8, argv[3], ".txt")) {
        const chat = try std.fmt.allocPrint(a, "[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{s}<|assistant|><think>", .{std.mem.trim(u8, text, " \n")});
        try prompt.appendSlice(a, try t.encode(a, chat));
    } else {
        var it = std.mem.tokenizeAny(u8, text, " \n\r\t,[]");
        while (it.next()) |w| try prompt.append(a, try std.fmt.parseInt(u32, w, 10));
    }
    const max = env("GLM_MAX", 64);
    for (0..env("GLM_REPEAT", 1)) |_| try once(e, io, &t, prompt.items, max, depth);
}

fn once(e: *g.engine.Engine, io: std.Io, t: *const tok.Tokenizer, prompt: []const u32, max: u64, depth: u32) !void {
    const forced = platform.c.getenv("GLM_FORCED") != null;
    var agree: usize = 0;
    const t0 = std.Io.Timestamp.now(io, .awake);
    try g.follow.begin(e, 0, prompt);
    var at: u32 = 0;
    var first: u32 = 0;
    while (at < prompt.len) {
        const n: u32 = @intCast(@min(env("GLM_CHUNK", g.gpu.prompt_rows), prompt.len - at));
        if (try g.follow.chunk(e, 0, at, n, forced)) |f| first = f;
        if (forced) for (e.picks(n), 0..) |p, r| {
            if (at + r + 1 < prompt.len and p == prompt[at + r + 1]) agree += 1;
        };
        at += n;
    }
    const t_prompt = seconds(io, t0);
    var reply: std.ArrayList(u32) = .empty;
    try reply.append(a, first);
    if (depth > 0) try g.follow.draft(e, &.{.{ .slot = 0, .prompt = true, .follow = &.{first}, .depth = depth }});
    var rounds: usize = 0;
    var gpu: f64 = 0;
    var drafted: usize = 0;
    var landed: usize = 0;
    var out: [16]u32 = undefined;
    const t1 = std.Io.Timestamp.now(io, .awake);
    while (reply.items.len < max and std.mem.indexOfScalar(u32, &e.m.c.eos, reply.items[reply.items.len - 1]) == null) {
        var toks: [16]u32 = undefined;
        toks[0] = reply.items[reply.items.len - 1];
        const h = e.held(0);
        @memcpy(toks[1..][0..h.len], h);
        const n = 1 + h.len;
        try g.follow.window(e, &.{.{ .slot = 0, .tokens = toks[0..n] }}, &out);
        gpu += e.gpu_ms;
        var kept: u32 = 1;
        while (kept < n and out[kept - 1] == toks[kept]) kept += 1;
        try g.follow.keep(e, 0, kept);
        try reply.appendSlice(a, out[0..kept]);
        rounds += 1;
        drafted += n - 1;
        landed += kept - 1;
        if (depth > 0) try g.follow.draft(e, &.{.{ .slot = 0, .prompt = false, .follow = out[0..kept], .depth = depth }});
    }
    const t_reply = seconds(io, t1);
    g.follow.release(e, 0);
    std.debug.print("reply ids:", .{});
    for (reply.items) |id| std.debug.print(" {d}", .{id});
    std.debug.print("\nreply: {s}\n", .{try t.decode(a, reply.items, false)});
    if (forced) std.debug.print("teacher-forced: {d} of {d} prompt rows pick the next prompt token\n", .{ agree, prompt.len - 1 });
    std.debug.print("\nprompt {d} tokens in {d:.2} s ({d:.1} tok/s); reply {d} tokens in {d} rounds, {d:.2} s ({d:.2} tok/s); drafts {d} landed of {d}\n", .{ prompt.len, t_prompt, @as(f64, @floatFromInt(prompt.len)) / t_prompt, reply.items.len, rounds, t_reply, @as(f64, @floatFromInt(reply.items.len - 1)) / t_reply, landed, drafted });
    std.debug.print("windows: {d:.2} ms of GPU time a round, {d:.2} ms a round in all\n", .{ gpu / @as(f64, @floatFromInt(@max(rounds, 1))), t_reply * 1000 / @as(f64, @floatFromInt(@max(rounds, 1))) });
}
