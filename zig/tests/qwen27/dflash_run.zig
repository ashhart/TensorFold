//! The draft prototype on real weights: target tokens and final state against ordinary native generation.
const std = @import("std");
const mtl = @import("metal");
const q = @import("qwen27");
const profile = @import("core").gpu_profile;
const help = "tf-qwen27-dflash-run --run --model TARGET --draft DRAFTER --tokens IDS.json --prototype bf16|q4 [--max-tokens 256 --nodes 15 --chunk 128 --tree-block 16 --compare-plain --force-length]";
fn ids(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20));
    return (try std.json.parseFromSlice([]u32, a, text, .{})).value;
}
pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 1 or (args.len == 2 and std.mem.eql(u8, args[1], "--help"))) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    var target_dir: ?[]const u8 = null;
    var draft_dir: ?[]const u8 = null;
    var tokens_file: ?[]const u8 = null;
    var mode: ?q.dflash.operators.Mode = null;
    var count: usize = 256;
    var chunk: usize = 128;
    var nodes: usize = 15;
    var tree_block: u32 = 16;
    var run = false;
    var compare = false;
    var force = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--run")) {
            run = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--compare-plain")) {
            compare = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--force-length")) {
            force = true;
            continue;
        }
        i += 1;
        if (i == args.len) return error.MissingValue;
        const value = args[i];
        if (std.mem.eql(u8, arg, "--model")) target_dir = value else if (std.mem.eql(u8, arg, "--draft")) draft_dir = value else if (std.mem.eql(u8, arg, "--tokens")) tokens_file = value else if (std.mem.eql(u8, arg, "--max-tokens")) count = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--chunk")) chunk = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--nodes")) nodes = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--tree-block")) tree_block = try std.fmt.parseInt(u32, value, 10) else if (std.mem.eql(u8, arg, "--prototype")) {
            mode = if (std.mem.eql(u8, value, "bf16")) .bf16_reference else if (std.mem.eql(u8, value, "q4")) .prepared_q4_reference else return error.BadPrototypeMode;
        } else return error.UnknownOption;
    }
    if (!run or target_dir == null or draft_dir == null or tokens_file == null or mode == null or chunk == 0 or chunk > 128 or nodes == 0 or nodes > 15 or count > 65536 or tree_block == 0 or tree_block > 16) return error.BadOptions;
    const prompt = try ids(a, io, tokens_file.?);
    if (prompt.len == 0 or prompt.len + count > 262144) return error.BadPrompt;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const load_start = std.Io.Clock.awake.now(io);
    const target = try q.model.Model.load(init.gpa, io, target_dir.?, 128);
    defer target.deinit();
    if (prompt.len + count > target.config.max_position) return error.ContextFull;
    for (prompt) |token| if (token >= target.config.vocab) return error.BadTokenId;
    var runner = try q.decode_round.Runner.init(init.gpa, target, 1, @intCast(@max(prompt.len + count + 16, 128)));
    defer runner.deinit();
    const draft = try q.dflash.runtime_model.Model.load(init.gpa, io, target, draft_dir.?, mode.?);
    defer draft.deinit();
    try draft.setTreeBlock(tree_block);
    var generation = try q.dflash.generation.Generation.init(init.gpa, &runner, draft);
    defer generation.deinit();
    const load_ns = load_start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const start = std.Io.Clock.awake.now(io);
    try generation.prefill(prompt, chunk);
    const prompt_ns = start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    var trace = profile.Trace{ .queue = target.queue };
    const profiled = std.c.getenv("QWEN27_PROFILE") != null;
    if (profiled) {
        runner.profile(&trace);
        trace.enabled = true;
    }
    const decode_start = std.Io.Clock.awake.now(io);
    const result = try generation.run(count, force, nodes);
    if (profiled) {
        inline for (@typeInfo(profile.Class).@"enum".field_names, 0..) |name, ci| std.debug.print("PROFILE class={s} calls={d} ms={d:.2}\n", .{ name, trace.calls[ci], trace.seconds[ci] * 1e3 });
        for (trace.shapes[0..trace.shape_count]) |sh| std.debug.print("PROFILE shape n={d} k={d} sk={d} calls={d} ms={d:.2}\n", .{ sh.n, sh.k, sh.sk, sh.calls, sh.gpu_seconds * 1e3 });
    }
    defer init.gpa.free(result.tokens);
    const decode_ns = decode_start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const session = q.session.Session{ .runner = &runner };
    const final = try session.fingerprint();
    var equal = false;
    var state_equal = false;
    var first_difference: ?usize = null;
    var plain_ns: i96 = 0;
    if (compare) {
        try runner.reset(0);
        try session.prefill(prompt, chunk);
        var plain: std.ArrayList(u32) = .empty;
        defer plain.deinit(init.gpa);
        const plain_start = std.Io.Clock.awake.now(io);
        for (0..count) |position| {
            const token = try session.greedy();
            try plain.append(init.gpa, token);
            if ((!force and target.config.isEos(token)) or position + 1 == count) break;
            try session.step(token);
        }
        plain_ns = plain_start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
        equal = std.mem.eql(u32, result.tokens, plain.items);
        state_equal = std.mem.eql(u8, &final, &try session.fingerprint());
        for (result.tokens, 0..) |token, position| if (position >= plain.items.len or token != plain.items[position]) {
            first_difference = position;
            break;
        };
        if (first_difference == null and result.tokens.len != plain.items.len) first_difference = result.tokens.len;
    }
    const record = try std.json.Stringify.valueAlloc(a, .{ .kind = "qwen27_dflash_prototype", .mode = @tagName(mode.?), .default_proposal_parity = false, .checkpoint_block = draft.graph.config.block, .tree_block = draft.tree_block, .draft_vocab_size = draft.backend.head.linear.n, .gpu_topk = true, .gpu_target_selection = true, .gpu_keep = true, .gpu_round = false, .tokens = result.tokens, .drafted_equal_plain = equal, .final_state_equal_plain = state_equal, .first_difference = first_difference, .rounds = result.rounds, .verified_rows = result.verified, .matched_draft_tokens = result.matched, .load_ns = load_ns, .prompt_ns = prompt_ns, .decode_ns = decode_ns, .plain_decode_ns = plain_ns, .tok_s = @as(f64, @floatFromInt(result.tokens.len)) * 1e9 / @as(f64, @floatFromInt(@max(decode_ns, 1))) }, .{});
    try std.Io.File.stdout().writeStreamingAll(io, record);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
    if (compare and (!equal or !state_equal)) return error.DraftPlainMismatch;
}
