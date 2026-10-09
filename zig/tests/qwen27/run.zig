//! The real-weight gate. Help returns before any device or checkpoint access.
const std = @import("std");
const q = @import("qwen27");
const mtl = @import("metal");
const sources = @import("qwen_runtime_sources");
const gdn_source = @import("qwen_gdn_source");
const profile = @import("core").gpu_profile;
const core_lane = @import("core_lane");

const help =
    \\tf-qwen27-run --run --model DIR --tokens IDS.json [--max-tokens 256]
    \\  [--chunk 128] [--compare-chunk 16] [--reference GENERATED.json] [--force-length]
    \\  [--no-warmup] [--profile-steps N] [--teacher]
    \\IDs and reference files are JSON arrays of unsigned token IDs, already chat-rendered.
    \\Greedy only, drafts off; JSON receipt on stdout, errors on stderr.
    \\--teacher feeds every reference token and records native top2 at all positions; margin timing is diagnostic.
    \\--max-tokens 0 reports prompt tok/s including the final single-row head.
    \\--compare-chunk reruns the whole prompt and decode and checks prompt state/KV/logits bytes.
    \\CPU only: --help or --emit-sources EXISTING_DIR, no Metal device or model load.
;

fn tokens(a: std.mem.Allocator, io: std.Io, file: []const u8) ![]const u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, file, a, .limited(16 << 20));
    return (try std.json.parseFromSlice([]u32, a, text, .{})).value;
}
fn write(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, text: []const u8) !void {
    const path = try std.fs.path.join(a, &.{ dir, name });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = text });
}
fn emit(a: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    try write(a, io, dir, "glue.metal", sources.glue ++ sources.copy);
    try write(a, io, dir, "attention.metal", sources.glue ++ sources.attention_mpp ++ sources.attention_io);
    try write(a, io, dir, "gdn.metal", gdn_source.text);
    try write(a, io, dir, "prompt.metal", sources.prompt_glue ++ sources.prompt_state);
    for ([_][]const u8{ "norm_input", "norm_xs", "mlp_xs", "xsum" }) |name| {
        const template = if (std.mem.eql(u8, name, "norm_input")) sources.norm_input else if (std.mem.eql(u8, name, "norm_xs")) sources.norm_xs else if (std.mem.eql(u8, name, "mlp_xs")) sources.mlp_xs else sources.xsum;
        const mark = std.mem.indexOf(u8, template, "Q27_SPECIALIZE") orelse return error.BadKernelSource;
        const defines = if (std.mem.eql(u8, name, "mlp_xs")) "constexpr int N=17408;" else "constexpr int K=5120, GS=64;";
        const text = try std.mem.concat(a, u8, &.{ template[0..mark], defines, template[mark + "Q27_SPECIALIZE".len ..] });
        try write(a, io, dir, try std.fmt.allocPrint(a, "{s}.metal", .{name}), text);
    }
    const shapes = [_][3]usize{ .{ 10240, 5120, 4 }, .{ 6240, 5120, 8 }, .{ 5120, 6144, 8 }, .{ 12288, 5120, 4 }, .{ 2048, 5120, 8 }, .{ 34816, 5120, 2 }, .{ 5120, 17408, 8 }, .{ 248320, 5120, 1 } };
    for (shapes, 0..) |shape, i| {
        const text = try core_lane.source(a, .{ .n = shape[0], .k = shape[1], .sk = shape[2], .format = .{ .bits = 4, .group = 64 }, .precompute_sums = true, .cooperative = true });
        try write(a, io, dir, try std.fmt.allocPrint(a, "lane-{d}.metal", .{i}), text);
    }
}
const Difference = struct { position: usize, expected: ?u32, actual: u32, top2: q.session.TopTwo };
const Generated = struct { tokens: []u32, first_difference: ?Difference, top2: ?[]q.session.TopTwo };
fn generated(a: std.mem.Allocator, s: q.session.Session, count: usize, force: bool, expected: ?[]const u32, trace: ?*profile.Trace, profile_steps: usize, teacher: bool) !Generated {
    var out: std.ArrayList(u32) = .empty;
    var difference: ?Difference = null;
    const margins = if (teacher) try a.alloc(q.session.TopTwo, count) else null;
    for (0..count) |i| {
        const token = try s.greedy();
        if (margins) |values| values[i] = try q.session.topTwo(s.runner.model.frame.get(.logits).buffer.slice(u16, s.runner.model.config.vocab));
        if (expected) |wanted| if (difference == null and (i >= wanted.len or token != wanted[i])) {
            difference = .{ .position = i, .expected = if (i < wanted.len) wanted[i] else null, .actual = token, .top2 = try q.session.topTwo(s.runner.model.frame.get(.logits).buffer.slice(u16, s.runner.model.config.vocab)) };
        };
        try out.append(a, token);
        if ((!teacher and !force and s.runner.model.config.isEos(token)) or i + 1 == count) break;
        if (trace) |t| t.enabled = i < profile_steps;
        try s.step(if (teacher) expected.?[i] else token);
    }
    if (trace) |t| t.enabled = false;
    return .{ .tokens = try out.toOwnedSlice(a), .first_difference = difference, .top2 = margins };
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(a);
    if (args.len == 1 or (args.len == 2 and std.mem.eql(u8, args[1], "--help"))) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    if (args.len == 3 and std.mem.eql(u8, args[1], "--emit-sources")) {
        try emit(a, io, args[2]);
        return;
    }
    var model: ?[]const u8 = null;
    var file: ?[]const u8 = null;
    var reference: ?[]const u8 = null;
    var profile_steps: usize = 0;
    var count: usize = 256;
    var chunk: usize = 128;
    var compare: usize = 0;
    var run = false;
    var force = false;
    var warmup = true;
    var teacher = false;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--run")) {
            run = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--force-length")) {
            force = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--teacher")) {
            teacher = true;
            continue;
        }
        if (std.mem.eql(u8, arg, "--no-warmup")) {
            warmup = false;
            continue;
        }
        if (i + 1 >= args.len) return error.MissingValue;
        i += 1;
        const value = args[i];
        if (std.mem.eql(u8, arg, "--model")) model = value else if (std.mem.eql(u8, arg, "--tokens")) file = value else if (std.mem.eql(u8, arg, "--reference")) reference = value else if (std.mem.eql(u8, arg, "--max-tokens")) count = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--chunk")) chunk = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--profile-steps")) profile_steps = try std.fmt.parseInt(usize, value, 10) else if (std.mem.eql(u8, arg, "--compare-chunk")) compare = try std.fmt.parseInt(usize, value, 10) else return error.UnknownOption;
    }
    if (!run or model == null or file == null or chunk == 0 or chunk > 128 or compare > 128 or count > 65536) return error.BadOptions;
    const ids = try tokens(a, io, file.?);
    if (ids.len == 0 or ids.len + count > 262144) return error.BadPrompt;
    const expected = if (reference) |path| try tokens(a, io, path) else null;
    if (teacher and (expected == null or expected.?.len != count)) return error.BadTeacherReference;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const load_start = std.Io.Clock.awake.now(io);
    const m = try q.model.Model.load(init.gpa, io, model.?, 128);
    defer m.deinit();
    for (ids) |token| if (token >= m.config.vocab) return error.BadTokenId;
    if (expected) |values| for (values) |token| if (token >= m.config.vocab) return error.BadTokenId;
    if (ids.len + count > m.config.max_position) return error.ContextFull;
    var runner = try q.decode_round.Runner.init(init.gpa, m, 1, @intCast(@max(ids.len + count, 128)));
    defer runner.deinit();
    const s = q.session.Session{ .runner = &runner };
    const load_ns = load_start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    if (warmup) {
        try s.prefill(ids[0..@min(ids.len, 128)], 128);
        try runner.reset(0);
    }
    const start = std.Io.Clock.awake.now(io);
    try s.prefill(ids, chunk);
    const prompt_ns = start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const digest = try s.fingerprint();
    var trace = profile.Trace{ .queue = m.queue };
    if (profile_steps > 0) runner.profile(&trace);
    const decode_start = std.Io.Clock.awake.now(io);
    const generation = try generated(a, s, count, force, expected, if (profile_steps > 0) &trace else null, profile_steps, teacher);
    const out = generation.tokens;
    const decode_ns = decode_start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds();
    const reference_equal = if (expected) |wanted| std.mem.eql(u32, out, wanted) else false;
    if (compare > 0) {
        try runner.reset(0);
        try s.prefill(ids, compare);
        if (!std.mem.eql(u8, &digest, &try s.fingerprint())) return error.PromptStateOrLogitsDiffer;
        const second = try generated(a, s, count, force, if (teacher) expected else null, null, 0, teacher);
        if (!std.mem.eql(u32, out, second.tokens)) return error.ChunkedTokensDiffer;
    }
    const digest_hex = std.fmt.bytesToHex(digest, .lower);
    const ProfileRecord = struct { class: []const u8, calls: u64, gpu_seconds: f64 };
    var timings: [@typeInfo(profile.Class).@"enum".field_names.len]ProfileRecord = undefined;
    for (std.enums.values(profile.Class), 0..) |kind, j| timings[j] = .{ .class = @tagName(kind), .calls = trace.calls[j], .gpu_seconds = trace.seconds[j] };
    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .kind = if (teacher) "qwen27_teacher" else "qwen27_greedy",
        .teacher_forced = teacher,
        .top2 = generation.top2,
        .timing_includes_margin_capture = teacher,
        .version = 1,
        .drafts = false,
        .prompt_tokens = ids.len,
        .tokens = out,
        .chunk = chunk,
        .compare_chunk = compare,
        .prompt_state_byte_equal = compare > 0,
        .reference_token_equal = reference_equal,
        .first_difference = generation.first_difference,
        .logit_dtype = "bf16",
        .profile_steps = profile_steps,
        .profile_dense_shapes = if (profile_steps > 0) @as(?[]const profile.Shape, trace.shapes[0..trace.shape_count]) else null,
        .profile_shapes_overflow = trace.shapes_overflow,
        .profile_split_buffers = profile_steps > 0,
        .profile = if (profile_steps > 0) @as(?[]const ProfileRecord, &timings) else null,
        .force_length = force,
        .warmed = warmup,
        .load_ns = load_ns,
        .prompt_ns = prompt_ns,
        .decode_ns = decode_ns,
        .prompt_tok_s = @as(f64, @floatFromInt(ids.len)) * 1e9 / @as(f64, @floatFromInt(@max(prompt_ns, 1))),
        .physical_prompt_rows = 128,
        .projection_rows = 16,
        .prompt_sha256 = @as([]const u8, &digest_hex),
    }, .{});
    try std.Io.File.stdout().writeStreamingAll(io, receipt);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
    if (!teacher and reference != null and !reference_equal) return error.ReferenceTokensDiffer;
}
