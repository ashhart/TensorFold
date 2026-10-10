//! ``tensorfold-native``: ``capabilities --json`` for the Python switch, and ``serve MODEL [flags]`` with no Python.
const std = @import("std");
const json = @import("json.zig");
const cli = @import("cli.zig");
const log = @import("log.zig");
const hub = @import("hub.zig");
const engines = @import("engines.zig");
const serve = @import("serve.zig");
const hf_text = @import("hf_text.zig");
const startup = @import("startup.zig");
const checkpoint_cli = @import("checkpoint_cli");

const usage_line = "usage: tensorfold serve [-h] [--host HOST] [--port PORT] [--name NAME] [--chat-template FILE] [--alias ALIAS] [--api-key API_KEY] [--api-key-file API_KEY_FILE] [--metrics-open] [--dashboard] [--context CONTEXT] [--speed-up SETTINGS] [--prompt-cache-gib PROMPT_CACHE_GIB] [--prompt-cache-over-cap] [--learn] [--learn-dir LEARN_DIR] [--learn-gib LEARN_GIB] [--max-tokens MAX_TOKENS] [--temperature TEMPERATURE] [--top-p TOP_P] [--top-k TOP_K] [--min-p MIN_P] [--thinking | --no-thinking] [--reasoning-effort {low,medium,high,xhigh}] [--thinking-budget THINKING_BUDGET] [--loop-guard] [--no-drafts] [--drafter DRAFTER] [--drafter-bits {0,4}] [--keep-warm SECONDS] [--compact-at COMPACT_AT] [--compact-keep COMPACT_KEEP] [--compact-memory COMPACT_MEMORY] [--slide] [--slide-graph SLIDE_GRAPH] [--parallel PARALLEL] [--no-update-check] [--backend {auto,mlx,cuda}] [--device DEVICE] [--segments SEGMENTS] model\n";

/// The flags past the common ones this binary's backend serves; cli.zig reads them from the root.
pub const serves = engines.serves;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    // These commands must work without a model, driver, GPU or checkout.
    if (argv.len == 2 and std.mem.eql(u8, argv[1], "--version")) {
        try std.Io.File.stdout().writeStreamingAll(io, "tensorfold-native " ++ @import("build_options").version ++ "\n");
        return 0;
    }
    if (argv.len == 2 and (std.mem.eql(u8, argv[1], "--help") or std.mem.eql(u8, argv[1], "-h"))) {
        try std.Io.File.stdout().writeStreamingAll(io, "usage: tensorfold-native --version | capabilities --json | models | info MODEL | pull REPO[@REVISION] | serve MODEL [flags]\n" ++ usage_line);
        return 0;
    }
    if (argv.len >= 3 and std.mem.eql(u8, argv[1], "serve")) {
        for (argv[2..]) |arg| if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try std.Io.File.stdout().writeStreamingAll(io, usage_line);
            return 0;
        };
    }
    // the checkpoint commands need no model, GPU or driver either
    if (argv.len >= 2 and checkpoint_cli.wants(@ptrCast(argv[1..]))) return checkpoint_cli.main(init, argv[1..]);
    log.init(io, false);
    if (argv.len == 3 and std.mem.eql(u8, argv[1], "capabilities") and std.mem.eql(u8, argv[2], "--json")) {
        var out: std.Io.Writer.Allocating = .init(a);
        try cli.capabilities(&out.writer, engines.capabilities(a));
        try std.Io.File.stdout().writeStreamingAll(io, out.written());
        return 0;
    }
    if (argv.len < 2 or !std.mem.eql(u8, argv[1], "serve")) {
        std.debug.print("usage: tensorfold-native capabilities --json | models | info MODEL | pull REPO[@REVISION] | serve MODEL [flags]\n", .{});
        return 2;
    }
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    var u: cli.Usage = .{};
    var args = cli.parse(a, argv[2..], &u) catch |e| switch (e) {
        error.Usage => {
            std.debug.print("{s}tensorfold serve: error: {s}\n", .{ usage_line, u.message });
            return 2;
        },
        else => |x| return x,
    };
    var problem: []const u8 = "";
    const dir = try hub.resolve(a, io, init.environ_map, args.model, &problem) orelse return fail(problem);
    // A drafter is named like a model: a directory, or a repo id `tensorfold pull` cached.
    if (args.drafter) |d| args.drafter = try hub.resolve(a, io, init.environ_map, d, &problem) orelse return fail(problem);
    const model_type = modelType(a, io, dir);
    if (args.rank > 0) { // a rank above 0 serves no HTTP and loads no text: it runs rank 0's steps
        if (!try engines.follow(a, gpa, io, dir, model_type, args, &problem)) return fail(problem);
        return 0;
    }
    var text_arena: std.heap.ArenaAllocator = .init(gpa); // the text's problem, written on its own thread
    defer text_arena.deinit();
    var text_problem: []const u8 = "";
    const up = startup.both(io, loadText, .{ gpa, io, dir, args.chat_template, text_arena.allocator(), &text_problem }, engines.open, .{ a, gpa, io, dir, model_type, args, &problem }) catch |e| {
        if (text_problem.len > 0) return fail(text_problem);
        if (problem.len > 0) return fail(problem); // an engine that failed after saying why
        return e;
    } orelse return fail(problem);
    defer up.text.deinit();
    defer up.engine.close(up.engine.ctx);
    return serve.run(gpa, io, args, .{
        .engine = up.engine.engine,
        .text = up.text.text(),
        .served = hub.servedName(args.name, args.model, dir),
        .sampling = try sampling(a, io, dir, args),
        .environ = init.environ_map,
        .started = started,
    });
}

/// HfText.load with every failure named in `problem`, so main tells the text's failure from the engine's.
fn loadText(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, template: ?[]const u8, pa: std.mem.Allocator, problem: *[]const u8) !*hf_text.HfText {
    return hf_text.HfText.load(gpa, io, dir, template, pa, problem) catch |e| {
        if (problem.len == 0) problem.* = @errorName(e);
        return e;
    };
}

fn fail(message: []const u8) u8 {
    std.debug.print("tensorfold: {s}\n", .{message});
    return 1;
}

fn modelType(a: std.mem.Allocator, io: std.Io, dir: []const u8) []const u8 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return "unknown";
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return "unknown";
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return "unknown";
    if (doc != .object) return "unknown";
    const t = doc.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

/// generation_config.json's sampling (``do_sample`` false is greedy), then the serve flags over it.
fn sampling(a: std.mem.Allocator, io: std.Io, dir: []const u8, args: cli.Args) !?json.Value {
    const out = try json.newObject(a);
    const path = try std.fs.path.join(a, &.{ dir, "generation_config.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20))) |bytes| {
        if ((try json.parse(a, bytes)) == .ok) {
            const cfg = (try json.parse(a, bytes)).ok;
            for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
            if (cfg.get("do_sample")) |d| if (d == .bool) {
                if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
            };
        }
    } else |_| {}
    if (args.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (args.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (args.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (args.min_p) |t| try out.put(a, "min_p", .{ .float = t });
    return if (out.count() > 0) json.Value{ .object = out } else null;
}
