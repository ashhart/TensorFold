//! `tensorfold lanes`: a prompts file through the lane core on CUDA, every stream at once or one at a time.

const std = @import("std");
const lanes = @import("lanes");
const core = @import("core");
const nemotron = @import("nemotron");

pub const Options = struct {
    prompts: []const u8,
    max_tokens: u32,
    sampling: ?lanes.Sampling,
    drafts: bool,
    solo: bool,
    report: ?[]const u8,
};

const Done = struct { name: []const u8, tokens: []const u32, sha: []const u8, rounds: u64, drafted: u64, accepted: u64 };

/// The capture's prompts.json: names to token ids, in file order.
pub fn readPrompts(gpa: std.mem.Allocator, io: std.Io, path: []const u8) !std.json.Parsed(std.json.ArrayHashMap([]u32)) {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 26));
    defer gpa.free(text);
    return std.json.parseFromSlice(std.json.ArrayHashMap([]u32), gpa, text, .{ .allocate = .alloc_always });
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, e: *nemotron.Engine, model_dir: []const u8, o: Options) !u8 {
    var prompts = try readPrompts(gpa, io, o.prompts);
    defer prompts.deinit();
    const names = prompts.value.map.keys();
    const ids = prompts.value.map.values();
    const head: ?*nemotron.Head = if (o.drafts) try nemotron.Head.init(e) else null;
    defer if (head) |h| h.deinit();
    var cuda = try nemotron.Lanes.init(gpa, e, head);
    defer cuda.deinit();
    try cuda.measure(io, model_dir);
    const rows: u32 = if (head != null) nemotron.state.max_rows else 1;
    var cfg = try lanes.Config.init(gpa, cuda.facts(), rows, rows - 1);
    defer cfg.deinit(gpa);
    var clock = lanes.backend.WallClock{ .io = io };
    const streams = try gpa.alloc(lanes.Stream, names.len);
    defer gpa.free(streams);
    for (streams, names, ids) |*s, name, p| s.* = try lanes.Stream.init(gpa, .{ .id = name, .prompt = p, .max_new = o.max_tokens, .eos = e.c.eos[0..e.c.eos_count], .sampling = o.sampling, .drafts = o.drafts });
    defer for (streams) |*s| s.deinit(gpa);
    const t0 = std.Io.Clock.awake.now(io);
    if (o.solo) {
        for (streams) |*s| try finish(gpa, &cfg, &cuda, clock.clock(), &.{s});
    } else {
        const all = try gpa.alloc(*lanes.Stream, streams.len);
        defer gpa.free(all);
        for (all, streams) |*a, *s| a.* = s;
        try finish(gpa, &cfg, &cuda, clock.clock(), all);
    }
    const seconds = nemotron.engine.seconds(io, t0);
    var done: std.ArrayList(Done) = .empty;
    defer {
        for (done.items) |d| gpa.free(d.sha);
        done.deinit(gpa);
    }
    var tokens: usize = 0;
    for (streams, names) |*s, name| {
        const text = try core.ids_json.write(gpa, s.emitted());
        defer gpa.free(text);
        var digest: [32]u8 = undefined;
        std.crypto.hash.sha2.Sha256.hash(text, &digest, .{});
        const hex = std.fmt.bytesToHex(digest, .lower);
        try done.append(gpa, .{ .name = name, .tokens = s.emitted(), .sha = try gpa.dupe(u8, hex[0..12]), .rounds = s.rounds, .drafted = s.drafted, .accepted = s.accepted });
        tokens += s.emitted().len;
        std.debug.print("{s}: {d} tokens sha {s} rounds {d} accepted {d}\n", .{ name, s.emitted().len, hex[0..12], s.rounds, s.accepted });
    }
    std.debug.print("{s}: {d} streams, {d} tokens in {d:.3} s\n", .{ if (o.solo) "solo" else "together", streams.len, tokens, seconds });
    if (o.report) |path| {
        const report = .{ .mode = if (o.solo) "solo" else "together", .seconds = seconds, .sampling = o.sampling, .drafts = o.drafts, .streams = done.items };
        const json = try std.json.Stringify.valueAlloc(gpa, report, .{});
        defer gpa.free(json);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
    }
    return 0;
}

/// Admit `streams` (each prefilled and drawing its first token), then step rounds until every one has finished.
fn finish(gpa: std.mem.Allocator, cfg: *const lanes.Config, cuda: *nemotron.Lanes, clock: lanes.backend.Clock, streams: []const *lanes.Stream) !void {
    var engine = lanes.Engine.init(gpa, cfg, cuda.backend(), clock);
    defer engine.deinit();
    for (streams) |s| try engine.addStream(s);
    while (engine.live.items.len > 0) try engine.step();
}
