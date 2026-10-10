//! `lanes <model dir> <prompts.json> <max tokens> [--solo] [--no-drafts] [--resume PREFIX] [--tp N --rank R]`

const std = @import("std");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const group_mod = @import("group.zig");
const ids_file = @import("ids_file.zig");
const lanes_session = @import("lanes_session.zig");

const Session = lanes_session.Session;

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 3) return error.MissingArgument;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const max_tokens = try std.fmt.parseInt(u32, args[2], 10);
    var job: Session.Job = .{ .prompts = try ids_file.read(arena, io, args[1]), .max_new = max_tokens };
    var report: ?[]const u8 = null;
    var resume_to: ?[]const u8 = null;
    var group: group_mod.Group = .{};
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        if (try group.option(args, &i)) continue;
        if (std.mem.eql(u8, args[i], "--resume")) {
            i += 1;
            resume_to = args[i];
        } else if (std.mem.eql(u8, args[i], "--solo")) job.solo = true else if (std.mem.eql(u8, args[i], "--no-drafts")) job.drafts = false else if (std.mem.eql(u8, args[i], "--report")) {
            i += 1;
            report = args[i];
        } else if (std.mem.eql(u8, args[i], "--seed")) {
            i += 1;
            job.sampling = .{ .seed = try std.fmt.parseInt(u64, args[i], 10), .temperature = if (job.sampling) |s| s.temperature else 1.0 };
        } else if (std.mem.eql(u8, args[i], "--temperature")) {
            i += 1;
            var s = job.sampling orelse lanes.Sampling{ .seed = 0 };
            s.temperature = try std.fmt.parseFloat(f64, args[i]);
            job.sampling = s;
        } else return error.UnknownOption;
    }
    var joined = try group.join(io);
    defer joined.close();
    const e = try group.engine(gpa, io, args[0], joined, .{ .capacity = ids_file.longest(job.prompts) + 2 * max_tokens + 64, .prompt_rows = ids_file.longest(job.prompts) + 2 * max_tokens + 64, .streams = job.prompts.len, .batch_rows = @max(32, job.prompts.len * qwen35.hip_lanes.Hip.max_window) });
    defer e.deinit();
    if (group.rank > 0) return group_mod.follow(gpa, e, &joined);
    var s: Session = undefined;
    try s.init(gpa, io, e, if (group.world > 1) &joined.link else null);
    defer s.deinit();
    if (resume_to != null) s.h.keepPrompts(8, 1 << 30);
    const done = try s.run(arena, job);
    for (done.replies) |r| std.debug.print("{s}: {d} tokens, rounds {d}, accepted {d}\n", .{ r.name, r.tokens.len, r.rounds, r.accepted });
    std.debug.print("{s}: {d} streams, {d} tokens in {d:.3} s, {d:.1} tok/s\n", .{ if (job.solo) "solo" else "together", done.replies.len, done.tokens(), done.seconds, @as(f64, @floatFromInt(done.tokens())) / done.seconds });
    if (report) |path| try writeJson(gpa, io, path, try named(arena, done.replies));
    if (resume_to) |prefix| try again(gpa, io, &s, arena, prefix, job, done);
}

fn writeJson(gpa: std.mem.Allocator, io: std.Io, path: []const u8, value: anytype) !void {
    const json = try std.json.Stringify.valueAlloc(gpa, value, .{});
    defer gpa.free(json);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = json });
}

const Named = struct { name: []const u8, tokens: []const u32 };
const Cached = struct { name: []const u8, cached: u32, tokens: []const u32 };

fn named(arena: std.mem.Allocator, rs: []const Session.Reply) ![]const Named {
    const out = try arena.alloc(Named, rs.len);
    for (out, rs) |*o, r| o.* = .{ .name = r.name, .tokens = r.tokens };
    return out;
}

/// Each prompt again, extended by its reply and its own first tokens, one at a time on the kept caches.
fn again(gpa: std.mem.Allocator, io: std.Io, s: *Session, arena: std.mem.Allocator, prefix: []const u8, first: Session.Job, done: Session.Done) !void {
    const extended = try lanes_session.extend(arena, first.prompts, done.replies);
    var second = first;
    second.prompts = extended;
    second.solo = true;
    const out = try s.run(arena, second);
    const kept = try arena.alloc(Cached, out.replies.len);
    const prompts = try arena.alloc(Named, extended.len);
    for (kept, prompts, out.replies, extended) |*k, *n, r, p| {
        std.debug.print("{s} again: {d} prompt tokens, {d} from the kept caches, {d} tokens\n", .{ r.name, p.ids.len, r.cached, r.tokens.len });
        k.* = .{ .name = r.name, .cached = r.cached, .tokens = r.tokens };
        n.* = .{ .name = p.name, .tokens = p.ids };
    }
    try writeJson(gpa, io, try std.fmt.allocPrint(arena, "{s}-prompts.json", .{prefix}), prompts);
    try writeJson(gpa, io, try std.fmt.allocPrint(arena, "{s}-replies.json", .{prefix}), kept);
}
