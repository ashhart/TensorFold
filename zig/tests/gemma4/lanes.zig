//! Gemma 4 dense on the lane core: copy-drafted and 4- and 8-stream shared rounds against the engine's plain replies.
const std = @import("std");
const tf = @import("tensorfold");

const lanes = tf.lanes;
const g4 = tf.gemma4;

const Prompt = struct { name: []const u8, ids: []const u32, want: []const u32 };

fn ms(io: std.Io, since: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(since.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
}

fn readJson(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, suffix: []const u8) ![]u32 {
    var p: [512]u8 = undefined;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.bufPrint(&p, "{s}/{s}{s}", .{ dir, name, suffix }), gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    const parsed = try std.json.parseFromSlice([]u32, gpa, bytes, .{});
    defer parsed.deinit();
    return gpa.dupe(u32, parsed.value);
}

/// The engine's own plain reply to `ids` (one stream, no drafts): the reference drafted runs must equal.
fn plain(gpa: std.mem.Allocator, engine: *lanes.Engine, ids: []const u32, count: u32) ![]u32 {
    var s = try lanes.Stream.init(gpa, .{ .id = "plain", .prompt = ids, .max_new = count, .drafts = false });
    defer s.deinit(gpa);
    try engine.addStream(&s);
    while (engine.activeCount() > 0) try engine.step();
    return gpa.dupe(u32, s.emitted());
}

/// Runs `streams` copies of the prompts (round-robin) together; returns how many streams' tokens differ.
fn case(gpa: std.mem.Allocator, io: std.Io, engine: *lanes.Engine, prompts: []const Prompt, streams: usize, copies: bool, label: []const u8) !usize {
    const ss = try gpa.alloc(lanes.Stream, streams);
    defer gpa.free(ss);
    const looks = try gpa.alloc(lanes.SuffixLookup, streams);
    defer gpa.free(looks);
    for (ss, looks, 0..) |*s, *l, i| {
        const p = prompts[i % prompts.len];
        l.* = try lanes.SuffixLookup.init(gpa, .{ .min_match = 4 });
        s.* = try lanes.Stream.init(gpa, .{ .id = p.name, .prompt = p.ids, .max_new = @intCast(p.want.len), .drafts = copies, .proposer = if (copies) l.proposer() else null });
    }
    defer for (ss, looks) |*s, *l| {
        s.deinit(gpa);
        l.deinit();
    };
    const t0 = std.Io.Clock.awake.now(io);
    for (ss) |*s| try engine.addStream(s);
    while (engine.activeCount() > 0) try engine.step();
    const elapsed = ms(io, t0);
    var bad: usize = 0;
    var tokens: usize = 0;
    var rounds: usize = 0;
    for (ss, 0..) |*s, i| {
        const p = prompts[i % prompts.len];
        const ok = std.mem.eql(u32, s.emitted(), p.want);
        bad += @intFromBool(!ok);
        tokens += s.emitted().len;
        rounds += s.rounds;
        if (!ok) {
            const at = for (s.emitted(), p.want, 0..) |a, b, j| {
                if (a != b) break j;
            } else @min(s.emitted().len, p.want.len);
            std.debug.print("  {s} stream {d} ({s}) differs at token {d}\n", .{ label, i, p.name, at });
        }
    }
    std.debug.print("{s}: {d} of {d} streams equal their reference, {d} tokens in {d} stream rounds ({d:.2} a round), {d:.0} tok/s\n", .{ label, streams - bad, streams, tokens, rounds, @as(f64, @floatFromInt(tokens)) / @as(f64, @floatFromInt(@max(rounds, 1))), @as(f64, @floatFromInt(tokens)) * 1e3 / elapsed });
    return bad;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    var args = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args.deinit();
    _ = args.next();
    const model_dir = args.next() orelse return error.Usage;
    const oracle = args.next() orelse return error.Usage;
    const extra = args.next(); // a directory with repeat.ids.json and edit.ids.json: copy-heavy prompts
    const m = try g4.Model.load(gpa, io, model_dir);
    defer m.deinit();
    const metal = try g4.backend.Metal.init(gpa, m, .{ .capacity = 4096, .streams = 8 });
    defer metal.deinit();
    var cfg = try lanes.Config.init(gpa, metal.facts(), g4.state.max_rows, g4.state.max_rows - 1);
    defer cfg.deinit(gpa);
    var clock = lanes.backend.WallClock{ .io = io };
    var prompts: [4]Prompt = undefined;
    for (&prompts, [_][]const u8{ "copy", "arithmetic", "story", "code" }) |*p, name| {
        p.* = .{ .name = name, .ids = try readJson(gpa, io, oracle, name, ".ids.json"), .want = &.{} };
    }
    defer for (prompts) |p| {
        gpa.free(p.ids);
        gpa.free(p.want);
    };
    var bad: usize = 0;
    var engine = lanes.Engine.init(gpa, &cfg, metal.backend(), clock.clock());
    defer engine.deinit();
    // the reference: the engine's own plain one-stream reply (prompts take the tensor-unit path, not the oracle's)
    for (&prompts) |*p| p.want = try plain(gpa, &engine, p.ids, 128);
    for (prompts) |p| bad += try case(gpa, io, &engine, &.{p}, 1, true, "copy drafts, one stream");
    bad += try case(gpa, io, &engine, &prompts, 4, true, "copy drafts, four streams");
    bad += try case(gpa, io, &engine, &prompts, 8, true, "copy drafts, eight streams");
    if (extra) |dir| {
        var copy_heavy: [2]Prompt = undefined;
        for (&copy_heavy, [_][]const u8{ "repeat", "edit" }) |*p, name| {
            const ids = try readJson(gpa, io, dir, name, ".ids.json");
            p.* = .{ .name = name, .ids = ids, .want = try plain(gpa, &engine, ids, 256) };
        }
        defer for (copy_heavy) |p| {
            gpa.free(p.ids);
            gpa.free(p.want);
        };
        for (copy_heavy) |p| bad += try case(gpa, io, &engine, &.{p}, 1, true, "copy drafts on a copying reply, one stream");
        const mixed = [_]Prompt{ copy_heavy[0], prompts[0], copy_heavy[1], prompts[1], copy_heavy[0], prompts[2], copy_heavy[1], prompts[3] };
        bad += try case(gpa, io, &engine, &mixed, 8, true, "copy drafts, eight mixed streams");
    }
    std.debug.print("{s}: {d} streams differ\n", .{ if (bad == 0) "all equal" else "differ", bad });
    if (bad > 0) std.process.exit(1);
}
