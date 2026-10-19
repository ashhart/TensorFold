//! The served engine without HTTP (tf-dsv41 follow / generate): replays a token trace; TF_DSV41_DRAFTS=0: no DSpark.

const std = @import("std");
const api = @import("engine_api");
const model = @import("cuda_model.zig");
const se = @import("serve_engine.zig");

fn drafts() bool {
    const v = std.c.getenv("TF_DSV41_DRAFTS") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

pub fn follow(gpa: std.mem.Allocator, io: std.Io, pack: []const u8, assets: []const u8) !u8 {
    const m = try model.Model.open(gpa, io, pack, assets, .{ .drafts = drafts(), .own_prefill = model.ownPrefillEnv(), .layers = try model.layersEnv(gpa) });
    defer m.close();
    if (m.rank() == 0) return error.RankZeroServes;
    std.debug.print("rank {d}: loaded, following rank 0\n", .{m.rank()});
    m.f.follow() catch |err| {
        std.log.err("rank {d}: following rank 0 failed: {s}", .{ m.rank(), @errorName(err) });
        return err;
    };
    std.debug.print("rank {d}: rank 0 stopped\n", .{m.rank()});
    return 0;
}

const Box = struct {
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    tokens: std.ArrayList(u32) = .empty,
    done: ?api.Reason = null,
    drafted: u64 = 0,
    accepted: u64 = 0,
    gpa: std.mem.Allocator,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        b.mutex.lockUncancelable(b.io);
        defer b.mutex.unlock(b.io);
        switch (e.*) {
            .tokens => |t| b.tokens.appendSlice(b.gpa, t) catch {},
            .finished => |f| {
                b.drafted = f.stats.drafted;
                b.accepted = f.stats.accepted;
                b.done = f.reason;
            },
            else => {},
        }
    }

    fn wait(b: *Box) api.Reason {
        while (true) {
            b.mutex.lockUncancelable(b.io);
            const d = b.done;
            b.mutex.unlock(b.io);
            if (d) |r| return r;
            std.Io.sleep(b.io, .fromMilliseconds(2), .awake) catch {};
        }
    }
};

pub fn generate(gpa: std.mem.Allocator, io: std.Io, pack: []const u8, assets: []const u8, trace: []const u8) !u8 {
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const text = try std.Io.Dir.cwd().readFileAlloc(io, trace, a, .limited(1 << 30));
    const s = try se.Served.open(gpa, io, pack, assets, drafts(), null);
    defer se.Served.close(s);
    const eng = s.engine();
    const Keyed = struct { seed: u64, temperature: f64, top_k: u32, top_p: f64, min_p: f64 = 0 };
    // "structure" ({kind, text}: the request's grammar, TF_DSV41_GRAMMAR=1): replayed under the same grammar
    const Structured = struct { kind: []const u8, text: []const u8 = "" };
    const Line = struct { prompt: []const u32, tokens: []const u32, drafted: ?u64 = null, sampling: ?Keyed = null, structure: ?Structured = null };
    var recs: std.ArrayList(Line) = .empty;
    var lines = std.mem.tokenizeScalar(u8, text, '\n');
    while (lines.next()) |line| try recs.append(a, try std.json.parseFromSliceLeaky(Line, a, line, .{ .ignore_unknown_fields = true }));
    // TF_DSV41_GEN_STREAMS=N: N requests in flight at once (the lanes batch them over the live slots); 1: one by one
    const streams: usize = if (std.c.getenv("TF_DSV41_GEN_STREAMS")) |v| @max(1, std.fmt.parseInt(usize, std.mem.span(v), 10) catch 1) else 1;
    const boxes = try a.alloc(Box, recs.items.len);
    const reqs = try a.alloc(api.Request, recs.items.len);
    var n: usize = 0;
    var equal: usize = 0;
    var drafted: u64 = 0;
    var accepted: u64 = 0;
    var at: usize = 0;
    while (at < recs.items.len) : (at += streams) {
        const end = @min(recs.items.len, at + streams);
        for (at..end) |i| {
            const rec = recs.items[i];
            boxes[i] = .{ .io = io, .gpa = gpa };
            const smp: ?api.Sampling = if (rec.sampling) |k| .{ .seed = k.seed, .temperature = k.temperature, .top_k = k.top_k, .top_p = k.top_p, .min_p = k.min_p } else null;
            reqs[i] = .{ .prompt = rec.prompt, .max_tokens = @intCast(rec.tokens.len), .drafts = drafts(), .sampling = smp };
            if (rec.structure) |st| reqs[i].structure = .{ .kind = std.meta.stringToEnum(@FieldType(api.Structure, "kind"), st.kind) orelse return error.BadStructure, .text = st.text };
            try eng.submit(@intCast(i + 1), &reqs[i], .{ .ctx = &boxes[i], .event = Box.event });
        }
        for (at..end) |i| {
            const rec = recs.items[i];
            const box = &boxes[i];
            defer box.tokens.deinit(gpa);
            const why = box.wait();
            const same = std.mem.eql(u32, box.tokens.items, rec.tokens);
            // the first token index where the replay leaves the recorded reply (-1: none), and both draft counts
            const first: i64 = if (same) -1 else @intCast(std.mem.indexOfDiff(u32, box.tokens.items, rec.tokens) orelse @min(box.tokens.items.len, rec.tokens.len));
            n += 1;
            equal += @intFromBool(same);
            drafted += box.drafted;
            accepted += box.accepted;
            std.debug.print("{{\"line\": {d}, \"prompt\": {d}, \"tokens\": {d}, \"recorded\": {d}, \"finished\": \"{t}\", \"equal\": {}, \"first_differing\": {d}, \"drafted\": {d}, \"recorded_drafted\": {?d}, \"accepted\": {d}, \"streams\": {d}}}\n", .{ n, rec.prompt.len, box.tokens.items.len, rec.tokens.len, why, same, first, box.drafted, rec.drafted, box.accepted, streams });
        }
    }
    const ok = n > 0 and equal == n;
    std.debug.print("{s} CLI == HTTP: {d}/{d} replies equal\n", .{ if (ok) "PASS" else "FAIL", equal, n });
    // DSpark's acceptance on these replies (drafts on): accepted / drafted, and tokens a verify round
    const rate: f64 = if (drafted > 0) @as(f64, @floatFromInt(accepted)) / @as(f64, @floatFromInt(drafted)) else 0;
    std.debug.print("DSpark: {d} drafted, {d} accepted ({d:.3})\n", .{ drafted, accepted, rate });
    return if (ok) 0 else 1;
}
