//! DeepSeek-V4.1's planner verdicts through sched_verdict.zig and a real lane host round into the load ledger.
const std = @import("std");
const api = @import("engine_api");
const lanes = @import("lanes");
const sched = @import("kv").sched;
const held = @import("sched_verdict.zig").held;
const Ledger = api.load_ledger.Ledger;
const Snapshot = api.load_ledger.Snapshot;
const Request = api.Request;
const t = std.testing;
const gpa = t.allocator;

/// The planner's side of `admit`, as kv/sched.zig answers: each verdict moves the counter the real planner moves.
const Planner = struct {
    counts: sched.Counts = .{},
    /// What a request whose prompt starts with 2 waits for: the pool's pages or the memory floor.
    floor: bool = false,

    fn self(ctx: *anyopaque) *Planner {
        return @ptrCast(@alignCast(ctx));
    }
    /// 1: a long prompt past the concurrent bound; 2: a hold; 3: more pages than the pool has.
    fn verdict(p: *Planner, first: u32) sched.Verdict {
        switch (first) {
            1 => {
                p.counts.long_waits += 1;
                return .skip;
            },
            2 => {
                if (p.floor) p.counts.mem_waits += 1 else p.counts.pool_waits += 1;
                return .wait;
            },
            else => return .{ .refuse = .too_large },
        }
    }
    fn admit(ctx: *anyopaque, ask: api.Rounds.Ask, words: *[]const u8) api.Rounds.Verdict {
        const p = self(ctx);
        const before = p.counts;
        const v = p.verdict(ask.prompt[0]);
        return held(v, before, p.counts, words);
    }
    fn none(_: *anyopaque) void {}
    fn admitted(_: *anyopaque) anyerror!void {}
    fn key(_: *anyopaque, _: usize) void {}
    fn began(_: *anyopaque, _: usize, _: bool) void {}
    fn plan(_: *anyopaque, pieces: *std.ArrayList(api.Rounds.Piece), finals: *std.ArrayList(usize)) anyerror!void {
        pieces.clearRetainingCapacity();
        finals.clearRetainingCapacity();
    }
    fn after(_: *anyopaque, _: f64, _: f64) void {}
    fn rounds(p: *Planner) api.Rounds {
        return .{ .ctx = p, .vtable = &.{ .begin_admit = none, .admit = admit, .admitted = admitted, .arm = key, .began = began, .plan = plan, .left = key, .after = after } };
    }
};

/// One request's end, as its sink heard it.
const Box = struct {
    reason: ?api.Reason = null,
    message: [128]u8 = undefined,
    len: usize = 0,

    fn event(ctx: *anyopaque, _: api.Id, e: *const api.Event) void {
        const b: *Box = @ptrCast(@alignCast(ctx));
        switch (e.*) {
            .finished => |f| {
                b.reason = f.reason;
                b.len = @min(f.message.len, b.message.len);
                @memcpy(b.message[0..b.len], f.message[0..b.len]);
            },
            else => {},
        }
    }
};

/// A lane host on the fake target whose rounds the test steps itself, with the ledger attached.
const Rig = struct {
    cfg: lanes.Config,
    target: lanes.fake.Fake,
    clock: lanes.fake.FixedClock = .{},
    core: lanes.Engine,
    host: api.LaneHost,
    planner: Planner = .{},
    ledger: Ledger,

    fn init(r: *Rig) !void {
        r.cfg = try lanes.Config.init(gpa, .{ .exact_width = 8, .gpu_tokens = true, .hidden_rows = true }, 8, 7);
        r.target = .{ .gpa = gpa };
        r.clock = .{};
        r.core = lanes.Engine.init(gpa, &r.cfg, r.target.pieceBackend(), r.clock.clock());
        r.host = api.LaneHost.init(gpa, t.io, &r.core, .{ .lanes = 2 });
        r.planner = .{};
        r.host.rounds = r.planner.rounds();
        r.ledger = .init(t.io);
        r.ledger.attach(r.host.engine());
    }

    fn deinit(r: *Rig) void {
        api.Rounds.Test.close(&r.host);
        r.ledger.detach(false);
        r.host.stop();
        r.core.deinit();
        r.target.deinit();
        r.cfg.deinit(gpa);
    }

    fn submit(r: *Rig, id: api.Id, request: *const Request, box: *Box) !void {
        try r.host.engine().submit(id, request, .{ .ctx = box, .event = Box.event });
    }

    fn read(r: *Rig) Snapshot {
        return r.ledger.read();
    }
};

fn waiting(s: Snapshot, w: api.Rounds.Wait) u32 {
    return s.requests.?.waiting_by_reason[@backingInt(w)];
}

const long = [_]u32{ 1, 5, 9, 2 };
const pages = [_]u32{ 2, 6, 5, 3 };
const huge = [_]u32{ 3, 5, 8, 9 };

test "wait reason ledger: DeepSeek's long-prompt skip, KV pool wait and memory floor wait reach the ledger as theirs" {
    var r: Rig = undefined;
    try r.init();
    defer r.deinit();
    var boxes: [2]Box = .{ .{}, .{} };
    const a: Request = .{ .prompt = &long, .max_tokens = 4 };
    const b: Request = .{ .prompt = &pages, .max_tokens = 4 };
    try r.submit(1, &a, &boxes[0]);
    try r.submit(2, &b, &boxes[1]);
    _ = api.Rounds.Test.step(&r.host);
    var s = r.read();
    try t.expectEqual(@as(u32, 2), s.requests.?.waiting);
    try t.expectEqual(@as(u32, 1), waiting(s, .prefill_concurrent));
    try t.expectEqual(@as(u32, 1), waiting(s, .kv_pool));
    try t.expectEqual(@as(u32, 0), waiting(s, .memory_floor));
    try t.expectEqual(@as(u32, 0), waiting(s, .engine_busy));
    r.planner.floor = true; // the same request now waits for the memory floor, not the pool
    _ = api.Rounds.Test.step(&r.host);
    s = r.read();
    try t.expectEqual(@as(u32, 1), waiting(s, .memory_floor));
    try t.expectEqual(@as(u32, 0), waiting(s, .memory_pressure)); // the floor is never pressure
    try t.expectEqual(@as(u32, 0), waiting(s, .kv_pool));
    try t.expectEqual(@as(u32, 1), waiting(s, .prefill_concurrent));
    try t.expectEqual(@as(u64, 0), s.failed(.planner_refused));
}

test "wait reason ledger: DeepSeek's refusal ends the request with its words and counts as planner_refused" {
    var r: Rig = undefined;
    try r.init();
    defer r.deinit();
    var box: Box = .{};
    const c: Request = .{ .prompt = &huge, .max_tokens = 4 };
    try r.submit(1, &c, &box);
    try t.expectEqual(@as(u64, 0), r.read().failed(.planner_refused));
    _ = api.Rounds.Test.step(&r.host);
    try t.expectEqual(@as(?api.Reason, .failed), box.reason);
    try t.expectEqualStrings("the request needs more KV pool pages than the pool has", box.message[0..box.len]);
    const s = r.read();
    try t.expectEqual(@as(u64, 1), s.failed(.planner_refused));
    try t.expectEqual(@as(u32, 0), s.requests.?.waiting);
}
