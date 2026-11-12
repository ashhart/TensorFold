//! The slide adapter: initial reader progress before its first potentially long step, and --slide's startup refusals.
const std = @import("std");
const api = @import("engine_api");
const Adapter = @import("slide_adapter.zig").Adapter;
const refusal = @import("slide_adapter.zig").refusal;
const Fixture = struct {
    pub const Example = api.Example;
    pub const Learner = struct {
        pub const Progress = struct { phase: enum { read, fit, values, check, done }, at: usize, of: usize, loss: f32 };
        pub const Placed = api.Neuron;
        pub fn init(_: std.mem.Allocator, _: std.Io, _: void) @This() {
            return .{};
        }
        pub fn deinit(_: *@This()) void {}
        pub fn begin(_: *@This(), _: anytype) !void {}
        pub fn abort(_: *@This()) void {}
        pub fn progress(_: *@This()) ?Progress {
            return .{ .phase = .read, .at = 0, .of = 100, .loss = 0 };
        }
        pub fn placed(_: *@This(), buf: []Placed) []const Placed {
            return buf[0..0];
        }
        pub fn step(_: *@This()) struct {
            done: bool = true,
            changed: bool = false,
            stale: bool = false,
            report: ?union(enum) { learned: struct { recalled: bool, steps: u32, loss: f32 }, failed: []const u8 } = null,
        } {
            return .{};
        }
    };
};
const Seen = struct {
    reads: usize = 0,
    fn on(ctx: *anyopaque, event: *const api.LearnEvent) void {
        const s: *Seen = @ptrCast(@alignCast(ctx));
        if (event.* == .progress and event.progress.phase == .reading and event.progress.at == 0 and event.progress.of == 100) s.reads += 1;
    }
};
test "lesson adapter: publish reading zero before the first long unit" {
    var a = Adapter(Fixture).init(std.testing.allocator, std.testing.io, {});
    defer a.deinit();
    var seen: Seen = .{};
    const hook = a.hook();
    try hook.begin(hook.ctx, &.{}, .{ .ctx = &seen, .event = Seen.on });
    try std.testing.expectEqual(@as(usize, 1), seen.reads);
}

const AuditFixture = struct {
    pub const Example = api.Example;
    pub const Learner = struct {
        const L = @This();
        pub const Progress = void;
        pub const Placed = api.Neuron;
        lessons: u32 = 17,
        begins: usize = 0,
        steps: usize = 0,
        aborts: usize = 0,
        pub const Audit = struct {
            pub fn init(_: *L) !@This() {
                return .{};
            }
            pub fn deinit(_: *@This()) void {}
            pub fn read(_: *@This(), _: *L, ids: []const u32, _: []const api.Neuron, out: []api.pocket_audit.Peak) !void {
                if (ids[0] == 0) return error.AuditNonfinite;
                for (out) |*p| p.* = .{ .activation = @floatFromInt(ids[0]) };
            }
        };
        pub fn init(_: std.mem.Allocator, _: std.Io, _: void) L {
            return .{};
        }
        pub fn deinit(_: *L) void {}
        pub fn begin(l: *L, _: anytype) !void {
            l.begins += 1;
        }
        pub fn step(l: *L) struct {
            done: bool = true,
            changed: bool = false,
            stale: bool = false,
            report: ?union(enum) { learned: struct { recalled: bool, steps: u32, loss: f32 }, failed: []const u8 } = null,
        } {
            l.steps += 1;
            return .{};
        }
        pub fn abort(l: *L) void {
            l.aborts += 1;
        }
        pub fn placed(_: *L, buf: []Placed) []const Placed {
            return buf[0..0];
        }
    };
};
const AuditSeen = struct {
    done: usize = 0,
    failed: bool = false,
    fn on(ctx: *anyopaque, event: *const api.LearnEvent) void {
        const s: *@This() = @ptrCast(@alignCast(ctx));
        if (event.* == .done) {
            s.done += 1;
            s.failed = event.done.message.len > 0;
        }
    }
};

test "pocket audit adapter: one prompt per step, no lesson begin step abort or arena reset" {
    var a = Adapter(AuditFixture).init(std.testing.allocator, std.testing.io, {});
    defer a.deinit();
    const kept = try a.arena.allocator().dupe(u8, "last lesson stays");
    var seen: AuditSeen = .{};
    const hook = a.hook();
    const ns = [_]api.Neuron{.{ .layer = 1, .neuron = 0, .key = .reply }};
    var out: [2]api.pocket_audit.Peak = undefined;
    var complete = false;
    const r: api.pocket_audit.Request = .{ .neurons = &ns, .probes = &.{ .{ .ids = &.{1} }, .{ .ids = &.{2} } }, .peaks = &out, .completed = &complete };
    const sink: api.LearnSink = .{ .ctx = &seen, .event = AuditSeen.on };
    try hook.begin(hook.ctx, &.{ .audit = &r }, sink);
    const first = hook.step(hook.ctx);
    try std.testing.expect(!first.done and !first.changed and !complete);
    const last = hook.step(hook.ctx);
    try std.testing.expect(last.done and !last.changed and complete and seen.done == 1 and !seen.failed);
    try std.testing.expectEqual(@as(f32, 1), out[0].activation);
    try std.testing.expectEqual(@as(f32, 2), out[1].activation);
    try std.testing.expectEqualStrings("last lesson stays", kept);
    try hook.begin(hook.ctx, &.{ .audit = &r }, sink);
    hook.abort(hook.ctx);
    try std.testing.expect(!complete and seen.failed);
    try std.testing.expect(a.learner.lessons == 17 and a.learner.begins == 0 and a.learner.steps == 0 and a.learner.aborts == 0);
    try std.testing.expectError(error.BadAudit, hook.begin(hook.ctx, &.{ .audit = &r, .save = true }, sink));
}

test "pocket audit adapter: a failed partial measurement never completes; unsupported has no side effects" {
    var a = Adapter(AuditFixture).init(std.testing.allocator, std.testing.io, {});
    defer a.deinit();
    var seen: AuditSeen = .{};
    const hook = a.hook();
    const ns = [_]api.Neuron{.{ .layer = 1, .neuron = 0, .key = .reply }};
    var out: [2]api.pocket_audit.Peak = undefined;
    var complete = false;
    const r: api.pocket_audit.Request = .{ .neurons = &ns, .probes = &.{ .{ .ids = &.{1} }, .{ .ids = &.{0} } }, .peaks = &out, .completed = &complete };
    const sink: api.LearnSink = .{ .ctx = &seen, .event = AuditSeen.on };
    try hook.begin(hook.ctx, &.{ .audit = &r }, sink);
    _ = hook.step(hook.ctx);
    const last = hook.step(hook.ctx);
    try std.testing.expect(last.done and !last.changed and !complete and seen.failed);
    var bare = Adapter(Fixture).init(std.testing.allocator, std.testing.io, {});
    defer bare.deinit();
    const bh = bare.hook();
    try std.testing.expectError(error.AuditUnsupported, bh.begin(bh.ctx, &.{ .audit = &r }, sink));
}
const NoLinks = struct {
    pub fn linked(_: std.mem.Allocator, _: std.Io, _: []const u8) !?[]const u8 {
        return null;
    }
};
test "slide refusal: a shard the index places in a folder is refused at startup, before any save" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    try std.testing.expect((try refusal(NoLinks, a, io, dir, "a copy")) == null);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = "{\"weight_map\":{\"a\":\"model.safetensors\"}}" });
    try std.testing.expect((try refusal(NoLinks, a, io, dir, "a copy")) == null);
    try tmp.dir.writeFile(io, .{
        .sub_path = "model.safetensors.index.json",
        .data = "{\"weight_map\":{\"a\":\"model.safetensors\",\"b\":\"sub/b.safetensors\"}}",
    });
    const why = (try refusal(NoLinks, a, io, dir, "a copy")) orelse return error.TestUnexpectedResult;
    try std.testing.expect(std.mem.indexOf(u8, why, "sub/b.safetensors") != null);
}
