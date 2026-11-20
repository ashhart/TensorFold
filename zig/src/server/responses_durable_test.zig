const std = @import("std");
const durable = @import("responses_durable.zig");
const codec = @import("responses_durable_record.zig");
const json = @import("json");
const errors = @import("errors.zig");

const t = std.testing;
const owner_a: [32]u8 = @splat(1);
const owner_b: [32]u8 = @splat(2);

fn metadata(key: ?[32]u8) durable.Metadata {
    return .{ .principal = .{ .key = key, .scope = "tenant-scope" }, .model = "model-sha256", .template = "template-sha256" };
}

const Fixture = struct {
    tmp: std.testing.TmpDir,
    root: [:0]const u8,
    store: durable.Store,

    fn init() !Fixture {
        var tmp = t.tmpDir(.{});
        const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
        errdefer t.allocator.free(root);
        return .{ .tmp = tmp, .root = root, .store = try durable.Store.init(t.allocator, t.io, root, 1 << 20) };
    }

    fn deinit(f: *Fixture) void {
        f.store.deinit();
        t.allocator.free(f.root);
        f.tmp.cleanup();
    }

    fn reopen(f: *Fixture) !void {
        f.store.deinit();
        f.store = try durable.Store.init(t.allocator, t.io, f.root, 1 << 20);
    }
};

fn parsed(a: std.mem.Allocator, text: []const u8) !json.Value {
    return (try json.parse(a, text)).ok;
}

test "durable Responses restart reconstructs the exact owner model and parent history" {
    var fixture = try Fixture.init();
    defer fixture.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    var first = [_]json.Value{try parsed(a, "{\"role\":\"user\",\"content\":\"private question\"}")};
    var second = [_]json.Value{try parsed(a, "{\"role\":\"assistant\",\"content\":\"committed answer\"}")};
    try fixture.store.putWhole(&cx, try parsed(a, "{\"id\":\"resp_a\",\"output\":[]}"), &first, metadata(owner_a));
    try fixture.store.put(&cx, try parsed(a, "{\"id\":\"resp_b\",\"previous_response_id\":\"resp_a\",\"output\":[]}"), &second, metadata(owner_a));
    try fixture.reopen();

    const history = (try fixture.store.history(a, metadata(owner_a), "resp_b", 0)).?;
    try t.expectEqual(@as(usize, 2), history.len);
    try t.expectEqualStrings("private question", history[0].strField("content").?);
    try t.expectEqualStrings("committed answer", history[1].strField("content").?);
    try t.expect((try fixture.store.get(a, metadata(owner_b), "resp_b", 0)) == null);
    var other_scope = metadata(owner_a);
    other_scope.principal.scope = "other-scope";
    try t.expect((try fixture.store.get(a, other_scope, "resp_b", 0)) == null);
    var other_model = metadata(owner_a);
    other_model.model = "other-model";
    try t.expectError(error.ConversationIdentityChanged, fixture.store.get(a, other_model, "resp_b", 0));

    try t.expect(try fixture.store.delete(a, metadata(owner_a).principal, "resp_a"));
    try t.expect((try fixture.store.history(a, metadata(owner_a), "resp_b", 0)) == null);
    try t.expect((try fixture.store.get(a, metadata(owner_a), "resp_b", 0)) != null);
}

test "durable Responses whole entries do not resurrect a public parent and expiry invalidates chains" {
    var fixture = try Fixture.init();
    defer fixture.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    var whole = [_]json.Value{try parsed(a, "{\"role\":\"user\",\"content\":\"complete supplied history\"}")};
    try fixture.store.putWhole(&cx, try parsed(a, "{\"id\":\"resp_whole\",\"previous_response_id\":\"resp_absent\",\"output\":[]}"), &whole, metadata(owner_a));
    const items = (try fixture.store.history(a, metadata(owner_a), "resp_whole", 0)).?;
    try t.expectEqual(@as(usize, 1), items.len);
    try t.expectEqualStrings("resp_absent", (try fixture.store.get(a, metadata(owner_a), "resp_whole", 0)).?.strField("previous_response_id").?);

    var expiring = metadata(owner_a);
    expiring.expires_at = 50;
    try fixture.store.putWhole(&cx, try parsed(a, "{\"id\":\"resp_expiring\",\"output\":[]}"), &whole, expiring);
    try t.expect((try fixture.store.get(a, expiring, "resp_expiring", 49)) != null);
    try t.expect((try fixture.store.get(a, expiring, "resp_expiring", 50)) == null);
    try fixture.store.put(&cx, try parsed(a, "{\"id\":\"resp_child\",\"previous_response_id\":\"resp_expiring\",\"output\":[]}"), &whole, expiring);
    try t.expect((try fixture.store.history(a, expiring, "resp_child", 50)) == null);
}

test "durable Responses retries are idempotent and conflicting committed items are refused" {
    var fixture = try Fixture.init();
    defer fixture.deinit();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var cx: errors.Cx = .{ .a = a };
    var messages = [_]json.Value{try parsed(a, "{\"role\":\"assistant\",\"content\":\"partial before stop\"}")};
    const stopped = "{\"id\":\"resp_stop\",\"status\":\"cancelled\",\"output\":[]}";
    try fixture.store.putWhole(&cx, try parsed(a, stopped), &messages, metadata(owner_a));
    try fixture.store.putWhole(&cx, try parsed(a, stopped), &messages, metadata(owner_a));
    try t.expectEqualStrings("cancelled", (try fixture.store.get(a, metadata(owner_a), "resp_stop", 0)).?.strField("status").?);
    try t.expectError(error.ConversationConflict, fixture.store.putWhole(&cx, try parsed(a, "{\"id\":\"resp_stop\",\"status\":\"completed\",\"output\":[]}"), &messages, metadata(owner_a)));
}

const Retry = struct {
    store: *durable.Store,
    failure: ?anyerror = null,

    fn run(r: *Retry) void {
        var arena: std.heap.ArenaAllocator = .init(std.heap.page_allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var cx: errors.Cx = .{ .a = a };
        const response = parsed(a, "{\"id\":\"resp_race\",\"status\":\"cancelled\",\"output\":[]}") catch |err| {
            r.failure = err;
            return;
        };
        var messages = [_]json.Value{parsed(a, "{\"role\":\"assistant\",\"content\":\"same committed prefix\"}") catch |err| {
            r.failure = err;
            return;
        }};
        r.store.putWhole(&cx, response, &messages, metadata(owner_a)) catch |err| {
            r.failure = err;
        };
    }
};

test "durable Responses concurrent identical retry and stop commits leave one recoverable record" {
    var fixture = try Fixture.init();
    defer fixture.deinit();
    var one: Retry = .{ .store = &fixture.store };
    var two: Retry = .{ .store = &fixture.store };
    const first = try std.Thread.spawn(.{}, Retry.run, .{&one});
    const second = try std.Thread.spawn(.{}, Retry.run, .{&two});
    first.join();
    second.join();
    if (one.failure) |err| return err;
    if (two.failure) |err| return err;
    try fixture.reopen();
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const response = (try fixture.store.get(arena.allocator(), metadata(owner_a), "resp_race", 0)).?;
    try t.expectEqualStrings("cancelled", response.strField("status").?);
}

test "durable Responses record decoding rejects malformed identity and parent fields" {
    try t.expectError(error.BadConversationRecord, codec.decode(t.allocator, "{}"));
    try t.expectError(error.BadConversationRecord, codec.decode(t.allocator, "{\"schema\":1}"));
}
