//! The producer over a synthetic engine state: file hand-offs replay, refusals leave reasons, each state is freed once.
const std = @import("std");
const fabric = @import("fabric");
const plan = @import("handoff/plan.zig");
const metal = @import("handoff/metal.zig");
const im = @import("handoff/import.zig");
const p2 = @import("handoff/p2.zig");
const s = @import("handoff/synth.zig");
const cx = @import("cuda_export.zig");
const cxt = @import("cuda_export_test.zig");
const file = @import("handoff_file.zig");
const serve = @import("cuda_export_serve.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const model = "dsv41-test";

/// A source whose state is a synthetic world's; counts what it hands out and gets back.
const Fake = struct {
    arena: std.heap.ArenaAllocator,
    snap: cxt.Snap,
    pos: u64,
    taken: u32 = 0,
    released: u32 = 0,
    fail: bool = false,

    fn init(upto: u64) !*Fake {
        const f = try gpa.create(Fake);
        f.* = .{ .arena = .init(gpa), .snap = undefined, .pos = upto };
        errdefer f.arena.deinit();
        const w = try s.make(f.arena.allocator(), .{ .upto = upto, .ds_rows = @intCast(@min(upto, 128)), .seed = upto });
        f.snap = try cxt.snapshotOf(f.arena.allocator(), w);
        return f;
    }

    fn deinit(f: *Fake) void {
        f.arena.deinit();
        gpa.destroy(f);
    }

    fn source(f: *Fake) cx.Source {
        return .{ .ptr = f, .take = take, .release = release };
    }

    fn take(ptr: *anyopaque, _: std.mem.Allocator, at: u64) anyerror!cx.Captured {
        const f: *Fake = @ptrCast(@alignCast(ptr));
        if (f.fail) return error.DeviceLost;
        f.taken += 1;
        // the engine hands out a copy: the export zeroes stale drafter rows in place
        const pieces = try f.arena.allocator().alloc(cx.Piece, f.snap.pieces.len);
        for (f.snap.pieces, pieces) |p, *o| o.* = .{ .role = p.role, .bytes = try f.arena.allocator().dupe(u8, p.bytes) };
        return .{ .ctx = f, .state = .{ .pos = at, .ds_rows = @min(at, 128), .pieces = pieces, .pages = f.snap.pages } };
    }

    fn release(ctx: *anyopaque) void {
        const f: *Fake = @ptrCast(@alignCast(ctx));
        f.released += 1;
    }
};

fn tokensOf(arena: std.mem.Allocator, n: u64) ![]u32 {
    const t = try arena.alloc(u32, n);
    for (t, 0..) |*x, i| x.* = @intCast(i * 11 + 2);
    return t;
}

fn target(arena: std.mem.Allocator) !metal.Packed {
    const probe: metal.Packed = .{ .live = &.{}, .storage = &.{}, .layers = 43, .capacity = 1024 };
    return .{ .live = try arena.alloc(u8, probe.liveBytes()), .storage = try arena.alloc(u8, metal.Packed.storageBytes(1024)), .layers = 43, .capacity = 1024 };
}

test "a handoff dumped to a directory replays into the importer; the tokens and no partial file sit beside it" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", a);
    const dump = try std.fs.path.join(a, &.{ dir, "nested", "out" }); // made on open

    const fake = try Fake.init(250);
    defer fake.deinit();
    var problem: []const u8 = "";
    const p = try serve.Produce.open(gpa, io, fake.source(), .{ .dump = dump, .model = model }, &problem);
    defer p.stop();

    const tokens = try tokensOf(a, 250);
    const id: [16]u8 = .{ 0xde, 0xad } ++ @as([14]u8, @splat(7));
    const saved = try p.save(250);
    p.exported(id, tokens, saved);
    try std.testing.expectEqual(@as(u32, 1), fake.taken);
    try std.testing.expectEqual(@as(u32, 1), fake.released);

    var name_buf: [32 + file.extension.len]u8 = undefined;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dump, file.fileName(&name_buf, id) }), a, .limited(64 << 20));
    const ids_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/{x}.tokens.json", .{ dump, &id }), a, .limited(1 << 20));
    const parsed = try std.json.parseFromSlice([]u32, a, ids_text, .{});
    try std.testing.expectEqualSlices(u32, tokens, parsed.value);
    try std.testing.expectError(error.FileNotFound, std.Io.Dir.cwd().access(io, try std.fmt.allocPrint(a, "{s}/{x}.mkvh2.part", .{ dump, &id }), .{}));

    const shape: plan.Shape = .{ .upto = 250, .ds_rows = 128 };
    const via = try target(a);
    var imp = try im.Import.init(a, shape, .{ .@"packed" = via }, .{});
    var sink: p2.ShardSink = .{ .imp = &imp, .shard = 0 };
    var reason: fabric.decoder.Reason = .{};
    const got = try file.pull(a, bytes, .{ .model = model, .tokens = tokens, .protocol = 2 }, sink.sink(), &reason);
    try p2.check(a, shape, got.manifest);
    const done = try imp.complete();
    try std.testing.expectEqual(@as(u64, 250), done.upto);
    try std.testing.expectEqual(@as(u64, 122), done.ds_first);
}

test "an export that fails leaves its reason in the dump directory, and its state is released" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const dump = try tmp.dir.realPathFileAlloc(io, ".", a);
    const fake = try Fake.init(100);
    defer fake.deinit();
    var problem: []const u8 = "";
    const p = try serve.Produce.open(gpa, io, fake.source(), .{ .dump = dump, .model = model }, &problem);
    defer p.stop();

    // a prompt whose length is not the state's position is not this state's
    const id: [16]u8 = @splat(3);
    p.exported(id, try tokensOf(a, 99), try p.save(100));
    try std.testing.expectEqual(@as(u32, 1), fake.released);
    const why = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/{x}.error", .{ dump, &id }), a, .limited(4096));
    try std.testing.expectEqualStrings("NotAHandoffPrompt", why);

    // a device read that fails is the lane host's `failed`
    fake.fail = true;
    try std.testing.expectError(error.DeviceLost, p.save(100));
    p.failed(@splat(4), "DeviceLost");
    const why2 = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(a, "{s}/{x}.error", .{ dump, &@as([16]u8, @splat(4)) }), a, .limited(4096));
    try std.testing.expectEqualStrings("DeviceLost", why2);
}

test "a link's table holds the export until the decoder closes it, refuses a repeated id and an export past the cap" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init(120);
    defer fake.deinit();
    // the link's listen mailbox is a device matter: the held-export bookkeeping is the producer's own
    var p: serve.Produce = .{ .gpa = gpa, .io = io, .source = fake.source(), .opts = .{ .link = "tf-test", .model = model, .held_bytes = 8 << 20 }, .table = .{ .gpa = gpa, .ttl_ns = std.time.ns_per_hour, .clock = fabric.words.nowNs, .cap_bytes = 8 << 20 } };
    defer p.table.deinit();
    defer {
        var it = p.held.valueIterator();
        while (it.next()) |h| {
            h.*.built.deinit(gpa);
            gpa.destroy(h.*);
        }
        p.held.deinit(gpa);
    }
    const tokens = try tokensOf(a, 120);
    const id: [16]u8 = @splat(9);
    p.exported(id, tokens, try p.save(120));
    const entry = p.table.get(id).?;
    try std.testing.expectEqual(@as(u16, 2), entry.ready.protocol());
    try std.testing.expectEqualSlices(u32, tokens, entry.ready.tokens);
    try std.testing.expectEqual(@as(u32, 1), fake.released);

    // the same id again: refused, the held export stays
    p.exported(id, tokens, try p.save(120));
    try std.testing.expectEqual(@as(usize, 1), p.held.count());
    try std.testing.expect(p.table.get(id).? == .ready);
    try std.testing.expectEqual(@as(u32, 2), fake.released);

    // an export past TF_PRODUCE_HELD_GIB's worth is refused by name: the decoder hears it
    const big: [16]u8 = @splat(10);
    p.exported(big, tokens, try p.save(120));
    try std.testing.expectEqualStrings("the producer already holds TF_PRODUCE_HELD_GIB of exports; it refuses more until the decoder closes some", p.table.get(big).?.failed);
    try std.testing.expectEqual(@as(usize, 1), p.held.count());

    // the decoder's CLOSE finishes it: the table reports its name and the producer frees it
    try p.table.finish(id);
    var done: std.ArrayList([]const u8) = .empty;
    defer done.deinit(gpa);
    try p.table.takeFinished(&done);
    try std.testing.expectEqual(@as(usize, 1), done.items.len);
    var hex: [32]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{&id}) catch unreachable;
    try std.testing.expectEqualStrings(&hex, done.items[0]);
}

fn readAll(r: *serve.Produce.Reading, into: *std.ArrayList(u8)) !void {
    while (try r.next()) |message| try into.appendSlice(gpa, message);
}

test "an HTTP producer streams a held export to any reader; a release or an expiry frees it at the last reader's close" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init(250);
    defer fake.deinit();
    var problem: []const u8 = "";
    const p = try serve.Produce.open(gpa, io, fake.source(), .{ .http = true, .model = model, .ttl_ns = std.time.ns_per_hour }, &problem);
    const tokens = try tokensOf(a, 250);
    const id: [16]u8 = @splat(0x21);
    p.exported(id, tokens, try p.save(250));
    try std.testing.expectEqual(@as(usize, 1), p.held.count());
    try std.testing.expectEqual(@as(u32, 1), fake.released);

    // what a reader gets is what the file holds, and the stream says its length first
    var want: std.Io.Writer.Allocating = .init(gpa);
    defer want.deinit();
    try file.write(gpa, &want.writer, p.table.get(id).?.ready, model, 0, 1);
    var why: []const u8 = "";
    const first = try p.read(id, &why);
    const second = try p.read(id, &why); // another reader at once, on the same export
    try std.testing.expectEqual(@as(u64, want.written().len), first.size());
    var got: std.ArrayList(u8) = .empty;
    defer got.deinit(gpa);
    try readAll(first, &got);
    try std.testing.expectEqualSlices(u8, want.written(), got.items);
    first.close();

    // the decoder deletes it while the second reader is mid-stream: new readers find nothing, the second one finishes
    const manifest_len = (try second.next()).?.len; // the manifest
    try std.testing.expect(p.discard(id));
    try std.testing.expectEqual(@as(usize, 0), p.held.count());
    try std.testing.expectError(error.NotHeld, p.read(id, &why));
    try std.testing.expect(!p.discard(id));
    var rest: usize = 0;
    while (try second.next()) |message| rest += message.len;
    try std.testing.expectEqual(want.written().len, manifest_len + rest);
    second.close(); // the last close frees the export: the leak check proves it

    // a failed export is refused with its reason
    p.failed(@splat(0x22), "DeviceLost");
    try std.testing.expectError(error.Refused, p.read(@splat(0x22), &why));
    try std.testing.expectEqualStrings("DeviceLost", why);
    try std.testing.expectError(error.NotHeld, p.read(@splat(0x23), &why));
    p.stop();
}

test "an HTTP producer's export expires, and one being read outlives its expiry" {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const fake = try Fake.init(130);
    defer fake.deinit();
    var problem: []const u8 = "";
    const p = try serve.Produce.open(gpa, io, fake.source(), .{ .http = true, .model = model, .ttl_ns = 20 * std.time.ns_per_ms }, &problem);
    const id: [16]u8 = @splat(0x31);
    p.exported(id, try tokensOf(a, 130), try p.save(130));
    var why: []const u8 = "";
    const reading = try p.read(id, &why);
    _ = try reading.next();
    try std.Io.sleep(io, .fromMilliseconds(60), .awake);
    try std.testing.expectError(error.NotHeld, p.read(id, &why)); // the sweep took it
    var rest: usize = 0;
    while (try reading.next()) |message| rest += message.len;
    try std.testing.expect(rest > 0); // the reader that held it still gets every frame
    reading.close();
    p.stop();
}
