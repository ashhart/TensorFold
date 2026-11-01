//! GLM content-keyed image resumes with the real prefix store, image admission, wire spans and splice dispatch.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const lanes = @import("lanes");
const core = @import("core");
const t = std.testing;
const a = t.allocator;
const D = glm.vision_config.out_hidden;
const pc = glm.prefix.radix;
const N = 8193;

const Model = struct {
    words: [2]u64 = .{ 17, 29 },
    serial: u64 = 0,
    restored: u32 = 0,
    live: usize = 0,
    const Snap = struct { kept: glm.engine.Kept, words: [2]u64 };
    fn bytes(_: *anyopaque, _: u32) u64 {
        return @sizeOf(Snap);
    }
    fn save(ptr: *anyopaque, _: ?*anyopaque, at: u32) anyerror!pc.Saved {
        const m: *Model = @ptrCast(@alignCast(ptr));
        const s = try a.create(Snap);
        m.serial += 1;
        s.* = .{ .kept = .{ .id = m.serial, .at = at }, .words = m.words };
        m.live += 1;
        return s;
    }
    fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: pc.Saved) anyerror!void {
        const m: *Model = @ptrCast(@alignCast(ptr));
        const s: *Snap = @ptrCast(@alignCast(saved));
        m.words = s.words;
        m.restored = s.kept.at;
        try glm.prefix.Prefix.restore(ptr, owner, saved);
    }
    fn drop(ptr: *anyopaque, saved: pc.Saved) void {
        const m: *Model = @ptrCast(@alignCast(ptr));
        a.destroy(@as(*Snap, @ptrCast(@alignCast(saved))));
        m.live -= 1;
    }
};

const Cache = struct {
    pages: glm.pages.Pages = undefined,
    leases: [1]core.kv_pool.Lease = .{.{}},
    used: [1]bool = .{false},
    ids: [16]u32 = undefined,
    prefix: *glm.prefix.Prefix = undefined,
    model: Model = .{},
    fn init(c: *Cache) !void {
        c.pages.pool = try core.kv_pool.Pool.init(a, 64, 512);
        errdefer c.pages.pool.deinit();
        c.pages.layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = 512, .snap_bytes = 0 };
        c.pages.leases = &c.leases;
        c.pages.used = &c.used;
        try t.expect(c.pages.pool.hold(&c.ids));
        errdefer for (c.ids) |id| c.pages.pool.release(id);
        try c.leases[0].ids.appendSlice(a, &c.ids);
        errdefer c.leases[0].ids.deinit(a);
        c.prefix = try glm.prefix.Prefix.initWith(a, &c.pages, 2048, 8, .{ .ptr = &c.model, .vtable = &.{ .bytes = Model.bytes, .save = Model.save, .restore = Model.restore, .drop = Model.drop } });
    }
    fn deinit(c: *Cache) void {
        c.prefix.deinit();
        for (c.ids) |id| c.pages.pool.release(id);
        c.leases[0].ids.deinit(a);
        c.pages.pool.deinit();
        std.debug.assert(c.model.live == 0);
    }
    fn keep(c: *Cache, s: *const lanes.Stream, at: u32, words: [2]u64) void {
        c.model.words = words;
        var keeps: glm.prefix.Keeps = undefined;
        const marks = c.prefix.marks(0, s, &.{at}, &keeps).?;
        marks.call(marks.ctx, at);
    }
    fn begin(c: *Cache, s: *const lanes.Stream) !u32 {
        var owner: glm.prefix.Owner = .{};
        var adopt: std.ArrayList(u32) = .empty;
        defer adopt.deinit(a);
        const plan = try c.prefix.beginStream(s, &owner, &adopt);
        defer a.free(plan.marks);
        defer c.pages.give(adopt.items);
        if (plan.from > 0) try t.expectEqual(plan.from, owner.pending.?.at);
        return plan.from;
    }
};

fn makeKey(ids: *const [N]u32, samples: []const f32, grid: [2]u32) [N]u32 {
    var key = ids.*;
    const identity = core.picture.identity;
    const hash = identity.withGrid(identity.ofBytes(7, std.mem.sliceAsBytes(samples)), grid);
    key[2499] = 0x8000_0000 | @as(u32, @truncate(hash & 0x7fff_ffff));
    key[2502] = 0x8000_0000 | @as(u32, @truncate(hash >> 31));
    return key;
}

const pixels = [2]f32{ 0.25, 0.75 };
const images = [_]lanes.stream.Image{
    .{ .pixels = &pixels, .gh = 2, .gw = 4, .at = 2500, .tokens = 2 },
    .{ .pixels = &.{0.5}, .gh = 3, .gw = 4, .at = 7000, .tokens = 3 },
};

fn stream(ids: []const u32, key: []const u32) !lanes.Stream {
    return lanes.Stream.init(a, .{ .id = "image-cache", .prompt = ids, .key = key, .images = &images, .max_new = 4, .history_len = 6144, .shared_prefixes = &.{2048} });
}

test "GLM image cache: the actual lane opening enables a content-keyed image lookup before admission" {
    var c: Cache = .{};
    try c.init();
    defer c.deinit();
    const ids: [N]u32 = @splat(9);
    const key = makeKey(&ids, &pixels, .{ 2, 4 });
    var s = try stream(&ids, &key);
    defer s.deinit(a);
    var e: glm.engine.Engine = undefined;
    e.gpa = a;
    e.capacity = N + 256;
    e.pages = &c.pages;
    var link: glm.follow.Link = undefined;
    var back: glm.lanes.Lanes = .{ .x = .{ .e = &e, .link = &link }, .prefix = c.prefix };
    defer back.lanes.deinit(a);
    // Refuse before a device operation, after the actual cache lookup and adopt rollback.
    c.pages.used = &.{};
    const b = back.backend();
    try t.expectError(error.LaneMemory, b.vtable.prefill_step.?(b.ptr, &s, 0));
    try t.expectEqual(@as(u64, 1), c.prefix.store.tree.counts.misses);
    c.pages.used = &c.used;
    c.keep(&s, 6144, .{ 31, 47 });
    c.pages.used = &.{};
    try t.expectError(error.LaneMemory, b.vtable.prefill_step.?(b.ptr, &s, 0));
    try t.expectEqual(@as(u32, 6144), c.model.restored);
    try t.expectEqual(@as(u64, 1), c.prefix.store.tree.counts.hits);
    try t.expectEqual(@as(usize, 0), back.lanes.count());
    for (c.ids[0..12]) |id| try t.expectEqual(@as(u32, 2), c.pages.pool.holders(id));
}

test "GLM image cache: A B A and a changed grid resume only matching content and prior text" {
    var c: Cache = .{};
    try c.init();
    defer c.deinit();
    const ids: [N]u32 = @splat(9);
    const ka = makeKey(&ids, &pixels, .{ 2, 4 });
    const changed_pixels = [2]f32{ 0.25, 0.5 };
    const kb = makeKey(&ids, &changed_pixels, .{ 2, 4 });
    const kg = makeKey(&ids, &pixels, .{ 4, 2 });
    var sa = try stream(&ids, &ka);
    defer sa.deinit(a);
    var sb = try stream(&ids, &kb);
    defer sb.deinit(a);
    var changed_image = images;
    changed_image[0].pixels = &changed_pixels;
    sb.images = &changed_image;
    var sg = try stream(&ids, &kg);
    defer sg.deinit(a);
    var changed_grid = images;
    changed_grid[0].gh = 4;
    changed_grid[0].gw = 2;
    sg.images = &changed_grid;
    try t.expectEqual(@as(u32, 0), try c.begin(&sa));
    c.keep(&sa, 2048, .{ 11, 13 });
    c.keep(&sa, 6144, .{ 17, 19 });
    try t.expectEqual(@as(u32, 2048), try c.begin(&sb));
    try t.expectEqual([2]u64{ 11, 13 }, c.model.words);
    c.keep(&sb, 6144, .{ 23, 29 });
    try t.expectEqual(@as(u32, 6144), try c.begin(&sa));
    try t.expectEqual([2]u64{ 17, 19 }, c.model.words);
    try t.expectEqual(@as(u32, 6144), try c.begin(&sb));
    try t.expectEqual([2]u64{ 23, 29 }, c.model.words);
    try t.expectEqual(@as(u32, 2048), try c.begin(&sg));
    sa.key = &.{};
    try t.expect(!glm.prefix.cacheable(&sa));
    try t.expectError(error.ImageCacheKeyRequired, c.begin(&sa));
    sa.key = ka[0..10];
    try t.expect(!glm.prefix.cacheable(&sa));
    sa.key = &ka;
    sa.prompt_logprobs = 0;
    try t.expect(!glm.prefix.cacheable(&sa));
    sa.prompt_logprobs = null;
    sa.untaught = true;
    try t.expect(!glm.prefix.cacheable(&sa));
    sa.untaught = false;
    sa.images = &.{};
    sa.key = &.{};
    try t.expectEqualSlices(u32, &ids, glm.prefix.cacheKey(&sa).?);
    try t.expect(glm.prefix.cacheable(&sa));
}

const Device = struct {
    var encodes: usize = 0;
    var fail_encode: ?usize = null;
    var launches: usize = 0;
    fn encode(_: *glm.vision.Tower, _: *const glm.vision.Run, p: []const f32, grid: [3]i64, out: u64) !void {
        encodes += 1;
        if (fail_encode == encodes) return error.EncodeFailed;
        const n: usize = @intCast(@divExact(grid[1] * grid[2], 4));
        const dst: [*]u16 = @ptrFromInt(out);
        for (0..n) |row| @memset(dst[row * D ..][0..D], @as(u16, @intFromFloat(p[0] * 100)) + @as(u16, @intCast(row)));
    }
    fn upload(dst: u64, src: ?*const anyopaque, n: usize) callconv(.c) c_int {
        @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrCast(src.?))[0..n]);
        return 0;
    }
    fn uploadAsync(dst: u64, src: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) c_int {
        return upload(dst, src, n);
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn hostAlloc(out: *?*anyopaque, n: usize, _: c_uint) callconv(.c) c_int {
        const b = a.alignedAlloc(u8, .@"16", n + 16) catch return cuda.abi.error_not_ready;
        std.mem.writeInt(usize, b[0..@sizeOf(usize)], n + 16, .little);
        out.* = b.ptr + 16;
        return 0;
    }
    fn hostFree(ptr: ?*anyopaque) callconv(.c) c_int {
        const b: [*]align(16) u8 = @alignCast(@as([*]u8, @ptrCast(ptr.?)) - 16);
        a.free(b[0..std.mem.readInt(usize, b[0..@sizeOf(usize)], .little)]);
        return 0;
    }
    fn eventCreate(out: *cuda.abi.Event, _: c_uint) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }
    fn eventRecord(_: cuda.abi.Event, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }
    fn eventDone(_: cuda.abi.Event) callconv(.c) c_int {
        return 0;
    }
    fn eventDestroy(_: cuda.abi.Event) callconv(.c) c_int {
        return 0;
    }
    fn arg(comptime T: type, args: [*]?*anyopaque, i: usize) T {
        return @as(*const T, @ptrCast(@alignCast(args[i].?))).*;
    }
    fn launch(_: cuda.abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: cuda.abi.Stream, argv: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) c_int {
        launches += 1;
        const args = argv.?;
        const rows: [*]const u16 = @ptrFromInt(arg(u64, args, 0));
        const emb: [*]u16 = @ptrFromInt(arg(u64, args, 1));
        const map: [*]const i32 = @ptrFromInt(arg(u64, args, 2));
        const n: usize = @intCast(arg(c_int, args, 3));
        const copies: usize = @intCast(arg(c_int, args, 5));
        for (0..n) |i| if (map[i] >= 0) {
            const src: usize = @intCast(map[i]);
            for (0..copies) |copy| @memcpy(emb[(i * copies + copy) * D ..][0..D], rows[src * D ..][0..D]);
        };
        return 0;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemcpyHtoD_v2 = upload;
        d.api.cuMemcpyHtoDAsync_v2 = uploadAsync;
        d.api.cuStreamSynchronize = sync;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventRecord = eventRecord;
        d.api.cuEventQuery = eventDone;
        d.api.cuEventDestroy_v2 = eventDestroy;
        d.api.cuLaunchKernel = launch;
        return d;
    }
};

fn imageOwner(d: *cuda.Driver, rows: []u16, map: []i32, device_map: []i32) glm.images.Images {
    var v: glm.images.Images = .{ .gpa = a, .io = t.io, .d = d, .module = undefined, .run = .{ .stream = .{ .d = d, .handle = null }, .f = undefined }, .tower = @as(glm.vision.Tower, undefined), .rows = .{ .d = d, .ptr = @intFromPtr(rows.ptr), .len = rows.len * 2, .borrowed = true }, .map = .{ .d = d, .ptr = @intFromPtr(device_map.ptr), .len = device_map.len * 4, .borrowed = true }, .host_map = map, .limits = .{ .image_tokens = 8, .max_images = 2 } };
    v.run.f.inject = .{ .d = d, .handle = null };
    return v;
}

fn reply(initial: [2]u64) [8]u32 {
    var state = initial;
    var out: [8]u32 = undefined;
    for (&out) |*token| {
        token.* = @intCast((state[0] ^ state[1]) % 251);
        state[0] = state[0] *% 131 +% token.*;
        state[1] = state[1] *% 133 +% token.*;
    }
    return out;
}

fn reference(ids: []const u32, from: usize, rows: []const u16, spans: []const glm.images.Span, initial: [2]u64) [2]u64 {
    var out = initial;
    for (ids[from..], from..) |id, row| {
        var value: u64 = id;
        for (spans) |sp| if (row >= sp.at and row < sp.at + sp.n) {
            value = rows[(sp.row + row - sp.at) * D];
        };
        for (&out, 0..) |*word, rank| word.* = word.* *% (131 + rank * 2) +% value;
    }
    return out;
}

test "GLM image cache: suffix encoding wire and actual splice dispatch preserve both ranks and a fresh reply" {
    var d = Device.driver();
    var leader_rows: [8 * D]u16 = @splat(0);
    var peer_rows: [8 * D]u16 = @splat(0);
    var map: [8]i32 = undefined;
    var device_map: [8]i32 = undefined;
    var leader = imageOwner(&d, &leader_rows, &map, &device_map);
    defer leader.spans.deinit(a);
    var peer = imageOwner(&d, &peer_rows, &map, &device_map);
    defer peer.spans.deinit(a);
    defer peer.run.stream.synchronize() catch unreachable;
    Device.encodes = 0;
    Device.fail_encode = null;
    const ids: [N]u32 = @splat(9);
    try leader.beginUsing(0, &images, 0, Device.encode);
    try t.expectEqual(@as(usize, 2), Device.encodes);
    const fresh = reference(&ids, 0, &leader_rows, leader.spans.items, .{ 17, 29 });
    const prior = reference(ids[0..6144], 0, &leader_rows, leader.spans.items, .{ 17, 29 });
    var cache: Cache = .{};
    try cache.init();
    defer cache.deinit();
    const key = makeKey(&ids, &pixels, .{ 2, 4 });
    var request = try stream(&ids, &key);
    defer request.deinit(a);
    cache.keep(&request, 6144, prior);
    cache.model.words = .{ 999, 777 };
    const from = try cache.begin(&request);
    try t.expectEqual(@as(u32, 6144), from);
    try leader.beginUsing(0, &images, from, Device.encode);
    try t.expectEqual(@as(usize, 3), Device.encodes);
    try t.expectEqual(@as(usize, 1), leader.spans.items.len);
    try t.expectEqual(@as(u32, 7000), leader.spans.items[0].at);
    const words = try glm.follow.spanWords(a, leader.spans.items, 12);
    defer a.free(words);
    try peer.take(0, words);
    @memcpy(peer_rows[0 .. peer.rowCount() * D], leader_rows[0 .. leader.rowCount() * D]);
    const resumed = reference(&ids, from, &peer_rows, peer.spans.items, cache.model.words);
    try t.expectEqual(fresh, resumed);
    const fresh_reply = reply(fresh);
    const resumed_reply = reply(resumed);
    try t.expectEqualSlices(u8, std.mem.asBytes(&fresh_reply), std.mem.asBytes(&resumed_reply));
    Device.launches = 0;
    var emb: [6 * D]u16 = @splat(9);
    try peer.splice(peer.run.stream, 0, 2499, 2503, @intFromPtr(&emb), 1);
    try t.expectEqual(@as(usize, 0), Device.launches);
    try peer.splice(peer.run.stream, 0, 6999, 7005, @intFromPtr(&emb), 1);
    try t.expectEqual(@as(usize, 1), Device.launches);
    try t.expectEqualSlices(i32, &.{ -1, 0, 1, 2, -1, -1 }, map[0..6]);
    try t.expectEqual(@as(u16, 9), emb[0]);
    for (0..3) |i| try t.expectEqual(@as(u16, 50 + @as(u16, @intCast(i))), emb[(i + 1) * D]);
    try peer.run.stream.synchronize();
    try leader.beginUsing(0, &images, 2501, Device.encode);
    try t.expectEqual(@as(usize, 5), Device.encodes);
    try t.expectEqual(@as(usize, 2), leader.spans.items.len);
    const overlap_words = try glm.follow.spanWords(a, leader.spans.items, 12);
    defer a.free(overlap_words);
    try peer.take(0, overlap_words);
    @memcpy(peer_rows[0 .. peer.rowCount() * D], leader_rows[0 .. leader.rowCount() * D]);
    @memset(&emb, 9);
    try peer.splice(peer.run.stream, 0, 2501, 2503, @intFromPtr(&emb), 1);
    try t.expectEqualSlices(i32, &.{ 1, -1 }, map[0..2]);
    try t.expectEqual(@as(u16, 26), emb[0]);
    try t.expectEqual(@as(u16, 9), emb[D]);
    Device.fail_encode = Device.encodes + 2;
    try t.expectError(error.EncodeFailed, leader.beginUsing(0, &images, 0, Device.encode));
    try t.expectEqual(@as(usize, 0), leader.spans.items.len);
    Device.fail_encode = null;
    try leader.beginUsing(0, &images, 7003, Device.encode);
    try t.expectEqual(@as(usize, 0), leader.rowCount());
    const none = try glm.follow.spanWords(a, leader.spans.items, 12);
    defer a.free(none);
    try peer.take(0, none);
    try t.expectEqual(@as(usize, 0), peer.rowCount());
}
