//! The backend's image path on a recording fake device: each slot's rows, encode at lane begin, refusals by name.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const mdl = @import("model.zig");
const kern = @import("kernels.zig");
const backend = @import("backend.zig");
const recipe = @import("recipe.zig");
const fixtures = @import("tiny_test.zig");
const fail = @import("device_fail_test.zig");
const Fake = fail.Fake;
const t = std.testing;

const mark: u32 = 500;
const row = 256 * 2;

const Cancel = struct {
    after: usize, // cancelled from this check on
    checks: usize = 0,
    fn check(p: *anyopaque) bool {
        const c: *Cancel = @ptrCast(@alignCast(p));
        c.checks += 1;
        return c.checks > c.after;
    }
};

fn image(at: u32, tokens: u32) recipe.Image {
    return .{ .pixels = &.{}, .gh = 4, .gw = 4, .at = at, .tokens = tokens };
}

/// One prefill step of a stream of `prompt` and `images`, cancelled after `after` checks.
fn step(b: lanes.backend.Backend, prompt: []const u32, images: []const recipe.Image, after: usize) !bool {
    var c: Cancel = .{ .after = after };
    var s = try lanes.Stream.init(t.allocator, .{ .id = "img", .prompt = prompt, .max_new = 4, .images = images, .cancel_check = .{ .ptr = &c, .check = Cancel.check } });
    defer s.deinit(t.allocator);
    return b.vtable.prefill_step.?(b.ptr, &s, 0);
}

fn Engine(comptime R: type) type {
    return struct {
        d: cuda.Driver,
        m: mdl.Model = undefined,
        k: kern.Kernels = undefined,
        own: R.Kernels = .{},
        w: R.Weights = .{},
        l: cuda.qmmf.Lane = fail.lane(),

        fn stream(e: *const @This()) cuda.Stream {
            return .{ .d = &e.d, .handle = Fake.queue };
        }

        fn dense(e: *@This(), v: backend.Dense(R).Vision) !backend.Dense(R) {
            e.m = try mdl.Model.init(t.allocator, &e.d, e.stream(), fixtures.tiny, 4096, &e.l);
            errdefer e.m.deinit();
            return backend.Dense(R).init(t.allocator, &e.d, &e.m, &e.k, &e.own, &e.w, e.stream(), v);
        }
    };
}

test "each slot owns its image rows: the first at init, the admitted lanes' at attach, freed with the engine" {
    var e: Engine(fixtures.TinyTower) = .{ .d = Fake.driver() };
    const held = Fake.live;
    var tower: fixtures.Fake = .{};
    var x = try e.dense(&tower);
    try t.expectEqual(@as(usize, 64 * row), x.feat_bytes);
    try t.expectEqual(@as(usize, 1), x.feats.items.len);
    try x.attach(2, 4);
    try t.expectEqual(@as(usize, 3), x.feats.items.len);
    for (x.feats.items) |b| try t.expectEqual(@as(usize, 64 * row), b.len);
    x.deinit();
    e.m.deinit();
    try t.expectEqual(held, Fake.live);
    var none = try e.dense(null); // served without --vision: no rows
    try none.attach(2, 4);
    try t.expectEqual(@as(usize, 0), none.feats.items.len);
    none.deinit();
    e.m.deinit();
    try t.expectEqual(held, Fake.live);
    cuda.copy_source.dropSpares(&e.d, cuda.copy_source.current());
}

test "a lane begins: its images through the tower in order into its slot's rows, a cancel check before each" {
    var e: Engine(fixtures.TinyTower) = .{ .d = Fake.driver() };
    var tower: fixtures.Fake = .{};
    var x = try e.dense(&tower);
    defer {
        x.deinit();
        e.m.deinit();
        cuda.copy_source.dropSpares(&e.d, cuda.copy_source.current());
    }
    try x.attach(2, 4);
    const b = x.backend();
    var buf: [40]u32 = @splat(7);
    @memset(buf[3..11], mark);
    @memset(buf[20..24], mark);
    const two = [_]recipe.Image{ image(3, 8), image(20, 4) };
    x.free[0] = false; // another stream holds slot 0: this lane takes slot 1
    const syncs = Fake.proofs;
    try t.expectError(error.Cancelled, step(b, &buf, &two, 2)); // both encode, then the chunk's check stops it
    try t.expectEqual(@as(usize, 2), tower.n);
    try t.expectEqual(fixtures.Fake.Call{ .at = 3, .tokens = 8, .out = x.feats.items[1].ptr }, tower.calls[0]);
    try t.expectEqual(fixtures.Fake.Call{ .at = 20, .tokens = 4, .out = x.feats.items[1].ptr + 8 * row }, tower.calls[1]);
    try t.expect(x.lanes.count() == 0 and x.free[1]); // dropped, after its stream drained
    try t.expect(!Fake.pending and Fake.proofs == syncs);
    tower.n = 0;
    try t.expectError(error.Cancelled, step(b, &buf, &two, 1)); // cancelled before the second image
    try t.expectEqual(@as(usize, 1), tower.n);
    try t.expect(x.lanes.count() == 0 and x.free[1]);
    tower.n = 0;
    try t.expectError(error.DenseImageSpan, step(b, &buf, &.{image(2, 8)}, 9)); // a text row inside: no slot taken
    try t.expectError(error.DenseImageSpan, step(b, &buf, &.{ image(3, 8), image(3, 8) }, 9));
    var wide: [80]u32 = @splat(mark);
    try t.expectError(error.DenseImageSpan, step(b, &wide, &.{ image(0, 40), image(40, 25) }, 9)); // 65 > 64 tokens
    try t.expectEqual(@as(usize, 0), tower.n);
    x.tower = null;
    try t.expectError(error.DenseServedWithoutVision, step(b, &buf, &two, 9));
    x.tower = &tower;
    try t.expect(x.lanes.count() == 0 and x.free[1] and x.free[2]);
}

test "a recipe without a tower refuses images by name, as before" {
    var e: Engine(fixtures.Tiny) = .{ .d = Fake.driver() };
    var x = try e.dense({});
    defer {
        x.deinit();
        e.m.deinit();
        cuda.copy_source.dropSpares(&e.d, cuda.copy_source.current());
    }
    try x.attach(0, 2);
    var buf: [16]u32 = @splat(mark);
    try t.expectError(error.DenseTextOnly, step(x.backend(), &buf, &.{image(0, 4)}, 9));
    try t.expectEqual(@as(usize, 0), x.feats.items.len);
}

/// TinyTower with no layer work: a prompt chunk runs embed, the splice and the layers on the fake device alone.
const Quiet = struct {
    const Base = fixtures.TinyTower;
    const P = @import("pass.zig").Pass(Quiet);
    pub const name = Base.name;
    pub const model_type = Base.model_type;
    pub const formats = Base.formats;
    pub const default_context = Base.default_context;
    pub const reasoning = false;
    pub const post_embed: ?fn (*const P) anyerror!void = null;
    pub const Tower: ?type = fixtures.Fake;
    pub const Weights = Base.Weights;
    pub const Kernels = Base.Kernels;
    pub const geometry = Base.geometry;
    pub const load = Base.load;
    pub const weightBytes = Base.weightBytes;
    pub const head = Base.head;
    pub const explain = Base.explain;
    pub fn embed(_: *const P) !void {}
    pub fn layer(_: *const P, _: usize) !void {}
    pub fn final(_: *const P) !void {}
};

test "a prompt chunk that cuts an image copies that image's rows from its slot over the chunk's embedding rows" {
    var e: Engine(Quiet) = .{ .d = Fake.driver() };
    var tower: fixtures.Fake = .{};
    var x = try e.dense(&tower);
    defer {
        x.deinit();
        e.m.deinit();
        cuda.copy_source.dropSpares(&e.d, cuda.copy_source.current());
    }
    try x.attach(2, 4);
    var buf: [40]u32 = @splat(7);
    @memset(buf[3..11], mark);
    const one = [_]recipe.Image{image(3, 8)};
    var c: Cancel = .{ .after = 99 };
    var s = try lanes.Stream.init(t.allocator, .{ .id = "img", .prompt = &buf, .max_new = 4, .images = &one, .reuse = .{ .marks = &.{ 7, 20 } }, .cancel_check = .{ .ptr = &c, .check = Cancel.check } });
    defer s.deinit(t.allocator);
    const b = x.backend();
    try t.expect(!try b.vtable.prefill_step.?(b.ptr, &s, 0)); // the chunk [0, 7) ends inside the image
    const src = x.feats.items[x.lanes.get(&s).?.slot].ptr;
    try t.expectEqual(@as(usize, 1), Fake.copied);
    try t.expectEqual(fail.Fake.Copy{ .dst = e.m.s.h + 3 * row, .src = src, .len = 4 * row }, Fake.copies[0]);
    Fake.copied = 0;
    // the chunk [7, 20): the image's last four rows at its first row
    try t.expect(!try b.vtable.prefill_step.?(b.ptr, &s, 0));
    try t.expectEqual(@as(usize, 1), Fake.copied);
    try t.expectEqual(fail.Fake.Copy{ .dst = e.m.s.h, .src = src + 4 * row, .len = 4 * row }, Fake.copies[0]);
    c.after = 0; // the next chunk's check drops the lane
    try t.expectError(error.Cancelled, b.vtable.prefill_step.?(b.ptr, &s, 0));
    try t.expect(x.lanes.count() == 0);
}
