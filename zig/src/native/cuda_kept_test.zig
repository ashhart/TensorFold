//! The shared kept image over a driver whose device memory is host memory, each family's view, and stale identities.
const std = @import("std");
const cuda = @import("cuda");
const api = @import("engine_api");
const kept = @import("cuda_kept.zig");
const disk = @import("cuda_disk.zig");
const rows = api.prompt_cache.rows;
const t = std.testing;

/// Device addresses are host addresses; a copy lands at once; `refuse` fails the next copy past `allowed`.
const Host = struct {
    var allowed: usize = std.math.maxInt(usize);
    var copies: usize = 0;
    var pending: usize = 0;
    fn queue() bool {
        if (copies == allowed) return false;
        copies += 1;
        pending += 1;
        return true;
    }
    fn toHost(out: ?*anyopaque, dev: u64, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (!queue()) return cuda.abi.error_not_ready; // check() returns NotReady for it, logging nothing
        const p: [*]const u8 = @ptrFromInt(dev);
        @memcpy(@as([*]u8, @ptrCast(out.?))[0..n], p[0..n]);
        return cuda.abi.success;
    }
    fn toDevice(dev: u64, in: ?*const anyopaque, n: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (!queue()) return cuda.abi.error_not_ready;
        const p: [*]u8 = @ptrFromInt(dev);
        @memcpy(p[0..n], @as([*]const u8, @ptrCast(in.?))[0..n]);
        return cuda.abi.success;
    }
    fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        pending = 0;
        return cuda.abi.success;
    }
    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuMemcpyDtoHAsync_v2 = toHost;
        d.api.cuMemcpyHtoDAsync_v2 = toDevice;
        d.api.cuStreamSynchronize = sync;
        return d;
    }
};

/// A family with a target's rows and a second buffer, as Gemma's drafter context.
const View = struct {
    pub const Target = struct {
        d: *const cuda.Driver,
        s: cuda.Stream,
        gpa: std.mem.Allocator = t.allocator,
        restored: u32 = 0,
    };
    pub const State = struct { at: u32, one: cuda.DeviceBuffer, two: ?cuda.DeviceBuffer };
    pub const probe_seed: u64 = 1;
    pub fn buffers(st: *const State) [2]?cuda.DeviceBuffer {
        return .{ st.one, st.two };
    }
    pub fn facts(_: *const Target) [2]u64 {
        return .{ 3, 4 };
    }
    pub fn parts(_: *const Target, at: u32, out: anytype, a: std.mem.Allocator) !void {
        try out.append(a, .{ .bytes = @as(u64, at) * 8, .row = 8 });
        try out.append(a, .{ .bytes = 32 });
    }
    pub fn blank(x: *Target, at: u32) !*State {
        const st = try x.gpa.create(State);
        const rows_ = try x.gpa.alloc(u8, one(at));
        st.* = .{ .at = at, .one = buffer(x.d, rows_), .two = buffer(x.d, try x.gpa.alloc(u8, 32)) };
        return st;
    }
    fn one(at: u32) usize {
        return @as(usize, at) * 8;
    }
};

/// The store's copies of `View`: a state's rows are its position's, so a state read back shows it was read.
const Copies = struct {
    fn of(p: *anyopaque) *View.Target {
        return @ptrCast(@alignCast(p));
    }
    fn state(p: api.prompt_cache.Saved) *View.State {
        return @ptrCast(@alignCast(p));
    }
    fn bytes(_: *anyopaque, at: u32) u64 {
        return View.one(at) + 32;
    }
    fn save(p: *anyopaque, _: ?*anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
        const st = try View.blank(of(p), at);
        for (hostOf(st.one), 0..) |*x, i| x.* = @truncate(i *% 31);
        @memset(hostOf(st.two.?), 9);
        return st;
    }
    fn restore(p: *anyopaque, _: ?*anyopaque, saved: api.prompt_cache.Saved) anyerror!void {
        of(p).restored = state(saved).at;
        for (hostOf(state(saved).one), 0..) |x, i| try t.expectEqual(@as(u8, @truncate(i *% 31)), x);
    }
    fn drop(p: *anyopaque, saved: api.prompt_cache.Saved) void {
        const st = state(saved);
        of(p).gpa.free(hostOf(st.one));
        of(p).gpa.free(hostOf(st.two.?));
        of(p).gpa.destroy(st);
    }
    const vtable: api.prompt_cache.Snapshots.VTable = .{
        .bytes = bytes,
        .save = save,
        .restore = restore,
        .drop = drop,
        .rows = &K.image,
    };
};

const K = kept.Kept(View);

fn hostOf(b: cuda.DeviceBuffer) []u8 {
    const p: [*]u8 = @ptrFromInt(b.ptr);
    return p[0..b.len];
}

fn buffer(d: *const cuda.Driver, bytes: []u8) cuda.DeviceBuffer {
    return .{ .d = d, .ptr = @intFromPtr(bytes.ptr), .len = bytes.len, .borrowed = true };
}

test "CUDA kept: an image's ranges cross the words and both buffers, and read back equal" {
    var d = Host.driver();
    var target: View.Target = .{ .d = &d, .s = .{ .d = &d, .handle = @ptrFromInt(331) } };
    var a: [32]u8 = undefined;
    var b: [32]u8 = undefined;
    for (&a, &b, 0..) |*x, *y, i| {
        x.* = @intCast(i);
        y.* = @intCast(100 + i);
    }
    var st: View.State = .{ .at = 4, .one = buffer(&d, &a), .two = buffer(&d, &b) };
    Host.allowed = std.math.maxInt(usize);
    var host: [16 + 64]u8 = undefined;
    const all = [_]rows.image.Range{.{ .off = 0, .len = host.len }};
    try K.image.gather(&target, &st, &all, &host);
    try t.expectEqual(@as(u64, 4), std.mem.readInt(u64, host[0..8], .little));
    try t.expectEqualSlices(u8, &a, host[16..48]);
    try t.expectEqualSlices(u8, &b, host[48..80]);
    @memset(&a, 0);
    @memset(&b, 0);
    const split = [_]rows.image.Range{ .{ .off = 40, .len = 40 }, .{ .off = 0, .len = 40 } };
    var parted: [80]u8 = undefined;
    @memcpy(parted[0..40], host[40..80]);
    @memcpy(parted[40..80], host[0..40]);
    try K.image.scatter(&target, &st, &split, &parted);
    try t.expectEqualSlices(u8, host[16..48], &a);
    try t.expectEqualSlices(u8, host[48..80], &b);
}

test "CUDA kept: a state whose words stand elsewhere is refused as a bad file" {
    var d = Host.driver();
    var target: View.Target = .{ .d = &d, .s = .{ .d = &d, .handle = @ptrFromInt(331) } };
    var a: [32]u8 = undefined;
    var st: View.State = .{ .at = 4, .one = buffer(&d, &a), .two = null };
    var host: [16]u8 = @splat(0);
    host[0] = 5;
    try t.expectError(error.KeptWordsDiffer, K.image.scatter(&target, &st, &.{.{ .off = 0, .len = 16 }}, &host));
}

test "Flash Next completion: a kept state's copy out or in that fails part way drains the copies it queued" {
    var d = Host.driver();
    var target: View.Target = .{ .d = &d, .s = .{ .d = &d, .handle = @ptrFromInt(331) } };
    var a: [1024]u8 = undefined;
    var st: View.State = .{ .at = 0, .one = buffer(&d, &a), .two = null };
    var host: [16 + 1024]u8 = @splat(0);
    // the words and half the state queue a copy; the driver refuses the other half, or that range runs past the state
    for ([_]bool{ true, false }) |refused| for ([_]bool{ true, false }) |out| {
        Host.copies = 0;
        Host.pending = 0;
        Host.allowed = 1;
        defer Host.allowed = std.math.maxInt(usize);
        const second: u64 = if (refused) 512 else 1024;
        const ranges = [_]rows.image.Range{ .{ .off = 0, .len = 16 + 512 }, .{ .off = 16 + 512, .len = second } };
        const image = K.image;
        const got = if (out)
            image.gather(&target, &st, &ranges, &host)
        else
            image.scatter(&target, &st, &ranges, &host);
        try t.expectError(if (refused) error.NotReady else error.BadKeptRange, got);
        try t.expectEqual(@as(usize, 1), Host.copies);
        try t.expectEqual(@as(usize, 0), Host.pending); // the caller's `host` may go: nothing copies on it
    };
}

test "CUDA kept: another tokenizer, template or drafter moves the key, a missing file does not fail" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    const bare = try kept.promptHash(a, t.io, dir, null);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = "one" });
    const template = try kept.promptHash(a, t.io, dir, null);
    try t.expect(bare != template);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = "two" });
    try t.expect(template != try kept.promptHash(a, t.io, dir, null));
    const same = try kept.promptHash(a, t.io, dir, null);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "tokenizer.json", .data = "{}" });
    try t.expect(same != try kept.promptHash(a, t.io, dir, null));
    const plain = try kept.promptHash(a, t.io, dir, null);
    (try tmp.dir.createDirPathOpen(t.io, "draft", .{})).close(t.io);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "draft/config.json", .data = "{}" });
    const drafted = try kept.promptHash(a, t.io, dir, try std.fs.path.join(a, &.{ dir, "draft" }));
    try t.expect(plain != drafted);
}

const Recording = struct {
    fn name(out: [*]u8, _: c_int, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        @memcpy(out[0..5], "fake\x00");
        return 0;
    }
    fn attribute(out: *c_int, _: cuda.abi.DeviceAttribute, _: cuda.abi.Device) callconv(.c) cuda.abi.Result {
        out.* = 12;
        return 0;
    }
    fn version(out: *c_int) callconv(.c) cuda.abi.Result {
        out.* = 13000;
        return 0;
    }
};

const rules: api.prompt_cache.Rules = .{ .lookahead = 0, .min_prompt = 0, .min_gap = 1 };

/// One turn of `prompt` through a store under `id`: where it resumes, its state written once it keeps one.
fn turn(a: std.mem.Allocator, root: []const u8, id: u64, target: *View.Target, prompt: []const u32, history: u32) !u32 {
    const copies: api.prompt_cache.Snapshots = .{ .ptr = target, .vtable = &Copies.vtable };
    const cap: u64 = 1 << 34;
    const options: rows.Options = .{
        .sync_writes = true,
        .root = root,
        .identity = id,
        .cap = cap,
        .floor = 0,
        .daily = cap,
    };
    const r = try rows.Rows.open(t.allocator, copies, options);
    defer r.deinit();
    var s = api.prompt_cache.Store.init(t.allocator, copies, rules, 1 << 30);
    defer s.deinit();
    s.disk = r;
    const plan = try s.begin(a, prompt, history, &.{}, &.{}, null, &.{});
    for (plan.marks) |mark| if (mark > plan.from) {
        _ = s.keep(prompt, mark, null, &.{}, &.{});
    };
    while (rows.write.step(r)) {}
    return plan.from;
}

test "CUDA kept: a saved state resumes under its own key, and a stale key misses and leaves it for its own" {
    var arena: std.heap.ArenaAllocator = .init(t.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(t.io, ".", a);
    try tmp.dir.writeFile(t.io, .{ .sub_path = "config.json", .data = "{}" });
    try tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = "one" });
    const root = try std.fs.path.join(a, &.{ dir, "states" });
    var d = Host.driver();
    d.api.cuDeviceGetName = Recording.name;
    d.api.cuDeviceGetAttribute = Recording.attribute;
    d.api.cuDriverGetVersion = Recording.version;
    const gpu: cuda.Context = .{ .d = &d, .device = 0, .handle = null };
    const key = struct {
        fn of(arena_: std.mem.Allocator, g: *const cuda.Context, folder: []const u8, form: u64) !u64 {
            const facts = [_]u64{form};
            const profile: @import("disk_identity.zig").Profile = .{ .family = "dory", .backend = "cuda" };
            const id = try disk.identityOf(arena_, t.io, g, folder, null, &facts, 42, profile);
            return K.prompted(arena_, t.io, .{ .dir = folder, .model_type = "dory" }, id); // as attachKept extends it
        }
    }.of;
    const id = try key(a, &gpu, dir, 1);
    var target: View.Target = .{ .d = &d, .s = .{ .d = &d, .handle = @ptrFromInt(331) } };
    const first = try a.alloc(u32, 1500);
    for (first, 0..) |*x, i| x.* = @intCast(1000 + i);
    const second = try std.mem.concat(a, u32, &.{ first[0..1401], &[_]u32{ 9, 8, 7, 6 } });
    try t.expectEqual(@as(u32, 0), try turn(a, root, id, &target, first, 1400));
    try t.expectEqual(@as(u32, 1400), try turn(a, root, id, &target, second, 1404));
    try t.expectEqual(@as(u32, 1400), target.restored);
    target.restored = 0;
    // another KV form, then another template: each its own namespace, a cold pass, the old state unread
    try t.expectEqual(@as(u32, 0), try turn(a, root, try key(a, &gpu, dir, 2), &target, second, 1404));
    try tmp.dir.writeFile(t.io, .{ .sub_path = "chat_template.jinja", .data = "two" });
    const moved = try key(a, &gpu, dir, 1);
    try t.expect(moved != id);
    try t.expectEqual(@as(u32, 0), try turn(a, root, moved, &target, second, 1404));
    try t.expectEqual(@as(u32, 0), target.restored);
    // its own key still resumes, at the state the second turn kept
    try t.expectEqual(@as(u32, 1404), try turn(a, root, id, &target, second, 1404));
}

/// `View` at a served window, which the store's permit takes whole.
const Windowed = struct {
    pub const Target = View.Target;
    pub const State = View.State;
    pub const probe_seed = View.probe_seed;
    pub const buffers = View.buffers;
    pub const facts = View.facts;
    pub const parts = View.parts;
    pub const blank = View.blank;
    pub var window: usize = 0;
    pub fn context(_: *const Target) usize {
        return window;
    }
};

test "CUDA kept: core's permit takes a state at the served window whole, within the RAM store's budget" {
    var d = Host.driver();
    var target: View.Target = .{ .d = &d, .s = .{ .d = &d, .handle = @ptrFromInt(331) } };
    const W = kept.Kept(Windowed);
    try t.expectEqual(@as(u32, 0), K.longest(&target)); // no window declared: the store's default permit
    Windowed.window = 1 << 28; // 2 GiB of rows at 8 bytes each, past the 1 GiB default
    try t.expectEqual(@as(u32, 1 << 28), W.longest(&target));
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(t.io, ".", t.allocator);
    defer t.allocator.free(root);
    const vt: api.prompt_cache.Snapshots.VTable = .{ .bytes = Copies.bytes, .save = Copies.save, .restore = Copies.restore, .drop = Copies.drop, .rows = &W.image };
    const copies: api.prompt_cache.Snapshots = .{ .ptr = &target, .vtable = &vt };
    for ([_]u64{ 8 << 30, 3 << 29 }) |budget| {
        var s = api.prompt_cache.Store.init(t.allocator, copies, rules, budget);
        const options: rows.Options = .{ .root = root, .identity = 1, .cap = 1 << 34, .floor = 0, .daily = 1 << 34, .longest = W.longest(&target) };
        const r = rows.attach(t.allocator, &s, options) catch |e| {
            s.deinit();
            return e;
        };
        defer r.deinit();
        defer s.deinit(); // the store before its disk, as the rows fixtures close them
        const whole = (@as(u64, 1 << 28) * 8) + 32 + 16;
        if (budget > whole) try t.expect(r.staging_bytes > whole) else try t.expectEqual(budget, r.staging_bytes);
    }
}

test "CUDA kept: Gemma 4, Dory and Spark2.5 take the store, a family without a view still refuses" {
    const gemma = @import("gemma4_cuda");
    const dory = @import("dory");
    inline for (.{ gemma.native, gemma.native31, dory.native, @import("spark25").native }) |F| {
        try t.expect(comptime disk.keptDisk(F) != null);
    }
    const open: api.Open = .{ .dir = "d", .model_type = "dory", .learn_conversations = true };
    try t.expect(@import("cuda_refusal.zig").family(dory.native, open) == null);
    try t.expect(comptime disk.keptDisk(@import("kolibri1").native) == null);
}
