//! Actual page ownership and refusal order under a recording CUDA driver, without devices.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const lanes = @import("lanes");
const t = std.testing;

const F = struct {
    var next: u64 = 1 << 30;
    var allocs: usize = 0;
    var frees: usize = 0;
    var context_waits: usize = 0;
    var pending = false;
    var fail_wait = false;
    var unsafe_frees: usize = 0;
    var fail_put = false;
    var unloads: usize = 0;
    fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(u64, n + 256, 256);
        allocs += 1;
        return 0;
    }
    fn free(_: u64) callconv(.c) cuda.abi.Result {
        frees += 1;
        unsafe_frees += @intFromBool(pending);
        return 0;
    }
    fn clear(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
    fn put(_: u64, _: ?*const anyopaque, _: usize) callconv(.c) cuda.abi.Result {
        return if (fail_put) cuda.abi.error_not_ready else 0;
    }
    fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
        unloads += 1;
        return 0;
    }
    fn wait(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
        if (fail_wait) return cuda.abi.error_not_ready;
        pending = false;
        return 0;
    }
    fn contextWait() callconv(.c) cuda.abi.Result {
        context_waits += 1;
        pending = false;
        return 0;
    }
};

const Fixture = struct {
    d: cuda.Driver,
    e: glm.engine.Engine,
    layers: [2]glm.weights.Layer,
    p: *glm.pages.Pages,
    fn init(f: *Fixture, streams: usize) !void {
        try f.prepare();
        f.p = try f.make(streams);
        f.e.pages = f.p;
    }
    fn prepare(f: *Fixture) !void {
        F.allocs = 0;
        F.frees = 0;
        F.context_waits = 0;
        F.unsafe_frees = 0;
        F.pending = false;
        F.fail_wait = false;
        F.fail_put = false;
        F.unloads = 0;
        f.d = undefined;
        f.d.api.cuMemAlloc_v2 = F.alloc;
        f.d.api.cuMemFree_v2 = F.free;
        f.d.api.cuMemsetD8_v2 = F.clear;
        f.d.api.cuMemcpyHtoD_v2 = F.put;
        f.d.api.cuStreamSynchronize = F.wait;
        f.d.api.cuCtxSynchronize = F.contextWait;
        f.e = undefined;
        f.e.gpa = t.allocator;
        f.e.d = &f.d;
        f.e.stream = .{ .d = &f.d, .handle = null };
        f.e.capacity = 4096;
        f.e.settings = .{};
        f.e.seqs = .empty;
        f.e.images = null;
        f.e.pages = null;
        f.e.w = undefined;
        f.e.w.config = .{ .layers = 2, .kv_lora = 16, .i_dim = 8, .hidden = 64 };
        f.e.w.share = try glm.shard.Share.of(&f.e.w.config, 0, 2);
        f.layers[0].mixer = .{ .mla = undefined };
        f.layers[1].mixer = .{ .kda = undefined };
        f.e.w.layers = &f.layers;
        try glm.pages.ownSequence(&f.e);
    }
    fn make(f: *Fixture, streams: usize) !*glm.pages.Pages {
        const layout = try glm.pages.layout(t.allocator, &f.e, 0);
        const p = try glm.pages.Pages.init(t.allocator, &f.e, layout, @max(streams, 1), 8, true);
        p.poison = false;
        return p;
    }
    fn deinit(f: *Fixture) void {
        F.fail_wait = false;
        F.pending = false;
        f.p.deinit();
        f.freeSequences();
    }
    fn freeSequences(f: *Fixture) void {
        for (f.e.seqs.items) |q| if (q) |s| {
            s.st.deinit();
            t.allocator.destroy(s);
        };
        f.e.seqs.deinit(t.allocator);
    }
};

test "GLM pages: a failed page-owner wait must fall back to the context before freeing" {
    var f: Fixture = undefined;
    try f.init(0);
    defer f.freeSequences();
    F.pending = true;
    F.fail_wait = true;
    f.p.deinit();
    F.pending = false;
    F.fail_wait = false;
    std.debug.print("page deinit: context_waits={d}, unsafe_frees={d}\n", .{ F.context_waits, F.unsafe_frees });
    try t.expect(F.context_waits > 0);
    try t.expectEqual(@as(usize, 0), F.unsafe_frees);
}

fn overflow(capped: bool) !void {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    var back: glm.lanes.Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer back.deinit();
    var ss: [3]lanes.Stream = undefined;
    for (&ss) |*s| s.* = try lanes.Stream.init(t.allocator, .{ .id = "page-test", .prompt = &.{100}, .max_new = 8 });
    defer for (&ss) |*s| s.deinit(t.allocator);
    try t.expectEqual(@as(u32, 0), (try back.claim(&ss[0])).slot);
    try t.expectEqual(@as(u32, 1), (try back.claim(&ss[1])).slot);
    _ = try f.p.open(0, 1, 8);
    _ = try f.p.open(1, 1, 8);
    try t.expect(!back.fits(1, 8));
    const before = F.allocs;
    const prior = if (capped) try cuda.memory.setAllocationLimit(cuda.memory.usage(false).device + cuda.memory.usage(false).host) else null;
    defer {
        if (capped) _ = cuda.memory.setAllocationLimit(prior) catch unreachable;
    }
    const b = back.backend();
    const result = b.vtable.prefill_step.?(b.ptr, &ss[2]);
    if (result) |_| return error.ExpectedFailure else |err| {
        std.debug.print("all-on-pages third opening: capped={}, error={s}, allocations={d}->{d}, slots={d}\n", .{ capped, @errorName(err), before, F.allocs, f.e.seqs.items.len });
        try t.expectEqual(error.LaneMemory, err);
    }
    try t.expectEqual(before, F.allocs);
    try t.expect(!back.lanes.contains(&ss[2]));
    try t.expectEqual(@as(usize, 2), back.lanes.count());
}

test "GLM pages: all-on-pages rejects the unpaged slot before allocating a whole window" {
    try overflow(false);
}

test "GLM pages: all-on-pages reaches LaneMemory even at the exact admitted allocation cap" {
    try overflow(true);
}

test "GLM pages: a failed final constructor wait drains before freeing every partial sequence" {
    var f: Fixture = undefined;
    try f.prepare();
    defer f.freeSequences();
    F.pending = true;
    F.fail_wait = true;
    try t.expectError(error.NotReady, f.make(2));
    F.fail_wait = false;
    try t.expectEqual(@as(usize, 1), F.context_waits);
    try t.expectEqual(@as(usize, 0), F.unsafe_frees);
    try t.expectEqual(F.allocs - 1, F.frees);
    try t.expect(f.e.seqs.items[0] != null);
    try t.expectEqual(@as(usize, 0), f.e.seqs.items[0].?.st.capacity);
    for (f.e.seqs.items[1..]) |q| try t.expect(q == null);
    f.p = try f.make(2);
    f.p.deinit();
}

test "GLM pages: table refusal cancels the leader lease and retry returns every page" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    F.fail_put = true;
    try t.expectError(error.NotReady, f.p.open(0, 600, 100));
    try t.expect(!f.p.used[0]);
    try t.expect(f.p.pool.idle());
    try t.expectEqual(@as(usize, 0), f.p.leases[0].ids.items.len);
    F.fail_put = false;
    _ = try f.p.open(0, 600, 100);
    const c = f.p.pool.census();
    try t.expectEqual(c.total, c.free + c.promised + c.kept + c.leased);
    f.p.close(0);
    try t.expectEqual(@as(usize, 8), f.p.pool.census().free);
}

test "GLM pages: a follower refuses a duplicate owner and rolls back a failed table upload" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    f.p.leader = false;
    F.fail_put = true;
    try t.expectError(error.NotReady, f.p.map(0, &.{ 3, 2 }));
    try t.expectEqual(@as(usize, 0), f.p.leases[0].ids.items.len);
    try t.expect(!f.p.used[0]);
    F.fail_put = false;
    try f.p.map(0, &.{ 3, 2 });
    try t.expectError(error.BadStep, f.p.map(0, &.{ 1, 0 }));
    try t.expectEqualSlices(u32, &.{ 3, 2 }, f.e.seqs.items[0].?.st.pages);
    f.p.close(0);
    try f.p.map(0, &.{ 1, 0 });
    f.p.close(0);
    try t.expect(f.p.pool.idle());
}

test "GLM pages: a failed begin packet cancels its lease and lane before a retry" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    try f.e.bind(0);
    f.e.dropped = .empty;
    f.e.m.glue.put_words = .{ .d = &f.d, .handle = null };
    var link: glm.follow.Link = undefined;
    link.ids = &.{};
    link.asleep = false;
    var back: glm.lanes.Lanes = .{ .x = .{ .e = &f.e, .link = &link } };
    defer back.deinit();
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "page-test", .prompt = &.{100}, .max_new = 8 });
    defer stream.deinit(t.allocator);
    const b = back.backend();
    const before = F.allocs;
    for (0..2) |_| {
        try t.expectError(error.TooManyIds, b.vtable.prefill_step.?(b.ptr, &stream));
        try t.expectEqual(@as(usize, 0), back.lanes.count());
        try t.expect(f.p.pool.idle());
        try t.expectEqual(@as(usize, 8), f.p.pool.census().free);
        try t.expectEqual(@as(u32, 0), f.e.slot);
        try t.expectEqual(before, F.allocs);
    }
}

test "GLM pages: a failed seat binding returns its pages and lane before retry" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    try f.e.bind(1);
    f.d.api.cuModuleUnload = F.unload;
    const v = try imageOwner(&f);
    v.slot = 1;
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    v.gpa = failing.allocator();
    f.e.images = v;
    defer {
        v.gpa = t.allocator;
        v.deinit();
        f.e.images = null;
    }
    var back: glm.lanes.Lanes = .{ .x = .{ .e = &f.e, .link = undefined } };
    defer back.deinit();
    var stream = try lanes.Stream.init(t.allocator, .{ .id = "page-test", .prompt = &.{100}, .max_new = 8 });
    defer stream.deinit(t.allocator);
    try t.expectError(error.OutOfMemory, back.seat(&stream, 1, 8));
    try t.expectEqual(@as(usize, 0), back.lanes.count());
    try t.expect(f.p.pool.idle());
    try t.expectEqual(@as(usize, 8), f.p.pool.census().free);
    try t.expectEqual(@as(u32, 1), f.e.slot);
    v.gpa = t.allocator;
    try t.expectEqual(@as(usize, 1), (try back.seat(&stream, 1, 8)).len);
    try t.expectEqual(@as(usize, 1), back.lanes.count());
    try t.expectEqual(@as(u32, 0), f.e.slot);
    f.e.release(0);
    try t.expect(f.p.pool.idle());
}

test "GLM pages: a refused image activation frees a new window and restores the previous binding" {
    var f: Fixture = undefined;
    try f.init(2);
    defer f.deinit();
    f.d.api.cuModuleUnload = F.unload;
    f.e.cur = f.e.seqs.items[1].?;
    f.e.slot = 1;
    var failing = t.FailingAllocator.init(t.allocator, .{ .fail_index = 0 });
    const v = try imageOwner(&f);
    v.gpa = failing.allocator();
    f.e.images = v;
    defer {
        v.deinit();
        f.e.images = null;
    }
    const live = F.allocs - F.frees;
    try t.expectError(error.OutOfMemory, f.e.bind(3));
    try t.expectEqual(live, F.allocs - F.frees);
    try t.expect(f.e.seqs.items[3] == null);
    try t.expectEqual(f.e.seqs.items[1].?, f.e.cur);
    try t.expectEqual(@as(u32, 1), f.e.slot);
}

const Open = struct {
    var phase: usize = 0;
    fn after(_: void) !void {
        return switch (phase) {
            0 => error.Fp8OnPages,
            1 => error.DrafterNeedsDrafts,
            2 => error.LinkRefused,
            3 => error.RoomRefused,
            else => {},
        };
    }
};

fn imageOwner(f: *Fixture) !*glm.images.Images {
    const a = t.allocator;
    const v = try a.create(glm.images.Images);
    errdefer a.destroy(v);
    const rows = try cuda.DeviceBuffer.alloc(&f.d, 128);
    errdefer {
        var b = rows;
        b.free();
    }
    const map = try cuda.DeviceBuffer.alloc(&f.d, 16);
    errdefer {
        var b = map;
        b.free();
    }
    v.* = .{
        .gpa = a,
        .io = t.io,
        .d = &f.d,
        .module = .{ .d = &f.d, .handle = null },
        .run = .{ .stream = f.e.stream, .f = undefined },
        .rows = rows,
        .map = map,
        .host_map = try a.alloc(i32, 4),
        .limits = .{ .image_tokens = 4, .max_images = 1 },
    };
    return v;
}

test "GLM pages: every post-image startup refusal drains and frees the actual image owner exactly once" {
    for (0..4) |phase| {
        var f: Fixture = undefined;
        try f.prepare();
        defer f.freeSequences();
        f.d.api.cuModuleUnload = F.unload;
        f.e.images = try imageOwner(&f);
        defer if (f.e.images) |v| v.deinit();
        const before = F.frees;
        F.pending = true;
        F.fail_wait = true;
        Open.phase = phase;
        if (glm.open_owner.finish(&f.e, {}, Open.after)) |_| return error.ExpectedRefusal else |_| {}
        try t.expect(f.e.images == null);
        try t.expectEqual(before + 2, F.frees);
        try t.expectEqual(@as(usize, 0), F.unsafe_frees);
        try t.expectEqual(@as(usize, 1), F.context_waits);
        try t.expectEqual(@as(usize, 1), F.unloads);
        F.fail_wait = false;
        f.e.images = try imageOwner(&f);
        Open.phase = 4;
        try glm.open_owner.finish(&f.e, {}, Open.after);
        f.e.images.?.deinit();
        f.e.images = null;
        try t.expectEqual(F.allocs - 1, F.frees);
        try t.expectEqual(@as(usize, 2), F.unloads);
    }
}

const registry = @import("core").contract.registry;
const limits = @import("core").contract.limits;
const launches = [_]struct { symbol: []const u8, image: []const u8, threads: u32, dynamic: u32 = 0 }{
    .{ .symbol = "tf_glm_lat_write", .image = "glm_glue", .threads = 128 },
    .{ .symbol = "tf_glm_pool_keys", .image = "glm_glue", .threads = 32 },
    .{ .symbol = "tf_glm_scores", .image = "glm_glue", .threads = 128 },
    .{ .symbol = "tf_glm_dsa_tokens", .image = "glm_ops", .threads = 256 },
    .{ .symbol = "tf_glm_chunks_16", .image = "glm_glue", .threads = 256, .dynamic = glm.glue.chunksSmem(16) },
    .{ .symbol = "tf_glm_chunks_32", .image = "glm_glue", .threads = 256, .dynamic = glm.glue.chunksSmem(32) },
};

fn inventory(records: []const registry.CudaRecord) !void {
    for (launches) |want| {
        const r = for (records) |r| {
            if (std.mem.eql(u8, want.symbol, r.launch.symbol)) break r;
        } else return error.MissingPagedLaunch;
        if (!std.mem.eql(u8, want.image, r.launch.image) or want.threads != r.launch.threads or
            want.dynamic != r.launch.dynamic or (want.dynamic > 48 * 1024) != r.launch.opt_in)
            return error.BadPagedLaunch;
        var notes: std.ArrayList(u8) = .empty;
        defer notes.deinit(t.allocator);
        for ([_]?limits.Nvcc{ null, .{ .major = 13, .minor = 0 }, .{ .major = 13, .minor = 2 }, .{ .major = 13, .minor = 3 } }) |version| {
            const usage = registry.usageFor(r, version);
            for (limits.sms, usage) |sm, u| {
                if (u.regs == 0) return error.MissingPagedUsage;
                if (u.local != 0 or u.static != (if (std.mem.eql(u8, want.symbol, "tf_glm_scores")) @as(u32, 32768) else 0))
                    return error.BadPagedUsage;
                if (try limits.cuda(r.launch, u, sm, &notes, t.allocator) != 0) return error.BadPagedUsage;
            }
        }
    }
}

test "GLM pages: all six changed symbols have complete launches and measured compiler declarations" {
    try inventory(&registry.all_cuda);
}

test "GLM pages: planted missing launch and wrong shared-memory controls are rejected" {
    var records: [6]registry.CudaRecord = undefined;
    for (launches, &records) |want, *out| {
        out.* = for (registry.all_cuda) |r| {
            if (std.mem.eql(u8, want.symbol, r.launch.symbol)) break r;
        } else return error.MissingPagedLaunch;
    }
    for (0..records.len) |missing| {
        var incomplete: std.ArrayList(registry.CudaRecord) = .empty;
        defer incomplete.deinit(t.allocator);
        for (records, 0..) |r, i| if (i != missing) try incomplete.append(t.allocator, r);
        try t.expectError(error.MissingPagedLaunch, inventory(incomplete.items));
    }
    var wrong = records;
    wrong[4].launch.dynamic -= 16;
    try t.expectError(error.BadPagedLaunch, inventory(&wrong));
    wrong = records;
    wrong[0].usage[0].regs = 0;
    try t.expectError(error.MissingPagedUsage, inventory(&wrong));
    wrong = records;
    wrong[2].usage[0].static -= 16;
    try t.expectError(error.BadPagedUsage, inventory(&wrong));
}
