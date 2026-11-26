//! Host-only invariants for Inkling 2 Mini Metal's GPU sampling dispatch, scratch layout, and flight metadata.
const std = @import("std");
const builtin = @import("builtin");
const lanes = @import("lanes");
const fwd = @import("forward.zig");
const st = @import("state.zig");
const flight = @import("flight.zig");
const backend = @import("backend.zig");

test {
    if (builtin.os.tag.isDarwin()) _ = @import("gpu_tokens_gpu_test.zig");
}

test "drawMode splits greedy, two-pass top-k (1..64), rule_1024 (65..1024), and full-vocabulary draws" {
    const dummy: fwd.Ref = .{ .buf = .{ .id = @ptrFromInt(16) }, .off = 0 };
    const vocab: u32 = st.vocabulary;
    try std.testing.expectEqual(fwd.DrawMode.greedy, fwd.drawMode(.{ .how = .greedy, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.full, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 0 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.topk, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.topk, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 20 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.topk, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 64 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 65 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 128 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1024 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.full, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 1025 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.full, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 2048 }, .position = 1 } }, .out = dummy }, vocab));
    try std.testing.expectEqual(fwd.DrawMode.rule_1024, fwd.drawMode(.{ .how = .{ .sampled = .{ .sampling = .{ .seed = 1, .temperature = 1.0, .top_k = 20 }, .position = 1 } }, .out = dummy }, 16));
}

test "flight metadata stride and offsets respect Metal alignment rules" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(flight.FlightMeta));
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(flight.FlightMeta, "pos") % 4);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(flight.FlightMeta, "seg_of") % 4);
    try std.testing.expectEqual(@as(usize, 0), @offsetOf(flight.FlightMeta, "seg") % 8);
}

test "scratch reserves picks and two-pass top-k buffers and Metal defaults gpu_tokens off for TTFT safety" {
    try std.testing.expect(st.scratchBytes(st.batch_rows) > st.scratchBytes(1));
    var m: backend.Metal = undefined;
    m.options = .{ .capacity = 16, .budget = 0 };
    m.timed = 0;
    m.shared_timed = 0;
    try std.testing.expect(!m.facts().gpu_tokens);
    m.options.gpu_tokens = true;
    try std.testing.expect(m.facts().gpu_tokens);
}

const FakeMtl = struct {
    pub const ResourceOptions = struct {
        pub const shared: u32 = 0;
    };

    pub const FlightRec = struct {
        fail: bool = false,
        waits: u32 = 0,
        releases: u32 = 0,
        slot: u64 = 0,
        write_token: ?u32 = null,
        wait_after_release: bool = false,
        release_before_wait: bool = false,
        acted_after_free: bool = false,
    };

    pub const State = struct {
        flights: [64]FlightRec = @splat(.{}),
        flight_count: usize = 0,
        tokens: [flight.ring]u32 = @splat(0),
        meta: [flight.ring]flight.FlightMeta = undefined,
        alloc_count: u32 = 0,
        tokens_freed: u32 = 0,
        meta_freed: u32 = 0,
        cache_freed: u32 = 0,
        early_ring_free: bool = false,
        early_cache_free: bool = false,

        fn allRetired(s: *const State) bool {
            for (s.flights[0..s.flight_count]) |f| {
                if (f.waits != 1 or f.releases != 1) return false;
            }
            return true;
        }

        fn addFlight(s: *State, slot: u64, fail: bool, write_token: ?u32) CommandBuffer {
            const idx = s.flight_count;
            s.flights[idx] = .{ .fail = fail, .slot = slot, .write_token = write_token };
            s.flight_count += 1;
            return .{ .state = s, .index = idx };
        }

        fn freeCache(s: *State) void {
            if (!s.allRetired()) s.early_cache_free = true;
            s.cache_freed += 1;
        }

        fn expectAllClean(s: *const State) !void {
            try std.testing.expect(!s.early_ring_free);
            try std.testing.expect(!s.early_cache_free);
            for (s.flights[0..s.flight_count]) |f| {
                try std.testing.expectEqual(@as(u32, 1), f.waits);
                try std.testing.expectEqual(@as(u32, 1), f.releases);
                try std.testing.expect(!f.wait_after_release);
                try std.testing.expect(!f.release_before_wait);
                try std.testing.expect(!f.acted_after_free);
            }
        }
    };

    pub const Device = struct {
        state: *State,

        pub fn buffer(d: Device, _: usize, _: u32) !Buffer {
            const kind: Buffer.Kind = if (d.state.alloc_count == 0) .tokens else .meta;
            d.state.alloc_count += 1;
            return .{ .state = d.state, .kind = kind };
        }
    };

    pub const Buffer = struct {
        const Kind = enum { tokens, meta };
        state: *State,
        kind: Kind,

        pub fn slice(b: Buffer, comptime T: type, len: usize) []T {
            if (T == u32) return b.state.tokens[0..len];
            if (T == flight.FlightMeta) return b.state.meta[0..len];
            @compileError("unexpected ring buffer element type");
        }

        pub fn deinit(b: Buffer) void {
            if (!b.state.allRetired()) b.state.early_ring_free = true;
            switch (b.kind) {
                .tokens => b.state.tokens_freed += 1,
                .meta => b.state.meta_freed += 1,
            }
        }
    };

    pub const CommandBuffer = struct {
        state: *State,
        index: usize,

        pub fn wait(cb: CommandBuffer) void {
            const f = &cb.state.flights[cb.index];
            if (f.releases > 0) f.wait_after_release = true;
            if (cb.state.tokens_freed > 0 or cb.state.meta_freed > 0 or cb.state.cache_freed > 0) f.acted_after_free = true;
            f.waits += 1;
            if (!f.fail) if (f.write_token) |tok| {
                cb.state.tokens[@intCast(f.slot % flight.ring)] = tok;
            };
        }

        pub fn failure(cb: CommandBuffer) ?[*:0]const u8 {
            return if (cb.state.flights[cb.index].fail) "injected gpu failure" else null;
        }

        pub fn release(cb: CommandBuffer) void {
            const f = &cb.state.flights[cb.index];
            if (f.waits == 0) f.release_before_wait = true;
            if (cb.state.tokens_freed > 0 or cb.state.meta_freed > 0 or cb.state.cache_freed > 0) f.acted_after_free = true;
            f.releases += 1;
        }
    };
};

const FakeRing = flight.RingOf(FakeMtl);

fn enqueueFakeFlight(r: *FakeRing, state: *FakeMtl.State, fail: bool, write_token: ?u32) !u64 {
    try r.prepare(1);
    const h = try r.take();
    const cb = state.addFlight(h, fail, write_token);
    r.flights.appendAssumeCapacity(.{ .end = h + 1, .cb = cb });
    return h;
}

test "failed flight followed by successful flight latches sticky failure across reads, allocations, and drains" {
    const vocab: usize = 1024;
    var state: FakeMtl.State = .{};
    state.tokens[0] = 42;
    state.tokens[1] = 99;
    var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });

    const h0 = try enqueueFakeFlight(&r, &state, true, null);
    const h1 = try enqueueFakeFlight(&r, &state, false, 77);
    try std.testing.expectEqual(@as(u64, 0), h0);
    try std.testing.expectEqual(@as(u64, 1), h1);

    try std.testing.expectError(error.GpuFailed, r.read(h0, vocab));
    try std.testing.expect(r.failed);
    try std.testing.expectEqual(@as(u64, 0), r.landed);
    try std.testing.expectEqual(@as(u32, 1), state.flights[0].waits);
    try std.testing.expectEqual(@as(u32, 1), state.flights[0].releases);
    try std.testing.expectEqual(@as(u32, 0), state.flights[1].waits);

    // Repeated reads on failed and trailing handles keep failing and never expose stale ring data.
    try std.testing.expectError(error.GpuFailed, r.read(h0, vocab));
    try std.testing.expectError(error.GpuFailed, r.read(h0, vocab));
    try std.testing.expectError(error.GpuFailed, r.read(h1, vocab));
    try std.testing.expectEqual(@as(u64, 0), r.landed);

    // Later allocations fail without mutating `next` or `landed`.
    try std.testing.expectError(error.GpuFailed, r.take());
    try std.testing.expectError(error.GpuFailed, r.prepare(1));
    try std.testing.expectError(error.GpuFailed, r.takeWrite(5));
    try std.testing.expectError(error.GpuFailed, r.land());
    try std.testing.expectEqual(@as(u64, 2), r.next);
    try std.testing.expectEqual(@as(u64, 0), r.landed);

    // Drain waits for and releases the trailing successful flight without landing failed output.
    try std.testing.expectError(error.GpuFailed, r.drain());
    try std.testing.expectEqual(@as(usize, 0), r.flights.items.len);
    try std.testing.expectEqual(@as(u64, 0), r.landed);
    try std.testing.expectEqual(@as(u32, 1), state.flights[1].waits);
    try std.testing.expectEqual(@as(u32, 1), state.flights[1].releases);

    // Repeated drains, reads, puts, and sync draws stay failed.
    try std.testing.expectError(error.GpuFailed, r.drain());
    try std.testing.expectError(error.GpuFailed, r.read(h0, vocab));
    try std.testing.expectError(error.GpuFailed, r.read(h1, vocab));
    try std.testing.expectError(error.GpuFailed, r.put(5));
    var sync_calls: usize = 0;
    try std.testing.expectError(error.GpuFailed, r.drawSync(&sync_calls, struct {
        fn call(n: *usize, _: u64) !void {
            n.* += 1;
        }
    }.call));
    try std.testing.expectEqual(@as(usize, 0), sync_calls);
    try std.testing.expectEqual(@as(u64, 0), r.landed);

    r.deinit();
    try state.expectAllClean();
    try std.testing.expectEqual(@as(u32, 1), state.tokens_freed);
    try std.testing.expectEqual(@as(u32, 1), state.meta_freed);
}

test "successful flight before failed flight stays landed while failed and trailing flights never land" {
    const vocab: usize = 1024;
    var state: FakeMtl.State = .{};
    state.tokens[1] = 42;
    var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });

    const h0 = try enqueueFakeFlight(&r, &state, false, 11);
    const h1 = try enqueueFakeFlight(&r, &state, true, null);
    const h2 = try enqueueFakeFlight(&r, &state, false, 33);

    try std.testing.expectEqual(@as(u32, 11), try r.read(h0, vocab));
    try std.testing.expectEqual(@as(u64, 1), r.landed);

    try std.testing.expectError(error.GpuFailed, r.read(h1, vocab));
    try std.testing.expectEqual(@as(u64, 1), r.landed);
    try std.testing.expectError(error.GpuFailed, r.read(h1, vocab));
    try std.testing.expectError(error.GpuFailed, r.read(h0, vocab));
    try std.testing.expectError(error.GpuFailed, r.read(h2, vocab));

    try std.testing.expectError(error.GpuFailed, r.drain());
    try std.testing.expectEqual(@as(u64, 1), r.landed);
    try std.testing.expectEqual(@as(usize, 0), r.flights.items.len);

    r.deinit();
    try state.expectAllClean();
}

test "release and deinit wait for and release every outstanding flight once before freeing cache and ring" {
    // Direct release then deinit with failed flight followed by successful flight.
    {
        var state: FakeMtl.State = .{};
        var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });
        _ = try enqueueFakeFlight(&r, &state, true, null);
        _ = try enqueueFakeFlight(&r, &state, false, 77);

        // Mirrors Metal.releaseFn: drain flights before freeing the stream's cache.
        try std.testing.expectError(error.GpuFailed, r.drain());
        try std.testing.expectEqual(@as(u64, 0), r.landed);
        state.freeCache();

        // Mirrors Metal.deinit: ring deinit must not double-wait or double-release.
        r.deinit();
        try state.expectAllClean();
        try std.testing.expectEqual(@as(u32, 1), state.cache_freed);
        try std.testing.expectEqual(@as(u32, 1), state.tokens_freed);
        try std.testing.expectEqual(@as(u32, 1), state.meta_freed);
    }

    // Failed read leaving a trailing flight outstanding, then release, then deinit.
    {
        var state: FakeMtl.State = .{};
        var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });
        const h0 = try enqueueFakeFlight(&r, &state, true, null);
        _ = try enqueueFakeFlight(&r, &state, false, 88);

        try std.testing.expectError(error.GpuFailed, r.read(h0, 1024));
        try std.testing.expectEqual(@as(u32, 0), state.flights[1].waits);

        try std.testing.expectError(error.GpuFailed, r.drain());
        try std.testing.expectEqual(@as(u64, 0), r.landed);
        state.freeCache();
        r.deinit();
        try state.expectAllClean();
    }

    // Direct deinit with multiple outstanding flights including failed and successful ones.
    {
        var state: FakeMtl.State = .{};
        var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });
        _ = try enqueueFakeFlight(&r, &state, true, null);
        _ = try enqueueFakeFlight(&r, &state, false, 55);
        _ = try enqueueFakeFlight(&r, &state, true, null);

        // Mirrors Metal.deinit: ring.deinit() finishes all flights before freeing ring and caches.
        r.deinit();
        state.freeCache();
        try std.testing.expectEqual(@as(u64, 0), r.landed);
        try std.testing.expect(r.failed);
        try state.expectAllClean();
    }
}

test "synchronous first-token failure leaves no readable or landed handle and latches sticky failure" {
    const vocab: usize = 1024;
    const FailSync = struct {
        calls: usize = 0,
        fn run(s: *@This(), _: u64) !void {
            s.calls += 1;
            return error.GpuFailed;
        }
    };

    // Fresh ring with stale token in slot 0.
    {
        var state: FakeMtl.State = .{};
        state.tokens[0] = 42;
        var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });
        defer r.deinit();

        var sync: FailSync = .{};
        try std.testing.expectError(error.GpuFailed, r.drawSync(&sync, FailSync.run));
        try std.testing.expectEqual(@as(usize, 1), sync.calls);
        try std.testing.expect(r.failed);
        try std.testing.expectEqual(@as(u64, 0), r.next);
        try std.testing.expectEqual(@as(u64, 0), r.landed);

        try std.testing.expectError(error.GpuFailed, r.read(0, vocab));
        try std.testing.expectError(error.GpuFailed, r.read(0, vocab));
        try std.testing.expectError(error.GpuFailed, r.take());
        try std.testing.expectError(error.GpuFailed, r.put(7));
        try std.testing.expectError(error.GpuFailed, r.drain());
        try std.testing.expectError(error.GpuFailed, r.drawSync(&sync, FailSync.run));
        try std.testing.expectEqual(@as(usize, 1), sync.calls);
        try std.testing.expectEqual(@as(u64, 0), r.next);
        try std.testing.expectEqual(@as(u64, 0), r.landed);
    }

    // Ring with a previously landed token, then failing synchronous draw into slot 1.
    {
        var state: FakeMtl.State = .{};
        state.tokens[1] = 55;
        var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });
        defer r.deinit();

        const h0 = try r.put(17);
        try std.testing.expectEqual(@as(u32, 17), try r.read(h0, vocab));
        var sync: FailSync = .{};
        try std.testing.expectError(error.GpuFailed, r.drawSync(&sync, FailSync.run));
        try std.testing.expect(r.failed);
        try std.testing.expectEqual(@as(u64, 1), r.next);
        try std.testing.expectEqual(@as(u64, 1), r.landed);

        try std.testing.expectError(error.GpuFailed, r.read(1, vocab));
        try std.testing.expectError(error.GpuFailed, r.read(1, vocab));
        try std.testing.expectError(error.GpuFailed, r.read(h0, vocab));
        try std.testing.expectError(error.GpuFailed, r.take());
        try std.testing.expectError(error.GpuFailed, r.drain());
        try std.testing.expectEqual(@as(u64, 1), r.landed);
    }
}

test "allocation backpressure landing a failed flight latches failure and deinit retires all remaining flights" {
    var state: FakeMtl.State = .{};
    var r = try FakeRing.init(std.testing.allocator, .{ .state = &state });
    for (0..flight.max_flights) |i| {
        _ = try enqueueFakeFlight(&r, &state, i == 0, @intCast(i + 10));
    }
    try std.testing.expectError(error.GpuFailed, r.take());
    try std.testing.expect(r.failed);
    try std.testing.expectEqual(@as(u64, flight.max_flights), r.next);
    try std.testing.expectEqual(@as(u64, 0), r.landed);
    try std.testing.expectError(error.GpuFailed, r.take());
    try std.testing.expectError(error.GpuFailed, r.read(0, 1024));
    r.deinit();
    try std.testing.expectEqual(@as(u64, 0), r.landed);
    try state.expectAllClean();
}
