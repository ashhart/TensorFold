//! Both GLM adapters use the shared affordable-cut planner and the actual prompt-sized state copies.
const std = @import("std");
const glm = @import("glm");
const cuda = @import("cuda");
const pc = @import("engine_api").prompt_cache;
const adapter = @import("copies");
const rec = @import("test_support/glm_snap_driver.zig").Driver;
const Fixture = @import("test_support/glm_snap_fixture.zig").Fixture;
const t = std.testing;

test "GLM budget fallback: both model types retain an affordable prefix in the real reusable snapshot pool" {
    inline for (.{ glm.native, glm.native_dsa }) |Family| {
        var d = rec.init();
        var f: Fixture = undefined;
        try f.initFormat(&d, 0, 2, true, true);
        defer f.deinit();
        const budget = f.e.stateBytes(8192);
        try t.expect(f.e.stateBytes(12288) > budget);
        _ = try Family.snap_ready(&f.e, budget);
        const copies: pc.Snapshots = .{ .ptr = &f.e, .vtable = &adapter.Snaps(Family, null).vtable };
        var store = pc.Store.init(t.allocator, copies, adapter.passRules(Family, Family.prompt_rows), budget);
        defer store.deinit();
        const ids = try t.allocator.alloc(u32, 12719);
        defer t.allocator.free(ids);
        @memset(ids, 7);
        var expected: u32 = 12288;
        while (f.e.stateBytes(expected) > budget) expected -= 2048;
        const p = try store.lookup(t.allocator, ids, 12705, &.{}, &.{}, &.{});
        defer t.allocator.free(p.marks);
        try t.expectEqualSlices(u32, &.{expected}, p.marks);
        try f.mark(expected, expected, 17);
        const before = try t.allocator.dupe(u8, rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len));
        defer t.allocator.free(before);
        try t.expect(store.keep(ids, expected, null, &.{}, &.{}));
        const hit = try store.lookup(t.allocator, ids, 12705, &.{}, &.{}, &.{});
        defer t.allocator.free(hit.marks);
        const entry = hit.entry orelse return error.ExpectedCacheHit;
        defer store.release(entry);
        const kept: *glm.engine.Kept = @ptrCast(@alignCast(entry.saved));
        @memset(rec.bytes(f.seq.st.arena.buf.ptr, f.seq.st.arena.buf.len), 0xED);
        try f.e.restoreState(kept.id);
        try f.expectPrefix(before, expected, expected);
        try t.expectEqual(@as(usize, expected), f.seq.st.pos);
        try t.expect(store.held + Family.snap_spare(&f.e) <= budget);
        std.debug.print("GLM fallback {s}: budget={d}, kept={d}, bytes={d}\n", .{ Family.model_type, budget, expected, entry.bytes });
    }
}

// Virtual addresses only: production layout arithmetic at full dimensions, with no GPU or model weights.
pub const Virtual = struct {
    var next: u64 = 1 << 40;
    pub fn alloc(out: *u64, n: usize) callconv(.c) cuda.abi.Result {
        out.* = next;
        next += std.mem.alignForward(u64, n, 256);
        return 0;
    }
    pub fn free(_: u64) callconv(.c) cuda.abi.Result {
        return 0;
    }
    pub fn zero(_: u64, _: u8, _: usize) callconv(.c) cuda.abi.Result {
        return 0;
    }
};

test "GLM budget fallback: actual full-shape CUDA sizes and greatest 300K cold cuts" {
    var d: cuda.Driver = undefined;
    d.api.cuMemAlloc_v2 = Virtual.alloc;
    d.api.cuMemFree_v2 = Virtual.free;
    d.api.cuMemsetD8_v2 = Virtual.zero;
    const full_json = try glm.config_dsa.dsaConfig(t.allocator, 78, 3, 256, 154880);
    defer t.allocator.free(full_json);
    const full = try glm.config_dsa.parse(t.allocator, full_json);
    const configs = [_]glm.config.Config{ .{}, .{}, full };
    for (configs, 0..) |config, mode| {
        var w: glm.weights.Weights = undefined;
        w.config = config;
        w.share = try glm.shard.Share.of(&w.config, 0, 2);
        var layers: [glm.config.max_layers]glm.weights.Layer = undefined;
        w.layers = layers[0..w.config.layers];
        for (w.layers, 0..) |*l, i| l.mixer = switch (w.config.kind(@intCast(i))) {
            .mla => .{ .mla = undefined },
            .kda => .{ .kda = undefined },
        };
        var e: glm.engine.Engine = undefined;
        var seq: @TypeOf(e.cur.*) = .{ .st = try glm.forward.State.initFormat(t.allocator, &d, &w, glm.native.capacity(524288), mode == 1) };
        defer seq.st.deinit();
        e.cur = &seq;
        e.local = &.{};
        e.c = w.config;
        e.mtp = true;
        e.capacity = seq.st.capacity;
        e.share_pages = false;
        const raw0 = try glm.snapshots.layout.Plan.size(&seq.st, &e.c, e.capacity, 65536, 65600, 0);
        const raw1 = try glm.snapshots.layout.Plan.size(&seq.st, &e.c, e.capacity, 67584, 67648, 0);
        const per_token = (raw1 - raw0) / 2048;
        const name: []const u8 = if (mode == 2) "glm_moe_dsa/bf16" else if (mode == 1) "glm5_next/fp8" else "glm5_next/bf16";
        try t.expectEqual(@as(usize, switch (mode) {
            0 => 13056,
            1 => 6960,
            else => 96640,
        }), per_token);
        for ([_]u64{ 4 << 30, 8 << 30, 16 << 30 }) |budget| {
            var at: u32 = 300000 / 2048 * 2048;
            while (at > 0 and glm.native.snap_bytes(&e, at) > budget) at -= 2048;
            try t.expect(at > 0);
            try t.expect(glm.native.snap_bytes(&e, at) <= budget);
            if (at + 2048 < 300000) try t.expect(glm.native.snap_bytes(&e, at + 2048) > budget);
            std.debug.print("GLM size {s} rank=0/world=2: growth={d} B/token, state300K={d}, budget={d}, cut={d}, cut_bytes={d}\n", .{ name, per_token, glm.native.snap_bytes(&e, 300000), budget, at, glm.native.snap_bytes(&e, at) });
        }
    }
}
