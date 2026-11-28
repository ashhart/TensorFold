//! Eight-layer GB10 GLM state-path check: cold, unshared resume and two shared-page resumes at the same starts.
const std = @import("std");
const cuda = @import("cuda");
const glm = @import("glm_cuda");
const check = @import("check.zig");
const Gpu = check.Gpu;
pub const capacity = 8256;
pub const layer_count = 8;
pub const memory_ceiling: u64 = 48 << 30;

pub fn checkSettings(c: glm.config.Config, original: glm.ranks.Settings) !glm.ranks.Settings {
    if (original.layers != layer_count) return error.SharedStartsNeedsEightLayers;
    if (c.dsa or c.exl3 or c.layers < layer_count) return error.SharedStartsPrefixCoverage;
    var mla: usize = 0;
    var dense: usize = 0;
    for (0..layer_count) |i| {
        mla += @intFromBool(c.kind(@intCast(i)) == .mla);
        dense += @intFromBool(!c.isMoe(@intCast(i)));
    }
    if (mla < 2 or layer_count - mla < 2 or dense == 0 or dense == layer_count) return error.SharedStartsPrefixCoverage;
    var set = original;
    set.pages = true;
    return set;
}

pub const Budget = struct {
    previous: ?u64,
    effective: u64,
    pub fn begin() !Budget {
        const previous = try cuda.memory.setAllocationLimit(memory_ceiling);
        errdefer _ = cuda.memory.setAllocationLimit(previous) catch {};
        const effective = @min(previous orelse memory_ceiling, memory_ceiling);
        _ = try cuda.memory.setAllocationLimit(effective);
        return .{ .previous = previous, .effective = effective };
    }
    pub fn close(b: Budget) void {
        _ = cuda.memory.setAllocationLimit(b.previous) catch {};
    }
};

fn memoryReceipt(phase: []const u8, limit: u64) void {
    const u = cuda.memory.usage(false);
    std.debug.print("MEMORY shared-starts phase={s} scope=layers0..7 device={d} pinned={d} device_peak={d} pinned_peak={d} tracked_limit={d}\n", .{ phase, u.device, u.host, u.peak, u.host_peak, limit });
}

const Save = struct {
    e: *glm.engine.Engine,
    at: u32,
    id: ?u64 = null,
    failed: bool = false,
    fn call(ctx: *anyopaque, at: u32) void {
        const s: *Save = @ptrCast(@alignCast(ctx));
        if (at != s.at) return;
        s.id = s.e.keepState() catch {
            s.failed = true;
            return;
        };
    }
};

/// Compact snapshot bytes in logical cache order, including the host metadata but excluding the device table.
fn stateHash(e: *glm.engine.Engine) ![32]u8 {
    const st = &e.cur.st;
    const p = try glm.snapshots.layout.Plan.init(st, &e.c, st.pos, null, 0);
    var hash = std.crypto.hash.sha2.Sha256.init(.{});
    hash.update(std.mem.asBytes(&[_]usize{ st.pos, st.mtp_len, st.base, st.mtp_base }));
    hash.update(std.mem.sliceAsBytes(st.cur));
    for (p.parts[0..p.count]) |part| {
        var done: usize = 0;
        while (done < part.len) {
            const piece = p.piece(part, done);
            const bytes = try e.gpa.alloc(u8, piece.len);
            defer e.gpa.free(bytes);
            try cuda.memory.download(e.d, piece.live, bytes);
            hash.update(bytes);
            done += piece.len;
        }
    }
    return hash.finalResult();
}

/// Clear only this slot's private bytes before begin remaps its table; shared pages are never destinations.
fn clear(e: *glm.engine.Engine, kept: usize) !void {
    const st = &e.cur.st;
    var fixed = st.arena.buf;
    if (st.table != 0) fixed.len = @intCast(st.table - fixed.ptr);
    try fixed.fill8(0, e.stream.handle);
    if (e.cur.st.table == 0) return;
    const p = e.pages.?;
    for (e.cur.st.pages[kept..]) |id| {
        const page: cuda.DeviceBuffer = .{ .d = e.d, .ptr = p.arena.?.ptr + id * p.page, .len = p.page, .borrowed = true };
        try page.fill8(0, e.stream.handle);
    }
}

fn prefixHash(e: *glm.engine.Engine, ids: []const u32) ![32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    const p = e.pages.?;
    const bytes = try e.gpa.alloc(u8, p.page);
    defer e.gpa.free(bytes);
    for (ids) |id| {
        try cuda.memory.download(e.d, p.arena.?.ptr + id * p.page, bytes);
        h.update(bytes);
    }
    return h.finalResult();
}

/// `glm-shared-starts-check CKPT` with TF_GLM_LAYERS=8: one GB10, eight-layer equality and two shared resumed slots.
pub fn sharedCheck(gpu: Gpu, checkpoint: []const u8) !void {
    try check.expect(std.mem.eql(u8, @import("core").profile.lookup("TF_GLM_REQUEST_STARTS") orelse "", "1"), "set TF_GLM_REQUEST_STARTS=1", .{});
    try check.expect(try @import("core").kv_prompts.sharing() > 0, "set TF_KV_SHARE=8", .{});
    var arena = std.heap.ArenaAllocator.init(gpu.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var problem: []const u8 = "";
    const path = try std.fs.path.join(a, &.{ checkpoint, "config.json" });
    const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, a, .limited(1 << 24));
    const config = try glm.config_dsa.of(a, text);
    const set = checkSettings(config, try glm.ranks.Settings.read(&glm.ranks.environment, &problem)) catch |err| {
        std.debug.print("shared-starts is an eight-layer state-path gate: set TF_GLM_LAYERS=8; full-checkpoint TP=2 qualification is separate\n", .{});
        return err;
    };
    const budget_limit = try Budget.begin();
    defer budget_limit.close();
    _ = cuda.memory.usage(true);
    defer memoryReceipt("cleanup", budget_limit.effective);
    std.debug.print("SCOPE shared-starts loaded_layers={d} checkpoint_layers={d} rank=0 world=1 mtp=false full_model=false\n", .{ set.layers, config.layers });
    var lib = try cuda.nccl.Library.open();
    defer lib.close();
    const comm = try @import("glm_lanes_tests.zig").openComm(gpu, &lib, 0, 1, "");
    defer _ = lib.api.ncclCommDestroy(comm);
    const e = try glm.engine.Engine.init(gpu.gpa, gpu.io, gpu.d, gpu.ctx, checkpoint, &lib, comm, 0, 1, .{ .capacity = capacity, .mtp = false, .settings = set });
    defer e.deinit();
    try check.expect(!e.c.dsa, "use GLM-5.3-Flash NVFP4", .{});
    try check.expect(e.w.layers.len == layer_count, "check did not load exactly eight layers", .{});
    var weight_bytes: u64 = 0;
    for (e.w.buffers.items) |buffer| weight_bytes += buffer.len;
    std.debug.print("WEIGHTS shared-starts device_resident_bytes={d} embedding_and_head=full selected_layers=0..7\n", .{weight_bytes});
    memoryReceipt("loaded", budget_limit.effective);
    var link = try glm.follow.Link.init(gpu.gpa, gpu.d, &lib, comm, e.stream, capacity);
    defer link.deinit();
    const x: glm.follow.Leader = .{ .e = e, .link = &link };
    defer x.end() catch {};
    const l = try glm.pages.layout(gpu.gpa, e, 0);
    const count = 3 * glm.pages.tableLen(capacity) + 8;
    try x.pool(3, count, l.pageBytes());
    e.pages = try glm.pages.Pages.init(gpu.gpa, e, l, 3, count, true);
    try check.expect(e.pages.?.arena != null, "this check uses device KV pages; disable host-KV modes", .{});
    const budget = 4 * e.stateBytes(capacity - 64);
    e.share_pages = true;
    try x.snapshots(budget);
    std.debug.print("STATE shared-starts page_bytes={d} pool_pages={d} snapshot_budget={d} slot_fixed={d}\n", .{ l.pageBytes(), count, budget, try glm.forward.State.pagedBytes(gpu.gpa, &e.w, capacity, glm.pages.tableLen(capacity)) });
    memoryReceipt("ready", budget_limit.effective);
    const Case = struct { at: u32, cuts: []const u32 };
    const cases = [_]Case{
        .{ .at = 511, .cuts = &.{511} },
        .{ .at = 512, .cuts = &.{512} },
        .{ .at = 513, .cuts = &.{513} },
        .{ .at = 805, .cuts = &.{805} },
        .{ .at = 2853, .cuts = &.{ 805, 2853 } },
    };
    const V = e.share.vocab.len();
    const logits = try a.alloc(u16, V);
    const reference = try a.alloc(u16, V);
    const decode_reference = try a.alloc(u16, 8 * V);
    const prompt = try a.alloc(u32, 4001);
    for (prompt, 0..) |*v, i| v.* = @intCast(100 + (i * 7919) % 5000);
    for (cases) |cs| {
        var full: ?u64 = null;
        var shared: ?u64 = null;
        defer {
            if (full) |id| e.dropState(id);
            if (shared) |id| e.dropState(id);
        }
        var want_token: u32 = 0;
        var want_state: [32]u8 = undefined;
        var want_final: [32]u8 = undefined;
        var want_next: [8]u32 = undefined;
        var prefix: [32]u8 = undefined;
        for (0..5) |mode| {
            const slot: u32 = if (mode < 2) 3 else @intCast(mode - 2);
            try x.bind(slot);
            var adopted: std.ArrayList(u32) = .empty;
            defer adopted.deinit(gpu.gpa);
            if (mode >= 3) try glm.snapshots.refs.adopt(e, shared.?, &adopted);
            const ids = if (mode >= 2) try e.pages.?.openWith(slot, prompt.len, 64, adopted.items) else &.{};
            try clear(e, adopted.items.len);
            try x.beginOn(null, ids, adopted.items.len);
            try x.cuts(cs.cuts);
            var save: Save = .{ .e = e, .at = cs.at };
            const marks: ?glm.engine.Marks = if (mode == 0 or mode == 2) .{ .at = &.{cs.at}, .ctx = &save, .call = Save.call } else null;
            const from: ?glm.engine.Resume = if (mode == 1 or mode >= 3) .{ .id = if (mode == 1) full.? else shared.?, .at = cs.at } else null;
            const started = check.now(gpu.io);
            const token = try x.prefill(prompt, from, null, marks);
            try e.stream.synchronize();
            const elapsed = check.now(gpu.io) - started;
            try cuda.memory.download(gpu.d, e.bp.logits, std.mem.sliceAsBytes(logits));
            const state = try stateHash(e);
            if (mode == 0) {
                full = save.id;
                want_token = token;
                @memcpy(reference, logits);
                want_state = state;
            } else {
                try check.expect(token == want_token, "cut={d} mode={d} token differs", .{ cs.at, mode });
                try check.sameBytes("request-start logits", std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(reference));
                try check.sameBytes("request-start committed state", &state, &want_state);
            }
            if (mode == 2) {
                shared = save.id;
                const snap = e.snaps.get(shared orelse return error.NoKeptState).?;
                prefix = try prefixHash(e, snap.shared_ids);
            }
            var next = token;
            for (0..8) |round| {
                var out: [1]u32 = undefined;
                try x.verify(&.{next}, &out, 0);
                try cuda.memory.download(gpu.d, e.bd.logits, std.mem.sliceAsBytes(logits));
                try x.keep(1);
                if (mode == 0) {
                    want_next[round] = out[0];
                    @memcpy(decode_reference[round * V ..][0..V], logits);
                } else {
                    try check.expect(out[0] == want_next[round], "cut={d} mode={d} decode={d} token differs", .{ cs.at, mode, round });
                    try check.sameBytes("request-start decode logits", std.mem.sliceAsBytes(logits), std.mem.sliceAsBytes(decode_reference[round * V ..][0..V]));
                }
                next = out[0];
            }
            const final = try stateHash(e);
            if (mode == 0) want_final = final else try check.sameBytes("request-start final committed state", &final, &want_final);
            if (mode >= 3) try check.sameBytes("other stream's complete prefix pages", &(try prefixHash(e, e.snaps.get(shared.?).?.shared_ids)), &prefix);
            try check.expect(!save.failed and (marks == null or save.id != null), "cut={d} snapshot was not saved", .{cs.at});
            check.pass("shared-starts cut={d} mode={d} token={d} state={x} prefill_ns={d}", .{ cs.at, mode, token, state, elapsed });
            memoryReceipt("mode", budget_limit.effective);
            if (mode < 2) try x.release(slot);
        }
        for (0..3) |slot| try x.release(@intCast(slot));
    }
}

test "GLM shared starts plan: a full-model or incomplete prefix cannot masquerade as a one-GB10 gate" {
    const t = std.testing;
    const c: glm.config.Config = .{};
    try t.expectError(error.SharedStartsNeedsEightLayers, checkSettings(c, .{}));
    for ([_]u32{ 1, 3, 4, 7, 9, 45 }) |n| try t.expectError(error.SharedStartsNeedsEightLayers, checkSettings(c, .{ .layers = n }));
    const plan = try checkSettings(c, .{ .layers = 8 });
    try t.expect(plan.pages and plan.layers == 8);
    var bad = c;
    bad.mla = .empty;
    try t.expectError(error.SharedStartsPrefixCoverage, checkSettings(bad, .{ .layers = 8 }));
    bad = c;
    bad.mla.setRangeValue(.{ .start = 0, .end = 8 }, true);
    try t.expectError(error.SharedStartsPrefixCoverage, checkSettings(bad, .{ .layers = 8 }));
    bad = c;
    bad.dense_layers = 8;
    try t.expectError(error.SharedStartsPrefixCoverage, checkSettings(bad, .{ .layers = 8 }));
    bad = c;
    bad.dense_layers = 0;
    try t.expectError(error.SharedStartsPrefixCoverage, checkSettings(bad, .{ .layers = 8 }));
    bad = c;
    bad.dsa = true;
    try t.expectError(error.SharedStartsPrefixCoverage, checkSettings(bad, .{ .layers = 8 }));
    bad = c;
    bad.layers = 7;
    try t.expectError(error.SharedStartsPrefixCoverage, checkSettings(bad, .{ .layers = 8 }));
    bad = c;
    bad.exl3 = true;
    try t.expectError(error.SharedStartsPrefixCoverage, checkSettings(bad, .{ .layers = 8 }));
}

test "GLM shared starts plan: tracked memory ceiling preserves a lower owner limit and restores it" {
    const t = std.testing;
    const old = try cuda.memory.setAllocationLimit(null);
    defer _ = cuda.memory.setAllocationLimit(old) catch {};
    {
        const b = try Budget.begin();
        defer b.close();
        try t.expect(b.previous == null and b.effective == 48 * (1 << 30));
    }
    _ = try cuda.memory.setAllocationLimit(1 << 30);
    {
        const b = try Budget.begin();
        defer b.close();
        try t.expect(b.previous.? == 1 << 30 and b.effective == 1 << 30);
    }
    try t.expectEqual(@as(?u64, 1 << 30), try cuda.memory.setAllocationLimit(null));
}
