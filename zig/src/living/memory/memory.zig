//! Folded Memory Weights v3: fast, drift-free, scalable Living Weights for hybrid & attention models.
pub const quant = @import("quant.zig");
pub const whitener = @import("whitener.zig");
pub const gate = @import("gate.zig");
pub const chunker = @import("chunker.zig");
pub const store = @import("store.zig");
pub const bank = @import("bank.zig");
pub const cuda_bridge = @import("cuda_bridge.zig");

pub const Bank = bank.Bank;
pub const Backend = bank.Backend;
pub const ArchSpec = bank.ArchSpec;
pub const RawState = bank.RawState;
pub const SnapHandle = bank.SnapHandle;
pub const TeachChunkInput = bank.TeachChunkInput;
pub const Family = bank.Family;
pub const CudaBridge = cuda_bridge.CudaBridge;

test {
    _ = quant;
    _ = whitener;
    _ = gate;
    _ = chunker;
    _ = store;
    _ = cuda_bridge;
    _ = @import("safety_test.zig");
    _ = @import("journal.zig");
    _ = @import("durable_test.zig");
    _ = @import("exact_test.zig");
    _ = @import("validate_test.zig");
    _ = @import("lesson_test.zig");
    _ = @import("replay_test.zig");
    _ = @import("transaction_test.zig");
}

test "memory bank: end-to-end teach, route, compose, safetensors save/reload (zero plaintext), and forget" {
    const std = @import("std");
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    const MockSnap = struct { len: u32 };
    const Mock = struct {
        gpa: std.mem.Allocator,
        arch: ArchSpec,
        snaps_alive: usize = 0,

        fn extractHidden(ctx: *anyopaque, ids: []const u32, _: usize, _: usize, out_lex: []f32, out_sem: []f32) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const d = self.arch.hidden;
            @memset(out_lex, 0);
            @memset(out_sem, 0);
            for (ids, 0..) |id, t| {
                // Background shared dims 0..2 + deterministic token-specific orthogonal direction
                out_lex[t * d + 0] = 4.0;
                out_lex[t * d + 1] = -3.0;
                out_lex[t * d + (4 + (id % (d - 4)))] = 5.0;
                out_sem[t * d + 0] = 3.0;
                out_sem[t * d + 2] = -2.0;
                out_sem[t * d + (4 + (id % (d - 4)))] = 5.0;
            }
        }

        fn compileState(ctx: *anyopaque, base: ?*const RawState, ids: []const u32, out: *RawState) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const off: usize = if (base) |b| b.len else 0;
            for (0..self.arch.num_attn) |a| {
                for (0..self.arch.kv_heads) |h| {
                    for (0..ids.len) |t| {
                        const base_i = (h * ids.len + t) * self.arch.head_dim;
                        for (0..self.arch.head_dim) |c| {
                            out.kv_k[a][base_i + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[t] + c) % 17)) * 0.1);
                            out.kv_v[a][base_i + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[t] + c + 3) % 19)) * 0.1);
                        }
                    }
                }
                quant.ropeRotateKvBf16(out.kv_k[a], self.arch.kv_heads, ids.len, self.arch.head_dim, off, self.arch.rotary_dim, self.arch.rope_theta, false);
            }
            for (0..self.arch.num_rec) |r| {
                if (base) |b| @memcpy(out.rec_state[r], b.rec_state[r]);
                for (out.rec_state[r], 0..) |*v, idx| v.* += @as(f32, @floatFromInt((ids[0] + idx) % 7)) * 0.05;
                for (out.rec_conv[r], 0..) |*v, idx| v.* = quant.f32ToBf16(@as(f32, @floatFromInt((ids[0] + idx) % 5)) * 0.1);
            }
        }

        fn buildSnap(ctx: *anyopaque, composed: *const RawState, suffix_ids: []const u32) !SnapHandle {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const s = try self.gpa.create(MockSnap);
            s.* = .{ .len = composed.len + @as(u32, @intCast(suffix_ids.len)) };
            self.snaps_alive += 1;
            return .{ .saved = s, .at = s.len };
        }

        fn dropSnap(ctx: *anyopaque, ptr: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const s: *MockSnap = @ptrCast(@alignCast(ptr));
            self.gpa.destroy(s);
            self.snaps_alive -= 1;
        }
    };

    const arch = ArchSpec{
        .family = .qwen3_5,
        .hidden = 64,
        .num_layers = 4,
        .layer_lex = 1,
        .layer_sem = 2,
        .num_attn = 2,
        .kv_heads = 2,
        .head_dim = 32,
        .rotary_dim = 16,
        .rope_theta = 10_000_000.0,
        .num_rec = 2,
        .rec_heads = 2,
        .rec_rows = 16,
        .rec_cols = 16,
        .conv_elems = 12,
    };

    var mock = Mock{ .gpa = gpa, .arch = arch };
    const backend = Backend{
        .ctx = &mock,
        .arch = arch,
        .extractHidden = Mock.extractHidden,
        .compileState = Mock.compileState,
        .buildSnap = Mock.buildSnap,
        .dropSnap = Mock.dropSnap,
    };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);

    var b = try Bank.init(gpa, io, dir, backend);
    defer b.deinit();

    // Calibration tokens occupy dims 4..23; novel taught fact occupies dims 30..35; orthogonal query dims 45..49
    const cal_a = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const cal_b = [_]u32{ 6, 7, 8, 9, 10, 11 };
    const cal_c = [_]u32{ 12, 13, 14, 15, 16, 17 };
    const cal_ids = [_][]const u32{ &cal_a, &cal_b, &cal_c };
    const pref_ids = [_]u32{ 1, 2, 3, 4 };
    const suf_ids = [_]u32{ 5, 6 };

    try b.ensureCalibrated(&cal_ids, &pref_ids, &suf_ids);
    const fact_ids = [_]u32{ 30, 31, 32, 33, 34 };
    _ = try b.teachChunk(.{
        .parent_id = 7,
        .body_ids = &fact_ids,
        .gate_ids = &fact_ids,
        .is_small_fact = true,
    });
    const bytes_written = try b.save();
    try std.testing.expect(bytes_written > 0);

    // Reload cleanly into a brand new Bank purely from living-weights.safetensors!
    var b2 = try Bank.init(gpa, io, dir, backend);
    defer b2.deinit();
    try std.testing.expect(try b2.loadIfPresent());
    try std.testing.expectEqual(@as(usize, 1), b2.units.items.len);

    // Matching query triggers and builds Snap of length prefix(4) + unit(5) + suffix(2) = 11
    const q_hit = [_]u32{ 30, 31, 32 };
    const snap = (try b2.routeAndCompose(&q_hit)) orelse return error.ExpectedMemoryHit;
    try std.testing.expectEqual(@as(u32, 11), snap.at);
    backend.dropSnap(backend.ctx, snap.saved);
    try std.testing.expectEqual(@as(usize, 0), mock.snaps_alive);

    // Unrelated query does NOT trigger -> null (zero drift!)
    const q_miss = [_]u32{ 45, 46, 47 };
    try std.testing.expect((try b2.routeAndCompose(&q_miss)) == null);

    // Standing policy unit activates EVEN on unrelated queries, survives save/reload, and reverts on forget!
    const pol_ids = [_]u32{ 50, 51, 52 };
    _ = try b2.teachChunk(.{
        .parent_id = 9,
        .body_ids = &pol_ids,
        .gate_ids = &.{},
        .is_standing = true,
    });
    _ = try b2.save();
    var b3 = try Bank.init(gpa, io, dir, backend);
    defer b3.deinit();
    try std.testing.expect(try b3.loadIfPresent());
    const snap_pol = (try b3.routeAndCompose(&q_miss)) orelse return error.ExpectedStandingHit;
    try std.testing.expectEqual(@as(u32, 4 + 3 + 2), snap_pol.at);
    backend.dropSnap(backend.ctx, snap_pol.saved);
    try std.testing.expectEqual(@as(u32, 1), b3.forgetParent(9));
    try std.testing.expect((try b3.routeAndCompose(&q_miss)) == null);

    // Exact deletion removes unit and deletes living-weights.safetensors
    try std.testing.expectEqual(@as(u32, 1), b3.forgetParent(7));
    try std.testing.expectEqual(@as(u64, 0), try b3.save());
    try std.testing.expect(!b3.hasUnits());
}

test "memory bank(v6): delta journal replay, temporal belief revision + undo restore, sleep-time consolidation, and pure-attention (num_rec=0)" {
    const std = @import("std");
    const gpa = std.testing.allocator;
    const io = std.testing.io;

    const MockSnap = struct { len: u32 };
    const Mock = struct {
        gpa: std.mem.Allocator,
        arch: ArchSpec,

        fn extractHidden(ctx: *anyopaque, ids: []const u32, _: usize, _: usize, out_lex: []f32, out_sem: []f32) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const d = self.arch.hidden;
            @memset(out_lex, 0);
            @memset(out_sem, 0);
            for (ids, 0..) |id, t| {
                out_lex[t * d + 0] = 4.0;
                out_lex[t * d + 1] = -3.0;
                out_lex[t * d + (4 + (id % (d - 4)))] = 5.0;
                out_sem[t * d + 0] = 3.0;
                out_sem[t * d + 2] = -2.0;
                out_sem[t * d + (4 + (id % (d - 4)))] = 5.0;
            }
        }

        fn compileState(ctx: *anyopaque, base: ?*const RawState, ids: []const u32, out: *RawState) !void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const off: usize = if (base) |b| b.len else 0;
            for (0..self.arch.num_attn) |a| {
                for (0..self.arch.kv_heads) |h| {
                    for (0..ids.len) |t| {
                        const base_i = (h * ids.len + t) * self.arch.head_dim;
                        for (0..self.arch.head_dim) |c| {
                            out.kv_k[a][base_i + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[t] + c) % 17)) * 0.1);
                            out.kv_v[a][base_i + c] = quant.f32ToBf16(@as(f32, @floatFromInt((ids[t] + c + 3) % 19)) * 0.1);
                        }
                    }
                }
                quant.ropeRotateKvBf16(out.kv_k[a], self.arch.kv_heads, ids.len, self.arch.head_dim, off, self.arch.rotary_dim, self.arch.rope_theta, false);
            }
            for (0..self.arch.num_rec) |r| {
                if (base) |b| @memcpy(out.rec_state[r], b.rec_state[r]);
                for (out.rec_state[r], 0..) |*v, idx| v.* += @as(f32, @floatFromInt((ids[0] + idx) % 7)) * 0.05;
                for (out.rec_conv[r], 0..) |*v, idx| v.* = quant.f32ToBf16(@as(f32, @floatFromInt((ids[0] + idx) % 5)) * 0.1);
            }
        }

        fn buildSnap(ctx: *anyopaque, composed: *const RawState, suffix_ids: []const u32) !SnapHandle {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const s = try self.gpa.create(MockSnap);
            s.* = .{ .len = composed.len + @as(u32, @intCast(suffix_ids.len)) };
            return .{ .saved = s, .at = s.len };
        }

        fn dropSnap(ctx: *anyopaque, ptr: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            const s: *MockSnap = @ptrCast(@alignCast(ptr));
            self.gpa.destroy(s);
        }
    };

    const cal_a = [_]u32{ 0, 1, 2, 3, 4, 5 };
    const cal_b = [_]u32{ 6, 7, 8, 9, 10, 11 };
    const cal_c = [_]u32{ 12, 13, 14, 15, 16, 17 };
    const cal_ids = [_][]const u32{ &cal_a, &cal_b, &cal_c };
    const pref_ids = [_]u32{ 1, 2, 3, 4 };
    const suf_ids = [_]u32{ 5, 6 };

    // Part A: hybrid bank: save, delta, supersession, reload replay, undo restore, consolidation
    {
        const arch = ArchSpec{
            .family = .qwen3_5,
            .hidden = 64,
            .num_layers = 4,
            .layer_lex = 1,
            .layer_sem = 2,
            .num_attn = 2,
            .kv_heads = 2,
            .head_dim = 32,
            .rotary_dim = 16,
            .rope_theta = 10_000_000.0,
            .num_rec = 2,
            .rec_heads = 2,
            .rec_rows = 16,
            .rec_cols = 16,
            .conv_elems = 12,
        };
        var mock = Mock{ .gpa = gpa, .arch = arch };
        const backend = Backend{ .ctx = &mock, .arch = arch, .extractHidden = Mock.extractHidden, .compileState = Mock.compileState, .buildSnap = Mock.buildSnap, .dropSnap = Mock.dropSnap };

        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);

        var b = try Bank.init(gpa, io, dir, backend);
        defer b.deinit();
        try b.ensureCalibrated(&cal_ids, &pref_ids, &suf_ids);

        // 1. Teach old belief (pid=1, 5 tokens) and write base safetensors
        const old_ids = [_]u32{ 30, 31, 32, 33, 34 };
        _ = try b.teachChunk(.{ .parent_id = 1, .body_ids = &old_ids, .gate_ids = &old_ids, .is_small_fact = true });
        _ = try b.save();
        try std.testing.expect(b.base_saved);
        try std.testing.expectEqual(@as(u32, 0), b.pending_deltas);

        // 2. Teach new belief (pid=2, 7 tokens) on overlapping entities -> conflict detector MUST find pid=1!
        const new_ids = [_]u32{ 30, 31, 32, 35, 36, 37, 38 };
        const conflicts = try b.findConflictParents(&new_ids, 4);
        defer gpa.free(conflicts);
        try std.testing.expectEqual(@as(usize, 1), conflicts.len);
        try std.testing.expectEqual(@as(u32, 1), conflicts[0].parent_id);

        _ = try b.teachChunk(.{ .parent_id = 2, .body_ids = &new_ids, .gate_ids = &new_ids, .is_small_fact = true });
        _ = try b.save(); // appends to living-weights.delta!
        try std.testing.expectEqual(@as(u32, 1), b.pending_deltas);

        try std.testing.expectEqual(@as(u32, 1), b.supersedeParent(1, 2));
        _ = try b.persistTombstone(.supersede_parent, 1, 2);
        try std.testing.expectEqual(@as(u32, 2), b.pending_deltas);

        // 3. Reload into fresh Bank -> MUST replay delta journal and activate ONLY pid=2 (len = 4 + 7 + 2 = 13)!
        var b2 = try Bank.init(gpa, io, dir, backend);
        defer b2.deinit();
        try std.testing.expect(try b2.loadIfPresent());
        try std.testing.expectEqual(@as(u32, 1), b2.activeUnitCount());
        const q = [_]u32{ 30, 31, 32 };
        const s_new = (try b2.routeAndCompose(&q)) orelse return error.ExpectedNewBelief;
        try std.testing.expectEqual(@as(u32, 4 + 7 + 2), s_new.at);
        backend.dropSnap(backend.ctx, s_new.saved);

        // 4. Undo/delete pid=2 -> MUST automatically restore pid=1 (len = 4 + 5 + 2 = 11)!
        try std.testing.expectEqual(@as(u32, 1), b2.forgetParent(2));
        const s_old = (try b2.routeAndCompose(&q)) orelse return error.ExpectedRestoredBelief;
        try std.testing.expectEqual(@as(u32, 4 + 5 + 2), s_old.at);
        backend.dropSnap(backend.ctx, s_old.saved);

        // 5. Teach 2 standing policies -> consolidation builds spectral `skill_rec` and compacts deltas
        const pol1 = [_]u32{ 50, 51, 52 };
        const pol2 = [_]u32{ 53, 54, 55 };
        _ = try b2.teachChunk(.{ .parent_id = 10, .body_ids = &pol1, .gate_ids = &.{}, .is_standing = true });
        _ = try b2.teachChunk(.{ .parent_id = 11, .body_ids = &pol2, .gate_ids = &.{}, .is_standing = true });
        const rep = try b2.consolidate(true, true);
        try std.testing.expectEqual(@as(u32, 2), rep.consolidated_policies);
        try std.testing.expectEqual(@as(usize, 2), b2.skill_rec.len);
        try std.testing.expectEqual(@as(u32, 0), b2.pending_deltas);

        // Reload after consolidation -> `skill_rec` survives pure-safetensors reload!
        var b3 = try Bank.init(gpa, io, dir, backend);
        defer b3.deinit();
        try std.testing.expect(try b3.loadIfPresent());
        try std.testing.expectEqual(@as(usize, 2), b3.skill_rec.len);
        try std.testing.expectEqual(@as(u32, 2), b3.skill_policy_count);
    }

    // Part B: Pure-attention Transformer (`family = .llama`, `num_rec = 0`) end-to-end save/reload/route!
    {
        const arch_llama = ArchSpec{
            .family = .llama,
            .hidden = 64,
            .num_layers = 4,
            .layer_lex = 1,
            .layer_sem = 2,
            .num_attn = 4,
            .kv_heads = 2,
            .head_dim = 32,
            .rotary_dim = 32,
            .rope_theta = 500_000.0,
            .num_rec = 0,
            .rec_heads = 0,
            .rec_rows = 0,
            .rec_cols = 0,
            .conv_elems = 0,
        };
        var mock_l = Mock{ .gpa = gpa, .arch = arch_llama };
        const backend_l = Backend{ .ctx = &mock_l, .arch = arch_llama, .extractHidden = Mock.extractHidden, .compileState = Mock.compileState, .buildSnap = Mock.buildSnap, .dropSnap = Mock.dropSnap };

        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(dir);

        var bl = try Bank.init(gpa, io, dir, backend_l);
        defer bl.deinit();
        try bl.ensureCalibrated(&cal_ids, &pref_ids, &suf_ids);
        const f_ids = [_]u32{ 30, 31, 32, 33 };
        _ = try bl.teachChunk(.{ .parent_id = 1, .body_ids = &f_ids, .gate_ids = &f_ids, .is_small_fact = true });
        try std.testing.expect((try bl.save()) > 0);

        var bl2 = try Bank.init(gpa, io, dir, backend_l);
        defer bl2.deinit();
        try std.testing.expect(try bl2.loadIfPresent());
        const s = (try bl2.routeAndCompose(&[_]u32{ 30, 31, 32 })) orelse return error.ExpectedLlamaHit;
        try std.testing.expectEqual(@as(u32, 4 + 4 + 2), s.at);
        backend_l.dropSnap(backend_l.ctx, s.saved);
    }
}
