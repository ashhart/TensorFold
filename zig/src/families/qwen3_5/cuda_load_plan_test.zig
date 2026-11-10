//! Real typed load ownership and refusal controls; no checkpoint weights or device are opened.
const std = @import("std");
const core = @import("core");
const plans = @import("cuda_load_plan.zig");
const weights = @import("cuda_weights.zig");
const hosted = @import("cuda_host_embed.zig");
const t = std.testing;
fn geometry() @import("cuda_geometry.zig").Geometry {
    return .{ .hidden = 5120, .intermediate = 17408, .layers = 64, .vocab = 248320, .k_heads = 16, .v_heads = 48, .q_heads = 24, .kv_heads = 4, .eps = 1e-6, .tied = false, .context = 131072, .linear = @splat(false) };
}
fn checkpoint() core.Checkpoint {
    return .{ .gpa = t.allocator, .io = t.io };
}
fn plan(other: bool) plans.LoadPlan {
    return .{ .checkpoint = checkpoint(), .source = if (other) .{ .checkpoint = checkpoint(), .plan = .{ .scope = .layers, .kinds = .{}, .weight_bytes = 128, .stage_bytes = 64 }, .dir = "alternate" } else null, .geometry = geometry(), .span = plans.Span.full(64), .workspace = @import("cuda_workspace_plan.zig").WorkspacePlan.forRows(geometry(), @import("qwen_prefill_config").max_rows) catch unreachable, .budgets = .{ .weight_bytes = if (other) 128 else null, .stage_bytes = 64 } };
}
const Mapping = struct {
    releases: usize = 0,
    fn release(ctx: *anyopaque) void {
        const m: *@This() = @ptrCast(@alignCast(ctx));
        m.releases += 1;
    }
    fn lease(m: *Mapping) plans.MappedSource {
        return .{ .ctx = m, .release = release };
    }
};
test "Qwen load plan: alternate profile refuses partial spans and mapped-primary combinations" {
    try plans.admit(plans.Span.full(64), false, false);
    try plans.admit(plans.Span.full(64), true, false);
    try t.expectError(error.AlternateQwenNeedsWholeModel, plans.admit(.{ .first = 1, .count = 63, .whole = 64 }, true, false));
    try t.expectError(error.UnqualifiedQwenLoadSpan, plans.admit(.{ .first = 0, .count = 32, .whole = 64 }, false, false));
    try t.expectError(error.InvalidQwenLoadSpan, plans.admit(.{ .first = 63, .count = 2, .whole = 64 }, false, false));
    try t.expectError(error.AlternateQwenCannotMapPrimary, plans.admit(plans.Span.full(64), true, true));
}
test "Qwen load plan: borrowed source pointers remain exact and all owned checkpoints close once" {
    var p = plan(true);
    try t.expect(p.primary() == &p.checkpoint);
    try t.expect(p.alternateSource().? == &p.source.?);
    try t.expect(p.primary() != &p.alternateSource().?.checkpoint);
    try t.expectEqual(@as(?usize, 128), p.budgets.weight_bytes);
    try t.expectEqual(@as(usize, 64), p.budgets.stage_bytes);
    p.deinit();
    try t.expect(!p.open);
    p.deinit();
    try t.expectError(error.QwenLoadClosed, p.requireCurrentUpload());
}
test "Qwen load plan: mapping transfer is atomic and release occurs once before checkpoint closure" {
    var p = plan(false);
    var m: Mapping = .{};
    try p.adoptMapped(m.lease());
    try t.expectEqual(@as(usize, 0), m.releases);
    try t.expectError(error.QwenLoadAlreadyBound, p.adoptMapped(m.lease()));
    try t.expectError(error.UnqualifiedQwenMappedUpload, p.requireCurrentUpload());
    p.finishMapped();
    try t.expectEqual(@as(usize, 1), m.releases);
    try p.requireCurrentUpload();
    p.deinit();
    try t.expectEqual(@as(usize, 1), m.releases);
}
test "Qwen load plan: refused alternate mapping remains caller-owned" {
    var p = plan(true);
    defer p.deinit();
    var m: Mapping = .{};
    try t.expectError(error.AlternateQwenCannotMapPrimary, p.adoptMapped(m.lease()));
    try t.expect(p.mapped == null and m.releases == 0);
    Mapping.release(&m);
    try t.expectEqual(@as(usize, 1), m.releases);
}
test "Qwen load plan: ordinary entry adapters and upload refusal preserve legacy signatures" {
    try t.expectEqual(@as(usize, 3), @typeInfo(@TypeOf(weights.load)).@"fn".param_types.len);
    try t.expectEqual(@as(usize, 5), @typeInfo(@TypeOf(hosted.arena)).@"fn".param_types.len);
    var p = plan(false);
    defer p.deinit();
    var m: Mapping = .{};
    try p.adoptMapped(m.lease());
    var e: @import("cuda_engine.zig").Engine = undefined;
    var stage: @import("cuda").DeviceBuffer = undefined;
    try t.expectError(error.UnqualifiedQwenMappedUpload, weights.loadPlanned(&e, &p, &stage));
}
test "Qwen load plan: weight stage and scratch budgets remain distinct with exact overflow refusal" {
    var b: plans.Budgets = .{ .stage_bytes = 1024 };
    try t.expectError(error.UnresolvedQwenWeightBudget, b.arenaBytes());
    b.weight_bytes = 4096;
    b.scratch_bytes = 2048;
    try t.expectEqual(@as(usize, 6144), try b.arenaBytes());
    try t.expectEqual(@as(usize, 1024), b.stage_bytes);
    b.weight_bytes = std.math.maxInt(usize);
    try t.expectError(error.Overflow, b.arenaBytes());
}

fn mappedCheckpoint(tmp: anytype) !core.Checkpoint {
    const header = "{\"probe\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    var bytes: [8 + header.len + 1]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    bytes[bytes.len - 1] = 37;
    try tmp.dir.writeFile(t.io, .{ .sub_path = "model.safetensors", .data = &bytes });
    const dir = try std.fmt.allocPrint(t.allocator, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    defer t.allocator.free(dir);
    return core.Checkpoint.openModel(t.allocator, t.io, dir);
}

test "Qwen load plan: real mapped primary and alternate checkpoints retain bytes and release all allocations" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var p = plan(true);
    p.checkpoint = try mappedCheckpoint(tmp);
    errdefer p.deinit();
    p.source.?.checkpoint = try mappedCheckpoint(tmp);
    const primary = try p.primary().expect("probe", .u8, &.{1});
    const other = try p.alternateSource().?.checkpoint.expect("probe", .u8, &.{1});
    try t.expectEqualSlices(u8, &.{37}, primary.bytes);
    try t.expectEqualSlices(u8, primary.bytes, other.bytes);
    try t.expect(primary.bytes.ptr != other.bytes.ptr);
    p.deinit();
    p.deinit(); // allocator and mapping cleanup must not execute twice
}

test "Qwen load plan: mapped upload lease releases while the real primary checkpoint is still borrowed" {
    var tmp = t.tmpDir(.{});
    defer tmp.cleanup();
    var p = plan(false);
    p.checkpoint = try mappedCheckpoint(tmp);
    defer p.deinit();
    const Observer = struct {
        p: *plans.LoadPlan,
        calls: usize = 0,
        live: bool = false,
        fn release(ctx: *anyopaque) void {
            const o: *@This() = @ptrCast(@alignCast(ctx));
            o.calls += 1;
            o.live = o.p.open and o.p.primary().has("probe");
        }
    };
    var observer: Observer = .{ .p = &p };
    try p.adoptMapped(.{ .ctx = &observer, .release = Observer.release });
    p.deinit();
    try t.expectEqual(@as(usize, 1), observer.calls);
    try t.expect(observer.live);
}
