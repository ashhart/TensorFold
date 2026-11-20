const std = @import("std");
const planner = @import("low_memory_plan.zig");
const transaction = @import("low_memory_plan_admission.zig");

const checkpoint: [32]u8 = @splat(7);
const draft_checkpoint: [32]u8 = @splat(8);

fn model() planner.Model {
    return .{
        .checkpoint = checkpoint,
        .weight_bytes = 6000,
        .max_context = 4096,
        .max_streams = 8,
        .kv = .{ .fixed_bytes = 10, .stream_bytes = 20, .shared_token_bytes = 1, .stream_token_bytes = 2 },
        .recurrent = .{ .fixed_bytes = 5, .stream_bytes = 7 },
        .drafter = .{
            .checkpoint = draft_checkpoint,
            .weight_bytes = 1000,
            .peak = .{ .fixed_bytes = 30, .stream_bytes = 40, .shared_token_bytes = 3, .stream_token_bytes = 4 },
        },
        .scratch = .{ .fixed_bytes = 50, .stream_bytes = 60, .shared_token_bytes = 5, .stream_token_bytes = 6 },
    };
}

fn placements() [2]planner.Placement {
    return .{
        .{
            .name = "resident",
            .checkpoint = checkpoint,
            .target = .{ .device = 6000 },
            .drafter = .{ .device = 1000 },
        },
        .{
            .name = "split-ssd",
            .checkpoint = checkpoint,
            .target = .{ .device = 2000, .host = 1000, .ssd = 3000 },
            .drafter = .{ .device = 400, .host = 200, .ssd = 400 },
            .device_expert_cache = 70,
            .host_expert_cache = 80,
            .ssd_expert_cache = 90,
        },
    };
}

fn request() planner.Request {
    return .{
        .context = 512,
        .streams = 2,
        .exact_context = true,
        .fixed_streams = true,
        .drafter = true,
        .allow_without_drafter = true,
        .prompt_cache = 100,
    };
}

fn limits(device: u64, reserve: u64) planner.MeasuredLimits {
    return .{
        .device = .{ .limit_bytes = device, .reserve_bytes = reserve },
        .host = .{ .limit_bytes = 1280 },
        .ssd = .{ .limit_bytes = 3490 },
    };
}

fn admitted(decision: planner.Decision) !planner.Plan {
    return switch (decision) {
        .admitted => |plan| plan,
        .refused => return error.UnexpectedRefusal,
    };
}

fn refused(decision: planner.Decision) !planner.Refusal {
    return switch (decision) {
        .admitted => return error.UnexpectedAdmission,
        .refused => |refusal| refusal,
    };
}

test "low-memory plans are deterministic per 16/24/48 GB target without inventing capacity" {
    const cases = [_]struct {
        envelope: planner.Envelope,
        device: u64,
        reserve: u64,
        placement: []const u8,
        headroom: u64,
    }{
        .{ .envelope = .gb16, .device = 20000, .reserve = 15, .placement = "split-ssd", .headroom = 170 },
        .{ .envelope = .gb24, .device = 24400, .reserve = 55, .placement = "resident", .headroom = 0 },
        .{ .envelope = .gb48, .device = 25345, .reserve = 0, .placement = "resident", .headroom = 1000 },
    };
    for (cases) |case| {
        const first = try admitted(try planner.choose(case.envelope, limits(case.device, case.reserve), model(), &placements(), request()));
        const second = try admitted(try planner.choose(case.envelope, limits(case.device, case.reserve), model(), &placements(), request()));
        try std.testing.expectEqualStrings(case.placement, first.placement.name);
        try std.testing.expectEqual(case.headroom, first.deviceHeadroom());
        try std.testing.expectEqualDeep(first, second);
        try std.testing.expectEqual(first.id, second.id);
        try std.testing.expect(first.full_service);
        try std.testing.expectEqual(@as(u64, 6000), first.placement.target.device + first.placement.target.host + first.placement.target.ssd);
        try std.testing.expectEqual(@as(u64, 1000), first.placement.drafter.device + first.placement.drafter.host + first.placement.drafter.ssd);
        var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
        defer out.deinit();
        try first.describe(&out.writer);
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
        defer parsed.deinit();
        const receipt = parsed.value.object;
        try std.testing.expectEqualStrings(case.envelope.label(), receipt.get("envelope").?.string);
        try std.testing.expectEqualStrings("admitted", receipt.get("outcome").?.string);
        try std.testing.expect(!receipt.get("physical_device_fit_established").?.bool);
    }
}

test "split placement charges every target, tier, KV, recurrent, draft and scratch byte once" {
    const plan = try admitted(try planner.choose(.gb16, limits(20000, 15), model(), &placements(), request()));
    try std.testing.expectEqual(@as(u64, 2610), plan.peaks.kv_device);
    try std.testing.expectEqual(@as(u64, 19), plan.peaks.recurrent_device);
    try std.testing.expectEqual(@as(u64, 6142), plan.peaks.draft_device);
    try std.testing.expectEqual(@as(u64, 8874), plan.peaks.scratch_device);
    try std.testing.expectEqual(@as(u64, 100), plan.peaks.prompt_cache_device);
    try std.testing.expectEqual(@as(u64, 2000 + 70 + 2610 + 19 + 6142 + 8874 + 100), plan.peaks.device_total);
    try std.testing.expectEqual(@as(u64, 1280), plan.peaks.host_total);
    try std.testing.expectEqual(@as(u64, 3490), plan.peaks.ssd_total);
    try std.testing.expectEqual(@as(u64, 0), plan.hostHeadroom());
    try std.testing.expectEqual(@as(u64, 0), plan.ssdHeadroom());
    try std.testing.expect(plan.fallback.placement);
    try std.testing.expect(!plan.fallback.context and !plan.fallback.concurrency and !plan.fallback.drafter);
}

test "fallback preserves dimensions before dropping a requested drafter" {
    const plan = try admitted(try planner.choose(.gb16, limits(18000, 0), model(), placements()[0..1], request()));
    try std.testing.expectEqualStrings("resident", plan.placement.name);
    try std.testing.expectEqual(@as(u32, 512), plan.context);
    try std.testing.expectEqual(@as(u32, 2), plan.streams);
    try std.testing.expect(!plan.drafter);
    try std.testing.expect(plan.fallback.drafter);
    try std.testing.expect(!plan.full_service);
    try std.testing.expectEqual(@as(u64, 0), plan.peaks.draft_device);
    try std.testing.expectEqual(@as(u64, 6000 + 2610 + 19 + 8874 + 100), plan.peaks.device_total);
}

test "bounded fallback lowers concurrency first and context only when streams are fixed" {
    var simple = model();
    simple.weight_bytes = 100;
    simple.max_context = 1024;
    simple.max_streams = 4;
    simple.kv = .{ .fixed_bytes = 10, .stream_token_bytes = 2 };
    simple.recurrent = .{ .stream_bytes = 3 };
    simple.drafter = null;
    simple.scratch = .{ .fixed_bytes = 20 };
    const candidates = [_]planner.Placement{.{ .name = "small", .checkpoint = checkpoint, .target = .{ .device = 100 } }};
    var ask: planner.Request = .{ .context = 1024, .streams = 4 };
    const concurrency = try admitted(try planner.choose(.gb24, limits(5000, 0), simple, &candidates, ask));
    try std.testing.expectEqual(@as(u32, 1024), concurrency.context);
    try std.testing.expectEqual(@as(u32, 2), concurrency.streams);
    try std.testing.expect(concurrency.fallback.concurrency and !concurrency.fallback.context);
    ask.fixed_streams = true;
    ask.streams = 2;
    const context = try admitted(try planner.choose(.gb24, limits(3000, 0), simple, &candidates, ask));
    try std.testing.expectEqual(@as(u32, 512), context.context);
    try std.testing.expectEqual(@as(u32, 2), context.streams);
    try std.testing.expect(context.fallback.context and !context.fallback.concurrency);
}

test "device and SSD failures choose another tier or refuse with the exact shortfall" {
    var candidates = placements();
    candidates[0].device_available = false;
    var required = request();
    required.allow_without_drafter = false;
    const split = try admitted(try planner.choose(.gb16, limits(20000, 15), model(), &candidates, required));
    try std.testing.expectEqualStrings("split-ssd", split.placement.name);
    candidates[1].ssd_available = false;
    const unavailable = try refused(try planner.choose(.gb16, limits(20000, 15), model(), &candidates, required));
    try std.testing.expectEqual(planner.RefusalReason.ssd_tier_unavailable, unavailable.reason);
    candidates[1].ssd_available = true;
    var measured = limits(20000, 15);
    measured.ssd.limit_bytes = 3489;
    const short = try refused(try planner.choose(.gb16, measured, model(), &candidates, required));
    try std.testing.expectEqual(planner.RefusalReason.ssd_tier_capacity, short.reason);
    try std.testing.expectEqual(@as(u64, 3490), short.peaks.ssd_total);
    try std.testing.expectEqual(@as(u64, 3489), try short.measured.ssd.usable());
}

test "refusal receipt is explicit at the caller-selected request floor" {
    var candidates = [_]planner.Placement{placements()[0]};
    var ask = request();
    ask.drafter = false;
    ask.allow_without_drafter = false;
    ask.exact_context = false;
    ask.fixed_streams = false;
    ask.context_step = 1;
    const refusal = try refused(try planner.choose(.gb16, limits(100, 0), model(), &candidates, ask));
    try std.testing.expectEqual(planner.RefusalReason.device_capacity, refusal.reason);
    try std.testing.expectEqual(@as(u32, 1), refusal.context);
    try std.testing.expectEqual(@as(u32, 1), refusal.streams);
    try std.testing.expect(refusal.peaks.device_total > try refusal.measured.device.usable());
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try refusal.describe(&out.writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    const receipt = parsed.value.object;
    try std.testing.expectEqualStrings("refused", receipt.get("outcome").?.string);
    try std.testing.expectEqualStrings("device_capacity", receipt.get("reason").?.string);
    try std.testing.expectEqual(@as(i64, 1), receipt.get("request_floor").?.object.get("context").?.integer);
    try std.testing.expect(!receipt.get("physical_device_fit_established").?.bool);
}

test "concurrency changes recompute every stream-dependent peak exactly" {
    var measured = limits(50000, 0);
    const two = try admitted(try planner.choose(.gb48, measured, model(), placements()[0..1], request()));
    var one_request = request();
    one_request.streams = 1;
    const one = try admitted(try planner.choose(.gb48, measured, model(), placements()[0..1], one_request));
    try std.testing.expectEqual(@as(u64, 1566), one.peaks.kv_device);
    try std.testing.expectEqual(@as(u64, 12), one.peaks.recurrent_device);
    try std.testing.expectEqual(@as(u64, 4654), one.peaks.draft_device);
    try std.testing.expectEqual(@as(u64, 5742), one.peaks.scratch_device);
    try std.testing.expectEqual(@as(u64, 18074), one.peaks.device_total);
    try std.testing.expectEqual(@as(u64, 24345), two.peaks.device_total);
    try std.testing.expect(!std.mem.eql(u8, &two.id, &one.id));
    measured.device.limit_bytes -= 1;
    const changed = try admitted(try planner.choose(.gb48, measured, model(), placements()[0..1], request()));
    try std.testing.expect(!std.mem.eql(u8, &two.id, &changed.id));
}

test "admission pressure preparation rolls back without changing the admitted receipt or promises" {
    const initial = try admitted(try planner.choose(.gb24, limits(25100, 0), model(), placements()[0..1], request()));
    var admission = transaction.Admission.init(initial);
    try std.testing.expectEqual(@as(u64, 755), admission.remaining());
    var lease: transaction.Lease = .{};
    try admission.promise(&lease, 700);
    try std.testing.expectEqual(@as(u64, 55), admission.remaining());
    var extra: transaction.Lease = .{};
    try std.testing.expectError(error.LaneMemory, admission.promise(&extra, 56));
    try std.testing.expect(!extra.active);
    try std.testing.expectEqual(@as(u64, 700), admission.promised);
    var one_stream = request();
    one_stream.streams = 1;
    try std.testing.expectError(error.ConcurrencyChangeWithLeases, admission.prepare(.gb24, limits(25000, 0), model(), placements()[0..1], one_stream));
    try std.testing.expectError(error.AdmittedBytesExceedPlan, admission.prepare(.gb24, limits(24600, 0), model(), placements()[0..1], request()));
    try std.testing.expect(admission.pending == null);
    try std.testing.expectEqual(initial.id, admission.active.id);
    var other = transaction.Admission.init(initial);
    try std.testing.expectError(error.ForeignLease, other.finish(&lease));
    try std.testing.expectEqual(@as(u64, 700), admission.promised);
    try admission.finish(&lease);
    try std.testing.expectEqual(@as(u64, 755), admission.remaining());
    const pressure_decision = try admission.prepare(.gb24, limits(24600, 0), model(), placements()[0..1], request());
    const pressure = try admitted(pressure_decision);
    try std.testing.expectEqual(@as(u64, 255), pressure.deviceHeadroom());
    try std.testing.expectEqual(initial.id, admission.active.id);
    admission.rollback();
    try std.testing.expect(admission.pending == null);
    try std.testing.expectEqual(initial.id, admission.active.id);
    _ = try admission.prepare(.gb24, limits(24600, 0), model(), placements()[0..1], request());
    var wrong = pressure.id;
    wrong[0] +%= 1;
    try std.testing.expectError(error.PlanReceiptChanged, admission.commit(wrong));
    try std.testing.expect(admission.pending != null);
    admission.rollback();
    _ = try admission.prepare(.gb24, limits(24600, 0), model(), placements()[0..1], request());
    try admission.commit(pressure.id);
    try std.testing.expectEqual(pressure.id, admission.active.id);
    try std.testing.expect(admission.pending == null);
}

test "concurrency replacement commits only after preparation and rollback keeps the old geometry" {
    const initial = try admitted(try planner.choose(.gb48, limits(50000, 0), model(), placements()[0..1], request()));
    var admission = transaction.Admission.init(initial);
    var ask = request();
    ask.streams = 1;
    const next = try admitted(try admission.prepare(.gb48, limits(50000, 0), model(), placements()[0..1], ask));
    try std.testing.expectEqual(@as(u32, 2), admission.active.streams);
    admission.rollback();
    try std.testing.expectEqual(@as(u32, 2), admission.active.streams);
    _ = try admission.prepare(.gb48, limits(50000, 0), model(), placements()[0..1], ask);
    try admission.commit(next.id);
    try std.testing.expectEqual(@as(u32, 1), admission.active.streams);
    try std.testing.expectEqualDeep(next.peaks, admission.active.peaks);
}

test "bad measurements, immutable weight sums and arithmetic overflow fail closed" {
    var measured = limits(25000, 0);
    measured.device.reserve_bytes = 25001;
    try std.testing.expectError(error.BadMeasuredLimits, planner.choose(.gb16, measured, model(), &placements(), request()));
    var candidates = placements();
    candidates[0].target.device = 5999;
    try std.testing.expectError(error.WeightConservation, planner.choose(.gb16, limits(25000, 0), model(), &candidates, request()));
    candidates = placements();
    candidates[0].checkpoint[0] +%= 1;
    try std.testing.expectError(error.CheckpointMismatch, planner.choose(.gb16, limits(25000, 0), model(), &candidates, request()));
    var huge = model();
    huge.scratch.fixed_bytes = std.math.maxInt(u64);
    huge.scratch.stream_bytes = 1;
    try std.testing.expectError(error.MemoryAccountingOverflow, planner.choose(.gb16, limits(25000, 0), huge, placements()[0..1], request()));
}
