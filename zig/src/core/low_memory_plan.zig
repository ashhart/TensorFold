//! Host-only placement and peak-budget plans from caller-measured limits; no physical-device fit is established.
const std = @import("std");

pub const max_placements: usize = 16;
pub const max_cells: u64 = 65536;

pub const Envelope = enum {
    gb16,
    gb24,
    gb48,

    pub fn label(e: Envelope) []const u8 {
        return switch (e) {
            .gb16 => "16GB",
            .gb24 => "24GB",
            .gb48 => "48GB",
        };
    }
};

pub const TierLimit = struct {
    limit_bytes: u64,
    reserve_bytes: u64 = 0,

    pub fn usable(l: TierLimit) !u64 {
        if (l.reserve_bytes > l.limit_bytes) return error.BadMeasuredLimits;
        return l.limit_bytes - l.reserve_bytes;
    }
};

pub const MeasuredLimits = struct {
    device: TierLimit,
    host: TierLimit = .{ .limit_bytes = 0 },
    ssd: TierLimit = .{ .limit_bytes = 0 },
};

pub const Curve = struct {
    fixed_bytes: u64 = 0,
    stream_bytes: u64 = 0,
    shared_token_bytes: u64 = 0,
    stream_token_bytes: u64 = 0,

    pub fn peak(c: Curve, context: u32, streams: u32) !u64 {
        const per_stream = try mul(c.stream_bytes, streams);
        const shared = try mul(c.shared_token_bytes, context);
        const tokens = try mul(@as(u64, context), streams);
        const per_stream_token = try mul(c.stream_token_bytes, tokens);
        return sum(&.{ c.fixed_bytes, per_stream, shared, per_stream_token });
    }
};

pub const Drafter = struct {
    checkpoint: [32]u8,
    weight_bytes: u64,
    peak: Curve = .{},
};

pub const Model = struct {
    checkpoint: [32]u8,
    weight_bytes: u64,
    max_context: u32,
    max_streams: u32,
    kv: Curve,
    recurrent: Curve = .{},
    drafter: ?Drafter = null,
    scratch: Curve = .{},
};

pub const Weights = struct {
    device: u64 = 0,
    host: u64 = 0,
    ssd: u64 = 0,
};

pub const Placement = struct {
    name: []const u8,
    checkpoint: [32]u8,
    target: Weights,
    drafter: Weights = .{},
    device_expert_cache: u64 = 0,
    host_expert_cache: u64 = 0,
    ssd_expert_cache: u64 = 0,
    device_available: bool = true,
    host_available: bool = true,
    ssd_available: bool = true,
};

pub const Request = struct {
    context: u32,
    streams: u32,
    exact_context: bool = false,
    fixed_streams: bool = false,
    context_step: u32 = 256,
    drafter: bool = false,
    allow_without_drafter: bool = false,
    prompt_cache: u64 = 0,
};

pub const Fallback = struct {
    placement: bool,
    context: bool,
    concurrency: bool,
    drafter: bool,
};

pub const Peaks = struct {
    device_weights: u64,
    device_expert_cache: u64,
    kv_device: u64,
    recurrent_device: u64,
    draft_device: u64,
    scratch_device: u64,
    prompt_cache_device: u64,
    device_total: u64,
    host_total: u64,
    ssd_total: u64,
};

pub const Plan = struct {
    envelope: Envelope,
    measured: MeasuredLimits,
    model: Model,
    placement: Placement,
    request: Request,
    context: u32,
    streams: u32,
    drafter: bool,
    peaks: Peaks,
    fallback: Fallback,
    full_service: bool,
    id: [32]u8,

    pub fn deviceHeadroom(p: Plan) u64 {
        return (p.measured.device.usable() catch unreachable) - p.peaks.device_total;
    }

    pub fn hostHeadroom(p: Plan) u64 {
        return (p.measured.host.usable() catch unreachable) - p.peaks.host_total;
    }

    pub fn ssdHeadroom(p: Plan) u64 {
        return (p.measured.ssd.usable() catch unreachable) - p.peaks.ssd_total;
    }

    pub fn describe(p: Plan, w: *std.Io.Writer) !void {
        try std.json.Stringify.value(.{
            .schema = 1,
            .outcome = "admitted",
            .identity = std.fmt.bytesToHex(p.id, .lower),
            .envelope = p.envelope.label(),
            .checkpoint = std.fmt.bytesToHex(p.model.checkpoint, .lower),
            .placement = p.placement.name,
            .weights = .{ .target = p.placement.target, .drafter = if (p.drafter) p.placement.drafter else null },
            .context = p.context,
            .streams = p.streams,
            .drafter = p.drafter,
            .full_service = p.full_service,
            .fallback = p.fallback,
            .measured_limits = p.measured,
            .peaks = p.peaks,
            .headroom = .{ .device = p.deviceHeadroom(), .host = p.hostHeadroom(), .ssd = p.ssdHeadroom() },
            .physical_device_fit_established = false,
        }, .{}, w);
        try w.writeByte('\n');
    }
};

pub const RefusalReason = enum {
    device_unavailable,
    host_tier_unavailable,
    ssd_tier_unavailable,
    device_capacity,
    host_tier_capacity,
    ssd_tier_capacity,
};

pub const Refusal = struct {
    envelope: Envelope,
    measured: MeasuredLimits,
    model: Model,
    placement: Placement,
    request: Request,
    context: u32,
    streams: u32,
    drafter: bool,
    peaks: Peaks,
    reason: RefusalReason,

    pub fn describe(r: Refusal, w: *std.Io.Writer) !void {
        const device_shortfall = r.peaks.device_total -| (r.measured.device.usable() catch unreachable);
        const host_shortfall = r.peaks.host_total -| (r.measured.host.usable() catch unreachable);
        const ssd_shortfall = r.peaks.ssd_total -| (r.measured.ssd.usable() catch unreachable);
        try std.json.Stringify.value(.{
            .schema = 1,
            .outcome = "refused",
            .envelope = r.envelope.label(),
            .checkpoint = std.fmt.bytesToHex(r.model.checkpoint, .lower),
            .placement = r.placement.name,
            .reason = @tagName(r.reason),
            .request_floor = .{ .context = r.context, .streams = r.streams, .drafter = r.drafter },
            .measured_limits = r.measured,
            .peaks = r.peaks,
            .shortfall = .{ .device = device_shortfall, .host = host_shortfall, .ssd = ssd_shortfall },
            .physical_device_fit_established = false,
        }, .{}, w);
        try w.writeByte('\n');
    }
};

pub const Decision = union(enum) {
    admitted: Plan,
    refused: Refusal,

    pub fn describe(d: Decision, w: *std.Io.Writer) !void {
        switch (d) {
            .admitted => |p| try p.describe(w),
            .refused => |r| try r.describe(w),
        }
    }
};

const Attempt = struct {
    peaks: Peaks,
    refusal: ?RefusalReason,
};

pub fn choose(envelope: Envelope, measured: MeasuredLimits, model: Model, placements: []const Placement, request: Request) !Decision {
    try validate(measured, model, placements, request);
    var last: ?Refusal = null;
    var context = request.context;
    while (context != 0) {
        var streams = request.streams;
        while (streams != 0) {
            var drop_drafter = false;
            while (true) {
                const drafting = request.drafter and !drop_drafter;
                for (placements, 0..) |placement, placement_index| {
                    const attempt = try evaluate(model, placement, measured, request.prompt_cache, context, streams, drafting);
                    if (attempt.refusal) |reason| {
                        last = .{
                            .envelope = envelope,
                            .measured = measured,
                            .model = model,
                            .placement = placement,
                            .request = request,
                            .context = context,
                            .streams = streams,
                            .drafter = drafting,
                            .peaks = attempt.peaks,
                            .reason = reason,
                        };
                        continue;
                    }
                    const fallback: Fallback = .{
                        .placement = placement_index != 0,
                        .context = context != request.context,
                        .concurrency = streams != request.streams,
                        .drafter = request.drafter and !drafting,
                    };
                    var plan: Plan = .{
                        .envelope = envelope,
                        .measured = measured,
                        .model = model,
                        .placement = placement,
                        .request = request,
                        .context = context,
                        .streams = streams,
                        .drafter = drafting,
                        .peaks = attempt.peaks,
                        .fallback = fallback,
                        .full_service = !fallback.context and !fallback.concurrency and !fallback.drafter,
                        .id = undefined,
                    };
                    plan.id = identity(plan);
                    return .{ .admitted = plan };
                }
                if (!request.drafter or !request.allow_without_drafter or drop_drafter) break;
                drop_drafter = true;
            }
            if (request.fixed_streams) break;
            streams -= 1;
        }
        if (request.exact_context) break;
        context = if (context % request.context_step != 0)
            context / request.context_step * request.context_step
        else
            context - request.context_step;
    }
    return .{ .refused = last.? };
}

fn validate(measured: MeasuredLimits, model: Model, placements: []const Placement, request: Request) !void {
    _ = try measured.device.usable();
    _ = try measured.host.usable();
    _ = try measured.ssd.usable();
    if (model.weight_bytes == 0 or model.max_context == 0 or model.max_streams == 0) return error.BadModelGeometry;
    if (model.drafter) |draft| if (draft.weight_bytes == 0) return error.BadModelGeometry;
    if (request.context == 0 or request.context > model.max_context or request.streams == 0 or
        request.streams > model.max_streams or request.context_step == 0) return error.BadFitRequest;
    if (request.drafter and model.drafter == null) return error.DrafterUnavailable;
    if (placements.len == 0 or placements.len > max_placements) return error.BadPlacementCount;
    const windows: u64 = if (request.exact_context) 1 else (@as(u64, request.context) + request.context_step - 1) / request.context_step;
    const counts: u64 = if (request.fixed_streams) 1 else request.streams;
    if (windows * counts > max_cells) return error.FitCensusTooLarge;
    for (placements) |placement| {
        if (placement.name.len == 0) return error.BadPlacement;
        if (!std.mem.eql(u8, &placement.checkpoint, &model.checkpoint)) return error.CheckpointMismatch;
        if (try weightTotal(placement.target) != model.weight_bytes) return error.WeightConservation;
        if (model.drafter) |draft| {
            if (try weightTotal(placement.drafter) != draft.weight_bytes) return error.WeightConservation;
        } else if (try weightTotal(placement.drafter) != 0) return error.WeightConservation;
    }
}

fn evaluate(model: Model, placement: Placement, measured: MeasuredLimits, prompt_cache: u64, context: u32, streams: u32, drafting: bool) !Attempt {
    const kv = try model.kv.peak(context, streams);
    const recurrent = try model.recurrent.peak(context, streams);
    const scratch = try model.scratch.peak(context, streams);
    const draft_curve = if (drafting) try model.drafter.?.peak.peak(context, streams) else 0;
    const draft_weights = if (drafting) placement.drafter.device else 0;
    const draft = try sum(&.{ draft_curve, draft_weights });
    const host_draft = if (drafting) placement.drafter.host else 0;
    const ssd_draft = if (drafting) placement.drafter.ssd else 0;
    const device_total = try sum(&.{ placement.target.device, placement.device_expert_cache, kv, recurrent, draft, scratch, prompt_cache });
    const host_total = try sum(&.{ placement.target.host, host_draft, placement.host_expert_cache });
    const ssd_total = try sum(&.{ placement.target.ssd, ssd_draft, placement.ssd_expert_cache });
    const peaks: Peaks = .{
        .device_weights = placement.target.device,
        .device_expert_cache = placement.device_expert_cache,
        .kv_device = kv,
        .recurrent_device = recurrent,
        .draft_device = draft,
        .scratch_device = scratch,
        .prompt_cache_device = prompt_cache,
        .device_total = device_total,
        .host_total = host_total,
        .ssd_total = ssd_total,
    };
    var refusal: ?RefusalReason = null;
    if (!placement.device_available) {
        refusal = .device_unavailable;
    } else if (host_total != 0 and !placement.host_available) {
        refusal = .host_tier_unavailable;
    } else if (ssd_total != 0 and !placement.ssd_available) {
        refusal = .ssd_tier_unavailable;
    } else if (device_total > try measured.device.usable()) {
        refusal = .device_capacity;
    } else if (host_total > try measured.host.usable()) {
        refusal = .host_tier_capacity;
    } else if (ssd_total > try measured.ssd.usable()) {
        refusal = .ssd_tier_capacity;
    }
    return .{ .peaks = peaks, .refusal = refusal };
}

fn weightTotal(w: Weights) !u64 {
    return sum(&.{ w.device, w.host, w.ssd });
}

fn mul(a: u64, b: anytype) !u64 {
    return std.math.mul(u64, a, @as(u64, b)) catch return error.MemoryAccountingOverflow;
}

fn sum(values: []const u64) !u64 {
    var total: u64 = 0;
    for (values) |value| total = std.math.add(u64, total, value) catch return error.MemoryAccountingOverflow;
    return total;
}

fn identity(p: Plan) [32]u8 {
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update("tensorfold-low-memory-plan-v1");
    hashInt(&h, @backingInt(p.envelope));
    hashLimit(&h, p.measured.device);
    hashLimit(&h, p.measured.host);
    hashLimit(&h, p.measured.ssd);
    hashModel(&h, p.model);
    hashText(&h, p.placement.name);
    h.update(&p.placement.checkpoint);
    hashWeights(&h, p.placement.target);
    hashWeights(&h, p.placement.drafter);
    hashInt(&h, p.placement.device_expert_cache);
    hashInt(&h, p.placement.host_expert_cache);
    hashInt(&h, p.placement.ssd_expert_cache);
    hashInt(&h, @intFromBool(p.placement.device_available));
    hashInt(&h, @intFromBool(p.placement.host_available));
    hashInt(&h, @intFromBool(p.placement.ssd_available));
    hashInt(&h, p.request.context);
    hashInt(&h, p.request.streams);
    hashInt(&h, @intFromBool(p.request.exact_context));
    hashInt(&h, @intFromBool(p.request.fixed_streams));
    hashInt(&h, p.request.context_step);
    hashInt(&h, @intFromBool(p.request.drafter));
    hashInt(&h, @intFromBool(p.request.allow_without_drafter));
    hashInt(&h, p.request.prompt_cache);
    hashInt(&h, p.context);
    hashInt(&h, p.streams);
    hashInt(&h, @intFromBool(p.drafter));
    hashPeaks(&h, p.peaks);
    return h.finalResult();
}

fn hashModel(h: *std.crypto.hash.sha2.Sha256, model: Model) void {
    h.update(&model.checkpoint);
    hashInt(h, model.weight_bytes);
    hashInt(h, model.max_context);
    hashInt(h, model.max_streams);
    hashCurve(h, model.kv);
    hashCurve(h, model.recurrent);
    hashCurve(h, model.scratch);
    hashInt(h, @intFromBool(model.drafter != null));
    if (model.drafter) |draft| {
        h.update(&draft.checkpoint);
        hashInt(h, draft.weight_bytes);
        hashCurve(h, draft.peak);
    }
}

fn hashCurve(h: *std.crypto.hash.sha2.Sha256, curve: Curve) void {
    hashInt(h, curve.fixed_bytes);
    hashInt(h, curve.stream_bytes);
    hashInt(h, curve.shared_token_bytes);
    hashInt(h, curve.stream_token_bytes);
}

fn hashPeaks(h: *std.crypto.hash.sha2.Sha256, peaks: Peaks) void {
    hashInt(h, peaks.device_weights);
    hashInt(h, peaks.device_expert_cache);
    hashInt(h, peaks.kv_device);
    hashInt(h, peaks.recurrent_device);
    hashInt(h, peaks.draft_device);
    hashInt(h, peaks.scratch_device);
    hashInt(h, peaks.prompt_cache_device);
    hashInt(h, peaks.device_total);
    hashInt(h, peaks.host_total);
    hashInt(h, peaks.ssd_total);
}

fn hashWeights(h: *std.crypto.hash.sha2.Sha256, weights: Weights) void {
    hashInt(h, weights.device);
    hashInt(h, weights.host);
    hashInt(h, weights.ssd);
}

fn hashLimit(h: *std.crypto.hash.sha2.Sha256, limit: TierLimit) void {
    hashInt(h, limit.limit_bytes);
    hashInt(h, limit.reserve_bytes);
}

fn hashText(h: *std.crypto.hash.sha2.Sha256, text: []const u8) void {
    hashInt(h, text.len);
    h.update(text);
}

fn hashInt(h: *std.crypto.hash.sha2.Sha256, value: anytype) void {
    var bytes: [8]u8 = undefined;
    std.mem.writeInt(u64, &bytes, @as(u64, value), .little);
    h.update(&bytes);
}

test {
    _ = @import("low_memory_plan_test.zig");
}
