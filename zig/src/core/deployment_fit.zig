//! Choose an existing resident/SSD placement from exact byte ledgers; no device or model execution occurs here.
const std = @import("std");

pub const quantum: u32 = 256;
pub const max_cells: u32 = 65536;
pub const max_placements: usize = 16;

pub const Model = struct {
    checkpoint: [32]u8,
    max_context: u32,
    max_streams: u32,
    /// Fixed cache bytes in each stream, plus bytes for each token in each stream.
    stream_fixed: u64,
    stream_token: u64,
};

/// `resident` includes the charged expert pool; `expert_pool` is the existing --ssd-experts budget.
pub const Placement = struct {
    name: []const u8,
    checkpoint: [32]u8,
    resident: u64,
    scratch: u64,
    expert_pool: u64 = 0,
    source_disk: u64 = 0,
    available: bool = true,
};

pub const Host = struct {
    memory_limit: u64,
    memory_reserve: u64,
    disk_free: u64 = 0,
    disk_floor: u64 = 0,
};

pub const Request = struct {
    context: u32,
    streams: u32,
    exact_context: bool = false,
    fixed_streams: bool = false,
    prompt_cache: u64 = 0,
};

pub const Plan = struct {
    placement: Placement,
    context: u32,
    streams: u32,
    prompt_cache: u64,
    memory_used: u64,
    memory_headroom: u64,
    full_service: bool,

    pub fn describe(p: Plan, w: *std.Io.Writer) !void {
        try std.json.Stringify.value(.{
            .schema = 1,
            .placement = p.placement.name,
            .checkpoint = std.fmt.bytesToHex(p.placement.checkpoint, .lower),
            .context = p.context,
            .streams = p.streams,
            .prompt_cache_bytes = p.prompt_cache,
            .ssd_experts_bytes = p.placement.expert_pool,
            .source_disk_bytes = p.placement.source_disk,
            .memory_used_bytes = p.memory_used,
            .memory_headroom_bytes = p.memory_headroom,
            .full_service = p.full_service,
        }, .{}, w);
        try w.writeByte('\n');
    }
};

pub fn fit(host: Host, model: Model, placements: []const Placement, request: Request) !Plan {
    try validate(model, placements, request);
    const windows: u64 = if (request.exact_context) 1 else (@as(u64, request.context) + quantum - 1) / quantum;
    const counts: u64 = if (request.fixed_streams) 1 else request.streams;
    if (windows * counts > max_cells) return error.FitCensusTooLarge;

    var context = request.context;
    while (context != 0) {
        var streams = request.streams;
        while (streams != 0) {
            for (placements) |placement| {
                const used = memoryUsed(placement, model, request.prompt_cache, context, streams) orelse continue;
                if (!fits(host, placement, used)) continue;
                const reserved = host.memory_reserve;
                return .{
                    .placement = placement,
                    .context = context,
                    .streams = streams,
                    .prompt_cache = request.prompt_cache,
                    .memory_used = used,
                    .memory_headroom = host.memory_limit - reserved - used,
                    .full_service = context == request.context and streams == request.streams,
                };
            }
            if (request.fixed_streams) break;
            streams -= 1;
        }
        if (request.exact_context) break;
        context = if (context % quantum != 0) context / quantum * quantum else context - quantum;
    }
    return error.NoDeploymentFit;
}

fn validate(model: Model, placements: []const Placement, request: Request) !void {
    if (model.max_context == 0 or model.max_streams == 0) return error.BadModelGeometry;
    if (request.context == 0 or request.context > model.max_context or
        request.streams == 0 or request.streams > model.max_streams) return error.BadFitRequest;
    if (placements.len == 0 or placements.len > max_placements) return error.BadPlacementCount;
    for (placements) |placement| {
        if (placement.name.len == 0) return error.BadPlacement;
        if (!std.mem.eql(u8, &placement.checkpoint, &model.checkpoint)) return error.CheckpointMismatch;
        if (placement.expert_pool == 0 and placement.source_disk != 0) return error.BadPlacement;
    }
}

fn memoryUsed(placement: Placement, model: Model, prompt_cache: u64, context: u32, streams: u32) ?u64 {
    const token_bytes = std.math.mul(u64, model.stream_token, context) catch return null;
    const per_stream = std.math.add(u64, model.stream_fixed, token_bytes) catch return null;
    const caches = std.math.mul(u64, per_stream, streams) catch return null;
    const standing = std.math.add(u64, placement.resident, placement.scratch) catch return null;
    const with_prompt = std.math.add(u64, standing, prompt_cache) catch return null;
    return std.math.add(u64, with_prompt, caches) catch null;
}

fn fits(host: Host, placement: Placement, used: u64) bool {
    if (!placement.available) return false;
    const memory = std.math.add(u64, used, host.memory_reserve) catch return false;
    if (memory > host.memory_limit) return false;
    const disk = std.math.add(u64, placement.source_disk, host.disk_floor) catch return false;
    return disk <= host.disk_free;
}

test {
    _ = @import("deployment_fit_test.zig");
}
