const std = @import("std");
const cuda = @import("cuda");
const kvpage = @import("cuda_exl3_kvpage.zig");
const weights = @import("cuda_prompt_wt.zig");

pub fn reserve(engine: anytype, cap: usize) !usize {
    const cfg = (engine.page_config orelse return 0).forChunk(engine.chunk);
    const bytes = try kvpage.sharedBytesFor(cap, cfg);
    if (cfg.shared_lanes == 0) return 0;
    const placement = @import("cuda_exl3_kv.zig").of(engine);
    if (placement.k8 != null or placement.heads != weights.kv_heads) return error.KvPagePlacement;
    const options: kvpage.shared_device.Options = .{ .cap = cap, .lanes = cfg.shared_lanes, .window = cfg.window, .rows = cfg.rows, .slots = cfg.slots };
    if (engine.shared_arena) |arena| {
        if (!std.meta.eql(arena.layout.options, options)) return error.SharedKvGeometry;
        return bytes;
    }
    engine.shared_arena = try kvpage.shared_device.Arena.init(engine.gpa, engine.driver, options);
    return bytes;
}

pub fn begin(engine: anytype, sequences: anytype) !?kvpage.shared_device.Lease {
    const cfg = engine.page_config orelse return null;
    if (cfg.shared_lanes == 0) return null;
    const arena = engine.shared_arena orelse return error.SharedKvArenaRequired;
    if (sequences.len > kvpage.shared_device.max_lanes) return error.SharedKvLeaseOwners;
    var owners: [kvpage.shared_device.max_lanes]kvpage.shared_contract.Sequence = undefined;
    for (sequences, 0..) |sequence, index| {
        const pg = sequence.paged orelse return error.SharedKvSequence;
        if (pg.cfg.shared_lanes == 0 or pg.shared_arena != arena) return error.SharedKvSequence;
        owners[index] = pg.shared_sequence;
    }
    const lease = try arena.acquireMany(owners[0..sequences.len]);
    errdefer arena.cancel(lease) catch unreachable;
    try arena.submittedOn(lease, engine.stream.*);
    return lease;
}

pub fn end(engine: anytype, lease: ?kvpage.shared_device.Lease) void {
    if (lease) |held| if (engine.shared_arena.?.active != null) engine.shared_arena.?.completeAfterConsumers(held) catch unreachable;
}

pub fn fail(engine: anytype, lease: ?kvpage.shared_device.Lease, sequences: anytype) void {
    if (lease) |held| {
        const arena = engine.shared_arena.?;
        if (!arena.active.?.failed) arena.fail(held) catch unreachable;
        for (sequences) |sequence| sequence.invalidated = true;
        cuda.completion.drain(engine.driver, engine.stream.*, "shared KV consumers did not drain");
        arena.drainFailed(held) catch @panic("shared KV failure still owns consumers");
    }
}

pub fn finish(engine: anytype, sequences: anytype, comptime complete: anytype) !void {
    var any = false;
    for (sequences) |q| if (q.paged) |pg| {
        try pg.stageStatus(engine.stream.*);
        any = true;
    };
    if (!any) return;
    try complete(engine);
    for (sequences) |q| if (q.paged) |pg| try pg.checkStatus();
}
