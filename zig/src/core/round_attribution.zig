//! Complete-round receipts: activity unions and inclusive host spans have distinct denominators.
const std = @import("std");
pub const schema_name = "tensorfold.cuda-round.v1";
pub const Phase = enum { other, forward, land, proposal, proposal_hold, proposal_select, candidate_frame, token_readback, logits_readback, keep_rollback, state_publication };
pub const Op = enum { other, qmmf, attention, logits_sampling, h2d, d2h, d2d, memset };
pub const Qmmf = struct { m: u32, n: u32, k: u32, sk: u32, cluster: [3]u32, reduce: bool, mode: []const u8 };
pub const Device = struct {
    correlation: u64,
    parent_scope: u64 = 0,
    graph_replay: ?u64 = null,
    stream: u64,
    phase: Phase,
    op: Op,
    symbol: []const u8,
    begin_ns: u64,
    end_ns: u64,
    bytes: u64 = 0,
    grid: [3]u32 = .{ 0, 0, 0 },
    block: [3]u32 = .{ 0, 0, 0 },
    qmmf: []const Qmmf = &.{},
};
pub const Host = struct { correlation: u64, parent_scope: u64 = 0, label: []const u8 = "", phase: Phase, kind: enum { span, wait, attribution_drain }, begin_ns: u64, end_ns: u64 };
pub const Request = struct { stream: []const u8, prompt_sha256: []const u8, prompt_tokens: usize, sampling_json: []const u8, cached: u64 };
pub const Window = struct { cached_after: u64 = 0, pending_after: ?u32 = null, rolled_back_rows: usize = 0, stream: []const u8, kind: []const u8, offered_rows: usize, proposal_nodes: usize, connected_accepted_depth: usize, kept_path: []const u32, committed_ids: []const u32 };
pub const Frame = struct { correlation: u64, live_rows: usize, windows: usize, shape_bucket: usize, shape_key: u64, mt2: bool };
pub const Budget = struct {
    complete_round_wall_ns: u64,
    functional_return_ns: u64,
    device_envelope_ns: u64,
    device_event_span_union_ns: u64,
    kernel_event_span_sum_ns: u64,
    device_envelope_gap_ns: u64,
    target_verify_device_event_union_ns: u64,
    proposal_device_event_union_ns: u64,
    device_by_operation_union_ns: [std.enums.values(Op).len]u64,
    device_by_phase_union_ns: [std.enums.values(Phase).len]u64,
    host_wait_union_ns: u64,
    state_publication_union_ns: u64,
    attribution_drain_union_ns: u64,
};
pub const Record = struct {
    schema: []const u8 = schema_name,
    round: u64,
    identity: []const u8,
    complete: bool,
    problem: ?[]const u8,
    requests: []const Request,
    frames: []const Frame,
    windows: []const Window,
    host: []const Host,
    device: []const Device,
    budget: Budget,
};
pub const Interval = struct { begin: u64, end: u64 };

/// Sum the covered time once, including touching intervals; reject backwards and overflowing arithmetic.
pub fn unionNs(a: std.mem.Allocator, intervals: []const Interval) !u64 {
    const sorted = try a.dupe(Interval, intervals);
    defer a.free(sorted);
    for (sorted) |s| if (s.end < s.begin) return error.BackwardsInterval;
    std.mem.sort(Interval, sorted, {}, struct {
        fn less(_: void, x: Interval, y: Interval) bool {
            return x.begin < y.begin;
        }
    }.less);
    var total: u64 = 0;
    var first: u64 = 0;
    var last: u64 = 0;
    for (sorted, 0..) |s, i| {
        if (i == 0) {
            first = s.begin;
            last = s.end;
            continue;
        }
        if (s.begin <= last) {
            last = @max(last, s.end);
            continue;
        }
        total = try std.math.add(u64, total, last - first);
        first = s.begin;
        last = s.end;
    }
    return std.math.add(u64, total, last - first);
}

pub fn budgets(a: std.mem.Allocator, wall: u64, functional: u64, envelope: u64, device: []const Device, host: []const Host) !Budget {
    if (functional > wall) return error.InvalidRoundBoundary;
    var spans: std.ArrayList(Interval) = .empty;
    defer spans.deinit(a);
    var out: Budget = .{ .complete_round_wall_ns = wall, .functional_return_ns = functional, .device_envelope_ns = envelope, .device_event_span_union_ns = 0, .kernel_event_span_sum_ns = 0, .device_envelope_gap_ns = 0, .target_verify_device_event_union_ns = 0, .proposal_device_event_union_ns = 0, .device_by_operation_union_ns = @splat(0), .device_by_phase_union_ns = @splat(0), .host_wait_union_ns = 0, .state_publication_union_ns = 0, .attribution_drain_union_ns = 0 };
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(a);
    for (device) |d| {
        if (d.end_ns < d.begin_ns or d.end_ns > envelope) return error.InvalidDeviceInterval;
        if ((try seen.getOrPut(a, d.correlation)).found_existing) return error.DuplicateCorrelation;
        try spans.append(a, .{ .begin = d.begin_ns, .end = d.end_ns });
        if (d.op == .other or d.op == .qmmf or d.op == .attention or d.op == .logits_sampling)
            out.kernel_event_span_sum_ns = try std.math.add(u64, out.kernel_event_span_sum_ns, d.end_ns - d.begin_ns);
    }
    out.device_event_span_union_ns = try unionNs(a, spans.items);
    if (out.device_event_span_union_ns > envelope) return error.InvalidDeviceEnvelope;
    out.device_envelope_gap_ns = envelope - out.device_event_span_union_ns;
    for (std.enums.values(Op), 0..) |op, i| {
        spans.clearRetainingCapacity();
        for (device) |d| if (d.op == op) {
            try spans.append(a, .{ .begin = d.begin_ns, .end = d.end_ns });
        };
        out.device_by_operation_union_ns[i] = try unionNs(a, spans.items);
    }
    for (std.enums.values(Phase), 0..) |phase, i| {
        spans.clearRetainingCapacity();
        for (device) |d| if (d.phase == phase) {
            try spans.append(a, .{ .begin = d.begin_ns, .end = d.end_ns });
        };
        out.device_by_phase_union_ns[i] = try unionNs(a, spans.items);
    }
    inline for (.{ "target_verify_device_event_union_ns", "proposal_device_event_union_ns" }, 0..) |field, which| {
        spans.clearRetainingCapacity();
        for (device) |d| {
            const verify = d.phase == .forward or d.phase == .land or d.phase == .token_readback or d.phase == .logits_readback;
            const proposal = d.phase == .proposal or d.phase == .proposal_hold or d.phase == .proposal_select or d.phase == .candidate_frame;
            if (if (which == 0) verify else proposal) try spans.append(a, .{ .begin = d.begin_ns, .end = d.end_ns });
        }
        @field(out, field) = try unionNs(a, spans.items);
    }
    for (host) |h| if (h.end_ns < h.begin_ns or h.end_ns > wall) return error.InvalidHostInterval;
    inline for (.{ "host_wait_union_ns", "state_publication_union_ns", "attribution_drain_union_ns" }, 0..) |field, which| {
        spans.clearRetainingCapacity();
        for (host) |h| {
            const take = switch (which) {
                0 => h.kind == .wait,
                1 => h.kind == .span and h.phase == .state_publication,
                2 => h.kind == .attribution_drain,
                else => unreachable,
            };
            if (take) try spans.append(a, .{ .begin = h.begin_ns, .end = h.end_ns });
        }
        @field(out, field) = try unionNs(a, spans.items);
    }
    return out;
}

fn isHash(s: []const u8) bool {
    if (s.len != 64) return false;
    for (s) |c| if (!std.ascii.isHex(c)) return false;
    return true;
}
pub fn validate(r: Record) !void {
    if (!std.mem.eql(u8, r.schema, schema_name) or r.round == 0 or !isHash(r.identity)) return error.InvalidIdentity;
    if (r.complete and (r.problem != null or r.requests.len == 0 or r.frames.len == 0 or r.windows.len == 0 or r.device.len == 0)) return error.IncompleteRound;
    for (r.requests) |q| if (!isHash(q.prompt_sha256)) return error.InvalidIdentity;
    for (r.frames) |frame| if (frame.live_rows == 0 or frame.windows == 0 or frame.live_rows > frame.shape_bucket) return error.InvalidFrame;
    for (r.windows) |w| {
        if (w.offered_rows == 0 or w.proposal_nodes != (if (std.mem.eql(u8, w.kind, "forced")) @as(usize, 0) else w.offered_rows - 1) or w.kept_path.len == 0 or w.kept_path[0] != 0 or w.connected_accepted_depth != (if (std.mem.eql(u8, w.kind, "forced")) @as(usize, 0) else w.kept_path.len - 1)) return error.InvalidWindow;
        var prev: u32 = 0;
        for (w.kept_path, 0..) |v, i| {
            if (v >= w.offered_rows or (i > 0 and v <= prev)) return error.InvalidWindow;
            prev = v;
        }
        if (w.committed_ids.len > w.kept_path.len or w.rolled_back_rows != w.offered_rows - w.kept_path.len) return error.InvalidWindow;
    }
}

pub fn json(a: std.mem.Allocator, r: Record) ![]u8 {
    try validate(r);
    return std.json.Stringify.valueAlloc(a, r, .{});
}
