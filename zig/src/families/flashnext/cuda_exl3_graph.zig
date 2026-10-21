//! CUDA graphs for an EXL3 Flash Next decode step with the expert offload pool (TF_FLASHNEXT_EXL3_GRAPH=1).
//!
//! With the pool planned on the device (cuda_exl3_dplan.zig) a decode step has no host wait between its first launch
//! and its logits, so the step is captured once and replayed: a decode round (Engine.round, one row a stream) and an
//! MTP verify window (Engine.forward with every row's logits, up to the planner's row bound) each take ONE graph
//! launch; at tp=2 the graph also holds the step's NCCL sums and the head's gathers (both ranks capture the same
//! calls in the same order, the rank engine keeps them in lockstep).
//!
//! What a replay may change and how (the graph-safety rules of exl3_graph.zig):
//!  - token ids, positions and the n-gram rows: pinned host bytes the graph's copies read, rewritten by `prepare`
//!    before each launch (upstream's cuda_round_graph.zig does the same for its tp round);
//!  - the experts each layer reads: the pool's pointer tables, rewritten in place by the device planner;
//!  - everything else (weights, scratch, caches, the sequence's buffers, the logits buffer) sits at fixed addresses,
//!    which the key holds: a different address, row count, attention chunk count or score-tile count captures anew.
//! A step the graph cannot take (more rows than the planner's bound, media) runs eagerly with the host cut.
//!
//! TF_FLASHNEXT_EXL3_DPLAN=1 alone plans on the device without capturing (the same kernels launched one by one).

const std = @import("std");
const cuda = @import("cuda");
const dplan = @import("cuda_exl3_dplan.zig");
const wt = @import("cuda_prompt_wt.zig");
const collective = @import("cuda_collective.zig");
const wire = @import("cuda_rank_wire.zig");
const profile_module = @import("exl3_graph_profile.zig");
const state_binding = @import("exl3_state_binding.zig");
const state_driver = @import("cuda_exl3_state_binding.zig");
pub var profile: profile_module.Profile(64, 1024) = .{};
fn profileEnabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_PROFILE") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}
pub const memory = @import("cuda_exl3_graph_memory.zig");

fn freeMemory(stream: cuda.Stream) !u64 {
    var free: usize = 0;
    var total: usize = 0;
    try stream.d.check(stream.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

pub fn uploadResident(exec: *cuda.graph.Exec, stream: cuda.Stream) !void {
    try exec.upload(stream);
    try stream.synchronize();
}

pub fn reserveMemory(stream: cuda.Stream, class: memory.Class, cohort: u64) !memory.Ticket {
    _ = try memory.startupReservation();
    while (true) {
        return memory.admitWithQuery(class, cohort, stream, freeMemory) catch |err| switch (err) {
            error.GraphBudgetBusy, error.GraphMemoryHeadroom, error.GraphEvictionRequired => {
                if (cohort != 0) return err;
                try stream.d.check(stream.d.api.cuCtxSynchronize(), "cuCtxSynchronize(graph retirement)");
                if (!evictOne()) return err;
                continue;
            },
            else => return err,
        };
    }
}

fn retirementBarrier(c: *collective.Collective, signature: u64, cohort: u64) !void {
    const decision = try c.graphAgree(signature, .{ .state = .ready, .cohort = cohort });
    if (decision.choice != .replay) return error.GraphCohortMismatch;
}

fn oldestCohort() u64 {
    var cohort: u64 = 0;
    var used: u64 = std.math.maxInt(u64);
    for (entries) |maybe| if (maybe) |entry| {
        if (entry.cohort != 0 and (entry.used < used or (entry.used == used and (cohort == 0 or entry.cohort < cohort)))) {
            cohort = entry.cohort;
            used = entry.used;
        }
    };
    return cohort;
}

test "exl3: TP eviction proposes the least recently used cohort and ignores local graphs" {
    entries = @splat(null);
    defer entries = @splat(null);
    const key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    entries[0] = .{ .key = key, .used = 30, .cohort = 3 };
    entries[1] = .{ .key = key, .used = 10, .cohort = 7 };
    entries[2] = .{ .key = key, .used = 0, .cohort = 0 };
    try std.testing.expectEqual(@as(u64, 7), oldestCohort());
    entries[0].?.used = 10;
    try std.testing.expectEqual(@as(u64, 3), oldestCohort());
}

fn retireCohort(cohort: u64) void {
    for (&entries) |*maybe| if (maybe.*) |*entry| {
        if (entry.cohort == cohort) {
            if (profileEnabled()) profile.evict(profile_module.Sample.fromKey(entry.key));
            drop(entry);
            maybe.* = null;
        }
    };
}

fn reserveTogether(c: *collective.Collective, stream: cuda.Stream, signature: u64, cohort: u64) !memory.Ticket {
    _ = try memory.startupReservation();
    var retired: usize = 0;
    while (true) {
        var ticket: ?memory.Ticket = memory.admitWithQuery(.decode, cohort, stream, freeMemory) catch |err| switch (err) {
            error.GraphBudgetBusy, error.GraphMemoryHeadroom, error.GraphEvictionRequired => null,
            else => return err,
        };
        errdefer if (ticket) |held| memory.release(held) catch unreachable;
        var free_slot = false;
        for (entries) |entry| free_slot = free_slot or entry == null;
        const agreement = try c.graphAgree(signature ^ 0x61646d6974, .{ .state = if (ticket != null and free_slot) .ready else .eager, .cohort = cohort });
        if (agreement.choice == .replay) return ticket.?;
        if (ticket) |held| {
            try memory.release(held);
            ticket = null;
        }
        const victim = try c.graphVictim(signature ^ 0x76696374696d, oldestCohort());
        if (victim == 0) {
            try stream.d.check(stream.d.api.cuCtxSynchronize(), "cuCtxSynchronize(TP local graph retirement)");
            try retirementBarrier(c, signature ^ 0x6c6f63616c7175, cohort);
            const reclaimed = evictOne();
            const any_reclaimed = try c.graphVictim(signature ^ 0x6c6f63616c646f, @intFromBool(reclaimed));
            if (any_reclaimed == 0) return error.GraphBudgetExceeded;
            continue;
        }
        if (retired >= entries.len * 2) return error.GraphBudgetExceeded;
        try stream.d.check(stream.d.api.cuCtxSynchronize(), "cuCtxSynchronize(TP graph retirement)");
        try retirementBarrier(c, signature ^ 0x71756965736365, victim);
        retireCohort(victim);
        try retirementBarrier(c, signature ^ 0x72657469726564, victim);
        retired += 1;
    }
}

fn evictOne() bool {
    var oldest: u64 = std.math.maxInt(u64);
    var decode: ?usize = null;
    var prefill: ?usize = null;
    for (entries, 0..) |maybe, index| if (maybe) |entry| {
        if (entry.cohort == 0 and entry.used < oldest) {
            oldest = entry.used;
            decode = index;
        }
    };
    for (pentries, 0..) |maybe, index| if (maybe) |entry| {
        if (entry.used < oldest) {
            oldest = entry.used;
            decode = null;
            prefill = index;
        }
    };
    if (decode) |index| {
        if (profileEnabled()) profile.evict(profile_module.Sample.fromKey(entries[index].?.key));
        drop(&entries[index].?);
        entries[index] = null;
        return true;
    }
    if (prefill) |index| {
        const entry = &pentries[index].?;
        dropPrefill(entry);
        pentries[index] = null;
        return true;
    }
    return @import("cuda_exl3_chain.zig").evictOne();
}

const dims = wt.dims;
const vocab = wt.vocab;
const att_rows = @import("cuda_batch_state.zig").att_rows;

/// TF_FLASHNEXT_EXL3_GRAPH=1: decode rounds and verify windows replay captured graphs.
pub fn enabled() bool {
    if (force) |f| return f == .graph;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// A test's override of the environment: the host cut, the device plan launched eagerly, or graphs.
pub const Mode = enum { off, dplan, graph };
pub var force: ?Mode = null;

const max_lanes = 32;

pub const Key = struct {
    /// 1: a decode round; 2: a window.
    kind: u8,
    rows: usize,
    /// Per lane (a window: lane 0): the sequence's buffers, its attention chunks and score tiles at the step's end.
    seq: [max_lanes][5]u64 = @splat(@splat(0)),
    paging: [max_lanes][6]u64 = @splat(@splat(0)),
    paging_logical: ?[max_lanes][2]u64 = null,
    chunks: [max_lanes]u32 = @splat(0),
    score_tiles: [max_lanes]u32 = @splat(0),
    /// The pass's scratch (its first buffer and its row capacity, which sets every other buffer's offset).
    scratch: u64,
    scratch_cap: usize = 0,
    scores: u64,
    output: u64,
    streams: u64 = 0,
    keep: u64 = 0,
    keep_nonstate: u64 = 0,
    /// The device planner's state (a new planner is new memory).
    plan: u64 = 0,

    fn seams(end: usize, chunks: *u32, tiles: *u32) void {
        chunks.* = @intCast((@min(end, wt.dense_keys) + 511) / 512);
        tiles.* = if (end / wt.index_ratio > wt.index_top) @intCast(((end + 3) / 4 + 63) / 64) else 0;
    }

    /// The DeltaNet state set is in the key: a drafting stream swaps it with its window's spare set (Draft.settle).
    fn seqOf(q: anytype) [5]u64 {
        return .{ if (q.kc.len > 0) q.kc[0].ptr else 0, if (q.conv.len > 0) q.conv[0].ptr else 0, q.tail.ptr, q.cap, if (q.state.len > 0) q.state[0].ptr else 0 };
    }
};

/// True while this module captures a step: the pass's pinned uploads become copy nodes (Uploads.copy).
pub var capturing: bool = false;

/// A copy node from pinned host bytes (the pass's staging): the graph reads them at every replay, as `prepare` left
/// them. A pageable source fails the capture (the driver refuses it), so stale custody copies cannot hide here.
pub fn pinnedCopy(s: cuda.Stream, dst: u64, bytes: []const u8) !void {
    if (bytes.len == 0) return;
    try s.d.check(s.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, s.handle), "cuMemcpyHtoDAsync");
}

/// `template`: the graph an update took this exec's parameters from, kept while the exec uses them (a tp graph's
/// NCCL resources are tied to the graph they were captured in).
const Entry = struct { key: Key, exec: ?cuda.graph.Exec = null, used: u64, template: ?cuda.graph.Graph = null, pending: ?*Pending = null, cohort: u64 = 0, admission: ?memory.Ticket = null, stream: ?cuda.Stream = null, binding: ?state_binding.Binding = null, poisoned: bool = false };

fn stateRebindEnabled() bool {
    const value = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_STATE_REBIND") orelse return false;
    return std.mem.eql(u8, std.mem.span(value), "1");
}

fn stateRequest(key: Key, pass: anytype) ?state_binding.Request {
    if (!stateRebindEnabled()) return null;
    return windowStateRequest(key, pass);
}

fn windowStateRequest(key: Key, pass: anytype) ?state_binding.Request {
    const Pass = @typeInfo(@TypeOf(pass)).pointer.child;
    if (comptime !@hasField(Pass, "lanes")) return null;
    if (key.kind != 2 or key.rows != 2 or pass.lanes.len != 0) return null;
    const gdn = @import("cuda_gdn.zig");
    var request: state_binding.Request = .{
        .width = key.rows,
        .chain = @intFromPtr(pass.k.chain.handle),
        .chain_to = @intFromPtr(pass.k.chain_to.handle),
        .geometry = .{ .grid = .{ gdn.nv, gdn.dv / 16, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = pass.s.gq, .gk = pass.s.gk, .gv = pass.s.gv, .ggb = pass.s.ggb, .rows = @intCast(key.rows), .ys = pass.s.ys },
    };
    for (pass.m.layers, 0..) |layer, index| {
        if (!layer.linear) continue;
        const linear = wt.linearIndex(index);
        if (request.count == state_binding.max_layers or linear >= pass.seq.state.len) return null;
        request.layers[request.count] = .{
            .layer = index,
            .state_in = pass.seq.state[linear].ptr,
            .state_out = if (pass.hooks.keep) |keep| keep.layers[index].state_out else 0,
        };
        if (request.layers[request.count].state_out == 0) return null;
        request.count += 1;
    }
    request.validate() catch return null;
    return request;
}

fn capturedBinding(key: Key, pass: anytype, graph: cuda.graph.Graph) ?state_binding.Binding {
    const request = stateRequest(key, pass) orelse return null;
    if (state_driver.nodeType(graph.d) == null) return null;
    return state_binding.Binding.init(@intFromPtr(graph.handle), request) catch null;
}

fn rebindCandidate(key: Key, request: state_binding.Request, tp: bool) ?*Entry {
    var candidate: ?*Entry = null;
    for (&entries) |*maybe| if (maybe.*) |*entry| {
        if (entry.poisoned or entry.pending != null or entry.exec == null or entry.template == null or (tp and entry.cohort == 0)) continue;
        const binding = entry.binding orelse continue;
        if (binding.poisoned or !std.meta.eql(state_binding.normalized(entry.key), state_binding.normalized(key)) or !binding.request.compatible(request)) continue;
        if (candidate == null or entry.cohort < candidate.?.cohort or (entry.cohort == candidate.?.cohort and entry.used > candidate.?.used)) candidate = entry;
    };
    return candidate;
}

fn prepareBinding(entry: *Entry) !?state_driver.Backend {
    const backend = state_driver.Backend.init(entry.template.?, entry.exec.?) orelse return null;
    entry.binding.?.load(backend) catch |err| switch (err) {
        error.UnsupportedBinding => {
            entry.binding.?.invalidate();
            entry.binding = null;
            return null;
        },
        else => return err,
    };
    return backend;
}

fn tryRebind(key: Key, pass: anytype) !bool {
    const request = stateRequest(key, pass) orelse return false;
    const entry = rebindCandidate(key, request, false) orelse {
        stats.rebind_declined += 1;
        return false;
    };
    const prepared = prepareBinding(entry) catch |err| {
        stats.rebind_failures += 1;
        return err;
    };
    const backend = prepared orelse {
        stats.rebind_declined += 1;
        return false;
    };
    stats.rebind_attempts += 1;
    entry.binding.?.apply(backend, request) catch |err| {
        entry.poisoned = true;
        stats.rebind_failures += 1;
        return err;
    };
    entry.key = key;
    entry.used = clock;
    stats.rebinds += 1;
    stats.replays += 1;
    note(key, .replay);
    try entry.exec.?.launchOn(pass.stream);
    return true;
}

fn tryRebindTogether(c: *collective.Collective, signature: u64, key: Key, pass: anytype) !bool {
    if (!stateRebindEnabled() or key.kind != 2 or key.rows != 2) return false;
    const request = stateRequest(key, pass);
    const candidate = if (request) |target| rebindCandidate(key, target, true) else null;
    var backend: ?state_driver.Backend = null;
    var failed = false;
    if (candidate) |entry| backend = prepareBinding(entry) catch blk: {
        stats.rebind_failures += 1;
        failed = true;
        break :blk null;
    };
    const selected = try c.graphAgree(signature ^ 0x737461746573656c, .{
        .state = if (failed) .failed else if (backend != null) .ready else .capture,
        .cohort = if (candidate) |entry| entry.cohort else 0,
    });
    if (selected.choice != .replay) {
        stats.rebind_declined += 1;
        return false;
    }
    const entry = candidate.?;
    stats.rebind_attempts += 1;
    entry.binding.?.applyTogether(backend.?, request.?, c, signature ^ 0x7374617465646f6e, selected.cohort) catch |err| {
        entry.poisoned = true;
        stats.rebind_failures += 1;
        return err;
    };
    entry.key = key;
    entry.used = clock;
    stats.rebinds += 1;
    stats.replays += 1;
    note(key, .replay);
    try entry.exec.?.launchOn(pass.stream);
    return true;
}

test "exl3: state binding normalized identity preserves all nonstate key guards" {
    const original: Key = .{ .kind = 2, .rows = 2, .scratch = 11, .scratch_cap = 16, .scores = 12, .output = 13, .keep = 14, .keep_nonstate = 15, .streams = 16, .plan = 17 };
    var flipped = original;
    flipped.seq[0][4] = 100;
    flipped.keep = 101;
    try std.testing.expect(!std.meta.eql(original, flipped));
    try std.testing.expectEqualDeep(state_binding.normalized(original), state_binding.normalized(flipped));
    inline for (@typeInfo(Key).@"struct".field_names) |field| {
        if (comptime !std.mem.eql(u8, field, "seq") and !std.mem.eql(u8, field, "paging") and
            !std.mem.eql(u8, field, "paging_logical") and
            !std.mem.eql(u8, field, "chunks") and !std.mem.eql(u8, field, "score_tiles") and !std.mem.eql(u8, field, "keep"))
        {
            var changed = original;
            @field(changed, field) += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
    }
    for (0..max_lanes) |lane| {
        for (0..5) |component| {
            if (lane == 0 and component == 4) continue;
            var changed = original;
            changed.seq[lane][component] += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
        for (0..6) |component| {
            var changed = original;
            changed.paging[lane][component] += 1;
            try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        }
        var changed = original;
        changed.chunks[lane] += 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        changed = original;
        changed.score_tiles[lane] += 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        changed = original;
        changed.paging_logical = @splat(@splat(0));
        changed.paging_logical.?[lane][0] = 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
        changed.paging_logical.?[lane][1] = 1;
        try std.testing.expect(!std.meta.eql(state_binding.normalized(original), state_binding.normalized(changed)));
    }
}

test "exl3: state binding candidate excludes pending poisoned retired and incompatible generations" {
    entries = @splat(null);
    defer entries = @splat(null);
    var request: state_binding.Request = .{
        .width = 2,
        .chain = 1,
        .chain_to = 2,
        .geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = 1, .gk = 2, .gv = 3, .ggb = 4, .rows = 2, .ys = 5 },
        .count = 1,
    };
    request.layers[0] = .{ .layer = 0, .state_in = 100, .state_out = 200 };
    const key: Key = .{ .kind = 2, .rows = 2, .scratch = 1, .scores = 2, .output = 3, .keep = 4, .keep_nonstate = 5 };
    entries[0] = .{
        .key = key,
        .used = 1,
        .cohort = 7,
        .exec = .{ .d = undefined, .handle = @ptrFromInt(1) },
        .template = .{ .d = undefined, .handle = @ptrFromInt(2) },
        .binding = try state_binding.Binding.init(2, request),
    };
    var target = request;
    std.mem.swap(u64, &target.layers[0].state_in, &target.layers[0].state_out);
    var flipped = key;
    flipped.seq[0][4] = 200;
    flipped.keep = 9;
    try std.testing.expect(find(flipped) == null);
    try std.testing.expect(rebindCandidate(flipped, target, true) == &entries[0].?);
    entries[0].?.pending = @ptrFromInt(@alignOf(Pending));
    try std.testing.expect(rebindCandidate(flipped, target, true) == null);
    entries[0].?.pending = null;
    entries[0].?.poisoned = true;
    try std.testing.expect(rebindCandidate(flipped, target, true) == null);
    entries[0].?.poisoned = false;
    target.layers[0].state_in = 300;
    try std.testing.expect(rebindCandidate(flipped, target, true) == null);
    target = request;
    flipped.output += 1;
    try std.testing.expect(rebindCandidate(flipped, target, true) == null);
    entries[0].?.binding.?.invalidate();
    try std.testing.expect(rebindCandidate(key, request, true) == null);
}

test "exl3: state binding helper contracts" {
    _ = @import("exl3_state_binding_test.zig");
    _ = state_driver;
}

test "exl3: state binding transaction uses real graphAgree before publication" {
    const Fake = struct {
        calls: usize = 0,
        fail: bool = false,

        pub fn set(self: *@This(), _: usize, _: usize, _: state_binding.Geometry, _: state_binding.Parameters) !void {
            self.calls += 1;
            if (self.fail and self.calls == 2) return error.SetterFailed;
        }
    };
    var request: state_binding.Request = .{
        .width = 2,
        .chain = 10,
        .chain_to = 11,
        .geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = 1, .gk = 2, .gv = 3, .ggb = 4, .rows = 2, .ys = 5 },
        .count = 2,
    };
    request.layers[0] = .{ .layer = 0, .state_in = 100, .state_out = 200 };
    request.layers[1] = .{ .layer = 2, .state_in = 300, .state_out = 400 };
    var target = request;
    for (target.layers[0..target.count]) |*mapping| std.mem.swap(u64, &mapping.state_in, &mapping.state_out);
    for (0..6) |scenario| {
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer for (pair) |descriptor| {
            _ = std.c.close(descriptor);
        };
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        for (pair) |descriptor| {
            const flags = std.c.fcntl(descriptor, std.c.F.GETFL);
            if (flags < 0 or std.c.fcntl(descriptor, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
        }
        const Peer = struct {
            fn run(link: wire.Link, current: usize) void {
                var control = GraphTestDriver.control(link);
                const selected = control.graphAgree(51, .{
                    .state = if (current == 2) .pending else if (current == 3) .capture else .ready,
                    .cohort = if (current == 1) 8 else 7,
                }) catch @panic("binding selection failed");
                if (selected.choice != .replay) return;
                const result = control.graphAgree(52, .{ .state = if (current == 5) .failed else .ready, .cohort = 7 });
                if (current == 4 or current == 5) {
                    if (result) |_| @panic("binding failure accepted") else |err| std.debug.assert(err == error.GraphInstantiationFailed);
                } else {
                    std.debug.assert((result catch @panic("binding commit failed")).choice == .replay);
                }
            }
        };
        const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, scenario });
        defer peer.join();
        var control = GraphTestDriver.control(.{ .fd = pair[0] });
        var state = try state_binding.Binding.init(42, request);
        for (request.layers[0..request.count], 0..) |mapping, index| {
            state.nodes[index] = index;
            state.parameters[index] = .{ .nonstate = request.nonstate, .state_in = mapping.state_in, .state_out = mapping.state_out };
        }
        state.loaded = true;
        var fake: Fake = .{ .fail = scenario == 4 };
        const selected = try control.graphAgree(51, .{ .state = .ready, .cohort = 7 });
        var published = false;
        if (selected.choice == .replay) {
            if (scenario == 4 or scenario == 5) {
                try std.testing.expectError(error.GraphInstantiationFailed, state.applyTogether(&fake, target, &control, 52, selected.cohort));
                try std.testing.expect(state.poisoned);
            } else {
                try state.applyTogether(&fake, target, &control, 52, selected.cohort);
                published = true;
            }
        }
        try std.testing.expectEqual(scenario == 0, published);
        try std.testing.expectEqual(@as(usize, if (scenario == 0 or scenario >= 4) 2 else 0), fake.calls);
    }
}

test "exl3: state binding current pass maps physical layers and declines in-place width1" {
    const batch = @import("cuda_batch.zig");
    var pass: batch.Pass(struct {}) = undefined;
    var kernels: batch.Kernels = undefined;
    kernels.chain = .{ .d = undefined, .handle = @ptrFromInt(10) };
    kernels.chain_to = .{ .d = undefined, .handle = @ptrFromInt(11) };
    var scratch: batch.Scratch = undefined;
    scratch.gq = 1;
    scratch.gk = 2;
    scratch.gv = 3;
    scratch.ggb = 4;
    scratch.ys = 5;
    var states: [wt.linear_n]cuda.DeviceBuffer = undefined;
    for (&states, 0..) |*state, index| state.ptr = 1000 + index;
    var layers: [wt.layers_n]@import("cuda_prompt_load.zig").Layer = undefined;
    var keep: @import("cuda_batch_state.zig").Keep = .{};
    for (&layers, 0..) |*layer, index| {
        layer.linear = wt.linearLayer(index);
        if (layer.linear) keep.layers[index].state_out = 2000 + wt.linearIndex(index);
    }
    pass.lanes = &.{};
    pass.k = &kernels;
    pass.s = &scratch;
    pass.seq.state = &states;
    pass.m.layers = &layers;
    pass.hooks.keep = &keep;
    var key: Key = .{ .kind = 2, .rows = 2, .scratch = 1, .scores = 2, .output = 3 };
    const request = windowStateRequest(key, &pass).?;
    try std.testing.expectEqual(wt.linear_n, request.count);
    for (request.layers[0..request.count], 0..) |mapping, index| {
        try std.testing.expectEqual(index, wt.linearIndex(mapping.layer));
        try std.testing.expectEqual(1000 + index, mapping.state_in);
        try std.testing.expectEqual(2000 + index, mapping.state_out);
    }
    key.rows = 1;
    try std.testing.expect(windowStateRequest(key, &pass) == null);
    key.rows = 3;
    try std.testing.expect(windowStateRequest(key, &pass) == null);
    key.rows = 2;
    keep.layers[0].state_out = 0;
    try std.testing.expect(windowStateRequest(key, &pass) == null);
    pass.hooks.keep = null;
    try std.testing.expect(windowStateRequest(key, &pass) == null);
}

/// A captured graph being instantiated on a worker thread (20-35 ms that the step need not wait for: it runs
/// eagerly meanwhile, and replays from the next sighting after the instantiation landed).
pub const Pending = struct {
    thread: std.Thread = undefined,
    done: std.atomic.Value(bool) = .init(false),
    graph: cuda.graph.Graph,
    exec: ?cuda.graph.Exec = null,
    ctx: ?*anyopaque,
    keep_template: bool,
    started_ns: u64 = 0,

    fn main(pd: *Pending) void {
        const api = ctxApi() orelse {
            pd.done.store(true, .release);
            return;
        };
        if (api.set(pd.ctx) != 0) {
            pd.done.store(true, .release);
            return;
        }
        pd.exec = pd.graph.instantiate() catch null;
        pd.done.store(true, .release);
    }
};

const CtxApi = struct { get: *const fn (*?*anyopaque) callconv(.c) c_int, set: *const fn (?*anyopaque) callconv(.c) c_int };
var ctx_api: ?CtxApi = null;
var ctx_lib: ?std.DynLib = null;
fn ctxApi() ?CtxApi {
    if (ctx_api) |a| return a;
    var lib = std.DynLib.open("libcuda.so.1") catch return null;
    const a: CtxApi = .{
        .get = lib.lookup(*const fn (*?*anyopaque) callconv(.c) c_int, "cuCtxGetCurrent") orelse return null,
        .set = lib.lookup(*const fn (?*anyopaque) callconv(.c) c_int, "cuCtxSetCurrent") orelse return null,
    };
    ctx_lib = lib;
    ctx_api = a;
    return a;
}

/// TF_FLASHNEXT_EXL3_GRAPH_ASYNC=0 instantiates on the step's thread.
pub fn asyncInstantiate() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_ASYNC") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Starts instantiating `g` on a worker thread (it owns `g` from here); null when that is not possible.
pub fn startInstantiate(g: cuda.graph.Graph, keep_template: bool) !?*Pending {
    if (!asyncInstantiate()) return null;
    const api = ctxApi() orelse return null;
    var ctx: ?*anyopaque = null;
    if (api.get(&ctx) != 0) return null;
    const pd = try std.heap.c_allocator.create(Pending);
    pd.* = .{ .graph = g, .ctx = ctx, .keep_template = keep_template, .started_ns = wire.now() };
    pd.thread = std.Thread.spawn(.{}, Pending.main, .{pd}) catch |err| {
        std.heap.c_allocator.destroy(pd);
        return err;
    };
    return pd;
}

/// The finished instantiation's exec (the graph template freed), or null while it runs; error when it failed.
pub fn takePending(pd: *Pending) !?cuda.graph.Exec {
    if (!pd.done.load(.acquire)) return null;
    pd.thread.join();
    defer std.heap.c_allocator.destroy(pd);
    pd.graph.deinit();
    return pd.exec orelse error.CudaFailed;
}

/// A pending entry whose instantiation finished becomes a usable graph (or is dropped if it failed); true if usable.
fn settle(x: *Entry) bool {
    if (x.poisoned) return false;
    const pd = x.pending orelse return x.exec != null;
    if (!pd.done.load(.acquire)) return false;
    pd.thread.join();
    x.pending = null;
    defer std.heap.c_allocator.destroy(pd);
    if (pd.exec) |e| {
        var exec = e;
        if (x.stream) |stream| uploadResident(&exec, stream) catch {
            exec.deinit();
            pd.graph.deinit();
            return false;
        };
        x.exec = exec;
        if (pd.keep_template) x.template = pd.graph else pd.graph.deinit();
        if (x.admission) |ticket| memory.installed(ticket) catch unreachable;
        return true;
    }
    pd.graph.deinit();
    return false;
}

fn drop(x: *Entry) void {
    if (x.binding) |*binding| binding.invalidate();
    if (x.stream) |stream| stream.synchronize() catch @panic("graph retirement synchronization failed");
    defer if (x.admission) |ticket| {
        memory.release(ticket) catch unreachable;
        x.admission = null;
    };
    if (x.pending) |pd| {
        pd.thread.join();
        if (pd.exec) |*e| e.deinit();
        pd.graph.deinit();
        std.heap.c_allocator.destroy(pd);
        return;
    }
    if (x.exec) |*e| e.deinit();
    if (x.template) |*g| g.deinit();
}

pub const Stats = struct {
    captures: u64 = 0,
    replays: u64 = 0,
    rebind_attempts: u64 = 0,
    rebinds: u64 = 0,
    rebind_failures: u64 = 0,
    rebind_declined: u64 = 0,
    /// Steps on the device plan launched one by one (TF_FLASHNEXT_EXL3_DPLAN=1 without graphs).
    eager_dplan: u64 = 0,
    /// Steps that left for the host cut (too many rows, media, no pools).
    declined: u64 = 0,
    capture_ns: u64 = 0,
    /// Captures that updated a cached graph of the same shape in place instead of instantiating one.
    updates: u64 = 0,
    /// Host time of `prepare` (ids, positions, n-gram rows) before a step's launch, and how many.
    prepare_ns: u64 = 0,
    prepares: u64 = 0,
};

pub var stats: Stats = .{};

/// Steps after which a graph nobody replayed counts as stale (another stream's, or a finished one's).
const stale_after: u64 = 512;

/// Graphs kept (LRU): a round and up to `max_rows` window sizes per stream, at two seams.
const cache_n = 128;
var entries: [cache_n]?Entry = @splat(null);
var clock: u64 = 0;
/// A capture that failed once turns graphs off for the process (the steps keep the device plan).
var broken: bool = false;

pub fn clear() void {
    clearPrefill();
    for (&entries) |*e| if (e.*) |*x| {
        drop(x);
        e.* = null;
    };
    seen = @splat(null);
}

fn references(key: Key, tail: u64) bool {
    if (tail == 0) return false;
    for (key.seq) |seq| if (seq[2] == tail) return true;
    return false;
}

pub fn releaseSeq(seq: anytype) void {
    const tail = seq.tail.ptr;
    if (tail == 0) return;
    for (&entries) |*entry| if (entry.*) |*x| if (references(x.key, tail)) {
        drop(x);
        entry.* = null;
    };
    for (&seen) |*entry| if (entry.*) |x| if (references(x.key, tail)) {
        entry.* = null;
    };
    for (&last_key) |*kinds| for (kinds) |*key| if (key.*) |k| if (references(k, tail)) {
        key.* = null;
    };
    for (&pentries) |*entry| if (entry.*) |*x| if (x.key.seq[2] == tail) {
        dropPrefill(x);
        entry.* = null;
    };
    for (&pseen) |*entry| if (entry.*) |x| if (x.key.seq[2] == tail) {
        entry.* = null;
    };
}

/// The engine's deinit: graphs and the device planner go.
pub fn releaseFor(owner: *const anyopaque) void {
    clear();
    dplan.releaseFor(owner);
}

fn find(key: Key) ?*Entry {
    for (&entries) |*e| if (e.*) |*x| if (std.meta.eql(x.key, key)) return x;
    return null;
}

fn slot() *?Entry {
    var best: usize = 0;
    var best_used: u64 = std.math.maxInt(u64);
    for (&entries, 0..) |*e, i| {
        if (e.* == null) return e;
        if (e.*.?.used < best_used) {
            best_used = e.*.?.used;
            best = i;
        }
    }
    if (profileEnabled()) profile.evict(profile_module.Sample.fromKey(entries[best].?.key));
    drop(&entries[best].?);
    entries[best] = null;
    return &entries[best];
}

/// The key without the addresses a cudaGraphExecUpdate may change: equal shapes capture the same topology.
fn sameShape(a: Key, b: Key) bool {
    return a.kind == b.kind and a.rows == b.rows and std.mem.eql(u32, &a.chunks, &b.chunks) and
        std.mem.eql(u32, &a.score_tiles, &b.score_tiles) and a.scratch_cap == b.scratch_cap and
        (a.keep != 0) == (b.keep != 0) and (a.streams != 0) == (b.streams != 0);
}

/// Keys seen without a graph, and how often (a key is captured at its `after`-th sighting; one-off shapes, e.g. a
/// request's rare window sizes, stay eager).
const Seen = struct { key: Key, n: u32, used: u64 };
var seen: [256]?Seen = @splat(null);

fn sighting(key: Key) u32 {
    var oldest: usize = 0;
    var oldest_used: u64 = std.math.maxInt(u64);
    for (&seen, 0..) |*e, i| {
        if (e.*) |*x| {
            if (std.meta.eql(x.key, key)) {
                x.n += 1;
                x.used = clock;
                return x.n;
            }
            if (x.used < oldest_used) {
                oldest_used = x.used;
                oldest = i;
            }
        } else {
            oldest = i;
            oldest_used = 0;
        }
    }
    seen[oldest] = .{ .key = key, .n = 1, .used = clock };
    return 1;
}

/// TF_FLASHNEXT_EXL3_GRAPH_TP_UPDATE=0: tp graphs are always instantiated; default: updated in place like tp=1's,
/// their captured template kept alive with the exec (NCCL's captured resources belong to it).
fn tpUpdate() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_TP_UPDATE") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// TF_FLASHNEXT_EXL3_GRAPH_AFTER: the sighting of a key that captures it (default 2; 1 captures at once). With a
/// blocking communicator a capture costs 3-7 ms and an in-place update 1-3 ms, and recycled stream buffers
/// (cuda_exl3_recycle.zig) keep keys stable across requests, so a key is captured once and replayed from then on.
fn after() u32 {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_AFTER") orelse return 2;
    return @max(1, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2);
}

/// Captures `pass.chunk(ids, pos0, null)` (prepared) under `key`, or replays the graph captured under it. A key not
/// yet seen `after()` times runs eagerly (prepared, uncaptured). A new key whose shape matches a cached graph's
/// updates that graph in place (cudaGraphExecUpdate) instead of instantiating one, except with a collective (NCCL's
/// captured resources belong to the graph they were captured in, so a tp graph is always instantiated).
/// Coverage by step kind (0 round, 1 window) and rows (1..16): replays, eager first sightings, captures that
/// instantiated, captures that updated in place. TF_FLASHNEXT_EXL3_GRAPH_LOG=N logs it every N steps (default 256).
pub const Outcome = enum { replay, eager, instantiated, updated };
pub var coverage: [2][17][4]u32 = @splat(@splat(@splat(0)));
var steps_seen: u64 = 0;

fn note(key: Key, o: Outcome) void {
    coverage[key.kind - 1][@min(key.rows, 16)][@backingInt(o)] += 1;
    steps_seen += 1;
    const every: u64 = if (std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_LOG")) |v| std.fmt.parseInt(u64, std.mem.span(v), 10) catch 256 else 256;
    if (every == 0 or steps_seen % every != 0) return;
    if (profileEnabled()) logProfile();
    std.log.info("EXL3 state rebind attempts/succeeded/failed/declined {d}/{d}/{d}/{d}", .{ stats.rebind_attempts, stats.rebinds, stats.rebind_failures, stats.rebind_declined });
    var buf: [1024]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    for (0..2) |k| for (1..17) |r| {
        const c = coverage[k][r];
        if (c[0] + c[1] + c[2] + c[3] == 0) continue;
        w.print(" {s}{d}:{d}/{d}/{d}/{d}", .{ if (k == 0) "round" else "win", r, c[0], c[1], c[2], c[3] }) catch break;
    };
    const rs = @import("cuda_exl3_recycle.zig").stats;
    const cs = @import("cuda_exl3_chain.zig").stats;
    std.log.info("flashnext exl3 graph coverage after {d} steps (replay/eager/instantiated/updated):{s}; recycled seqs {d}/{d} drafts {d}/{d} (reused/kept); draft chains {d}: {d} replays, {d} captures ({d} updates), {d} eager; decode prepare {d} ms/{d}, capture {d} ms; prefill graphs {d}/{d}/{d}/{d}, capture {d} ms", .{ steps_seen, w.buffered(), rs.seq_reused, rs.seq_kept, rs.draft_reused, rs.draft_kept, cs.chains, cs.replays, cs.captures, cs.updates, cs.eager, stats.prepare_ns / std.time.ns_per_ms, stats.prepares, stats.capture_ns / std.time.ns_per_ms, prefill_stats.replays, prefill_stats.eager, prefill_stats.captures, prefill_stats.updates, prefill_stats.capture_ns / std.time.ns_per_ms });
}

fn logProfile() void {
    std.log.info("EXL3 graph profile: misses mean absent keys, not missed replays; present-unready means pending or no exec at entry, before settlement; readiness is not replay proof; use runtime replay counters; fingerprint estimates, not allocation sizes; lane/key overflow {d}/{d}; changes overlap and compare within width except width transitions", .{ profile.lane_overflow, profile.key_overflow });
    for (profile.counters, 0..) |kind, kind_index| for (kind, 0..) |counts, width| {
        if (counts.accesses == 0) continue;
        std.log.info("EXL3 graph profile {s}{d}: accesses/key-presence-misses/present-unready/new/returning-key-misses/evicted/post-eviction-key-misses {d}/{d}/{d}/{d}/{d}/{d}/{d}; width/state/keep/seam/paging/allocation changes {d}/{d}/{d}/{d}/{d}/{d}; orientation overflow {d}", .{ if (kind_index == 0) "round" else "win", width, counts.accesses, counts.misses, counts.present_unready, counts.first_keys, counts.returning_misses, counts.pressure_evictions, counts.post_eviction_misses, counts.width_changes, counts.state_changes, counts.keep_changes, counts.seam_changes, counts.paging_changes, counts.allocation_changes, counts.orientation_overflow });
    };
    for (profile.lanes, 0..) |maybe, index| if (maybe) |lane| {
        std.log.info("EXL3 graph profile lane {d}: width mask {x}; accesses/key-presence-misses/present-unready/new/returning-key-misses/evicted {d}/{d}/{d}/{d}/{d}/{d}; orientation A/B/other accesses {d}/{d}/{d}", .{ index, lane.width_mask, lane.counters.accesses, lane.counters.misses, lane.counters.present_unready, lane.counters.first_keys, lane.counters.returning_misses, lane.counters.pressure_evictions, lane.orientation_accesses[0], lane.orientation_accesses[1], lane.orientation_accesses[2] });
        std.log.info("EXL3 graph profile lane {d}: new fingerprints by width 0..16 {any}", .{ index, lane.new_keys_by_width });
    };
}

/// TF_FLASHNEXT_EXL3_GRAPH_DEBUG=1: which key fields changed since the last step of the same kind and rows.
var last_key: [2][17]?Key = @splat(@splat(null));
fn debugKey(key: Key) void {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_DEBUG") orelse return;
    if (!std.mem.eql(u8, std.mem.span(v), "1")) return;
    const lk = &last_key[key.kind - 1][@min(key.rows, 16)];
    defer lk.* = key;
    const prev = lk.* orelse return;
    var buf: [256]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    inline for (@typeInfo(Key).@"struct".field_names) |f| {
        if (!std.meta.eql(@field(prev, f), @field(key, f))) w.print(" {s}", .{f}) catch {};
    }
    std.log.info("flashnext exl3 graph key ({s} of {d} rows) changed:{s} keep {x} seq {x} {x} {x} {d} {x} out {x} streams {x} scratch {x} scores {x}", .{ if (key.kind == 1) "round" else "window", key.rows, w.buffered(), key.keep, key.seq[0][0], key.seq[0][1], key.seq[0][2], key.seq[0][3], key.seq[0][4], key.output, key.streams, key.scratch, key.scores });
}

fn run(key: Key, pass: anytype, ids: []const u32, pos0: usize) !void {
    if (profileEnabled()) {
        const presence: profile_module.Presence = if (find(key)) |entry|
            if (entry.pending != null or entry.exec == null) .present_unready else .present_ready
        else
            .absent;
        profile.observeEntry(profile_module.Sample.fromKey(key), presence);
    }
    debugKey(key);
    clock += 1;
    if (pass.m.collective) |c| return runTogether(c, key, pass, ids, pos0);
    if (find(key)) |e| {
        e.used = clock;
        if (!settle(e)) {
            if (e.pending == null) {
                // the instantiation failed: forget the key, run eagerly
                for (&entries) |*en| if (en.*) |*x| if (x == e) {
                    drop(x);
                    en.* = null;
                };
                return error.CudaFailed;
            }
            stats.eager_dplan += 1;
            note(key, .eager);
            pass.prepared = true;
            return pass.chunk(ids, pos0, null);
        }
        stats.replays += 1;
        note(key, .replay);
        return e.exec.?.launchOn(pass.stream);
    }
    if (try tryRebind(key, pass)) return;
    if (sighting(key) < after()) {
        stats.eager_dplan += 1;
        note(key, .eager);
        pass.prepared = true;
        return pass.chunk(ids, pos0, null);
    }
    const t0 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    try pass.stream.synchronize();
    const admission = try reserveMemory(pass.stream, .decode, 0);
    var admitted = true;
    defer if (admitted) memory.release(admission) catch unreachable;
    try cuda.graph.beginCapture(pass.stream, .thread_local);
    pass.prepared = true;
    capturing = true;
    defer capturing = false;
    pass.chunk(ids, pos0, null) catch |err| {
        if (cuda.graph.endCapture(pass.stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return err;
    };
    var graph = try cuda.graph.endCapture(pass.stream);
    var kept = false;
    defer if (!kept) graph.deinit();
    const t1 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    var how: []const u8 = "instantiated";
    const tp = pass.m.collective != null;
    const at: *Entry = blk: {
        // a same-shape graph is updated in place only when it is stale (its key not used for a while, e.g. a finished
        // stream's) or the cache is full: two live variants of one shape (a window's state ping-pong) keep two graphs
        var free_slot = false;
        for (&entries) |*e| free_slot = free_slot or e.* == null;
        if ((!tp or tpUpdate()) and capturedBinding(key, pass, graph) == null) {
            for (&entries) |*e| if (e.*) |*x| if (!x.poisoned and x.exec != null and x.pending == null and sameShape(x.key, key) and (!free_slot or clock - x.used > stale_after)) {
                // every earlier launch finished (each step ends on the host), so the old parameters are free to go
                const r = try x.exec.?.update(graph);
                if (r == .success) {
                    if (x.binding) |*binding| binding.invalidate();
                    x.binding = null;
                    try uploadResident(&x.exec.?, pass.stream);
                    x.key = key;
                    x.used = clock;
                    if (x.template) |*old| old.deinit();
                    x.template = null;
                    if (tp) {
                        x.template = graph;
                        kept = true;
                    }
                    stats.updates += 1;
                    note(key, .updated);
                    how = "updated";
                    break :blk x;
                }
            };
        }
        if (asyncInstantiate()) if (ctxApi()) |api| {
            var ctx: ?*anyopaque = null;
            if (api.get(&ctx) == 0) {
                const sl = slot();
                const pd = try std.heap.c_allocator.create(Pending);
                const binding = capturedBinding(key, pass, graph);
                pd.* = .{ .graph = graph, .ctx = ctx, .keep_template = tp or binding != null, .started_ns = wire.now() };
                kept = true;
                pd.thread = std.Thread.spawn(.{}, Pending.main, .{pd}) catch |err| {
                    kept = false;
                    std.heap.c_allocator.destroy(pd);
                    return err;
                };
                sl.* = .{ .key = key, .used = clock, .pending = pd, .admission = admission, .stream = pass.stream, .binding = binding };
                admitted = false;
                note(key, .instantiated);
                stats.captures += 1;
                // this step runs eagerly while the graph instantiates
                capturing = false;
                pass.prepared = true;
                return pass.chunk(ids, pos0, null);
            }
        };
        const sl = slot();
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try uploadResident(&exec, pass.stream);
        try memory.installed(admission);
        sl.* = .{ .key = key, .exec = exec, .used = clock, .admission = admission, .stream = pass.stream };
        sl.*.?.binding = capturedBinding(key, pass, graph);
        if (sl.*.?.binding != null) {
            sl.*.?.template = graph;
            kept = true;
        }
        admitted = false;
        note(key, .instantiated);
        break :blk &sl.*.?;
    };
    stats.captures += 1;
    const t2 = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    stats.capture_ns += @intCast(t2 - t0);
    std.log.info("flashnext exl3 graph: captured a {s} of {d} rows in {d:.1} ms + {s} in {d:.1} ms ({d} captures, {d} updates, {d} replays, {d} eager, {d} declined steps, {d} residency pulls so far)", .{ if (key.kind == 1) "decode round" else "window", key.rows, @as(f64, @floatFromInt(t1 - t0)) / 1e6, how, @as(f64, @floatFromInt(t2 - t1)) / 1e6, stats.captures, stats.updates, stats.replays, stats.eager_dplan, stats.declined, if (dplan.current()) |dp| dp.pulls else 0 });
    try at.exec.?.launchOn(pass.stream);
}

fn graphSignature(key: Key, pos0: usize) u64 {
    var hash = std.hash.Wyhash.init(0);
    std.hash.autoHash(&hash, .{ key.kind, key.rows, key.chunks, key.score_tiles, key.scratch_cap, key.keep != 0, key.streams != 0, pos0 });
    std.hash.autoHash(&hash, stateRebindEnabled());
    for (key.seq, key.paging, 0..) |seq, paging, index| {
        const identity: [2]u64 = if (key.paging_logical) |logical| logical[index] else .{ paging[4], paging[5] };
        std.hash.autoHash(&hash, .{ seq[3], identity[0], identity[1] });
    }
    return hash.final();
}

fn graphVote(entry: ?*Entry, timeout_ms: u64) collective.GraphVote {
    const x = entry orelse return .{ .state = .capture };
    if (settle(x)) return .{ .state = .ready, .cohort = x.cohort };
    if (x.pending) |pd| {
        const elapsed = wire.now() -| pd.started_ns;
        return .{ .state = if (elapsed >= timeout_ms *| std.time.ns_per_ms) .timed_out else .pending, .cohort = x.cohort };
    }
    return .{ .state = .failed, .cohort = x.cohort };
}

fn runTogether(c: *collective.Collective, key: Key, pass: anytype, ids: []const u32, pos0: usize) !void {
    errdefer c.failGraph();
    const signature = graphSignature(key, pos0);
    var entry = find(key);
    if (entry) |x| x.used = clock;
    const vote = if (entry != null) graphVote(entry, c.graphTimeoutMs()) else collective.GraphVote{
        .state = if (stateRequest(key, pass)) |request|
            if (rebindCandidate(key, request, true) != null) .capture else if (sighting(key) < after()) .eager else .capture
        else if (sighting(key) < after()) .eager else .capture,
    };
    const decision = try c.graphAgree(signature, vote);
    switch (decision.choice) {
        .eager => {
            stats.eager_dplan += 1;
            note(key, .eager);
            pass.prepared = true;
            return pass.chunk(ids, pos0, null);
        },
        .replay => {
            stats.replays += 1;
            note(key, .replay);
            return entry.?.exec.?.launchOn(pass.stream);
        },
        .capture => {},
    }
    if (try tryRebindTogether(c, signature, key, pass)) return;
    const started = std.Io.Clock.awake.now(pass.io).toNanoseconds();
    try pass.stream.synchronize();
    const admission = try reserveTogether(c, pass.stream, signature, decision.cohort);
    entry = find(key);
    var admitted = true;
    defer if (admitted) memory.release(admission) catch unreachable;
    try cuda.graph.beginCapture(pass.stream, .thread_local);
    pass.prepared = true;
    capturing = true;
    pass.chunk(ids, pos0, null) catch |err| {
        capturing = false;
        if (cuda.graph.endCapture(pass.stream)) |g| {
            var discarded = g;
            discarded.deinit();
        } else |_| {}
        return err;
    };
    capturing = false;
    var graph = try cuda.graph.endCapture(pass.stream);
    var kept = false;
    defer if (!kept) graph.deinit();
    var updated = false;
    var candidate: ?*Entry = null;
    if (tpUpdate() and capturedBinding(key, pass, graph) == null) for (&entries) |*maybe| if (maybe.*) |*current| {
        if (!current.poisoned and current.cohort != 0 and current.exec != null and current.pending == null and sameShape(current.key, key) and
            (std.meta.eql(current.key, key) or clock - current.used > stale_after))
        {
            if (candidate == null or current.used < candidate.?.used) candidate = current;
        }
    };
    const update_agreement = try c.graphAgree(signature ^ 0x757064617465, if (candidate) |current|
        .{ .state = .ready, .cohort = current.cohort }
    else
        .{ .state = .capture });
    if (update_agreement.choice == .replay) {
        const current = candidate.?;
        const result = try current.exec.?.update(graph);
        if (result == .success) {
            if (current.binding) |*binding| binding.invalidate();
            current.binding = null;
            if (current.template) |*old| old.deinit();
            current.template = graph;
            kept = true;
            updated = true;
        }
        const result_agreement = try c.graphAgree(signature ^ 0x75706461746564, .{ .state = if (updated) .ready else .capture, .cohort = decision.cohort });
        if (result_agreement.choice == .replay) {
            if (current.admission) |old| try memory.release(old);
            current.admission = admission;
            admitted = false;
            try uploadResident(&current.exec.?, pass.stream);
            try memory.installed(admission);
            current.key = key;
            current.used = clock;
            current.cohort = decision.cohort;
            current.stream = pass.stream;
            entry = current;
        } else {
            if (updated) {
                current.template = null;
                kept = false;
            }
            drop(current);
            for (&entries) |*slot_entry| if (slot_entry.*) |*cached| if (cached == current) {
                slot_entry.* = null;
            };
            entry = find(key);
            updated = false;
            try retirementBarrier(c, signature ^ 0x72656275696c64, decision.cohort);
        }
    }
    if (!updated) {
        const target = if (entry) |old| blk: {
            drop(old);
            for (&entries) |*slot_entry| if (slot_entry.*) |*x| if (x == old) {
                slot_entry.* = null;
                break :blk slot_entry;
            };
            unreachable;
        } else blk: {
            for (&entries) |*slot_entry| if (slot_entry.* == null) break :blk slot_entry;
            return error.GraphBudgetBusy;
        };
        if (try startInstantiate(graph, true)) |pending| {
            target.* = .{ .key = key, .used = clock, .pending = pending, .cohort = decision.cohort, .admission = admission, .stream = pass.stream, .binding = capturedBinding(key, pass, graph) };
        } else {
            var exec = try graph.instantiate();
            errdefer exec.deinit();
            try uploadResident(&exec, pass.stream);
            try memory.installed(admission);
            target.* = .{ .key = key, .exec = exec, .used = clock, .template = graph, .cohort = decision.cohort, .admission = admission, .stream = pass.stream, .binding = capturedBinding(key, pass, graph) };
        }
        admitted = false;
        kept = true;
        entry = &target.*.?;
    }
    stats.captures += 1;
    if (updated) stats.updates += 1;
    note(key, if (updated) .updated else .instantiated);
    stats.capture_ns += @intCast(std.Io.Clock.awake.now(pass.io).toNanoseconds() - started);
    const installed = try c.graphAgree(signature ^ 0x696e7374616c6c, graphVote(entry, c.graphTimeoutMs()));
    switch (installed.choice) {
        .replay => try entry.?.exec.?.launchOn(pass.stream),
        .eager => {
            stats.eager_dplan += 1;
            pass.prepared = true;
            try pass.chunk(ids, pos0, null);
        },
        .capture => return error.GraphRendezvousMismatch,
    }
}

/// A round's host inputs (what chunk's stage and gather would write), outside capture and replay.
fn prepareRound(pass: anytype, ids: []const u32) !void {
    const s = pass.s;
    try s.ready();
    for (s.hostIds(ids.len), ids) |*out, token| out.* = @intCast(token);
    const positions = s.hostPos(ids.len * 4);
    @memset(positions, 0);
    for (pass.lanes, 0..) |lane, i| {
        positions[i * 4] = @intCast(lane.pos);
        try pass.m.ngram.rows(lane.seq.history[0..], ids[i..][0..1], s.hostEmb(ids.len)[i * dims ..][0..dims], s.hostXs(ids.len)[i * (dims / 32) ..][0 .. dims / 32]);
    }
}

/// A window's host inputs: ids, one position a block of attention rows, the n-gram rows after the sequence's history.
fn prepareWindow(pass: anytype, ids: []const u32, pos0: usize) !void {
    const s = pass.s;
    const n = ids.len;
    try s.ready();
    for (s.hostIds(n), ids) |*out, token| out.* = @intCast(token);
    const blocks = (n + att_rows - 1) / att_rows;
    const positions = s.hostPos(blocks * 4);
    @memset(positions, 0);
    for (0..blocks) |b| positions[4 * b] = @intCast(pos0 + b * att_rows);
    try pass.m.ngram.rows(pass.seq.history[0..], ids, s.hostEmb(n), s.hostXs(n));
}

/// The device planner for `e`, or null when its decode steps keep the host cut.
fn planner(e: anytype, rows: usize) !?*dplan.DevPlan {
    if (!dplan.enabled()) return null;
    const dp = (try dplan.forLayers(e.gpa, e.driver, e, e.layers[0..e.n_layers])) orelse return null;
    if (rows > dp.max_rows) {
        stats.declined += 1;
        return null;
    }
    return dp;
}

/// Engine.round's hook: true when this module ran the round (device plan, captured or not); false leaves it to the
/// engine. `p` is the round's pass (lanes set), `out` its logits [rows, vocab].
pub fn round(e: anytype, p: anytype, seqs: anytype, ids: []const u32, scores: u64, out: u64) !bool {
    for (seqs) |q| if (q.media != null) return false;
    if (seqs.len > max_lanes) return false;
    for (seqs) |q| if (q.paged) |pg| try pg.validateSharedLeaseOn(p.stream);
    const dp = (try planner(e, ids.len)) orelse return false;
    try dp.toDevice(p.stream);
    dplan.active = dp;
    defer dplan.active = null;
    if (enabled() and broken) return error.GraphCaptureFailed;
    if (!enabled()) {
        stats.eager_dplan += 1;
        try p.chunk(ids, 0, null);
        return true;
    }
    for (seqs) |q| if (q.paged) |pg| for (0..wt.attn_n) |ai| try pg.validateRange(ai, q.fed, 1);
    defer for (seqs) |q| if (q.paged) |pg| {
        pg.graph_prepared = false;
    };
    for (seqs) |q| if (q.paged) |pg| try pg.prepareGraph(p.stream, q.fed, 1);
    var key: Key = .{ .kind = 1, .rows = ids.len, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = scores, .output = out, .plan = dp.state.ptr };
    for (seqs, 0..) |q, i| {
        key.seq[i] = Key.seqOf(q);
        if (q.paged) |pg| {
            key.paging[i] = pg.graphKey();
            if (key.paging_logical == null) key.paging_logical = @splat(@splat(0));
            key.paging_logical.?[i] = pg.logicalGraphKey();
        }
        Key.seams(q.fed + 1, &key.chunks[i], &key.score_tiles[i]);
    }
    const tp0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
    try prepareRound(p, ids);
    stats.prepare_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - tp0);
    stats.prepares += 1;
    run(key, p, ids, 0) catch |err| {
        if (p.m.collective) |c| {
            c.failGraph();
            return err;
        }
        broken = true;
        return err;
    };
    return true;
}

/// Engine.forward's hook for a window: every row's logits into hooks.logits_out (an MTP verify window, or a one-row
/// step), at most the planner's rows. True when this module ran it (seq.fed and seq.logits advanced as forward does).
pub fn forward(e: anytype, p: anytype, seq: anytype, ids: []const u32) !bool {
    const hooks = p.hooks;
    if (hooks.logits != .all or hooks.logits_out == 0 or seq.media != null or ids.len == 0 or ids.len > att_rows) return false;
    if (seq.fed + ids.len > seq.cap or ids.len > p.s.cap) return false;
    if (seq.paged) |pg| try pg.validateSharedLeaseOn(p.stream);
    const dp = (try planner(e, ids.len)) orelse return false;
    try dp.toDevice(p.stream);
    dplan.active = dp;
    defer dplan.active = null;
    const n = ids.len;
    if (enabled() and broken) return error.GraphCaptureFailed;
    if (!enabled()) {
        stats.eager_dplan += 1;
        try p.chunk(ids, seq.fed, null);
    } else {
        defer if (seq.paged) |pg| {
            pg.graph_prepared = false;
        };
        if (seq.paged) |pg| try pg.prepareGraph(p.stream, seq.fed, n);
        var key: Key = .{ .kind = 2, .rows = n, .scratch = p.s.ids, .scratch_cap = p.s.cap, .scores = p.seq.scores, .output = hooks.logits_out, .streams = hooks.streams, .plan = dp.state.ptr };
        key.seq[0] = Key.seqOf(seq);
        if (seq.paged) |pg| {
            key.paging[0] = pg.graphKey();
            key.paging_logical = @splat(@splat(0));
            key.paging_logical.?[0] = pg.logicalGraphKey();
        }
        Key.seams(seq.fed + n, &key.chunks[0], &key.score_tiles[0]);
        if (hooks.keep) |kp| {
            key.keep = std.hash.Wyhash.hash(1, std.mem.asBytes(kp));
            if (stateRebindEnabled()) key.keep_nonstate = state_binding.keepHash(kp.*);
        }
        const tp0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
        try prepareWindow(p, ids, seq.fed);
        stats.prepare_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - tp0);
        stats.prepares += 1;
        run(key, p, ids, seq.fed) catch |err| {
            if (p.m.collective) |c| {
                c.failGraph();
                return err;
            }
            broken = true;
            return err;
        };
    }
    if (hooks.output_tail) |tail| try tail.enqueue(tail.ptr, p.stream);
    // the pass's head(.all) copies the last row's scores to the host after the step (outside the graph)
    if (p.m.collective) |c| try c.finish(p.stream) else try p.stream.synchronize();
    const d = p.stream.d;
    try d.check(d.api.cuMemcpyDtoH_v2(seq.logits.ptr, hooks.logits_out + (n - 1) * vocab * 2, vocab * 2), "cuMemcpyDtoH");
    seq.fed += n;
    return true;
}

// ---- prefill: one layer's sub-blocks of a deep chunk -------------------------------------------------------------
//
// The expert-major prefill (cuda_exl3_prefill.zig) runs each layer as its dense sub-blocks (HC read-outs, DeltaNet or
// attention, write-backs, router and top-k over every sub-block of the deep chunk) and then the layer's experts, whose
// passes the host plans from the layer's picks (the ring, the members, the copies): that half stays eager. The dense
// half has no host dependency and is captured per layer: a DeltaNet layer's graph depends only on the chunk's shape
// (rows, sub-block size) and replays for every later chunk of that shape; an attention layer's launch geometry follows
// the chunk's position, so each chunk re-captures it and updates the cached graph in place when the topology holds.

pub const Prefill = enum { eager, replayed, capturing };

const PKey = struct {
    layer: usize,
    rows: usize,
    sub: usize,
    /// The chunk's first position (attention layers only; 0 for DeltaNet layers).
    pos0: usize,
    deep: u64,
    base: u64,
    base_cap: usize,
    seq: [5]u64,
    paging: [6]u64,
    wait_count: usize,
    wait_key: u64,
    scores: u64,
    tiles: bool,
};

const PEntry = struct { key: PKey, exec: cuda.graph.Exec, used: u64, admission: ?memory.Ticket = null, stream: ?cuda.Stream = null };

fn dropPrefill(entry: *PEntry) void {
    if (entry.stream) |stream| stream.synchronize() catch @panic("prefill retirement synchronization failed");
    entry.exec.deinit();
    if (entry.admission) |ticket| memory.release(ticket) catch unreachable;
}
const pcache_n = 112;
var pentries: [pcache_n]?PEntry = @splat(null);
var pseen: [128]?struct { key: PKey, n: u32 } = @splat(null);
var pseen_at: usize = 0;
var pcur: ?PKey = null;
var pcur_admission: ?memory.Ticket = null;

pub var prefill_stats: struct { captures: u64 = 0, updates: u64 = 0, replays: u64 = 0, eager: u64 = 0, admission_misses: u64 = 0, capture_ns: u64 = 0 } = .{};

fn prefillAdmissionMiss() Prefill {
    prefill_stats.eager += 1;
    prefill_stats.admission_misses += 1;
    const misses = prefill_stats.admission_misses;
    if (misses == 1 or misses % 256 == 0) std.log.info("flashnext exl3 prefill graph budget admission declines: {d}; retained cache preserved, layer runs eagerly", .{misses});
    return .eager;
}

/// TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1 captures prefill layers under TF_FLASHNEXT_EXL3_GRAPH=1 (default off pending
/// measurements). TP=2 graphs capture each incoming sub-block reduction as an external event wait.
fn prefillEnabled() bool {
    if (!enabled() or broken) return false;
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_PREFILL") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

fn pshape(k: PKey) PKey {
    var s = k;
    s.pos0 = 0;
    s.paging[5] = 0;
    return s;
}

test "prefill graph update candidates tolerate dynamic position and page-window origins only" {
    const base: PKey = .{ .layer = 7, .rows = 4096, .sub = 256, .pos0 = 4096, .deep = 1, .base = 2, .base_cap = 4096, .seq = .{ 3, 4, 5, 4096, 6 }, .paging = .{ 7, 8, 9, 10, 11, 12 }, .wait_count = 0, .wait_key = 0, .scores = 13, .tiles = true };
    var moved = base;
    moved.pos0 = 8192;
    moved.paging[5] = 14;
    try std.testing.expect(!std.meta.eql(base, moved));
    try std.testing.expect(std.meta.eql(pshape(base), pshape(moved)));

    moved.seq[0] += 1;
    try std.testing.expect(!std.meta.eql(pshape(base), pshape(moved)));
    moved = base;
    moved.paging[0] += 1;
    try std.testing.expect(!std.meta.eql(pshape(base), pshape(moved)));
    moved = base;
    moved.rows += 1;
    try std.testing.expect(!std.meta.eql(pshape(base), pshape(moved)));
    moved = base;
    moved.wait_key += 1;
    try std.testing.expect(!std.meta.eql(pshape(base), pshape(moved)));
}

fn psighting(k: PKey) u32 {
    const s = pshape(k);
    for (&pseen) |*e| if (e.*) |*x| if (std.meta.eql(x.key, s)) {
        x.n += 1;
        return x.n;
    };
    pseen[pseen_at % pseen.len] = .{ .key = s, .n = 1 };
    pseen_at += 1;
    return 1;
}

/// TF_FLASHNEXT_EXL3_GRAPH_PREFILL_AFTER: the sighting of a layer shape that captures it (default 2: a chunk takes
/// seconds, a capture milliseconds).
fn prefillAfter() u32 {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_GRAPH_PREFILL_AFTER") orelse return 2;
    return @max(1, std.fmt.parseInt(u32, std.mem.span(v), 10) catch 2);
}

pub fn clearPrefill() void {
    for (&pentries) |*e| if (e.*) |*x| {
        dropPrefill(x);
        e.* = null;
    };
    pseen = @splat(null);
}

/// Before a layer's sub-blocks: `.replayed` (the graph ran; skip the sub-blocks), `.capturing` (run them: they are
/// recorded; then `prefillEnd`), or `.eager` (run them; `prefillEnd` is a no-op).
/// External event waits for expert-parallel sub-blocks are captured in order with the dense work.
pub fn prefillBegin(p: anytype, linear: bool, layer: usize, rows: usize, sub: usize, pos0: usize, deep: u64, base: anytype, tiles: bool, waits: []const cuda.Event) !Prefill {
    if (enabled() and broken) return error.GraphCaptureFailed;
    if (p.seq.paged) |pg| try pg.validateSharedLeaseOn(p.stream);
    if (!prefillEnabled()) return .eager;
    const q = p.seq;
    if (!linear) if (q.paged) |pg| {
        try pg.prepareLayer(p.stream, wt.attnIndex(layer), pos0, rows);
        pg.graph_prepared = true;
    };
    errdefer if (q.paged) |pg| {
        pg.graph_prepared = false;
    };
    var wait_key: u64 = 0;
    for (waits) |ev| {
        const handle = if (ev.handle) |h| @intFromPtr(h) else 0;
        wait_key = std.hash.Wyhash.hash(wait_key, std.mem.asBytes(&handle));
    }
    const key: PKey = .{
        .layer = layer,
        .rows = rows,
        .sub = sub,
        .pos0 = if (linear) 0 else pos0,
        .deep = deep,
        .base = base.ids,
        .base_cap = base.cap,
        .seq = .{ if (q.conv.len > 0) q.conv[0].ptr else 0, if (q.kc.len > 0) q.kc[0].ptr else 0, q.tail.ptr, q.cap, if (q.state.len > 0) q.state[0].ptr else 0 },
        .paging = if (!linear) (if (q.paged) |pg| pg.graphKey() else @splat(0)) else @splat(0),
        .wait_count = waits.len,
        .wait_key = wait_key,
        .scores = q.scores,
        .tiles = tiles,
    };
    clock += 1;
    for (&pentries) |*e| if (e.*) |*x| if (std.meta.eql(x.key, key)) {
        x.used = clock;
        prefill_stats.replays += 1;
        try x.exec.launchOn(p.stream);
        return .replayed;
    };
    if (psighting(key) < prefillAfter()) {
        prefill_stats.eager += 1;
        return .eager;
    }
    var free_slot = false;
    for (pentries) |entry| free_slot = free_slot or entry == null;
    if (!free_slot) return prefillAdmissionMiss();
    _ = try memory.startupReservation();
    const admission = memory.admitWithQuery(.prefill, 0, p.stream, freeMemory) catch |err| switch (err) {
        error.GraphBudgetBusy, error.GraphBudgetExceeded, error.GraphMemoryHeadroom, error.GraphEvictionRequired => return prefillAdmissionMiss(),
        else => return err,
    };
    errdefer memory.release(admission) catch unreachable;
    try p.stream.synchronize();
    try cuda.graph.beginCapture(p.stream, .thread_local);
    capturing = true;
    pcur = key;
    pcur_admission = admission;
    return .capturing;
}

/// After a layer's sub-blocks ran under `.capturing`: the graph is instantiated (or a cached one of the same shape
/// updated) and launched.
pub fn prefillEnd(p: anytype, how: Prefill) !void {
    defer if (p.seq.paged) |pg| {
        pg.graph_prepared = false;
    };
    if (how != .capturing) return;
    const t0 = std.Io.Clock.awake.now(p.io).toNanoseconds();
    capturing = false;
    const key = pcur.?;
    pcur = null;
    const admission = pcur_admission.?;
    pcur_admission = null;
    var admitted = true;
    defer if (admitted) memory.release(admission) catch unreachable;
    var graph = try cuda.graph.endCapture(p.stream);
    defer graph.deinit();
    const at: *PEntry = blk: {
        for (&pentries) |*e| if (e.*) |*x| if (std.meta.eql(pshape(x.key), pshape(key))) {
            if ((try x.exec.update(graph)) == .success) {
                try uploadResident(&x.exec, p.stream);
                x.key = key;
                x.used = clock;
                prefill_stats.updates += 1;
                break :blk x;
            }
        };
        var best: usize = 0;
        var best_used: u64 = std.math.maxInt(u64);
        for (&pentries, 0..) |*e, i| {
            if (e.* == null) {
                best = i;
                break;
            }
            if (e.*.?.used < best_used) {
                best_used = e.*.?.used;
                best = i;
            }
        }
        if (pentries[best]) |*old| {
            dropPrefill(old);
            pentries[best] = null;
        }
        var exec = try graph.instantiate();
        errdefer exec.deinit();
        try uploadResident(&exec, p.stream);
        try memory.installed(admission);
        pentries[best] = .{ .key = key, .exec = exec, .used = clock, .admission = admission, .stream = p.stream };
        admitted = false;
        break :blk &pentries[best].?;
    };
    prefill_stats.captures += 1;
    prefill_stats.capture_ns += @intCast(std.Io.Clock.awake.now(p.io).toNanoseconds() - t0);
    try at.exec.launchOn(p.stream);
}

/// A failed capture: closed and dropped; prefill graphs stay off for the process (the chunk's error propagates).
pub fn prefillAbort(p: anytype, how: Prefill) void {
    if (p.seq.paged) |pg| pg.graph_prepared = false;
    if (how != .capturing) return;
    capturing = false;
    pcur = null;
    defer if (pcur_admission) |ticket| {
        memory.release(ticket) catch unreachable;
        pcur_admission = null;
    };
    if (cuda.graph.endCapture(p.stream)) |g| {
        var d = g;
        d.deinit();
    } else |_| {}
    std.log.warn("flashnext exl3 graph: a prefill layer could not be captured; prefill and decode graphs are off", .{});
    broken = true;
}

const GraphTestDriver = struct {
    var exec_drops: usize = 0;
    var graph_drops: usize = 0;
    var launches: usize = 0;
    var exec_updates: usize = 0;

    fn dropExec(_: cuda.abi.GraphExec) callconv(.c) c_int {
        exec_drops += 1;
        return 0;
    }

    fn dropGraph(_: cuda.abi.Graph) callconv(.c) c_int {
        graph_drops += 1;
        return 0;
    }

    fn instantiate(out: *cuda.abi.GraphExec, _: cuda.abi.Graph, _: u64) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }

    fn getContext(out: *?*anyopaque) callconv(.c) c_int {
        out.* = @ptrFromInt(1);
        return 0;
    }

    fn noContext(_: *?*anyopaque) callconv(.c) c_int {
        return 1;
    }

    fn setContext(_: ?*anyopaque) callconv(.c) c_int {
        return 0;
    }

    fn driver() cuda.Driver {
        var d: cuda.Driver = undefined;
        d.api.cuGraphExecDestroy = dropExec;
        d.api.cuGraphDestroy = dropGraph;
        d.api.cuGraphLaunch = launch;
        d.api.cuGraphInstantiateWithFlags = instantiate;
        d.api.cuMemGetInfo_v2 = memoryInfo;
        d.api.cuCtxSynchronize = synchronize;
        d.api.cuStreamSynchronize = synchronizeStream;
        d.api.cuStreamBeginCapture_v2 = beginCapture;
        d.api.cuStreamEndCapture = endCapture;
        d.api.cuGraphUpload = upload;
        d.api.cuGraphExecUpdate_v2 = update;
        return d;
    }

    fn memoryInfo(free: *usize, total: *usize) callconv(.c) c_int {
        free.* = 100;
        total.* = 100;
        return 0;
    }

    fn synchronize() callconv(.c) c_int {
        return 0;
    }

    fn synchronizeStream(_: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }

    fn beginCapture(_: cuda.abi.Stream, _: cuda.abi.CaptureMode) callconv(.c) c_int {
        return 0;
    }

    fn endCapture(_: cuda.abi.Stream, out: *cuda.abi.Graph) callconv(.c) c_int {
        out.* = @ptrFromInt(2);
        return 0;
    }

    fn upload(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
        return 0;
    }

    fn update(_: cuda.abi.GraphExec, _: cuda.abi.Graph, info: *cuda.abi.ExecUpdateResultInfo) callconv(.c) c_int {
        exec_updates += 1;
        info.result = .success;
        return 0;
    }

    fn launch(_: cuda.abi.GraphExec, _: cuda.abi.Stream) callconv(.c) c_int {
        launches += 1;
        return 0;
    }

    fn abort(_: cuda.nccl.Comm) callconv(.c) cuda.nccl.Result {
        return 0;
    }

    fn control(link: wire.Link) collective.Collective {
        var c: collective.Collective = undefined;
        c.comm = @ptrFromInt(1);
        c.api.ncclCommAbort = abort;
        c.link = link;
        c.failed = false;
        c.timeout_ms = 1000;
        c.graph_step = 0;
        return c;
    }

    fn finishPending(pd: *Pending, driver_: *cuda.Driver) void {
        pd.exec = .{ .d = driver_, .handle = @ptrFromInt(1) };
        pd.done.store(true, .release);
    }
};

test "exl3: sequence graph retirement removes mixed lanes and pending work without evicting a live owner" {
    clear();
    defer clear();
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    var driver = GraphTestDriver.driver();
    const exec: cuda.graph.Exec = .{ .d = &driver, .handle = @ptrFromInt(1) };
    var key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    key.seq[0][2] = 100;
    const owner_key = key;
    entries[0] = .{ .key = key, .exec = exec, .used = 1 };
    key.seq[0][2] = 200;
    const live_key = key;
    entries[1] = .{ .key = key, .exec = exec, .used = 1 };
    key.rows = 2;
    key.seq[1][2] = 100;
    entries[2] = .{ .key = key, .exec = exec, .used = 1 };
    const pending = try std.heap.c_allocator.create(Pending);
    pending.* = .{ .graph = .{ .d = &driver, .handle = @ptrFromInt(1) }, .ctx = null, .keep_template = true };
    pending.thread = try std.Thread.spawn(.{}, GraphTestDriver.finishPending, .{ pending, &driver });
    entries[3] = .{ .key = owner_key, .pending = pending, .used = 1 };
    const live_pending = try std.heap.c_allocator.create(Pending);
    live_pending.* = .{ .graph = .{ .d = &driver, .handle = @ptrFromInt(1) }, .ctx = null, .keep_template = true };
    live_pending.thread = try std.Thread.spawn(.{}, GraphTestDriver.finishPending, .{ live_pending, &driver });
    var live_pending_key = live_key;
    live_pending_key.rows = 3;
    entries[4] = .{ .key = live_pending_key, .pending = live_pending, .used = 1 };
    const pkey: PKey = .{ .layer = 0, .rows = 32, .sub = 16, .pos0 = 0, .deep = 1, .base = 2, .base_cap = 32, .seq = .{ 1, 2, 100, 32, 3 }, .paging = @splat(0), .wait_count = 0, .wait_key = 0, .scores = 4, .tiles = false };
    pentries[0] = .{ .key = pkey, .exec = exec, .used = 1 };
    var live_pkey = pkey;
    live_pkey.seq[2] = 200;
    pentries[1] = .{ .key = live_pkey, .exec = exec, .used = 1 };
    seen[0] = .{ .key = owner_key, .n = 10, .used = 1 };
    seen[1] = .{ .key = live_key, .n = 10, .used = 1 };
    pseen[0] = .{ .key = pkey, .n = 10 };
    pseen[1] = .{ .key = live_pkey, .n = 10 };
    const Seq = struct { tail: struct { ptr: u64 } };
    releaseSeq(Seq{ .tail = .{ .ptr = 100 } });
    try std.testing.expect(find(owner_key) == null);
    try std.testing.expect(find(live_key) != null);
    try std.testing.expect(entries[2] == null and entries[3] == null);
    try std.testing.expect(entries[4].?.pending == live_pending);
    try std.testing.expect(pentries[0] == null and pentries[1] != null);
    try std.testing.expect(seen[0] == null and seen[1] != null);
    try std.testing.expect(pseen[0] == null and pseen[1] != null);
    try std.testing.expectEqual(@as(usize, 4), GraphTestDriver.exec_drops);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.graph_drops);
    try std.testing.expectEqual(@as(u32, 1), sighting(owner_key));
    releaseSeq(Seq{ .tail = .{ .ptr = 100 } });
    try std.testing.expectEqual(@as(usize, 4), GraphTestDriver.exec_drops);
}

test "exl3: pending graph deadlines and failed instantiation never expose an executable" {
    var pending: Pending = .{ .graph = undefined, .ctx = null, .keep_template = true, .started_ns = wire.now() };
    var entry: Entry = .{ .key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 }, .pending = &pending, .used = 1, .cohort = 7 };
    try std.testing.expectEqual(collective.GraphState.pending, graphVote(&entry, 1000).state);
    pending.started_ns = wire.now() -| 2 * std.time.ns_per_ms;
    try std.testing.expectEqual(collective.GraphState.timed_out, graphVote(&entry, 1).state);
    entry.pending = null;
    try std.testing.expectEqual(collective.GraphState.failed, graphVote(&entry, 1000).state);
    drop(&entry);
}

test "exl3: asynchronous TP instantiation preserves its NCCL template until owner retirement" {
    if (!asyncInstantiate()) return error.SkipZigTest;
    const previous_api = ctx_api;
    defer ctx_api = previous_api;
    ctx_api = .{ .get = GraphTestDriver.getContext, .set = GraphTestDriver.setContext };
    var driver = GraphTestDriver.driver();
    const graph: cuda.graph.Graph = .{ .d = &driver, .handle = @ptrFromInt(1) };
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    const pending = (try startInstantiate(graph, true)).?;
    var entry: Entry = .{ .key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 }, .pending = pending, .used = 1, .cohort = 9 };
    defer drop(&entry);
    const until = wire.deadline(1000);
    while (!pending.done.load(.acquire)) {
        if (wire.now() >= until) return error.RankTimeout;
        wire.pause();
    }
    const vote = graphVote(&entry, 1000);
    try std.testing.expectEqual(collective.GraphState.ready, vote.state);
    try std.testing.expectEqual(@as(u64, 9), vote.cohort);
    try std.testing.expect(entry.pending == null and entry.exec != null and entry.template != null);
    try std.testing.expectEqual(@as(usize, 0), GraphTestDriver.graph_drops);
}

test "exl3: rendezvous signatures compare topology and paging origins without device addresses" {
    var key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    key.seq[0] = .{ 1, 2, 100, 4096, 3 };
    key.paging[0] = .{ 10, 20, 30, 40, 50, 60 };
    const signature = graphSignature(key, 4096);
    key.seq[0][0] = 1000;
    key.seq[0][2] = 1001;
    key.paging[0][0] = 1002;
    key.output = 1003;
    try std.testing.expectEqual(signature, graphSignature(key, 4096));
    try std.testing.expect(signature != graphSignature(key, 4097));
    key.paging[0][5] += 1;
    try std.testing.expect(signature != graphSignature(key, 4096));
}

test "exl3: shared paging rendezvous ignores physical ownership but local graph keys retain it" {
    var first: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    first.seq[0] = .{ 1, 2, 3, 4096, 4 };
    first.paging[0] = .{ 10, 20, 30, 40, 50, 60 };
    first.paging_logical = @splat(@splat(0));
    first.paging_logical.?[0] = .{ 100, 200 };
    var second = first;
    second.paging[0] = .{ 1000, 2000, 3000, 4000, 5000, 60 };
    try std.testing.expect(!std.meta.eql(first, second));
    try std.testing.expect(!std.meta.eql(state_binding.normalized(first), state_binding.normalized(second)));
    try std.testing.expectEqual(graphSignature(first, 4096), graphSignature(second, 4096));
    second.paging_logical.?[0][0] += 1;
    try std.testing.expect(graphSignature(first, 4096) != graphSignature(second, 4096));
    second = first;
    second.paging_logical.?[0][1] += 1;
    try std.testing.expect(graphSignature(first, 4096) != graphSignature(second, 4096));
    second = first;
    second.seq[0][3] += 1;
    try std.testing.expect(graphSignature(first, 4096) != graphSignature(second, 4096));
}

test "exl3: TP decode dispatch stays eager while its peer instantiates and replays only after agreement" {
    clear();
    defer clear();
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    const key: Key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 };
    const Peer = struct {
        fn main(link: wire.Link, signature: u64) void {
            var c = GraphTestDriver.control(link);
            const first = c.graphAgree(signature, .{ .state = .pending, .cohort = 7 }) catch @panic("first peer agreement failed");
            std.debug.assert(first.choice == .eager);
            const second = c.graphAgree(signature, .{ .state = .ready, .cohort = 7 }) catch @panic("second peer agreement failed");
            std.debug.assert(second.choice == .replay);
        }
    };
    const thread = try std.Thread.spawn(.{}, Peer.main, .{ wire.Link{ .fd = pair[1] }, graphSignature(key, 0) });
    defer thread.join();
    var c = GraphTestDriver.control(.{ .fd = pair[0] });
    var driver = GraphTestDriver.driver();
    GraphTestDriver.launches = 0;
    entries[0] = .{ .key = key, .used = 1, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .cohort = 7 };
    const Pass = struct {
        io: std.Io = std.testing.io,
        stream: cuda.Stream,
        m: struct { collective: ?*collective.Collective },
        prepared: bool = false,
        chunks: usize = 0,

        fn chunk(p: *@This(), _: []const u32, _: usize, _: ?[]u8) !void {
            p.chunks += 1;
        }
    };
    var pass: Pass = .{ .stream = cuda.Stream.borrowed(&driver, null), .m = .{ .collective = &c } };
    try run(key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 1), pass.chunks);
    try std.testing.expectEqual(@as(usize, 0), GraphTestDriver.launches);
    try run(key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 1), pass.chunks);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.launches);
}

test "exl3: TP pressure on either rank retires cohorts before retrying admission" {
    clear();
    defer clear();
    for ([_]bool{ true, false }) |local_pressure| {
        try memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 }, .prefill_charge = .{ .exec = 1, .template = 0, .transient = 0 } });
        var driver = GraphTestDriver.driver();
        GraphTestDriver.exec_drops = 0;
        const local = try memory.admit(.prefill, 0, 100);
        try memory.installed(local);
        pentries[0] = .{ .key = std.mem.zeroes(PKey), .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 0, .admission = local };
        if (local_pressure) for (0..2) |index| {
            const ticket = try memory.admit(.decode, index + 7, 100);
            try memory.installed(ticket);
            entries[index] = .{ .key = .{ .kind = 1, .rows = index + 1, .scratch = 1, .scores = 2, .output = 3 }, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = index + 1, .cohort = index + 7, .admission = ticket };
        };
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer for (pair) |fd| {
            _ = std.c.close(fd);
        };
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        for (pair) |fd| {
            const flags = std.c.fcntl(fd, std.c.F.GETFL);
            if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
        }
        const Peer = struct {
            fn run(link: wire.Link, pressure: bool) void {
                var control = GraphTestDriver.control(link);
                const first = control.graphAgree(42 ^ 0x61646d6974, .{ .state = if (pressure) .ready else .eager, .cohort = 11 }) catch @panic("pressure agreement failed");
                std.debug.assert(first.choice == .eager);
                const victim = control.graphVictim(42 ^ 0x76696374696d, if (pressure) 8 else 7) catch @panic("victim agreement failed");
                std.debug.assert(victim == 7);
                retirementBarrier(&control, 42 ^ 0x71756965736365, victim) catch @panic("quiescence agreement failed");
                retirementBarrier(&control, 42 ^ 0x72657469726564, victim) catch @panic("retirement agreement failed");
                const admitted = control.graphAgree(42 ^ 0x61646d6974, .{ .state = .ready, .cohort = 11 }) catch @panic("retry agreement failed");
                std.debug.assert(admitted.choice == .replay);
            }
        };
        const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, local_pressure });
        defer peer.join();
        var control = GraphTestDriver.control(.{ .fd = pair[0] });
        const ticket = try reserveTogether(&control, cuda.Stream.borrowed(&driver, null), 42, 11);
        try std.testing.expectEqual(@as(u64, if (local_pressure) 71 else 41), memory.charged());
        try std.testing.expectEqual(@as(usize, if (local_pressure) 1 else 0), GraphTestDriver.exec_drops);
        try std.testing.expect(pentries[0] != null);
        if (local_pressure) try std.testing.expectEqual(@as(u64, 8), entries[1].?.cohort);
        try memory.release(ticket);
        clear();
    }
}

test "exl3: exhausted TP victims reclaim locals incrementally with asymmetric peer caches" {
    clear();
    defer clear();
    for ([_]bool{ false, true }) |local_cached| for ([_]bool{ false, true }) |peer_cached| {
        try memory.configure(.{ .reservation_bytes = 100, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
        var driver = GraphTestDriver.driver();
        GraphTestDriver.exec_drops = 0;
        if (local_cached) {
            const prefill = try memory.admit(.prefill, 0, 100);
            try memory.installed(prefill);
            pentries[0] = .{ .key = std.mem.zeroes(PKey), .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1, .admission = prefill };
            const local = try memory.admit(.decode, 0, 100);
            try memory.installed(local);
            entries[0] = .{ .key = .{ .kind = 1, .rows = 1, .scratch = 1, .scores = 2, .output = 3 }, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 2, .admission = local };
        }
        var pair: [2]std.posix.socket_t = undefined;
        if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
        defer for (pair) |fd| {
            _ = std.c.close(fd);
        };
        const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
        for (pair) |fd| {
            const flags = std.c.fcntl(fd, std.c.F.GETFL);
            if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
        }
        const Peer = struct {
            fn run(link: wire.Link, local_cached_peer: bool, cached: bool) void {
                var control = GraphTestDriver.control(link);
                const first = control.graphAgree(42 ^ 0x61646d6974, .{ .state = .eager, .cohort = 11 }) catch @panic("pressure agreement failed");
                std.debug.assert(first.choice == .eager);
                const victim = control.graphVictim(42 ^ 0x76696374696d, 0) catch @panic("victim agreement failed");
                std.debug.assert(victim == 0);
                retirementBarrier(&control, 42 ^ 0x6c6f63616c7175, 11) catch @panic("local quiescence failed");
                const reclaimed = control.graphVictim(42 ^ 0x6c6f63616c646f, @intFromBool(cached)) catch @panic("local destruction acknowledgment failed");
                std.debug.assert(reclaimed == @as(u64, @intFromBool(local_cached_peer or cached)));
                if (reclaimed == 0) return;
                const admitted = control.graphAgree(42 ^ 0x61646d6974, .{ .state = .ready, .cohort = 11 }) catch @panic("retry agreement failed");
                std.debug.assert(admitted.choice == .replay);
            }
        };
        const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, local_cached, peer_cached });
        defer peer.join();
        var control = GraphTestDriver.control(.{ .fd = pair[0] });
        const result = reserveTogether(&control, cuda.Stream.borrowed(&driver, null), 42, 11);
        if (local_cached or peer_cached) {
            const ticket = try result;
            try std.testing.expectEqual(@as(u64, if (local_cached) 70 else 40), memory.charged());
            try memory.release(ticket);
        } else {
            try std.testing.expectError(error.GraphBudgetExceeded, result);
            try std.testing.expectEqual(@as(u64, 0), memory.charged());
        }
        try std.testing.expectEqual(@as(usize, @intFromBool(local_cached)), GraphTestDriver.exec_drops);
        try std.testing.expect(pentries[0] == null);
        try std.testing.expectEqual(local_cached, entries[0] != null);
        clear();
    };
}

test "exl3: TP updates agree the old cohort and transfer template and admission together" {
    try testTPUpdate(true);
}

test "exl3: asymmetric TP update rejection retires both execs before re-instantiating" {
    try testTPUpdate(false);
}

fn testTPUpdate(peer_accept: bool) !void {
    if (!tpUpdate()) return error.SkipZigTest;
    clear();
    defer clear();
    const previous_api = ctx_api;
    defer ctx_api = previous_api;
    ctx_api = .{ .get = GraphTestDriver.noContext, .set = GraphTestDriver.setContext };
    try memory.configure(.{ .reservation_bytes = 120, .headroom_bytes = 20, .charge = .{ .exec = 20, .template = 10, .transient = 10 } });
    var driver = GraphTestDriver.driver();
    GraphTestDriver.exec_updates = 0;
    GraphTestDriver.exec_drops = 0;
    GraphTestDriver.graph_drops = 0;
    const old_ticket = try memory.admit(.decode, 7, 100);
    try memory.installed(old_ticket);
    const key: Key = .{ .kind = 1, .rows = 1, .scratch = 2, .scores = 2, .output = 3 };
    var old_key = key;
    old_key.scratch = 1;
    entries[0] = .{ .key = old_key, .exec = .{ .d = &driver, .handle = @ptrFromInt(1) }, .template = .{ .d = &driver, .handle = @ptrFromInt(1) }, .used = 1, .cohort = 7, .admission = old_ticket };
    var old_binding_request: state_binding.Request = .{
        .width = 2,
        .chain = 10,
        .chain_to = 11,
        .geometry = .{ .grid = .{ 16, 8, 1 }, .block = .{ 128, 1, 1 } },
        .nonstate = .{ .gq = 1, .gk = 2, .gv = 3, .ggb = 4, .rows = 2, .ys = 5 },
        .count = 1,
    };
    old_binding_request.layers[0] = .{ .layer = 0, .state_in = 100, .state_out = 200 };
    entries[0].?.binding = try state_binding.Binding.init(1, old_binding_request);
    seen[0] = .{ .key = key, .n = std.math.maxInt(u32) - 1, .used = 1 };
    const previous_clock = clock;
    defer clock = previous_clock;
    clock = 1024;
    var pair: [2]std.posix.socket_t = undefined;
    if (std.c.socketpair(std.posix.AF.UNIX, std.posix.SOCK.STREAM, 0, &pair) != 0) return error.SocketPair;
    defer for (pair) |fd| {
        _ = std.c.close(fd);
    };
    const nonblock: c_int = @bitCast(@as(u32, @bitCast(std.c.O{ .NONBLOCK = true })));
    for (pair) |fd| {
        const flags = std.c.fcntl(fd, std.c.F.GETFL);
        if (flags < 0 or std.c.fcntl(fd, std.c.F.SETFL, flags | nonblock) < 0) return error.Nonblocking;
    }
    const Peer = struct {
        fn run(link: wire.Link, signature: u64, accept: bool) void {
            var control = GraphTestDriver.control(link);
            control.graph_step = 10;
            const capture = control.graphAgree(signature, .{ .state = .capture }) catch @panic("capture agreement failed");
            std.debug.assert(capture.choice == .capture and capture.cohort == 11);
            retirementBarrier(&control, signature ^ 0x61646d6974, 11) catch @panic("admission failed");
            retirementBarrier(&control, signature ^ 0x757064617465, 7) catch @panic("old cohort agreement failed");
            const result = control.graphAgree(signature ^ 0x75706461746564, .{ .state = if (accept) .ready else .capture, .cohort = 11 }) catch @panic("update result agreement failed");
            std.debug.assert(result.choice == if (accept) collective.GraphChoice.replay else collective.GraphChoice.capture);
            if (!accept) retirementBarrier(&control, signature ^ 0x72656275696c64, 11) catch @panic("rebuild retirement agreement failed");
            retirementBarrier(&control, signature ^ 0x696e7374616c6c, 11) catch @panic("installation agreement failed");
        }
    };
    const peer = try std.Thread.spawn(.{}, Peer.run, .{ wire.Link{ .fd = pair[1] }, graphSignature(key, 0), peer_accept });
    defer peer.join();
    var control = GraphTestDriver.control(.{ .fd = pair[0] });
    control.graph_step = 10;
    const Pass = struct {
        io: std.Io = std.testing.io,
        stream: cuda.Stream,
        prepared: bool = false,
        chunks: usize = 0,
        fn chunk(pass: *@This(), _: []const u32, _: usize, _: ?[]u8) !void {
            pass.chunks += 1;
        }
    };
    var pass: Pass = .{ .stream = cuda.Stream.borrowed(&driver, null) };
    try runTogether(&control, key, &pass, &.{1}, 0);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.exec_updates);
    try std.testing.expectEqual(@as(usize, 1), GraphTestDriver.graph_drops);
    try std.testing.expectEqual(@as(usize, if (peer_accept) 0 else 1), GraphTestDriver.exec_drops);
    try std.testing.expectEqual(@as(u64, 30), memory.charged());
    try std.testing.expectEqual(@as(u64, 11), find(key).?.cohort);
    try std.testing.expect(find(key).?.binding == null);
    try std.testing.expectError(error.StaleGraphTicket, memory.release(old_ticket));
    try std.testing.expectEqual(@as(usize, 1), pass.chunks);
}
