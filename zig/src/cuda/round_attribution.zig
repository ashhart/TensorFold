//! Opt-in driver event correlation, retaining captured graph launches and complete host round boundaries.
const std = @import("std");
const platform = @import("sys");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
pub const model = @import("profile").attribution_model;
pub const meta = @import("round_attribution_meta.zig");
pub const wrappers = @import("round_attribution_api.zig").API(Trace);
pub const EventNode = *const fn (abi.GraphNode, *abi.Event) callconv(.c) abi.Result;
pub var owner: ?*Trace = null;
const Pair = struct { begin: abi.Event, end: abi.Event, stream: abi.Stream, desc: meta.Meta, owned: bool = false, inflight: bool = false, free: bool = false, submission: u64 = 0, submitted: u64 = 0 };
const Pending = struct { pair: usize, correlation: u64, replay: ?u64, phase: model.Phase, scope: u64 };
pub const Group = struct { pairs: []usize, refs: usize = 1 };
pub const Scope = struct {
    trace: ?*Trace = null,
    previous: model.Phase = .other,
    phase: model.Phase = .other,
    begin: u64 = 0,
    correlation: u64 = 0,
    parent: u64 = 0,
    previous_readback: model.Phase = .token_readback,
    pub fn end(s: Scope) void {
        const t = s.trace orelse return;
        t.addHost(.{ .correlation = s.correlation, .parent_scope = s.parent, .phase = s.phase, .kind = .span, .begin_ns = s.begin, .end_ns = t.now() });
        t.phase = s.previous;
        t.scope_id = s.parent;
        t.readback = s.previous_readback;
    }
};
pub inline fn scope(phase: model.Phase) Scope {
    const t = owner orelse return .{};
    if (!t.active) return .{};
    const s: Scope = .{ .trace = t, .previous = t.phase, .phase = phase, .begin = t.now(), .correlation = t.next(), .parent = t.scope_id, .previous_readback = t.readback };
    t.phase = phase;
    t.scope_id = s.correlation;
    return s;
}
pub inline fn land(sampled: bool) Scope {
    const s = scope(.land);
    if (s.trace) |trace| trace.readback = if (sampled) .token_readback else .logits_readback;
    return s;
}
pub const Trace = struct {
    a: std.mem.Allocator,
    io: std.Io,
    driver: *Driver,
    original: abi.Api,
    event_node: ?EventNode = null,
    permanent: std.heap.ArenaAllocator,
    round_arena: std.heap.ArenaAllocator,
    functions: std.AutoHashMapUnmanaged(usize, []const u8) = .empty,
    pairs: std.ArrayList(Pair) = .empty,
    free: std.ArrayList(usize) = .empty,
    capture: std.ArrayList(usize) = .empty,
    capturing: bool = false,
    graphs: std.AutoHashMapUnmanaged(usize, *Group) = .empty,
    execs: std.AutoHashMapUnmanaged(usize, *Group) = .empty,
    pending: std.ArrayList(Pending) = .empty,
    device: std.ArrayList(model.Device) = .empty,
    host: std.ArrayList(model.Host) = .empty,
    frames: std.ArrayList(model.Frame) = .empty,
    requests: std.ArrayList(model.Request) = .empty,
    windows: std.ArrayList(model.Window) = .empty,
    begin_event: abi.Event = null,
    end_event: abi.Event = null,
    stream: abi.Stream = null,
    context: abi.Context = null,
    identity: []const u8 = "",
    header: std.json.Value = .null,
    seq: u64 = 0,
    round: u64 = 0,
    active: bool = false,
    phase: model.Phase = .other,
    scope_id: u64 = 0,
    root_scope: u64 = 0,
    readback: model.Phase = .token_readback,
    started: i96 = 0,
    thread: std.Thread.Id = undefined,
    problem: ?[]const u8 = null,
    persistent_problem: ?[]const u8 = null,
    previous: ?meta.Meta = null,

    pub fn install(a: std.mem.Allocator, io: std.Io, d: *const Driver) !?*Trace {
        if (comptime !@import("profile").round_attribution.enabled) return null;
        if (!try platform.environment.read(bool, "TF_CUDA_ROUND_TRACE", false)) return null;
        if (owner != null) return error.AttributionNeedsOneCudaEngine;
        if (!sourceValid(@import("profile").round_attribution.source)) return error.AttributionSourceIdentityRequired;
        const t = try a.create(Trace);
        t.* = .{ .a = a, .io = io, .driver = @constCast(d), .original = d.api, .permanent = std.heap.ArenaAllocator.init(a), .round_arena = std.heap.ArenaAllocator.init(a) };
        errdefer {
            t.permanent.deinit();
            t.round_arena.deinit();
            a.destroy(t);
        }
        try d.check(t.original.cuEventCreate(&t.begin_event, 0), "attribution begin event");
        errdefer _ = t.original.cuEventDestroy_v2(t.begin_event);
        try d.check(t.original.cuEventCreate(&t.end_event, 0), "attribution end event");
        errdefer _ = t.original.cuEventDestroy_v2(t.end_event);
        t.event_node = t.driver.lib.lookup(EventNode, "cuGraphEventRecordNodeGetEvent") orelse return error.AttributionEventNodeQueryUnavailable;
        owner = t;
        wrappers.install(t);
        return t;
    }
    pub fn deinit(t: *Trace) void {
        if (t.context != null) _ = t.original.cuCtxSetCurrent(t.context);
        _ = t.original.cuCtxSynchronize();
        t.driver.api = t.original;
        owner = null;
        wrappers.owner = null;
        var gs = t.graphs.valueIterator();
        while (gs.next()) |g| t.release(g.*);
        var es = t.execs.valueIterator();
        while (es.next()) |g| t.release(g.*);
        for (t.pairs.items) |p| {
            _ = t.original.cuEventDestroy_v2(p.begin);
            _ = t.original.cuEventDestroy_v2(p.end);
        }
        _ = t.original.cuEventDestroy_v2(t.begin_event);
        _ = t.original.cuEventDestroy_v2(t.end_event);
        t.graphs.deinit(t.a);
        t.execs.deinit(t.a);
        t.functions.deinit(t.a);
        t.pairs.deinit(t.a);
        t.free.deinit(t.a);
        t.capture.deinit(t.a);
        t.pending.deinit(t.a);
        t.device.deinit(t.a);
        t.host.deinit(t.a);
        t.frames.deinit(t.a);
        t.requests.deinit(t.a);
        t.windows.deinit(t.a);
        t.round_arena.deinit();
        t.permanent.deinit();
        const a = t.a;
        a.destroy(t);
    }
    pub fn isTimingNode(t: *Trace, node: abi.GraphNode) bool {
        const query = t.event_node orelse return false;
        var event: abi.Event = null;
        if (query(node, &event) != 0) return false;
        for (t.pairs.items) |p| if (p.begin == event or p.end == event) return true;
        return false;
    }
    pub fn next(t: *Trace) u64 {
        t.seq += 1;
        return t.seq;
    }
    pub fn now(t: *Trace) u64 {
        return @intCast(@max(0, std.Io.Clock.awake.now(t.io).toNanoseconds() - t.started));
    }
    pub fn fail(t: *Trace, why: []const u8) void {
        if (t.problem == null) t.problem = why;
        if (!t.active and t.persistent_problem == null) t.persistent_problem = why;
    }
    pub fn addHost(t: *Trace, h: model.Host) void {
        t.host.append(t.a, h) catch t.fail("host record allocation");
    }
    pub fn name(t: *Trace, f: abi.Function) []const u8 {
        return t.functions.get(@intFromPtr(f)) orelse t.driver.functionName(f);
    }

    pub fn bind(t: *Trace, e: anytype, ctx: anytype, dir: []const u8, checkpoint: anytype) !void {
        const a = t.permanent.allocator();
        if (e.g.hidden != 5120 or e.g.layers != 64 or e.g.experts != 0 or e.g.exl3 or e.g.tfq3 or !e.g.kinds.modelopt or e.g.first != 0 or e.g.whole != 0) return error.AttributionNeedsE0DenseTarget;
        t.stream = e.stream.handle;
        t.context = ctx.handle;
        const path = try std.fmt.allocPrint(a, "{s}/config.json", .{dir});
        const text = try std.Io.Dir.cwd().readFileAlloc(t.io, path, a, .limited(1 << 24));
        const config = try std.json.parseFromSlice(std.json.Value, a, text, .{ .allocate = .alloc_always });
        const quant = config.value.object.get("quantization_config") orelse config.value.object.get("quantization") orelse .null;
        const conditions = @import("profile").get(.TF_CUDA_ROUND_CONDITIONS);
        const receipt = .{ .schema = "tensorfold.cuda-round-identity.v1", .source = @import("profile").round_attribution.source, .binary_sha256 = std.fmt.bytesToHex(try platform.file_identity.running(a, t.io), .lower), .checkpoint = checkpoint, .drafter = @as(?u8, null), .checkpoint_config_sha256 = std.fmt.bytesToHex(digest(text), .lower), .quantization = quant, .geometry = e.g, .kv8 = e.kv8, .driver = try t.driver.version(), .sm_major = try ctx.attribute(.compute_capability_major), .sm_minor = try ctx.attribute(.compute_capability_minor), .operation_names = @typeInfo(model.Op).@"enum".field_names, .phase_names = @typeInfo(model.Phase).@"enum".field_names, .conditions = conditions, .coverage = "single-context driver-table events; captured graphs retained; errors invalidate budgets", .instrumentation = "timing event nodes and diagnostic end drain; measured round includes profiling overhead; event brackets include possible enqueue gaps, not profiler kernel-busy time" };
        const json = try std.json.Stringify.valueAlloc(a, receipt, .{});
        t.header = (try std.json.parseFromSlice(std.json.Value, a, json, .{ .allocate = .alloc_always })).value;
        try t.publishIdentity();
    }
    fn publishIdentity(t: *Trace) !void {
        const a = t.permanent.allocator();
        const json = try std.json.Stringify.valueAlloc(a, t.header, .{});
        t.identity = try a.dupe(u8, &std.fmt.bytesToHex(digest(json), .lower));
        std.debug.print("TF_ROUND_IDENTITY {s} {s}\n", .{ t.identity, json });
    }
    pub fn annotate(t: *Trace, key: []const u8, json: []const u8) !void {
        const a = t.permanent.allocator();
        const parsed = try std.json.parseFromSlice(std.json.Value, a, json, .{ .allocate = .alloc_always });
        try t.header.object.put(a, try a.dupe(u8, key), parsed.value);
        try t.publishIdentity();
    }
    pub fn begin(t: *Trace) void {
        _ = t.round_arena.reset(.retain_capacity);
        t.pending.clearRetainingCapacity();
        t.device.clearRetainingCapacity();
        t.host.clearRetainingCapacity();
        t.frames.clearRetainingCapacity();
        t.requests.clearRetainingCapacity();
        t.windows.clearRetainingCapacity();
        t.round += 1;
        t.active = true;
        t.problem = t.persistent_problem;
        t.phase = .other;
        t.previous = null;
        t.root_scope = t.next();
        t.scope_id = t.root_scope;
        t.thread = std.Thread.getCurrentId();
        t.started = std.Io.Clock.awake.now(t.io).toNanoseconds();
        if (t.context != null and t.original.cuCtxSetCurrent(t.context) != 0) t.fail("round context binding");
        if (t.original.cuEventRecord(t.begin_event, t.stream) != 0) t.fail("round origin event");
    }
    pub fn threadOk(t: *Trace) bool {
        if (t.active and t.thread != std.Thread.getCurrentId()) {
            t.fail("concurrent foreign CUDA thread");
            return false;
        }
        return true;
    }
    pub fn before(t: *Trace, stream: abi.Stream, desc: meta.Meta) ?usize {
        if ((!t.active and !t.capturing) or !t.threadOk()) return null;
        if (t.active and stream != t.stream) t.fail("secondary stream outside supported envelope");
        const i = if (t.free.pop()) |x| x else blk: {
            if (t.pairs.items.len >= 65536) {
                t.fail("event capacity");
                return null;
            }
            var b: abi.Event = null;
            var end: abi.Event = null;
            if (t.original.cuEventCreate(&b, 0) != 0) {
                t.fail("event allocation");
                return null;
            }
            if (t.original.cuEventCreate(&end, 0) != 0) {
                _ = t.original.cuEventDestroy_v2(b);
                t.fail("event allocation");
                return null;
            }
            t.pairs.append(t.a, .{ .begin = b, .end = end, .stream = stream, .desc = desc }) catch {
                _ = t.original.cuEventDestroy_v2(b);
                _ = t.original.cuEventDestroy_v2(end);
                t.fail("event metadata allocation");
                return null;
            };
            break :blk t.pairs.items.len - 1;
        };
        const p = &t.pairs.items[i];
        p.submission = t.next();
        p.free = false;
        p.stream = stream;
        p.desc = desc;
        p.owned = t.capturing;
        const res = if (t.capturing) t.original.cuEventRecordWithFlags(p.begin, stream, abi.event_record_external) else t.original.cuEventRecord(p.begin, stream);
        if (res != 0) t.fail("event begin");
        p.submitted = t.now();
        return i;
    }
    pub fn after(t: *Trace, index: ?usize, result: abi.Result) void {
        const i = index orelse return;
        const p = &t.pairs.items[i];
        const res = if (t.capturing) t.original.cuEventRecordWithFlags(p.end, p.stream, abi.event_record_external) else t.original.cuEventRecord(p.end, p.stream);
        if (result != 0 or res != 0) t.fail("observed CUDA operation failed");
        if (t.active and !t.capturing) t.addHost(.{ .correlation = p.submission, .parent_scope = t.scope_id, .label = p.desc.symbol, .phase = t.phase, .kind = .span, .begin_ns = p.submitted, .end_ns = t.now() });
        if (t.capturing) t.capture.append(t.a, i) catch t.fail("capture metadata allocation") else t.enqueue(i, null);
    }
    pub fn enqueue(t: *Trace, i: usize, replay_id: ?u64) void {
        const p = &t.pairs.items[i];
        if (p.inflight) {
            t.fail("event reused before existing wait collected it");
            return;
        }
        t.pending.append(t.a, .{ .pair = i, .correlation = if (replay_id == null) p.submission else t.next(), .replay = replay_id, .phase = if (t.phase == .land and p.desc.op == .d2h) t.readback else t.phase, .scope = t.scope_id }) catch {
            t.fail("pending metadata allocation");
            if (!p.owned) t.recycle(i);
            return;
        };
        p.inflight = true;
    }
    pub fn collect(t: *Trace, stream: ?abi.Stream) void {
        var at: usize = 0;
        while (at < t.pending.items.len) {
            const n = t.pending.items[at];
            const p = &t.pairs.items[n.pair];
            if (stream) |s| if (p.stream != s) {
                at += 1;
                continue;
            };
            var start_ms: f32 = 0;
            var end_ms: f32 = 0;
            if (t.original.cuEventElapsedTime(&start_ms, t.begin_event, p.begin) != 0 or t.original.cuEventElapsedTime(&end_ms, t.begin_event, p.end) != 0 or !std.math.isFinite(start_ms) or !std.math.isFinite(end_ms) or start_ms < 0 or end_ms < start_ms) {
                t.fail("unavailable or backwards device timestamp");
            } else {
                const qs: []const model.Qmmf = t.round_arena.allocator().dupe(model.Qmmf, p.desc.projections[0..p.desc.count]) catch blk: {
                    t.fail("projection metadata allocation");
                    break :blk &.{};
                };
                t.device.append(t.a, .{ .correlation = n.correlation, .parent_scope = n.scope, .graph_replay = n.replay, .stream = @intFromPtr(p.stream), .phase = n.phase, .op = p.desc.op, .symbol = p.desc.symbol, .begin_ns = @intFromFloat(@as(f64, start_ms) * 1e6), .end_ns = @intFromFloat(@as(f64, end_ms) * 1e6), .bytes = p.desc.bytes, .grid = p.desc.grid, .block = p.desc.block, .qmmf = qs }) catch t.fail("device record allocation");
                if (p.desc.op == .qmmf and p.desc.count == 0) t.fail("unresolved QMMF projection metadata");
            }
            p.inflight = false;
            _ = t.pending.orderedRemove(at);
            if (!p.owned) t.recycle(n.pair);
        }
    }
    fn recycle(t: *Trace, i: usize) void {
        const p = &t.pairs.items[i];
        if (p.free or p.inflight or p.owned) return;
        p.free = true;
        t.free.append(t.a, i) catch t.fail("event pool metadata allocation");
    }
    pub fn release(t: *Trace, g: *Group) void {
        g.refs -= 1;
        if (g.refs != 0) return;
        for (g.pairs) |i| {
            t.pairs.items[i].owned = false;
            t.recycle(i);
        }
        t.a.free(g.pairs);
        t.a.destroy(g);
    }
    pub fn captured(t: *Trace, graph: abi.Graph, ok: bool) void {
        defer {
            t.capturing = false;
            t.capture.clearRetainingCapacity();
        }
        if (!ok) {
            for (t.capture.items) |i| {
                t.pairs.items[i].owned = false;
                t.recycle(i);
            }
            return;
        }
        const g = t.a.create(Group) catch {
            t.fail("capture group allocation");
            return;
        };
        g.* = .{ .pairs = t.a.dupe(usize, t.capture.items) catch {
            t.a.destroy(g);
            t.fail("capture list allocation");
            return;
        } };
        t.graphs.put(t.a, @intFromPtr(graph), g) catch {
            t.release(g);
            t.fail("capture map allocation");
        };
    }
    pub fn replay(t: *Trace, exec: abi.GraphExec) ?u64 {
        if (!t.active) return null;
        const id = t.next();
        const g = t.execs.get(@intFromPtr(exec)) orelse {
            t.fail("untracked graph replay");
            return id;
        };
        for (g.pairs) |i| t.enqueue(i, id);
        return id;
    }
    pub fn finish(t: *Trace, aborted: bool) void {
        const functional = t.now();
        t.addHost(.{ .correlation = t.root_scope, .label = "round", .phase = .other, .kind = .span, .begin_ns = 0, .end_ns = functional });
        if (aborted) t.fail("round aborted");
        if (t.original.cuEventRecord(t.end_event, t.stream) != 0) t.fail("round end event");
        const drain = t.now();
        if (t.original.cuCtxSynchronize() != 0) t.fail("attribution drain");
        const drained = t.now();
        t.addHost(.{ .correlation = t.next(), .phase = .other, .kind = .attribution_drain, .begin_ns = drain, .end_ns = drained });
        t.collect(null);
        var envelope: f32 = 0;
        if (t.original.cuEventElapsedTime(&envelope, t.begin_event, t.end_event) != 0 or !std.math.isFinite(envelope) or envelope < 0) {
            t.fail("device envelope");
            envelope = 0;
        }
        const wall = t.now();
        const budget = model.budgets(t.round_arena.allocator(), wall, functional, @intFromFloat(@as(f64, envelope) * 1e6), t.device.items, t.host.items) catch |err| {
            std.debug.print("TF_ROUND_ERROR round={d} budget={s}\n", .{ t.round, @errorName(err) });
            t.active = false;
            return;
        };
        const r: model.Record = .{ .round = t.round, .identity = t.identity, .complete = t.problem == null, .problem = t.problem, .requests = t.requests.items, .frames = t.frames.items, .windows = t.windows.items, .host = t.host.items, .device = t.device.items, .budget = budget };
        const json = model.json(t.round_arena.allocator(), r) catch |err| {
            std.debug.print("TF_ROUND_ERROR round={d} record={s}\n", .{ t.round, @errorName(err) });
            t.active = false;
            return;
        };
        std.debug.print("TF_ROUND {s}\n", .{json});
        t.active = false;
    }
};

pub fn digest(bytes: []const u8) [32]u8 {
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return out;
}

pub fn sourceValid(source: []const u8) bool {
    if (source.len != 81 or source[40] != ':') return false;
    for (source, 0..) |c, i| if (i != 40 and !std.ascii.isHex(c)) return false;
    return true;
}
