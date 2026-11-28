//! One opt-in prefill owner using CUDA attribution's driver-table and event-pair ownership pattern.
const std = @import("std");
const sys = @import("sys");
const abi = @import("abi.zig");
const Driver = @import("driver.zig").Driver;
const Pair = @import("attribution_events.zig").Pair;
pub const ledger = @import("prefill_ledger.zig");
const API = @import("prefill_attribution_api.zig").API(Trace);
pub var owner: ?*Trace = null;
pub const Ticket = ?usize;
const Entry = struct {
    row: ledger.Row,
    stream: abi.Stream,
    pair: ?Pair = null,
    stopped_at: ?u64 = null,
    work_at: u64,
    ended: bool = false,
    emitted: bool = false,
};
pub const Trace = struct {
    a: std.mem.Allocator,
    io: std.Io,
    driver: *Driver,
    original: abi.Api,
    stream: abi.Stream,
    request: ?*anyopaque = null,
    finished: bool = false,
    active_thread: std.atomic.Value(u64) = .init(0),
    active: bool = false,
    capturing: bool = false,
    tap_functions: [4]abi.Function = @splat(null),
    generation: u64 = 0,
    entries: std.ArrayList(Entry) = .empty,
    stack: [24]usize = undefined,
    depth: usize = 0,
    dropped: u64 = 0,
    restoration: ledger.Resume = .{},
    prompt_tokens: u64 = 0,
    scratch_rows: u64 = 0,
    scratch_bytes: u64 = 0,
    sink: ?*const fn (ledger.Row) void = null,

    pub fn create(a: std.mem.Allocator, io: std.Io, d: *const Driver, stream: abi.Stream, enabled: bool) !void {
        if (!enabled) return;
        if (owner != null) return error.PrefillTraceAlreadyOwned;
        const t = try a.create(Trace);
        errdefer a.destroy(t);
        t.* = .{ .a = a, .io = io, .driver = @constCast(d), .original = d.api, .stream = stream };
        try t.entries.ensureTotalCapacity(a, 512);
        owner = t;
        API.install(t);
    }
    pub fn close() void {
        const t = owner orelse return;
        t.active_thread.store(@intCast(std.Thread.getCurrentId()), .release);
        t.collect(true);
        t.driver.api = t.original;
        for (t.entries.items) |e| if (e.pair) |p| p.deinit(&t.original);
        t.entries.deinit(t.a);
        const a = t.a;
        a.destroy(t);
        owner = null;
    }
    pub fn enter(t: *Trace, request: *anyopaque, tokens: usize, scratch_rows: usize, scratch_bytes: usize) bool {
        if (t.finished or (t.request != null and t.request != request)) return false;
        t.request = request;
        t.prompt_tokens = tokens;
        t.scratch_rows = scratch_rows;
        t.scratch_bytes = scratch_bytes;
        t.active_thread.store(@intCast(std.Thread.getCurrentId()), .release);
        t.active = true;
        return true;
    }
    pub fn leave(t: *Trace) void {
        t.collect(false);
        t.active = false;
    }
    pub fn owns(t: *Trace) bool {
        return t.active_thread.load(.acquire) == @as(u64, @intCast(std.Thread.getCurrentId()));
    }
    pub fn recording(t: *Trace) bool {
        return t.owns() and t.active and t.depth > 0;
    }
    pub fn now(_: *Trace) u64 {
        var ts: sys.c.timespec = undefined;
        if (sys.c.clock_gettime(sys.c.CLOCK.MONOTONIC, &ts) != 0) return 0;
        return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
    }
    pub fn begin(t: *Trace, row: ledger.Row, stream: abi.Stream, gpu: bool) Ticket {
        if (!t.owns() or !t.active) return null;
        if (t.depth == t.stack.len) {
            t.dropped += 1;
            return null;
        }
        var e: Entry = .{ .row = row, .stream = stream, .work_at = t.generation };
        e.row.sequence = t.entries.items.len;
        e.row.parent = if (t.depth > 0) t.stack[t.depth - 1] else null;
        e.row.wall_begin_ns = t.now();
        e.row.stream = @intFromPtr(stream);
        if (gpu and !t.capturing) {
            e.pair = Pair.init(&t.original, stream);
            if (e.pair == null) e.row.complete = false;
        } else if (gpu) {
            e.row.event_status = .capture_only;
            e.row.graph_mode = .capture;
        }
        t.entries.append(t.a, e) catch {
            if (e.pair) |p| p.deinit(&t.original);
            t.dropped += 1;
            return null;
        };
        const index = t.entries.items.len - 1;
        t.stack[t.depth] = index;
        t.depth += 1;
        return index;
    }
    pub fn end(t: *Trace, ticket: Ticket, result: i32) void {
        const index = ticket orelse return;
        const e = &t.entries.items[index];
        std.debug.assert(t.depth > 0 and t.stack[t.depth - 1] == index);
        t.depth -= 1;
        e.row.wall_end_ns = t.now();
        e.row.wall_ns = e.row.wall_end_ns -| e.row.wall_begin_ns;
        e.row.result = result;
        e.row.complete = e.row.complete and result == 0;
        e.row.restoration = t.restoration;
        e.row.prompt_tokens = t.prompt_tokens;
        e.row.scratch_rows = t.scratch_rows;
        e.row.scratch_bytes = t.scratch_bytes;
        if (e.row.graph_captures > 0) e.row.graph_mode = if (e.row.graph_replays > 0) .capture_and_replay else .capture else if (e.row.graph_replays > 0) e.row.graph_mode = .replay;
        if (e.pair) |p| {
            if (t.capturing) {
                e.row.event_status = .unavailable;
                e.row.complete = false;
            } else if (e.row.launches == 0 and e.row.event_status != .capture_only) e.row.event_status = .not_launched else {
                if (e.stopped_at != t.generation) {
                    if (p.stop(&t.original, e.stream)) {
                        e.stopped_at = t.generation;
                    } else {
                        e.stopped_at = null;
                        e.row.complete = false;
                    }
                }
                e.row.event_status = .pending;
            }
        } else if (e.row.launches > 0 and e.row.event_status != .capture_only) {
            e.row.event_status = .unavailable;
            e.row.complete = false;
        }
        e.row.memory = t.memory();
        e.ended = true;
        t.collect(false);
    }
    pub fn work(t: *Trace, stream: abi.Stream, result: abi.Result) void {
        if (!t.recording()) return;
        if (result != 0) {
            t.dropped += 1;
            return;
        }
        if (t.capturing) return;
        t.generation += 1;
        for (t.stack[0..t.depth]) |index| {
            const e = &t.entries.items[index];
            e.row.launches += 1;
            if (e.stream != stream) e.row.complete = false;
        }
    }
    pub fn graph(t: *Trace, capture: bool) void {
        if (!t.recording()) return;
        for (t.stack[0..t.depth]) |index| {
            const row = &t.entries.items[index].row;
            if (capture) row.graph_captures += 1 else row.graph_replays += 1;
        }
    }
    pub fn beforeWait(t: *Trace, stream: ?abi.Stream) void {
        if (!t.recording() or t.capturing) return;
        for (t.stack[0..t.depth]) |index| {
            const e = &t.entries.items[index];
            if (stream) |s| if (s != e.stream) continue;
            if (e.pair) |p| if (p.stop(&t.original, e.stream)) {
                e.stopped_at = t.generation;
            } else {
                e.row.complete = false;
            };
        }
    }
    pub fn collect(t: *Trace, final: bool) void {
        if (!t.owns() or t.capturing) return;
        for (t.entries.items) |*e| {
            if (!e.ended or e.emitted) continue;
            if (e.row.event_status == .pending) {
                if (e.pair) |p| if (e.stopped_at != null) if (p.elapsed(&t.original)) |ns| {
                    e.row.cuda_event_ns = ns;
                    e.row.event_status = .complete;
                };
                if (e.row.event_status == .pending) {
                    if (!final) continue;
                    e.row.event_status = .unavailable;
                    e.row.complete = false;
                }
            }
            e.row.dropped_stages = t.dropped;
            e.row.complete = e.row.complete and t.dropped == 0;
            if (t.sink) |sink| sink(e.row) else std.debug.print("{f}", .{e.row});
            e.emitted = true;
            if (e.pair) |p| p.deinit(&t.original);
            e.pair = null;
        }
    }
    fn memory(t: *Trace) ledger.Memory {
        var m: ledger.Memory = .{};
        if (sys.process_resident.read(t.io)) |physical| {
            m.rss_bytes = physical.resident_bytes;
            m.physical_peak_bytes = physical.peak_bytes;
        }
        const used = @import("memory.zig").usage(false);
        m.device_bytes = used.device;
        m.device_peak_bytes = used.peak;
        m.pinned_bytes = used.host;
        m.pinned_peak_bytes = used.host_peak;
        return m;
    }
};
