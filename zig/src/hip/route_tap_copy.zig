//! Async device-to-host copy of the global top-k before localisation rewrites local ids.
const tap = @import("core").route_tap;
const runtime = @import("runtime.zig");
const abi = @import("abi.zig");
const Event = @import("stream.zig").Event;
const HostBuffer = @import("memory.zig").HostBuffer;

var ids_buf: ?HostBuffer = null;
var wts_buf: ?HostBuffer = null;
var ev: ?Event = null;
var pending: u32 = 0;

fn pin(r: *const runtime.Runtime) void {
    if (ids_buf != null) return;
    const n = tap.ring_slots * tap.max_ids * 4;
    const ids = HostBuffer.alloc(r, n) catch return;
    const wts = HostBuffer.alloc(r, n) catch {
        var b = ids;
        b.free();
        return;
    };
    ids_buf = ids;
    wts_buf = wts;
    tap.adoptHost(ids.slice(i32), wts.slice(f32));
    ev = Event.init(r, false) catch null;
    tap.onStop(&release);
}

fn release() void {
    if (ev) |*e| {
        e.synchronize() catch {};
        tap.publishTo(pending);
        e.deinit();
        ev = null;
    }
    if (ids_buf) |*b| {
        b.free();
        ids_buf = null;
    }
    if (wts_buf) |*b| {
        b.free();
        wts_buf = null;
    }
    pending = 0;
}

pub fn afterSelect(r: *const runtime.Runtime, stream: abi.Stream, picks: u64, wts: u64, rows: usize, slots: usize) void {
    if (!tap.live()) return;
    pin(r);
    const n: u16 = @intCast(@min(rows * slots, tap.max_ids));
    if (n == 0) return;
    const sl = tap.reserve(n) orelse return;
    const nb = @as(usize, n) * 4;
    const ids_ok = runtime.check(r.api.hipMemcpyDtoHAsync(@ptrCast(sl.ids.ptr), @ptrFromInt(picks), nb, stream));
    const wts_ok = runtime.check(r.api.hipMemcpyDtoHAsync(@ptrCast(sl.weights.ptr), @ptrFromInt(wts), nb, stream));
    const s = @import("stream.zig").Stream{ .r = r, .handle = stream };
    if (ev) |e| e.record(s) catch {};
    if (ids_ok) |_| {} else |_| {
        if (ev) |e| e.synchronize() catch {};
        tap.dropSlot(sl);
        pending = tap.reservedCount();
        return;
    }
    if (wts_ok) |_| {} else |_| {
        if (ev) |e| e.synchronize() catch {};
        tap.dropSlot(sl);
        pending = tap.reservedCount();
        return;
    }
    pending = tap.reservedCount();
}

/// Wait the event behind this step's copies, then publish only those slots.
pub fn land() void {
    if (!tap.live()) return;
    const end = pending;
    if (end <= tap.publishedCount()) return;
    if (ev) |e| e.synchronize() catch {};
    tap.publishTo(end);
}
