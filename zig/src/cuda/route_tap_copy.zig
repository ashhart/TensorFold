//! Async device-to-host copy of one layer's global top-k into a pinned host ring (rank 0, when live).
const tap = @import("core").route_tap;
const Event = @import("stream.zig").Event;
const HostBuffer = @import("memory.zig").HostBuffer;

var ids_buf: ?HostBuffer = null;
var wts_buf: ?HostBuffer = null;
var ev: ?Event = null;
var pending: u32 = 0;

fn pin(s: anytype) void {
    if (ids_buf != null) return;
    const n = tap.ring_slots * tap.max_ids * 4;
    const ids = HostBuffer.alloc(s.d, n) catch return;
    const wts = HostBuffer.alloc(s.d, n) catch {
        var b = ids;
        b.free();
        return;
    };
    ids_buf = ids;
    wts_buf = wts;
    tap.adoptHost(ids.slice(i32), wts.slice(f32));
    ev = Event.init(s.d, false) catch null;
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

/// Copy `rows * slots` int32 picks and fp32 weights from device pointers, before any localisation.
pub fn afterPick(s: anytype, picks: u64, wts: u64, rows: usize, slots: usize) void {
    if (!tap.live()) return;
    pin(s);
    const n: u16 = @intCast(@min(rows * slots, tap.max_ids));
    if (n == 0) return;
    const sl = tap.reserve(n) orelse return;
    const nb = @as(usize, n) * 4;
    const ids_ok = s.d.check(s.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(sl.ids.ptr), picks, nb, s.handle), "cuMemcpyDtoHAsync");
    const wts_ok = s.d.check(s.d.api.cuMemcpyDtoHAsync_v2(@ptrCast(sl.weights.ptr), wts, nb, s.handle), "cuMemcpyDtoHAsync");
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
