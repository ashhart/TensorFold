//! Staggered prompt segments, for any family: one prompt chunk split into segments, each on its own Metal queue.
//! Segment k runs each layer's mixer after segment k-1's (an MTLEvent between the queues) and the rest of its layer
//! beside the other segments, so one segment's scans and glue overlap another's matrix work. Each segment does what a
//! serial chunk of its rows does, so the output equals the chunks run in order. Measured on Flash Next 6-bit (M5
//! Ultra): two segments +9% at 8k-64k, three +2%, four time out.
const std = @import("std");
const mtl = @import("metal");
const Fence = @import("fence.zig").Fence;
const stagger = @import("stagger.zig");

/// Segments one chunk can take.
pub const MAX = stagger.MAX;

/// Where layer i waits for the segment before (core/stagger.zig).
pub const Wait = stagger.Wait;

/// A segment's command buffer and open encoder; hooks encode into `enc` and leave it open.
pub const Lane = struct {
    k: usize,
    cb: mtl.CommandBuffer,
    enc: mtl.ComputeEncoder,
    fence: Fence, // orders the segment's encoders: buffers are untracked, so encoders could overlap
};

pub const rows = stagger.rows;
pub const start = stagger.start;
pub const Call = stagger.Call;
pub const next = stagger.next;

/// Runs one chunk's segments, segment k on queues[k], over `layers` layers, through the family's hooks on `fam`:
///   begin(lane), pre(lane, i), mixer(lane, i), post(lane, i), finish(lane): the segment's work into lane.enc;
///   wait(i) Wait: where layer i waits for the segment before; handoff(lane, i): right after that wait.
/// Hooks leave the encoder open and commit nothing; what segment k+1 reads of segment k's must be written by k's
/// mixer or earlier. Commits every segment and waits for all, failed or not; returns the longest segment's GPU seconds.
/// One queue is a serial chunk.
pub fn run(device: mtl.Device, queues: []const mtl.Queue, layers: usize, mode: mtl.DispatchType, fam: anytype) !f64 {
    const n = queues.len;
    if (n == 0 or n > MAX) return error.Segments;
    var evs: [MAX - 1]mtl.Event = undefined;
    var n_evs: usize = 0;
    defer for (evs[0..n_evs]) |e| e.deinit();
    while (n_evs + 1 < n) : (n_evs += 1) evs[n_evs] = try device.event();
    var lanes: [MAX]Lane = undefined;
    var n_lanes: usize = 0;
    defer for (lanes[0..n_lanes]) |l| l.fence.deinit();
    for (queues) |q| {
        const fence = try Fence.init(device);
        const cb = q.commandBuffer();
        lanes[n_lanes] = .{ .k = n_lanes, .cb = cb, .enc = cb.compute(mode), .fence = fence };
        n_lanes += 1;
    }
    const X = struct {
        fam: @TypeOf(fam),
        lanes: []Lane,
        evs: []const mtl.Event,
        mode: mtl.DispatchType,

        /// Ends the lane's encoder, waits for (or signals) `value` between its command buffer's encoders, and opens
        /// the next encoder behind the fence.
        fn sync(x: *const @This(), l: *Lane, ev: mtl.Event, value: u64, wait_: bool) void {
            l.fence.update(l.enc);
            l.enc.end();
            if (wait_) l.cb.waitForEvent(ev, value) else l.cb.signalEvent(ev, value);
            l.enc = l.cb.compute(x.mode);
            l.fence.wait(l.enc);
        }
        pub fn begin(x: *const @This(), k: usize) !void {
            try x.fam.begin(&x.lanes[k]);
        }
        pub fn wait(x: *const @This(), i: usize) Wait {
            return x.fam.wait(i);
        }
        pub fn waitFor(x: *const @This(), k: usize, i: usize) !void {
            x.sync(&x.lanes[k], x.evs[k - 1], i + 1, true);
            try x.fam.handoff(&x.lanes[k], i);
        }
        pub fn signal(x: *const @This(), k: usize, i: usize) !void {
            x.sync(&x.lanes[k], x.evs[k], i + 1, false);
        }
        pub fn pre(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.pre(&x.lanes[k], i);
        }
        pub fn mixer(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.mixer(&x.lanes[k], i);
        }
        pub fn post(x: *const @This(), k: usize, i: usize) !void {
            try x.fam.post(&x.lanes[k], i);
        }
        pub fn finish(x: *const @This(), k: usize) !void {
            try x.fam.finish(&x.lanes[k]);
        }
    };
    const x: X = .{ .fam = fam, .lanes = lanes[0..n], .evs = evs[0..n_evs], .mode = mode };
    stagger.drive(n, layers, &x) catch |err| { // nothing committed yet: close the open encoders, drop the command buffers
        for (lanes[0..n]) |l| l.enc.end();
        return err;
    };
    for (lanes[0..n]) |l| l.enc.end();
    for (lanes[0..n]) |l| l.cb.commit();
    for (lanes[0..n]) |l| l.cb.wait(); // every segment ends before a failure returns and its buffers are reused
    var gpu: f64 = 0;
    for (lanes[0..n]) |l| {
        if (l.cb.failure()) |msg| {
            std.log.err("command buffer failed: {s}", .{msg});
            return error.GpuFailed;
        }
        gpu = @max(gpu, l.cb.gpuSeconds());
    }
    return gpu;
}

test {
    _ = stagger;
}
