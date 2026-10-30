//! A queued request's disk state read ahead while other streams run: it waits for the read, then restores from memory.
const std = @import("std");
const api = @import("engine_api.zig");
const LaneHost = @import("lane_host.zig").LaneHost;
const Job = LaneHost.Job;
const prefetch = @import("rows_prefetch.zig");
const write = @import("rows_write.zig");
const Name = @import("rows_chain.zig").Name;

/// A queued job's read ahead: looked for once while others run, and the disk state it waits for.
pub const Ticket = struct { looked: bool = false, name: ?Name = null, writing: ?Name = null, since: i128 = 0 };

/// Reads no queued request names go; then reads start for queued requests a disk state serves (the lock held).
pub fn queue(h: *LaneHost) void {
    const store = h.cache orelse return;
    const r = store.disk orelse return;
    if (r.fetch) |f| f.sweep(h, named);
    if (h.admitted.items.len == 0) return; // alone: its state is read where it is admitted, as before
    var fetcher: ?*prefetch.Prefetch = null;
    for (h.queued.items) |job| {
        if (job.fetch.looked or job.request.background or !keyed(job.request)) continue; // warm passes read where they run
        const f = fetcher orelse prefetch.ensure(r, h.io, .{ .ctx = h, .call = wake }) orelse return;
        fetcher = f;
        if (!f.on()) return; // TENSORFOLD_RESTORE_PREFETCH=0: nothing is looked up or read here
        write.reap(r);
        const q = job.request;
        if (job.fetch.writing) |name| {
            if (r.worker) |w| if (w.names(name)) {
                if (write.nowNs() - job.fetch.since < f.limits.wait_ns) continue;
                job.fetch.looked = true;
                job.fetch.writing = null;
                continue;
            };
            job.fetch.writing = null;
        } else if (prefetch.writing(store, q.cacheKey(), q.chunks, h.spans(q))) |name| {
            job.fetch.writing = name;
            job.fetch.since = write.nowNs();
            continue;
        }
        job.fetch.looked = true;
        job.fetch.name = f.start(r, prefetch.choose(store, q.cacheKey(), q.chunks, h.spans(q)) orelse continue);
    }
}

/// Whether queued `job` waits for its read: still going and inside the bound (the lock held).
pub fn held(h: *LaneHost, job: *const Job) bool {
    const r = (h.cache orelse return false).disk orelse return false;
    const f = r.fetch orelse return false;
    if (job.fetch.writing) |name| {
        if (write.nowNs() - job.fetch.since >= f.limits.wait_ns) return false;
        return if (r.worker) |w| w.names(name) else false;
    }
    const name = job.fetch.name orelse return false;
    return f.pendingSince(name, if (job.fetch.since == 0) null else job.fetch.since);
}

/// Stop notifications after engine drain; a blocked read keeps only its private memory.
pub fn stop(h: *LaneHost) void {
    const f = ((h.cache orelse return).disk orelse return).fetch orelse return;
    f.stopAt(write.nowNs());
}

fn named(h: *LaneHost, name: Name) bool {
    for (h.queued.items) |j| if (j.fetch.name) |n| if (std.mem.eql(u8, &n, &name)) return true;
    return false;
}

/// A request the host's lookup keys by its tokens alone (no pictures), as `lane_stream.open` looks it up.
fn keyed(r: *const api.Request) bool {
    return r.embedding == null and r.prompt_logprobs == null and !r.untaught and r.images.len == 0 and r.cacheKey().len > 0;
}

/// A read finished: the engine thread, perhaps idle with every queued request waiting, looks again.
fn wake(ctx: *anyopaque) void {
    const h: *LaneHost = @ptrCast(@alignCast(ctx));
    h.lock();
    defer h.unlock();
    h.wake.signal(h.io);
}

test {
    _ = @import("lane_host_fetch_test.zig");
}
