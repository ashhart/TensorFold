//! A hand-off's pool pages converted while the prompt prefills: a thread reads each finished segment's pages back.
const std = @import("std");
const cuda = @import("cuda");
const cx = @import("cuda_export.zig");
const ex = @import("handoff/export.zig");
const plan = @import("handoff/plan.zig");
const kvs = @import("cuda_kv.zig");
const log = std.log.scoped(.dsv41);

/// The bytes one part's pages are read into: two parts' worth, the format's part size being 2 MiB.
const stage_bytes: usize = 2 * plan.chunk_bytes;

/// One finished segment: the fence after its writes, the position the prompt stands at, and the slot's physical pages.
const Job = struct { fence: u64, pos: u64, pages: []u32 };

pub const Stream = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    built: *cx.Built,
    kx: *kvs.Kv,
    thread: ?std.Thread = null,
    lock: std.atomic.Mutex = .unlocked, // `jobs`: the engine thread queues, the worker takes
    jobs: std.ArrayList(Job) = .empty,
    queued: std.atomic.Value(u32) = .init(0), // jobs queued or under way
    closing: std.atomic.Value(bool) = .init(false),
    broken: std.atomic.Value(bool) = .init(false),
    converted: []bool,

    /// A stream feeding `built`'s paged parts from `kx`'s pool; the pool's pages are read for each `progress`.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, kx: *kvs.Kv, built: *cx.Built) !*Stream {
        const s = try gpa.create(Stream);
        errdefer gpa.destroy(s);
        s.* = .{ .gpa = gpa, .io = io, .built = built, .kx = kx, .converted = try gpa.alloc(bool, built.parts.len) };
        errdefer gpa.free(s.converted);
        @memset(s.converted, false);
        s.thread = try std.Thread.spawn(.{}, run, .{s});
        return s;
    }

    /// Stops the worker and frees the stream; its export's parts are no longer written.
    pub fn stop(s: *Stream) void {
        s.closing.store(true, .release);
        if (s.thread) |t| t.join();
        for (s.jobs.items) |j| s.gpa.free(j.pages);
        s.jobs.deinit(s.gpa);
        s.gpa.free(s.converted);
        s.gpa.destroy(s);
    }

    /// The engine thread: the prompt stands at `pos` and the GPU has the writes of everything before it queued.
    pub fn progress(s: *Stream, pos: u64) void {
        if (s.broken.load(.acquire)) return;
        s.queue(pos) catch |err| {
            log.warn("handoff stream: a segment's pages were not queued ({t}); the cut reads them", .{err});
            s.broken.store(true, .release);
        };
    }

    fn queue(s: *Stream, pos: u64) !void {
        const kx = s.kx;
        const at = @min(pos, s.built.shape.upto);
        const npages: usize = @intCast((at + kx.pool.page - 1) / kx.pool.page);
        if (kx.slot.len < npages) return error.BadWindow;
        const fence = try kx.dev.store().fence();
        const pages = try s.gpa.dupe(u32, kx.slot.pages[0..npages]);
        errdefer s.gpa.free(pages);
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
        defer s.lock.unlock();
        try s.jobs.append(s.gpa, .{ .fence = fence, .pos = pos, .pages = pages });
        _ = s.queued.fetchAdd(1, .acq_rel);
    }

    /// The cut: waits for the queued segments; whether every paged part is converted (else the cut reads them).
    pub fn drain(s: *Stream) bool {
        while (s.queued.load(.acquire) > 0 and !s.broken.load(.acquire)) std.Io.sleep(s.io, .fromMilliseconds(1), .awake) catch {};
        if (s.broken.load(.acquire)) return false;
        const pg = s.built.exp.progress.?;
        for (s.built.parts, 0..) |p, i| if (paged(p.spec.kind) and !pg.isReady(i)) return false;
        return true;
    }

    fn paged(kind: plan.Kind) bool {
        return kind == .comp or kind == .ixk;
    }

    fn take(s: *Stream) ?Job {
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
        defer s.lock.unlock();
        if (s.jobs.items.len == 0) return null;
        return s.jobs.orderedRemove(0);
    }

    fn run(s: *Stream) void {
        s.work() catch |err| {
            log.warn("handoff stream: stopped ({t}); the cut reads the pages left", .{err});
            s.broken.store(true, .release);
        };
        // jobs queued after a stop or a failure are dropped, so a drain does not wait on them
        while (s.take()) |j| {
            s.gpa.free(j.pages);
            _ = s.queued.fetchSub(1, .acq_rel);
        }
    }

    fn work(s: *Stream) !void {
        const dev = &s.kx.dev;
        try dev.ctx.makeCurrent();
        var stream = try cuda.Stream.init(dev.d, true);
        defer stream.deinit();
        var stage = try cuda.HostBuffer.alloc(dev.d, stage_bytes);
        defer stage.free();
        while (true) {
            const job = s.take() orelse {
                if (s.closing.load(.acquire)) return;
                std.Io.sleep(s.io, .fromMilliseconds(1), .awake) catch {};
                continue;
            };
            defer {
                s.gpa.free(job.pages);
                _ = s.queued.fetchSub(1, .acq_rel);
            }
            if (s.closing.load(.acquire)) return;
            try dev.store().waitFence(job.fence);
            try s.convertReady(job, stream, stage.bytes);
        }
    }

    /// Every paged part whose pages the segment finished, read back and converted, in plan order.
    fn convertReady(s: *Stream, job: Job, stream: cuda.Stream, stage: []u8) !void {
        const b = s.built;
        const shape = b.shape;
        const kx = s.kx;
        const page = kx.pool.page;
        const pg = b.exp.progress.?;
        for (b.parts, 0..) |*p, i| {
            if (s.converted[i] or !paged(p.spec.kind)) continue;
            if (s.closing.load(.acquire)) return;
            const src = shape.geo.sourceIndex(p.spec.layer).?;
            const total = shape.pages(src);
            const end = @min(total, @as(u64, p.spec.first_page) + shape.span(src, p.spec.kind));
            if (job.pos < shape.upto and end * page > job.pos) continue; // its last page is still being written
            const first: usize = p.spec.first_page;
            const fam = (if (p.spec.kind == .comp) kx.layout.compOf(p.spec.layer) else kx.layout.indexOf(p.spec.layer)).?;
            const pb: usize = @intCast(kx.layout.families()[fam].pageBytes(page));
            const n: usize = @intCast(end - first);
            if (n * pb > stage.len or end > job.pages.len) return error.BadWindow;
            var at: usize = 0;
            while (at < n) {
                var run_end = at + 1;
                while (run_end < n and job.pages[first + run_end] == job.pages[first + run_end - 1] + 1) run_end += 1;
                try kx.dev.tensors[fam].downloadAsync(@as(usize, job.pages[first + at]) * pb, stage[at * pb .. run_end * pb], stream.handle);
                at = run_end;
            }
            try stream.synchronize();
            const bytes = stage[0 .. n * pb];
            const paged_view: ex.Paged = .{ .split = false, .ranks = &.{bytes}, .first = first };
            const source: ex.SourceState = .{ .layer = p.spec.layer, .comp = paged_view, .index = paged_view };
            const view: ex.Cuda = .{ .index_format = if (kx.layout.opts.index == .fp8) .fp8 else .bf16, .sources = &.{source}, .swa = &.{}, .carries = &.{} };
            try ex.convert(p, shape, view);
            s.converted[i] = true;
            pg.mark(i);
        }
    }
};
