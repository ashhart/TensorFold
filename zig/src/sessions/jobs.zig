//! Parks and restores off the round: a tier thread runs each job's file I/O until the plan settles it (Store.settle).

const std = @import("std");
const Io = std.Io;
const disk_mod = @import("disk.zig");
const Disk = disk_mod.Disk;

pub const Kind = enum { park, restore };
pub const State = enum(u8) { queued, running, done, failed };

pub const Job = struct {
    kind: Kind,
    /// the store's entry
    entry: u32,
    state: std.atomic.Value(State) = .init(.queued),
    err: ?anyerror = null,
    started_ns: i96 = 0,
    elapsed_ns: u64 = 0,
    /// park: the file to write (owned buffers below) and its size once written
    src: disk_mod.WriteSrc = undefined,
    size: u64 = 0,
    /// restore: files base first, the sink, the bounded state read
    extents: []disk_mod.Extent = &.{},
    sink: disk_mod.ReadSink = undefined,
    restored: ?disk_mod.Restored = null,
    /// restore: the slot it fills
    slot: u32 = 0,
    /// pool pages the job holds until settled
    held: []u32 = &.{},
    /// owned buffers
    ids: []i32 = &.{},
    lists: [][]u32 = &.{},
    blobs: []disk_mod.Blob = &.{},

    pub fn done(j: *const Job) bool {
        const s = j.state.load(.acquire);
        return s == .done or s == .failed;
    }

    pub fn deinit(j: *Job, gpa: std.mem.Allocator) void {
        gpa.free(j.held);
        gpa.free(j.ids);
        for (j.lists) |l| gpa.free(l);
        gpa.free(j.lists);
        for (j.blobs) |b| {
            gpa.free(b.name);
            gpa.free(b.bytes);
        }
        gpa.free(j.blobs);
        gpa.free(j.extents);
        if (j.restored) |*r| r.deinit(gpa);
        gpa.destroy(j);
    }

    fn run(j: *Job, d: *Disk, io: Io) void {
        j.state.store(.running, .release);
        const t0 = Io.Clock.awake.now(io).toNanoseconds();
        j.err = switch (j.kind) {
            .park => if (d.writeFile(&j.src)) |n| blk: {
                j.size = n;
                break :blk null;
            } else |e| e,
            .restore => j.restore(d),
        };
        j.elapsed_ns = @intCast(Io.Clock.awake.now(io).toNanoseconds() - t0);
        j.state.store(if (j.err == null) .done else .failed, .release);
    }

    fn restore(j: *Job, d: *Disk) ?anyerror {
        for (j.extents, 0..) |e, i| {
            const last = i + 1 == j.extents.len;
            const r = d.readFile(e, &j.sink, last) catch |err| return err;
            if (last) j.restored = r;
        }
        j.sink.store.publish() catch |err| return err;
        return null;
    }
};

/// The tier thread and its queue (null thread: jobs run inline).
pub const Runner = struct {
    gpa: std.mem.Allocator,
    io: Io,
    disk: *Disk,
    thread: ?std.Thread = null,
    queue: std.ArrayList(*Job) = .empty,
    head: usize = 0,
    mutex: Io.Mutex = .init,
    cond: Io.Condition = .init,
    stop: bool = false,

    pub fn init(gpa: std.mem.Allocator, io: Io, d: *Disk) Runner {
        return .{ .gpa = gpa, .io = io, .disk = d };
    }

    /// Starts the tier thread: jobs then run off the caller's thread.
    pub fn startThread(r: *Runner) !void {
        if (r.thread != null) return;
        r.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, loop, .{r});
    }

    pub fn deinit(r: *Runner) void {
        if (r.thread) |t| {
            r.mutex.lockUncancelable(r.io);
            r.stop = true;
            r.cond.broadcast(r.io);
            r.mutex.unlock(r.io);
            t.join();
        }
        r.queue.deinit(r.gpa);
    }

    pub fn submit(r: *Runner, j: *Job) !void {
        if (r.thread == null) return j.run(r.disk, r.io);
        r.mutex.lockUncancelable(r.io);
        defer r.mutex.unlock(r.io);
        if (r.head == r.queue.items.len) {
            r.queue.clearRetainingCapacity();
            r.head = 0;
        }
        try r.queue.append(r.gpa, j);
        r.cond.signal(r.io);
    }

    /// Waits for a job (the plan settles it next).
    pub fn wait(r: *Runner, j: *Job) void {
        while (!j.done()) {
            r.mutex.lockUncancelable(r.io);
            defer r.mutex.unlock(r.io);
            if (!j.done()) r.cond.waitTimeout(r.io, &r.mutex, .{ .duration = .{ .raw = .fromMilliseconds(5), .clock = .awake } }) catch {};
        }
    }

    fn loop(r: *Runner) void {
        while (true) {
            const j = blk: {
                r.mutex.lockUncancelable(r.io);
                defer r.mutex.unlock(r.io);
                while (r.head == r.queue.items.len and !r.stop) r.cond.waitUncancelable(r.io, &r.mutex);
                if (r.head == r.queue.items.len) return;
                r.head += 1;
                break :blk r.queue.items[r.head - 1];
            };
            j.run(r.disk, r.io);
            r.mutex.lockUncancelable(r.io);
            r.cond.broadcast(r.io);
            r.mutex.unlock(r.io);
        }
    }
};
