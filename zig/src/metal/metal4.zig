//! Metal 4 under command.zig's calls: slots of command buffers, argument tables, constants pages, barriers, an event.
const std = @import("std");
const sys = @import("sys");
const objc = @import("objc.zig");
const types = @import("types.zig");
const Timeline = @import("metal4_timeline.zig").Timeline;

const Id = objc.Id;
const heap = std.heap.c_allocator;

pub const stage_dispatch: usize = 1 << 27;
pub const stage_blit: usize = 1 << 28;
const queue_stages = stage_dispatch | stage_blit;
const flush_device: usize = 1;

/// Command buffers a queue keeps; a slot is begun again once its last commit has completed.
pub const ring = 64;
const page_bytes: usize = 256 << 10;
const max_pages = 32;
const max_events = 4;
/// Feedback kept a slot, by commit: a reader is never more than one commit behind.
const history = 4;
/// How long a reader waits for a completed commit's feedback: spins, then naps of 10 us (2 s in all).
const spins = 4096;
const feedback_naps = 200_000;

/// The Objective-C block ABI for the commit feedback handler: a global block, so nothing copies or frees it.
const BlockDescriptor = extern struct { reserved: c_ulong, size: c_ulong, signature: [*:0]const u8 };
const Block = extern struct {
    isa: *const anyopaque,
    flags: c_int,
    reserved: c_int,
    invoke: *const fn (*const Block, Id) callconv(.c) void,
    descriptor: *const BlockDescriptor,
    slot: *Slot,
};
extern const _NSConcreteGlobalBlock: anyopaque;
const block_global: c_int = 1 << 28;
const block_signature: c_int = 1 << 30;
const descriptor: BlockDescriptor = .{ .reserved = 0, .size = @sizeOf(Block), .signature = "v16@?0@8" };

const Page = struct { buf: Id, ptr: [*]u8, addr: u64 };
const Event = struct { id: Id, value: u64 };

pub const Slot = struct {
    q: *State,
    cb: Id,
    allocator: Id,
    table: Id,
    block: Block,
    /// how many times the slot was begun: a handle from an earlier use finds its work done
    use: u32 = 0,
    commits: u32 = 0,
    /// the event value this use's commit signals; 0 before it commits
    value: u64 = 0,
    open: bool = false,
    serial: bool = false,
    blit: bool = false,
    dirty: bool = false,
    pages: [max_pages]?Page = @splat(null),
    page: usize = 0,
    bump: usize = 0,
    waits: [max_events]Event = undefined,
    nwaits: usize = 0,
    signals: [max_events]Event = undefined,
    nsignals: usize = 0,
    /// commit feedbacks received; the newest ones' GPU span and whether they failed
    seen: std.atomic.Value(u32) = .init(0),
    start: [history]f64 = @splat(0),
    end: [history]f64 = @splat(0),
    bad: [history]bool = @splat(false),

    /// A compute encoder on this slot's command buffer, after every earlier command of the queue.
    pub fn compute(s: *Slot, serial: bool) Id {
        const e = objc.msg(Id, s.cb, "computeCommandEncoder", .{});
        objc.msg(void, e, "setArgumentTable:", .{s.table});
        objc.msg(void, e, "barrierAfterQueueStages:beforeStages:visibilityOptions:", .{ queue_stages, queue_stages, flush_device });
        s.serial = serial;
        s.blit = false;
        s.dirty = false;
        if (s.q.timeline) |t| t.started(e);
        return e;
    }

    /// Copies and fills, in order: Metal 4 encodes them on a compute encoder.
    pub fn blitEncoder(s: *Slot) Id {
        const e = s.compute(true);
        s.blit = true;
        return e;
    }

    pub fn bind(s: *Slot, buffer: Id, offset: usize, index: usize) void {
        objc.msg(void, s.table, "setAddress:atIndex:", .{ s.q.address(buffer) + offset, index });
    }

    /// Constants copied into this use's pages, read by address.
    pub fn bytes(s: *Slot, data: []const u8, index: usize) void {
        const len = std.mem.alignForward(usize, @max(data.len, 1), 64);
        if (len > page_bytes) std.debug.panic("metal 4: {d} bytes of constants in one call", .{data.len});
        if (s.bump + len > page_bytes) {
            s.page += 1;
            s.bump = 0;
        }
        const p = s.pageAt(s.page);
        @memcpy(p.ptr[s.bump..][0..data.len], data);
        objc.msg(void, s.table, "setAddress:atIndex:", .{ p.addr + s.bump, index });
        s.bump += len;
    }

    fn pageAt(s: *Slot, i: usize) Page {
        if (i >= max_pages) std.debug.panic("metal 4: more than {d} KiB of constants in one command buffer", .{max_pages * page_bytes >> 10});
        if (s.pages[i]) |p| return p;
        const buf = objc.msg(?Id, s.q.device, "newBufferWithLength:options:", .{ page_bytes, types.ResourceOptions.shared }) orelse
            std.debug.panic("metal 4: no buffer for constants", .{});
        const p: Page = .{ .buf = buf, .ptr = objc.msg([*]u8, buf, "contents", .{}), .addr = s.q.address(buf) };
        s.pages[i] = p;
        return p;
    }

    pub fn pipeline(s: *Slot, id: Id) void {
        if (s.q.timeline) |t| t.piped(id);
    }

    pub fn dispatched(s: *Slot, e: Id) void {
        s.dirty = true;
        if (s.q.timeline) |t| t.dispatched();
        if (s.serial) s.barrier(e);
    }

    pub fn dispatchIndirect(s: *Slot, e: Id, args: Id, offset: usize, threads: types.Size) void {
        objc.msg(void, e, "dispatchThreadgroupsWithIndirectBuffer:threadsPerThreadgroup:", .{ s.q.address(args) + offset, threads });
        s.dispatched(e);
    }

    /// Later commands of this encoder see every write before it; with nothing since the last one it does nothing.
    pub fn barrier(s: *Slot, e: Id) void {
        if (!s.dirty) return;
        s.dirty = false;
        const st = if (s.blit) stage_blit else stage_dispatch;
        objc.msg(void, e, "barrierAfterEncoderStages:beforeEncoderStages:visibilityOptions:", .{ st, st, flush_device });
        if (s.q.timeline) |t| t.stage(e);
    }

    pub fn copy(s: *Slot, e: Id, from: Id, from_offset: usize, to: Id, to_offset: usize, size: usize) void {
        _ = s.q.address(from);
        _ = s.q.address(to);
        objc.msg(void, e, "copyFromBuffer:sourceOffset:toBuffer:destinationOffset:size:", .{ from, from_offset, to, to_offset, size });
        s.dispatched(e);
    }

    pub fn fill(s: *Slot, e: Id, buffer: Id, offset: usize, size: usize, value: u8) void {
        _ = s.q.address(buffer);
        objc.msg(void, e, "fillBuffer:range:value:", .{ buffer, types.Range{ .location = offset, .length = size }, value });
        s.dispatched(e);
    }

    pub fn ending(s: *Slot, e: Id) void {
        if (s.q.timeline) |t| if (s.dirty) t.ended(e);
    }

    /// Event waits and signals at the command buffer's edges: Metal 4 puts them on the queue, around the commit.
    pub fn waitEvent(s: *Slot, event: Id, value: u64) void {
        if (s.nwaits == max_events) std.debug.panic("metal 4: more than {d} event waits in one command buffer", .{max_events});
        s.waits[s.nwaits] = .{ .id = event, .value = value };
        s.nwaits += 1;
    }

    pub fn signalEvent(s: *Slot, event: Id, value: u64) void {
        if (s.nsignals == max_events) std.debug.panic("metal 4: more than {d} event signals in one command buffer", .{max_events});
        s.signals[s.nsignals] = .{ .id = event, .value = value };
        s.nsignals += 1;
    }

    pub fn commit(s: *Slot) void {
        s.q.commit(s);
    }

    /// Blocks until use `use` of this slot completed (at once for a later use's handle or one never committed).
    pub fn wait(s: *Slot, use: u32) void {
        if (use != s.use or s.value == 0) return;
        s.q.reach(s.value);
        s.q.drain();
    }

    /// The GPU's start and end of use `use`, in host seconds; null when gone or, unless `block`, not yet reported.
    pub fn times(s: *Slot, use: u32, block: bool) ?[2]f64 {
        if (use != s.use or s.value == 0) return null;
        const k = s.commits;
        var tries: u32 = 0;
        while (s.seen.load(.acquire) < k) : (tries += 1) {
            if (!block or tries == 100_000) return null;
            nap(10_000);
        }
        const i = (k - 1) % history;
        return .{ s.start[i], s.end[i] };
    }

    /// A failure the queue has not reported yet; this use's own feedback (apart from its GPU event) is awaited first.
    pub fn failure(s: *Slot, use: u32) ?[*:0]const u8 {
        const q = s.q;
        if (use == s.use and s.value != 0 and !s.fed(s.commits, feedback_naps)) return "metal 4: a command buffer's feedback never arrived";
        if (q.failures.load(.acquire) == q.reported) return null;
        q.reported = q.failures.load(.acquire);
        return @ptrCast(&q.failure_text);
    }

    /// Waits for commit `k`'s feedback: a short spin (it follows the GPU closely), then up to `naps` naps of 10 us.
    fn fed(s: *const Slot, k: u32, naps: u32) bool {
        var i: u32 = 0;
        while (s.seen.load(.acquire) < k) : (i += 1) {
            if (i < spins) {
                std.atomic.spinLoopHint();
                continue;
            }
            if (i - spins >= naps) return false;
            nap(10_000);
        }
        return true;
    }

    pub fn status(s: *Slot, use: u32) types.CommandBufferStatus {
        if (use != s.use) return .completed;
        if (s.value == 0) return if (s.open) .not_enqueued else .completed;
        if (objc.msg(u64, s.q.event, "signaledValue", .{}) < s.value) return .committed;
        const k = s.commits;
        if (s.seen.load(.acquire) >= k and s.bad[(k - 1) % history]) return .@"error";
        return .completed;
    }

    fn feedback(b: *const Block, fb: Id) callconv(.c) void {
        const s = b.slot;
        const n = s.seen.load(.monotonic);
        const i = n % history;
        s.start[i] = objc.msg(f64, fb, "GPUStartTime", .{});
        s.end[i] = objc.msg(f64, fb, "GPUEndTime", .{});
        const err = objc.msg(?Id, fb, "error", .{});
        s.bad[i] = err != null;
        if (err) |e| s.q.fail(e);
        s.seen.store(n + 1, .release);
    }
};

/// Buffers bound so far and their GPU addresses: open addressing, read without the lock and written under it.
const Table = struct {
    keys: []std.atomic.Value(usize),
    addrs: []u64,
    used: usize = 0,

    const tomb: usize = 1;

    fn init(n: usize) *Table {
        const t = heap.create(Table) catch @panic("metal 4: out of memory");
        t.* = .{ .keys = heap.alloc(std.atomic.Value(usize), n) catch @panic("metal 4: out of memory"), .addrs = heap.alloc(u64, n) catch @panic("metal 4: out of memory") };
        for (t.keys) |*k| k.* = .init(0);
        return t;
    }

    fn deinit(t: *Table) void {
        heap.free(t.keys);
        heap.free(t.addrs);
        heap.destroy(t);
    }

    fn home(t: *const Table, k: usize) usize {
        return @as(usize, @truncate(((k >> 4) *% 0x9E3779B97F4A7C15) >> 24)) & (t.keys.len - 1);
    }

    fn find(t: *const Table, k: usize) ?usize {
        var i = t.home(k);
        while (true) : (i = (i + 1) & (t.keys.len - 1)) {
            const key = t.keys[i].load(.acquire);
            if (key == k) return i;
            if (key == 0) return null;
        }
    }

    /// Under the lock, a key that is not there: its address first, so a reader that sees the key sees the address.
    fn put(t: *Table, k: usize, addr: u64) void {
        var i = t.home(k);
        while (true) : (i = (i + 1) & (t.keys.len - 1)) {
            const key = t.keys[i].load(.monotonic);
            if (key != 0 and key != tomb) continue;
            if (key == 0) t.used += 1;
            t.addrs[i] = addr;
            t.keys[i].store(k, .release);
            return;
        }
    }
};

/// Queues whose buffers Buffer.deinit must hand back.
var registry: [8]std.atomic.Value(?*State) = @splat(.init(null));
var registered: std.atomic.Value(u32) = .init(0);

/// A buffer's owner let it go: queues that bound it release it from their sets once their work in flight is done.
pub fn forget(buffer: Id) void {
    if (registered.load(.acquire) == 0) return;
    for (&registry) |*r| if (r.load(.acquire)) |q| q.drop(buffer);
}

fn nap(ns: u64) void {
    const t: std.c.timespec = .{ .sec = 0, .nsec = @intCast(ns) };
    _ = sys.c.nanosleep(&t, null);
}

pub const State = struct {
    device: Id,
    queue: Id,
    event: Id,
    set: Id,
    lock_word: std.c.os_unfair_lock = .{},
    tick: u64 = 0,
    next: usize = 0,
    /// slots begun and not yet committed
    unsent: usize = 0,
    table: std.atomic.Value(*Table),
    retired: std.ArrayList(*Table) = .empty,
    /// buffers whose owners freed them: still in the set until the queue has nothing in flight
    dropped: std.ArrayList(Id) = .empty,
    set_dirty: bool = false,
    set_grew: bool = false,
    failures: std.atomic.Value(u32) = .init(0),
    reported: u32 = 0,
    failure_text: [256]u8 = @splat(0),
    timeline: ?*Timeline = null,
    slots: [ring]Slot = undefined,

    /// A Metal 4 queue on `device` with its slots and its residency set, or null where Metal 4 is missing.
    pub fn open(device: Id, timeline_rounds: u32) ?*State {
        const queue = objc.msg(?Id, device, "newMTL4CommandQueue", .{}) orelse return null;
        const q = heap.create(State) catch @panic("metal 4: out of memory");
        q.* = .{
            .device = device,
            .queue = queue,
            .event = objc.msg(?Id, device, "newSharedEvent", .{}) orelse @panic("metal 4: no shared event"),
            .set = newSet(device),
            .table = .init(Table.init(4096)),
        };
        const td = objc.msg(Id, objc.msg(Id, objc.class("MTL4ArgumentTableDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(td);
        objc.msg(void, td, "setMaxBufferBindCount:", .{@as(usize, 31)});
        for (&q.slots) |*s| {
            var err: ?Id = null;
            s.* = .{
                .q = q,
                .cb = objc.msg(?Id, device, "newCommandBuffer", .{}) orelse @panic("metal 4: no command buffer"),
                .allocator = objc.msg(?Id, device, "newCommandAllocator", .{}) orelse @panic("metal 4: no command allocator"),
                .table = objc.msg(?Id, device, "newArgumentTableWithDescriptor:error:", .{ td, &err }) orelse @panic("metal 4: no argument table"),
                .block = .{ .isa = &_NSConcreteGlobalBlock, .flags = block_global | block_signature, .reserved = 0, .invoke = Slot.feedback, .descriptor = &descriptor, .slot = s },
            };
        }
        objc.msg(void, queue, "addResidencySet:", .{q.set});
        if (timeline_rounds > 0) q.timeline = Timeline.init(device, timeline_rounds);
        for (&registry) |*r| if (r.cmpxchgStrong(null, q, .acq_rel, .acquire) == null) {
            _ = registered.fetchAdd(1, .acq_rel);
            break;
        };
        return q;
    }

    fn newSet(device: Id) Id {
        const d = objc.msg(Id, objc.msg(Id, objc.class("MTLResidencySetDescriptor"), "alloc", .{}), "init", .{});
        defer objc.release(d);
        objc.msg(void, d, "setInitialCapacity:", .{@as(usize, 1024)});
        var err: ?Id = null;
        return objc.msg(?Id, device, "newResidencySetWithDescriptor:error:", .{ d, &err }) orelse @panic("metal 4: no residency set");
    }

    fn lock(q: *State) void {
        std.c.os_unfair_lock_lock(&q.lock_word);
    }

    fn unlock(q: *State) void {
        std.c.os_unfair_lock_unlock(&q.lock_word);
    }

    /// The next slot, begun: its previous use has completed (it is waited for if not) and its allocator is reset.
    pub fn acquire(q: *State) *Slot {
        q.lock();
        defer q.unlock();
        const s = &q.slots[q.next];
        q.next = (q.next + 1) % ring;
        if (s.open) {
            objc.msg(void, s.cb, "endCommandBuffer", .{});
            s.open = false;
            q.unsent -= 1;
        }
        if (s.value > 0) q.reach(s.value);
        objc.msg(void, s.allocator, "reset", .{});
        objc.msg(void, s.cb, "beginCommandBufferWithAllocator:", .{s.allocator});
        s.use +%= 1;
        s.value = 0;
        s.page = 0;
        s.bump = 0;
        s.nwaits = 0;
        s.nsignals = 0;
        s.open = true;
        q.unsent += 1;
        return s;
    }

    fn commit(q: *State, s: *Slot) void {
        objc.msg(void, s.cb, "endCommandBuffer", .{});
        const options = objc.msg(Id, objc.msg(Id, objc.class("MTL4CommitOptions"), "alloc", .{}), "init", .{});
        defer objc.release(options);
        objc.msg(void, options, "addFeedbackHandler:", .{&s.block});
        q.lock();
        defer q.unlock();
        if (q.set_dirty) {
            objc.msg(void, q.set, "commit", .{});
            if (q.set_grew) objc.msg(void, q.set, "requestResidency", .{});
            q.set_dirty = false;
            q.set_grew = false;
        }
        for (s.waits[0..s.nwaits]) |w| objc.msg(void, q.queue, "waitForEvent:value:", .{ w.id, w.value });
        const list = [1]Id{s.cb};
        objc.msg(void, q.queue, "commit:count:options:", .{ @as([*]const Id, &list), @as(usize, 1), options });
        q.tick += 1;
        s.value = q.tick;
        s.commits += 1;
        s.open = false;
        q.unsent -= 1;
        objc.msg(void, q.queue, "signalEvent:value:", .{ q.event, q.tick });
        for (s.signals[0..s.nsignals]) |g| objc.msg(void, q.queue, "signalEvent:value:", .{ g.id, g.value });
    }

    /// Blocks until the queue's event reaches `v`.
    fn reach(q: *State, v: u64) void {
        if (objc.msg(u64, q.event, "signaledValue", .{}) >= v) return;
        var seconds: u32 = 0;
        while (!objc.msg(bool, q.event, "waitUntilSignaledValue:timeoutMS:", .{ v, @as(u64, 1000) })) {
            seconds += 1;
            if (seconds == 10) std.log.warn("metal 4: a command buffer has run for 10 s", .{});
        }
    }

    /// With nothing in flight or being encoded, the buffers their owners freed leave the set.
    fn drain(q: *State) void {
        if (q.dropped.items.len == 0) return;
        q.lock();
        defer q.unlock();
        if (q.unsent != 0 or objc.msg(u64, q.event, "signaledValue", .{}) < q.tick) return;
        for (q.dropped.items) |id| objc.msg(void, q.set, "removeAllocation:", .{id});
        q.dropped.clearRetainingCapacity();
        objc.msg(void, q.set, "commit", .{});
    }

    /// A buffer's GPU address, the buffer joining the residency set the first time it is seen.
    pub fn address(q: *State, buffer: Id) u64 {
        const k = @intFromPtr(buffer);
        const t = q.table.load(.acquire);
        if (t.find(k)) |i| return t.addrs[i];
        return q.admit(buffer);
    }

    fn admit(q: *State, buffer: Id) u64 {
        const k = @intFromPtr(buffer);
        q.lock();
        defer q.unlock();
        var t = q.table.load(.monotonic);
        if (t.find(k)) |i| return t.addrs[i];
        if ((t.used + 1) * 2 > t.keys.len) t = q.grow(t);
        const addr = objc.msg(u64, buffer, "gpuAddress", .{});
        objc.msg(void, q.set, "addAllocation:", .{buffer});
        q.set_dirty = true;
        q.set_grew = true;
        t.put(k, addr);
        return addr;
    }

    /// A table with room, its live keys moved over; the old one stays readable until the queue closes.
    fn grow(q: *State, t: *Table) *Table {
        var live: usize = 0;
        for (t.keys) |*k| live += @intFromBool(k.load(.monotonic) > Table.tomb);
        var n = t.keys.len;
        while ((live + 1) * 4 > n) n *= 2;
        const g = Table.init(n);
        for (t.keys, t.addrs) |*k, a| {
            const key = k.load(.monotonic);
            if (key > Table.tomb) g.put(key, a);
        }
        q.retired.append(heap, t) catch @panic("metal 4: out of memory");
        q.table.store(g, .release);
        return g;
    }

    fn drop(q: *State, buffer: Id) void {
        const k = @intFromPtr(buffer);
        q.lock();
        defer q.unlock();
        const t = q.table.load(.monotonic);
        const i = t.find(k) orelse return;
        t.keys[i].store(Table.tomb, .release);
        q.dropped.append(heap, buffer) catch {
            objc.msg(void, q.set, "removeAllocation:", .{buffer});
            q.set_dirty = true;
        };
    }

    fn fail(q: *State, err: Id) void {
        const text = std.mem.span(objc.errorText(err));
        std.log.err("metal 4: a command buffer failed: {s}", .{text});
        const n = @min(text.len, q.failure_text.len - 1);
        @memcpy(q.failure_text[0..n], text[0..n]);
        q.failure_text[n] = 0;
        _ = q.failures.fetchAdd(1, .acq_rel);
    }

    /// Everything committed has run and reported; then the queue, its slots and its set go.
    pub fn close(q: *State) void {
        _ = q.closeWithin(feedback_naps);
    }

    /// As close with at most `naps` naps for owed feedback; while one is owed the State (its handlers' storage) stays.
    pub fn closeWithin(q: *State, naps: u32) bool {
        for (&registry) |*r| if (r.cmpxchgStrong(q, null, .acq_rel, .acquire) == null) {
            _ = registered.fetchSub(1, .acq_rel);
            break;
        };
        if (q.tick > 0) q.reach(q.tick);
        var left = naps;
        var owed = false;
        for (&q.slots) |*s| {
            while (s.seen.load(.acquire) < s.commits and !owed) {
                if (left == 0) owed = true else nap(10_000);
                left -|= 1;
            }
            if (s.open) objc.msg(void, s.cb, "endCommandBuffer", .{});
            for (s.pages) |p| if (p) |pg| objc.release(pg.buf);
            for ([_]Id{ s.table, s.allocator, s.cb }) |id| objc.release(id);
        }
        objc.msg(void, q.set, "removeAllAllocations", .{});
        objc.msg(void, q.set, "commit", .{});
        objc.msg(void, q.queue, "removeResidencySet:", .{q.set});
        if (q.timeline) |t| t.deinit();
        for ([_]Id{ q.set, q.event, q.queue }) |id| objc.release(id);
        for (q.retired.items) |t| t.deinit();
        q.retired.deinit(heap);
        q.dropped.deinit(heap);
        q.table.load(.monotonic).deinit();
        if (owed) {
            std.log.warn("metal 4: a command buffer's feedback was still owed at close; the queue's memory is kept for it", .{});
            return true;
        }
        heap.destroy(q);
        return false;
    }

    /// A round of named work: under TF_METAL4_TIMELINE some rounds of each name get a timestamp at every stage.
    pub fn roundBegin(q: *State, name: []const u8) void {
        if (q.timeline) |t| t.begin(name);
    }

    /// After the round's last wait: a stamped round's stages are read and summed.
    pub fn roundEnd(q: *State) void {
        if (q.timeline) |t| t.end();
    }
};

test "the follow table: lookups, a dropped key, growth keeps every live address" {
    const t = Table.init(8);
    defer t.deinit();
    for (1..4) |i| t.put(i * 64, i * 1000);
    try std.testing.expectEqual(@as(?usize, null), t.find(5 * 64));
    try std.testing.expectEqual(@as(u64, 2000), t.addrs[t.find(2 * 64).?]);
    t.keys[t.find(2 * 64).?].store(Table.tomb, .release);
    try std.testing.expectEqual(@as(?usize, null), t.find(2 * 64));
    try std.testing.expectEqual(@as(u64, 3000), t.addrs[t.find(3 * 64).?]);
    t.put(2 * 64, 2222);
    try std.testing.expectEqual(@as(u64, 2222), t.addrs[t.find(2 * 64).?]);
    try std.testing.expectEqual(@as(usize, 3), t.used);
}
