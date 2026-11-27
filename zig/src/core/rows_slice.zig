//! Sliced write-behind gather: files charge the permit one by one and mid-job cases abandon cleanly (#3370 PR 1).
const std = @import("std");
const pc = @import("prompt_cache.zig");
const store = @import("rows_store.zig");
const image = @import("rows_image.zig");
const file = @import("rows_file.zig");
const chain = @import("rows_chain.zig");
const sync = @import("rows_write_sync.zig");
const io = @import("rows_io.zig");
const Rows = store.Rows;

/// Slice budget between decode rounds (rows_host.answer).
pub const round_slice_ns: i128 = 10 * std.time.ns_per_ms;

/// Slice budget while idle or draining (rows_host.step and rows_write.drain).
pub const idle_slice_ns: i128 = 40 * std.time.ns_per_ms;

/// Engine-thread state of one sliced job whose files are still being gathered into worker payloads.
pub const Active = struct {
    job: *io.Job,
    entry: *pc.Entry,
    parts: []image.Part,
    wants: []sync.Want,
    own: []image.Range,
    next: usize = 0,
};

/// Sliced-gather and eviction-allowance state kept on Rows without growing rows_store.zig.
pub const State = struct {
    active: ?*Active = null,
    eviction_bytes: u64 = 0,
    waiting_used: u64 = 0,
    waiting_slot: bool = false,
    abandoned: u64 = 0,
    payload_allocator: std.mem.Allocator = io.allocator,
};

/// Eviction whole-image staging ceiling (#2388's allowance applies only to evictions, never normal saves).
pub fn evictionAllowance(r: *const Rows) u64 {
    return @max(r.staging_bytes, r.sliced.eviction_bytes);
}

/// Worst-case host RAM the write path can hold: sliced cap alone when spill is off, plus eviction allowance when on.
pub fn hostBytes(r: *const Rows) u64 {
    return r.staging_bytes + if (r.spill == .off) 0 else evictionAllowance(r);
}

/// Startup memory-plan clause naming the write path's host RAM ceiling.
pub fn formatHostPlan(r: *const Rows, buf: []u8) []const u8 {
    if (r.spill == .off) {
        return std.fmt.bufPrint(buf, "staging up to {d} MiB host RAM", .{r.staging_bytes >> 20}) catch "";
    }
    return std.fmt.bufPrint(buf, "staging up to {d} MiB host RAM ({d} MiB sliced + {d} MiB eviction)", .{
        hostBytes(r) >> 20,
        r.staging_bytes >> 20,
        evictionAllowance(r) >> 20,
    }) catch "";
}

/// Log the write path's host RAM ceiling once at startup (silent in tests).
pub fn noteHostPlan(r: *const Rows) void {
    if (@import("builtin").is_test) return;
    var buf: [128]u8 = undefined;
    std.log.info("imprint: {s}", .{formatHostPlan(r, &buf)});
}

pub fn ensureWorker(r: *Rows) !*io.Worker {
    if (r.worker == null) {
        r.worker = try io.Worker.open(r.dir, r.staging_bytes, r.sliced.eviction_bytes, r.file_io, r.sliced.payload_allocator);
    }
    return r.worker.?;
}

/// Submit a normal write-behind job with its trailers ready and its file payloads deferred to `gatherSlice`.
pub fn start(r: *Rows, prepared: *const sync.Job) !void {
    const w = try ensureWorker(r);
    if (w.names(prepared.name)) return;
    const e = prepared.entry;
    const metadata = 4 * (3 + e.tokens.len + 2 * e.decode_spans.len + prepared.starts.len);
    var promised: io.Promised = .empty;
    defer promised.deinit(r.gpa);
    var count: usize = 0;
    try w.promised(r.gpa, &promised);
    var bytes = newBytes(r, &promised, prepared, metadata, &count);
    while (true) {
        if (!w.hasSlot()) {
            r.sliced.waiting_slot = true;
            return error.StagingBusy;
        }
        const first = firstPayloadBytes(r, &promised, prepared);
        if (!w.canCharge(first)) {
            r.sliced.waiting_slot = false;
            r.sliced.waiting_used = w.slicedUsed();
            return error.StagingBusy;
        }
        const reserved = w.reserved();
        if (!r.budget(bytes + reserved)) return error.DailyBudgetSpent;
        if (!r.room(bytes + reserved)) return error.OverCap;
        try w.promised(r.gpa, &promised);
        var again_count: usize = 0;
        const again = newBytes(r, &promised, prepared, metadata, &again_count);
        if (again == bytes) break;
        bytes = again;
        count = again_count;
    }
    if (r.admission.reserve(r.dir, bytes) != .ready) return error.UnderFreeDiskFloor;
    errdefer r.admission.finish(r.dir, bytes, true);
    const j = try io.allocator.create(io.Job);
    errdefer io.allocator.destroy(j);
    var arena = std.heap.ArenaAllocator.init(io.allocator);
    errdefer arena.deinit();
    const a = arena.allocator();
    const files = try a.alloc(io.File, count);
    const need_wants = try r.gpa.alloc(sync.Want, count - 1);
    errdefer r.gpa.free(need_wants);
    var built: usize = 0;
    for (prepared.wants) |want| if (lacks(r, &promised, want)) {
        files[built] = .{
            .trailer = .{ .kind = @backingInt(file.Kind.block), .payload = 0, .payload_hash = 0, .name = want.name, .block = r.keying.block },
        };
        need_wants[built] = want;
        built += 1;
    };
    const own = &files[built];
    own.* = .{
        .trailer = .{ .kind = @backingInt(file.Kind.state), .payload = 0, .payload_hash = 0, .name = prepared.name, .at = e.at, .block = r.keying.block, .flags = @intFromBool(e.shared) },
    };
    const tokens = try a.dupe(u32, e.tokens);
    const spans = try a.dupe(@import("cache_modes.zig").Span, e.decode_spans);
    const starts = try a.dupe(u32, prepared.starts);
    const counts = try a.create([3]u32);
    counts.* = .{ @intCast(tokens.len), @intCast(spans.len), @intCast(starts.len) };
    own.meta = try a.dupe([]const u8, &.{ std.mem.asBytes(counts), std.mem.sliceAsBytes(tokens), std.mem.sliceAsBytes(spans), std.mem.sliceAsBytes(starts) });
    const wants = try a.dupe(sync.Want, prepared.wants);
    const act = try r.gpa.create(Active);
    errdefer r.gpa.destroy(act);
    const parts_copy = try r.gpa.dupe(image.Part, prepared.parts);
    errdefer r.gpa.free(parts_copy);
    const own_copy = try r.gpa.dupe(image.Range, prepared.own);
    errdefer r.gpa.free(own_copy);
    j.* = .{
        .arena = arena,
        .payload_alloc = w.payload_alloc,
        .id = r.next_job,
        .generation = w.generation.load(.acquire),
        .name = prepared.name,
        .at = e.at,
        .shared = e.shared,
        .sliced = true,
        .tokens = tokens,
        .spans = spans,
        .starts = starts,
        .wants = wants,
        .files = files,
        .bytes = bytes,
        .staged = 0,
        .verify = r.verify,
        .began = prepared.began,
    };
    act.* = .{ .job = j, .entry = e, .parts = parts_copy, .wants = need_wants, .own = own_copy };
    w.submitSliced(j);
    r.sliced.active = act;
}

/// Gather at least one file of the active sliced job, continuing until `budget_ns` elapses or the permit fills.
pub fn gatherSlice(r: *Rows, budget_ns: i128) bool {
    if (r.sliced.active == null) return false;
    const began = io.nowNs();
    var progressed = false;
    while (r.sliced.active) |act| {
        switch (gatherOne(r, act)) {
            .gathered => {
                progressed = true;
                if (r.sliced.active == null or io.nowNs() - began >= budget_ns) break;
            },
            .full => break,
            .ended => {
                progressed = true;
                break;
            },
        }
    }
    return progressed;
}

const StepOutcome = enum { gathered, full, ended };

fn gatherOne(r: *Rows, act: *Active) StepOutcome {
    const w = r.worker orelse {
        clearActive(r);
        return .ended;
    };
    if (!w.valid(act.job) or act.job.done.load(.acquire)) {
        clearActive(r);
        return .ended;
    }
    if (r.img.whole) |whole| if (!whole(r.family.ptr, act.entry.saved)) {
        abandon(r, error.NotWhole);
        return .ended;
    };
    var block_ranges: std.ArrayList(image.Range) = .empty;
    defer block_ranges.deinit(r.gpa);
    const ranges: []const image.Range = if (act.next < act.wants.len) blk: {
        const want = act.wants[act.next];
        image.blockRanges(act.parts, r.keying.block, want.window, want.k, &block_ranges, r.gpa) catch {
            abandon(r, error.OutOfMemory);
            return .ended;
        };
        break :blk block_ranges.items;
    } else act.own;
    const payload_len = image.total(ranges);
    if (!w.chargeFile(payload_len)) return .full;
    const payload: []u8 = if (payload_len > 0) w.payload_alloc.alloc(u8, @intCast(payload_len)) catch {
        w.releaseCharge(payload_len);
        abandon(r, error.OutOfMemory);
        return .ended;
    } else &.{};
    if (payload_len > 0) {
        r.img.gather(r.family.ptr, act.entry.saved, ranges, payload) catch |err| {
            w.payload_alloc.free(payload);
            w.releaseCharge(payload_len);
            abandon(r, err);
            return .ended;
        };
    }
    const f = &act.job.files[act.next];
    f.payload = payload;
    f.charge = payload_len;
    f.ready.store(true, .release);
    act.next += 1;
    if (act.next == act.job.files.len) clearActive(r);
    return .gathered;
}

/// Abandon the active sliced job mid-gather (PR 1): detach entry immediately and tell the worker to abort.
pub fn abandon(r: *Rows, err: anyerror) void {
    const act = r.sliced.active orelse return;
    act.job.abandon_err = err;
    act.job.abandoned.store(true, .release);
    r.sliced.abandoned += 1;
    clearActive(r);
}

pub fn canStep(r: *const Rows) bool {
    const act = r.sliced.active orelse return false;
    const w = r.worker orelse return false;
    if (!w.valid(act.job) or act.job.done.load(.acquire)) return true;
    return w.canCharge(nextFileBytes(act));
}

pub fn onEntryDropped(r: *Rows, e: *pc.Entry) void {
    if (r.sliced.active) |act| if (act.entry == e) abandon(r, error.EntryDropped);
}

pub fn onReaped(r: *Rows, j: *io.Job) void {
    if (r.sliced.active) |act| if (act.job == j) clearActive(r);
}

pub fn clearActive(r: *Rows) void {
    const act = r.sliced.active orelse return;
    r.sliced.active = null;
    r.gpa.free(act.parts);
    r.gpa.free(act.wants);
    r.gpa.free(act.own);
    r.gpa.destroy(act);
}

fn nextFileBytes(act: *const Active) u64 {
    if (act.next < act.wants.len) return act.wants[act.next].bytes - @sizeOf(file.Trailer);
    return image.total(act.own);
}

fn firstPayloadBytes(r: *const Rows, promised: *const io.Promised, j: *const sync.Job) u64 {
    for (j.wants) |want| if (lacks(r, promised, want)) return want.bytes - @sizeOf(file.Trailer);
    return image.total(j.own);
}

pub fn lacks(r: *const Rows, promised: *const io.Promised, want: sync.Want) bool {
    const n = chain.stem(want.name);
    return !r.blocks.contains(n) and !promised.contains(n);
}

pub fn newBytes(r: *const Rows, promised: *const io.Promised, j: *const sync.Job, metadata: usize, count: *usize) u64 {
    var bytes = image.total(j.own) + metadata + @sizeOf(file.Trailer);
    count.* = 1;
    for (j.wants) |want| if (lacks(r, promised, want)) {
        bytes += want.bytes;
        count.* += 1;
    };
    return bytes;
}
