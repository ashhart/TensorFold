//! TF_FLASHNEXT_SCALE_INDEX: the 4/32 expert projections' scales and biases as index and table buffers (scale_pairs).
const std = @import("std");
const replay = @import("replay.zig");
const variants = @import("variants.zig");
const experts4 = @import("experts4.zig");
const pairs = @import("scale_pairs.zig");
const Run = replay.Run;
const Buf = replay.Buf;

/// A projection's codes buffer: the key of its index and table.
pub const Key = struct { id: @FieldType(@FieldType(Buf, "b"), "id"), off: usize };

pub fn keyOf(codes: Buf) Key {
    return .{ .id = codes.b.id, .off = codes.off };
}

/// The projections only expert kernels read (routed gate, up, down): their stored scales and biases are released.
pub fn routed(j: usize) bool {
    return j == 0 or j == 1 or j == 4;
}

/// An expert set (weight, scales, biases a projection) and its widths; null widths are the recorded 6/32.
pub const Set = struct { ex: *[18]Buf, xw: ?variants.Experts };

const Job = struct { ex: *[18]Buf, j: usize, ix: Buf, table: []u32 = &.{}, state: enum { waiting, done, too_many, failed } = .waiting };

const max_workers = 8;

/// Jobs taken in turn: each projection's pairs indexed into its buffer, its table kept on the host for the caller.
fn work(jobs: []Job, next: *std.atomic.Value(usize)) void {
    const gpa = std.heap.page_allocator;
    const table = gpa.create([pairs.most]u32) catch return;
    defer gpa.destroy(table);
    const seen = gpa.create([pairs.slots]u32) catch return;
    defer gpa.destroy(seen);
    while (true) {
        const at = next.fetchAdd(1, .monotonic);
        if (at >= jobs.len) return;
        const job = &jobs[at];
        const s = job.ex[job.j * 3 + 1];
        const b = job.ex[job.j * 3 + 2];
        const groups = (s.b.length() - s.off) / 2;
        const sp: [*]align(1) const u16 = @ptrCast(s.b.contents() + s.off);
        const bp: [*]align(1) const u16 = @ptrCast(b.b.contents() + b.off);
        const ip: [*]align(1) u16 = @ptrCast(job.ix.b.contents());
        const n = pairs.index(sp[0..groups], bp[0..groups], ip[0..groups], table, seen) catch {
            job.state = .too_many;
            continue;
        };
        job.table = gpa.dupe(u32, table[0..n]) catch {
            job.state = .failed;
            continue;
        };
        job.state = .done;
    }
}

/// Every 4/32 gate/up and down group of `sets` indexed, routed arrays released and their slots given index and table.
pub fn build(r: *Run, sets: []const Set) !void {
    const gpa = std.heap.page_allocator;
    var jobs: std.ArrayList(Job) = .empty;
    defer jobs.deinit(gpa);
    defer for (jobs.items) |job| if (job.table.len != 0) gpa.free(job.table);
    for (sets) |set| {
        const w = set.xw orelse continue;
        var c: [4]variants.Const = undefined;
        for ([_][2]usize{ .{ 0, 4 }, .{ 4, 6 } }, [_][4]usize{ w.gu, w.down }) |span, width| {
            if (!experts4.q4(variants.four(&c, width))) continue;
            for (span[0]..span[1]) |j| {
                const s = set.ex[j * 3 + 1];
                try jobs.append(gpa, .{ .ex = set.ex, .j = j, .ix = .{ .b = try r.buffer(s.b.length() - s.off) } });
            }
        }
    }
    if (jobs.items.len == 0) {
        r.sb_index = false;
        return;
    }
    var next: std.atomic.Value(usize) = .init(0);
    var threads: [max_workers]?std.Thread = @splat(null);
    const workers = @min(max_workers, std.Thread.getCpuCount() catch 1, jobs.items.len);
    for (threads[1..workers]) |*t| t.* = std.Thread.spawn(.{}, work, .{ jobs.items, &next }) catch null;
    work(jobs.items, &next);
    for (threads) |t| if (t) |thread| thread.join();
    for (jobs.items) |job| if (job.state != .done) {
        for (jobs.items) |made| r.release(made.ix.b);
        r.sb_index = false;
        if (job.state != .too_many) return error.OutOfMemory;
        std.log.warn("TF_FLASHNEXT_SCALE_INDEX: a tensor has more than {d} scale and bias pairs; the index is off", .{pairs.most});
        return;
    };
    for (jobs.items) |job| {
        const table = try r.buffer(job.table.len * 4);
        @memcpy(table.contents()[0 .. job.table.len * 4], std.mem.sliceAsBytes(job.table));
        try r.sbix.put(r.arena, keyOf(job.ex[job.j * 3]), .{ job.ix, .{ .b = table } });
        r.sb_pairs = @max(r.sb_pairs, job.table.len);
        if (!routed(job.j)) continue;
        r.release(job.ex[job.j * 3 + 1].b);
        r.release(job.ex[job.j * 3 + 2].b);
        job.ex[job.j * 3 + 1] = job.ix;
        job.ex[job.j * 3 + 2] = .{ .b = table };
    }
}

/// `e` as the indexed kernels take it: the shared projections' index and table in their scales' and biases' slots.
pub fn bind(r: *const Run, e: *[18]Buf, gate_up: bool, down: bool) !void {
    for (0..6) |j| {
        if (routed(j) or !(if (j < 4) gate_up else down)) continue;
        const p = r.sbix.get(keyOf(e[j * 3])) orelse return error.NoScaleIndex;
        e[j * 3 + 1] = p[0];
        e[j * 3 + 2] = p[1];
    }
}
