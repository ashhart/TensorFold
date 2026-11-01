//! Off-by-default MoE top-k tap: a host ring and a versioned trace file for inspecting routing.
//! Unset `TF_ROUTE_TAP` leaves this module inert (no thread, no copies). The ring is independent of the file writer.

const std = @import("std");

/// `TF_ROUTE_TAP=/path` enables the tap; empty or unset means off.
pub const env_name = "TF_ROUTE_TAP";
/// File magic and format version (little-endian header, then records).
pub const magic = "TFR1";
pub const version: u16 = 1;
pub const flag_weights: u16 = 1;
pub const header_bytes = 16;
/// Slots in the host ring; wrap drops the newest record when the writer lags.
pub const ring_slots = 128;
/// Expert ids (and weights) one record may hold; a longer layer is truncated.
pub const max_ids = 4096;

/// Families whose global top-k is copied before localisation. Every other name is refused.
pub const tapped = [_][]const u8{
    "glm5_next",
    "glm_moe_dsa",
    "qwen3_5_moe",
    "qwen4_exp",
    "nemotron_h",
    "nemotron_h_omni",
    "minimax_m3_vl",
    "step5_vl",
    "lfm2_moe",
};

pub const Header = extern struct {
    magic: [4]u8 = magic.*,
    version: u16 = version,
    flags: u16 = flag_weights,
    top_k: u16 = 0, // 0: each record's n is rows × k (variable); not a fixed k
    reserved: u16 = 0,
};

pub const Rec = struct {
    step: u32 = 0,
    layer: u16 = 0,
    n_ids: u16 = 0,
};

pub const Slot = struct {
    index: u32 = 0,
    rec: Rec = .{},
    ids: []i32 = &.{},
    weights: []f32 = &.{},
};

/// SPSC ring: one producer reserves, publishes after the step wait; the writer pops published slots.
pub const Ring = struct {
    recs: []Rec,
    ids: []i32,
    weights: []f32,
    cap: u32,
    max_ids: u16,
    reserved: std.atomic.Value(u32) = .init(0),
    published: std.atomic.Value(u32) = .init(0),
    read: std.atomic.Value(u32) = .init(0),
    drops: std.atomic.Value(u32) = .init(0),

    pub fn init(a: std.mem.Allocator, cap: u32, ids_each: u16) !Ring {
        const recs = try a.alloc(Rec, cap);
        errdefer a.free(recs);
        const ids = try a.alloc(i32, @as(usize, cap) * ids_each);
        errdefer a.free(ids);
        const weights = try a.alloc(f32, @as(usize, cap) * ids_each);
        @memset(recs, .{});
        @memset(ids, 0);
        @memset(weights, 0);
        return .{ .recs = recs, .ids = ids, .weights = weights, .cap = cap, .max_ids = ids_each };
    }

    pub fn deinit(r: *Ring, a: std.mem.Allocator) void {
        a.free(r.weights);
        a.free(r.ids);
        a.free(r.recs);
        r.* = undefined;
    }

    pub fn slot(r: *Ring, i: u32) Slot {
        const n: usize = r.max_ids;
        const at = @as(usize, i) * n;
        return .{ .index = i, .rec = r.recs[i], .ids = r.ids[at .. at + n], .weights = r.weights[at .. at + n] };
    }

    /// Next write index, or null when `reserved - read == cap` (wrap would overwrite unread).
    pub fn reserve(r: *Ring) ?u32 {
        const rd = r.read.load(.acquire);
        const rs = r.reserved.load(.monotonic);
        if (rs - rd >= r.cap) {
            _ = r.drops.fetchAdd(1, .monotonic);
            return null;
        }
        r.reserved.store(rs + 1, .release);
        return rs % r.cap;
    }

    pub fn put(r: *Ring, i: u32, rec: Rec, ids: []const i32, wts: []const f32) void {
        r.recs[i] = rec;
        const n = @min(@as(usize, rec.n_ids), r.max_ids);
        const at = @as(usize, i) * r.max_ids;
        @memcpy(r.ids[at .. at + n], ids[0..n]);
        if (wts.len >= n) @memcpy(r.weights[at .. at + n], wts[0..n]);
    }

    /// Make every reserved slot visible to the writer (call after the step's existing wait).
    pub fn publish(r: *Ring) void {
        r.published.store(r.reserved.load(.monotonic), .release);
    }

    /// Next published slot, or null. The slot stays reserved until `consume`.
    pub fn pop(r: *Ring) ?u32 {
        const pubd = r.published.load(.acquire);
        const rd = r.read.load(.monotonic);
        if (rd == pubd) return null;
        return rd % r.cap;
    }

    pub fn consume(r: *Ring) void {
        _ = r.read.fetchAdd(1, .release);
    }
};

pub fn envPath() ?[]const u8 {
    const v = std.mem.span(std.c.getenv(env_name) orelse return null);
    return if (v.len == 0) null else v;
}

pub fn isTapped(model_type: []const u8) bool {
    for (tapped) |name| if (std.mem.eql(u8, name, model_type)) return true;
    return false;
}

/// Null when the tap is off or the family is tapped; otherwise a refusal naming the family.
pub fn refuse(a: std.mem.Allocator, model_type: []const u8, problem: *[]const u8) bool {
    if (envPath() == null) return false;
    if (isTapped(model_type)) return false;
    problem.* = std.fmt.allocPrint(a, "TF_ROUTE_TAP: {s} is not tapped; the switch is refused", .{model_type}) catch
        "TF_ROUTE_TAP: this family is not tapped; the switch is refused";
    return true;
}

/// Refuse an unsupported family, or start the rank-0 writer when the env is set.
pub fn admit(a: std.mem.Allocator, gpa: std.mem.Allocator, model_type: []const u8, rank: u32, problem: *[]const u8) bool {
    if (refuse(a, model_type, problem)) return false;
    start(gpa, rank) catch |err| {
        problem.* = std.fmt.allocPrint(a, "TF_ROUTE_TAP: cannot write the trace ({s})", .{@errorName(err)}) catch
            "TF_ROUTE_TAP: cannot write the trace";
        return false;
    };
    return true;
}

pub fn encodeHeader(out: *[header_bytes]u8, flags: u16) void {
    const h = Header{ .flags = flags };
    @memcpy(out[0..4], &h.magic);
    std.mem.writeInt(u16, out[4..6], h.version, .little);
    std.mem.writeInt(u16, out[6..8], h.flags, .little);
    std.mem.writeInt(u16, out[8..10], h.top_k, .little);
    std.mem.writeInt(u16, out[10..12], h.reserved, .little);
    @memset(out[12..16], 0);
}

pub fn encodeRec(out: []u8, rec: Rec, ids: []const i32, wts: []const f32) usize {
    std.debug.assert(out.len >= 8 + ids.len * 4 + wts.len * 4);
    std.mem.writeInt(u32, out[0..4], rec.step, .little);
    std.mem.writeInt(u16, out[4..6], rec.layer, .little);
    std.mem.writeInt(u16, out[6..8], rec.n_ids, .little);
    var at: usize = 8;
    for (ids[0..rec.n_ids]) |id| {
        std.mem.writeInt(i32, out[at..][0..4], id, .little);
        at += 4;
    }
    for (wts[0..rec.n_ids]) |w| {
        std.mem.writeInt(u32, out[at..][0..4], @bitCast(w), .little);
        at += 4;
    }
    return at;
}

const Session = struct {
    gpa: std.mem.Allocator,
    ring: Ring,
    threaded: std.Io.Threaded,
    file: std.Io.File,
    thread: ?std.Thread = null,
    stop: std.atomic.Value(bool) = .init(false),
    step: std.atomic.Value(u32) = .init(0),
    layer: std.atomic.Value(u16) = .init(0),
    step_open: std.atomic.Value(bool) = .init(false),
};

var g_session: ?*Session = null;

/// True only after rank 0 opened a tapped family with `TF_ROUTE_TAP` set.
pub fn live() bool {
    return g_session != null;
}

pub fn start(gpa: std.mem.Allocator, rank: u32) !void {
    if (envPath() == null) return;
    if (rank != 0) return;
    if (g_session != null) return;
    const path = envPath().?;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var file = if (std.fs.path.isAbsolute(path))
        try std.Io.Dir.createFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().createFile(io, path, .{});
    errdefer file.close(io);
    var hdr: [header_bytes]u8 = undefined;
    encodeHeader(&hdr, flag_weights);
    try file.writeStreamingAll(io, &hdr);
    const s = try gpa.create(Session);
    errdefer gpa.destroy(s);
    s.* = .{ .gpa = gpa, .ring = try Ring.init(gpa, ring_slots, max_ids), .threaded = threaded, .file = file };
    s.thread = try std.Thread.spawn(.{ .stack_size = 1 << 20 }, writerMain, .{s});
    g_session = s;
}

pub fn stop() void {
    const s = g_session orelse return;
    g_session = null;
    s.ring.publish();
    s.stop.store(true, .release);
    if (s.thread) |t| t.join();
    s.file.close(s.threaded.io());
    s.ring.deinit(s.gpa);
    s.gpa.destroy(s);
}

/// Host-side reserve of the next ring slot (ids/weights filled by the caller, including device copies).
pub fn reserve(n_ids: u16) ?Slot {
    @branchHint(.unlikely);
    const s = g_session orelse return null;
    if (!s.step_open.load(.monotonic)) {
        _ = s.step.fetchAdd(1, .monotonic);
        s.layer.store(0, .monotonic);
        s.step_open.store(true, .monotonic);
    }
    const i = s.ring.reserve() orelse return null;
    const layer = s.layer.fetchAdd(1, .monotonic);
    var sl = s.ring.slot(i);
    const n = @min(n_ids, s.ring.max_ids);
    sl.index = i;
    sl.rec = .{ .step = s.step.load(.monotonic) - 1, .layer = layer, .n_ids = n };
    s.ring.recs[i] = sl.rec;
    sl.ids = sl.ids[0..n];
    sl.weights = sl.weights[0..n];
    return sl;
}

/// Byte offset of slot `index`'s ids in a shared ids|weights ring buffer (ids first, then weights).
pub fn sharedOff(index: u32) usize {
    return @as(usize, index) * max_ids * 4;
}

/// Reserved slot count (absolute); Metal records this at encode so a later wait publishes only that CB.
pub fn reservedCount() u32 {
    const s = g_session orelse return 0;
    return s.ring.reserved.load(.monotonic);
}

/// After a Metal wait: copy shared-storage ids/weights into the host ring, then publish.
pub fn absorbShared(ids: []align(1) const i32, wts: []align(1) const f32) void {
    absorbSharedTo(ids, wts, reservedCount());
}

/// Copy only slots `[published, end)` from the shared ring (one landed command buffer).
pub fn absorbSharedTo(ids: []align(1) const i32, wts: []align(1) const f32, end: u32) void {
    const s = g_session orelse return;
    var i = s.ring.published.load(.monotonic);
    const last = @min(end, s.ring.reserved.load(.monotonic));
    while (i < last) : (i += 1) {
        const k = i % s.ring.cap;
        const n = s.ring.recs[k].n_ids;
        const at = @as(usize, k) * s.ring.max_ids;
        if (ids.len >= at + n) @memcpy(s.ring.ids[at .. at + n], ids[at .. at + n]);
        if (wts.len >= at + n) @memcpy(s.ring.weights[at .. at + n], wts[at .. at + n]);
    }
}

pub fn setLayer(layer: u16) void {
    if (g_session) |s| s.layer.store(layer, .monotonic);
}

/// After the step wait: ring slots become readable; the next reserve starts a new step.
pub fn publish() void {
    publishTo(reservedCount());
}

/// Publish only slots below `end` (a landed command buffer). Later reserved slots stay unpublished.
pub fn publishTo(end: u32) void {
    const s = g_session orelse return;
    const last = @min(end, s.ring.reserved.load(.monotonic));
    const cur = s.ring.published.load(.monotonic);
    if (last > cur) s.ring.published.store(last, .release);
    s.step_open.store(false, .release);
}

/// True when a record has no picks (Metal published a slot before its blit landed).
pub fn blank(ids: []const i32, wts: []const f32) bool {
    if (ids.len == 0) return false;
    for (ids) |id| if (id != 0) return false;
    for (wts) |w| if (w != 0) return false;
    return true;
}

pub fn storeHost(sl: Slot, ids: []const i32, wts: []const f32) void {
    const n = sl.rec.n_ids;
    if (ids.len >= n) @memcpy(sl.ids[0..n], ids[0..n]);
    if (wts.len >= n) @memcpy(sl.weights[0..n], wts[0..n]);
}

fn writerMain(s: *Session) void {
    var buf: [8 + max_ids * 8]u8 = undefined;
    while (true) {
        const stopping = s.stop.load(.acquire);
        while (s.ring.pop()) |i| {
            const rec = s.ring.recs[i];
            const sl = s.ring.slot(i);
            const n = rec.n_ids;
            if (blank(sl.ids[0..n], sl.weights[0..n])) {
                s.ring.consume();
                continue;
            }
            const used = encodeRec(buf[0..], rec, sl.ids[0..n], sl.weights[0..n]);
            s.ring.consume();
            s.file.writeStreamingAll(s.threaded.io(), buf[0..used]) catch {};
        }
        if (stopping) break;
        std.Io.sleep(s.threaded.io(), .fromMilliseconds(2), .awake) catch {};
    }
}

test "ring wraparound: unread slots are not overwritten; publish/pop wrap the index" {
    const a = std.testing.allocator;
    var r = try Ring.init(a, 4, 2);
    defer r.deinit(a);
    var i: u32 = 0;
    while (i < 4) : (i += 1) {
        const k = r.reserve() orelse return error.Reserve;
        r.put(k, .{ .step = 0, .layer = @intCast(i), .n_ids = 2 }, &.{ @intCast(i), @intCast(10 + i) }, &.{ 0.25, 0.75 });
    }
    try std.testing.expect(r.reserve() == null);
    try std.testing.expectEqual(@as(u32, 1), r.drops.load(.monotonic));
    r.publish();
    var seen: [4]u16 = undefined;
    var n: usize = 0;
    while (r.pop()) |k| {
        seen[n] = r.recs[k].layer;
        n += 1;
        r.consume();
    }
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqualSlices(u16, &.{ 0, 1, 2, 3 }, &seen);
    const k = r.reserve() orelse return error.AfterWrap;
    r.put(k, .{ .step = 1, .layer = 9, .n_ids = 1 }, &.{42}, &.{1.0});
    r.publish();
    const j = r.pop() orelse return error.PopWrap;
    try std.testing.expectEqual(@as(u32, 0), j);
    r.consume();
    try std.testing.expectEqual(@as(u16, 9), r.recs[j].layer);
    try std.testing.expectEqual(@as(i32, 42), r.ids[0]);
}

test "record header and one step/layer record round-trip" {
    var hdr: [header_bytes]u8 = undefined;
    encodeHeader(&hdr, flag_weights);
    try std.testing.expectEqualStrings(magic, hdr[0..4]);
    try std.testing.expectEqual(version, std.mem.readInt(u16, hdr[4..6], .little));
    try std.testing.expectEqual(flag_weights, std.mem.readInt(u16, hdr[6..8], .little));
    try std.testing.expectEqual(@as(u16, 0), std.mem.readInt(u16, hdr[8..10], .little));
    var rec_buf: [8 + 8 + 8]u8 = undefined;
    const rec = Rec{ .step = 3, .layer = 11, .n_ids = 2 };
    const n = encodeRec(&rec_buf, rec, &.{ 7, 4 }, &.{ 0.5, 0.25 });
    try std.testing.expectEqual(@as(usize, 24), n);
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, rec_buf[0..4], .little));
    try std.testing.expectEqual(@as(u16, 11), std.mem.readInt(u16, rec_buf[4..6], .little));
    try std.testing.expectEqual(@as(i32, 7), std.mem.readInt(i32, rec_buf[8..12], .little));
    try std.testing.expectEqual(@as(i32, 4), std.mem.readInt(i32, rec_buf[12..16], .little));
}

test "tapped families are named; DeepSeek V4.1 and dense families are refused only when the env is set" {
    for (tapped) |name| try std.testing.expect(isTapped(name));
    try std.testing.expect(!isTapped("deepseek_v41"));
    try std.testing.expect(!isTapped("qwen3_5"));
    try std.testing.expect(!isTapped("gemma4"));
    try std.testing.expect(!isTapped("qwen3_8_flash_next"));
    var problem: []const u8 = "";
    try std.testing.expect(!refuse(std.testing.allocator, "deepseek_v41", &problem));
    try std.testing.expect(!refuse(std.testing.allocator, "gemma4", &problem));
}

test "unset env keeps live() false and reserve a no-op" {
    try std.testing.expect(envPath() == null);
    try std.testing.expect(!live());
    try std.testing.expect(reserve(8) == null);
}

test "absorbShared takes align(1) slices from Metal contents()" {
    const f: *const fn ([]align(1) const i32, []align(1) const f32) void = &absorbShared;
    _ = f;
    absorbShared(&.{}, &.{});
}

test "pop does not release a slot until consume; blank picks are detected" {
    const a = std.testing.allocator;
    var r = try Ring.init(a, 2, 2);
    defer r.deinit(a);
    const i = r.reserve() orelse return error.Reserve;
    r.put(i, .{ .step = 0, .layer = 0, .n_ids = 2 }, &.{ 3, 4 }, &.{ 0.5, 0.5 });
    r.publish();
    try std.testing.expectEqual(@as(u32, 0), r.pop().?);
    try std.testing.expectEqual(@as(u32, 0), r.pop().?);
    r.consume();
    try std.testing.expect(r.pop() == null);
    try std.testing.expect(blank(&.{ 0, 0 }, &.{ 0, 0 }));
    try std.testing.expect(!blank(&.{ 3, 0 }, &.{ 0, 0 }));
    try std.testing.expect(!blank(&.{}, &.{}));
}

test "publishTo leaves later reserved slots unpublished" {
    const a = std.testing.allocator;
    var r = try Ring.init(a, 8, 1);
    defer r.deinit(a);
    _ = r.reserve();
    _ = r.reserve();
    _ = r.reserve();
    r.published.store(1, .release);
    try std.testing.expectEqual(@as(u32, 0), r.pop().?);
    r.consume();
    try std.testing.expect(r.pop() == null);
    r.published.store(3, .release);
    try std.testing.expectEqual(@as(u32, 1), r.pop().?);
}
