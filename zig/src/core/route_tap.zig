//! Optional MoE route trace: inert when TF_ROUTE_TAP is unset, with storage independent of its writer.

const std = @import("std");

const format = @import("route_tap_format.zig");
pub const env_name = format.env_name;
pub const magic = format.magic;
pub const version = format.version;
pub const flag_weights = format.flag_weights;
pub const flag_rows = format.flag_rows;
pub const flag_truncated = format.flag_truncated;
pub const record_bytes = format.record_bytes;
pub const header_bytes = format.header_bytes;
pub const ring_slots = format.ring_slots;
pub const max_ids = format.max_ids;
pub const Header = format.Header;
pub const Rec = format.Rec;
pub const Slot = format.Slot;
pub const Ring = format.Ring;
pub const encodeHeader = format.encodeHeader;
pub const encodeRec = format.encodeRec;

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

pub fn envPath() ?[]const u8 {
    const v = @import("sys").environment.Process.get(.{}, env_name) orelse return null;
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

/// Family and backend pairs that never copy (empty traces if admitted). Named, not inferred.
pub fn neverCopies(backend: []const u8, model_type: []const u8) bool {
    if (std.mem.eql(u8, backend, "xpu") and std.mem.eql(u8, model_type, "nemotron_h")) return true;
    if (std.mem.eql(u8, backend, "xpu") and std.mem.eql(u8, model_type, "qwen4_exp")) return true;
    if (std.mem.eql(u8, backend, "metal") and std.mem.eql(u8, model_type, "qwen4_exp")) return true;
    if (std.mem.eql(u8, backend, "metal") and std.mem.eql(u8, model_type, "qwen3_5_moe")) return true;
    return false;
}

/// Refuse a family/backend pair that never copies, when the env is set.
pub fn refusePair(a: std.mem.Allocator, model_type: []const u8, backend: []const u8, problem: *[]const u8) bool {
    if (envPath() == null) return false;
    if (!neverCopies(backend, model_type)) return false;
    problem.* = std.fmt.allocPrint(a, "TF_ROUTE_TAP: {s} on {s} never copies; the switch is refused", .{ model_type, backend }) catch
        "TF_ROUTE_TAP: this family never copies on this backend; the switch is refused";
    return true;
}

/// Refuse an unsupported family or a no-copy pair, or start the rank-0 writer when the env is set.
pub fn admit(a: std.mem.Allocator, gpa: std.mem.Allocator, model_type: []const u8, rank: u32, backend: []const u8, problem: *[]const u8) bool {
    if (refuse(a, model_type, problem)) return false;
    if (refusePair(a, model_type, backend, problem)) return false;
    start(gpa, rank) catch |err| {
        problem.* = std.fmt.allocPrint(a, "TF_ROUTE_TAP: cannot write the trace ({s})", .{@errorName(err)}) catch
            "TF_ROUTE_TAP: cannot write the trace";
        return false;
    };
    return true;
}

/// Frees the session on scope exit unless `dismiss` ran (the engine opened and `close` will stop).
pub const Guard = struct {
    keep: bool = false,

    pub fn dismiss(g: *Guard) void {
        g.keep = true;
    }

    pub fn deinit(g: *Guard) void {
        if (!g.keep) stop();
    }
};

/// Keep the session if `opened` is a successful engine; otherwise `Guard.deinit` calls `stop`.
pub fn held(g: *Guard, opened: anytype) @TypeOf(opened) {
    const T = @TypeOf(opened);
    if (comptime @typeInfo(T) == .optional) {
        if (opened != null) g.dismiss();
    } else {
        g.dismiss();
    }
    return opened;
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
var g_release: ?*const fn () void = null;

/// True only after rank 0 opened a tapped family with `TF_ROUTE_TAP` set.
pub fn live() bool {
    return g_session != null;
}

/// CUDA/HIP pin release: wait the step event, then free `cuMemHostAlloc` / `hipHostMalloc` buffers.
pub fn onStop(f: *const fn () void) void {
    g_release = f;
}

/// Replace the GPA ids/weights with a backend's pinned host ring. Recs stay on the allocator.
pub fn adoptHost(ids: []i32, weights: []f32) void {
    const s = g_session orelse return;
    if (s.ring.owned) {
        s.gpa.free(s.ring.ids);
        s.gpa.free(s.ring.weights);
        s.ring.owned = false;
    }
    s.ring.ids = ids;
    s.ring.weights = weights;
}

pub fn start(gpa: std.mem.Allocator, rank: u32) !void {
    if (envPath() == null) return;
    if (rank != 0) return;
    if (g_session != null) return;
    const path = (try @import("sys").environment.path(env_name, envPath())) orelse return;
    var threaded: std.Io.Threaded = .init_single_threaded;
    const io = threaded.io();
    var file = if (std.fs.path.isAbsolute(path))
        try std.Io.Dir.createFileAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().createFile(io, path, .{});
    errdefer file.close(io);
    var hdr: [header_bytes]u8 = undefined;
    encodeHeader(&hdr, flag_weights | flag_rows);
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
    if (g_release) |f| f();
    g_release = null;
    s.ring.publish();
    s.stop.store(true, .release);
    if (s.thread) |t| t.join();
    s.file.close(s.threaded.io());
    const n = s.ring.drops.load(.monotonic);
    if (n > 0) std.log.info("TF_ROUTE_TAP: dropped {d} records", .{n});
    s.ring.deinit(s.gpa);
    s.gpa.destroy(s);
}

/// Host-side reserve of the next ring slot (ids/weights filled by the caller, including device copies).
pub fn reserve(rows: usize, slots: usize) ?Slot {
    @branchHint(.unlikely);
    const s = g_session orelse return null;
    if (!s.step_open.load(.monotonic)) {
        _ = s.step.fetchAdd(1, .monotonic);
        s.layer.store(0, .monotonic);
        s.step_open.store(true, .monotonic);
    }
    const layer = s.layer.fetchAdd(1, .monotonic);
    if (rows == 0 or rows > 65535 or slots == 0 or slots > 255 or slots > s.ring.max_ids) {
        _ = s.ring.drops.fetchAdd(1, .monotonic);
        return null;
    }
    const i = s.ring.reserve() orelse return null;
    var sl = s.ring.slot(i);
    const n: u16 = @intCast(@min(rows, s.ring.max_ids / slots) * slots);
    sl.index = i;
    sl.rec = .{ .step = s.step.load(.monotonic) - 1, .layer = layer, .n_ids = n, .rows = @intCast(rows), .slots = @intCast(slots), .flags = if (@as(usize, n) < rows * slots) flag_truncated else 0 };
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

/// Slots already marked readable. CUDA/HIP land only up to the event behind this step's copies.
pub fn publishedCount() u32 {
    const s = g_session orelse return 0;
    return s.ring.published.load(.monotonic);
}

/// A failed device copy: clear the slot so a later lap cannot publish the previous ids under this header.
pub fn dropSlot(sl: Slot) void {
    const s = g_session orelse return;
    @memset(sl.ids, 0);
    @memset(sl.weights, 0);
    _ = s.ring.drops.fetchAdd(1, .monotonic);
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
    if (last <= cur) return;
    s.ring.published.store(last, .release);
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
    var buf: [record_bytes + max_ids * 8]u8 = undefined;
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
    encodeHeader(&hdr, flag_weights | flag_rows);
    try std.testing.expectEqualStrings(magic, hdr[0..4]);
    try std.testing.expectEqual(version, std.mem.readInt(u16, hdr[4..6], .little));
    try std.testing.expectEqual(flag_weights | flag_rows, std.mem.readInt(u16, hdr[6..8], .little));
    try std.testing.expectEqual(@as(u16, 0), std.mem.readInt(u16, hdr[8..10], .little));
    var rec_buf: [record_bytes + 8 + 8]u8 = undefined;
    const rec = Rec{ .step = 3, .layer = 11, .n_ids = 2, .rows = 1, .slots = 2 };
    const n = encodeRec(&rec_buf, rec, &.{ 7, 4 }, &.{ 0.5, 0.25 });
    try std.testing.expectEqual(@as(usize, 28), n);
    try std.testing.expectEqual(@as(u32, 3), std.mem.readInt(u32, rec_buf[0..4], .little));
    try std.testing.expectEqual(@as(u16, 11), std.mem.readInt(u16, rec_buf[4..6], .little));
    try std.testing.expectEqual(@as(u16, 1), std.mem.readInt(u16, rec_buf[8..10], .little));
    try std.testing.expectEqualSlices(u8, &.{ 2, 0 }, rec_buf[10..12]);
    try std.testing.expectEqual(@as(i32, 7), std.mem.readInt(i32, rec_buf[12..16], .little));
    try std.testing.expectEqual(@as(i32, 4), std.mem.readInt(i32, rec_buf[16..20], .little));
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

test "named family/backend pairs that never copy are refused by name; others copy" {
    try std.testing.expect(neverCopies("xpu", "nemotron_h"));
    try std.testing.expect(neverCopies("xpu", "qwen4_exp"));
    try std.testing.expect(neverCopies("metal", "qwen4_exp"));
    try std.testing.expect(neverCopies("metal", "qwen3_5_moe"));
    try std.testing.expect(!neverCopies("cuda", "nemotron_h"));
    try std.testing.expect(!neverCopies("hip", "nemotron_h"));
    try std.testing.expect(!neverCopies("metal", "nemotron_h"));
    try std.testing.expect(!neverCopies("cuda", "qwen4_exp"));
    try std.testing.expect(!neverCopies("hip", "qwen4_exp"));
    try std.testing.expect(!neverCopies("cuda", "qwen3_5_moe"));
    try std.testing.expect(!neverCopies("metal", "glm5_next"));
    try std.testing.expect(!neverCopies("xpu", "qwen3_5"));
    var problem: []const u8 = "";
    try std.testing.expect(!refusePair(std.testing.allocator, "nemotron_h", "xpu", &problem));
    try std.testing.expect(!refusePair(std.testing.allocator, "qwen4_exp", "metal", &problem));
}

test "unset env keeps live() false and reserve a no-op" {
    try std.testing.expect(envPath() == null);
    try std.testing.expect(!live());
    try std.testing.expect(reserve(1, 8) == null);
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

test "a failed copy clears the slot so a later lap cannot keep the previous ids" {
    const a = std.testing.allocator;
    var r = try Ring.init(a, 2, 2);
    defer r.deinit(a);
    const i = r.reserve() orelse return error.Reserve;
    r.put(i, .{ .step = 0, .layer = 0, .n_ids = 2 }, &.{ 9, 8 }, &.{ 0.5, 0.5 });
    var sl = r.slot(i);
    sl.ids = sl.ids[0..2];
    sl.weights = sl.weights[0..2];
    @memset(sl.ids, 0);
    @memset(sl.weights, 0);
    _ = r.drops.fetchAdd(1, .monotonic);
    try std.testing.expect(blank(sl.ids, sl.weights));
    try std.testing.expectEqual(@as(u32, 1), r.drops.load(.monotonic));
}

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

test "stop frees the session and ring under the testing allocator" {
    const a = std.testing.allocator;
    try std.testing.expect(!live());
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, ".zig-cache/tmp/{s}/route-tap.tfr1", .{tmp.sub_path}, 0);

    const prev = if (std.c.getenv("TF_ROUTE_TAP")) |p| try a.dupeSentinel(u8, std.mem.span(p), 0) else null;
    defer if (prev) |p| a.free(p);
    try std.testing.expectEqual(@as(c_int, 0), setenv(env_name, path, 1));
    defer {
        if (prev) |p| _ = setenv(env_name, p, 1) else _ = unsetenv(env_name);
    }

    try start(a, 0);
    defer stop();
    try std.testing.expect(live());
    const sl = reserve(1, 2) orelse return error.Reserve;
    storeHost(sl, &.{ 3, 4 }, &.{ 0.5, 0.5 });
    stop();
    try std.testing.expect(!live());
    stop();
}

test "a failed open after admit frees the ring under the testing allocator" {
    const a = std.testing.allocator;
    try std.testing.expect(!live());
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var path_buf: [128]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&path_buf, ".zig-cache/tmp/{s}/route-tap-fail.tfr1", .{tmp.sub_path}, 0);

    const prev = if (std.c.getenv("TF_ROUTE_TAP")) |p| try a.dupeSentinel(u8, std.mem.span(p), 0) else null;
    defer if (prev) |p| a.free(p);
    try std.testing.expectEqual(@as(c_int, 0), setenv(env_name, path, 1));
    defer {
        if (prev) |p| _ = setenv(env_name, p, 1) else _ = unsetenv(env_name);
        stop();
    }

    var problem: []const u8 = "";
    try std.testing.expect(!admit(a, a, "qwen4_exp", 0, "xpu", &problem));
    try std.testing.expect(!live());
    a.free(problem);
    problem = "";
    try std.testing.expect(admit(a, a, "nemotron_h", 0, "cuda", &problem));
    try std.testing.expect(live());
    {
        var g: Guard = .{};
        defer g.deinit();
        try std.testing.expect(live());
    }
    try std.testing.expect(!live());

    try std.testing.expect(admit(a, a, "nemotron_h", 0, "cuda", &problem));
    try std.testing.expect(live());
    {
        var g: Guard = .{};
        defer g.deinit();
        const opened: ?u8 = 1;
        _ = held(&g, opened);
        try std.testing.expect(g.keep);
    }
    try std.testing.expect(live());
    stop();
    try std.testing.expect(!live());
}

test "route tap: a dropped call still advances the layer ordinal" {
    const a = std.testing.allocator;
    var s: Session = .{ .gpa = a, .ring = try Ring.init(a, 2, 1), .threaded = .init_single_threaded, .file = undefined };
    defer s.ring.deinit(a);
    try std.testing.expect(g_session == null);
    g_session = &s;
    defer g_session = null;
    _ = reserve(1, 1).?;
    _ = reserve(1, 1).?;
    try std.testing.expect(reserve(1, 1) == null);
    s.ring.publish();
    s.ring.consume();
    const later = reserve(1, 1).?;
    try std.testing.expectEqual(@as(u16, 3), later.rec.layer);
}

test "route tap: long records keep whole routed rows and flag the missing tail" {
    const a = std.testing.allocator;
    var s: Session = .{ .gpa = a, .ring = try Ring.init(a, 2, 16), .threaded = .init_single_threaded, .file = undefined };
    defer s.ring.deinit(a);
    try std.testing.expect(g_session == null);
    g_session = &s;
    defer g_session = null;
    const sl = reserve(5, 3).?;
    try std.testing.expectEqual(@as(u16, 15), sl.rec.n_ids);
    try std.testing.expectEqual(@as(u8, 0), sl.rec.flags);
    const long = reserve(10, 3).?;
    try std.testing.expectEqual(@as(u16, 15), long.rec.n_ids);
    try std.testing.expectEqual(@as(u16, 10), long.rec.rows);
    try std.testing.expectEqual(@as(u8, 3), long.rec.slots);
    try std.testing.expectEqual(flag_truncated, long.rec.flags);
}

test "route tap: one completed buffer never absorbs or publishes its future sibling" {
    const a = std.testing.allocator;
    var s: Session = .{ .gpa = a, .ring = try Ring.init(a, 2, 1), .threaded = .init_single_threaded, .file = undefined };
    defer s.ring.deinit(a);
    try std.testing.expect(g_session == null);
    g_session = &s;
    defer g_session = null;
    _ = reserve(1, 1).?;
    const first = reservedCount();
    _ = reserve(1, 1).?;
    const second = reservedCount();
    var ids = [_]i32{ 3, 0 };
    var weights = [_]f32{ 1, 0 };
    absorbSharedTo(&ids, &weights, first);
    publishTo(first);
    try std.testing.expectEqual(first, publishedCount());
    try std.testing.expectEqual(@as(i32, 0), s.ring.ids[1]);
    ids[1] = 7;
    weights[1] = 1;
    absorbSharedTo(&ids, &weights, second);
    publishTo(second);
    try std.testing.expectEqual(@as(i32, 7), s.ring.ids[1]);
    publishTo(first);
    try std.testing.expectEqual(second, publishedCount());
}
