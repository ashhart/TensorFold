//! Real backend observer/cancellation/yield and saved-prefix qualification.
const std = @import("std");
const tf = @import("tensorfold");
const q = tf.lfm3;
const lanes = tf.lanes;
const Sha = std.crypto.hash.sha2.Sha256;
const Observed = struct { at: u32, logits: [32]u8, state: [32]u8 };
const Context = struct {
    metal: *q.backend.Metal,
    stream: *lanes.Stream,
    events: [8]Observed = undefined,
    count: usize = 0,
    saved: ?*q.snapshot.Snap = null,
    failed: bool = false,
    fn stop(ptr: *anyopaque) bool {
        const s: *Context = @ptrCast(@alignCast(ptr));
        return s.stream.fill_at.load(.monotonic) >= 257;
    }
    fn mark(ptr: *anyopaque, stream: *lanes.Stream, at: u32) void {
        const s: *Context = @ptrCast(@alignCast(ptr));
        const cache = s.metal.caches.get(stream) orelse {
            s.failed = true;
            return;
        };
        const snap = q.snapshot.save(s.metal.gpa, &s.metal.model.config.layer_types, cache, at) catch {
            s.failed = true;
            return;
        };
        if (s.count >= s.events.len) {
            q.snapshot.drop(s.metal.gpa, snap);
            s.failed = true;
            return;
        }
        var item = Observed{ .at = at, .logits = undefined, .state = undefined };
        Sha.hash(cache.logits.contents()[0 .. q.config.vocab * 2], &item.logits, .{});
        Sha.hash(snap.bytes, &item.state, .{});
        s.events[s.count] = item;
        s.count += 1;
        if (s.saved) |old| q.snapshot.drop(s.metal.gpa, old);
        s.saved = snap;
    }
};
const Result = struct { events: [8]Observed = undefined, count: usize, final_state: [32]u8, final_logits: [32]u8, first: u64 };
fn once(gpa: std.mem.Allocator, m: *q.Model, ids: []const u32, enabled: bool, mode: usize) !Result {
    const metal = try gpa.create(q.backend.Metal);
    metal.* = .{ .gpa = gpa, .model = m, .options = .{ .capacity = ids.len + 32, .chunk = 128, .streams = 1, .defer_prompt_head = enabled }, .scratch = try q.state.Scratch.init(gpa, m.device, &m.config.layer_types, 128) };
    defer metal.deinit();
    var stream = try lanes.Stream.init(gpa, .{ .id = "deferred-callback", .prompt = ids, .max_new = 16, .eos = &.{} });
    defer stream.deinit(gpa);
    var context = Context{ .metal = metal, .stream = &stream };
    defer if (context.saved) |s| q.snapshot.drop(gpa, s);
    stream.reuse = .{ .marks = &.{ 129, 257 }, .hook = .{ .ptr = &context, .at = Context.mark } };
    if (mode == 1) stream.cancel_check = .{ .ptr = &context, .check = Context.stop };
    if (mode == 2) stream.yield_check = .{ .ptr = &context, .check = Context.stop };
    const be = metal.backend();
    var interrupted = false;
    be.vtable.prefill(be.ptr, &stream) catch |err| {
        if (mode == 1 and err == error.Cancelled or mode == 2 and err == error.Yielded) interrupted = true else return err;
    };
    if (context.failed or context.count < 2 or (mode != 0 and !interrupted)) return error.ObserverFailure;
    if (mode != 0 and metal.caches.contains(&stream)) return error.InterruptedCacheNotReleased;
    var resumed = try lanes.Stream.init(gpa, .{ .id = "deferred-resume", .prompt = ids, .max_new = 16, .eos = &.{} });
    defer resumed.deinit(gpa);
    resumed.reuse = .{ .saved = @ptrCast(context.saved.?), .at = context.saved.?.at };
    try be.vtable.prefill(be.ptr, &resumed);
    if (resumed.cached != 257) return error.ResumePrefixMismatch;
    const cache = metal.caches.get(&resumed) orelse return error.MissingCache;
    const snap = try q.snapshot.save(gpa, &m.config.layer_types, cache, @intCast(ids.len));
    defer q.snapshot.drop(gpa, snap);
    var out = Result{ .count = context.count, .final_state = undefined, .final_logits = undefined, .first = try be.vtable.first(be.ptr, &resumed, ids.len) };
    @memcpy(out.events[0..context.count], context.events[0..context.count]);
    Sha.hash(snap.bytes, &out.final_state, .{});
    Sha.hash(cache.logits.contents()[0 .. q.config.vocab * 2], &out.final_logits, .{});
    return out;
}
pub fn check(gpa: std.mem.Allocator, m: *q.Model, ids: []const u32) !void {
    for (0..3) |mode| {
        const a = try once(gpa, m, ids, false, mode);
        const b = try once(gpa, m, ids, true, mode);
        if (a.count != b.count or a.first != b.first or !std.mem.eql(u8, &a.final_state, &b.final_state) or !std.mem.eql(u8, &a.final_logits, &b.final_logits)) return error.CallbackFinalMismatch;
        for (a.events[0..a.count], b.events[0..b.count]) |x, y| if (x.at != y.at or !std.mem.eql(u8, &x.logits, &y.logits) or !std.mem.eql(u8, &x.state, &y.state)) return error.CallbackBoundaryMismatch;
        std.debug.print("DEFERRED_CALLBACK_PASS mode={d} events={d} resume257 first={d}\n", .{ mode, a.count, a.first });
    }
}
