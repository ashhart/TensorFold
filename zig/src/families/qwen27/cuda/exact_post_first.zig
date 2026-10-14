//! A bounded host-only reservation moves optional snapshot work past the committed-token handoff.
const std = @import("std");
const lanes = @import("lanes");
const st = @import("state.zig");
const dr = @import("draft.zig");

pub fn enabled(raw: ?[]const u8, cache_on: bool) !bool {
    const value = raw orelse return false;
    if (std.mem.eql(u8, value, "0")) return false;
    if (!std.mem.eql(u8, value, "1")) return error.InvalidPostFirstCaptureFlag;
    if (!cache_on) return error.PostFirstCaptureNeedsExactCache;
    return true;
}

pub fn configured(cache_on: bool) !bool {
    const raw = std.c.getenv("TF_PR608_EXACT_CACHE_POST_FIRST");
    return enabled(if (raw) |value| std.mem.span(value) else null, cache_on);
}

pub const Reservation = struct {
    stream: *lanes.Stream,
    seq: *st.Seq,
    ctx: ?*dr.Context,
    tokens: usize,
    bytes: usize,
    cur: u1,
};

pub const Control = struct {
    enabled: bool = false,
    reserved: ?Reservation = null,
    reservations: u64 = 0,
    saves: u64 = 0,
    skips: u64 = 0,
    failures: u64 = 0,

    pub fn take(control: *Control, s: *lanes.Stream) ?Reservation {
        const value = control.reserved orelse return null;
        if (value.stream != s) return null;
        control.reserved = null;
        return value;
    }

    pub fn forget(control: *Control, s: *lanes.Stream) void {
        if (control.take(s) != null) control.skips += 1;
    }
};

/// No device work or active-entry eviction occurs while deciding the one pending owner.
pub fn reserve(cache: anytype, s: *lanes.Stream, seq: *st.Seq, ctx: ?*dr.Context, bytes: usize) void {
    if (seq.kv8 or seq.media != null or seq.rope_delta != 0 or !cache.eligible(s) or
        s.drafts != (ctx != null) or (cache.active != null and cache.active.?.matches(s))) return;
    if (cache.post_first.reserved != null) {
        cache.stats.refusals += 1;
        return;
    }
    const policy = @import("exact_cache_policy.zig");
    switch (policy.admission(cache.stats.active_bytes, cache.stats.pending_bytes, bytes)) {
        .refuse, .busy => {
            cache.stats.refusals += 1;
            return;
        },
        .evict, .retain => {},
    }
    cache.post_first.reserved = .{ .stream = s, .seq = seq, .ctx = ctx, .tokens = s.prompt().len, .bytes = bytes, .cur = if (ctx) |dc| dc.cur else 0 };
    cache.post_first.reservations += 1;
}

/// This executes on the engine thread before initial proposals can consume shared scratch.
pub fn after(p: *anyopaque, s: *lanes.Stream) !void {
    const Cuda = @import("lanes.zig").Cuda;
    const x: *Cuda = @ptrCast(@alignCast(p));
    if (x.exact_cache.post_first.reserved == null or x.exact_cache.post_first.reserved.?.stream != s) return;
    finish(x, s) catch |err| {
        if (err != error.Cancelled) {
            @import("exact_seq_reuse.zig").fail(x);
            x.exact_cache.failed = true;
        }
        x.exact_cache.abandon(s, x.e.ops()) catch |purge_error| {
            std.log.err("fatal unresolved post-first capture fence before device frees: {s}", .{@errorName(purge_error)});
            std.process.exit(1);
        };
        return err;
    };
}

fn finish(x: anytype, s: *lanes.Stream) !void {
    const seq = x.seqs.get(s) orelse return error.UnknownQwenStream;
    const ctx = if (x.drafting.get(s)) |ds| &ds.ctx else null;
    try x.exact_cache.finishAfterFirst(s, seq, ctx, x.e.ops(), x.e.w.g, x.e.scratch.logits);
}

test {
    _ = @import("exact_post_first_test.zig");
}
