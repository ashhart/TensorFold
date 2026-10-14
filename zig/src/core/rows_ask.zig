//! The Mac app's questions about a conversation's kept states (/v1/imprint), answered on the engine thread.
const std = @import("std");
const pc = @import("prompt_cache.zig");
const rows = pc.rows;

pub const Op = enum { where, load, forget };

/// One question: a rendered conversation (its tokens, chunk starts and decoded spans), and the answer once `done`.
pub const Ask = struct {
    op: Op,
    prompt: []const u32 = &.{},
    starts: []const u32 = &.{},
    spans: []const [2]u32 = &.{},
    out: rows.read.Resume = .{},
    loaded: u32 = 0, // load: the tokens a disk read brought into RAM (0: none was needed, or it did not pay)
    load_seconds: f64 = 0,
    freed: u64 = 0, // forget: bytes removed from disk
    disk: bool = false, // the engine keeps conversation states on disk
    done: std.atomic.Value(bool) = .init(false),

    /// The engine thread's answer from its kept states (`store` null: it keeps none).
    pub fn answer(a: *Ask, store: ?*pc.Store) void {
        defer a.done.store(true, .release);
        const s = store orelse return;
        a.disk = s.disk != null;
        switch (a.op) {
            .where => {},
            .load => {
                const have = s.find(a.prompt, a.starts, a.spans);
                const t0 = rows.write.nowNs();
                const got = rows.read.recall(s, a.prompt, @intCast(a.prompt.len), a.starts, a.spans, have);
                if (got != null and got != have) {
                    a.loaded = got.?.at;
                    a.load_seconds = @as(f64, @floatFromInt(rows.write.nowNs() - t0)) / 1e9;
                }
            },
            .forget => if (s.disk) |r| {
                a.freed = r.forgetAll();
            },
        }
        a.out = rows.read.resumable(s, a.prompt, a.starts, a.spans);
    }

    /// The answer within `ms` milliseconds (false: none came).
    pub fn wait(a: *const Ask, io: std.Io, ms: u32) bool {
        var waited: u32 = 0;
        while (!a.done.load(.acquire)) : (waited += 2) {
            if (waited >= ms) return false;
            std.Io.sleep(io, .fromMilliseconds(2), .awake) catch {};
        }
        return true;
    }
};
