//! The idle keepalive's Metal target: one tiny command buffer on the engine's queue, touching its residency sets.
const std = @import("std");
pub const Target = TargetFor(@import("metal.zig"));

/// A queue and the residency sets one commit keeps wired. The engine owns both; this only reads them.
pub fn TargetFor(comptime Metal: type) type {
    return struct {
        const Self = @This();
        queue: Metal.Queue,
        sets: []const Metal.ResidencySet = &.{},
        scratch: ?Metal.Buffer = null, // a few bytes each tick copies within, so its command buffer runs on the GPU
        flush: ?Flush = null, // the sets' waiting changes committed first: while idle, never inside a request
        hold: ?Hold = null, // the owner's lock over the sets: a probe beside a request commits under it

        pub const Flush = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) void };
        pub const Hold = struct { ctx: *anyopaque, lock: *const fn (ctx: *anyopaque) void, unlock: *const fn (ctx: *anyopaque) void };

        /// The core ticker's tick: the sets' waiting changes, then the probe.
        pub fn tick(ctx: *anyopaque) void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const pool = Metal.objc.Pool.push();
            defer pool.pop();
            if (t.flush) |f| f.run(f.ctx);
            _ = t.probe();
        }

        /// The ticker's probe while a request runs: the sets stay in use and their waiting changes stay waiting.
        pub fn busy(ctx: *anyopaque) void {
            const t: *Self = @ptrCast(@alignCast(ctx));
            const pool = Metal.objc.Pool.push();
            defer pool.pop();
            if (t.hold) |h| h.lock(h.ctx); // the engine's thread adds to and removes from the sets while a request runs
            defer if (t.hold) |h| h.unlock(h.ctx);
            _ = t.probe();
        }

        /// One committed command buffer that touches every set, without the flush; a request may wait for it.
        pub fn probe(t: *const Self) Metal.CommandBuffer {
            const cmd = t.queue.commandBuffer();
            for (t.sets) |set| cmd.useResidencySet(set);
            if (t.scratch) |b| { // an empty command buffer need not reach the GPU, and then its sets stay as they were
                const blit = cmd.blit();
                blit.copy(b, 0, b, 32, 16);
                blit.end();
            }
            cmd.commit();
            return cmd;
        }
    };
}

/// A Metal that records what one target asks of it, in order: the CPU tests' stand-in.
const Recorded = struct {
    const Step = enum { flush, lock, unlock, use, copy, commit };
    var steps: [16]Step = undefined;
    var count: usize = 0;

    fn note(step: Step) void {
        steps[count] = step;
        count += 1;
    }

    fn noteFlush(_: *anyopaque) void {
        note(.flush);
    }

    fn noteLock(_: *anyopaque) void {
        note(.lock);
    }

    fn noteUnlock(_: *anyopaque) void {
        note(.unlock);
    }

    pub const objc = struct {
        pub const Pool = struct {
            pub fn push() Pool {
                return .{};
            }
            pub fn pop(_: Pool) void {}
        };
    };
    pub const ResidencySet = struct {};
    pub const Buffer = struct {};
    pub const Blit = struct {
        pub fn copy(_: Blit, _: Buffer, _: usize, _: Buffer, _: usize, _: usize) void {
            note(.copy);
        }
        pub fn end(_: Blit) void {}
    };
    pub const CommandBuffer = struct {
        pub fn useResidencySet(_: CommandBuffer, _: ResidencySet) void {
            note(.use);
        }
        pub fn blit(_: CommandBuffer) Blit {
            return .{};
        }
        pub fn commit(_: CommandBuffer) void {
            note(.commit);
        }
    };
    pub const Queue = struct {
        pub fn commandBuffer(_: Queue) CommandBuffer {
            return .{};
        }
    };
};

test "residency keepalive: the busy probe commits under the owner's hold and never flushes; the idle tick flushes first" {
    const T = TargetFor(Recorded);
    var owner: u8 = 0;
    const sets = [1]Recorded.ResidencySet{.{}};
    var t: T = .{ .queue = .{}, .sets = &sets, .scratch = .{}, .flush = .{ .ctx = &owner, .run = Recorded.noteFlush }, .hold = .{ .ctx = &owner, .lock = Recorded.noteLock, .unlock = Recorded.noteUnlock } };
    Recorded.count = 0;
    T.busy(&t);
    try std.testing.expectEqualSlices(Recorded.Step, &.{ .lock, .use, .copy, .commit, .unlock }, Recorded.steps[0..Recorded.count]);
    Recorded.count = 0;
    T.tick(&t);
    try std.testing.expectEqualSlices(Recorded.Step, &.{ .flush, .use, .copy, .commit }, Recorded.steps[0..Recorded.count]);
    t.hold = null; // a bare queue has no owner: the probe alone
    Recorded.count = 0;
    T.busy(&t);
    try std.testing.expectEqualSlices(Recorded.Step, &.{ .use, .copy, .commit }, Recorded.steps[0..Recorded.count]);
}
