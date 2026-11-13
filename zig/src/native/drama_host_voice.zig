//! A reference WAV file's codes for POST /v1/voices, encoded on the speech host's own thread by an idle lane's codec.
const std = @import("std");
const api = @import("engine_api");

/// One waiting encode; the asking thread sleeps on `done` under the host's mutex.
const Ask = struct {
    gpa: std.mem.Allocator,
    wav: []const u8,
    codes: [][10]i32 = &.{},
    err: ?api.EncodeVoiceError = null,
    done: bool = false,
    cond: std.Io.Condition = .init,
};

/// Asks waiting for the host thread: one at a time, in arrival order.
pub const Queue = struct { turn: std.Io.Mutex = .init, ask: ?*Ask = null };

/// Errors of the file itself: the request's to fix, so the engine stays open.
fn input(err: anyerror) bool {
    for ([_]anyerror{ error.NotWav, error.BadWav, error.UnsupportedWav, error.BadReference, error.EmptyAudio }) |e| if (err == e) return true;
    return false;
}

pub fn For(comptime Host: type) type {
    return struct {
        /// The engine's encode_voice: hands the file to the host thread and waits for its codes.
        pub fn encode(p: *anyopaque, gpa: std.mem.Allocator, wav: []const u8) api.EncodeVoiceError![][10]i32 {
            const h: *Host = @ptrCast(@alignCast(p));
            h.voices.turn.lockUncancelable(h.io);
            defer h.voices.turn.unlock(h.io);
            var ask: Ask = .{ .gpa = gpa, .wav = wav };
            h.mutex.lockUncancelable(h.io);
            defer h.mutex.unlock(h.io);
            if (h.closing or h.failed.closed) return error.Closed;
            h.voices.ask = &ask;
            h.wake.signal(h.io);
            while (!ask.done) ask.cond.waitUncancelable(h.io, &h.mutex);
            return ask.err orelse ask.codes;
        }

        /// On the host thread with its mutex held: runs a waiting encode on an idle lane; true: done, mutex released.
        pub fn step(h: *Host) bool {
            const ask = h.voices.ask orelse return false;
            if (h.closing or h.failed.closed) {
                refuse(h);
                h.mutex.unlock(h.io);
                return true;
            }
            const lane = for (h.jobs[0..h.queue.lanes], 0..) |job, i| {
                if (job == null) break i;
            } else return false; // every lane busy: the ask waits for one to finish
            h.voices.ask = null;
            h.mutex.unlock(h.io);
            const t = h.sessions[lane].t;
            const outcome: ?api.EncodeVoiceError = if (t.encodeWav(ask.wav)) |codes| copy: {
                defer t.c.gpa.free(codes);
                ask.codes = ask.gpa.dupe([10]i32, codes) catch break :copy error.OutOfMemory;
                break :copy null;
            } else |err| if (err == error.OutOfMemory) error.OutOfMemory else if (input(err)) error.BadWav else fail: {
                std.log.warn("drama3: reference encode failed, closing the speech engine: {s}", .{@errorName(err)});
                break :fail error.Failed;
            };
            h.mutex.lockUncancelable(h.io);
            if (outcome) |e| if (e == error.Failed) {
                h.failed.closed = true; // a device error: as in advance, the engine stops serving
            };
            ask.err = outcome;
            ask.done = true;
            ask.cond.signal(h.io);
            h.mutex.unlock(h.io);
            return true;
        }

        /// With the host's mutex held and the engine closing: a waiting encode gets Closed.
        pub fn refuse(h: *Host) void {
            const ask = h.voices.ask orelse return;
            h.voices.ask = null;
            ask.err = error.Closed;
            ask.done = true;
            ask.cond.signal(h.io);
        }
    };
}

test {
    _ = @import("drama_host_voice_test.zig");
}
