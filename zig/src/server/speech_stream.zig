const std = @import("std");
const platform = @import("sys");
const api = @import("engine_api");
const log = @import("log.zig");
const clock = @import("clock.zig");
const Mailbox = @import("mailbox.zig").Mailbox;
const Server = @import("server.zig").Server;
const Conn = @import("http_conn.zig").Conn;
const Cx = @import("errors.zig").Cx;
const Metadata = @import("speech_metadata.zig").Metadata;
const speech = @import("speech.zig");
const Plan = speech.Plan;
const Format = speech.Format;
const quantizeInPlace = speech.quantizeInPlace;
const fail = speech.fail;

/// The client left before the reply ended; the synthesis was cancelled and waited out.
pub fn left(srv: *Server, c: *Conn, id: []const u8, prompt_len: usize, received: i96) void {
    c.close = true;
    srv.metrics.disconnected(srv.io);
    var buf: [1024]u8 = undefined;
    log.line("{s}", .{log.ended(&buf, id, null, prompt_len, 0, clock.seconds(clock.nowNs(srv.io) - received))});
}

/// Why a finished synthesis has no complete audio, or null; an empty reply is no fault, length and silence clip it.
pub fn why(box: *const Mailbox, samples: usize, capacity: usize) ?[]const u8 {
    return switch (box.reason) {
        .stop, .length, .silence => if (samples > capacity) "the engine wrote more audio than the reply holds" else null,
        .cancelled => "speech synthesis was cancelled",
        .failed => if (box.message.len > 0) box.message else "speech synthesis failed",
    };
}

/// An error reply when synthesis ends before any audio; else the chunks, and the terminator only after a complete one.
pub fn streamReply(srv: *Server, c: *Conn, cx: *Cx, id: []const u8, sp: api.SpeechInfo, plan: Plan, received: i96, box: *Mailbox, engine_id: api.Id, pcm: []f32, result: *const api.SpeechResult, metadata: Metadata, lane_released: *bool) void {
    const sent = streamAudio(srv, c, sp, box, engine_id, pcm, metadata) orelse return left(srv, c, id, plan.prompt.len, received);
    // The engine is finished: release before final output, including an error reply or an empty stream's head.
    speech.releaseLane(srv, lane_released);
    const mismatch: ?[]const u8 = if (result.samples != sent.samples) "the finished synthesis does not match the audio it published" else null;
    const problem = why(box, result.samples, pcm.len) orelse mismatch;
    if (sent.samples == 0) {
        if (problem) |message| return fail(c, cx, id, metadata, .server, message);
        // the model closed without a frame: an empty stream, not a fault
        startStream(c, sp.sample_rate, metadata) catch {
            c.close = true;
            return;
        };
        c.writeAll(if (box.reason == .length) length_chunk else last_chunk) catch {
            c.close = true;
            return;
        };
        c.response_complete = true;
        log.line("speech {s} prompt={d} samples=0 format=pcm stream=true finish={s} seconds={d:.2}", .{ id, plan.prompt.len, @tagName(box.reason), clock.seconds(clock.nowNs(srv.io) - received) });
        return;
    }
    if (problem) |message| {
        c.close = true; // no terminating chunk: the client's HTTP/1.1 parser reports an incomplete body
        log.line("speech {s} stream ended incomplete after samples={d}: {s}", .{ id, sent.samples, message });
        return;
    }
    c.writeAll(if (box.reason == .silence) silence_chunk else if (box.reason == .length) length_chunk else last_chunk) catch {
        c.close = true;
        return;
    };
    c.response_complete = true;
    log.line("speech {s} prompt={d} samples={d} format=pcm stream=true finish={s} first_audio_ready_seconds={d:.3} first_frame_written_seconds={d:.3} seconds={d:.2}", .{
        id, plan.prompt.len, sent.samples, @tagName(box.reason), clock.seconds(sent.first_ready - received), clock.seconds(sent.first_written - received), clock.seconds(clock.nowNs(srv.io) - received),
    });
}

/// What a stream sent, when this thread first saw published audio, and when frame 0's own bytes reached the socket.
const Sent = struct { samples: usize = 0, first_ready: i96 = 0, first_written: i96 = 0 };

/// Sends everything published since the last send as one chunk, converted in place; null: the client left.
fn streamAudio(srv: *Server, c: *Conn, sp: api.SpeechInfo, box: *Mailbox, engine_id: api.Id, pcm: []f32, metadata: Metadata) ?Sent {
    var sent: Sent = .{};
    var size: [20]u8 = undefined;
    while (true) {
        const now = settle(srv, c, box, engine_id, sent.samples) orelse return null;
        if (now == sent.samples) return sent; // finished, and everything published is sent
        const first = sent.samples == 0;
        if (first) {
            sent.first_ready = clock.nowNs(srv.io);
            startStream(c, sp.sample_rate, metadata) catch return abandon(srv, box, engine_id);
        }
        // the audio event made pcm[..now] final: this thread converts it in place
        const bytes = quantizeInPlace(pcm[sent.samples..now]) catch return abandon(srv, box, engine_id);
        // frame 0's bytes go apart, so their hand-off time does not depend on the chunk's size
        const split: usize = if (first) @min(2 * sp.frame_length, bytes.len) else 0;
        c.writeAll(std.fmt.bufPrint(&size, "{x}\r\n", .{bytes.len}) catch unreachable) catch return abandon(srv, box, engine_id);
        c.writeAll(bytes[0..split]) catch return abandon(srv, box, engine_id);
        if (first) sent.first_written = clock.nowNs(srv.io);
        c.writeAll(bytes[split..]) catch return abandon(srv, box, engine_id);
        c.writeAll("\r\n") catch return abandon(srv, box, engine_id);
        sent.samples = now;
    }
}

/// Cancels a synthesis whose client is gone, then waits for its finished event; null, for the callers' `return`.
fn abandon(srv: *Server, box: *Mailbox, engine_id: api.Id) ?Sent {
    srv.engine.cancel(engine_id);
    box.mutex.lockUncancelable(box.io);
    defer box.mutex.unlock(box.io);
    while (!box.finished) box.cond.waitUncancelable(box.io, &box.mutex);
    return null;
}

/// Waits for `finished` or (with `sent`) samples past it, cancelling if the client leaves; samples so far, or null.
pub fn settle(srv: *Server, c: *Conn, box: *Mailbox, engine_id: api.Id, sent: ?usize) ?usize {
    while (true) {
        box.mutex.lockUncancelable(box.io);
        const idle = if (sent) |n| box.audio == n else true;
        if (!box.finished and idle) box.cond.waitTimeout(box.io, &box.mutex, .{ .duration = .{ .raw = .fromMilliseconds(50), .clock = .awake } }) catch {};
        const finished = box.finished;
        const published = box.audio;
        box.mutex.unlock(box.io);
        if (finished or (if (sent) |n| published > n else false)) return published;
        if (c.peerGone()) {
            _ = abandon(srv, box, engine_id);
            return null;
        }
    }
}

/// 200 with raw PCM16LE in HTTP/1.1 chunks, under the reference's sample-format headers.
fn startStream(c: *Conn, sample_rate: u32, metadata: Metadata) error{Closed}!void {
    c.startResponse(200, null) catch return error.Closed;
    c.addHeader("Content-Type", Format.pcm.contentType()) catch return error.Closed;
    c.addHeader("Transfer-Encoding", "chunked") catch return error.Closed;
    var rate: [16]u8 = undefined;
    c.addHeader("X-Sample-Rate", std.fmt.bufPrint(&rate, "{d}", .{sample_rate}) catch unreachable) catch return error.Closed;
    c.addHeader("X-Channels", "1") catch return error.Closed;
    c.addHeader("X-Bit-Depth", "16") catch return error.Closed;
    c.addHeader("Trailer", "X-Finish-Reason") catch return error.Closed;
    metadata.add(c) catch return error.Closed;
    if (c.close) c.addHeader("Connection", "close") catch return error.Closed;
    // each chunk leaves at once rather than waiting on the client's acknowledgement of the last (no effect off TCP)
    const one: c_int = 1;
    if (platform.windows) platform.posix.setsockopt(c.fd, platform.posix.IPPROTO.TCP, platform.posix.TCP.NODELAY, std.mem.asBytes(&one)) catch {} else _ = std.c.setsockopt(c.fd, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, &one, @sizeOf(c_int));
    // a blocking write into a full socket returns after the stall limit, so writeAll's poll can end a stalled stream
    const stall: platform.posix.timeval = .{ .sec = @divTrunc(c.timeouts.write_ms, 1000), .usec = @rem(c.timeouts.write_ms, 1000) * 1000 };
    platform.posix.setsockopt(c.fd, platform.posix.SOL.SOCKET, platform.posix.SO.SNDTIMEO, std.mem.asBytes(&stall)) catch {};
    try c.finish("");
    c.response_complete = false;
}

/// Ends a complete stream; length and silence use an X-Finish-Reason trailer instead.
const last_chunk = "0\r\n\r\n";
const length_chunk = "0\r\nX-Finish-Reason: length\r\n\r\n";
const silence_chunk = "0\r\nX-Finish-Reason: silence\r\n\r\n";
