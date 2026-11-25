//! A speech server's voices, Drama 3's store or a ZONOS3 release's own: loaded at startup, listed on `/v1/voices`.
const std = @import("std");
const server = @import("server.zig");
const zonos = @import("zonos_speech.zig");

/// Fills `srv.voices` from `--voices`, or `srv.zonos` for a release, which refuses Drama 3's voice flags.
pub fn start(srv: *server.Server, gpa: std.mem.Allocator, io: std.Io, config: server.Config) !void {
    const voice = config.voice;
    srv.voices = .{ .gpa = gpa, .io = io, .dir = voice.dir };
    if (srv.info.speech == null and (voice.dir != null or voice.inline_voices or voice.manage)) return error.VoicesRequireSpeech;
    if (srv.info.speech != null) srv.zonos = try zonos.load(srv.arena.allocator(), io, config.model_dir, voice.default);
    if (srv.zonos != null) {
        if (voice.dir != null or voice.inline_voices or voice.manage) return error.VoiceFlagsNeedDrama;
        return;
    }
    if (voice.dir) |dir| try srv.voices.load(dir);
    _ = try srv.voices.select(srv.arena.allocator(), voice.default, "default");
}

/// The `/v1/voices` reply.
pub fn listing(srv: *server.Server, a: std.mem.Allocator) !@import("json").Value {
    if (srv.zonos) |state| return zonos.listing(a, state);
    return srv.voices.listing(a);
}
