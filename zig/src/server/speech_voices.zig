//! A speech server's voices at startup: Drama 3's catalog store, or a ZONOS3 Mini release's own voices.
const std = @import("std");
const server = @import("server.zig");
const zonos = @import("zonos_speech.zig");

/// Fills `srv.voices` for Drama 3 or `srv.zonos` for a release, whose voices are its own: Drama 3's flags are refused.
pub fn start(srv: *server.Server, gpa: std.mem.Allocator, io: std.Io, config: server.Config) !void {
    const voice = config.voice;
    srv.voices = .{ .gpa = gpa, .io = io, .dir = voice.dir };
    if (srv.info.speech == null and (voice.dir != null or voice.inline_voices or voice.manage)) return error.VoicesRequireSpeech;
    if (config.zonos.steer_bundle != null and srv.info.speech == null) return error.SteerBundleNeedsZonos;
    if (srv.info.speech != null) srv.zonos = try zonos.load(srv.arena.allocator(), io, config.model_dir, voice.default, config.zonos.steer_bundle);
    if (srv.zonos != null) {
        if (voice.dir != null or voice.inline_voices or voice.manage) return error.VoiceFlagsNeedDrama;
        return;
    }
    if (voice.dir) |dir| try srv.voices.load(dir);
    _ = try srv.voices.select(srv.arena.allocator(), voice.default, "default");
}
