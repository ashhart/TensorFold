//! The voices a speech server holds: Drama 3's catalog (--voices, inline, managed), or a ZONOS3 Mini release's own.
const std = @import("std");
const server = @import("server.zig");

/// At startup: Drama 3's catalog and its default voice, or the ZONOS3 release's, which takes no catalog options.
pub fn open(srv: *server.Server, gpa: std.mem.Allocator, io: std.Io, config: server.Config) !void {
    srv.voices = .{ .gpa = gpa, .io = io, .dir = config.voice.dir };
    const catalog = config.voice.dir != null or config.voice.inline_voices or config.voice.manage;
    if (srv.info.speech == null and catalog) return error.VoicesRequireSpeech;
    const z = config.zonos;
    if (z.steer_bundle != null and srv.info.speech == null) return error.SteerBundleNeedsZonos;
    if ((z.laci_heads != null or z.short_text != null) and srv.info.speech == null) return error.LaciHeadsNeedZonos;
    if (srv.info.speech != null) srv.zonos = try @import("zonos_speech.zig").load(srv.arena.allocator(), io, config.model_dir, config.voice.default, z.steer_bundle, z.laci_heads, z.short_text);
    if (srv.zonos != null) {
        if (catalog) return error.VoicesAreDrama3s; // a catalog ZONOS3 would never read
        return;
    }
    if (config.voice.dir) |dir| try srv.voices.load(dir);
    _ = try srv.voices.select(srv.arena.allocator(), config.voice.default, "default");
}
