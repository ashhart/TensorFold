//! A speech server's voices: Drama 3's catalog store, loaded at startup and listed on `/v1/voices`.
const std = @import("std");
const server = @import("server.zig");

/// Fills `srv.voices` from `--voices` and selects the default voice; voice flags without a speech model are refused.
pub fn start(srv: *server.Server, gpa: std.mem.Allocator, io: std.Io, config: server.Config) !void {
    const voice = config.voice;
    srv.voices = .{ .gpa = gpa, .io = io, .dir = voice.dir };
    if (srv.info.speech == null and (voice.dir != null or voice.inline_voices or voice.manage)) return error.VoicesRequireSpeech;
    if (voice.dir) |dir| try srv.voices.load(dir);
    _ = try srv.voices.select(srv.arena.allocator(), voice.default, "default");
}

/// The `/v1/voices` reply.
pub fn listing(srv: *server.Server, a: std.mem.Allocator) !@import("json").Value {
    return srv.voices.listing(a);
}
