//! Validate a complete bounded-state image before uploading or publishing any metadata.
const std = @import("std");
const sg = @import("sessions_gpu.zig");
const ced = @import("ced.zig");
const eh = @import("engram_host.zig");
const Blob = @import("kv").sessions.disk.Blob;
pub const Metadata = struct { pos: u64, tail_len: usize, tail: [eh.max_ngram]u32, stash: ced.Stash, ds: u64 };
pub fn validate(s: *const sg.Sessions, bs: []const Blob) !Metadata {
    if (bs.len != s.pieces.len + 1) return error.BadBlob;
    var slot: ?[]const u8 = null;
    for (bs, 0..) |b, i| {
        for (bs[0..i]) |prior| if (std.mem.eql(u8, b.name, prior.name)) return error.BadBlob;
        if (std.mem.eql(u8, b.name, "slot")) {
            slot = b.bytes;
            continue;
        }
        const pc = for (s.pieces) |p| {
            if (std.mem.eql(u8, p.role, b.name)) break p;
        } else return error.BadBlob;
        if (pc.len != b.bytes.len) return error.BadBlob;
    }
    const bytes = slot orelse return error.BadBlob;
    if (bytes.len != sg.Sessions.slot_bytes) return error.BadBlob;
    const pos = std.mem.readInt(u64, bytes[0..8], .little);
    const tail_len = std.mem.readInt(u64, bytes[8..16], .little);
    if (pos > s.kx.opts.capacity or tail_len >= eh.max_ngram or tail_len > pos) return error.BadBlob;
    var m: Metadata = .{ .pos = pos, .tail_len = @intCast(tail_len), .tail = @splat(0), .stash = .{}, .ds = std.mem.readInt(u64, bytes[sg.Sessions.ds_at..][0..8], .little) };
    for (&m.tail, 0..) |*t, i| {
        t.* = std.mem.readInt(u32, bytes[16 + 4 * i ..][0..4], .little);
        if (i < tail_len and t.* >= s.f.cfg.vocab) return error.BadBlob;
    }
    const c = bytes[sg.Sessions.ced_at..];
    m.stash.from = std.mem.readInt(u64, c[0..8], .little);
    m.stash.hi = std.mem.readInt(u64, c[8..16], .little);
    if (m.stash.from > m.stash.hi or m.stash.hi > pos or m.stash.hi - m.stash.from > ced.ring) return error.BadBlob;
    if (!s.measure.replay and (m.stash.from != 0 or m.stash.hi != 0)) return error.BadBlob;
    for (&m.stash.ids, 0..) |*t, i| t.* = std.mem.readInt(u32, c[16 + 4 * i ..][0..4], .little);
    var at = m.stash.from;
    while (at < m.stash.hi) : (at += 1) if (m.stash.ids[@intCast(at % ced.ring)] >= s.f.cfg.vocab) return error.BadBlob;
    if (m.ds > pos or m.ds > s.measure.window or (!s.ds_rings and m.ds != 0)) return error.BadBlob;
    return m;
}
