//! The prompt cache's Nemotron side: the Mamba state and last row hip_serve's prefix copies, the pages a lane swaps in.
const std = @import("std");
const hip = @import("hip");
const serve = @import("hip_serve");
const radix = @import("engine_api").prompt_radix;
const engine = @import("engine.zig");
const pages = @import("pages.zig");
const lane_mod = @import("lanes.zig");

const Engine = engine.Engine;
const Lane = lane_mod.Lane;

/// A mark other than the history's is kept only this far from every other mark and the resume point.
pub const min_gap = 256;

/// The family's side of the cache: a snapshot is a lane's Mamba state and its last final-normed row.
pub const Side = struct {
    e: *Engine,
    state_bytes: usize,
    row_bytes: usize,

    pub const Owner = Lane;

    pub fn idOf(l: *const Lane) u32 {
        return l.id;
    }

    /// The lane's pages in position order.
    pub fn table(l: *const Lane) []const u32 {
        return l.s.ids.items;
    }

    /// Bytes one snapshot holds.
    pub fn snapBytes(f: *const Side) u64 {
        return f.state_bytes + f.row_bytes;
    }

    pub fn save(f: *Side, l: *Lane, buf: hip.DeviceBuffer) !void {
        const stream = f.e.stream.handle;
        try hip.raw.copy(buf, 0, l.s.mem.base(), f.state_bytes, stream);
        try hip.raw.copy(buf, f.state_bytes, l.hidden.base(), f.row_bytes, stream);
    }

    pub fn restore(f: *Side, l: *Lane, buf: hip.DeviceBuffer) !void {
        const stream = f.e.stream.handle;
        try hip.raw.copy(l.s.mem, 0, buf.base(), f.state_bytes, stream);
        try hip.raw.copy(l.hidden, 0, buf.base() + f.state_bytes, f.row_bytes, stream);
    }

    /// The lane's leading pages become `own`, held once more by it (the ones it had are released).
    pub fn adopt(f: *Side, l: *Lane, own: []const u32) !void {
        try l.s.adopt(&f.e.pool, own, true);
    }
};

pub const Prefix = serve.prefix.Prefix(Side);

/// The cache's rules: cuts on page edges, a gap between snapshots, the tail kept.
pub const rules: radix.Rules = .{ .page = pages.tokens, .min_gap = min_gap, .min_prompt = 0, .tail = true };

/// The cache over `side`, holding pages of the engine's pool and snapshots on its device.
pub fn init(gpa: std.mem.Allocator, side: *Side, tp: *serve.protocol.Sender) !*Prefix {
    const e = side.e;
    return Prefix.init(gpa, side, &e.driver, &e.pool.ids, e.pool.pageBytes(), rules, tp);
}
