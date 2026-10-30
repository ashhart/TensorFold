//! The prompt cache's Qwen3.5 side: what hip_serve's prefix copies for a snapshot, and the pages a stream swaps in.

const std = @import("std");
const hip = @import("hip");
const serve = @import("hip_serve");
const radix = @import("engine_api").prompt_radix;
const memory = @import("memory.zig");
const state = @import("../forward/state.zig");
const pages = @import("../forward/pages.zig");
const Engine = @import("engine.zig").Engine;

/// The family's side of the cache: linear snapshots of the engine's streams.
pub const Side = struct {
    gpa: std.mem.Allocator,
    e: *Engine,

    /// What a stream gives the prompt cache: its caches and the id the other ranks know it by.
    pub const Owner = struct { caches: *state.Caches, id: u32 };

    pub fn idOf(o: *const Owner) u32 {
        return o.id;
    }

    /// The stream's pages in position order.
    pub fn table(o: *const Owner) []const u32 {
        return o.caches.table.items;
    }

    /// Bytes one snapshot holds: the linear layers' state.
    pub fn snapBytes(f: *const Side) u64 {
        return memory.linearBytes(f.e.model().spec);
    }

    pub fn save(f: *Side, o: *Owner, buf: hip.DeviceBuffer) !void {
        try o.caches.saveLinear(buf, f.e.stream.handle);
    }

    pub fn restore(f: *Side, o: *Owner, buf: hip.DeviceBuffer) !void {
        try o.caches.loadLinear(buf, f.e.stream.handle);
    }

    /// The stream's leading pages become `own`, held once more by it (the ones it had are released).
    pub fn adopt(f: *Side, o: *Owner, own: []const u32) !void {
        for (own) |id| f.e.pool.retain(id);
        try o.caches.set(f.gpa, 0, own);
    }
};

pub const Prefix = serve.prefix.Prefix(Side);

/// The cache's rules: cuts on page edges, a gap between snapshots, the tail kept.
pub const rules: radix.Rules = .{
    .page = pages.tokens,
    .min_gap = @import("prefix.zig").min_gap,
    .min_prompt = 0,
    .tail = true,
};

/// The cache over `side`, holding pages of the engine's pool and snapshots on its device.
pub fn init(gpa: std.mem.Allocator, side: *Side, tp: *serve.protocol.Sender) !*Prefix {
    const e = side.e;
    return Prefix.init(gpa, side, &e.driver, &e.pool.ids, e.pool.pageBytes(), rules, tp);
}
