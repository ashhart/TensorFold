//! Opt-in mapped-host KV pages; recurrent state and verify runs stay on the device.
const std = @import("std");
const environment = @import("sys").environment;

pub fn onHost() bool {
    const v = @import("profile").get(.TF_QWEN_KV_HOST) orelse return false;
    return std.mem.eql(u8, v, "1");
}

/// Only the explicit host path reads the GiB cap or the model's geometry.
pub fn room(l: anytype) !u64 {
    if (!onHost()) return 0;
    const cap = try environment.read(u64, "TF_QWEN_KV_HOST_GIB", 0);
    const p = @import("cuda_pages.zig");
    const layout = p.layout(l.e, l.capacity, l.page_tokens, l.ringBytes(), l.extra);
    return bytes(l.capacity + p.window_rows, l.page_tokens, layout.pageBytes(), if (cap == 0) null else cap << 30);
}

/// A whole window of KV pages, never state runs or unused host placeholders for them.
pub fn bytes(rows: usize, tokens: usize, page_bytes: u64, cap: ?u64) !u64 {
    if (tokens == 0 or page_bytes == 0) return error.QwenKvHostNeedsPages;
    const n = try std.math.divCeil(u64, rows, tokens);
    const total = try std.math.mul(u64, n, page_bytes);
    if (cap) |limit| if (limit < total) return error.QwenKvHostCap;
    return total;
}

pub fn checkDevice(integrated: bool, pages: usize, all: bool) !void {
    if (pages == 0) return error.QwenKvHostNeedsPages;
    if (!all) return error.QwenKvHostNeedsPagedEngine;
    if (integrated) return error.QwenKvHostIntegrated;
}

pub fn checkShared(snaps: usize) !void {
    if (snaps > 0) return error.QwenKvHostShare;
}

/// Host capacity mode admits one stream and no disk-backed prompt snapshots.
pub fn refusal(o: anytype) ?[]const u8 {
    if (!onHost()) return null;
    if (o.lanes != 1) return "TF_QWEN_KV_HOST supports one stream: set --parallel 1";
    if (o.spill != .off) return "TF_QWEN_KV_HOST cannot use --spill-gib; serve without spill";
    if (o.learn != null or o.learn_conversations) return "TF_QWEN_KV_HOST cannot use disk snapshots; serve without --learn or --learn-conversations";
    return null;
}
