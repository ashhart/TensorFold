//! TF_QWEN27_METAL_KV_PAGES: the 27B's keys and values in pages of the core pool, a slot a lane (decode_round.Runner).
const std = @import("std");
const mtl = @import("metal");
const attn = @import("attention.zig");
const Ref = @import("projection.zig").Ref;
const Model = @import("model.zig").Model;
const kv_pool = @import("core").kv_pool;
pub const kvp = @import("core").metal_kv_pages;

/// TF_QWEN27_METAL_KV_PAGES=1: every slot's keys and values in pages of the core pool (off by default).
pub fn on() bool {
    const v = std.c.getenv("TF_QWEN27_METAL_KV_PAGES") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// A page: each attention layer's keys [kv_heads][256][head_dim] then its values, 256 rows of every layer.
pub fn pagedLayout(config: anytype, attention_layers: usize) kv_pool.Layout {
    return .{ .fixed_bytes = 0, .token_bytes = attention_layers * 2 * config.kvDim() * 2, .page_tokens = kvp.page_tokens, .snap_bytes = 0 };
}

/// The pool's pages, arena and tables a paged runner holds; slot i is lane i, promised as its stream's rows come.
pub const KvPages = struct { pages: *kvp.Pages, arena: mtl.Buffer, tables: mtl.Buffer };

/// A span of a slot's K/V rows in a buffer, as snapshot and digest copy and hash them.
pub const KvSpan = struct { buffer: mtl.Buffer, offset: usize, len: usize };

/// Slot `slot`'s first `at` rows of each attention layer's keys then values, head by head, page by page.
pub fn spans(r: anytype, a: std.mem.Allocator, slot: u32, at: u32, list: *std.ArrayList(KvSpan)) !void {
    const head_dim = r.model.config.head_dim;
    const row_bytes = head_dim * @sizeOf(u16);
    for (0..r.caches.len / r.slots) |layer| {
        const cache = r.caches[layer * r.slots + slot];
        for ([_]Ref{ cache.keys, cache.values }, [_]u32{ cache.key_stride, cache.value_stride }) |ref, stride| {
            if (r.kv) |k| {
                const ids = k.pages.ids(slot);
                if (ids.len * kvp.page_tokens < at) return error.PagesNotCovered;
                for (0..r.model.config.kv_heads) |head| {
                    var row: usize = 0;
                    while (row < at) : (row += kvp.page_tokens) {
                        const rows = @min(kvp.page_tokens, at - row);
                        const page = ref.offset + @as(usize, @intCast(k.pages.arena.offset(ids[row / kvp.page_tokens])));
                        try list.append(a, .{ .buffer = ref.buffer, .offset = page + head * kvp.page_tokens * row_bytes, .len = rows * row_bytes });
                    }
                }
                continue;
            }
            const head_stride = @as(usize, if (stride == 0) cache.capacity * head_dim else stride) * 2;
            for (0..r.model.config.kv_heads) |head| try list.append(a, .{ .buffer = ref.buffer, .offset = ref.offset + head * head_stride, .len = @as(usize, at) * row_bytes });
        }
    }
}

/// The pool over `slots` lanes of `capacity` rows each at most; each layer's caches its part of a page.
pub fn caches(a: std.mem.Allocator, m: *Model, layers: usize, slots: u32, capacity: u32, out: []attn.Cache) !KvPages {
    const l = pagedLayout(m.config, layers);
    const pages = @as(usize, slots) * l.pages(capacity);
    const sized = kvp.Pages.bytes(l, slots, pages, capacity);
    if (sized.arena > m.device.maxBufferLength()) return error.ArenaTooLarge;
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const arena = try m.device.buffer(@intCast(sized.arena), opts);
    errdefer arena.deinit();
    const tables = try m.device.buffer(@intCast(sized.tables), opts);
    errdefer tables.deinit();
    const p = try kvp.Pages.initIn(a, l, slots, pages, capacity, arena, tables);
    errdefer p.deinit();
    const part = @as(usize, kvp.page_tokens) * m.config.kvDim() * 2;
    for (0..layers) |layer| for (0..slots) |slot| {
        const table: Ref = .{ .buffer = tables, .offset = p.tables.offset(slot) };
        out[layer * slots + slot] = .{ .keys = .{ .buffer = arena, .offset = 2 * layer * part }, .values = .{ .buffer = arena, .offset = (2 * layer + 1) * part }, .capacity = capacity, .pages = .{ .table = table, .page_bytes = l.pageBytes() } };
    };
    std.log.info("qwen27: KV in pages of the core pool: {d} slots over {d} pages of 256 rows ({d:.2} GiB, {d:.2} MiB a page)", .{ slots, pages, @as(f64, @floatFromInt(sized.arena)) / (1 << 30), @as(f64, @floatFromInt(l.pageBytes())) / (1 << 20) });
    return .{ .pages = p, .arena = arena, .tables = tables };
}

/// Every lane finished, every page must be free again: the census says so, or warns.
pub fn free(k: KvPages) void {
    for (0..k.pages.leases.len) |lane| k.pages.finish(lane);
    const c = k.pages.pool.census();
    if (k.pages.idle() and c.free == c.total) {
        std.log.info("qwen27: KV pages idle, {d} of {d} free", .{ c.free, c.total });
    } else std.log.warn("qwen27: every slot closed, yet {d} of {d} KV pages are not free", .{ c.total - c.free, c.total });
    k.pages.deinit();
    k.arena.deinit();
    k.tables.deinit();
}
