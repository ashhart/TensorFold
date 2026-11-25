//! Host-tier admission: frame reservations, not the logical context size, decide whether a pool still fits.
const std = @import("std");
const kv = @import("core").kv_pool;
const host_tier = @import("exl3/kv/cuda_exl3_kvhost.zig");
const kvt = @import("exl3/kv/cuda_exl3_kvtier.zig");

pub fn Tests(comptime Lane: type, comptime Pool: type, comptime windowRows: fn () usize) type {
    return struct {
        test "exl3: host-tier admission follows frame reservations rather than logical context size" {
            const t = std.testing;
            const run_pages = 2;
            const slots = 9;
            const frames = comptime kvt.framesFor(run_pages, slots);
            const logical = frames + run_pages + 8;
            const page_bytes = 256;
            var backing: [2 * logical * page_bytes]u8 = undefined;
            var host: host_tier.Host = .{ .tier = undefined, .store = undefined, .kv_page = page_bytes };
            try host.tier.init(t.allocator, &backing, frames, page_bytes);
            defer host.tier.deinit();
            var lanes = [_]Lane{.{ .seq = undefined }};
            const pool: Pool = .{
                .gpa = t.allocator,
                .e = undefined,
                .layout = .{ .fixed_bytes = 0, .token_bytes = 1, .page_tokens = page_bytes, .snap_bytes = 0, .run_bytes = run_pages * page_bytes },
                .capacity = logical * page_bytes,
                .head = null,
                .plain = undefined,
                .pool = &host.tier.pool,
                .host = &host,
                .slot_frames = slots,
                .arena = undefined,
                .ibase = 0,
                .lanes = &lanes,
            };
            try t.expect(pool.fits(pool.capacity));
            {
                var lease: kv.tier.Lease = .{};
                try host.tier.reserve(&lease, 1, frames);
                defer host.tier.finish(&lease);
                try t.expect(!pool.fits(pool.capacity));
            }
            try t.expect(pool.fits(pool.capacity));
            {
                var lease: kv.tier.Lease = .{};
                try host.tier.reserve(&lease, 2 * logical, 0);
                defer host.tier.finish(&lease);
                try t.expect(!pool.fits(pool.capacity));
            }
            try t.expect(pool.fits(pool.capacity));
            lanes[0].used = true;
            try t.expect(!pool.fits(pool.capacity));
            lanes[0].used = false;
            try t.expect(!pool.fits(pool.capacity + windowRows() + 1));
        }
    };
}
