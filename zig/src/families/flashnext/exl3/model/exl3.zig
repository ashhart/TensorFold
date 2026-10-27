//! Host-side EXL3 index for Qwen3.8-Flash-Next: the float64 oracles and header rules kept as tests under option C

pub const shape = @import("../pack/exl3_shape.zig");
pub const scan = @import("../pack/exl3_scan.zig");
pub const tp = @import("../tp/exl3_tp.zig");
pub const cache = @import("exl3_cache.zig");
pub const cpu = @import("exl3_cpu.zig");
pub const place = @import("../pack/exl3_place.zig");
pub const route = @import("exl3_route.zig");
pub const router = @import("exl3_router.zig");
pub const kv = @import("../kv/exl3_kv.zig");
pub const norm = @import("exl3_norm.zig");
pub const hc = @import("exl3_hc.zig");
pub const moe = @import("exl3_moe.zig");
pub const gdn_cpu = @import("exl3_gdn_cpu.zig");
pub const attn_cpu = @import("exl3_attn_cpu.zig");
pub const fwd_cpu = @import("exl3_fwd_cpu.zig");

test {
    _ = shape;
    _ = scan;
    _ = tp;
    _ = cache;
    _ = cpu;
    _ = place;
    _ = route;
    _ = router;
    _ = kv;
    _ = norm;
    _ = hc;
    _ = moe;
    _ = gdn_cpu;
    _ = attn_cpu;
    _ = fwd_cpu;
}
