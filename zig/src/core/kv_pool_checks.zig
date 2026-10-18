//! The page pool's checks beside engine_api: under the paged prompt store and a lane host (kv_pool is its own module).
test {
    _ = @import("kv_pool_store_test.zig");
    _ = @import("kv_pool_host_test.zig");
}
