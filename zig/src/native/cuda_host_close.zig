//! Release a stopped native CUDA host in ownership order.
const failed_load = @import("cuda_load_gpu.zig");
const kv_pages = @import("cuda_kv_pool.zig");
pub fn close(h: anytype) void {
    h.host.stop();
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    if (h.recording) |r| r.free(h.gpa);
    failed_load.closeEntry(if (h.gpu) |g| &g.ctx else null) catch {};
    if (h.store) |s| kv_pages.closeStore(h.gpa, s);
    if (h.slide) |s| h.slide_free.?(h.gpa, s);
    h.gpa.free(h.dir);
    h.served.close(h.gpa);
    h.release(h.family);
    if (h.gpu) |g| {
        g.ctx.deinit();
        g.driver.close();
        h.gpa.destroy(g);
    }
    h.gpa.free(h.startup);
    h.gpa.destroy(h);
}
