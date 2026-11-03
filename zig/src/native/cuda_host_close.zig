//! Release a stopped native CUDA host in ownership order.
const failed_load = @import("cuda_load_gpu.zig");
const kv_pages = @import("cuda_kv_pool.zig");
pub fn close(h: anytype) void {
    h.host.stop();
    if (comptime @hasField(@typeInfo(@TypeOf(h)).pointer.child, "before_close")) {
        wait(h) catch {
            Guard(@typeInfo(@TypeOf(h)).pointer.child).keep(h);
            return;
        };
    }
    finish(h);
}
fn wait(h: anytype) !void {
    if (h.before_close) |f| try f(h.family);
}
fn Guard(comptime H: type) type {
    return @import("close_guard.zig").Queue(H, struct {
        fn f(h: *H) anyerror!void {
            try wait(h);
        }
    }.f, struct {
        fn f(h: *H) void {
            finish(h);
        }
    }.f);
}
pub fn recover(comptime H: type) void {
    Guard(H).recover();
}
fn finish(h: anytype) void {
    kv_pages.unwatch(&h.kv); // the engine thread is joined: the pool's teardown below reports to no one
    h.core.deinit();
    h.cfg.deinit(h.gpa);
    if (h.recording) |r| r.free(h.gpa);
    failed_load.closeEntry(if (h.gpu) |g| &g.ctx else null) catch {};
    if (h.store) |s| kv_pages.closeStore(h.gpa, s);
    if (h.prompts) |x| x.close(); // its pages back to the pool before the family frees the pool
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
