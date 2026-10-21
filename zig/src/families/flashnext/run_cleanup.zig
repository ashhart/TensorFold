//! Drain every owned queue before buffers, pipelines or the pair's no-copy endpoint are released.
const mtl = @import("metal");
const platform = @import("sys");

pub fn drop(r: anytype) void {
    if (r.closed) return;
    r.closed = true;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    if (r.tp) |t| t.stopService();
    r.objects.drain();
    const cb = r.queue.commandBuffer();
    cb.commit();
    cb.wait();
    while (r.held.items.len > 0) {
        const b = r.held.items[r.held.items.len - 1];
        r.held.items.len -= 1;
        if (r.follow) |f| f.leave(f.ctx, b);
        if (r.tp) |t| if (b.id == t.wbuf.id or b.id == t.one.id or b.id == t.pick_tmp.id) continue;
        b.deinit();
    }
    if (r.rows_cleanup) |f| f.call(f.ctx);
    var wide = r.wide_pipes.valueIterator();
    while (wide.next()) |p| p.deinit();
    r.objects.release();
    for (r.maps.items) |m| m.deinit();
    r.maps.clearRetainingCapacity();
    for (r.files.items) |fd| _ = platform.c.close(fd);
    r.files.clearRetainingCapacity();
    r.queue.deinit();
    if (r.tp) |t| {
        t.deinit();
        r.tp = null;
    }
}
