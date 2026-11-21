//! Pair.init's link and GPU objects in the order they are made, generic over the kit that makes them.
const std = @import("std");

/// The started pair, or the error with every object made before it released once; `jobs` is the layout's JOBS.
pub fn start(comptime K: type, k: *K, gpa: std.mem.Allocator, comptime jobs: usize) !*K.Made {
    // each object's errdefer follows it: a failed step releases what came before it, newest first, each once
    const ep = try k.endpoint();
    errdefer ep.deinit();
    // the window's buffer and the pipelines: released before the endpoint frees the memory the buffer aliases
    var owned: K.Owned = .{};
    errdefer owned.deinit();
    const rd = ep.rdma();
    const win = rd.window();
    const lib = try k.library();
    defer lib.deinit();
    const fence = try k.fence();
    errdefer fence.deinit();
    const posted = try k.event();
    errdefer posted.deinit();
    const landed = try k.event();
    errdefer landed.deinit();
    const p = try gpa.create(K.Made);
    errdefer gpa.destroy(p);
    const wbuf = try owned.keep(gpa, try k.window(win));
    const stage_pipe = try owned.keep(gpa, try k.pipeline(lib, "pair_stage"));
    const gather_pipe = try owned.keep(gpa, try k.pipeline(lib, "pair_gather"));
    const argmax_pipe = try owned.keep(gpa, try k.pipeline(lib, "pair_argmax"));
    const pick_pipe = try owned.keep(gpa, try k.pipeline(lib, "pair_pick"));
    const bytes = try gpa.alloc(u32, jobs);
    errdefer gpa.free(bytes);
    k.assemble(p, .{
        .ep = ep,
        .rd = rd,
        .win = win,
        .wbuf = wbuf,
        .stage_pipe = stage_pipe,
        .gather_pipe = gather_pipe,
        .argmax_pipe = argmax_pipe,
        .pick_pipe = pick_pipe,
        .fence = fence,
        .posted = posted,
        .landed = landed,
        .bytes = bytes,
    });
    // both Macs up before the first exchange, then the service thread: the last two steps that can fail
    try k.hello(p);
    p.thread = try k.spawn(p);
    // the pair owns them only now: until here `owned`'s errdefer alone releases them, so never twice and never none
    p.owned = owned;
    return p;
}
