//! Compile the actual native tail callbacks, including generic verify and the serial MTP wrappers, without execution.
const hip = @import("hip");
const Engine = @import("hip/engine.zig").Engine;
const Fx = @import("hip/lanes.zig").Fx;
const cache = @import("hip/cache.zig");
const forward = @import("hip/forward.zig");
const tail = @import("hip/tail.zig");
const Window = @import("lanes").backend.Window;

export fn flashnextHipTailCompile(e: *Engine, f: *Fx, c: *cache.Caches, windows: [*]const Window, done: *const @import("hip/engine.zig").Done, segs: [*]const forward.Seg, ones: *const hip.DeviceBuffer) void {
    tail.verify(f, windows[0..1], done.*, true) catch return;
    e.keepWindow(c, &.{1}, 1) catch return;
    const out = tail.run(e, e.ops(&e.arena), segs[0..1], done.out.streams, ones.*) catch return;
    var tokens: [1]u32 = undefined;
    var probs: [1]f32 = undefined;
    tail.draws(e, e.ops(&e.arena), out.mixed, &.{1}, &tokens, &probs) catch return;
}
