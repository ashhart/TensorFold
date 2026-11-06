//! The prompt-only exceptional selector. The finite dispatch uses the original pipeline and argument layout.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");
const frags = @import("../../core/frags.zig");
pub const Args = extern struct { len: u32, cap: u32, scale: f32, subs: u32 };
pub const Ref = struct { buf: mtl.Buffer, off: usize = 0 };
pub const Pipelines = struct {
    masked: mtl.Pipeline,
    scan: mtl.Pipeline,
    pick: mtl.Pipeline,
    pub fn load(device: mtl.Device) !Pipelines {
        const a = std.heap.c_allocator;
        const body = try frags.source(device, a, sources.lfm2_prompt_masked.source);
        defer a.free(body);
        const text = try std.mem.concat(a, u8, &.{ sources.lfm2, "\n", body });
        defer a.free(text);
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const masked = try mtl.Pipeline.init(device, lib, "lfm2_attn_masked", false);
        errdefer masked.deinit();
        const scan = try mtl.Pipeline.init(device, lib, "lfm2_prompt_scan", false);
        errdefer scan.deinit();
        return .{ .masked = masked, .scan = scan, .pick = try mtl.Pipeline.init(device, lib, "lfm2_prompt_pick", false) };
    }
    pub fn deinit(p: Pipelines) void {
        p.masked.deinit();
        p.scan.deinit();
        p.pick.deinit();
    }
};
/// The caller owns and zero-initializes 64 bytes of pick scratch, reused after each pick clears its flag.
pub fn launch(e: mtl.ComputeEncoder, pipes: Pipelines, original: mtl.Pipeline, inputs: [3]Ref, partials: [3]Ref, scratch: Ref, p: Args, rows: usize) void {
    e.setPipeline(pipes.scan);
    e.setBuffer(inputs[2].buf, inputs[2].off, 0);
    e.setValue(p, 1);
    e.setBuffer(scratch.buf, scratch.off, 2);
    e.setValue(@as(u32, @intCast(rows)), 3);
    e.dispatchGroups(mtl.Size.of(1, @min((rows + 32) / 2, 32), 8), mtl.Size.of(128, 1, 1));
    e.barrier();
    e.setPipeline(pipes.pick);
    e.setBuffer(scratch.buf, scratch.off, 0);
    e.setValue([3]u32{ @intCast((rows + 15) / 16), 8, @intCast((p.len + rows + 4095) / 4096) }, 1);
    e.setBuffer(scratch.buf, scratch.off + 16, 2);
    e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(1, 1, 1));
    e.barrier();
    for ([_]mtl.Pipeline{ original, pipes.masked }, [_]usize{ 16, 28 }) |pipe, offset| {
        e.setPipeline(pipe);
        for (inputs, 0..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(p, 3);
        for (partials, 4..) |r, i| e.setBuffer(r.buf, r.off, i);
        e.setValue(@as(u32, @intCast(rows)), 7);
        e.dispatchIndirect(scratch.buf, scratch.off + offset, mtl.Size.of(128, 1, 1));
    }
}
