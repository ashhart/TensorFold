//! tf-idol-birefnet on Metal: idol_birefnet_run.zig's case driver on the system GPU.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const run_ = @import("idol_birefnet_run.zig");
const bn = tf.idol.birefnet;
const dec = tf.idol.birefnet_decoder;

const Metal = struct {
    pub const st = tf.safetensors;
    pub const pre = tf.idol.preprocess;
    pub const gt = tf.idol.birefnet_reference;
    pub const cfg = tf.idol.birefnet_config;
    pub const seconds = mtl.clock.seconds;
    pub const Net = bn.Net;

    fn ref(n: *Net, slot: run_.Slot) tf.idol.birefnet_kernels.Ref {
        return switch (slot) {
            .ta => n.ta,
            .tb => n.tb,
            .x => n.x,
            .xs => |i| n.xs[i],
            .lin => |l| n.lin[l],
            .out => |l| n.out[l],
            .logits => n.logits,
        };
    }

    /// The slot's first `len` floats in place: the scratch buffer is shared memory.
    pub fn read(n: *Net, slot: run_.Slot, len: usize) ![]f32 {
        return Net.floats(ref(n, slot), len);
    }

    pub fn write(n: *Net, slot: run_.Slot, data: []const f32) !void {
        const dst = Net.floats(ref(n, slot), data.len);
        if (dst.ptr != data.ptr) @memcpy(dst, data);
    }

    pub fn stage(n: *Net, step: run_.Step) !void {
        const cb = n.queue.commandBuffer();
        const e = cb.compute(.serial);
        switch (step) {
            .layer => |at| n.layer(e, at[0], at[1], n.ta, n.tb),
            .encode => n.encode(e),
            .squeeze => dec.squeeze(n, e),
            .block => |l| dec.block(n, e, &n.w.dec[l], n.lin[l], cfg.level_in[l], cfg.level_side[l], n.out[l], cfg.level_out[l], 0),
            .decode => dec.decode(n, e),
        }
        _ = try Net.finish(cb, e);
    }

    pub const Device = struct {
        pool: mtl.objc.Pool,
        device: mtl.Device,

        pub fn init(g: *Device) !void {
            g.pool = mtl.objc.Pool.push();
            errdefer g.pool.pop();
            g.device = try mtl.Device.init();
        }

        pub fn deinit(g: *Device) void {
            g.pool.pop();
        }

        pub fn name(g: *Device) []const u8 {
            return std.mem.span(g.device.name());
        }

        pub fn load(g: *Device, gpa: std.mem.Allocator, _: std.Io, path: [:0]const u8) !*Net {
            return Net.load(gpa, g.device, path);
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return run_.run(Metal, init);
}
