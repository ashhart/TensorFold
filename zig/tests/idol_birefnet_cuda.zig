//! tf-idol-birefnet on CUDA: idol_birefnet_run.zig's driver on the first GPU (sm_120/sm_121, others refused).
const std = @import("std");
const cuda = @import("cuda");
const br = @import("idol_birefnet");
const run_ = @import("idol_birefnet_run.zig");

const Cuda = struct {
    pub const st = br.safetensors;
    pub const pre = br.preprocess;
    pub const gt = br.reference;
    pub const cfg = br.config;
    pub const seconds = br.seconds;
    pub const Net = br.Net;

    /// The host's copy of a slot the stages read (valid until the next read).
    var mirror: std.ArrayList(f32) = .empty;

    fn at(n: *Net, slot: run_.Slot) u64 {
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

    pub fn read(n: *Net, slot: run_.Slot, len: usize) ![]f32 {
        try mirror.resize(n.gpa, len);
        try n.download(mirror.items, at(n, slot));
        return mirror.items;
    }

    pub fn write(n: *Net, slot: run_.Slot, data: []const f32) !void {
        try n.upload(at(n, slot), data);
    }

    pub fn stage(n: *Net, step: run_.Step) !void {
        switch (step) {
            .layer => |l| try n.layer(l[0], l[1], n.ta, n.tb),
            .encode => try n.encode(),
            .squeeze => try br.decoder.squeeze(n),
            .block => |l| try br.decoder.block(n, &n.w.dec[l], n.lin[l], cfg.level_in[l], cfg.level_side[l], n.out[l], cfg.level_out[l], 0),
            .decode => try br.decoder.decode(n),
        }
        try n.stream.synchronize();
    }

    pub const Device = struct {
        driver: cuda.Driver,
        ctx: cuda.Context,
        buf: [256]u8,

        pub fn init(g: *Device) !void {
            g.driver = try cuda.Driver.open();
            errdefer g.driver.close();
            g.ctx = try cuda.Context.init(&g.driver, 0);
        }

        pub fn deinit(g: *Device) void {
            mirror.deinit(std.heap.page_allocator);
            g.ctx.deinit();
            g.driver.close();
        }

        pub fn name(g: *Device) []const u8 {
            return g.ctx.name(&g.buf) catch "a CUDA GPU";
        }

        pub fn load(g: *Device, gpa: std.mem.Allocator, io: std.Io, path: [:0]const u8) !*Net {
            return Net.load(gpa, io, &g.ctx, path);
        }

        /// The device bytes the net holds and the context's use at the end, printed and recorded (CUDA alone).
        pub fn footprint(g: *Device, n: *Net, out: *std.ArrayList(u8), a: std.mem.Allocator) !void {
            const mi = try g.ctx.memInfo();
            const used = mi.total - mi.free;
            std.debug.print("device bytes: BiRefNet holds {d:.2} GB (weights {d:.2}); the context uses {d:.2} GB\n", .{ gb(n.deviceBytes()), gb(n.w.buf.len), gb(used) });
            try out.appendSlice(a, try std.fmt.allocPrint(a, ", \"device_bytes\": {d}, \"weight_bytes\": {d}, \"context_used_bytes\": {d}", .{ n.deviceBytes(), n.w.buf.len, used }));
        }

        fn gb(v: usize) f64 {
            return @as(f64, @floatFromInt(v)) / 1e9;
        }
    };
};

pub fn main(init: std.process.Init) !u8 {
    return run_.run(Cuda, init);
}
