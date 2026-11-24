//! An inert backend for testing the actual MOSS-Image case driver without any device or checkpoint.
const std = @import("std");
const Prompt = struct {
    pub fn deinit(_: *Prompt) void {}
};
const Timing = struct { wall: f64 = 0, gpu: f64 = 0, host: f64 = 0 };
const Model = struct {
    a: std.mem.Allocator,
    kind: enum { normal, element, negative, repeat, nonfinite },
    calls: usize = 0,

    pub fn deinit(m: *Model) void {
        m.a.destroy(m);
    }
    pub fn prompt(_: *Model, _: []const u16) !Prompt {
        return .{};
    }
    pub fn forward(m: *Model, _: *Prompt, input: []const f32, timestep: f32, skip: ?usize, out: []f32) !Timing {
        for (out, input) |*o, x| {
            const bits: u32 = @bitCast(x);
            o.* = @bitCast((bits +% 0x7fff +% ((bits >> 16) & 1)) & 0xffff0000);
        }
        if (timestep == 600 and ((m.kind == .element and skip == null) or (m.kind == .negative and skip != null))) out[3] += 0.125;
        if (m.kind == .nonfinite and timestep == 600) out[2] = std.math.nan(f32);
        if (m.kind == .repeat and m.calls == 8) out[0] += 0.125;
        m.calls += 1;
        return .{};
    }
};
const Backend = struct {
    pub const st = @import("safetensors");
    pub const mi = struct {
        pub const config = struct {
            pub const channels = 4;
            pub const context = 2;
        };
        pub const Model = @import("moss_image_run_host.zig").Model;
        pub const Prompt = @import("moss_image_run_host.zig").Prompt;
        pub const noise = struct {
            /// The driver's --seed path without a torch stream: a fixed ramp, so a seeded run still runs.
            pub fn latents(_: std.mem.Allocator, _: u64, _: usize, _: usize, out: []f32) !void {
                for (out, 0..) |*v, i| v.* = @floatFromInt(i);
            }
        };
    };
    pub fn seconds() f64 {
        return 0;
    }
    pub const Device = struct {
        pub fn init(_: *Device) !void {}
        pub fn deinit(_: *Device) void {}
        pub fn name(_: *Device) []const u8 {
            return "inert teacher fixture";
        }
        pub fn load(_: *Device, a: std.mem.Allocator, _: std.Io, dir: []const u8, _: usize, _: usize, _: usize) !*Model {
            const m = try a.create(Model);
            m.* = .{ .a = a, .kind = if (std.mem.eql(u8, dir, "element-fault")) .element else if (std.mem.eql(u8, dir, "negative-fault")) .negative else if (std.mem.eql(u8, dir, "repeat-fault")) .repeat else if (std.mem.eql(u8, dir, "nonfinite-fault")) .nonfinite else .normal };
            return m;
        }
    };
};
pub fn main(init: std.process.Init) !u8 {
    return @import("moss_image_run.zig").run(Backend, init);
}
