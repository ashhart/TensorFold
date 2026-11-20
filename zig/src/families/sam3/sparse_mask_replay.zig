const std = @import("std");
const gpu = @import("tensor_metal");
const st = @import("safetensors");

pub const Probe = struct {
    input: gpu.Tensor,
    weight: gpu.Tensor,
    conditioning: u32,

    pub fn init(g: *gpu.Gpu, file: *const st.File) !Probe {
        const input = file.get("source_input") orelse return error.MissingMaskInput;
        const weight = file.get("source_weight") orelse return error.MissingMaskWeight;
        if (!input.is(.f32, &.{ 1, 32, 1152, 1152 }) or !weight.is(.f32, &.{ 16, 32, 3, 3 })) return error.SparseMaskCaptureShape;
        const pixels = 1152 * 1152;
        const conditioning = std.mem.readInt(u32, input.bytes[16 * pixels * 4 ..][0..4], .little);
        if (conditioning != 0 and conditioning != 0x3f800000) return error.SparseMaskConditioning;
        for (1..32) |channel| for (0..pixels) |pixel| {
            const value = std.mem.readInt(u32, input.bytes[(channel * pixels + pixel) * 4 ..][0..4], .little);
            if (value != (if (channel == 16) conditioning else @as(u32, 0))) return error.DenseMaskCapture;
        };
        const x = try g.alloc(pixels, .f32);
        errdefer x.deinit();
        @memcpy(x.slice(u8, pixels * 4), input.bytes[0 .. pixels * 4]);
        const w = try g.alloc(weight.numel(), .f32);
        errdefer w.deinit();
        @memcpy(w.slice(u8, weight.bytes.len), weight.bytes);
        try x.checkFinite("sparse mask input");
        try w.checkFinite("sparse mask weights");
        return .{ .input = x, .weight = w, .conditioning = @intFromBool(conditioning != 0) };
    }

    pub fn deinit(self: Probe) void {
        self.input.deinit();
        self.weight.deinit();
    }

    pub fn run(self: Probe, g: *gpu.Gpu, bias: gpu.Tensor) !gpu.Tensor {
        const output = try g.allocDevice(576 * 576 * 16, .f32);
        errdefer output.deinit();
        try g.dispatch("im_sam_mask_downsample", &.{ self.input, self.weight, bias, output }, self.conditioning, gpu.mtl.Size.of(output.count, 1, 1), gpu.mtl.Size.of(256, 1, 1));
        return output;
    }
};
