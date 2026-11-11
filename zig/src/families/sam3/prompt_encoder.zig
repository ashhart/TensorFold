// SAM prompt embeddings adapted from facebookresearch/sam3 under LICENSES/SAM.txt.
const std = @import("std");
const ops = @import("ops.zig");
const vision = @import("vision.zig");
const prompts = @import("prompts.zig");
const Runtime = ops.Runtime;
const Tensor = ops.Tensor;

pub const Encoded = struct { sparse: ?Tensor = null, dense: ?Tensor = null };

pub fn encode(r: *Runtime, prompt: prompts.Prompt, size: prompts.Size) !Encoded {
    try prompt.validate(size);
    const prefix = r.config.promptPrefix();
    var out = Encoded{};
    const start = r.mark();
    const padding: usize = if (prompt.boxes.len == 0 and prompt.points.len != 0) 1 else 0;
    const count = try std.math.add(usize, try std.math.add(usize, prompt.points.len, try std.math.mul(usize, prompt.boxes.len, 2)), padding);
    if (count != 0) {
        const coords = try r.a.alloc([2]f32, count);
        const labels = try r.a.alloc(i8, count);
        const transform = prompts.Transform{ .source = size };
        for (prompt.points, 0..) |point, i| {
            const p = try transform.point(point);
            coords[i] = .{ (p[0] + 0.5) / 1008, (p[1] + 0.5) / 1008 };
            labels[i] = if (point.positive) 1 else 0;
        }
        for (prompt.boxes, 0..) |box, i| {
            if (!box.positive) return error.NegativeBoxNeedsConceptPrompt;
            const b = try transform.box(box);
            const at = prompt.points.len + i * 2;
            coords[at] = .{ (b[0] + 0.5) / 1008, (b[1] + 0.5) / 1008 };
            coords[at + 1] = .{ (b[2] + 0.5) / 1008, (b[3] + 0.5) / 1008 };
            labels[at] = 2;
            labels[at + 1] = 3;
        }
        if (padding != 0) {
            coords[count - 1] = .{ 0, 0 };
            labels[count - 1] = -1;
        }
        const encoded = try vision.fourier(r, try r.name("{s}.pe_layer.positional_encoding_gaussian_matrix", .{prefix}), coords);
        const sparse = try r.alloc(count * 256);
        for (labels, 0..) |label, i| {
            const embedding = try r.weight(if (label < 0)
                try r.name("{s}.not_a_point_embed.weight", .{prefix})
            else
                try r.name("{s}.point_embeddings.{d}.weight", .{ prefix, label }));
            for (0..256) |c| sparse.slice(f32, sparse.count)[i * 256 + c] =
                (if (label < 0) @as(f32, 0) else encoded.get(i * 256 + c)) + embedding.get(c);
        }
        out.sparse = sparse;
    }
    if (prompt.mask) |mask| {
        var f = ops.feature(try r.upload(mask.values), mask.size.height, mask.size.width, 1);
        if (mask.format == .probabilities) {
            f = try r.resize(f, 1008, 1008, true);
            f = try r.conv(try r.name("{s}.{s}", .{ r.config.trackerPrefix(), if (r.config.version == .sam31) "interactive_mask_downsample" else "mask_downsample" }), f, 4, 0, false, false);
        }
        f = try r.resize(f, 288, 288, true);
        const key = try r.name("{s}.mask_downscaling", .{prefix});
        f = try r.conv(try r.name("{s}.0", .{key}), f, 2, 0, false, false);
        f.tensor = try r.gelu(try r.norm(try r.name("{s}.1", .{key}), f.tensor, 4, 1e-6));
        f = try r.conv(try r.name("{s}.3", .{key}), f, 2, 0, false, false);
        f.tensor = try r.gelu(try r.norm(try r.name("{s}.4", .{key}), f.tensor, 16, 1e-6));
        f = try r.conv(try r.name("{s}.6", .{key}), f, 1, 0, false, false);
        out.dense = f.tensor;
    }
    var keep: [2]Tensor = undefined;
    var n: usize = 0;
    if (out.sparse) |tensor| {
        keep[n] = tensor;
        n += 1;
    }
    if (out.dense) |tensor| {
        keep[n] = tensor;
        n += 1;
    }
    r.trim(start, keep[0..n]);
    return out;
}
