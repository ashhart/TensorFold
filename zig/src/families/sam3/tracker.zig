// SAM tracking architecture adapted from facebookresearch/sam3 under LICENSES/SAM.txt.
const std = @import("std");
const ops = @import("ops.zig");
const vision = @import("vision.zig");
const decoder = @import("decoder.zig");
const memory = @import("memory.zig");
const prompt_encoder = @import("prompt_encoder.zig");
const prompts = @import("prompts.zig");
const Runtime = ops.Runtime;
const Tensor = ops.Tensor;
const Feature = ops.Feature;
const view = ops.view;
pub const Result = decoder.Result;
pub const Memory = memory.Memory;

pub fn seed(r: *Runtime, features: [3]Feature, prompt: prompts.Prompt, size: prompts.Size) !Result {
    const start = r.mark();
    const encoded = try prompt_encoder.encode(r, prompt, size);
    const key = try r.name("{s}.{s}", .{ r.config.trackerPrefix(), if (r.config.version == .sam31) "interactivity_no_mem_embed" else "no_mem_embed" });
    const image = try r.add(features[2].tensor, try r.weight(key));
    var result = try decoder.decode(r, features, image, .{
        .dense = encoded.dense,
        .sparse = encoded.sparse,
        .multimask = prompt.points.len == 1 and prompt.boxes.len == 0 and prompt.mask == null,
    });
    if (prompt.mask) |mask| {
        if (prompt.points.len == 0 and prompt.boxes.len == 0 and mask.format == .probabilities) {
            const logits = try r.alloc(mask.values.len);
            for (mask.values, logits.slice(f32, logits.count)) |value, *out| out.* = value * 20 - 10;
            result.input_masks = ops.feature(logits, mask.size.height, mask.size.width, 1);
            result.masks = try r.resize(result.input_masks.?, 288, 288, true);
            result.scores[0] = if (std.mem.max(f32, mask.values) > 0) 10 else -10;
            result.ious[0] = 1;
            if (result.scores[0] <= 0) result.pointer = if (r.config.version == .sam31)
                try r.linear(try r.name("{s}.no_obj_ptr_linear", .{r.config.trackerPrefix()}), result.pointer, 256)
            else
                try r.weight(try r.name("{s}.no_obj_ptr", .{r.config.trackerPrefix()}));
        }
    }
    var retained: [3]Tensor = undefined;
    r.trim(start, result.retained(&retained));
    return result;
}

pub fn distance(a: usize, b: usize) usize {
    return @max(a, b) - @min(a, b);
}

fn spatial(past: Memory, index: usize) bool {
    return past.spatial_live and (past.conditioning or distance(index, past.index) < 7);
}

fn temporalIndex(multiplex: bool, conditioning: bool, delta: usize) usize {
    return if ((!multiplex and conditioning) or delta == 0 or delta >= 7) 6 else 6 - delta;
}

test "SAM3.1 temporally encodes conditioning frames while SAM3 uses its conditioning indicator" {
    try std.testing.expectEqual(@as(usize, 6), temporalIndex(false, true, 1));
    try std.testing.expectEqual(@as(usize, 5), temporalIndex(true, true, 1));
    try std.testing.expectEqual(@as(usize, 0), temporalIndex(true, true, 6));
    try std.testing.expectEqual(@as(usize, 6), temporalIndex(true, true, 7));
    try std.testing.expectEqual(@as(usize, 4), temporalIndex(false, false, 2));
}

fn pointerPosition(r: *Runtime, max_pointers: usize, delta: usize) !Tensor {
    const key = try r.name("pointer.positions.{d}.{d}", .{ max_pointers, delta });
    if (delta < 16) if (r.constant(key)) |cached| return cached;
    const pe = try r.alloc(256);
    for (0..128) |c| {
        const angle = @as(f32, @floatFromInt(delta)) / @as(f32, @floatFromInt(@max(1, max_pointers - 1))) /
            std.math.pow(f32, 10000, @as(f32, @floatFromInt(c / 2 * 2)) / 128);
        pe.slice(f32, 256)[c] = @sin(angle);
        pe.slice(f32, 256)[c + 128] = @cos(angle);
    }
    const projected = try r.linear(try r.name("{s}.obj_ptr_tpos_proj", .{r.config.trackerPrefix()}), pe, 256);
    const repeated = try r.alloc(r.config.bucket_size * 256);
    const width = r.config.memory_width;
    for (0..repeated.count / width) |slot|
        @memcpy(repeated.slice(f32, repeated.count)[slot * width ..][0..width], projected.slice(f32, width));
    return if (delta < 16) r.remember(key, repeated) else repeated;
}

fn oldAttention(r: *Runtime, key: []const u8, q: Tensor, k: Tensor, v: Tensor, input_width: usize, spatial_keys: usize) !Tensor {
    const qp = try r.linear(try r.name("{s}.q_proj", .{key}), q, 256);
    const kp = try r.linear(try r.name("{s}.k_proj", .{key}), k, input_width);
    const vp = try r.linear(try r.name("{s}.v_proj", .{key}), v, input_width);
    const attended = try r.attend(try r.rope(qp, 1, 256, 72, false, 5184), try r.rope(kp, 1, 256, 72, false, spatial_keys), vp, 1, 256, null, false);
    return r.linear(try r.name("{s}.out_proj", .{key}), attended, 256);
}

pub fn propagate(r: *Runtime, features: [3]Feature, history: []const Memory, index: usize, frames: usize, active: u16) !Result {
    if (history.len == 0) return error.MissingConditioningFrame;
    const start = r.mark();
    const root = r.config.trackerPrefix();
    const multiplex = r.config.version == .sam31;
    const width = r.config.memory_width;
    const pos = try vision.sine(r, 72, 72);
    const memory_pe = try vision.sineWidth(r, 72, 72, width);
    const temporal = try r.weight(try r.name("{s}.maskmem_tpos_enc", .{root}));
    const a = r.a;
    var pointers: std.ArrayList(Tensor) = .empty;
    var pointer_positions: std.ArrayList(Tensor) = .empty;
    const max_pointers = @min(frames, 16);
    for (history) |past| {
        const delta = distance(index, past.index);
        if (!past.conditioning and delta >= max_pointers) continue;
        try pointers.append(a, past.pointer);
        try pointer_positions.append(a, try pointerPosition(r, max_pointers, delta));
    }
    const pointer_count = pointers.items.len * r.config.bucket_size * 256 / width;
    const zero = try r.alloc(pointer_count * 256);
    @memset(zero.slice(f32, zero.count), 0);
    try r.g.beginBatch();
    errdefer if (r.g.batch != null) r.g.endBatch() catch {};
    var memories: std.ArrayList(Tensor) = .empty;
    var images: std.ArrayList(Tensor) = .empty;
    var positions: std.ArrayList(Tensor) = .empty;
    for (history) |past| {
        if (!spatial(past, index)) continue;
        const delta = distance(index, past.index);
        const t = temporalIndex(multiplex, past.conditioning, delta);
        try memories.append(a, past.encoded);
        try images.append(a, past.image);
        try positions.append(a, try r.add(memory_pe, view(temporal, t * width, width)));
    }
    if (memories.items.len == 0) return error.MissingSpatialMemory;
    const spatial_count = memories.items.len * 5184;
    try memories.appendSlice(a, pointers.items);
    try positions.appendSlice(a, pointer_positions.items);
    try images.append(a, zero);
    const cached = multiplex and r.precision == .mixed;
    const encoded = try r.concat(if (cached) pointers.items else memories.items);
    const image_memory = if (cached or !multiplex) zero else try r.concat(images.items);
    const memory_pos = try r.concat(positions.items);
    const image = features[2].tensor;
    var h = try r.add(image, try r.scale(pos, 0.1));
    for (0..4) |i| {
        const mark = r.mark();
        const layer = try r.name("{s}.transformer.encoder.layers.{d}", .{ root, i });
        const norm = try r.norm(try r.name("{s}.norm1", .{layer}), h, 256, 1e-5);
        var next: Tensor = undefined;
        if (!multiplex) {
            next = try r.add(h, try oldAttention(r, try r.name("{s}.self_attn", .{layer}), norm, norm, norm, 256, 5184));
            const cross_norm = try r.norm(try r.name("{s}.norm2", .{layer}), next, 256, 1e-5);
            next = try r.add(next, try oldAttention(r, try r.name("{s}.cross_attn_image", .{layer}), cross_norm, try r.add(encoded, memory_pos), encoded, width, spatial_count));
            const up = try r.unary(try r.linear(try r.name("{s}.linear1", .{layer}), try r.norm(try r.name("{s}.norm3", .{layer}), next, 256, 1e-5), 256), 0);
            h = try r.add(next, try r.linear(try r.name("{s}.linear2", .{layer}), up, 2048));
        } else {
            const q = try r.rope(try r.linear(try r.name("{s}.self_attn_q_proj", .{layer}), norm, 256), 8, 256, 72, false, 5184);
            const k = try r.rope(try r.linear(try r.name("{s}.self_attn_k_proj", .{layer}), norm, 256), 8, 256, 72, false, 5184);
            const v = try r.linear(try r.name("{s}.self_attn_v_proj", .{layer}), norm, 256);
            next = try r.add(h, try r.linear(try r.name("{s}.self_attn_out_proj", .{layer}), try r.attend(q, k, v, 8, 256, null, false), 256));
            const n = try r.norm(try r.name("{s}.norm2", .{layer}), next, 256, 1e-5);
            const cq = try r.add(try r.linear(try r.name("{s}.image_cross_attn_q_proj", .{layer}), image, 256), try r.linear(try r.name("{s}.cross_attn_q_proj", .{layer}), n, 256));
            var ck: Tensor = undefined;
            var cv: Tensor = undefined;
            if (cached) {
                var keys: std.ArrayList(Tensor) = .empty;
                var values: std.ArrayList(Tensor) = .empty;
                for (history) |past| {
                    if (!spatial(past, index)) continue;
                    try keys.append(a, past.projected.?[i].key);
                    try values.append(a, past.projected.?[i].value);
                }
                try keys.append(a, try r.add(try r.linear(try r.name("{s}.cross_attn_k_proj", .{layer}), encoded, 256), try r.weight(try r.name("{s}.image_cross_attn_k_proj.bias", .{layer}))));
                try values.append(a, try r.linear(try r.name("{s}.cross_attn_v_proj", .{layer}), encoded, 256));
                ck = try r.add(try r.concat(keys.items), memory_pos);
                cv = try r.concat(values.items);
            } else {
                ck = try r.add(try r.add(try r.linear(try r.name("{s}.image_cross_attn_k_proj", .{layer}), image_memory, 256), try r.linear(try r.name("{s}.cross_attn_k_proj", .{layer}), encoded, 256)), memory_pos);
                cv = try r.linear(try r.name("{s}.cross_attn_v_proj", .{layer}), encoded, 256);
            }
            const cross = try r.attend(try r.rope(cq, 8, 256, 72, false, 5184), try r.rope(ck, 8, 256, 72, false, spatial_count), cv, 8, 256, null, false);
            next = try r.add(next, try r.linear(try r.name("{s}.cross_attn_out_proj", .{layer}), cross, 256));
            h = try r.add(next, try vision.feedForward(r, layer, try r.norm(try r.name("{s}.norm3", .{layer}), next, 256, 1e-5)));
        }
        r.trim(mark, &.{h});
    }
    h = try r.norm(try r.name("{s}.transformer.encoder.norm", .{root}), h, 256, 1e-5);
    try r.g.endBatch();
    const out = try decoder.decode(r, features, h, .{ .multiplex = multiplex, .active = active });
    r.trim(start, &.{ out.masks.tensor, out.pointer });
    return out;
}
