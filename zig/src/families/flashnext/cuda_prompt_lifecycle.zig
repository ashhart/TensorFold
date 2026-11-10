const std = @import("std");
const prompt = @import("cuda_prompt.zig");

pub fn deinit(sequence: *prompt.Seq, allocator: std.mem.Allocator) void {
    if (sequence.stream) |stream| @import("cuda").completion.drain(stream.d, stream, "Flash Next sequence work did not drain before release");
    if (sequence.media) |*media| media.deinit();
    for ([_][]@import("cuda").DeviceBuffer{ sequence.conv, sequence.state, sequence.kc, sequence.vc, sequence.ikc, sequence.pooled }) |buffers| {
        for (buffers) |*buffer| buffer.free();
        allocator.free(buffers);
    }
    if (sequence.tail.ptr != 0) sequence.tail.free();
    if (sequence.paged) |*table| table.deinit(allocator);
    allocator.free(sequence.logits);
}
