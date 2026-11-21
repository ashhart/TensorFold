//! The single-reply host offers only landed prompt state to the same cancellation hook as lane backends.
const std = @import("std");
const lanes = @import("tensorfold").lanes;
const Mark = *const fn (*anyopaque, usize) void;

const Bridge = struct {
    ptr: *anyopaque,
    mark: Mark,
    fn kept(ptr: *anyopaque, _: *lanes.Stream, at: u32) void {
        const b: *Bridge = @ptrCast(@alignCast(ptr));
        b.mark(b.ptr, at);
    }
};

pub fn keep(gpa: std.mem.Allocator, prefilling: bool, at: u32, cached: u32, prompt_len: usize, ptr: *anyopaque, mark: Mark) void {
    if (!prefilling) return;
    var bridge: Bridge = .{ .ptr = ptr, .mark = mark };
    var stream = lanes.Stream.init(gpa, .{ .id = "cancelled-prompt", .prompt = &.{}, .max_new = 0, .reuse = .{ .hook = .{ .ptr = &bridge, .at = Bridge.kept } } }) catch return;
    defer stream.deinit(gpa);
    stream.cached = cached;
    stream.prompt_len = prompt_len;
    stream.keepInterruptedPrefill(at);
}

test "Flash Next cancellation: only a newly landed prompt prefix reaches the existing keep callback" {
    const Heard = struct {
        at: ?usize = null,
        fn mark(ptr: *anyopaque, at: usize) void {
            const h: *@This() = @ptrCast(@alignCast(ptr));
            h.at = at;
        }
    };
    const Case = struct { prefilling: bool = true, at: u32, cached: u32 = 0, want: ?usize = null };
    for ([_]Case{
        .{ .at = 0 },                      .{ .at = 4, .want = 4 },            .{ .at = 8, .want = 8 },
        .{ .at = 4, .cached = 4 },         .{ .at = 10 },                      .{ .at = 11 },
        .{ .prefilling = false, .at = 4 }, .{ .prefilling = false, .at = 11 },
    }) |case| {
        var heard: Heard = .{};
        keep(std.testing.allocator, case.prefilling, case.at, case.cached, 10, &heard, Heard.mark);
        try std.testing.expectEqual(case.want, heard.at);
    }
}
