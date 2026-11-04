const flash = @import("flashnext");

export fn exl3PrefillForwardBody(engine: *flash.prompt.Engine, sequence: *flash.prompt.Seq, launch: *const flash.Tri, ids: [*]const u32, count: usize) void {
    engine.forward(flash.Tri, launch.*, sequence, ids[0..count], .{ .logits = .last }) catch {};
}
