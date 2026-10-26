test {
    _ = @import("src/families/flashnext/backend.zig");
}

test {
    _ = @import("src/families/flashnext/grammar_host_test.zig");
}

test {
    _ = @import("src/families/flashnext/vision_contract.zig");
}

test {
    _ = @import("src/families/flashnext/window_config_test.zig");
}

test {
    _ = @import("flashnext_rotary_bounds_tests.zig");
    _ = @import("contract_bounds_host_test.zig");
}

test {
    _ = @import("flashnext_precise_rotary_test.zig");
    _ = @import("rotary_sweep_tests.zig");
    _ = @import("rotary_receipt_host_test.zig");
    _ = @import("src/families/flashnext/rotary_cache_test.zig");
}

test "Flash Next memory: construction and preflight consume the same prompt sizes" {
    const std = @import("std");
    const mtl = @import("metal");
    const fz = @import("src/families/flashnext/replay.zig");
    const Counter = struct {
        total: u64 = 0,
        pub fn buffer(c: *@This(), n: usize) error{}!mtl.Buffer {
            c.total += @max(n, 64);
            return undefined;
        }
    };
    var counter: Counter = .{};
    _ = try fz.Prompt.buffers(&counter);
    try std.testing.expectEqual(@import("flashnext_memory").prompt.bytes(), counter.total);
    try std.testing.expectEqual(@as(usize, (fz.PLE_TAIL + fz.PMAX) * fz.WIDE * 2), @import("flashnext_memory").prompt.sizes.cin);
}

test {
    _ = @import("flashnext_memory_checkpoint_test.zig");
}

test {
    _ = @import("src/families/flashnext/load_weights_test.zig");
}

test {
    _ = @import("src/families/flashnext/vis_prompt.zig");
}

test {
    _ = @import("flashnext_fsr_tests.zig");
}
