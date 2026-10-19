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
}
