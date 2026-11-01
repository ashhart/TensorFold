//! Row-store host regressions, kept separate from the storage owner.

test {
    _ = @import("rows_image.zig");
    _ = @import("rows_chain.zig");
    _ = @import("rows_file.zig");
    _ = @import("rows_write.zig").io;
    _ = @import("rows_test.zig");
    _ = @import("rows_spill_test.zig");
    _ = @import("rows_async_test.zig");
    _ = @import("rows_close_test.zig");
    _ = @import("rows_stage_test.zig");
    _ = @import("rows_select_test.zig");
    _ = @import("rows_recall_select_test.zig");
    _ = @import("rows_charge_test.zig");
    _ = @import("rows_pool_select_test.zig");
    _ = @import("rows_prefetch.zig");
    _ = @import("rows_prefetch_stack_test.zig");
    _ = @import("rows_refusal_test.zig");
}
