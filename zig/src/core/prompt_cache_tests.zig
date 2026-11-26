//! The prompt-cache store's complete host test suite.
test {
    _ = @import("prompt_cache_report_test.zig");
    _ = @import("prompt_cache_review_test.zig");
    _ = @import("prompt_cache_promised_test.zig");
    _ = @import("prompt_cache_queue_room_test.zig");
    _ = @import("prompt_cache_index_test.zig");
    _ = @import("prompt_cache_test.zig");
    _ = @import("prompt_cache_learn_test.zig");
    _ = @import("prompt_cache_rewind_test.zig");
    _ = @import("prompt_cache_interior_test.zig");
    _ = @import("prompt_cache_intermediate_test.zig");
    _ = @import("prompt_cache_branch_test.zig");
    _ = @import("prompt_cache_newest_test.zig");
    _ = @import("prompt_cache_fork_test.zig");
    _ = @import("prompt_cache_give_test.zig");
    _ = @import("prompt_cache_lifetime_test.zig");
    _ = @import("prompt_cache_scan_test.zig");
    _ = @import("learned_fault_test.zig");
    _ = @import("cache_modes.zig");
    _ = @import("rows_store.zig");
    _ = @import("prompt_cache_repeat_test.zig");
    _ = @import("prompt_cache_tail_test.zig");
    _ = @import("prompt_cache_others_test.zig");
    _ = @import("prompt_cache_miss_test.zig");
    _ = @import("prompt_cache_message_starts_test.zig");
    _ = @import("prompt_cache_why_test.zig");
}
