const std = @import("std");
test {
    _ = @import("kv_io_test.zig");
    inline for (.{ @import("forward.zig").Forward, @import("run.zig").Runner, @import("graphs.zig").Statics, @import("graphs.zig").EngramStage, @import("cuda_kv.zig").Kv, @import("forward_prefill.zig"), @import("longpf.zig"), @import("ced.zig"), @import("sessions_gpu.zig").Sessions, @import("slots.zig").SlotSet, @import("engine_hooks.zig") }) |m| std.testing.refAllDecls(m);
    _ = @import("kv_state.zig");
    _ = @import("ced_test.zig");
    _ = @import("longpf_test.zig");
    _ = @import("mhc_pfdec_test.zig");
    _ = @import("kv_glue_emit_test.zig");
    _ = @import("pf4k_test.zig");
    _ = @import("pftbo_test.zig");
    _ = @import("pfovl_test.zig");
    _ = @import("streamrb_test.zig");
}
fn force(comptime T: type) void {
    inline for (comptime std.meta.declarations(T)) |name| {
        const value = @field(T, name);
        if (@typeInfo(@TypeOf(value)) == .@"fn" and !@typeInfo(@TypeOf(value)).@"fn".is_generic)
            std.mem.doNotOptimizeAway(&value);
    }
}
test "Snapshot state: emitted Linux bodies, never run on this host" {
    if (@import("builtin").os.tag == .linux) {
        inline for (.{ @import("forward.zig").Forward, @import("run.zig").Runner, @import("graphs.zig").Statics, @import("graphs.zig").EngramStage, @import("cuda_kv.zig").Kv, @import("cuda_transaction.zig").Transaction, @import("forward_prefill.zig"), @import("longpf.zig"), @import("sessions_gpu.zig").Sessions, @import("slots.zig").SlotSet, @import("engine_hooks.zig") }) |m| force(m);
    }
}

test {
    _ = @import("review_workspace_test.zig");
}

test {
    _ = @import("review_session_test.zig");
}

test {
    _ = @import("review_copy_test.zig");
}
