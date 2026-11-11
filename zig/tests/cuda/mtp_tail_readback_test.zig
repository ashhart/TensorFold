//! Exercise the production tail, including its completion fence and integrated-memory staging.
const cuda = @import("cuda");
const tail = @import("tail");
const Fake = @import("staged_copy_fake.zig").Fake;

var completed = false;
var original_copy: *const fn (?*anyopaque, u64, usize) callconv(.c) cuda.abi.Result = undefined;

fn complete(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
    completed = true;
    return 0;
}

fn copy(dst: ?*anyopaque, src: u64, n: usize) callconv(.c) cuda.abi.Result {
    if (!completed) return cuda.abi.error_not_ready;
    return original_copy(dst, src, n);
}

test "MTP tail readback waits before copying and preserves bounded integrated/discrete output" {
    for ([_]bool{ true, false }) |integrated| {
        var rig = Fake.init(integrated);
        defer rig.deinit();
        completed = false;
        original_copy = rig.d.api.cuMemcpyDtoH_v2;
        rig.d.api.cuMemcpyDtoH_v2 = copy;
        rig.d.api.cuStreamSynchronize = complete;
        const out = rig.dest(u8, 83);
        @memset(out, 0xaa);
        try tail.download(cuda.Stream.borrowed(&rig.d, null), out, rig.src);
        try rig.check(rig.src, out);
    }
}
