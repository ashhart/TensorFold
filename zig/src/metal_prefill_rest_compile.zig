//! Compile actual Metal prefill callback bodies; this object is never executed.
const tf = @import("tensorfold");
const lanes = @import("lanes");

export fn gemma4PrefillCompile(b: *tf.gemma4.backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn lfm2PrefillCompile(b: *tf.lfm2_moe.backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn step5PrefillCompile(b: *tf.step5.backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn sparkPrefillCompile(b: *tf.spark25_metal.backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn minimaxPrefillCompile(b: *tf.minimax_m3.backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn llamaPrefillCompile(b: *tf.llama.backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn qwen35PrefillCompile(b: *tf.qwen35.backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn dsaPrefillCompile(b: *tf.glm_dsa.backend.Backend, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}

export fn kimiPrefillCompile(b: *@import("kimi").backend.Metal, s: *lanes.Stream) bool {
    b.backend().prefill(s) catch return false;
    return true;
}
