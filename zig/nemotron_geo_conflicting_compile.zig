const source = "template <int K, int GS, int RPS>\n[[max_total_threads_per_threadgroup(64)]]\n[[max_total_threads_per_threadgroup(SG * 32)]]\n[[kernel]] void sample() { tf_rowdot;  for (int k0 = 0; k0 < FULL; k0 += 512) { }\n}\n\ntemplate [[host_name(\"sample\")]] [[kernel]] decltype(sample) sample;";
comptime {
    _ = @import("src/families/nemotron/geo_source.zig").body(source, "sample");
}
