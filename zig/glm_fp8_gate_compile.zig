//! Force compilation of the real synthetic device gate without running or building GPU code.
comptime {
    _ = &@import("gate").run;
}
