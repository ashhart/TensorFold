//! The shape of one serve flag: its name, how it takes its value and what this binary honours.

pub const Kind = enum {
    value, // --flag VALUE or --flag=VALUE
    store_true, // --flag
    append, // repeatable --flag VALUE
};

pub const Flag = struct {
    name: []const u8,
    kind: Kind = .value,
    /// argparse's choices; empty: any value of the flag's type.
    choices: []const []const u8 = &.{},
    /// This binary honours the flag (and lists it in ``capabilities``); the rest are refused as unsupported.
    native: bool = false,
    /// Listed in ``capabilities``: a flag a build parses only to refuse it by name sets false.
    advertised: bool = true,
    /// The values it honours when only some of the choices.
    native_values: ?[]const []const u8 = null,
    /// The value names a file or folder: a config file's relative one is read from that file's folder (cli_config.zig).
    path: bool = false,
};
