// nemotron_h_puzzle (Bolt): group_norm_xs.metal's body at mamba inner XD 8192 in groups of GS 1024, one MP each
template [[host_name("tf_nemotron_group_norm_xs_bolt_16")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<16>) tf_nemotron_group_norm_xs_bolt<16>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_32")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<32>) tf_nemotron_group_norm_xs_bolt<32>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_48")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<48>) tf_nemotron_group_norm_xs_bolt<48>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_64")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<64>) tf_nemotron_group_norm_xs_bolt<64>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_80")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<80>) tf_nemotron_group_norm_xs_bolt<80>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_96")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<96>) tf_nemotron_group_norm_xs_bolt<96>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_112")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<112>) tf_nemotron_group_norm_xs_bolt<112>;
template [[host_name("tf_nemotron_group_norm_xs_bolt_128")]] [[kernel]] decltype(tf_nemotron_group_norm_xs_bolt<128>) tf_nemotron_group_norm_xs_bolt<128>;
