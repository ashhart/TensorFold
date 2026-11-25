// TFQ3 only (the GLM kernels' enableTfq3 compiles it): KDA parts and window templates at 8-bit/64 gates (FB = GB = 8).
template [[host_name("glm_kda_parts_q8")]] [[kernel]] decltype(glm_kda_parts<64, 128, 4, 16, 8, 8, 8, 2>) glm_kda_parts<64, 128, 4, 16, 8, 8, 8, 2>;
template [[host_name("glm_kda_window_q8")]] [[kernel]] decltype(glm_kda_window<64, 128, 4, 32, 8, 8, 8>) glm_kda_window<64, 128, 4, 32, 8, 8, 8>;
