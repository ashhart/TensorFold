/* Host-only HIP ABI fixture, never linked into the runtime. */
#ifndef OMIT_INIT
int hipInit(unsigned int flags) {
    return flags == 0 ? INIT_RESULT : 1;
}
#endif
int hipGetDeviceCount(int *count) {
    *count = 2;
    return 0;
}
