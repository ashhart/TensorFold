// Filtered Linux recording tests never call these; real Metal modules still link their declarations.
void *objc_getClass(const char *name) {
    (void)name;
    return 0;
}

void *sel_registerName(const char *name) {
    (void)name;
    return 0;
}

void *objc_msgSend(void) {
    return 0;
}
