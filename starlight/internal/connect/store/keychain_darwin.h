//go:build darwin && cgo

#include <stdlib.h>
#include <Security/Security.h>

OSStatus tf_keychain_set(const char *service, const char *account, const void *data, int length);
OSStatus tf_keychain_get(const char *service, const char *account, void **out, int *length);
OSStatus tf_keychain_delete(const char *service, const char *account);
