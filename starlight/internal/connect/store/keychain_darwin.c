//go:build darwin && cgo

#include <stdlib.h>
#include <string.h>
#include "keychain_darwin.h"

static CFDictionaryRef tf_query(const char *service, const char *account, int extra, const void **keys, const void **values) {
	CFStringRef svc = CFStringCreateWithCString(NULL, service, kCFStringEncodingUTF8);
	CFStringRef acc = CFStringCreateWithCString(NULL, account, kCFStringEncodingUTF8);
	const void *k[8] = { kSecClass, kSecAttrService, kSecAttrAccount };
	const void *v[8] = { kSecClassGenericPassword, svc, acc };
	for (int i = 0; i < extra && i < 5; i++) {
		k[3 + i] = keys[i];
		v[3 + i] = values[i];
	}
	CFDictionaryRef q = CFDictionaryCreate(NULL, k, v, 3 + extra, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	CFRelease(svc);
	CFRelease(acc);
	return q;
}

OSStatus tf_keychain_set(const char *service, const char *account, const void *data, int length) {
	CFDataRef value = CFDataCreate(NULL, data, length);
	CFDictionaryRef q = tf_query(service, account, 0, NULL, NULL);
	const void *uk[] = { kSecValueData };
	const void *uv[] = { value };
	CFDictionaryRef update = CFDictionaryCreate(NULL, uk, uv, 1, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks);
	OSStatus status = SecItemUpdate(q, update);
	if (status == errSecItemNotFound) {
		CFDictionaryRef add = tf_query(service, account, 1, uk, uv);
		status = SecItemAdd(add, NULL);
		CFRelease(add);
	}
	CFRelease(update);
	CFRelease(q);
	CFRelease(value);
	return status;
}

OSStatus tf_keychain_get(const char *service, const char *account, void **out, int *length) {
	const void *k[] = { kSecReturnData, kSecMatchLimit };
	const void *v[] = { kCFBooleanTrue, kSecMatchLimitOne };
	CFDictionaryRef q = tf_query(service, account, 2, k, v);
	CFTypeRef result = NULL;
	OSStatus status = SecItemCopyMatching(q, &result);
	CFRelease(q);
	if (status != errSecSuccess) {
		return status;
	}
	CFDataRef data = (CFDataRef)result;
	*length = (int)CFDataGetLength(data);
	*out = malloc(*length > 0 ? *length : 1);
	if (*out == NULL) {
		CFRelease(result);
		return errSecAllocate;
	}
	memcpy(*out, CFDataGetBytePtr(data), *length);
	CFRelease(result);
	return status;
}

OSStatus tf_keychain_delete(const char *service, const char *account) {
	CFDictionaryRef q = tf_query(service, account, 0, NULL, NULL);
	OSStatus status = SecItemDelete(q);
	CFRelease(q);
	return status;
}
