#ifndef TENSORFOLD_WINDOWS_SANDBOX_H
#define TENSORFOLD_WINDOWS_SANDBOX_H
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0A00
#endif
#ifndef UNICODE
#define UNICODE 1
#endif
#ifndef _UNICODE
#define _UNICODE 1
#endif
#include <windows.h>
#include <userenv.h>
#include <aclapi.h>
#include <sddl.h>
#include <wchar.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  const wchar_t *root;
  const wchar_t *workspace;
  const wchar_t *scratch;
  const wchar_t *executable;
  wchar_t *command_line;
  DWORD parent_pid;
  DWORD timeout_ms;
  BOOL network;
  BOOL writable_workspace;
  wchar_t profile[80];
  PSID sid;
} SandboxPlan;

BOOL sandbox_inside(const wchar_t *root, const wchar_t *path);
BOOL sandbox_validate_directory(const wchar_t *path);
BOOL sandbox_prepare_directory(const wchar_t *path);
BOOL sandbox_validate_input(const wchar_t *path);
BOOL sandbox_export(const wchar_t *source, const wchar_t *destination, BOOL input_links);
BOOL sandbox_grant_tree(SandboxPlan *plan);
DWORD sandbox_execute(SandboxPlan *plan);
wchar_t *sandbox_command_line(int count, wchar_t **args);
DWORD sandbox_child_probe(void);
DWORD sandbox_probe(void);
#endif
