#include "sandbox.h"
#include <objbase.h>

#ifndef PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY
#define PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY 0x0002000F
#endif

DWORD sandbox_execute(SandboxPlan *plan) {
  DWORD result = 126;
  GUID guid;
  wchar_t identity[40], cancellation[32768];
  HANDLE job = NULL, parent = NULL, inherited[3] = {NULL, NULL, NULL};
  PROCESS_INFORMATION process = {0};
  STARTUPINFOEXW startup = {0};
  SIZE_T size = 0;
  BOOL attributes_initialized = FALSE;
  PSID network_sid = NULL;
  if (CoCreateGuid(&guid) != S_OK || !StringFromGUID2(&guid, identity, 40)) goto done;
  swprintf(plan->profile, 80, L"TensorFold.%ls", identity + 1);
  plan->profile[wcslen(plan->profile) - 1] = L'\0';
  HRESULT created = CreateAppContainerProfile(plan->profile, L"TensorFold isolated process", L"Temporary project isolation", NULL, 0, &plan->sid);
  if (FAILED(created) || !plan->sid || !sandbox_grant_tree(plan)) goto done;
  if (swprintf(cancellation, 32768, L"%ls\\cancel", plan->root) < 0) goto done;
  parent = OpenProcess(SYNCHRONIZE, FALSE, plan->parent_pid);
  job = CreateJobObjectW(NULL, NULL);
  if (!parent || !job) goto done;
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits = {0};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
  limits.BasicLimitInformation.ActiveProcessLimit = 64;
  JOBOBJECT_BASIC_UI_RESTRICTIONS ui = {0};
  ui.UIRestrictionsClass = JOB_OBJECT_UILIMIT_HANDLES | JOB_OBJECT_UILIMIT_READCLIPBOARD | JOB_OBJECT_UILIMIT_WRITECLIPBOARD
    | JOB_OBJECT_UILIMIT_SYSTEMPARAMETERS | JOB_OBJECT_UILIMIT_DISPLAYSETTINGS | JOB_OBJECT_UILIMIT_GLOBALATOMS
    | JOB_OBJECT_UILIMIT_DESKTOP | JOB_OBJECT_UILIMIT_EXITWINDOWS;
  if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, &limits, sizeof(limits))
    || !SetInformationJobObject(job, JobObjectBasicUIRestrictions, &ui, sizeof(ui))) goto done;
  SECURITY_CAPABILITIES capabilities = {0};
  SID_AND_ATTRIBUTES capability = {0};
  capabilities.AppContainerSid = plan->sid;
  if (plan->network) {
    if (!ConvertStringSidToSidW(L"S-1-15-3-1", &network_sid)) goto done;
    capability.Sid = network_sid; capability.Attributes = SE_GROUP_ENABLED;
    capabilities.Capabilities = &capability; capabilities.CapabilityCount = 1;
  }
  InitializeProcThreadAttributeList(NULL, 3, 0, &size);
  startup.lpAttributeList = HeapAlloc(GetProcessHeap(), 0, size);
  if (!startup.lpAttributeList || !InitializeProcThreadAttributeList(startup.lpAttributeList, 3, 0, &size)) goto done;
  attributes_initialized = TRUE;
  DWORD opt_out = 1;
  for (DWORD i = 0; i < 3; i++) {
    HANDLE source = GetStdHandle(i == 0 ? STD_INPUT_HANDLE : i == 1 ? STD_OUTPUT_HANDLE : STD_ERROR_HANDLE);
    if (!source || source == INVALID_HANDLE_VALUE
      || !DuplicateHandle(GetCurrentProcess(), source, GetCurrentProcess(), &inherited[i], 0, TRUE, DUPLICATE_SAME_ACCESS)) goto done;
  }
  if (!UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES, &capabilities, sizeof(capabilities), NULL, NULL)
    || !UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_ALL_APPLICATION_PACKAGES_POLICY, &opt_out, sizeof(opt_out), NULL, NULL)
    || !UpdateProcThreadAttribute(startup.lpAttributeList, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited, sizeof(inherited), NULL, NULL)) goto done;
  if (WaitForSingleObject(parent, 0) == WAIT_OBJECT_0 || GetFileAttributesW(cancellation) != INVALID_FILE_ATTRIBUTES) { result = 130; goto done; }
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = inherited[0]; startup.StartupInfo.hStdOutput = inherited[1]; startup.StartupInfo.hStdError = inherited[2];
  if (!CreateProcessW(plan->executable, plan->command_line, NULL, NULL, TRUE,
    CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
    NULL, plan->workspace, &startup.StartupInfo, &process)) goto done;
  if (!AssignProcessToJobObject(job, process.hProcess)) goto done;
  wchar_t ready[32768];
  if (swprintf(ready, 32768, L"%ls\\ready", plan->root) < 0) goto done;
  HANDLE marker = CreateFileW(ready, GENERIC_WRITE, 0, NULL, CREATE_NEW, FILE_ATTRIBUTE_NORMAL, NULL);
  if (marker == INVALID_HANDLE_VALUE) goto done;
  CloseHandle(marker);
  if (ResumeThread(process.hThread) == (DWORD)-1) goto done;
  ULONGLONG start = GetTickCount64();
  HANDLE wait[] = {process.hProcess, parent};
  while (TRUE) {
    DWORD state = WaitForMultipleObjects(2, wait, FALSE, 100);
    if (state == WAIT_OBJECT_0) { if (!GetExitCodeProcess(process.hProcess, &result)) result = 126; break; }
    if (state == WAIT_OBJECT_0 + 1 || GetFileAttributesW(cancellation) != INVALID_FILE_ATTRIBUTES) { result = 130; break; }
    if (state == WAIT_FAILED) break;
    if (plan->timeout_ms && GetTickCount64() - start >= plan->timeout_ms) { result = 124; break; }
  }
done:
  if (job) {
    TerminateJobObject(job, result);
    ULONGLONG deadline = GetTickCount64() + 5000;
    JOBOBJECT_BASIC_ACCOUNTING_INFORMATION accounting;
    while (QueryInformationJobObject(job, JobObjectBasicAccountingInformation, &accounting, sizeof(accounting), NULL)
      && accounting.ActiveProcesses && GetTickCount64() < deadline) Sleep(10);
  }
  if (process.hProcess) { TerminateProcess(process.hProcess, result); WaitForSingleObject(process.hProcess, 5000); CloseHandle(process.hProcess); }
  if (process.hThread) CloseHandle(process.hThread);
  if (job) CloseHandle(job);
  if (parent) CloseHandle(parent);
  for (DWORD i = 0; i < 3; i++) if (inherited[i]) CloseHandle(inherited[i]);
  if (attributes_initialized) DeleteProcThreadAttributeList(startup.lpAttributeList);
  if (startup.lpAttributeList) HeapFree(GetProcessHeap(), 0, startup.lpAttributeList);
  if (network_sid) LocalFree(network_sid);
  if (plan->sid) { FreeSid(plan->sid); DeleteAppContainerProfile(plan->profile); plan->sid = NULL; }
  if (result == 126) fprintf(stderr, "TensorFold AppContainer isolation failed (%lu). No unrestricted process was started.\n", GetLastError());
  return result;
}
