#include "thread_utils.h"
#include <algorithm>

#include <core/common/make_unique.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <thread>

namespace onnxruntime {
namespace concurrency {
#ifdef _WIN32
// This function doesn't support systems with more than 64 logical processors
static int GetNumCpuCores() {
  // Indeed 64 should be enough. However, it's harmless to have a little more.
  SYSTEM_LOGICAL_PROCESSOR_INFORMATION buffer[256];
  DWORD returnLength = sizeof(buffer);
  if (GetLogicalProcessorInformation(buffer, &returnLength) == FALSE) {
    // try GetSystemInfo
    SYSTEM_INFO sysInfo;
    GetSystemInfo(&sysInfo);
    if (sysInfo.dwNumberOfProcessors <= 0) {
      return static_cast<int>(std::thread::hardware_concurrency() / 2);
    }
    // This is the number of logical processors in the current group
    return sysInfo.dwNumberOfProcessors;
  }
  int processorCoreCount = 0;
  int count = (int)(returnLength / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
  for (int i = 0; i != count; ++i) {
    if (buffer[i].Relationship == RelationProcessorCore) {
      ++processorCoreCount;
    }
  }
  if (!processorCoreCount)
    return static_cast<int>(std::thread::hardware_concurrency() / 2);
  return processorCoreCount;
}
#else
static int GetNumCpuCores() {
  return static_cast<int>(std::thread::hardware_concurrency() / 2);
}
#endif
std::unique_ptr<ThreadPool> CreateThreadPool(int thread_pool_size, Env* env, const ThreadOptions& thread_options,
                                             const ORTCHAR_T* name, bool allow_spinning, Eigen::Allocator* allocator) {
  if (thread_pool_size <= 0) {  // default
    thread_pool_size = std::max<int>(1, GetNumCpuCores());
  }
  // since we use the main thread for execution we don't have to create any threads on the thread pool when
  // the requested size is 1. For other cases, we will have thread_pool_size + 1 threads for execution
  if (thread_pool_size == 1)
    return nullptr;
  return onnxruntime::make_unique<ThreadPool>(env, thread_options, name, thread_pool_size, allow_spinning, allocator);
}
}  // namespace concurrency
}  // namespace onnxruntime