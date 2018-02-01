/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/platform/default/gpu/cupti_wrapper.h"

#if GOOGLE_CUDA

#include <string>

#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/stream_executor.h"

namespace perftools {
namespace gputools {
namespace profiler {

namespace dynload {

#define LIBCUPTI_WRAP(__name)                                               \
  struct DynLoadShim__##__name {                                            \
    static const char* kName;                                               \
    using FuncPointerT = std::add_pointer<decltype(::__name)>::type;        \
    static void* GetDsoHandle() {                                           \
      static auto status = perftools::gputools::internal::CachedDsoLoader:: \
          GetLibcuptiDsoHandle();                                           \
      return status.ValueOrDie();                                           \
    }                                                                       \
    static FuncPointerT DynLoad() {                                         \
      static void* f;                                                       \
      TF_CHECK_OK(::tensorflow::Env::Default()->GetSymbolFromLibrary(       \
          GetDsoHandle(), kName, &f))                                       \
          << "could not find " << kName << "in libcupti DSO";               \
      return reinterpret_cast<FuncPointerT>(f);                             \
    }                                                                       \
    template <typename... Args>                                             \
    CUptiResult operator()(Args... args) {                                  \
      return DynLoad()(args...);                                            \
    }                                                                       \
  } __name;                                                                 \
  const char* DynLoadShim__##__name::kName = #__name;

LIBCUPTI_WRAP(cuptiActivityDisable);
LIBCUPTI_WRAP(cuptiActivityEnable);
LIBCUPTI_WRAP(cuptiActivityFlushAll);
LIBCUPTI_WRAP(cuptiActivityGetNextRecord);
LIBCUPTI_WRAP(cuptiActivityGetNumDroppedRecords);
LIBCUPTI_WRAP(cuptiActivityRegisterCallbacks);
LIBCUPTI_WRAP(cuptiGetTimestamp);
LIBCUPTI_WRAP(cuptiEnableCallback);
LIBCUPTI_WRAP(cuptiEnableDomain);
LIBCUPTI_WRAP(cuptiSubscribe);
LIBCUPTI_WRAP(cuptiUnsubscribe);
// added events and metric wraps, by Hujian
LIBCUPTI_WRAP(cuptiMetricGetIdFromName);
LIBCUPTI_WRAP(cuptiMetricGetNumEvents);
LIBCUPTI_WRAP(cuptiMetricEnumEvents);
LIBCUPTI_WRAP(cuptiMetricCreateEventGroupSets);
LIBCUPTI_WRAP(cuptiSetEventCollectionMode);
LIBCUPTI_WRAP(cuptiEventGroupSetAttribute);
LIBCUPTI_WRAP(cuptiEventGroupEnable);
LIBCUPTI_WRAP(cuptiEventGroupGetAttribute);
LIBCUPTI_WRAP(cuptiDeviceGetEventDomainAttribute);
LIBCUPTI_WRAP(cuptiEventGroupReadEvent);
LIBCUPTI_WRAP(cuptiEventGroupResetAllEvents);
LIBCUPTI_WRAP(cuptiMetricGetValue);
LIBCUPTI_WRAP(cuptiGetDeviceId);
LIBCUPTI_WRAP(cuptiEventGroupDisable);
LIBCUPTI_WRAP(cuptiGetContextId);
LIBCUPTI_WRAP(cuptiEnableKernelReplayMode);
LIBCUPTI_WRAP(cuptiMetricGetAttribute);
LIBCUPTI_WRAP(cuptiDisableKernelReplayMode);
LIBCUPTI_WRAP(cuptiEventGroupSetsDestroy);

}  // namespace dynload

CUptiResult CuptiWrapper::ActivityDisable(CUpti_ActivityKind kind) {
  return dynload::cuptiActivityDisable(kind);
}

CUptiResult CuptiWrapper::ActivityEnable(CUpti_ActivityKind kind) {
  return dynload::cuptiActivityEnable(kind);
}

CUptiResult CuptiWrapper::ActivityFlushAll(uint32_t flag) {
  return dynload::cuptiActivityFlushAll(flag);
}

CUptiResult CuptiWrapper::ActivityGetNextRecord(uint8_t* buffer,
                                                size_t valid_buffer_size_bytes,
                                                CUpti_Activity** record) {
  return dynload::cuptiActivityGetNextRecord(buffer, valid_buffer_size_bytes,
                                             record);
}

CUptiResult CuptiWrapper::ActivityGetNumDroppedRecords(CUcontext context,
                                                       uint32_t stream_id,
                                                       size_t* dropped) {
  return dynload::cuptiActivityGetNumDroppedRecords(context, stream_id,
                                                    dropped);
}

CUptiResult CuptiWrapper::ActivityRegisterCallbacks(
    CUpti_BuffersCallbackRequestFunc func_buffer_requested,
    CUpti_BuffersCallbackCompleteFunc func_buffer_completed) {
  return dynload::cuptiActivityRegisterCallbacks(func_buffer_requested,
                                                 func_buffer_completed);
}

CUptiResult CuptiWrapper::GetDeviceId(CUcontext context, uint32_t* deviceId) {
  return dynload::cuptiGetDeviceId(context, deviceId);
}

CUptiResult CuptiWrapper::GetTimestamp(uint64_t* timestamp) {
  return dynload::cuptiGetTimestamp(timestamp);
}

CUptiResult CuptiWrapper::EnableCallback(uint32_t enable,
                                         CUpti_SubscriberHandle subscriber,
                                         CUpti_CallbackDomain domain,
                                         CUpti_CallbackId cbid) {
  return dynload::cuptiEnableCallback(enable, subscriber, domain, cbid);
}

CUptiResult CuptiWrapper::EnableDomain(uint32_t enable,
                                       CUpti_SubscriberHandle subscriber,
                                       CUpti_CallbackDomain domain) {
  return dynload::cuptiEnableDomain(enable, subscriber, domain);
}

CUptiResult CuptiWrapper::Subscribe(CUpti_SubscriberHandle* subscriber,
                                    CUpti_CallbackFunc callback,
                                    void* userdata) {
  return dynload::cuptiSubscribe(subscriber, callback, userdata);
}

CUptiResult CuptiWrapper::Unsubscribe(CUpti_SubscriberHandle subscriber) {
  return dynload::cuptiUnsubscribe(subscriber);
}

// added events and metric apis, by Hujian
CUptiResult CuptiWrapper::cuptiMetricGetIdFromName(CUdevice device,
                                              const char *metricName,
                                              CUpti_MetricID *metric) {
  return dynload::cuptiMetricGetIdFromName(device, metricName, metric);
}

CUptiResult CuptiWrapper::cuptiMetricGetNumEvents(CUpti_MetricID metric, uint32_t *numEvents) {
  return dynload::cuptiMetricGetNumEvents(metric, numEvents);
}

CUptiResult CuptiWrapper::cuptiMetricEnumEvents(CUpti_MetricID metric,
                                  size_t *eventIdArraySizeBytes,
                                  CUpti_EventID *eventIdArray) {
  return dynload::cuptiMetricEnumEvents(metric, eventIdArraySizeBytes, eventIdArray);
}

CUptiResult CuptiWrapper::cuptiMetricCreateEventGroupSets(CUcontext context,
                                            size_t metricIdArraySizeBytes,
                                            CUpti_MetricID *metricIdArray,
                                            CUpti_EventGroupSets **eventGroupPasses) {
  return dynload::cuptiMetricCreateEventGroupSets(context, metricIdArraySizeBytes,
                                                  metricIdArray, eventGroupPasses);
}

CUptiResult CuptiWrapper::cuptiSetEventCollectionMode(CUcontext context,
                                        CUpti_EventCollectionMode mode) {
  return dynload::cuptiSetEventCollectionMode(context, mode);
}

CUptiResult CuptiWrapper::cuptiEventGroupSetAttribute(CUpti_EventGroup eventGroup,
                                          CUpti_EventGroupAttribute attrib,
                                          size_t valueSize,
                                          void *value) {
  return dynload::cuptiEventGroupSetAttribute(eventGroup, attrib, valueSize, value);
}

CUptiResult CuptiWrapper::cuptiEventGroupEnable(CUpti_EventGroup eventGroup) {
  return dynload::cuptiEventGroupEnable(eventGroup);
}

CUptiResult CuptiWrapper::cuptiEventGroupGetAttribute(CUpti_EventGroup eventGroup,
                                        CUpti_EventGroupAttribute attrib,
                                        size_t *valueSize,
                                        void *value) {
  return dynload::cuptiEventGroupGetAttribute(eventGroup, attrib, valueSize, value);
}

CUptiResult CuptiWrapper::cuptiDeviceGetEventDomainAttribute(CUdevice device,
                                                 CUpti_EventDomainID eventDomain,
                                                 CUpti_EventDomainAttribute attrib,
                                                 size_t *valueSize,
                                                 void *value) {
  return dynload::cuptiDeviceGetEventDomainAttribute(device, eventDomain, attrib, 
                                                     valueSize, value);
}

CUptiResult CuptiWrapper::cuptiEventGroupReadEvent(CUpti_EventGroup eventGroup,
                                       CUpti_ReadEventFlags flags,
                                       CUpti_EventID event,
                                       size_t *eventValueBufferSizeBytes,
                                       uint64_t *eventValueBuffer) {
  return dynload::cuptiEventGroupReadEvent(eventGroup, flags, event,
                                           eventValueBufferSizeBytes, eventValueBuffer);
}

CUptiResult CuptiWrapper::cuptiEventGroupResetAllEvents(CUpti_EventGroup eventGroup) {
  return dynload::cuptiEventGroupResetAllEvents(eventGroup);
}

CUptiResult CuptiWrapper::cuptiMetricGetValue(CUdevice device,
                                  CUpti_MetricID metric,
                                  size_t eventIdArraySizeBytes,
                                  CUpti_EventID *eventIdArray,
                                  size_t eventValueArraySizeBytes,
                                  uint64_t *eventValueArray,
                                  uint64_t timeDuration,
                                  CUpti_MetricValue *metricValue) {
  return dynload::cuptiMetricGetValue(device, metric, eventIdArraySizeBytes, eventIdArray,
                                      eventValueArraySizeBytes, eventValueArray, timeDuration,
                                      metricValue);
}

CUptiResult CuptiWrapper::cuptiEventGroupDisable(CUpti_EventGroup eventGroup) {
  return dynload::cuptiEventGroupDisable(eventGroup);
}

CUptiResult CuptiWrapper::cuptiGetContextId(CUcontext context, uint32_t *contextId){
  return dynload::cuptiGetContextId(context, contextId);
}

CUptiResult CuptiWrapper::cuptiEnableKernelReplayMode(CUcontext context) {
  return dynload::cuptiEnableKernelReplayMode(context);
}

CUptiResult CuptiWrapper::cuptiMetricGetAttribute(CUpti_MetricID metric,
                                    CUpti_MetricAttribute attrib,
                                    size_t *valueSize,
                                    void *value) {
  return dynload::cuptiMetricGetAttribute(metric, attrib, valueSize, value);
}

CUptiResult CuptiWrapper::cuptiDisableKernelReplayMode(CUcontext context) {
  return dynload::cuptiDisableKernelReplayMode(context);
}

CUptiResult CuptiWrapper::cuptiEventGroupSetsDestroy(CUpti_EventGroupSets *eventGroupSets) {
  return dynload::cuptiEventGroupSetsDestroy(eventGroupSets);
}

}  // namespace profiler
}  // namespace gputools
}  // namespace perftools

#endif  // GOOGLE_CUDA
