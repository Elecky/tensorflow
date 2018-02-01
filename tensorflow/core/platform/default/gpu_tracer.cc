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

#include "tensorflow/core/platform/gpu_tracer.h"

#if GOOGLE_CUDA

#include <stdlib.h>
#include <memory>
#include <fstream>
#include <string>
#include <cstdlib>
#include <unordered_set>
#include <functional>
#include <mutex>
#include <unordered_map>
#include <atomic>

#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/cupti_wrapper.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/stream_executor/cuda/cuda_driver.h"

namespace {

// Maps a MemcpyKind enum to a const string.
const char *getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      break;
  }
  return "<unknown>";
}

// Maps a MemoryKind enum to a const string.
const char *getMemoryKindString(CUpti_ActivityMemoryKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_MEMORY_KIND_UNKNOWN:
      return "Unknown";
    case CUPTI_ACTIVITY_MEMORY_KIND_PAGEABLE:
      return "Pageable";
    case CUPTI_ACTIVITY_MEMORY_KIND_PINNED:
      return "Pinned";
    case CUPTI_ACTIVITY_MEMORY_KIND_DEVICE:
      return "Device";
    case CUPTI_ACTIVITY_MEMORY_KIND_ARRAY:
      return "Array";
    default:
      break;
  }
  return "<unknown>";
}

// Maps an OverheadKind enum to a const string.
const char *getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
  switch (kind) {
    case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
      return "COMPILER";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
      return "BUFFER_FLUSH";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
      return "INSTRUMENTATION";
    case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
      return "RESOURCE";
    default:
      break;
  }
  return "<unknown>";
}

}  // namespace

namespace tensorflow {
namespace gputracer {

// Forward declaration.
class CUPTIManager;

// Returns a pointer to the CUPTIManager singleton.
CUPTIManager *GetCUPTIManager();

// Callback interface for consumers of CUPTI tracing.
class CUPTIClient {
 public:
  virtual ~CUPTIClient() {}

  // Invoked for each CUPTI activity reported.
  virtual void ActivityCallback(const CUpti_Activity &activity) = 0;
};

#define CUPTI_CALL(call)                                            \
  do {                                                              \
    CUptiResult _status = cupti_wrapper_->call;                     \
    if (_status != CUPTI_SUCCESS) {                                 \
      LOG(ERROR) << "cuda call " << #call << " failed " << _status; \
    }                                                               \
  } while (0)

// Singleton class to manage registration of CUPTI callbacks.
class CUPTIManager {
 public:
  CUPTIManager() {
    cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
    CUPTI_CALL(ActivityRegisterCallbacks(BufferRequested, BufferCompleted));
  }

  // Enables tracing and delivers event callbacks to 'client'.
  // Does not take ownership of client.  Client's lifetime must persist
  // until tracing is disabled.
  Status EnableTrace(CUPTIClient *client);

  // Disable tracing.  No further events will be delivered to 'client'.
  Status DisableTrace();

 private:
  // Static functions which we can use as CUPTI callbacks.
  static void BufferRequested(uint8_t **buffer, size_t *size,
                              size_t *maxNumRecords) {
    GetCUPTIManager()->InternalBufferRequested(buffer, size, maxNumRecords);
  }
  static void BufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer,
                              size_t size, size_t validSize) {
    GetCUPTIManager()->InternalBufferCompleted(ctx, streamId, buffer, size,
                                               validSize);
  }
  // These methods are called by the static stubs above.
  void InternalBufferRequested(uint8_t **buffer, size_t *size,
                               size_t *maxNumRecords);
  void InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                               uint8_t *buffer, size_t size, size_t validSize);

  // Size of buffers used for CUPTI tracing.
  static constexpr size_t kBufferSize = 32 * 1024;
  // Required alignment of CUPTI buffers.
  static constexpr size_t kBufferAlignment = 8;

  mutex mu_;
  CUPTIClient *client_ GUARDED_BY(mu_);
  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;

  TF_DISALLOW_COPY_AND_ASSIGN(CUPTIManager);
};

Status CUPTIManager::EnableTrace(CUPTIClient *client) {
  mutex_lock l(mu_);
  // TODO(pbar) Work out the minimal set to trace.
  // We can currently manage without driver/runtime tracing.
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  // These might be useful for annotations but require NVTX API.
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  // CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));

  CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY2));
  CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  CUPTI_CALL(ActivityEnable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
  client_ = client;
  return Status::OK();
}

Status CUPTIManager::DisableTrace() {
  // We turn off all tracing regardless.
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MARKER));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_OVERHEAD));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY2));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(ActivityDisable(CUPTI_ACTIVITY_KIND_ENVIRONMENT));
  CUPTI_CALL(ActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED));
  {
    // Don't acquire this lock until Flush returns, since Flush
    // will potentially cause callbacks into BufferCompleted.
    mutex_lock l(mu_);
    client_ = nullptr;
  }
  return Status::OK();
}

void CUPTIManager::InternalBufferRequested(uint8_t **buffer, size_t *size,
                                           size_t *maxNumRecords) {
  VLOG(2) << "BufferRequested";
  void *p = port::AlignedMalloc(kBufferSize, kBufferAlignment);
  *size = kBufferSize;
  *buffer = reinterpret_cast<uint8_t *>(p);
  *maxNumRecords = 0;
}

void CUPTIManager::InternalBufferCompleted(CUcontext ctx, uint32_t streamId,
                                           uint8_t *buffer, size_t size,
                                           size_t validSize) {
  VLOG(2) << "BufferCompleted";
  CUptiResult status;
  CUpti_Activity *record = nullptr;
  mutex_lock l(mu_);  // Hold mu_ while using client_.
  if (client_ && validSize > 0) {
    do {
      status =
          cupti_wrapper_->ActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        client_->ActivityCallback(*record);
      } else {
        break;
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(ActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      LOG(WARNING) << "Dropped " << dropped << " activity records";
    }
  }
  port::AlignedFree(buffer);
}

CUPTIManager *GetCUPTIManager() {
  static CUPTIManager *manager = new CUPTIManager();
  return manager;
}

#ifdef _MSC_VER
#define __thread __declspec(thread)
#endif

// TODO(pbar) Move this to platform specific header file?
// Static thread local variable for POD types.
#define TF_STATIC_THREAD_LOCAL_POD(_Type_, _var_)                  \
  static __thread _Type_ s_obj_##_var_;                            \
  namespace {                                                      \
  class ThreadLocal_##_var_ {                                      \
   public:                                                         \
    ThreadLocal_##_var_() {}                                       \
    void Init() {}                                                 \
    inline _Type_ *pointer() const { return &s_obj_##_var_; }      \
    inline _Type_ *safe_pointer() const { return &s_obj_##_var_; } \
    _Type_ &get() const { return s_obj_##_var_; }                  \
    bool is_native_tls() const { return true; }                    \
                                                                   \
   private:                                                        \
    TF_DISALLOW_COPY_AND_ASSIGN(ThreadLocal_##_var_);              \
  } _var_;                                                         \
  }  // namespace

// Thread-local state recording the most recent annotation (if any).
// When non-null, this points to a string in the active annotation
// of the current thread.  The annotation is guaranteed to remain live
// for the duration of the CUPTI API callback.
TF_STATIC_THREAD_LOCAL_POD(const char *, tls_current_annotation);

class GPUTracerImpl : public GPUTracer,
                      public CUPTIClient,
                      public port::Tracing::Engine {
 public:
  GPUTracerImpl();
  ~GPUTracerImpl() override;

  // GPUTracer interface:
  Status Start() override;
  Status Stop() override;
  Status Collect(StepStatsCollector *collector) override;

  // port::Tracing::Engine interface:
  bool IsEnabled() const override {
    // We only register the Engine while tracing is enabled.
    return true;
  }
  Annotation *PushAnnotation(StringPiece name) override {
    VLOG(2) << "PushAnnotation " << name;
    struct Impl : public port::Tracing::Engine::Annotation {
      string annotation;
      explicit Impl(StringPiece n) : annotation(n.ToString()) {
        // Remember the most recent ScopedAnnotation for each thread.
        tls_current_annotation.get() = annotation.c_str();
      }
      ~Impl() override { tls_current_annotation.get() = nullptr; }
    };
    return new Impl(name);
  }
  Tracer *StartTracing(StringPiece label) override {
    // We don't do anything with 'TraceMe' regions yet.
    return nullptr;
  }

 protected:
  // This callback is used exclusively by CUPTIManager.
  friend class CUPTIManager;
  void ActivityCallback(const CUpti_Activity &activity) override;

 private:
  // Internal struct to record kernel launches.
  struct KernelRecord {
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint32 device_id;
    uint32 stream_id;
    uint32 correlation_id;
  };
  // Internal struct to record memcpy operations.
  struct MemcpyRecord {
    uint64_t start_timestamp;
    uint64_t end_timestamp;
    uint32 device_id;
    uint32 stream_id;
    uint32 correlation_id;
    uint8 copyKind;
    uint8 srcKind;
    uint8 dstKind;
    uint64 bytes;
  };

  // Internal struct to record environment stats. By HuJian
  struct EnvironmentRecord {
    uint64_t timestamp;
    uint32 device_id;
    // environment kind:
    // 0 for unknown, 1 for speed, 2 for temperature, 3 for power, 4 for cooling.
    uint32 environment_kind;

    union {
      // data for speed
      struct {
        // The SM frequency in MHz
        uint32 smClock;
        // The memory frequency in MHz
        uint32 memoryClock;
        // The PCIe link generation.
        uint32 pcieLinkGen;
        // The PCIe link width.
        uint32 pcieLinkWidth;
        // The clocks throttle reasons. check for document
        uint32 clocksThrottleReasons;
      } speed;
      // data for temperature
      struct {
        // GPU temperature in degrees C(maybe Celsius)
        uint32 gpuTemperature;
      } temperature;
      // data for power
      struct {
        // The power in milliwatts consumed by GPU and associated
        // circuitry.
        uint32 power;
        // The power in milliwatts that will trigger power management
        // algorithm.
        uint32 powerLimit;
      } power;
      // data for cooling
      struct {
        // The fan speed as percentage of maximum.
        uint32 fanSpeed;
      } cooling;
    } data;
  };

  struct MetricRecord {
    uint64_t start, end;
    std::string name;  // the scope name
    std::vector<CUpti_MetricValue> Values;
  };

  // Internal struct to record event values related to metrics, by Hujian
  // each context has one structure.
  struct MetricData {
    // init a MetricData structure
    MetricData();
    MetricData(CUcontext context);
    void Init(CUcontext context);
    ~MetricData();

    int StartRecord();

    void ToCSV(std::ofstream &out, bool writeHeader);

    // stop record, and save event values to savedEvents
    int EndRecord(uint32_t correlation_id);

    void ComputeMetric(uint64_t start, uint64_t end, uint32_t correlation_id,
                       const std::string &annotate);

    // the device where metric is being collected
    CUdevice device;
    // the context id
    uint32_t contextId;
    CUcontext context;
    // the metric names and ids been sampled
    CUpti_MetricID *metricIds = nullptr;
    std::vector<std::string> metrics;
    std::size_t numMetrics;
    // the sets object, stored to be destroyed when destruction
    CUpti_EventGroupSets *sets;
    // the set of event groups to collect for a pass
    CUpti_EventGroupSet *eventGroups;
    std::size_t passes;
    // the current number of events collected in eventIdArray and
    // eventValueArray
    // uint32_t eventIdx;
    // the number of entries in eventIdArray and eventValueArray
    uint32_t numEvents;
    std::vector<std::size_t> eventIdxToGroupId;
    std::vector<CUpti_EventID> eventIdArray;
    std::vector<uint32_t> numTotalInstances;  // the total instances of groups
    std::vector<uint32_t> numInstances;  // the event counter instances of groups
    std::unique_ptr<uint64_t[]> valuesBuffer;

    // saved event ids and values
    std::unordered_map<uint32_t, 
                       std::pair<std::unique_ptr<CUpti_EventID[]>, 
                                 std::unique_ptr<uint64_t[]> > > savedEvents;
    std::atomic_flag eventMapLock = ATOMIC_FLAG_INIT;

    // event it to index
    //std::unordered_map<CUpti_EventID, std::size_t> eventIdToIndex;
    // metric values
    std::vector<MetricRecord> metricValues;
    // timestamp of last record
    // uint64_t lastTimestamp;
    // state
    int state;

    std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;

    std::atomic_flag semaphore = ATOMIC_FLAG_INIT;
  };

  // This is the subscriber callback which is invoked directly by CUPTI.
  // The 'userdata' argument will be a pointer to the active 'GPUTracerImpl'.
  static void CUPTIAPI ApiCallback(void *userdata, CUpti_CallbackDomain domain,
                                   CUpti_CallbackId cbid, const void *cbdata);

  // Records the mapping between correlation ID and kernel name.
  void AddCorrelationId(uint32 correlation_id, const string &name);

  // Returns the current system time in microseconds.
  inline int64 NowInUsec() { return Env::Default()->NowMicros(); }

  std::atomic_flag metricLock = ATOMIC_FLAG_INIT;

  CUPTIManager *cupti_manager_;
  std::unique_ptr<perftools::gputools::profiler::CuptiWrapper> cupti_wrapper_;
  CUpti_SubscriberHandle subscriber_;

  mutex trace_mu_;
  static constexpr size_t kMaxRecords = 1024 * 1024;
  std::map<uint32, string> correlations_ GUARDED_BY(trace_mu_);
  std::vector<KernelRecord> kernel_records_ GUARDED_BY(trace_mu_);
  std::vector<MemcpyRecord> memcpy_records_ GUARDED_BY(trace_mu_);
  // vector to store environment records. By HuJian
  std::vector<EnvironmentRecord> environment_records_ GUARDED_BY(trace_mu_);

  mutex mu_;
  bool enabled_ GUARDED_BY(mu_);
  int64 start_walltime_us_ GUARDED_BY(mu_);
  int64 end_walltime_us_ GUARDED_BY(mu_);
  uint64_t start_timestamp_ GUARDED_BY(mu_);
  uint64_t end_timestamp_ GUARDED_BY(mu_);

  // event data related to metric, by Hujian
  std::unordered_map<CUcontext, MetricData> metricDatas;

  int _sampleMetric = -1;
  bool sampleMetric() {
    if (_sampleMetric == -1) {
      char *s = getenv("hirp_sample_metric");
      if (s == nullptr)
        _sampleMetric = 0;
      else {
        if (std::string(s) == std::string("true"))
          _sampleMetric = 1;
        else
          _sampleMetric = 0;
      }
    }
    return _sampleMetric != 0;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(GPUTracerImpl);
};

GPUTracerImpl::GPUTracerImpl() {
  VLOG(1) << "GPUTracer created.";
  cupti_manager_ = GetCUPTIManager();
  CHECK(cupti_manager_);
  cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());
  enabled_ = false;
  _sampleMetric = -1;
}

GPUTracerImpl::~GPUTracerImpl() {
  // Unregister the CUPTI callbacks if needed to prevent them from accessing
  // freed memory.
  Stop().IgnoreError();
}

Status GPUTracerImpl::Start() {
  VLOG(1) << "GPUTracer::Start";
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("GPUTracer is already enabled.");
  }
  // There can only be one CUPTI subscriber.  If we can't create one then
  // there is another trace in progress (possibly by external code).
  CUptiResult ret;
  ret = cupti_wrapper_->Subscribe(
      &subscriber_, static_cast<CUpti_CallbackFunc>(ApiCallback), this);
  if (ret == CUPTI_ERROR_MAX_LIMIT_REACHED) {
    return errors::Unavailable("CUPTI subcriber limit reached.");
  } else if (ret != CUPTI_SUCCESS) {
    return errors::Internal("Failed to create CUPTI subcriber.");
  }

  // Register as a TraceEngine to receive ScopedAnnotations.
  port::Tracing::RegisterEngine(this);

  // Intercept launch and memcpy calls to capture the Op name annotation.
  // TODO(pbar) Add callbacks for memcpy variants.
  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel));
  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_RUNTIME_API,
                            CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020));
  CUPTI_CALL(EnableCallback(
      /*enable=*/1, subscriber_, CUPTI_CB_DOMAIN_RUNTIME_API,
      CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020));

  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2));
  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2));
  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2));
  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2));
  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2));
  CUPTI_CALL(EnableCallback(/*enable=*/1, subscriber_,
                            CUPTI_CB_DOMAIN_DRIVER_API,
                            CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2));

  //StartMetricSampling();

  TF_RETURN_IF_ERROR(cupti_manager_->EnableTrace(this));

  CUPTI_CALL(GetTimestamp(&start_timestamp_));
  start_walltime_us_ = NowInUsec();
  enabled_ = true;
  return Status::OK();
}

Status GPUTracerImpl::Stop() {
  VLOG(1) << "GPUTracer::Stop";
  mutex_lock l(mu_);
  if (!enabled_) {
    return Status::OK();
  }
  CUPTI_CALL(Unsubscribe(subscriber_));
  port::Tracing::RegisterEngine(nullptr);
  TF_RETURN_IF_ERROR(cupti_manager_->DisableTrace());
  end_walltime_us_ = NowInUsec();
  CUPTI_CALL(GetTimestamp(&end_timestamp_));
  enabled_ = false;
  return Status::OK();
}

void GPUTracerImpl::AddCorrelationId(uint32 correlation_id,
                                     const string &name) {
  VLOG(2) << correlation_id << " : " << name;
  mutex_lock l(trace_mu_);
  if (correlations_.size() >= kMaxRecords) return;
  correlations_.emplace(correlation_id, name);
}

/*static*/ void GPUTracerImpl::ApiCallback(void *userdata,
                                           CUpti_CallbackDomain domain,
                                           CUpti_CallbackId cbid,
                                           const void *cbdata) {
  auto *cbInfo = reinterpret_cast<const CUpti_CallbackData *>(cbdata);
  GPUTracerImpl *tracer = reinterpret_cast<GPUTracerImpl *>(userdata);
  VLOG(2) << "ApiCallback " << domain << ":" << cbid
          << " func: " << cbInfo->functionName;

  // API callbacks are invoked synchronously on the thread making the
  // CUDA API call.  If this pointer is non-null then the ScopedAnnotation
  // must be valid.
  const char *tls_annotation = tls_current_annotation.get();

  if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
      (cbid == CUPTI_DRIVER_TRACE_CBID_cuLaunchKernel)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      // metric sampling related code, by hujian
      // start metric sampling at every kernel launch
      if (tracer->sampleMetric()) {
        cudaDeviceSynchronize();  // wait for all work been done on GPU

        MetricData *metricDataP;
        // require lock, in case unordered_map accessed on different thread
        // tracer->metricMutex.lock();  
        while (tracer->metricLock.test_and_set(std::memory_order_acquire))
          ; // spin
        auto it = tracer->metricDatas.find(cbInfo->context);
        if (it != tracer->metricDatas.end()) {
          // a MetricData on this context already exists
          metricDataP = &(it->second);
        }
        else {
          tracer->metricDatas[cbInfo->context].Init(cbInfo->context);
          metricDataP = &(tracer->metricDatas[cbInfo->context]);
        }
        // release lock
        //tracer->metricMutex.unlock();  
        tracer->metricLock.clear(std::memory_order_release);
        if (metricDataP != nullptr)
          metricDataP->StartRecord();  // start record
      }
      // end of metric sampling code

      auto *params = reinterpret_cast<const cuLaunchKernel_params *>(
          cbInfo->functionParams);
      if (VLOG_IS_ON(2)) {
        VLOG(2) << "LAUNCH stream " << params->hStream << " correllation "
                << cbInfo->correlationId << " kernel " << cbInfo->symbolName;
      }
      const string annotation =
          tls_annotation ? tls_annotation : cbInfo->symbolName;
      tracer->AddCorrelationId(cbInfo->correlationId, annotation);
    }
    else if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      // metric sampling related code, by hujian
      if (tracer->sampleMetric()) {
        cudaDeviceSynchronize();  // wait for queued kernel to finish
        // if this is kernel exiting, end record
        //tracer->metricMutex.lock();
        while (tracer->metricLock.test_and_set(std::memory_order_acquire))
          ; // spin

        MetricData &metricData = tracer->metricDatas[cbInfo->context];
        //tracer->metricMutex.unlock();

        tracer->metricLock.clear(std::memory_order_release);
        metricData.EndRecord(cbInfo->correlationId);
      }
      // end of metric sampling code
    }
  } else if ((domain == CUPTI_CB_DOMAIN_RUNTIME_API) &&
             (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020 ||
              cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020)) {
    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      if (VLOG_IS_ON(2)) {
        auto *funcParams = reinterpret_cast<const cudaMemcpy_v3020_params *>(
            cbInfo->functionParams);
        size_t count = funcParams->count;
        enum cudaMemcpyKind kind = funcParams->kind;
        VLOG(2) << "MEMCPY count " << count << " kind " << kind;
      }
      if (tls_annotation) {
        const string annotation = tls_annotation;
        tracer->AddCorrelationId(cbInfo->correlationId, annotation);
      }
    }
  } else if ((domain == CUPTI_CB_DOMAIN_DRIVER_API) &&
             (cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoD_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoH_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoD_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyHtoDAsync_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoHAsync_v2 ||
              cbid == CUPTI_DRIVER_TRACE_CBID_cuMemcpyDtoDAsync_v2)) {
    if (cbInfo->callbackSite == CUPTI_API_EXIT && tls_annotation) {
      const string annotation = tls_annotation;
      tracer->AddCorrelationId(cbInfo->correlationId, annotation);
    }
  } else {
    VLOG(1) << "Unhandled API Callback for " << domain << " " << cbid;
  }
}

void GPUTracerImpl::ActivityCallback(const CUpti_Activity &record) {
  VLOG(2) << "ActivityCallback " << record.kind;
  mutex_lock l(trace_mu_);
  switch (record.kind) {
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      if (memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy *>(&record);
      memcpy_records_.push_back(MemcpyRecord{
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes});
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY2: {
      if (memcpy_records_.size() >= kMaxRecords) return;
      auto *memcpy = reinterpret_cast<const CUpti_ActivityMemcpy2 *>(&record);
      memcpy_records_.push_back(MemcpyRecord{
          memcpy->start, memcpy->end, memcpy->deviceId, memcpy->streamId,
          memcpy->correlationId, memcpy->copyKind, memcpy->srcKind,
          memcpy->dstKind, memcpy->bytes});
      break;
    }
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      if (kernel_records_.size() >= kMaxRecords) return;
      auto *kernel = reinterpret_cast<const CUpti_ActivityKernel3 *>(&record);
      kernel_records_.push_back(KernelRecord{kernel->start, kernel->end,
                                             kernel->deviceId, kernel->streamId,
                                             kernel->correlationId});

      // sample metric, by hujian
      if (sampleMetric()) {
        // iterate throw all MetricData samplers, find the one related this context
        MetricData *metricDataP = nullptr;
        while (metricLock.test_and_set(std::memory_order_acquire))
          ; // spin
        for (auto &item : metricDatas) {
          if (item.second.contextId == kernel->contextId) {
            metricDataP = &(item.second);
            break;
          }
        }
        metricLock.clear(std::memory_order_release);

        if (metricDataP != nullptr) {
          auto it = correlations_.find(kernel->correlationId);
          const std::string name = (it != correlations_.cend()) ? it->second : "unknown";
          metricDataP->ComputeMetric(kernel->start, kernel->end, 
                                     kernel->correlationId, name);
        }
      }
      break;
    }
    // handling environment kind activity. By HuJian
    case CUPTI_ACTIVITY_KIND_ENVIRONMENT: {
      if (environment_records_.size() >= kMaxRecords) return;
      auto *env = reinterpret_cast<const CUpti_ActivityEnvironment *>(&record);
      environment_records_.push_back(EnvironmentRecord());
      EnvironmentRecord &rec = environment_records_.back();
      rec.timestamp = env->timestamp;
      rec.device_id = env->deviceId;
      rec.environment_kind = env->environmentKind;
      switch (env->environmentKind) {
        // related to speed
        case 1: {
          rec.data.speed.smClock = env->data.speed.smClock;
          rec.data.speed.memoryClock = env->data.speed.memoryClock;
          rec.data.speed.pcieLinkGen = env->data.speed.pcieLinkGen;
          rec.data.speed.pcieLinkWidth = env->data.speed.pcieLinkWidth;
          rec.data.speed.clocksThrottleReasons = env->data.speed.clocksThrottleReasons;
          break;
        }
        // related to temparature
        case 2: {
          rec.data.temperature.gpuTemperature = env->data.temperature.gpuTemperature;
          break;
        }
        // related to power
        case 3: {
          rec.data.power.power = env->data.power.power;
          rec.data.power.powerLimit = env->data.power.powerLimit;
          break;
        }
        case 4: {
          rec.data.cooling.fanSpeed = env->data.cooling.fanSpeed;
          break;
        }
        default:{
          VLOG(1) << "unhandled environment kind";
        }
      }
      break;
    }
    default:
      VLOG(1) << "ActivityCallback unhandled kind";
      break;
  }
}

std::vector<std::string> SplitStr(const std::string &text, char sep) {
  std::vector<std::string> tokens;
  if (text.length() == 0)
    return tokens;
  
  std::size_t start = 0, end = 0;
  while ((end = text.find(sep, start)) != std::string::npos) {
    if (end != start)
      tokens.push_back(text.substr(start, end - start));
    start = end + 1;
  }
  if (end != start)
    tokens.push_back(text.substr(start));
  return tokens;
}

GPUTracerImpl::MetricData::MetricData(){
  cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());

  state = 0;
  passes = 1;
  sets = nullptr;
  metricIds = nullptr;
}

GPUTracerImpl::MetricData::MetricData(CUcontext context) {
  cupti_wrapper_.reset(new perftools::gputools::profiler::CuptiWrapper());

  state = 0;
  passes = 1;
  sets = nullptr;
  Init(context);
}

void GPUTracerImpl::MetricData::Init(CUcontext context) {
  //std::ofstream file("/home/jian/Documents/RunData/2018-1-21/metrics.log", std::ios::trunc);
  //file << "trying to start metric sampling" << std::endl;
  // get the device realted to this context
  this->context = context;
  uint32_t deviceId = 0;
  CUPTI_CALL(GetDeviceId(context, &deviceId));
  device = deviceId;
  // get the context id
  CUPTI_CALL(cuptiGetContextId(context, &contextId));
  // set kernel replay mode on
  // there is no other choice, since we can't rerun the application,
  // and still get the correlations between different kernels

  // get metric names and numbers
  char *c_str = getenv("hirp_metrics");
  if (c_str == nullptr || *c_str == '\0') {
    state = 2;
    return;
  }

  metrics = SplitStr(std::string(c_str), ',');
  numMetrics = metrics.size();

  // allocate space to store metric Ids, and query metric ids
  metricIds = new CUpti_MetricID[numMetrics];
  for (std::size_t i = 0; i != numMetrics; ++i) {
    std::string metric = metrics[i];
    // first get metric idCUpti_MetricID
    CUpti_MetricID metricId = 0;
    CUPTI_CALL(cuptiMetricGetIdFromName(device, metric.c_str(), &metricId));
    metricIds[i] = metricId;
  }

  {
    // hack to get how many replays needed to get those metrics
    CUPTI_CALL(cuptiDisableKernelReplayMode(context));
    CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(CUpti_MetricID) * numMetrics,
                                             metricIds, &sets));
    passes = sets->numSets;
    CUPTI_CALL(cuptiEventGroupSetsDestroy(sets));
    sets = nullptr;
  }

  // create the event group sets required to compute the metrics,
  // this also indicates how many passes will be needed
  CUPTI_CALL(cuptiEnableKernelReplayMode(context));
  CUPTI_CALL(cuptiMetricCreateEventGroupSets(context, sizeof(CUpti_MetricID) * numMetrics,
                                             metricIds, &sets));
  if (sets->numSets > 1) {
    //file << "can't collect those metrics in one run, absort" << std::endl;
    //cout << "can't collect those metrics in one run, absort";
    state = 2;
    CUPTI_CALL(cuptiDisableKernelReplayMode(context));
    // maybe do clean up here?
  }

  eventGroups = sets->sets;

  // iterate throw all groups to count all events needed, and store event Ids
  numEvents = 0;
  uint32_t maxNumInstances = 0;
  // resize numInstances and numTotalInstances vector
  numInstances.resize(eventGroups->numEventGroups);
  numTotalInstances.resize(eventGroups->numEventGroups);

  for (std::size_t i = 0; i < eventGroups->numEventGroups; ++i) {
    CUpti_EventGroup group = eventGroups->eventGroups[i];

    // enable event group
    uint32 all = 1;
    CUPTI_CALL(cuptiEventGroupSetAttribute(eventGroups->eventGroups[i],
                                           CUPTI_EVENT_GROUP_ATTR_PROFILE_ALL_DOMAIN_INSTANCES,
                                           sizeof(all), &all));
    CUPTI_CALL(cuptiEventGroupEnable(eventGroups->eventGroups[i]));

    CUpti_EventDomainID groupDomain;
    uint32_t numEventsG;
    CUpti_EventID *eventIds;
    std::size_t groupDomainSize = sizeof(groupDomain);
    std::size_t numEventsSize = sizeof(numEventsG);
    std::size_t numInstancesSize = sizeof(uint32_t);
    std::size_t numTotalInstancesSize = sizeof(uint32_t);
    std::size_t eventIdsSize;

    CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENT_DOMAIN_ID,
                                           &groupDomainSize, &groupDomain));
    CUPTI_CALL(cuptiDeviceGetEventDomainAttribute(device, groupDomain,
                                           CUPTI_EVENT_DOMAIN_ATTR_TOTAL_INSTANCE_COUNT,
                                           &numTotalInstancesSize, &numTotalInstances[i]));
    CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_INSTANCE_COUNT,
                                           &numInstancesSize, &numInstances[i]));
    CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_NUM_EVENTS,
                                           &numEventsSize, &numEventsG));

    numEvents += numEventsG;

    maxNumInstances = maxNumInstances > numInstances[i] ? maxNumInstances : numInstances[i];

    // acquire event ids in this group.
    eventIdsSize = numEventsG * sizeof(CUpti_EventID);
    eventIds = new CUpti_EventID[numEventsG];
    CUPTI_CALL(cuptiEventGroupGetAttribute(group, CUPTI_EVENT_GROUP_ATTR_EVENTS,
                                           &eventIdsSize, eventIds));
    // store event ids
    for (std::size_t j = 0; j < numEventsG; ++j) {
      eventIdArray.push_back(eventIds[j]);
      eventIdxToGroupId.push_back(i);  // record to which group this event belongs.
    }
  }

  // allocate event value buffer
  valuesBuffer.reset(new uint64_t[maxNumInstances]);

  // set event coolection mode
  //CUPTI_CALL(cuptiSetEventCollectionMode(context, CUPTI_EVENT_COLLECTION_MODE_KERNEL));

  // clear the vector, in case of other data stored
  metricValues.clear();
  
  state = 1;
}

GPUTracerImpl::MetricData::~MetricData() {
  if (state == 1) {
    delete[] metricIds;
    metricIds = nullptr;

    // disable all groups
    for (std::size_t i = 0; i < eventGroups->numEventGroups; ++i) {
      CUpti_EventGroup group = eventGroups->eventGroups[i];
      CUPTI_CALL(cuptiEventGroupDisable(group));
    }

    CUPTI_CALL(cuptiDisableKernelReplayMode(context));
  }
  // destroy event group sets
  if (sets != nullptr) {
    CUPTI_CALL(cuptiEventGroupSetsDestroy(sets));
  }
}

int GPUTracerImpl::MetricData::StartRecord() {
  if (state == 1) {
    // only one kernel is allowed to run at once when sampling metric
    // so acquire lock
    while (semaphore.test_and_set(std::memory_order_acquire))
      ;  // spin
    return 0;
  }
  else {
    //std::ofstream file("/home/jian/Documents/RunData/2018-1-21/metrics.log", std::ios_base::app);
    //file << "metric sampling data is not inited" << std::endl;
    return 2;
  }
}

int GPUTracerImpl::MetricData::EndRecord(uint32_t correlation_id) {
  //std::ofstream file("/home/jian/Documents/RunData/2018-1-21/metrics.log", std::ios_base::app);
  //file << "metric sampling end record started" << std::endl;
  if (state != 1) {
    //file << "metric sampling data is not inited" << std::endl;
    return -1;
  }

  std::unique_ptr<uint64_t[]> buffer(new uint64_t[numEvents]);
  // iterate through all events.
  for (std::size_t idx = 0; idx < numEvents; ++idx) {
    std::size_t gIdx = eventIdxToGroupId[idx];
    CUpti_EventGroup group = eventGroups->eventGroups[gIdx];
    uint64_t normalized, sum;
    std::size_t valuesSize = numInstances[gIdx] * sizeof(uint64_t);  // the buffer is at least larger than this
    // read event values
    CUPTI_CALL(cuptiEventGroupReadEvent(group, CUPTI_EVENT_READ_FLAG_NONE,
                                        this->eventIdArray[idx], 
                                        &valuesSize, valuesBuffer.get()));
    sum = 0;
    for (uint32 k = 0; k < numInstances[gIdx]; ++k) {
      sum += valuesBuffer[k];
    }

    normalized = (sum * numTotalInstances[gIdx]) / numInstances[gIdx];

    // save id and normalized value 
    buffer[idx] = normalized;
  }

  semaphore.clear(std::memory_order_release);  // after all event data has been read, release the lock.

  // protect from racing conditions for accessing savedEvents map.
  while (eventMapLock.test_and_set(std::memory_order_acquire))
    ;  // spin
  auto &_pair = savedEvents[correlation_id];

  std::unique_ptr<CUpti_EventID[]> &eventIdArray = _pair.first;
  eventIdArray.reset(new CUpti_EventID[numEvents]);
  std::unique_ptr<uint64_t[]> &eventValueArray = _pair.second;
  eventValueArray.reset(new uint64_t[numEvents]);
  for (std::size_t idx = 0; idx < numEvents; ++idx) {
    eventIdArray[idx] = this->eventIdArray[idx];  // maybe try to
    eventValueArray[idx] = buffer[idx];
  }

  // release lock
  eventMapLock.clear(std::memory_order_release);

  /*
  for (std::size_t i = 0; i < eventGroups->numEventGroups; ++i) {
    CUpti_EventGroup group = eventGroups->eventGroups[i];
    CUPTI_CALL(cuptiEventGroupResetAllEvents(group));
  }*/

  return 0;
}

void GPUTracerImpl::MetricData::ComputeMetric(uint64_t start, uint64_t end, 
                                  uint32_t correlation_id, const std::string &annotate) {
  uint64_t duration = end - start;
  
  MetricRecord record;
  record.start = start;
  record.end = end;
  record.name = annotate;

  // compute and save metric data
  // use all the collected events to calculate the metric values
  record.Values.resize(numMetrics);

  while (eventMapLock.test_and_set(std::memory_order_acquire))
    ;  // spin

  auto it = savedEvents.find(correlation_id);
  if (it == savedEvents.end()) {
    // maybe the event values is lost, or have not been writen
    return;
  }

  auto &_pair = it->second;

  eventMapLock.clear(std::memory_order_release);

  std::unique_ptr<CUpti_EventID[]> &eventIdArray = _pair.first;
  std::unique_ptr<uint64_t[]> &eventValueArray = _pair.second;

  for (std::size_t i = 0; i < numMetrics; ++i) {
    CUPTI_CALL(cuptiMetricGetValue(device, metricIds[i],
                                   numEvents * sizeof(CUpti_EventID),
                                   eventIdArray.get(),
                                   numEvents * sizeof(uint64),
                                   eventValueArray.get(),
                                   duration / passes,
                                   &record.Values[i]));
    //file << record.Values[i] << std::endl;
  }
  // remove this event data
  while (eventMapLock.test_and_set(std::memory_order_acquire))
    ;  // spin
  savedEvents.erase(correlation_id);
  eventMapLock.clear(std::memory_order_release);

  metricValues.push_back(record);
}

void GPUTracerImpl::MetricData::ToCSV(std::ofstream &out, bool writeHeader) {
  if (out) {
    if (writeHeader) {
      out << "device_id" << ", context_id" << ", kernel_start" << ", kernel_end"
          << ", name";
      for (const string &metricName : metrics){
        out << ", " << metricName;
      }
      out << std::endl;
    }

    // read out all metric kinds
    std::unique_ptr<CUpti_MetricValueKind[]> kinds(new CUpti_MetricValueKind[numMetrics]);
    for (std::size_t i = 0; i < numMetrics; ++i) {
      std::size_t valueKindSize = sizeof(CUpti_MetricValueKind);
      CUPTI_CALL(cuptiMetricGetAttribute(metricIds[i], CUPTI_METRIC_ATTR_VALUE_KIND,
                                         &valueKindSize, &kinds[i]));
    }

    // now iterate through all metric records, and write them
    for (const MetricRecord &record : metricValues) {
      out << device << ", " << contextId
          << ", " << record.start << ", " << record.end
          << ", " << record.name;
      // iterate through all metrics
      const std::vector<CUpti_MetricValue> &values = record.Values;
      for (std::size_t i = 0; i < numMetrics; ++i) {
        out << ", ";
        switch (kinds[i]) {
        case CUPTI_METRIC_VALUE_KIND_DOUBLE:
          out << values[i].metricValueDouble;
          break;
        case CUPTI_METRIC_VALUE_KIND_UINT64:
          out << values[i].metricValueUint64;
          break;
        case CUPTI_METRIC_VALUE_KIND_INT64:
          out << values[i].metricValueInt64;
          break;
        case CUPTI_METRIC_VALUE_KIND_PERCENT:
          out << values[i].metricValuePercent;
          break;
        case CUPTI_METRIC_VALUE_KIND_THROUGHPUT:
          out << values[i].metricValueThroughput;
          break;
        case CUPTI_METRIC_VALUE_KIND_UTILIZATION_LEVEL:
          out << values[i].metricValueUtilizationLevel;
          break;
        default:
          out << "NAN";
        }
      }
      out << std::endl;
    }
  }
}

Status GPUTracerImpl::Collect(StepStatsCollector *collector) {
  mutex_lock l(mu_);
  if (enabled_) {
    return errors::FailedPrecondition("GPUTracer is still enabled.");
  }

  // TODO(pbar) Handle device IDs and prefix properly.
  const string prefix = "";
  const int id = 0;
  const string stream_device = strings::StrCat(prefix, "/device:GPU:", id, "/stream:");
  const string memcpy_device = strings::StrCat(prefix, "/device:GPU:", id, "/memcpy");

  mutex_lock l2(trace_mu_);
  for (const auto &rec : kernel_records_) {
    auto it = correlations_.find(rec.correlation_id);
    const string name = (it != correlations_.cend()) ? it->second : "unknown";
    NodeExecStats *ns = new NodeExecStats;
    ns->set_all_start_micros(start_walltime_us_ +
                             ((rec.start_timestamp - start_timestamp_) / 1000));
    ns->set_op_start_rel_micros(0);
    auto elapsed_us =
        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
    ns->set_op_end_rel_micros(elapsed_us);
    ns->set_all_end_rel_micros(elapsed_us);
    ns->set_node_name(name);
    // TODO(pbar) Generate details based on the kernel activity record.
    // ns->set_timeline_label(details);
    auto nscopy = new NodeExecStats;
    *nscopy = *ns;
    collector->Save(strings::StrCat(stream_device, "all"), ns);
    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
  }
  for (const auto &rec : memcpy_records_) {
    auto it = correlations_.find(rec.correlation_id);
    const string name = (it != correlations_.cend()) ? it->second : "unknown";
    NodeExecStats *ns = new NodeExecStats;
    ns->set_all_start_micros(start_walltime_us_ +
                             ((rec.start_timestamp - start_timestamp_) / 1000));
    ns->set_op_start_rel_micros(0);
    auto elapsed_us =
        std::max<int64>((rec.end_timestamp - rec.start_timestamp) / 1000, 1);
    ns->set_op_end_rel_micros(elapsed_us);
    ns->set_all_end_rel_micros(elapsed_us);
    auto copyKind = static_cast<CUpti_ActivityMemcpyKind>(rec.copyKind);
    auto srcKind = static_cast<CUpti_ActivityMemoryKind>(rec.srcKind);
    auto dstKind = static_cast<CUpti_ActivityMemoryKind>(rec.dstKind);
    const string details = strings::Printf(
        "MEMCPY%s %llu bytes (%s to %s)", getMemcpyKindString(copyKind),
        rec.bytes, getMemoryKindString(srcKind), getMemoryKindString(dstKind));
    ns->set_node_name(
        strings::StrCat(name, ":MEMCPY", getMemcpyKindString(copyKind)));
    ns->set_timeline_label(details);
    auto nscopy = new NodeExecStats;
    *nscopy = *ns;
    collector->Save(memcpy_device, ns);
    collector->Save(strings::StrCat(stream_device, rec.stream_id), nscopy);
  }

  // directly write environment data to file. By HuJian
  // TODO: find a better way to pass those data to higher layers.
  {
    char *c_strDir = getenv("hirp_rundata_dir");
    char *c_strFN = getenv("hirp_env_data_fn");
    if (c_strDir != nullptr && c_strFN != nullptr) {
      std::string saveDir(c_strDir);
      std::string envDataFN(c_strFN);
      if (!saveDir.empty() && !envDataFN.empty()) {
        // this works with linux, but not sure when on windows
        std::string envDataFP(saveDir + "/"+ envDataFN);

        std::ofstream file(envDataFP, std::ios::trunc);
        if (file) { // if file can be open write
          // write header
          file << "deviceId" << ", timestamp(ns)" << ", environmentKind"
              << ", smClock(MHz)" << ", memoryClock(MHz)" << ", pcieLinkGen"
              << ", pcieLinkWidth" << ", clocksThrotteReasons"
              << ", gpuTemperature" << ", power(mW)" << ", powerLimit(mW)"
              << ", fanSpeed(%)" << std::endl;
          for (const EnvironmentRecord& rec : environment_records_) {
            file << rec.device_id << ", " << rec.timestamp << ", "
                << rec.environment_kind;
            // if this environment activity record is related to speed
            if (rec.environment_kind == 1) {
              const auto & speed = rec.data.speed;
              file << ", " << speed.smClock << ", " << speed.memoryClock
                  << ", " << speed.pcieLinkGen << ", " << speed.pcieLinkWidth
                  << ", " << speed.clocksThrottleReasons;
            }
            else
              file << ", , , , , ";
            
            // if this environment activity record is related to temperature
            if (rec.environment_kind == 2) {
              file << ", " << rec.data.temperature.gpuTemperature;
            }
            else
              file << ", ";

            // if this environment activity record is related to power
            if (rec.environment_kind == 3) {
              const auto & power = rec.data.power;
              file << ", " << power.power << ", " << power.powerLimit;
            }
            else
              file << ", , ";

            // if this environment activity record is related to cooling
            if (rec.environment_kind == 4) {
              file << ", " << rec.data.cooling.fanSpeed;
            }
            else
              file << ", ";

            file << '\n';
          }
          // flush the file.
          file.flush();
        }  // end if file opend
      }
    }  // end if env variables exists
  }  // end collecting environment activity data

  // write metric data to file
  {
    char *c_strDir = getenv("hirp_rundata_dir");
    char *c_strFN = getenv("hirp_metric_data_fn");
    if (sampleMetric() && c_strDir != nullptr && c_strFN != nullptr) {
      std::string saveDir(c_strDir);
      std::string envDataFN(c_strFN);
      if (!saveDir.empty() && !envDataFN.empty()) {
        std::string metricDataFP(saveDir + '/' + c_strFN);
        std::ofstream file(metricDataFP, std::ios::trunc);

        if (file){
          bool first = true;
          for (auto &item : metricDatas) {
            MetricData &data = item.second;
            data.ToCSV(file, first);
            first = false;
          }

          metricDatas.clear();  // delete all metric sampling object, release resources
        }
      }
    }
  }

  return Status::OK();
}

}  // namespace gputracer

std::unique_ptr<GPUTracer> CreateGPUTracer() {
  std::unique_ptr<GPUTracer> tracer(new gputracer::GPUTracerImpl());
  return tracer;
}

}  // namespace tensorflow

#else  // GOOGLE_CUDA

namespace tensorflow {

std::unique_ptr<GPUTracer> CreateGPUTracer() { return nullptr; }

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
