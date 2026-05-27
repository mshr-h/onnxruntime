// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/llm/flex_attention.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "core/common/common.h"
#include "core/common/safeint.h"
#include "core/framework/op_kernel_context_internal.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/framework/utils.h"
#include "core/graph/graph_viewer.h"
#include "core/mlas/inc/mlas.h"
#include "core/platform/threadpool.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"

namespace onnxruntime {

#define REGISTER_FLEX_ATTENTION_TYPED(T)                                                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                                            \
      FlexAttention,                                                                         \
      kOnnxPreviewDomain,                                                                    \
      1,                                                                                     \
      T,                                                                                     \
      kCpuExecutionProvider,                                                                 \
      (*KernelDefBuilder::Create()).TypeConstraint("T1", DataTypeImpl::GetTensorType<T>()), \
      FlexAttention<T>);

REGISTER_FLEX_ATTENTION_TYPED(float)
REGISTER_FLEX_ATTENTION_TYPED(MLFloat16)

namespace {

int64_t DimAt(const TensorShape& shape, size_t dim) {
  return shape.GetDims()[dim];
}

Status ValidateInputs(const Tensor& q, const Tensor& k, const Tensor& v) {
  ORT_RETURN_IF_NOT(q.Shape().NumDimensions() == 4 && k.Shape().NumDimensions() == 4 && v.Shape().NumDimensions() == 4,
                    "FlexAttention CPU kernel requires rank-4 Q, K, V tensors.");
  ORT_RETURN_IF_NOT(DimAt(q.Shape(), 0) == DimAt(k.Shape(), 0) && DimAt(q.Shape(), 0) == DimAt(v.Shape(), 0),
                    "FlexAttention Q, K, V batch dimensions must match.");
  ORT_RETURN_IF_NOT(DimAt(k.Shape(), 1) == DimAt(v.Shape(), 1),
                    "FlexAttention K and V head counts must match.");
  ORT_RETURN_IF_NOT(DimAt(k.Shape(), 2) == DimAt(v.Shape(), 2),
                    "FlexAttention K and V sequence lengths must match.");
  ORT_RETURN_IF_NOT(DimAt(q.Shape(), 3) == DimAt(k.Shape(), 3),
                    "FlexAttention Q and K head sizes must match.");
  ORT_RETURN_IF_NOT(DimAt(q.Shape(), 1) > 0 && DimAt(k.Shape(), 1) > 0 && DimAt(q.Shape(), 1) % DimAt(k.Shape(), 1) == 0,
                    "FlexAttention q_num_heads must be a positive multiple of kv_num_heads.");
  return Status::OK();
}

void GemmFloat(CBLAS_TRANSPOSE trans_a, CBLAS_TRANSPOSE trans_b,
               int64_t m, int64_t n, int64_t k,
               float alpha, const float* a, int lda, const float* b, int ldb,
               float beta, float* c, int ldc,
               const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* selector) {
  math::GemmEx<float, concurrency::ThreadPool>(trans_a, trans_b,
                                               static_cast<ptrdiff_t>(m), static_cast<ptrdiff_t>(n), static_cast<ptrdiff_t>(k),
                                               alpha, a, lda, b, ldb, beta, c, ldc, nullptr, selector);
}

void ConvertHalfToFloat(const MLFloat16* src, float* dst, size_t count) {
  MlasConvertHalfToFloatBuffer(src, dst, count);
}

void ConvertFloatToHalf(const float* src, MLFloat16* dst, size_t count) {
  MlasConvertFloatToHalfBuffer(src, dst, count);
}

}  // namespace

template <typename T>
FlexAttention<T>::FlexAttention(const OpKernelInfo& info) : controlflow::IControlFlowKernel(info) {
  SetupMlasBackendKernelSelectorFromConfigOptions(mlas_backend_kernel_selector_config_, info.GetConfigOptions());

  const auto& attrs = info.node().GetAttributes();
  has_score_mod_ = attrs.find("score_mod") != attrs.end();
  ORT_ENFORCE(attrs.find("prob_mod") == attrs.end(),
              "ai.onnx.preview::FlexAttention CPU kernel does not support prob_mod yet.");

  scale_ = info.GetAttrOrDefault<float>("scale", std::numeric_limits<float>::quiet_NaN());
  softmax_precision_ = info.GetAttrOrDefault<int64_t>("softmax_precision", ONNX_NAMESPACE::TensorProto_DataType_FLOAT);
  ORT_ENFORCE(softmax_precision_ == ONNX_NAMESPACE::TensorProto_DataType_FLOAT,
              "ai.onnx.preview::FlexAttention CPU kernel currently supports only softmax_precision=float.");
}

template <typename T>
Status FlexAttention<T>::SetupSubgraphExecutionInfo(const SessionState& /*session_state*/,
                                                    const std::string& attribute_name,
                                                    const SessionState& subgraph_session_state) {
  ORT_RETURN_IF_NOT(attribute_name == "score_mod",
                    "Unsupported FlexAttention subgraph attribute: ", attribute_name);
  ORT_RETURN_IF_NOT(score_mod_info_ == nullptr && score_mod_feeds_fetches_manager_ == nullptr,
                    "FlexAttention score_mod execution info should only be initialized once.");

  const auto& graph = subgraph_session_state.GetGraphViewer();
  const auto& inputs = graph.GetInputs();
  const auto& outputs = graph.GetOutputs();
  ORT_RETURN_IF_NOT(inputs.size() == 1 && outputs.size() == 1,
                    "FlexAttention score_mod subgraph must have exactly one input and one output.");

  auto info = std::make_unique<FlexAttentionScoreModInfo>();
  info->input_names.push_back(inputs[0]->Name());
  info->output_names.push_back(outputs[0]->Name());

  std::unique_ptr<FeedsFetchesManager> ffm;
  ORT_RETURN_IF_ERROR(FeedsFetchesManager::Create(info->input_names, info->output_names,
                                                  subgraph_session_state.GetOrtValueNameIdxMap(), ffm));
  ORT_RETURN_IF_ERROR(utils::InitializeFeedFetchCopyInfo(subgraph_session_state, *ffm));

  const OrtDevice cpu_device;
  std::vector<OrtDevice> feed_locations{cpu_device};
  std::vector<const OrtDevice*> fetch_locations{&cpu_device};
  utils::FinalizeFeedFetchCopyInfo(*ffm, feed_locations, fetch_locations);

  score_mod_info_ = std::move(info);
  score_mod_feeds_fetches_manager_ = std::move(ffm);
  return Status::OK();
}

template <typename T>
Status FlexAttention<T>::ExecuteScoreMod(OpKernelContext* context, const TensorShape& score_shape,
                                         std::vector<float>& scores) const {
  if (!has_score_mod_) {
    return Status::OK();
  }

  ORT_RETURN_IF_NOT(score_mod_info_ && score_mod_feeds_fetches_manager_,
                    "FlexAttention score_mod subgraph execution info is not initialized.");

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  OrtValue feed;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), score_shape, allocator, feed);
  auto* feed_tensor = feed.GetMutable<Tensor>();
  std::memcpy(feed_tensor->MutableData<float>(), scores.data(), scores.size() * sizeof(float));

  std::vector<OrtValue> feeds;
  feeds.push_back(feed);
  std::vector<OrtValue> fetches(1);
  std::unordered_map<size_t, IExecutor::CustomAllocator> fetch_allocators;

  auto* ctx_internal = static_cast<OpKernelContextInternal*>(context);
  const SessionState* score_mod_session_state = ctx_internal->SubgraphSessionState("score_mod");
  ORT_RETURN_IF_NOT(score_mod_session_state != nullptr,
                    "FlexAttention score_mod subgraph SessionState was not found.");

  ORT_RETURN_IF_ERROR(utils::ExecuteSubgraph(*score_mod_session_state,
                                             *score_mod_feeds_fetches_manager_,
                                             feeds,
                                             fetches,
                                             fetch_allocators,
                                             ExecutionMode::ORT_SEQUENTIAL,
                                             ctx_internal->GetTerminateFlag(),
                                             ctx_internal->Logger(),
                                             ctx_internal->GetComputeStream(),
                                             /*sync_subgraph_fetches*/ true,
                                             ctx_internal->GetRunProfiler()));

  ORT_RETURN_IF_NOT(fetches.size() == 1 && fetches[0].IsTensor(),
                    "FlexAttention score_mod must return one tensor.");
  const auto& output = fetches[0].Get<Tensor>();
  ORT_RETURN_IF_NOT(output.IsDataType<float>(),
                    "FlexAttention score_mod output must have float element type.");
  ORT_RETURN_IF_NOT(output.Shape() == score_shape,
                    "FlexAttention score_mod output shape must match input score shape.");

  std::memcpy(scores.data(), output.Data<float>(), scores.size() * sizeof(float));
  return Status::OK();
}

template <typename T>
Status FlexAttention<T>::Compute(OpKernelContext* context) const {
  const Tensor* q = context->Input<Tensor>(0);
  const Tensor* k = context->Input<Tensor>(1);
  const Tensor* v = context->Input<Tensor>(2);
  ORT_RETURN_IF_NOT(q != nullptr && k != nullptr && v != nullptr,
                    "FlexAttention requires Q, K, V inputs.");
  ORT_RETURN_IF_ERROR(ValidateInputs(*q, *k, *v));

  const int64_t batch = DimAt(q->Shape(), 0);
  const int64_t q_heads = DimAt(q->Shape(), 1);
  const int64_t q_seq = DimAt(q->Shape(), 2);
  const int64_t head_size = DimAt(q->Shape(), 3);
  const int64_t kv_heads = DimAt(k->Shape(), 1);
  const int64_t kv_seq = DimAt(k->Shape(), 2);
  const int64_t v_head_size = DimAt(v->Shape(), 3);
  const float scale = std::isnan(scale_) ? 1.0f / std::sqrt(static_cast<float>(head_size)) : scale_;

  TensorShape output_shape{batch, q_heads, q_seq, v_head_size};
  Tensor* y = context->Output(0, output_shape);

  const size_t score_count = SafeInt<size_t>(batch) * q_heads * q_seq * kv_seq;
  std::vector<float> scores(score_count);

  const T* q_data = q->Data<T>();
  const T* k_data = k->Data<T>();
  const T* v_data = v->Data<T>();
  T* y_data = y->MutableData<T>();

  const size_t q_matrix_size = SafeInt<size_t>(q_seq) * head_size;
  const size_t k_matrix_size = SafeInt<size_t>(kv_seq) * head_size;
  const size_t v_matrix_size = SafeInt<size_t>(kv_seq) * v_head_size;

  std::vector<float> q_buffer;
  std::vector<float> k_buffer;
  if constexpr (std::is_same<T, MLFloat16>::value) {
    q_buffer.resize(q_matrix_size);
    k_buffer.resize(k_matrix_size);
  }

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < q_heads; ++h) {
      const int64_t kv_h = h * kv_heads / q_heads;
      const T* q_head = q_data + ((b * q_heads + h) * q_seq * head_size);
      const T* k_head = k_data + ((b * kv_heads + kv_h) * kv_seq * head_size);
      float* score_head = scores.data() + ((b * q_heads + h) * q_seq * kv_seq);

      const float* q_float = nullptr;
      const float* k_float = nullptr;
      if constexpr (std::is_same<T, float>::value) {
        q_float = q_head;
        k_float = k_head;
      } else {
        ConvertHalfToFloat(q_head, q_buffer.data(), q_matrix_size);
        ConvertHalfToFloat(k_head, k_buffer.data(), k_matrix_size);
        q_float = q_buffer.data();
        k_float = k_buffer.data();
      }

      GemmFloat(CblasNoTrans, CblasTrans,
                q_seq, kv_seq, head_size,
                scale, q_float, static_cast<int>(head_size),
                k_float, static_cast<int>(head_size),
                0.0f, score_head, static_cast<int>(kv_seq),
                &mlas_backend_kernel_selector_config_);
    }
  }

  TensorShape score_shape{batch, q_heads, q_seq, kv_seq};
  ORT_RETURN_IF_ERROR(ExecuteScoreMod(context, score_shape, scores));

  auto* tp = context->GetOperatorThreadPool();
  MlasComputeSoftmax(scores.data(), scores.data(), SafeInt<size_t>(batch) * q_heads * q_seq,
                     static_cast<size_t>(kv_seq), false, false, 0.0f, tp);

  std::vector<float> v_buffer;
  std::vector<float> y_buffer;
  if constexpr (std::is_same<T, MLFloat16>::value) {
    v_buffer.resize(v_matrix_size);
    y_buffer.resize(SafeInt<size_t>(q_seq) * v_head_size);
  }

  for (int64_t b = 0; b < batch; ++b) {
    for (int64_t h = 0; h < q_heads; ++h) {
      const int64_t kv_h = h * kv_heads / q_heads;
      const float* probs_head = scores.data() + ((b * q_heads + h) * q_seq * kv_seq);
      const T* v_head = v_data + ((b * kv_heads + kv_h) * kv_seq * v_head_size);
      T* y_head = y_data + ((b * q_heads + h) * q_seq * v_head_size);

      const float* v_float = nullptr;
      float* y_float = nullptr;
      if constexpr (std::is_same<T, float>::value) {
        v_float = v_head;
        y_float = y_head;
      } else {
        ConvertHalfToFloat(v_head, v_buffer.data(), v_matrix_size);
        v_float = v_buffer.data();
        y_float = y_buffer.data();
      }

      GemmFloat(CblasNoTrans, CblasNoTrans,
                q_seq, v_head_size, kv_seq,
                1.0f, probs_head, static_cast<int>(kv_seq),
                v_float, static_cast<int>(v_head_size),
                0.0f, y_float, static_cast<int>(v_head_size),
                &mlas_backend_kernel_selector_config_);

      if constexpr (std::is_same<T, MLFloat16>::value) {
        ConvertFloatToHalf(y_buffer.data(), y_head, SafeInt<size_t>(q_seq) * v_head_size);
      }
    }
  }

  return Status::OK();
}

template class FlexAttention<float>;
template class FlexAttention<MLFloat16>;

}  // namespace onnxruntime
