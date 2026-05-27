// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <vector>

#include "core/framework/feeds_fetches_manager.h"
#include "core/providers/cpu/controlflow/utils.h"
#include "core/providers/cpu/mlas_backend_kernel_selector_config_utils.h"

namespace onnxruntime {

class SessionState;

struct FlexAttentionScoreModInfo {
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

template <typename T>
class FlexAttention final : public controlflow::IControlFlowKernel {
 public:
  explicit FlexAttention(const OpKernelInfo& info);

  Status SetupSubgraphExecutionInfo(const SessionState& session_state,
                                    const std::string& attribute_name,
                                    const SessionState& subgraph_session_state) override;

  Status Compute(OpKernelContext* context) const override;

 private:
  Status ExecuteScoreMod(OpKernelContext* context, const TensorShape& score_shape,
                         std::vector<float>& scores) const;

  bool has_score_mod_ = false;
  float scale_ = 0.0f;
  int64_t softmax_precision_ = 0;
  std::unique_ptr<FlexAttentionScoreModInfo> score_mod_info_;
  std::unique_ptr<FeedsFetchesManager> score_mod_feeds_fetches_manager_;
  MLAS_BACKEND_KERNEL_SELECTOR_CONFIG mlas_backend_kernel_selector_config_;
};

}  // namespace onnxruntime
