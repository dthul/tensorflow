/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/constant_folding.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/tools/graph_transforms/fold_constants_lib.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace graph_transforms {

Status SwapTransMulAdd(const GraphDef& input_graph_def,
                       const TransformFuncContext& context,
                       GraphDef* output_graph_def) {
  // Make sure we don't get rid of any nodes used as graph inputs or outputs.
  std::set<string> required_nodes;
  for (const string& input : context.input_names) {
    required_nodes.insert(NodeNameFromInput(input));
  }
  for (const string& output : context.output_names) {
    required_nodes.insert(NodeNameFromInput(output));
  }
  // LOG(INFO) << "Here 1";

  std::map<string, string> inputs_to_rename;
  std::unordered_set<string> nodes_to_ignore;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"AddV2",
        {
          {"Mul",
            {
              {"Transpose",
                {
                  {"*"},     // input_node
                  {"Const"}  // permutation
                }
              },
              {"Const"},     // weights
            }
          },
          {"Const"},         // bias
        }
      },  // clang-format on
      [&inputs_to_rename, &nodes_to_ignore, &required_nodes](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& add_node = match.node;
        // if (!add_node.attr().contains("fused_activation_function")) {
        //   // Return original nodes.
        //   // LOG(INFO) << "Skipping replacement for " << add_node.name();
        //   CopyOriginalMatch(match, new_nodes);
        //   return Status::OK();
        // }
        // LOG(INFO) << "Here 2" << match.node.name();
        // LOG(INFO) << add_node.attr().at("fused_activation_function").s();
        const NodeDef& bias_node = match.inputs[1].node;  // Const
        const NodeDef& mul_node = match.inputs[0].node;
        const NodeDef& weights_node = match.inputs[0].inputs[1].node;  // Const
        const NodeDef& transpose_node = match.inputs[0].inputs[0].node;
        const NodeDef& perm_node =
            match.inputs[0].inputs[0].inputs[1].node;  // Const
        const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;

        // Check that nodes that we use are not used somewhere else.
        for (const auto& node : {add_node, bias_node, mul_node, weights_node,
                                 transpose_node, perm_node}) {
          if (required_nodes.count(node.name())) {
            // Return original nodes.
            LOG(INFO) << "Skipping replacement for " << node.name();
            CopyOriginalMatch(match, new_nodes);
            return Status::OK();
          }
        }
        Tensor perm = GetNodeTensorAttr(perm_node, "value");
        Tensor weights = GetNodeTensorAttr(weights_node, "value");
        Tensor bias = GetNodeTensorAttr(bias_node, "value");
        auto perm_vector = perm.flat<int32_t>();
        auto weights_vector = weights.tensor<float, 4>();
        auto bias_vector = bias.tensor<float, 4>();
        if (perm_vector(0) != 0 || perm_vector(1) != 3 || perm_vector(2) != 1 ||
            perm_vector(3) != 2) {
          // Return original nodes.
          LOG(INFO) << "Skipping replacement for " << transpose_node.name();
          CopyOriginalMatch(match, new_nodes);
          return Status::OK();
        }
        Eigen::array<int, 4> shuffling({0, 2, 3, 1});
        auto new_weights_vector = weights_vector.shuffle(shuffling);
        auto new_bias_vector = bias_vector.shuffle(shuffling);
        auto weight_shape = weights.shape();
        auto bias_shape = bias.shape();
        Tensor new_weights(
            DT_FLOAT, {weight_shape.dim_size(0), weight_shape.dim_size(2),
                       weight_shape.dim_size(3), weight_shape.dim_size(1)});
        Tensor new_bias(DT_FLOAT,
                        {bias_shape.dim_size(0), bias_shape.dim_size(2),
                         bias_shape.dim_size(3), bias_shape.dim_size(1)});
        new_weights.tensor<float, 4>() = new_weights_vector;
        new_bias.tensor<float, 4>() = new_bias_vector;

        // Construct the new nodes.
        NodeDef new_weights_node;
        new_weights_node.set_op("Const");
        new_weights_node.set_name(weights_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_weights_node);
        SetNodeTensorAttr<float>("value", new_weights, &new_weights_node);

        NodeDef new_mul_node;
        new_mul_node.set_op("Mul");
        new_mul_node.set_name(mul_node.name());
        SetNodeAttr("T", DT_FLOAT, &new_mul_node);
        AddNodeInput(input_node.name(), &new_mul_node);
        AddNodeInput(new_weights_node.name(), &new_mul_node);

        NodeDef new_bias_node;
        new_bias_node.set_op("Const");
        new_bias_node.set_name(bias_node.name());
        SetNodeAttr("dtype", DT_FLOAT, &new_bias_node);
        SetNodeTensorAttr<float>("value", new_bias, &new_bias_node);

        NodeDef new_add_node;
        new_add_node.set_op("AddV2");
        new_add_node.set_name(add_node.name());
        // new_add_node.set_device(add_node.device());
        if (add_node.attr().contains("fused_activation_function")) {
          LOG(INFO) << "HAS FUSED ACTIVATION" << std::endl;
          CopyNodeAttr(add_node, "fused_activation_function",
                       "fused_activation_function", &new_add_node);
        }
        SetNodeAttr("T", DT_FLOAT, &new_add_node);
        AddNodeInput(new_mul_node.name(), &new_add_node);
        AddNodeInput(new_bias_node.name(), &new_add_node);

        NodeDef new_transpose_node;
        new_transpose_node = transpose_node;
        new_transpose_node.mutable_input()->at(0) = new_add_node.name();

        new_nodes->push_back(input_node);
        new_nodes->push_back(new_weights_node);
        new_nodes->push_back(new_mul_node);
        new_nodes->push_back(new_bias_node);
        new_nodes->push_back(new_add_node);
        new_nodes->push_back(perm_node);
        new_nodes->push_back(new_transpose_node);

        // Rename references appropriately
        string target_name = new_transpose_node.name();
        inputs_to_rename[add_node.name()] = target_name;
        inputs_to_rename["^" + add_node.name()] = "^" + target_name;
        nodes_to_ignore.insert(new_transpose_node.name());

        return Status::OK();
      },
      {true}, &replaced_graph_def));
  // Make sure all references to removed nodes now point to their inputs.
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      nodes_to_ignore, output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("swap_trans_mul_add", SwapTransMulAdd);

}  // namespace graph_transforms
}  // namespace tensorflow
