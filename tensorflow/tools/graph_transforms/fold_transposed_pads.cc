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

Status FoldTransposedPads(const GraphDef& input_graph_def,
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

  std::map<string, string> inputs_to_rename;
  GraphDef replaced_graph_def;
  TF_RETURN_IF_ERROR(ReplaceMatchingOpTypes(
      input_graph_def,  // clang-format off
      {"Transpose",
        {
          {"Pad",
            {
              {"Transpose",
                {
                  {"*"},     // input_node
                  {"Const"}  // permutation
                }
              },        
              {"Const"},     // paddings
            }
          },
          {"Const"},         // permutation
        }
      },  // clang-format on
      [&input_graph_def, &inputs_to_rename, &required_nodes](
          const NodeMatch& match, const std::set<string>& input_nodes,
          const std::set<string>& output_nodes,
          std::vector<NodeDef>* new_nodes) {
        // Find all the nodes we expect in the subgraph.
        const NodeDef& transpose2_node = match.node;
        const NodeDef& perm2_node = match.inputs[1].node;  // Const
        const NodeDef& pad_node = match.inputs[0].node;
        const NodeDef& paddings_node = match.inputs[0].inputs[1].node;  // Const
        const NodeDef& transpose1_node = match.inputs[0].inputs[0].node;
        const NodeDef& perm1_node =
            match.inputs[0].inputs[0].inputs[1].node;  // Const
        const NodeDef& input_node = match.inputs[0].inputs[0].inputs[0].node;

        // Check that nodes that we use are not used somewhere else.
        for (const auto& node : {transpose2_node, perm2_node, pad_node,
                                 paddings_node, transpose1_node, perm1_node}) {
          if (required_nodes.count(node.name())) {
            // Return original nodes.
            LOG(INFO) << "Skipping replacement for " << node.name();
            CopyOriginalMatch(match, new_nodes);
            return Status::OK();
          }
        }
        // for (const auto& node : {perm2_node, pad_node, paddings_node,
        //                          transpose1_node, perm1_node}) {
        //   string old_transpose2_name = transpose2_node.name();
        //   for (const NodeDef& node : input_graph_def.node()) {
        //     for (const string& input : node.input()) {
        //       if (!input.compare(0, )) }
        //   }
        // }
        Tensor perm2 = GetNodeTensorAttr(perm2_node, "value");
        Tensor paddings = GetNodeTensorAttr(paddings_node, "value");
        Tensor perm1 = GetNodeTensorAttr(perm1_node, "value");
        auto perm1_vector = perm1.flat<int32_t>();
        auto perm2_vector = perm2.flat<int32_t>();
        auto paddings_vector = paddings.flat_outer_dims<int32_t, 2>();
        // LOG(INFO) << "Paddings:\n" << paddings_vector;
        if (perm1_vector(0) != 0 || perm1_vector(1) != 3 ||
            perm1_vector(2) != 1 || perm1_vector(3) != 2) {
          // Return original nodes.
          LOG(INFO) << "Skipping replacement for " << pad_node.name();
          CopyOriginalMatch(match, new_nodes);
          return Status::OK();
        }
        if (perm2_vector(0) != 0 || perm2_vector(1) != 2 ||
            perm2_vector(2) != 3 || perm2_vector(3) != 1) {
          // Return original nodes.
          LOG(INFO) << "Skipping replacement for " << pad_node.name();
          CopyOriginalMatch(match, new_nodes);
          return Status::OK();
        }
        Tensor new_paddings(DT_INT32, paddings.shape());
        auto new_paddings_vector = new_paddings.flat_outer_dims<int32_t, 2>();
        // This abomination is just to re-order the order of the rows of
        // paddings.
        // -_-
        new_paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
            {0, 0}, {1, paddings_vector.dimension(1)}) =
            paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
                {0, 0}, {1, paddings_vector.dimension(1)});
        new_paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
            {1, 0}, {1, paddings_vector.dimension(1)}) =
            paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
                {2, 0}, {1, paddings_vector.dimension(1)});
        new_paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
            {2, 0}, {1, paddings_vector.dimension(1)}) =
            paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
                {3, 0}, {1, paddings_vector.dimension(1)});
        new_paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
            {3, 0}, {1, paddings_vector.dimension(1)}) =
            paddings_vector.slice<Eigen::array<int, 2>, Eigen::array<int, 2>>(
                {1, 0}, {1, paddings_vector.dimension(1)});
        // This doesn't work (compiles but produces gargabe):
        // new_paddings_vector(0) = paddings_vector(0);
        // new_paddings_vector(1) = paddings_vector(2);
        // new_paddings_vector(2) = paddings_vector(3);
        // new_paddings_vector(3) = paddings_vector(1);
        // LOG(INFO) << "New Paddings:\n" << new_paddings_vector;

        // Construct the new nodes.
        NodeDef new_paddings_node;
        new_paddings_node.set_op("Const");
        new_paddings_node.set_name(paddings_node.name());
        SetNodeAttr("dtype", DT_INT32, &new_paddings_node);
        SetNodeTensorAttr<float>("value", new_paddings, &new_paddings_node);

        NodeDef new_pad_node;
        new_pad_node.set_op("Pad");
        new_pad_node.set_name(pad_node.name());
        new_pad_node.set_device(pad_node.device());
        SetNodeAttr("T", DT_FLOAT, &new_pad_node);
        AddNodeInput(input_node.name(), &new_pad_node);
        AddNodeInput(new_paddings_node.name(), &new_pad_node);
        new_nodes->push_back(new_paddings_node);
        new_nodes->push_back(new_pad_node);
        new_nodes->push_back(input_node);
        // We keep the original first transpose node, because the network makes
        // use of it
        new_nodes->push_back(transpose1_node);
        new_nodes->push_back(perm1_node);

        // Rename references appropriately
        string target_name = new_pad_node.name();
        inputs_to_rename[transpose2_node.name()] = target_name;
        inputs_to_rename["^" + transpose2_node.name()] = "^" + target_name;

        return Status::OK();
      },
      {true}, &replaced_graph_def));
  // Make sure all references to removed nodes now point to their inputs.
  TF_RETURN_IF_ERROR(RenameNodeInputs(replaced_graph_def, inputs_to_rename,
                                      std::unordered_set<string>(),
                                      output_graph_def));
  return Status::OK();
}

REGISTER_GRAPH_TRANSFORM("fold_transposed_pads", FoldTransposedPads);

}  // namespace graph_transforms
}  // namespace tensorflow
