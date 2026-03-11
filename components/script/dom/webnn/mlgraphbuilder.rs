/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use dom_struct::dom_struct;
use half::f16;
use js::rust::HandleObject;
use rustnn::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};
use rustnn::operator_options::OperatorOptions;
use script_bindings::cformat;
use script_bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;
use script_bindings::record::Record;
use script_bindings::str::USVString;
use webnn_traits::WebNNMsg;

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConv2dFilterOperandLayout,
    MLConv2dOptions, MLConvTranspose2dFilterOperandLayout, MLConvTranspose2dOptions,
    MLCumulativeSumOptions, MLEluOptions, MLGatherOptions, MLGemmOptions, MLGraphBuilderMethods,
    MLHardSigmoidOptions, MLInputOperandLayout, MLInstanceNormalizationOptions,
    MLInterpolationMode, MLLayerNormalizationOptions, MLLeakyReluOptions, MLLinearOptions,
    MLOperandDataType, MLOperandDescriptor, MLOperatorOptions, MLPadOptions, MLPaddingMode,
    MLPool2dOptions, MLReduceOptions, MLResample2dOptions, MLReverseOptions, MLRoundingType,
    MLSoftmaxOptions, MLSplitOptions, MLTransposeOptions, MLTriangularOptions,
};
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mloperand::MLOperand;
use crate::dom::{MLContext, MLTensor};
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder>
pub(crate) struct MLGraphBuilder {
    reflector_: Reflector,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-context-slot>
    context: Dom<MLContext>,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-hasbuilt-slot>
    /// Built state represented by `graph_info: Option<GraphInfo>` (None => built).

    /// <https://webmachinelearning.github.io/webnn/#computational-graph-input>
    // NOTE: DOM-level storage for `inputs`, `constants`, and `operand_map` was removed.
    // All graph bookkeeping (backend operands, input/constant ids, constant bytes, etc.)
    // now lives in `graph_info` (the rustnn `GraphInfo`). The builder still creates and
    // returns `MLOperand` DOM objects to callers, but it no longer keeps separate DOM
    // lists or a mapping from backend id -> DOM operand.

    /// Implementation-defined graph representation (rustnn `GraphInfo`) used to build backend graphs.
    #[no_trace]
    #[ignore_malloc_size_of = "rustnn::GraphInfo is external; skip malloc-size accounting"]
    graph_info: DomRefCell<Option<GraphInfo>>,

    /// Next operand id counter for rustnn GraphInfo.
    next_operand_id: Cell<u32>,
}

impl MLGraphBuilder {
    /// Helper: map WebNN data type strings to binding enum variants used by descriptors.
    fn data_type_enum_from_str(data_type: &str) -> MLOperandDataType {
        match data_type {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            "int32" => MLOperandDataType::Int32,
            "uint32" => MLOperandDataType::Uint32,
            "int64" => MLOperandDataType::Int64,
            "uint64" => MLOperandDataType::Uint64,
            "int8" => MLOperandDataType::Int8,
            "uint8" => MLOperandDataType::Uint8,
            _ => MLOperandDataType::Float32,
        }
    }

    pub(crate) fn new_inherited(context: &MLContext) -> MLGraphBuilder {
        MLGraphBuilder {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            graph_info: DomRefCell::new(Some(GraphInfo {
                operands: Vec::new(),
                input_operands: Vec::new(),
                output_operands: Vec::new(),
                operations: Vec::new(),
                constant_operand_ids_to_handles: HashMap::new(),
                id_to_constant_tensor_operand_map: HashMap::new(),
                quantized: false,
            })),
            next_operand_id: Cell::new(0),
        }
    }

    pub(crate) fn new(
        context: &MLContext,
        global: &GlobalScope,
        can_gc: CanGc,
    ) -> DomRoot<MLGraphBuilder> {
        reflect_dom_object(
            Box::new(MLGraphBuilder::new_inherited(context)),
            global,
            can_gc,
        )
    }

    pub(crate) fn context(&self) -> Dom<MLContext> {
        self.context.clone()
    }

    fn can_build(&self) -> bool {
        self.graph_info.borrow().is_some() && !self.context().is_lost()
    }

    fn validate_operand(&self, operand: &MLOperand) -> bool {
        operand.builder() == Dom::from_ref(self)
    }

    /// Internal helper accepting a reference to an MLOperand (used by generated bindings
    /// that pass `&MLOperand` directly).
    fn validate_operand_ref(&self, operand: &MLOperand) -> bool {
        operand.builder() == Dom::from_ref(self)
    }

    /// Allocate an operand id, append `operand` into `graph_info.operands`,
    /// optionally record the id in `graph_info.input_operands`, and return the id.
    ///
    /// This centralizes the `next_operand_id` increment + GraphInfo mutation logic
    /// so callers only need to construct a rustnn `Operand` and call this helper.
    fn push_operand_to_graph(&self, operand: Operand, add_to_inputs: bool) -> u32 {
        // Step 1: Read the next backend operand id.
        let id = self.next_operand_id.get();
        // Step 2: Append the operand to GraphInfo and optionally mark it as an input operand.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operands.push(operand);
            if add_to_inputs {
                gi.input_operands.push(id);
            }
        }
        // Step 3: Advance the id counter and return the id assigned to this operand.
        self.next_operand_id.set(id + 1);
        id
    }

    /// Create a rustnn `Operand` from a WebNN data-type string, shape, kind and optional name.
    /// Centralizes descriptor -> `OperandDescriptor` + `Operand` construction.
    fn create_rust_operand(
        &self,
        data_type_str: &str,
        shape: Vec<u32>,
        kind: OperandKind,
        name: Option<String>,
    ) -> Operand {
        // Step 1: Convert WebNN data type strings to rustnn graph data types.
        let rust_data_type = match data_type_str {
            "float32" => DataType::Float32,
            "float16" => DataType::Float16,
            "int32" => DataType::Int32,
            "uint32" => DataType::Uint32,
            "int8" => DataType::Int8,
            "uint8" => DataType::Uint8,
            "int64" => DataType::Int64,
            _ => DataType::Float32,
        };
        // Step 2: Construct the rustnn OperandDescriptor with converted shape dimensions.
        let desc = OperandDescriptor {
            data_type: rust_data_type,
            shape: rustnn::graph::to_dimension_vector(&shape),
            pending_permutation: Vec::new(),
        };
        // Step 3: Return a rustnn Operand carrying descriptor, kind, and optional name.
        Operand {
            descriptor: desc,
            kind,
            name,
        }
    }

    fn push_unary_operation(
        &self,
        op_name: &str,
        input_id: u32,
        output_id: u32,
        attributes: serde_json::Value,
        label: Option<String>,
    ) {
        // Step 1: Append a unary Operation record into GraphInfo.operations.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: op_name.to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(op_name, attributes),
                label,
            });
        }
    }

    fn push_binary_operation(
        &self,
        op_name: &str,
        input_ids: Vec<u32>,
        output_id: u32,
        attributes: serde_json::Value,
        label: Option<String>,
    ) {
        // Step 1: Append an Operation record with a dynamic input list into GraphInfo.operations.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: op_name.to_string(),
                input_operands: input_ids,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(op_name, attributes),
                label,
            });
        }
    }

    fn operator_attributes(op_name: &str, attributes: serde_json::Value) -> OperatorOptions {
        OperatorOptions::from_json_with_op_type(op_name, &attributes).unwrap_or_default()
    }

    fn label_from_operator_options(options: &MLOperatorOptions) -> Option<String> {
        // Step 1: Read operator label.
        // Step 2: Normalize empty labels to None.
        let label = options.label.clone();
        if label.is_empty() {
            None
        } else {
            Some(label.to_string())
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#mlgraphbuilder-element-wise-logical-op>
    fn create_an_element_wise_logical_operation(
        &self,
        op_name: &str,
        a: &MLOperand,
        b: Option<&MLOperand>,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: [=Assert=]: |op| is one of "equal", "notEqual", "greater", "greaterOrEqual", "lesser", "lesserOrEqual", "logicalNot", "logicalAnd", "logicalOr", "logicalXor", "isNaN", "isInfinite".
        debug_assert!(
            [
                "equal",
                "notEqual",
                "greater",
                "greaterOrEqual",
                "lesser",
                "lesserOrEqual",
                "logicalNot",
                "logicalAnd",
                "logicalOr",
                "logicalXor",
                "isNaN",
                "isInfinite",
            ]
            .contains(&op_name)
        );

        // Step 2: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 3: If [=MLGraphBuilder/validating operand=] with [=this=] and |a| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(a) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 4: If |op| is one of "logicalNot", "logicalAnd", "logicalOr", "logicalXor", then:
        // Step 4.1: If |a|'s [=MLOperand/dataType=] is not {{MLOperandDataType/"uint8"}}, then [=exception/throw=] a {{TypeError}}.
        let a_dtype = a.descriptor_data_type();
        if ["logicalNot", "logicalAnd", "logicalOr", "logicalXor"].contains(&op_name) &&
            a_dtype != "uint8"
        {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // Step 5: If |op| is one of "isNaN", "isInfinite", then:
        // Step 5.1: If |a|'s [=MLOperand/dataType=] is not one of « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », then [=exception/throw=] a {{TypeError}}.
        if ["isNaN", "isInfinite"].contains(&op_name) &&
            a_dtype != "float32" &&
            a_dtype != "float16"
        {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // Step 6/7: If |b| is passed, validate |b|, validate matching dataType, and infer |outputShape| by bidirectional broadcasting.
        // Step 7: Otherwise, clone |a|'s shape as |outputShape|.
        let output_shape = if let Some(b_operand) = b {
            if !self.validate_operand_ref(b_operand) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }

            if a_dtype != b_operand.descriptor_data_type() {
                return Err(Error::Type(c"input dataType must match".to_owned()));
            }

            bidirectionally_broadcast_shapes(a.descriptor_shape(), b_operand.descriptor_shape())
                .ok_or_else(|| {
                    Error::Type(c"shapes are not bidirectionally broadcastable".to_owned())
                })?
        } else {
            a.descriptor_shape().clone()
        };

        // Step 8: Let |descriptor| be the result of [=creating an MLOperandDescriptor=] given {{MLOperandDataType/"uint8"}} and |outputShape|.
        let descriptor = MLOperandDescriptor {
            dataType: MLOperandDataType::Uint8,
            shape: output_shape.clone(),
        };

        // Step 9: *Make graph connections:*
        // Step 9.1: Let |output| be the result of [=creating an MLOperand=] given [=this=] and |descriptor|.
        let a_id = a
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

        let b_id = if let Some(b_operand) = b {
            Some(
                b_operand
                    .id()
                    .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?,
            )
        } else {
            None
        };

        let rust_operand =
            self.create_rust_operand("uint8", output_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 9.2: Let |operator| be an [=operator=] for the |op| operation, given |a| and (if |b| is passed) |b|, and |options|.
        // Step 9.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 9.4: Set |operator|'s [=operator/inputs=] to |a| and (if |b| is passed) |b|.
        // Step 9.5: Set |operator|'s [=operator/output=] to |output|.
        let mut input_operands = vec![a_id];
        if let Some(id) = b_id {
            input_operands.push(id);
        }

        self.push_binary_operation(
            op_name,
            input_operands,
            output_id,
            serde_json::json!({}),
            Self::label_from_operator_options(options),
        );

        let operand = create_an_mloperand(
            self,
            Some(&descriptor),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );
        // Step 10: Return |output|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#mlgraphbuilder-create-reduction-operation>
    fn create_reduction_operation(
        &self,
        op_name: &str,
        input: &MLOperand,
        options: &MLReduceOptions,
        allowed_data_types: Option<&[&str]>,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: [=Assert=]: |op| is one of "reduceL1", "reduceL2", "reduceLogSum", "reduceLogSumExp", "reduceMax", "reduceMean", "reduceMin", "reduceProduct", "reduceSum", "reduceSumSquare".
        debug_assert!(
            [
                "reduceL1",
                "reduceL2",
                "reduceLogSum",
                "reduceLogSumExp",
                "reduceMax",
                "reduceMean",
                "reduceMin",
                "reduceProduct",
                "reduceSum",
                "reduceSumSquare",
            ]
            .contains(&op_name)
        );

        // Step 2: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 3: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 4: If |allowedDataTypes| is given and it does not [=list/contain=] |input|'s [=MLOperand/dataType=], then [=exception/throw=] a {{TypeError}}.
        let out_dtype = input.descriptor_data_type();
        if let Some(allowed_data_types) = allowed_data_types {
            if !allowed_data_types.contains(&out_dtype) {
                return Err(Error::Type(c"unsupported input dataType".to_owned()));
            }
        }

        // Step 5: Let |outputShape| be the result of [=MLGraphBuilder/calculating reduction output sizes=] given |input|'s [=MLOperand/shape=], |options|.{{MLReduceOptions/axes}} (if it [=map/exists=]), and |options|.{{MLReduceOptions/keepDimensions}}. If that returns failure, then [=exception/throw=] a {{TypeError}}.
        let reduce_options = rustnn::shape_inference::ReduceOptions {
            axes: options.axes.as_ref().map(|v| v.clone()).unwrap_or_default(),
            keep_dimensions: options.keepDimensions,
        };
        let output_shape =
            rustnn::shape_inference::infer_reduce_shape(input.descriptor_shape(), &reduce_options)
                .map_err(|e| Error::Type(cformat!("{e}")))?;

        // Step 6: Let |desc| be the result of [=creating an MLOperandDescriptor=] given |input|'s [=MLOperand/dataType=] and |outputShape|.
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };

        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

        // Step 7: *Make graph connections:*
        // Step 7.1: Let |output| be the result of [=creating an MLOperand=] given [=this=] and |desc|.
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 7.2: Let |operator| be an [=operator=] for the |op| operation, given |options|.
        // Step 7.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 7.4: Set |operator|'s [=operator/input=] to |input|.
        // Step 7.5: Set |operator|'s [=operator/output=] to |output|.
        self.push_unary_operation(
            op_name,
            input_id,
            output_id,
            serde_json::json!({"axes": reduce_options.axes, "keepDimensions": reduce_options.keep_dimensions}),
            Self::label_from_operator_options(&options.parent),
        );

        // Step 8: Return |output|.
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-quantizelinear>
    fn create_quantize_or_dequantize_linear_operation(
        &self,
        op_name: &str,
        input: &MLOperand,
        scale: &MLOperand,
        zero_point: &MLOperand,
        quantize: bool,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Validate builder state and all operand ownership.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) ||
            !self.validate_operand_ref(scale) ||
            !self.validate_operand_ref(zero_point)
        {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        // Step 2: Select output data type from zeroPoint (quantize) or scale (dequantize).
        let out_dtype_str = if quantize {
            zero_point.descriptor_data_type()
        } else {
            scale.descriptor_data_type()
        };
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype_str),
            shape: input.descriptor_shape().clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let scale_id = scale
            .id()
            .ok_or_else(|| Error::Type(c"scale operand has no backend id".to_owned()))?;
        let zero_id = zero_point
            .id()
            .ok_or_else(|| Error::Type(c"zeroPoint operand has no backend id".to_owned()))?;
        // Step 3: Create output operand id, record operation metadata, and return output operand.
        let rust_operand = self.create_rust_operand(
            out_dtype_str,
            input.descriptor_shape().clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);
        self.push_binary_operation(
            op_name,
            vec![input_id, scale_id, zero_id],
            output_id,
            serde_json::json!({}),
            Self::label_from_operator_options(options),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-scatterelements>
    fn create_scatter_elements_operation(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        updates: &MLOperand,
        axis: i32,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Validate builder state and all operand ownership.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) ||
            !self.validate_operand_ref(indices) ||
            !self.validate_operand_ref(updates)
        {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        if !["int32", "uint32", "int64"].contains(&indices.descriptor_data_type()) {
            return Err(Error::Type(
                c"unsupported indices dataType for scatterElements".to_owned(),
            ));
        }
        if updates.descriptor_data_type() != input.descriptor_data_type() {
            return Err(Error::Type(
                c"updates must have same dataType as input".to_owned(),
            ));
        }
        // Step 2: Validate scatter-elements shape constraints and infer output descriptor.
        rustnn::shape_inference::infer_scatter_elements_shape(
            input.descriptor_shape(),
            indices.descriptor_shape(),
            updates.descriptor_shape(),
            axis,
        )
        .map_err(|e| Error::Type(cformat!("{e}")))?;
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: input.descriptor_shape().clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let indices_id = indices
            .id()
            .ok_or_else(|| Error::Type(c"indices operand has no backend id".to_owned()))?;
        let updates_id = updates
            .id()
            .ok_or_else(|| Error::Type(c"updates operand has no backend id".to_owned()))?;
        // Step 3: Create output operand id, record operation metadata, and return output operand.
        let rust_operand = self.create_rust_operand(
            out_dtype,
            input.descriptor_shape().clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);
        self.push_binary_operation(
            "scatterElements",
            vec![input_id, indices_id, updates_id],
            output_id,
            serde_json::json!({"axis": axis}),
            Self::label_from_operator_options(options),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-scatternd>
    fn create_scatter_nd_operation(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        updates: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Validate builder state and all operand ownership.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) ||
            !self.validate_operand_ref(indices) ||
            !self.validate_operand_ref(updates)
        {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        if !["int32", "uint32", "int64"].contains(&indices.descriptor_data_type()) {
            return Err(Error::Type(
                c"unsupported indices dataType for scatterND".to_owned(),
            ));
        }
        if updates.descriptor_data_type() != input.descriptor_data_type() {
            return Err(Error::Type(
                c"updates must have same dataType as input".to_owned(),
            ));
        }
        // Step 2: Validate scatter-ND shape constraints and infer output descriptor.
        rustnn::shape_inference::infer_scatter_nd_shape(
            input.descriptor_shape(),
            indices.descriptor_shape(),
            updates.descriptor_shape(),
        )
        .map_err(|e| Error::Type(cformat!("{e}")))?;
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: input.descriptor_shape().clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let indices_id = indices
            .id()
            .ok_or_else(|| Error::Type(c"indices operand has no backend id".to_owned()))?;
        let updates_id = updates
            .id()
            .ok_or_else(|| Error::Type(c"updates operand has no backend id".to_owned()))?;
        // Step 3: Create output operand id, record operation metadata, and return output operand.
        let rust_operand = self.create_rust_operand(
            out_dtype,
            input.descriptor_shape().clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);
        self.push_binary_operation(
            "scatterND",
            vec![input_id, indices_id, updates_id],
            output_id,
            serde_json::json!({}),
            Self::label_from_operator_options(options),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#mlgraphbuilder-pooling-op>
    fn create_a_pooling_operation(
        &self,
        op_name: &str,
        input: &MLOperand,
        options: &MLPool2dOptions,
        allowed_data_types: Option<&[&str]>,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Assert: |op| is one of "averagePool2d", "l2Pool2d", "maxPool2d".
        if !matches!(op_name, "averagePool2d" | "l2Pool2d" | "maxPool2d") {
            debug_assert!(false, "unexpected pooling op: {op_name}");
            return Err(Error::Type(c"invalid pooling op".to_owned()));
        }

        // Step 2: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 3: If validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 4: If |allowedDataTypes| is given and it does not contain |input|'s dataType, then throw a TypeError.
        let out_dtype = input.descriptor_data_type();
        if let Some(allowed_data_types) = allowed_data_types {
            if !allowed_data_types.contains(&out_dtype) {
                return Err(Error::Type(c"unsupported input dataType".to_owned()));
            }
        }

        // Step 5: If |input|'s rank is not 4, then throw a TypeError.
        let input_shape = input.descriptor_shape();
        if input_shape.len() != 4 {
            return Err(Error::Type(c"input must be a 4-D tensor".to_owned()));
        }

        // Step 6: Switch on |options|.layout and extract batch/channel/spatial dimensions.
        let (layout_str, batches, channels, input_height, input_width) = match options.layout {
            MLInputOperandLayout::Nchw => (
                "nchw",
                input_shape[0],
                input_shape[1],
                input_shape[2],
                input_shape[3],
            ),
            MLInputOperandLayout::Nhwc => (
                "nhwc",
                input_shape[0],
                input_shape[3],
                input_shape[1],
                input_shape[2],
            ),
        };

        // Step 7: If |windowDimensions| does not exist, then set it to « inputHeight, inputWidth ».
        let window_dimensions = options
            .windowDimensions
            .as_ref()
            .cloned()
            .unwrap_or_else(|| vec![input_height, input_width]);

        // Step 8: If |windowDimensions|'s size is not 2, then throw a TypeError.
        if window_dimensions.len() != 2 {
            return Err(Error::Type(c"windowDimensions must be length 2".to_owned()));
        }

        // Step 9: If any item in |windowDimensions| is equal to 0, then throw a TypeError.
        if window_dimensions.contains(&0) {
            return Err(Error::Type(
                c"windowDimensions values must be >= 1".to_owned(),
            ));
        }

        // Step 10: If |outputSizes| exists, or if |padding| does not exist, then set |padding| to « 0, 0, 0, 0 ».
        let pads = options
            .padding
            .as_ref()
            .cloned()
            .unwrap_or_else(|| vec![0, 0, 0, 0]);

        // Step 11: If |padding|'s size is not 4, then throw a TypeError.
        if pads.len() != 4 {
            return Err(Error::Type(c"padding must be length 4".to_owned()));
        }

        // Step 12: If |strides| does not exist, then set it to « 1, 1 ».
        let strides = options
            .strides
            .as_ref()
            .cloned()
            .unwrap_or_else(|| vec![1, 1]);

        // Step 13: If |strides|'s size is not 2, then throw a TypeError.
        if strides.len() != 2 {
            return Err(Error::Type(c"strides must be length 2".to_owned()));
        }

        // Step 14: If any item in |strides| is 0, then throw a TypeError.
        if strides.contains(&0) {
            return Err(Error::Type(c"strides values must be >= 1".to_owned()));
        }

        // Step 15: If |outputSizes| exists, then validate its size and relationship with |strides|.
        let output_sizes = options.outputSizes.as_ref().cloned();
        if let Some(output_sizes) = output_sizes.as_ref() {
            if output_sizes.len() != 2 {
                return Err(Error::Type(c"outputSizes must be length 2".to_owned()));
            }
            if output_sizes.contains(&0) {
                return Err(Error::Type(c"outputSizes values must be >= 1".to_owned()));
            }
            if output_sizes[0] < strides[0] || output_sizes[1] < strides[1] {
                return Err(Error::Type(
                    c"outputSizes values must be >= corresponding strides".to_owned(),
                ));
            }
            if options.outputShapeRounding != MLRoundingType::Floor {
                return Err(Error::Type(
                    c"outputShapeRounding must be 'floor' when outputSizes is provided".to_owned(),
                ));
            }
        }

        // Step 16: If |dilations| does not exist, then set it to « 1, 1 ».
        let dilations = options
            .dilations
            .as_ref()
            .cloned()
            .unwrap_or_else(|| vec![1, 1]);

        // Step 17: If |dilations|'s size is not 2, then throw a TypeError.
        if dilations.len() != 2 {
            return Err(Error::Type(c"dilations must be length 2".to_owned()));
        }

        // Step 18: If any item in |dilations| is 0, then throw a TypeError.
        if dilations.contains(&0) {
            return Err(Error::Type(c"dilations values must be >= 1".to_owned()));
        }

        // Step 19: Let |desc| be a copy of |input|.[[descriptor]].
        let mut desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: input_shape.clone(),
        };

        // Step 20: Calculate the output shape.
        let window_height = window_dimensions[0] as f64;
        let window_width = window_dimensions[1] as f64;
        let stride_height = strides[0] as f64;
        let stride_width = strides[1] as f64;
        let dilation_height = dilations[0] as f64;
        let dilation_width = dilations[1] as f64;
        let pad_beginning_height = pads[0] as f64;
        let pad_ending_height = pads[1] as f64;
        let pad_beginning_width = pads[2] as f64;
        let pad_ending_width = pads[3] as f64;

        // Step 20.1: Let « calculatedOutputHeight, calculatedOutputWidth » be the result of calculating conv2d output sizes.
        let effective_window_height = (window_height - 1.0) * dilation_height + 1.0;
        let effective_window_width = (window_width - 1.0) * dilation_width + 1.0;
        let calculated_output_height = 1.0 +
            ((input_height as f64) - effective_window_height +
                pad_beginning_height +
                pad_ending_height) /
                stride_height;
        let calculated_output_width = 1.0 +
            ((input_width as f64) - effective_window_width +
                pad_beginning_width +
                pad_ending_width) /
                stride_width;

        // Step 20.2/20.3: Resolve outputSizes or outputShapeRounding when outputSizes is absent.
        let (output_height, output_width) = if let Some(output_sizes) = output_sizes.as_ref() {
            let output_height = output_sizes[0] as f64;
            let output_width = output_sizes[1] as f64;
            let floor_matches = output_height == calculated_output_height.floor() &&
                output_width == calculated_output_width.floor();
            let ceil_matches = output_height == calculated_output_height.ceil() &&
                output_width == calculated_output_width.ceil();

            if !floor_matches && !ceil_matches {
                return Err(Error::Type(
                    c"outputSizes are inconsistent with calculated output shape".to_owned(),
                ));
            }

            (output_sizes[0], output_sizes[1])
        } else {
            let (rounded_height, rounded_width) = match options.outputShapeRounding {
                MLRoundingType::Floor => (
                    calculated_output_height.floor(),
                    calculated_output_width.floor(),
                ),
                MLRoundingType::Ceil => (
                    calculated_output_height.ceil(),
                    calculated_output_width.ceil(),
                ),
            };

            if !rounded_height.is_finite() ||
                !rounded_width.is_finite() ||
                rounded_height < 1.0 ||
                rounded_width < 1.0
            {
                return Err(Error::Type(c"invalid output shape".to_owned()));
            }
            if rounded_height > (u32::MAX as f64) || rounded_width > (u32::MAX as f64) {
                return Err(Error::Type(c"invalid output shape".to_owned()));
            }

            (rounded_height as u32, rounded_width as u32)
        };

        // Step 20.4: If either outputHeight or outputWidth is not a valid dimension, then throw a TypeError.
        if output_height == 0 || output_width == 0 {
            return Err(Error::Type(c"invalid output shape".to_owned()));
        }

        // Step 20.5: Set |outputShape| according to |options|.layout.
        let output_shape = if layout_str == "nchw" {
            vec![batches, channels, output_height, output_width]
        } else {
            vec![batches, output_height, output_width, channels]
        };

        // Step 20.6: Set |desc|.shape to |outputShape|.
        desc.shape = output_shape.clone();

        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

        // Step 21.1: Let |output| be the result of creating an MLOperand given this and |desc|.
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 21.2/21.3/21.4/21.5: Create and connect the pooling operator metadata.
        self.push_unary_operation(
            op_name,
            input_id,
            output_id,
            serde_json::json!({
                "windowDimensions": window_dimensions,
                "strides": strides,
                "dilations": dilations,
                "padding": pads,
                "layout": layout_str,
                "outputShapeRounding": if options.outputShapeRounding == MLRoundingType::Floor { "floor" } else { "ceil" },
                "outputSizes": output_sizes,
            }),
            Self::label_from_operator_options(&options.parent),
        );

        // Step 22: Return |output|.
        Ok(copy_an_mloperand(
            input,
            Some(&desc),
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-convtranspose2d>
    fn create_conv_transpose2d_operation(
        &self,
        input: &MLOperand,
        filter: &MLOperand,
        options: &MLConvTranspose2dOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Validate builder state and operand ownership.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) || !self.validate_operand_ref(filter) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        // Step 2: Normalize transpose-convolution options and infer output shape.
        use rustnn::shape_inference::{
            Conv2dFilterLayout, Conv2dInputLayout, ConvTranspose2dOptions,
        };
        let input_layout = match options.inputLayout {
            MLInputOperandLayout::Nchw => Conv2dInputLayout::Nchw,
            MLInputOperandLayout::Nhwc => Conv2dInputLayout::Nhwc,
        };
        let input_layout_str = match options.inputLayout {
            MLInputOperandLayout::Nchw => "nchw",
            MLInputOperandLayout::Nhwc => "nhwc",
        };
        let filter_layout = match options.filterLayout {
            MLConvTranspose2dFilterOperandLayout::Iohw => Conv2dFilterLayout::Oihw,
            MLConvTranspose2dFilterOperandLayout::Hwoi => Conv2dFilterLayout::Ihwo,
            MLConvTranspose2dFilterOperandLayout::Ohwi => Conv2dFilterLayout::Ohwi,
            MLConvTranspose2dFilterOperandLayout::Oihw => Conv2dFilterLayout::Hwio,
        };
        let filter_layout_str = match options.filterLayout {
            MLConvTranspose2dFilterOperandLayout::Iohw => "iohw",
            MLConvTranspose2dFilterOperandLayout::Hwoi => "hwoi",
            MLConvTranspose2dFilterOperandLayout::Ohwi => "ohwi",
            MLConvTranspose2dFilterOperandLayout::Oihw => "oihw",
        };
        let strides = options
            .strides
            .as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| vec![1, 1]);
        let dilations = options
            .dilations
            .as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| vec![1, 1]);
        let pads = options
            .padding
            .as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| vec![0, 0, 0, 0]);
        let output_padding = options
            .outputPadding
            .as_ref()
            .map(|v| v.clone())
            .unwrap_or_else(|| vec![0, 0]);
        let conv_options = ConvTranspose2dOptions {
            strides: strides.clone(),
            dilations: dilations.clone(),
            pads: pads.clone(),
            output_padding: output_padding.clone(),
            output_sizes: options.outputSizes.as_ref().map(|v| v.clone()),
            groups: options.groups,
            input_layout,
            filter_layout,
        };
        let output_shape = rustnn::shape_inference::infer_conv_transpose2d_shape(
            input.descriptor_shape(),
            filter.descriptor_shape(),
            &conv_options,
        )
        .map_err(|e| Error::Type(cformat!("{e}")))?;
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let filter_id = filter
            .id()
            .ok_or_else(|| Error::Type(c"filter operand has no backend id".to_owned()))?;
        // Step 3: Create output operand id, record operation metadata, and return output operand.
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);
        let mut input_ids = vec![input_id, filter_id];
        if let Some(bias) = options.bias.as_ref() {
            let bias_id = bias
                .id()
                .ok_or_else(|| Error::Type(c"bias operand has no backend id".to_owned()))?;
            input_ids.push(bias_id);
        }

        let mut attributes = serde_json::json!({
            "strides": strides,
            "dilations": dilations,
            "padding": pads,
            "outputPadding": output_padding,
            "groups": options.groups,
            "inputLayout": input_layout_str,
            "filterLayout": filter_layout_str,
        });
        if let Some(output_sizes) = options.outputSizes.as_ref() {
            attributes["outputSizes"] = serde_json::json!(output_sizes);
        }
        if options.bias.is_some() {
            attributes["hasBias"] = serde_json::json!(true);
        }

        self.push_binary_operation(
            "convTranspose2d",
            input_ids,
            output_id,
            attributes,
            Self::label_from_operator_options(&options.parent),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-split>
    fn create_split_operation(
        &self,
        input: &MLOperand,
        split_spec: rustnn::shape_inference::SplitSpec,
        axis: u32,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<Vec<DomRoot<MLOperand>>> {
        // Step 1: Validate builder state and input ownership.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        // Step 2: Infer all split output shapes.
        let output_shapes = rustnn::shape_inference::infer_split_shapes(
            input.descriptor_shape(),
            &split_spec,
            axis,
        )
        .map_err(|e| Error::Type(cformat!("{e}")))?;
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let out_dtype = input.descriptor_data_type();
        let out_dtype_enum = Self::data_type_enum_from_str(out_dtype);
        let mut output_ids = Vec::with_capacity(output_shapes.len());
        let mut outputs = Vec::with_capacity(output_shapes.len());
        // Step 3: Allocate backend output operands and corresponding DOM outputs.
        for shape in output_shapes {
            let rust_operand =
                self.create_rust_operand(out_dtype, shape.clone(), OperandKind::Output, None);
            let output_id = self.push_operand_to_graph(rust_operand, false);
            output_ids.push(output_id);
            let desc = MLOperandDescriptor {
                dataType: out_dtype_enum,
                shape,
            };
            outputs.push(create_an_mloperand(
                self,
                Some(&desc),
                None,
                None,
                false,
                false,
                Some(output_id),
                can_gc,
            ));
        }
        // Step 4: Record the split operation with all output ids and return DOM outputs.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "split".to_string(),
                input_operands: vec![input_id],
                output_operand: None,
                output_operands: output_ids,
                attributes: Self::operator_attributes("split", serde_json::json!({"axis": axis})),
                label: Self::label_from_operator_options(options),
            });
        }
        Ok(outputs)
    }

    /// <https://webmachinelearning.github.io/webnn/#mlgraphbuilder-argminmax-op>
    fn mlgraphbuilder_argminmax_op(
        &self,
        _op_name: &str,
        input: &MLOperand,
        axis: u32,
        options: &MLArgMinMaxOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1 (Assert): |op| is one of "argMin", "argMax".
        debug_assert!(_op_name == "argMin" || _op_name == "argMax");

        // Step 2: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 3: If MLGraphBuilder/validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 4: If |axis| is greater than or equal to |input|'s rank, then throw a TypeError.
        let in_shape = input.descriptor_shape();
        if (axis as usize) >= in_shape.len() {
            return Err(Error::Type(c"axis out of range".to_owned()));
        }

        // Step 5: Validate |options|.outputDataType is allowed (int32 or int64 for argMin/argMax).
        let out_dtype_str = options.outputDataType.as_str();
        if out_dtype_str != "int32" && out_dtype_str != "int64" {
            return Err(Error::Type(
                c"outputDataType must be 'int32' or 'int64'".to_owned(),
            ));
        }

        // Step 6: If input.shape[axis] is greater than outputDataType's max value, then throw.
        let axis_dim = in_shape[axis as usize] as u128;
        match out_dtype_str {
            "int32" => {
                if axis_dim > (i32::MAX as u128) {
                    return Err(Error::Type(c"dimension too large for int32".to_owned()));
                }
            },
            "int64" => {
                if axis_dim > (i64::MAX as u128) {
                    return Err(Error::Type(c"dimension too large for int64".to_owned()));
                }
            },
            _ => {},
        }

        // Step 7: Let |outputShape| be the result of calculating reduction output sizes.
        let output_shape = match rustnn::shape_inference::infer_arg_reduce_shape(
            in_shape,
            axis,
            options.keepDimensions,
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(cformat!("{e}"))),
        };

        // Step 8: Let |desc| be the result of creating an MLOperandDescriptor given
        // |options.outputDataType| and |outputShape|.
        let desc = MLOperandDescriptor {
            dataType: if out_dtype_str == "int32" {
                MLOperandDataType::Int32
            } else {
                MLOperandDataType::Int64
            },
            shape: output_shape.clone(),
        };

        // Step 9: *Make graph connections:* (record an operator in GraphInfo.operations)
        // Implementation note: the DOM MLOperand object does not store an [[operator]] slot
        // in this binding; we reflect the operator by recording it in the implementation
        // GraphInfo (rustnn::GraphInfo.operations).

        // Ensure the input has a backend operand id.
        let input_id = match input.id() {
            Some(i) => i,
            None => return Err(Error::Type(c"input operand has no backend id".to_owned())),
        };

        // Create backend operand for the output now (implementation detail: backend id is needed
        // to record the operator). The spec's conceptual order is preserved (we create a
        // descriptor/operator relationship), but the concrete backend id must exist first.
        let rust_operand = self.create_rust_operand(
            out_dtype_str,
            output_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Build operation attributes per the README/spec.
        let mut attributes = serde_json::json!({
            "axis": axis,
            "keepDimensions": options.keepDimensions,
        });
        // Record the explicit outputDataType when present.
        attributes["outputDataType"] = serde_json::json!(out_dtype_str);

        // Optional label (MLOperatorOptions.label). Use empty => none.
        let label = {
            let l = options.parent.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // Push an Operation record into the builder's GraphInfo.operations so the
        // backend has the operator + connectivity metadata available at Build().
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: _op_name.to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(_op_name, attributes),
                label,
            });
        }

        // Step 9.2: Let |output| be the result of creating an MLOperand given this and |desc|.
        // (Also return it per Step 10.)
        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );
        Ok(operand)
    }
}

/// <https://webmachinelearning.github.io/webnn/#mloperanddescriptor-check-dimensions>
pub(crate) fn check_dimensions(descriptor: &MLOperandDescriptor) -> bool {
    // Step 1: If any item of |descriptor|.shape is not a valid dimension, then return false.
    // A valid dimension is an integer greater than zero and in the range of `long`.
    if descriptor.shape.iter().any(|&d| d == 0) {
        return false;
    }

    // Step 2: If |descriptor|.shape's list/size is too large to be supported by the implementation,
    // then return false.  Use an implementation-defined conservative upper bound for now.
    // TODO (spec: #mloperanddescriptor-check-dimensions): choose a platform-appropriate limit.
    const MAX_OPERAND_DIMS: usize = 8; // conservative default
    if descriptor.shape.len() > MAX_OPERAND_DIMS {
        return false;
    }

    // Step 3: If |descriptor|'s byte length is not supported by the implementation, then return false.
    // Compute byte length per-spec: elementLength * elementSize, detect overflow.
    let mut element_length: u128 = 1;
    for &dim in descriptor.shape.iter() {
        element_length = match element_length.checked_mul(dim as u128) {
            Some(v) => v,
            None => return false, // overflow => unsupported
        };
    }

    let element_size: u128 = match descriptor.dataType.as_str() {
        "float32" => 4,
        "float16" => 2,
        "int32" => 4,
        "uint32" => 4,
        "int8" => 1,
        "uint8" => 1,
        "int64" => 8,
        "uint64" => 8,
        _ => 4,
    };

    let byte_length = match element_length.checked_mul(element_size) {
        Some(v) => v,
        None => return false, // overflow
    };

    if byte_length > (usize::MAX as u128) {
        return false;
    }

    // Implementation may impose further limits (not enforced here).
    true
}

/// <https://webmachinelearning.github.io/webnn/#cast>
fn cast_number_to_data_type(x: f64, data_type: &str) -> f64 {
    // Step 1: Switch on |dataType|.
    match data_type {
        // Step 1.1: {{MLOperandDataType/"float32"}} → Return [=ConvertToFloat=](|x|, 32).
        "float32" => convert_to_float(x, 32),

        // Step 1.2: {{MLOperandDataType/"float16"}} → Return [=ConvertToFloat=](|x|, 16).
        "float16" => convert_to_float(x, 16),

        // Step 1.3: {{MLOperandDataType/"int64"}} → Return [=ConvertToInt=](|x|, 64, "signed").
        "int64" => convert_to_int(x, 64, "signed"),

        // Step 1.4: {{MLOperandDataType/"uint64"}} → Return [=ConvertToInt=](|x|, 64, "unsigned").
        "uint64" => convert_to_int(x, 64, "unsigned"),

        // Step 1.5: {{MLOperandDataType/"int32"}} → Return [=ConvertToInt=](|x|, 32, "signed").
        "int32" => convert_to_int(x, 32, "signed"),

        // Step 1.6: {{MLOperandDataType/"uint32"}} → Return [=ConvertToInt=](|x|, 32, "signed").
        // Note: the current WebNN spec text says "signed" for `uint32`; this helper intentionally mirrors the spec prose verbatim.
        "uint32" => convert_to_int(x, 32, "signed"),

        // Step 1.7: {{MLOperandDataType/"int8"}} → Return [=ConvertToInt=](|x|, 8, "signed").
        "int8" => convert_to_int(x, 8, "signed"),

        // Step 1.8: {{MLOperandDataType/"uint8"}} → Return [=ConvertToInt=](|x|, 8, "unsigned").
        "uint8" => convert_to_int(x, 8, "unsigned"),

        _ => {
            debug_assert!(
                false,
                "unexpected MLOperandDataType for cast algorithm: {data_type}"
            );
            x
        },
    }
}

/// <https://webmachinelearning.github.io/webnn/#converttofloat>
fn convert_to_float(x: f64, bit_length: u8) -> f64 {
    // Step 1: If |x| is NaN, then return NaN.
    if x.is_nan() {
        return f64::NAN;
    }

    // Step 2: Switch on |bitLength|.
    let y = match bit_length {
        // Step 2.1: 32.
        // Step 3: Let |y| be the number in |S| that is closest to |x|, selecting the number with an even significand if there are two [=equally close values=].
        // Note: Rust's `f32` conversion uses IEEE-754 round-to-nearest, ties-to-even semantics and yields infinities for out-of-range finite values, which matches the spec's outcome.
        32 => (x as f32) as f64,

        // Step 2.2: 16.
        // Step 3: Let |y| be the number in |S| that is closest to |x|, selecting the number with an even significand if there are two [=equally close values=].
        // Note: `half::f16` provides IEEE-754 binary16 conversion with the rounding behavior required by the spec.
        16 => f16::from_f64(x).to_f64(),

        _ => {
            debug_assert!(
                false,
                "unexpected bit length for ConvertToFloat: {bit_length}"
            );
            (x as f32) as f64
        },
    };

    // Step 4: If |y| is |upperBound|, then return +Infinity.
    // Step 5: If |y| is |lowerBound|, then return -Infinity.
    // Note: the conversions above already produce infinities for out-of-range inputs, so no extra code is needed here.

    // Step 6: If |y| is +0 and |x| is negative, then return -0.
    if y == 0.0 && x.is_sign_negative() {
        return -0.0;
    }

    // Step 7: Return |y|.
    y
}

/// <https://webmachinelearning.github.io/webnn/#converttoint>
fn convert_to_int(x: f64, bit_length: u8, signedness: &str) -> f64 {
    // Step 1: If |signedness| is "unsigned", then:
    // Step 1.1: Let |lowerBound| be 0.
    // Step 1.2: Let |upperBound| be 2^|bitLength| - 1.
    // Step 2: Otherwise:
    // Step 2.1: Let |lowerBound| be -(2^|bitLength| - 1).
    // Step 2.2: Let |upperBound| be 2^|bitLength| - 1 - 1.
    let (lower_bound, upper_bound) = if signedness == "unsigned" {
        (0.0, (2_f64).powi(i32::from(bit_length)) - 1.0)
    } else {
        (
            -((2_f64).powi(i32::from(bit_length) - 1)),
            (2_f64).powi(i32::from(bit_length) - 1) - 1.0,
        )
    };

    // Step 3: If |x| is -0, then set |x| to +0.
    let x = if x == 0.0 { 0.0 } else { x };

    // Step 4: If |x| is NaN, then return +0.
    if x.is_nan() {
        return 0.0;
    }

    // Step 5: Set |x| to min(max(|x|, |lowerBound|), |upperBound|).
    let x = x.clamp(lower_bound, upper_bound);

    // Step 6: Round |x| to the nearest integer, choosing the even integer if it lies halfway between two, and choosing +0 rather than -0.
    let x = x.round_ties_even();

    // Step 7: Return |x|.
    if x == 0.0 { 0.0 } else { x }
}

/// <https://webmachinelearning.github.io/webnn/#create-an-mloperand>
fn create_an_mloperand(
    builder: &MLGraphBuilder,
    descriptor: Option<&MLOperandDescriptor>,
    tensor: Option<&MLTensor>,
    name: Option<DOMString>,
    is_input: bool,
    is_constant: bool,
    operand_id: Option<u32>,
    can_gc: CanGc,
) -> DomRoot<MLOperand> {
    // Step 1: Let |realm| be |builder|'s relevant realm.
    // Step 2: Let |operand| be a new MLOperand in |realm|.
    let global = &builder.global();
    let operand = if let Some(t) = tensor {
        MLOperand::new_from_tensor(
            builder,
            global,
            t,
            name,
            is_input,
            is_constant,
            operand_id,
            can_gc,
        )
    } else if let Some(desc) = descriptor {
        MLOperand::new(
            builder,
            global,
            desc,
            name,
            is_input,
            is_constant,
            operand_id,
            can_gc,
        )
    } else {
        // Internal invariant: caller must provide either a descriptor or a tensor.
        // Use `debug_assert!` (not `panic!`) for internal invariants so release builds
        // do not abort; provide a safe fallback value for release to preserve
        // resilience in production builds.
        debug_assert!(false, "create_an_mloperand requires a descriptor or tensor");
        return reflect_dom_object::<crate::DomTypeHolder, MLOperand, GlobalScope>(
            Box::new(MLOperand::new_inherited(
                builder,
                operand_id,
                "float32".to_string(),
                Vec::new(),
                None,
                false,
                false,
            )),
            &builder.global(),
            can_gc,
        );
    };

    // Step 3: Set |operand|.[[builder]] to |builder| and set |operand|.[[descriptor]] to |desc|.
    // (Handled by the MLOperand constructors above.)

    // Step 4: Return |operand|.
    operand
}

/// <https://webmachinelearning.github.io/webnn/#api-mloperand-create>
fn copy_an_mloperand(
    operand: &MLOperand,
    descriptor_override: Option<&MLOperandDescriptor>,
    operand_id: Option<u32>,
    can_gc: CanGc,
) -> DomRoot<MLOperand> {
    // Step 1: Let |builder| be |operand|.[[builder]].
    let builder = operand.builder();

    // Step 2: Let |realm| be |builder|'s relevant realm.
    let global = builder.global();

    // Step 3: Let |result| be a new MLOperand in |realm|.
    // Step 4: Set |result|.[[builder]] to |builder|.
    // Step 5: Set |result|.[[descriptor]] to |operand|.[[descriptor]].
    let copied_desc = MLOperandDescriptor {
        dataType: match operand.descriptor_data_type() {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            "int32" => MLOperandDataType::Int32,
            "uint32" => MLOperandDataType::Uint32,
            "int64" => MLOperandDataType::Int64,
            "uint64" => MLOperandDataType::Uint64,
            "int8" => MLOperandDataType::Int8,
            "uint8" => MLOperandDataType::Uint8,
            _ => MLOperandDataType::Float32,
        },
        shape: operand.descriptor_shape().clone(),
    };

    // Note: this binding computes output descriptors eagerly for operator results,
    // so callers may provide an override descriptor instead of the copied descriptor.
    let descriptor = descriptor_override.unwrap_or(&copied_desc);

    // Step 6: If |operand|.[[name]] exists, then set |result|.[[name]] to |operand|.[[name]].
    let name = operand.name();

    let result = MLOperand::new(
        &builder, &global, descriptor, name, false, false, operand_id, can_gc,
    );

    // Step 7: Return |result|.
    result
}

/// <https://webmachinelearning.github.io/webnn/#bidirectionally-broadcasting>
fn bidirectionally_broadcast_shapes(shape_a: &[u32], shape_b: &[u32]) -> Option<Vec<u32>> {
    // Step 1: Let |sizeA| be |shapeA|'s [=list/size=].
    let size_a = shape_a.len();

    // Step 2: Let |sizeB| be |shapeB|'s [=list/size=].
    let size_b = shape_b.len();

    // Step 3: Let |outputSize| be the maximum of |sizeA| and |sizeB|.
    let output_size = std::cmp::max(size_a, size_b);

    // Step 4: Let |paddedA| be a [=list/clone=] of |shapeA|.
    let mut padded_a = shape_a.to_vec();

    // Step 5: While |paddedA|'s [=list/size=] is less than |outputSize|, [=list/prepend=] 1 to |paddedA|.
    while padded_a.len() < output_size {
        padded_a.insert(0, 1);
    }

    // Step 6: Let |paddedB| be a [=list/clone=] of |shapeB|.
    let mut padded_b = shape_b.to_vec();

    // Step 7: While |paddedB|'s [=list/size=] is less than |outputSize|, [=list/prepend=] 1 to |paddedB|.
    while padded_b.len() < output_size {
        padded_b.insert(0, 1);
    }

    // Step 8: Let |outputShape| be a new [=/list=].
    let mut output_shape = Vec::with_capacity(output_size);

    // Step 9: [=list/For each=] |index| in [=the range=] 0 to |outputSize|, exclusive:
    for index in 0..output_size {
        // Step 9.1: Let |dimA| be |paddedA|[|index|].
        let dim_a = padded_a[index];

        // Step 9.2: Let |dimB| be |paddedB|[|index|].
        let dim_b = padded_b[index];

        // Step 9.3: If |dimA| is not equal to |dimB|, and |dimA| is not equal to 1, and |dimB| is not equal to 1, then return failure.
        if dim_a != dim_b && dim_a != 1 && dim_b != 1 {
            return None;
        }

        // Step 9.4: [=list/Append=] the maximum of |dimA| and |dimB| to |outputShape|.
        output_shape.push(std::cmp::max(dim_a, dim_b));
    }

    // Step 10: Return |outputShape|.
    Some(output_shape)
}

fn unidirectionally_broadcastable_to_shape(source: &[u32], target: &[u32]) -> bool {
    if source.len() > target.len() {
        return false;
    }

    for index in 0..target.len() {
        let source_index = source.len().checked_sub(1 + index);
        let source_dim = source_index.map_or(1, |i| source[i]);
        let target_dim = target[target.len() - 1 - index];
        if source_dim != 1 && source_dim != target_dim {
            return false;
        }
    }

    true
}

impl MLGraphBuilder {
    /// <https://webmachinelearning.github.io/webnn/#mlgraphbuilder-element-wise-unary-op>
    fn create_an_element_wise_unary_operation(
        &self,
        op_name: &str,
        input: &MLOperand,
        allowed_data_types: Option<&[&str]>,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: [=Assert=]: |op| is one of "abs", "ceil", "cos", "erf", "exp", "floor", "identity", "log", "neg", "reciprocal", "roundEven", "sin", "sign", "sqrt", "tan".
        debug_assert!(
            [
                "abs",
                "ceil",
                "cos",
                "erf",
                "exp",
                "floor",
                "identity",
                "log",
                "neg",
                "reciprocal",
                "roundEven",
                "sin",
                "sign",
                "sqrt",
                "tan",
                "tanh",
                "elu",
                "gelu",
                "hardSigmoid",
                "hardSwish",
                "leakyRelu",
                "linear",
                "sigmoid",
                "softplus",
                "softsign",
                "softmax",
                "reverse",
            ]
            .contains(&op_name)
        );

        // Step 2: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 3: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 4: If |allowedDataTypes| is given and it does not [=list/contain=] |input|'s [=MLOperand/dataType=], then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if let Some(allowed_data_types) = allowed_data_types {
            if !allowed_data_types.contains(&input_data_type) {
                return Err(Error::Type(c"unsupported input dataType".to_owned()));
            }
        }

        // Step 5: *Make graph connections:*
        // Step 5.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        // Note: implementation allocates backend output id before creating the DOM operand
        // so operator metadata can point to concrete input/output ids.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 5.2: Let |operator| be an [=operator=] for the |op| operation given |options|.
        let label = {
            let value = options.label.clone();
            if value.is_empty() {
                None
            } else {
                Some(value.clone().to_string())
            }
        };

        // Step 5.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 5.4: Set |operator|'s [=operator/input=] to |input|.
        // Step 5.5: Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: op_name.to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(op_name, serde_json::json!({})),
                label,
            });
        }

        // Step 5.1: Let |output| be the result of copying an MLOperand given |input|.
        let output = copy_an_mloperand(input, None, Some(output_id), can_gc);

        // Step 6: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#mlgraphbuilder-element-wise-binary-op>
    fn create_an_element_wise_binary_operation(
        &self,
        op_name: &str,
        a: &MLOperand,
        b: &MLOperand,
        options_label: Option<String>,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: [=Assert=]: |op| is one of "add", "sub", "mul", "div", "max", "min", "pow".
        debug_assert!(["add", "sub", "mul", "div", "max", "min", "pow"].contains(&op_name));

        // Step 2: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 3: If validating operand with this and any of |a| and |b| returns false, then throw a TypeError.
        if !self.validate_operand_ref(a) || !self.validate_operand_ref(b) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 4: If |a|'s dataType is not equal to |b|'s dataType, then throw a TypeError.
        // (We enforce this invariant here; the spec permits implementation-defined promotions.)
        let a_dtype = a.descriptor_data_type();
        if a_dtype != b.descriptor_data_type() {
            return Err(Error::Type(c"input dataType must match".to_owned()));
        }

        // Step 5: Let |outputShape| be the result of [=bidirectionally broadcasting=] |a|'s [=MLOperand/shape=] and |b|'s [=MLOperand/shape=].
        // Step 6: If |outputShape| is failure, then throw a TypeError.
        let out_shape =
            bidirectionally_broadcast_shapes(a.descriptor_shape(), b.descriptor_shape())
                .ok_or_else(|| {
                    Error::Type(c"shapes are not bidirectionally broadcastable".to_owned())
                })?;

        // Step 7: Let |outputDescriptor| be the result of creating an MLOperandDescriptor given |a|'s [=MLOperand/dataType=] and |outputShape|.
        let out_dtype_enum = match a_dtype {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            "int32" => MLOperandDataType::Int32,
            "uint32" => MLOperandDataType::Uint32,
            "int64" => MLOperandDataType::Int64,
            "uint64" => MLOperandDataType::Uint64,
            "int8" => MLOperandDataType::Int8,
            "uint8" => MLOperandDataType::Uint8,
            _ => MLOperandDataType::Float32,
        };

        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: out_shape.clone(),
        };

        // Step 8: *Make graph connections:*
        // Step 8.1: Let |output| be the result of creating an MLOperand given this and |outputDescriptor|.
        // Step 8.2: Let |operator| be an operator for the |op| operation, given |options|.
        // Step 8.3: Set |operator|'s [=operator/inputs=] to « |a|, |b| ».
        // Step 8.4: Set |operator|'s [=operator/output=] to |output|.
        // Note: implementation allocates backend ids before creating the DOM output object
        // so operation records can reference concrete input/output operand ids.
        let a_id = a
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let b_id = b
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

        let rust_operand =
            self.create_rust_operand(a_dtype, out_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        let attributes = serde_json::json!({});
        let label = options_label
            .map(|s| if s.is_empty() { None } else { Some(s) })
            .flatten();

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: op_name.to_string(),
                input_operands: vec![a_id, b_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(op_name, attributes),
                label,
            });
        }

        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );

        // Step 9: Return |output|.
        Ok(operand)
    }
}

impl MLGraphBuilderMethods<crate::DomTypeHolder> for MLGraphBuilder {
    fn Constructor(
        global: &GlobalScope,
        _proto: Option<HandleObject>,
        can_gc: CanGc,
        context: &MLContext,
    ) -> DomRoot<MLGraphBuilder> {
        MLGraphBuilder::new(context, global, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-input>
    fn Input(
        &self,
        name: DOMString,
        descriptor: &MLOperandDescriptor,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If |name| is empty, then throw a TypeError.
        if name.is_empty() {
            return Err(Error::Type(c"name is empty".to_owned()));
        }

        // Step 3: If any MLOperand in this graph's computational graph/inputs has [[name]] == |name|, then throw a TypeError.
        if let Some(ref gi) = self.graph_info.borrow().as_ref() {
            for &input_id in gi.input_operands.iter() {
                if let Some(op) = gi.operands.get(input_id as usize) {
                    if let Some(op_name) = &op.name {
                        if op_name.as_str() == name.str().as_ref() {
                            return Err(Error::Type(c"duplicate input name".to_owned()));
                        }
                    }
                }
            }
        }

        // Step 4: If MLOperandDescriptor/checking dimensions given |descriptor| returns false, then throw a TypeError.
        if !check_dimensions(descriptor) {
            return Err(Error::Type(c"invalid operand descriptor".to_owned()));
        }

        // Step 5: *Make graph connections:*
        // Step 5.1: Let |operand| be the result of creating an MLOperand given this and |descriptor|.
        // Step 5.2: Set |operand|.[[name]] to |name|.
        // Step 5.3: Add |operand| to this graph's computational graph/inputs and record backend operand id
        //          in the implementation-defined `GraphInfo` (append to `GraphInfo.operands` and `GraphInfo.input_operands`).
        // Note: this implements "Add operand to this's graph's inputs" — append the operand to the
        // implementation-defined `GraphInfo.operands` vector and record its id in `GraphInfo.input_operands`.
        let rust_operand = self.create_rust_operand(
            descriptor.dataType.as_str(),
            descriptor.shape.clone(),
            OperandKind::Input,
            Some(name.clone().to_string()),
        );
        let id = self.push_operand_to_graph(rust_operand, true);

        // Step 5.1: Let |operand| be the result of creating an MLOperand given this and |descriptor|.
        let operand = create_an_mloperand(
            self,
            Some(descriptor),
            None,
            Some(name.clone()),
            true,
            false,
            Some(id),
            can_gc,
        );

        // Step 5.2: Set |operand|.[[name]] to |name| (already supplied to the constructor above).
        // Step 5.3: DOM operands are no longer stored on the builder; backend bookkeeping
        // for the operand (id) was already recorded in `graph_info` above.

        // Step 6: Return |operand|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-constant>
    fn Constant(
        &self,
        descriptor: &MLOperandDescriptor,
        buffer: ArrayBufferViewOrArrayBuffer,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then return an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If MLOperandDescriptor/checking dimensions given |descriptor| returns false, then throw a TypeError.
        if !check_dimensions(descriptor) {
            return Err(Error::Type(c"invalid operand descriptor".to_owned()));
        }

        // Step 3: If validating buffer with descriptor given |buffer| and |descriptor| returns false, then throw a TypeError.
        // Step 3: TODO — validate the provided buffer against |descriptor| (byteLength, element type and shape).
        // TODO (spec: #api-mlgraphbuilder-constant-buffer): implement buffer validation (byte length, data type, shape).

        // Step 4: *Make graph connections:*
        // Step 4.1: Let |operand| be the result of creating an MLOperand given this and |descriptor|.
        let bytes: Vec<u8> = match buffer {
            ArrayBufferViewOrArrayBuffer::ArrayBufferView(view) => view.to_vec(),
            ArrayBufferViewOrArrayBuffer::ArrayBuffer(buf) => buf.to_vec(),
        };

        // ask context for a tensor id and queue the backend allocation
        let tensor_id = self.context().allocate_constant_tensor_for_builder(bytes);

        // Step 4.3: Add |operand| to this graph's computational graph/constants with |bytes| as value (persist bytes in GraphInfo).
        // Note: done below.

        // Create rustnn operand descriptor and operand id for the constant itself.
        let rust_operand = self.create_rust_operand(
            descriptor.dataType.as_str(),
            descriptor.shape.clone(),
            OperandKind::Constant,
            None,
        );
        let id = self.push_operand_to_graph(rust_operand, false);

        // Record mapping so the manager can resolve the bytes later.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.id_to_constant_tensor_operand_map
                .insert(id, tensor_id.to_string());
        }

        // Step 4.1: Let |operand| be the result of creating an MLOperand given this and |descriptor|.
        let operand = create_an_mloperand(
            self,
            Some(descriptor),
            None,
            None,
            false,
            true,
            Some(id),
            can_gc,
        );
        // Step 4.2: DOM operands are no longer kept by the builder; backend bookkeeping
        // for the operand (id and bytes) must be persisted in `graph_info`.
        // Step 5: Return |operand|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-constant-tensor>
    fn Constant_(&self, tensor: &MLTensor, can_gc: CanGc) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If tensor.[[context]] is not this.[[context]], throw a TypeError.
        if tensor.context() != self.context() {
            return Err(Error::Type(
                c"tensor is not owned by this builder's context".to_owned(),
            ));
        }

        // Step 2: If |tensor|.[[isDestroyed]] is true, then throw a TypeError.
        if tensor.is_destroyed() {
            return Err(Error::Type(c"tensor is destroyed".to_owned()));
        }

        // Step 3: If |tensor|.[[isConstant]] is false, then throw a TypeError.
        if !tensor.is_constant() {
            return Err(Error::Type(c"tensor is not constant".to_owned()));
        }

        // Step 4: If this can not build, throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 5: *Make graph connections:*
        // Step 5.1: Let |operand| be the result of creating an MLOperand given this and |tensor|.[[descriptor]].
        // Step 5.2: Set |operand|.[[constantTensor]] to |tensor|.
        // Step 5.3: Add |operand| to this graph's computational graph/constants with |tensor| as value.
        let rust_operand = self.create_rust_operand(
            tensor.data_type(),
            tensor.shape().iter().map(|&d| d as u32).collect(),
            OperandKind::Constant,
            None,
        );
        let id = self.push_operand_to_graph(rust_operand, false);
        // Step 5.3: Persist a reference so we can materialize the constant later.  The
        // library currently has no synchronous access to |tensor|'s data (it lives in the
        // manager thread), so we record the tensor id in
        // `id_to_constant_tensor_operand_map` and let the manager fill in the bytes at
        // dispatch time.  This mirrors the behaviour of the Python reference binding.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            // store tensor id as string; it will be parsed later by the manager
            gi.id_to_constant_tensor_operand_map
                .insert(id, tensor.tensor_id().to_string());
        }

        // Step 5.1: Let |operand| be the result of creating an MLOperand given this and |tensor|.[[descriptor]].
        let operand = create_an_mloperand(
            self,
            None,
            Some(tensor),
            None,
            false,
            true,
            Some(id),
            can_gc,
        );
        // Step 5.2: Set |operand|.[[constantTensor]] to |tensor| (TODO: MLOperand currently does not store a reference).
        // Step 5.3: DOM operands are not stored on the builder; persist tensor bytes in `graph_info`
        // so that Build() can materialize the constant.
        // Step 6: Return |operand|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-build>
    fn Build(&self, outputs: Record<USVString, DomRoot<MLOperand>>, can_gc: CanGc) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: If this can not build, then return a new promise in |realm| rejected with an InvalidStateError.
        if !self.can_build() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::InvalidState(None), can_gc);
            return p;
        }

        // Step 3: If |outputs| is empty, then return a new promise in |realm| rejected with a TypeError.
        if outputs.is_empty() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::Type(c"outputs is empty".to_owned()), can_gc);
            return p;
        }

        // Step 4: For each |operand| of |outputs|, run the per-operand validations from the spec.
        // Along the way we also ensure names are unique and don’t conflict with inputs.
        let mut seen_output_names = std::collections::HashSet::new();
        if let Some(ref gi) = self.graph_info.borrow().as_ref() {
            for (name, operand) in outputs.iter() {
                // Step 4.1: If |name| is empty, then return a rejected promise with a TypeError.
                if name.is_empty() {
                    let p = Promise::new(global, can_gc);
                    p.reject_error(Error::Type(c"operand name is empty".to_owned()), can_gc);
                    return p;
                }

                // Duplicate check for outputs.
                if !seen_output_names.insert(name.as_ref().to_string()) {
                    let p = Promise::new(global, can_gc);
                    p.reject_error(Error::Type(c"duplicate output name".to_owned()), can_gc);
                    return p;
                }

                // Check collision with any existing input name recorded in GraphInfo.
                for &input_id in gi.input_operands.iter() {
                    if let Some(op) = gi.operands.get(input_id as usize) {
                        if let Some(op_name) = &op.name {
                            if op_name.as_str() == name.as_ref() {
                                let p = Promise::new(global, can_gc);
                                p.reject_error(
                                    Error::Type(c"output name conflicts with input".to_owned()),
                                    can_gc,
                                );
                                return p;
                            }
                        }
                    }
                }

                // Step 4.2: If MLGraphBuilder/validating operand given |this| and |operand| returns false, then reject.
                if !self.validate_operand(operand) {
                    let p = Promise::new(global, can_gc);
                    p.reject_error(Error::Type(c"invalid operand".to_owned()), can_gc);
                    return p;
                }

                // Step 4.3: If |operand| is in this graph's input operands or constants, then reject.
                if operand.is_input() || operand.is_constant() {
                    let p = Promise::new(global, can_gc);
                    p.reject_error(
                        Error::Type(c"operand cannot be an input or constant".to_owned()),
                        can_gc,
                    );
                    return p;
                }

                // Step 4.4: If |operand|.[[constantTensor]] exists and |operand|.[[constantTensor]].[[isDestroyed]] is true, then reject.
                // TODO (spec: #api-mlgraphbuilder-build): MLOperand currently does not keep a reference to an
                // associated constant MLTensor. Add tracking or a helper so this validation can be implemented.
            }
        }

        // Step 5: Let |graph_info| be the GraphInfo we built and clear our slot.
        // Step 6: Set this.[[hasBuilt]] to true by dropping the GraphInfo from the builder.
        let mut graph_info = self
            .graph_info
            .borrow_mut()
            .take()
            .expect("can_build() ensured graph_info is Some");

        // Assign the names and output operand bookkeeping as before.
        for (name, operand) in outputs.iter() {
            if let Some(id) = operand.id() {
                if let Some(op) = graph_info.operands.get_mut(id as usize) {
                    op.name = Some(name.clone().to_string());
                }
            }
        }
        for operand in outputs.values() {
            if let Some(id) = operand.id() {
                graph_info.output_operands.push(id);
            }
        }

        // generate a unique id for the graph
        let graph_id = self.context().next_graph_id();

        // record the promise **and a copy of the GraphInfo** on the context.
        // The stored info will be used later when the compile callback runs
        // so that the MLGraph created for the script thread carries the same
        // descriptor data needed for dispatch validation.
        let p = Promise::new(global, can_gc);
        self.context()
            .register_build(graph_id, graph_info.clone(), p.clone());

        // send compile request to the manager.  Move the original `graph_info`
        // into the message; the backend/thread cache owns that copy.
        let cb = self.global().get_ml().get_or_setup_callback(global);
        let _ = self.global().webnn_sender().send(WebNNMsg::Compile(
            cb,
            graph_id,
            self.context().context_id(),
            graph_info,
        ));

        // Step 7/8: Return the promise without resolving it yet. It will be
        // resolved by the compile callback when compilation completes.
        p
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-argmin>
    fn ArgMin(
        &self,
        input: &MLOperand,
        axis: u32,
        options: &MLArgMinMaxOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Delegate to the shared helper that implements
        // `#mlgraphbuilder-argminmax-op` per the spec.
        self.mlgraphbuilder_argminmax_op("argMin", input, axis, options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-argmax>
    fn ArgMax(
        &self,
        input: &MLOperand,
        axis: u32,
        options: &MLArgMinMaxOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Delegate to the shared helper that implements
        // `#mlgraphbuilder-argminmax-op` per the spec.
        self.mlgraphbuilder_argminmax_op("argMax", input, axis, options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-where>
    fn Where(
        &self,
        condition: &MLOperand,
        true_value: &MLOperand,
        false_value: &MLOperand,
        _options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: Validate operands belong to this builder.
        if !self.validate_operand_ref(condition) ||
            !self.validate_operand_ref(true_value) ||
            !self.validate_operand_ref(false_value)
        {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Validate data types per spec
        if condition.descriptor_data_type() != "uint8" {
            return Err(Error::Type(
                c"condition must have dataType 'uint8'".to_owned(),
            ));
        }
        if true_value.descriptor_data_type() != false_value.descriptor_data_type() {
            return Err(Error::Type(
                c"trueValue and falseValue must have the same dataType".to_owned(),
            ));
        }

        // Infer output shape using rustnn shape inference helper
        let output_shape = match rustnn::shape_inference::infer_where_shape(
            condition.descriptor_shape(),
            true_value.descriptor_shape(),
            false_value.descriptor_shape(),
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(cformat!("{e}"))),
        };

        // Create output descriptor using trueValue's data type
        let out_dtype_str = true_value.descriptor_data_type();
        let out_dtype_enum = match out_dtype_str {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            "int32" => MLOperandDataType::Int32,
            "uint32" => MLOperandDataType::Uint32,
            "int64" => MLOperandDataType::Int64,
            "uint64" => MLOperandDataType::Uint64,
            "int8" => MLOperandDataType::Int8,
            "uint8" => MLOperandDataType::Uint8,
            _ => MLOperandDataType::Float32,
        };

        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: output_shape.clone(),
        };

        // Create backend operand and DOM operand
        let rust_operand = self.create_rust_operand(
            out_dtype_str,
            output_shape.clone(),
            OperandKind::Output,
            None,
        );
        let id = self.push_operand_to_graph(rust_operand, false);
        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(id),
            can_gc,
        );
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-cast>
    fn Cast(
        &self,
        input: &MLOperand,
        data_type: MLOperandDataType,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and input returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: if dataType is not output tensor’s allowed data types (according to this table), then throw a TypeError.
        // (This implementation accepts all MLOperandDataType enum variants.)

        // Make graph connections:
        // Let operator be an operator for the "cast" operation, given dataType and options.
        // Let output be the result of copying an MLOperand given input.
        // Set output.[[operator]] to operator.
        // Set operator’s input to input.
        // Set operator’s output to output.

        let in_shape = input.descriptor_shape();

        // Map enum to string for backend descriptor creation / attributes.
        let out_dtype_str = match data_type {
            MLOperandDataType::Float32 => "float32",
            MLOperandDataType::Float16 => "float16",
            MLOperandDataType::Int32 => "int32",
            MLOperandDataType::Uint32 => "uint32",
            MLOperandDataType::Int64 => "int64",
            MLOperandDataType::Uint64 => "uint64",
            MLOperandDataType::Int8 => "int8",
            MLOperandDataType::Uint8 => "uint8",
        };

        let desc = MLOperandDescriptor {
            dataType: data_type,
            shape: in_shape.clone(),
        };

        // Ensure the input has a backend operand id.
        let input_id = match input.id() {
            Some(i) => i,
            None => return Err(Error::Type(c"input operand has no backend id".to_owned())),
        };

        // Create backend operand for the output now (backend id required to record the operator).
        let rust_operand =
            self.create_rust_operand(out_dtype_str, in_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Build operation attributes and optional label (recording operator metadata).
        let attributes = serde_json::json!({ "dataType": out_dtype_str });

        let label = {
            let l = options.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "cast".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("cast", attributes),
                label,
            });
        }

        // Note: the spec's copy-an-mloperand algorithm copies input's descriptor,
        // but Cast changes output dataType. This binding applies an override descriptor
        // so DOM-visible output metadata matches the cast result.
        let operand = copy_an_mloperand(input, Some(&desc), Some(output_id), can_gc);
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-clamp>
    fn Clamp(
        &self,
        input: &MLOperand,
        options: &MLClampOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and input returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        let in_dtype = input.descriptor_data_type();

        // Step 3: Let |minValue| be the |options|.{{MLClampOptions/minValue}} if given, or Infinity otherwise.
        let min_opt = options.minValue.as_ref().map(|v| **v);

        // Step 5: Let |maxValue| be the |options|.{{MLClampOptions/maxValue}} if given, or -Infinity otherwise.
        let max_opt = options.maxValue.as_ref().map(|v| **v);

        // Step 4: Set |options|.{{MLClampOptions/minValue}} to the result of [=casting=] |minValue| to |input|'s [=MLOperand/dataType=].
        let min_casted = min_opt.map(|value| cast_number_to_data_type(value, in_dtype));

        // Step 6: Set |options|.{{MLClampOptions/maxValue}} to the result of [=casting=] |maxValue| to |input|'s [=MLOperand/dataType=].
        let max_casted = max_opt.map(|value| cast_number_to_data_type(value, in_dtype));

        // Step 7: If |options|.{{MLClampOptions/minValue}} is greater than |options|.{{MLClampOptions/maxValue}}, then [=exception/throw=] a {{TypeError}}.
        if let (Some(min_value), Some(max_value)) = (min_casted, max_casted) {
            if min_value > max_value {
                return Err(Error::Type(
                    c"minValue must not be greater than maxValue".to_owned(),
                ));
            }
        }

        // Note: the implementation records the casted values in operator attributes instead of mutating the bindings object.
        let min_value = min_casted.map(|value| match in_dtype {
            "float32" | "float16" => serde_json::json!(value),
            "int8" => serde_json::json!(value as i8),
            "uint8" => serde_json::json!(value as u8),
            "int32" => serde_json::json!(value as i32),
            "uint32" => serde_json::json!(value as u32),
            "int64" => serde_json::json!(value as i64),
            "uint64" => serde_json::json!(value as u64),
            _ => serde_json::json!(value),
        });
        let max_value = max_casted.map(|value| match in_dtype {
            "float32" | "float16" => serde_json::json!(value),
            "int8" => serde_json::json!(value as i8),
            "uint8" => serde_json::json!(value as u8),
            "int32" => serde_json::json!(value as i32),
            "uint32" => serde_json::json!(value as u32),
            "int64" => serde_json::json!(value as i64),
            "uint64" => serde_json::json!(value as u64),
            _ => serde_json::json!(value),
        });

        // Step 8.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        // Note: the implementation allocates the backend output id before constructing the DOM operand so the operator record can reference concrete ids.
        let in_shape = input.descriptor_shape();

        // Step 8.4: Set |operator|'s [=operator/input=] to |input|.
        let input_id = match input.id() {
            Some(i) => i,
            None => return Err(Error::Type(c"input operand has no backend id".to_owned())),
        };

        // Step 8.5: Set |operator|'s [=operator/output=] to |output|.
        let rust_operand =
            self.create_rust_operand(in_dtype, in_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 8.2: Let |operator| be an [=operator=] for the "clamp" operation, given |options|.
        let mut attributes = serde_json::json!({
            "hasMinValue": min_opt.is_some(),
            "hasMaxValue": max_opt.is_some(),
        });
        if let Some(mv) = min_value {
            attributes["minValue"] = mv;
        }
        if let Some(mv) = max_value {
            attributes["maxValue"] = mv;
        }

        // Optional label
        let label = {
            let l = options.parent.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // Step 8.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "clamp".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("clamp", attributes),
                label,
            });
        }

        // Step 8.1: Let |output| be the result of copying an MLOperand given |input|.
        let operand = copy_an_mloperand(input, None, Some(output_id), can_gc);

        // Step 9: Return |output|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-triangular>
    fn Triangular(
        &self,
        input: &MLOperand,
        options: &MLTriangularOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and input returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If input's rank is not one of its allowed ranks, then throw a TypeError.
        let in_shape = input.descriptor_shape();
        if let Err(e) = rustnn::shape_inference::infer_triangular_shape(&in_shape) {
            return Err(Error::Type(cformat!("{e}")));
        }

        // Step 4.1: Let output be the result of copying an MLOperand given input.
        let out_dtype_str = input.descriptor_data_type();

        let rust_operand =
            self.create_rust_operand(out_dtype_str, in_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 4.2: Let operator be an operator for the "triangular" operation, given options.
        let attributes = serde_json::json!({
            "upper": options.upper,
            "diagonal": options.diagonal,
        });

        let label = {
            let l = options.parent.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // Step 4.3-4.5: Record operator->input/output graph connections.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            let input_id = input
                .id()
                .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

            gi.operations.push(Operation {
                op_type: "triangular".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("triangular", attributes),
                label,
            });
        }

        // Step 5: Return output.
        let operand = copy_an_mloperand(input, None, Some(output_id), can_gc);
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-concat>
    fn Concat(
        &self,
        inputs: Vec<DomRoot<MLOperand>>,
        axis: u32,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and any item in inputs returns false, then throw a TypeError.
        if inputs.is_empty() {
            return Err(Error::Type(c"inputs is empty".to_owned()));
        }
        for inp in inputs.iter() {
            if !self.validate_operand(inp) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }
        }

        // Step 4: Let first be inputs[0].
        let first = inputs.get(0).expect("inputs non-empty; checked above");
        let first_shape = first.descriptor_shape();
        // Step 5: If axis is >= first.rank, then throw.
        if (axis as usize) >= first_shape.len() {
            return Err(Error::Type(c"axis out of range".to_owned()));
        }

        // Step 6: Let desc be descriptor created from first's dataType and shape.
        let first_dtype = first.descriptor_data_type();
        let mut desc = MLOperandDescriptor {
            dataType: match first_dtype {
                "float32" => MLOperandDataType::Float32,
                "float16" => MLOperandDataType::Float16,
                "int32" => MLOperandDataType::Int32,
                "uint32" => MLOperandDataType::Uint32,
                "int64" => MLOperandDataType::Int64,
                "uint64" => MLOperandDataType::Uint64,
                "int8" => MLOperandDataType::Int8,
                "uint8" => MLOperandDataType::Uint8,
                _ => MLOperandDataType::Float32,
            },
            shape: first_shape.clone(),
        };

        // Step 8: For each subsequent input validate dtype/rank and accumulate axis dimension.
        for index in 1..inputs.len() {
            let input = &inputs[index];
            let in_shape = input.descriptor_shape();

            if input.descriptor_data_type() != first_dtype {
                return Err(Error::Type(
                    c"input dataType must match first input".to_owned(),
                ));
            }
            if in_shape.len() != first_shape.len() {
                return Err(Error::Type(c"input rank must match first input".to_owned()));
            }

            for dim in 0..in_shape.len() {
                if dim != (axis as usize) {
                    if in_shape[dim] != first_shape[dim] {
                        return Err(Error::Type(
                            c"input shapes must match on all dims except axis".to_owned(),
                        ));
                    }
                } else {
                    // Sum sizes on axis and check validity (no overflow / non-zero).
                    let size_sum = (desc.shape[dim] as u128)
                        .checked_add(in_shape[dim] as u128)
                        .ok_or_else(|| Error::Type(c"dimension size overflow".to_owned()))?;
                    if size_sum == 0 || size_sum > (u32::MAX as u128) {
                        return Err(Error::Type(
                            c"invalid concatenated dimension size".to_owned(),
                        ));
                    }
                    desc.shape[dim] = size_sum as u32;
                }
            }
        }

        // Make graph connections: create backend operand and Operation record.
        let out_dtype_str = first_dtype;
        let rust_operand =
            self.create_rust_operand(out_dtype_str, desc.shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Build attributes and optional label
        let attributes = serde_json::json!({ "axis": axis });
        let label = {
            let l = options.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // Collect input backend ids
        let mut input_ids: Vec<u32> = Vec::with_capacity(inputs.len());
        for inp in inputs.iter() {
            let id = inp
                .id()
                .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
            input_ids.push(id);
        }

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "concat".to_string(),
                input_operands: input_ids,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("concat", attributes),
                label,
            });
        }

        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-conv2d>
    fn Conv2d(
        &self,
        input: &MLOperand,
        filter: &MLOperand,
        options: &MLConv2dOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: Validate operands belong to this builder (input, filter, bias if present).
        if !self.validate_operand_ref(input) || !self.validate_operand_ref(filter) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        if options.bias.is_some() {
            if !self.validate_operand_ref(options.bias.as_ref().unwrap()) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }
        }

        // Step 3: Input's dataType must be floating-point per spec.
        let in_dtype = input.descriptor_data_type();
        if in_dtype != "float32" && in_dtype != "float16" {
            return Err(Error::Type(
                c"input dataType must be 'float32' or 'float16'".to_owned(),
            ));
        }

        // Step 4: Input must be 4-D.
        let in_shape = input.descriptor_shape();
        if in_shape.len() != 4 {
            return Err(Error::Type(c"input must be a 4-D tensor".to_owned()));
        }

        // Step 5: Filter must be 4-D.
        let filter_shape = filter.descriptor_shape();
        if filter_shape.len() != 4 {
            return Err(Error::Type(c"filter must be a 4-D tensor".to_owned()));
        }

        // Step 6: Filter's dataType must match input's dataType.
        if filter.descriptor_data_type() != in_dtype {
            return Err(Error::Type(
                c"filter must have same dataType as input".to_owned(),
            ));
        }

        // Steps for options: apply defaults and validate lengths/values.
        let pads = match &options.padding {
            Some(p) if !p.is_empty() => p.clone(),
            _ => vec![0u32, 0u32, 0u32, 0u32],
        };
        if pads.len() != 4 {
            return Err(Error::Type(c"padding must be length 4".to_owned()));
        }

        let strides = match &options.strides {
            Some(s) if !s.is_empty() => s.clone(),
            _ => vec![1u32, 1u32],
        };
        if strides.len() != 2 {
            return Err(Error::Type(c"strides must be length 2".to_owned()));
        }
        if strides[0] < 1 || strides[1] < 1 {
            return Err(Error::Type(c"strides must be >= 1".to_owned()));
        }

        let dilations = match &options.dilations {
            Some(d) if !d.is_empty() => d.clone(),
            _ => vec![1u32, 1u32],
        };
        if dilations.len() != 2 {
            return Err(Error::Type(c"dilations must be length 2".to_owned()));
        }
        if dilations[0] < 1 || dilations[1] < 1 {
            return Err(Error::Type(c"dilations must be >= 1".to_owned()));
        }

        let groups = options.groups;
        if groups == 0 {
            return Err(Error::Type(c"groups must be >= 1".to_owned()));
        }

        // Determine logical inputChannels depending on inputLayout.
        let input_layout_str = match options.inputLayout {
            MLInputOperandLayout::Nchw => "nchw",
            MLInputOperandLayout::Nhwc => "nhwc",
        };
        let input_channels = if input_layout_str == "nchw" {
            in_shape[1]
        } else {
            in_shape[3]
        };

        // Determine filterInputChannels and outputChannels depending on filterLayout.
        let filter_layout_str = match options.filterLayout {
            MLConv2dFilterOperandLayout::Oihw => "oihw",
            MLConv2dFilterOperandLayout::Hwio => "hwio",
            MLConv2dFilterOperandLayout::Ohwi => "ohwi",
            MLConv2dFilterOperandLayout::Ihwo => "ihwo",
        };
        let (filter_input_channels, output_channels) = match filter_layout_str {
            "oihw" => (filter_shape[1], filter_shape[0]),
            "hwio" => (filter_shape[2], filter_shape[3]),
            "ohwi" => (filter_shape[3], filter_shape[0]),
            "ihwo" => (filter_shape[0], filter_shape[3]),
            _ => (filter_shape[1], filter_shape[0]),
        };

        // Validate grouped conv invariants.
        if (input_channels % groups) != 0 {
            return Err(Error::Type(c"inputChannels % groups must be 0".to_owned()));
        }
        if (input_channels / groups) != filter_input_channels {
            return Err(Error::Type(
                c"inputChannels / groups must equal filterInputChannels".to_owned(),
            ));
        }

        // If bias exists validate shape and dtype.
        if let Some(b) = options.bias.as_ref() {
            if b.descriptor_shape().len() != 1 {
                return Err(Error::Type(c"bias must be a 1-D tensor".to_owned()));
            }
            if b.descriptor_shape()[0] != output_channels {
                return Err(Error::Type(
                    c"bias size must equal the filter output channels".to_owned(),
                ));
            }
            if b.descriptor_data_type() != in_dtype {
                return Err(Error::Type(
                    c"bias must have same dataType as input".to_owned(),
                ));
            }
        }

        // Build rustnn shape-inference options and infer output shape.
        let rust_input_layout = if input_layout_str == "nchw" {
            rustnn::shape_inference::Conv2dInputLayout::Nchw
        } else {
            rustnn::shape_inference::Conv2dInputLayout::Nhwc
        };
        let rust_filter_layout = match filter_layout_str {
            "oihw" => rustnn::shape_inference::Conv2dFilterLayout::Oihw,
            "hwio" => rustnn::shape_inference::Conv2dFilterLayout::Hwio,
            "ohwi" => rustnn::shape_inference::Conv2dFilterLayout::Ohwi,
            "ihwo" => rustnn::shape_inference::Conv2dFilterLayout::Ihwo,
            _ => rustnn::shape_inference::Conv2dFilterLayout::Oihw,
        };
        let infer_options = rustnn::shape_inference::Conv2dOptions {
            strides: strides.clone(),
            dilations: dilations.clone(),
            pads: pads.clone(),
            groups,
            input_layout: rust_input_layout,
            filter_layout: rust_filter_layout,
        };

        let output_shape = match rustnn::shape_inference::infer_conv2d_shape(
            &in_shape,
            &filter_shape,
            &infer_options,
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(cformat!("{e}"))),
        };

        // Create output descriptor and backend operand.
        let out_dtype_enum = match in_dtype {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            _ => MLOperandDataType::Float32,
        };
        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: output_shape.clone(),
        };

        let rust_operand =
            self.create_rust_operand(in_dtype, output_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Build attributes and optional label.
        let mut attributes = serde_json::json!({
            "strides": strides,
            "dilations": dilations,
            "padding": pads,
            "groups": groups,
            "inputLayout": input_layout_str,
            "filterLayout": filter_layout_str,
        });
        if options.bias.is_some() {
            attributes["hasBias"] = serde_json::json!(true);
        }

        let label = {
            let l = options.parent.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // Collect input backend ids
        let mut input_ids: Vec<u32> = Vec::new();
        input_ids.push(
            input
                .id()
                .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?,
        );
        input_ids.push(
            filter
                .id()
                .ok_or_else(|| Error::Type(c"filter operand has no backend id".to_owned()))?,
        );
        if let Some(b) = options.bias.as_ref() {
            input_ids.push(b.id().unwrap());
        }

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "conv2d".to_string(),
                input_operands: input_ids,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("conv2d", attributes),
                label,
            });
        }

        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-batchnormalization>
    fn BatchNormalization(
        &self,
        input: &MLOperand,
        mean: &MLOperand,
        variance: &MLOperand,
        options: &MLBatchNormalizationOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and any of input, mean, variance,
        //         options.scale (if it exists), and options.bias (if it exists) returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) ||
            !self.validate_operand_ref(mean) ||
            !self.validate_operand_ref(variance)
        {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }
        if options.scale.is_some() {
            if !self.validate_operand_ref(options.scale.as_ref().unwrap()) {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
        }
        if options.bias.is_some() {
            if !self.validate_operand_ref(options.bias.as_ref().unwrap()) {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
        }

        // Step 3: If input’s dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        let in_dtype = input.descriptor_data_type();
        if in_dtype != "float32" && in_dtype != "float16" {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }

        // Step 4: If options.axis is not in the range 0 to input’s rank, exclusive, then throw a TypeError.
        let in_shape = input.descriptor_shape();
        let axis = options.axis as usize;
        if axis >= in_shape.len() {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }

        // Step 5: If mean’s dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        if mean.descriptor_data_type() != in_dtype {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }

        // Step 6: If mean’s shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if mean.descriptor_shape().len() != 1 {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }
        if mean.descriptor_shape()[0] != in_shape[axis] {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }

        // Step 7: If variance’s dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        if variance.descriptor_data_type() != in_dtype {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }

        // Step 8: If variance’s shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if variance.descriptor_shape().len() != 1 {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }
        if variance.descriptor_shape()[0] != in_shape[axis] {
            return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
        }

        // Step 9: Set options.epsilon to the result of casting options.epsilon to input’s dataType.
        let epsilon = cast_number_to_data_type(*options.epsilon, in_dtype);

        // Step 10.1: If options.scale exists and its dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        // Step 10.2: If options.scale exists and its shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if let Some(s) = options.scale.as_ref() {
            if s.descriptor_data_type() != in_dtype {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
            if s.descriptor_shape().len() != 1 {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
            if s.descriptor_shape()[0] != in_shape[axis] {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
        }

        // Step 11.1: If options.bias exists and its dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        // Step 11.2: If options.bias exists and its shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if let Some(b) = options.bias.as_ref() {
            if b.descriptor_data_type() != in_dtype {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
            if b.descriptor_shape().len() != 1 {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
            if b.descriptor_shape()[0] != in_shape[axis] {
                return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
            }
        }

        // Make graph connections per the spec:
        // Step 12.1: Let operator be an operator for the "batchNormalization" operation, given input, mean, variance and options.
        // Step 12.2: Let output be the result of creating an MLOperand given this and input.[[descriptor]].
        // Step 12.3: Set output.[[operator]] to operator.
        // Step 12.4: Set operator’s inputs to input, mean, and variance.
        // Step 12.5: If options.scale exists, then add it to operator’s inputs.
        // Step 12.6: If options.bias exists, then add it to operator’s inputs.
        // Step 12.7: Set operator’s output to output.

        // Infer output shape (implementation helper; spec's "Let output be the result of creating an MLOperand given this and input.[[descriptor]]").
        let output_shape = match rustnn::shape_inference::infer_batch_normalization_shape(in_shape)
        {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(cformat!("{e}"))),
        };

        // Create output descriptor and backend operand (maps to Step 12.2 & 12.7).
        let out_dtype_enum = match in_dtype {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            _ => MLOperandDataType::Float32,
        };
        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: output_shape.clone(),
        };

        let rust_operand =
            self.create_rust_operand(in_dtype, output_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Build operation attributes and optional label (recording operator metadata).
        let attributes = serde_json::json!({
            "epsilon": epsilon,
            "axis": options.axis,
            "scale": options.scale.as_ref().and_then(|operand| operand.id()),
            "bias": options.bias.as_ref().and_then(|operand| operand.id()),
            "hasScale": options.scale.is_some(),
            "hasBias": options.bias.is_some(),
        });

        let label = {
            let l = options.parent.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // Push an Operation record into the builder's GraphInfo.operations so the backend has the operator metadata (implements Steps 12.1, 12.4-12.6, 12.7).
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            let mut input_operands = vec![
                match input.id() {
                    Some(i) => i,
                    None => return Err(Error::Type(c"[batchNormalization_?_123]".to_owned())),
                },
                match mean.id() {
                    Some(i) => i,
                    None => return Err(Error::Type(c"[batchNormalization_?_123]".to_owned())),
                },
                match variance.id() {
                    Some(i) => i,
                    None => {
                        return Err(Error::Type(c"[batchNormalization_?_123]".to_owned()));
                    },
                },
            ];
            if let Some(s) = options.scale.as_ref() {
                input_operands.push(s.id().unwrap());
            }
            if let Some(b) = options.bias.as_ref() {
                input_operands.push(b.id().unwrap());
            }

            gi.operations.push(Operation {
                op_type: "batchNormalization".to_string(),
                input_operands,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("batchNormalization", attributes),
                label,
            });
        }

        // Step 13: Return output.
        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-abs>
    fn Abs(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "abs", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}}, {{MLOperandDataType/"int64"}}, {{MLOperandDataType/"int32"}}, {{MLOperandDataType/"int8"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16", "int64", "int32", "int8"];
        let output = self.create_an_element_wise_unary_operation(
            "abs",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-ceil>
    fn Ceil(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "ceil", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "ceil",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-cos>
    fn Cos(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "cos", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "cos",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-erf>
    fn Erf(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "erf", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "erf",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-exp>
    fn Exp(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "exp", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "exp",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-floor>
    fn Floor(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "floor", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "floor",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-identity>
    fn Identity(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "identity" |input|, and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let output =
            self.create_an_element_wise_unary_operation("identity", input, None, options, can_gc)?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-log>
    fn Log(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "log", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "log",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-neg>
    fn Neg(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "neg", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}}, {{MLOperandDataType/"int64"}}, {{MLOperandDataType/"int32"}}, {{MLOperandDataType/"int8"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16", "int64", "int32", "int8"];
        let output = self.create_an_element_wise_unary_operation(
            "neg",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reciprocal>
    fn Reciprocal(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "reciprocal", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "reciprocal",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-roundeven>
    fn RoundEven(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "roundEven", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "roundEven",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sin>
    fn Sin(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "sin", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "sin",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sign>
    fn Sign(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "sign", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}}, {{MLOperandDataType/"int64"}}, {{MLOperandDataType/"int32"}}, {{MLOperandDataType/"int8"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16", "int64", "int32", "int8"];
        let output = self.create_an_element_wise_unary_operation(
            "sign",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sqrt>
    fn Sqrt(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "sqrt", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "sqrt",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-tan>
    fn Tan(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of [=MLGraphBuilder/element-wise-unary-op|creating an element-wise unary operation=] given "tan", |input|, « {{MLOperandDataType/"float32"}}, {{MLOperandDataType/"float16"}} », and |options|.
        // Step 1.1: If that [=exception/throws=] an error, then re-[=exception/throw=] the error.
        let allowed_data_types = ["float32", "float16"];
        let output = self.create_an_element_wise_unary_operation(
            "tan",
            input,
            Some(&allowed_data_types),
            options,
            can_gc,
        )?;

        // Step 2: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-tanh>
    fn Tanh(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-tanh)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // Step 4: *Make graph connections:*
        // Step 4.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        // Note: implementation allocates backend output id before creating the DOM operand
        // so operator metadata can point to concrete input/output ids.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 4.2: Let |operator| be an [=operator=] for the "tanh" operation, given |options|.
        let label = {
            let value = options.label.clone();
            if value.is_empty() {
                None
            } else {
                Some(value.clone().to_string())
            }
        };

        // Step 4.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 4.4: Set |operator|'s [=operator/input=] to |input|.
        // Step 4.5: Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "tanh".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("tanh", serde_json::json!({})),
                label,
            });
        }

        // Step 4.1: Let |output| be the result of copying an MLOperand given |input|.
        let output = copy_an_mloperand(input, None, Some(output_id), can_gc);

        // Step 5: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-add>
    fn Add(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // The <dfn method for=MLGraphBuilder>add(|a|, |b|, |options|)</dfn> method steps are:
        // 1. Let |output| be the result of creating an element-wise binary operation given "add", |a|, |b|, and |options|.
        //    1. If that throws an error, then rethrow the error.
        // 2. Return |output|.
        let label = options.label.clone().to_string();
        self.create_an_element_wise_binary_operation("add", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sub>
    fn Sub(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // The <dfn method for=MLGraphBuilder>sub(|a|, |b|, |options|)</dfn> method steps are:
        // 1. Let |output| be the result of creating an element-wise binary operation given "sub", |a|, |b|, and |options|.
        //    1. If that throws an error, then rethrow the error.
        // 2. Return |output|.
        let label = options.label.clone().to_string();
        self.create_an_element_wise_binary_operation("sub", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-mul>
    fn Mul(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // The <dfn method for=MLGraphBuilder>mul(|a|, |b|, |options|)</dfn> method steps are:
        // 1. Let |output| be the result of creating an element-wise binary operation given "mul", |a|, |b|, and |options|.
        //    1. If that throws an error, then rethrow the error.
        // 2. Return |output|.
        let label = options.label.clone().to_string();
        self.create_an_element_wise_binary_operation("mul", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-div>
    fn Div(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // The <dfn method for=MLGraphBuilder>div(|a|, |b|, |options|)</dfn> method steps are:
        // 1. Let |output| be the result of creating an element-wise binary operation given "div", |a|, |b|, and |options|.
        //    1. If that throws an error, then rethrow the error.
        // 2. Return |output|.
        let label = options.label.clone().to_string();
        self.create_an_element_wise_binary_operation("div", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-max>
    fn Max(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // The <dfn method for=MLGraphBuilder>max(|a|, |b|, |options|)</dfn> method steps are:
        // 1. Let |output| be the result of creating an element-wise binary operation given "max", |a|, |b|, and |options|.
        //    1. If that throws an error, then rethrow the error.
        // 2. Return |output|.
        let label = options.label.clone().to_string();
        self.create_an_element_wise_binary_operation("max", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-min>
    fn Min(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // The <dfn method for=MLGraphBuilder>min(|a|, |b|, |options|)</dfn> method steps are:
        // 1. Let |output| be the result of creating an element-wise binary operation given "min", |a|, |b|, and |options|.
        //    1. If that throws an error, then rethrow the error.
        // 2. Return |output|.
        let label = options.label.clone().to_string();
        self.create_an_element_wise_binary_operation("min", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-pow>
    fn Pow(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // The <dfn method for=MLGraphBuilder>pow(|a|, |b|, |options|)</dfn> method steps are:
        // 1. Let |output| be the result of creating an element-wise binary operation given "pow", |a|, |b|, and |options|.
        //    1. If that throws an error, then rethrow the error.
        // 2. Return |output|.
        let label = options.label.clone().to_string();
        self.create_an_element_wise_binary_operation("pow", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-matmul>
    fn Matmul(&self, a: &MLOperand, b: &MLOperand, can_gc: CanGc) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If MLGraphBuilder/validating operand with this and any of |a| and |b| returns false, then throw a TypeError.
        if !self.validate_operand_ref(a) || !self.validate_operand_ref(b) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If the MLOperand/dataType of any of |a| or |b| is not one of its allowed data types, then throw a TypeError.
        // (This implementation requires matching data types; promotion is not performed.)
        let a_dtype = a.descriptor_data_type();
        if a_dtype != b.descriptor_data_type() {
            return Err(Error::Type(c"input dataType must match".to_owned()));
        }

        // Step 4 (substeps 4.1–4.11): Calculate the output shape. The rustnn helper validates ranks, transposes
        // and inner-dimension compatibility, broadcasts batch shapes and appends spatial dims.
        let output_shape = match rustnn::shape_inference::infer_matmul_shape(
            &a.descriptor_shape(),
            &b.descriptor_shape(),
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(cformat!("{e}"))),
        };

        let out_dtype_enum = match a_dtype {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            _ => MLOperandDataType::Float32,
        };
        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: output_shape.clone(),
        };

        let a_id = a
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let b_id = b
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

        let rust_operand =
            self.create_rust_operand(a_dtype, output_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 5: *Make graph connections:* record the `matmul` operator, its inputs and the output operand in GraphInfo.operations.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "matmul".to_string(),
                input_operands: vec![a_id, b_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("matmul", serde_json::json!({})),
                label: None,
            });
        }

        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-gemm>
    fn Gemm(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLGemmOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If MLGraphBuilder/validating operand with this and any of |a| and |b| returns false, then throw a TypeError.
        if !self.validate_operand_ref(a) || !self.validate_operand_ref(b) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // NOTE (early check): we do a quick ownership check for options.c here, but
        // full spec Step 12 (broadcastability + dtype checks for `c`) is performed
        // at the location in the spec order (see TODO below).
        if options.c.is_some() {
            if !self.validate_operand_ref(options.c.as_ref().unwrap()) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }
        }

        // Step 3: If the MLOperand/dataType of any of |a| or |b| is not one of its allowed data types, then throw a TypeError.
        let a_dtype = a.descriptor_data_type();
        if a_dtype != "float32" && a_dtype != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }
        // enforce matching data types (implementation-defined promotion not supported here)
        if a_dtype != b.descriptor_data_type() {
            return Err(Error::Type(c"input dataType must match".to_owned()));
        }

        // Step 4: Validate ranks.
        if a.descriptor_shape().len() != 2 || b.descriptor_shape().len() != 2 {
            return Err(Error::Type(c"gemm inputs must be 2-D tensors".to_owned()));
        }

        // Step 5: Set |options|.{{MLGemmOptions/alpha}} to the result of [=casting=] |options|.{{MLGemmOptions/alpha}} to |a|'s [=MLOperand/dataType=].
        let alpha = cast_number_to_data_type(*options.alpha, a_dtype);

        // Step 6: Set |options|.{{MLGemmOptions/beta}} to the result of [=casting=] |options|.{{MLGemmOptions/beta}} to |a|'s [=MLOperand/dataType=].
        let beta = cast_number_to_data_type(*options.beta, a_dtype);

        // Steps 7–11: Shape/rank/transposition validations and inner-dimension checks.
        // Step 4: If the MLOperand/rank of any of |a| or |b| is not its allowed rank, then throw a TypeError.
        // Step 7–11: clone shapes, apply transposes, and verify shapeA[1] == shapeB[0].
        // Implementation delegates these checks to `rustnn::shape_inference::infer_gemm_shape`.
        let mut shape_a = a.descriptor_shape().clone();
        let mut shape_b = b.descriptor_shape().clone();
        if options.aTranspose {
            shape_a.reverse();
        }
        if options.bTranspose {
            shape_b.reverse();
        }
        if shape_a[1] != shape_b[0] {
            return Err(Error::Type(
                c"shapeA[1] must equal shapeB[0] for gemm".to_owned(),
            ));
        }

        let output_shape = match rustnn::shape_inference::infer_gemm_shape(
            &a.descriptor_shape(),
            &b.descriptor_shape(),
            options.aTranspose,
            options.bTranspose,
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(cformat!("{e}"))),
        };

        // Step 12: Validate optional |c| against target shape « shapeA[0], shapeB[1] » and data type.
        if let Some(c_op) = options.c.as_ref() {
            if c_op.descriptor_data_type() != a_dtype {
                return Err(Error::Type(c"c must have same dataType as a".to_owned()));
            }
            if c_op.descriptor_shape().len() > 2 {
                return Err(Error::Type(c"c rank must be in range [0, 2]".to_owned()));
            }
            let c_target = vec![shape_a[0], shape_b[1]];
            if !unidirectionally_broadcastable_to_shape(c_op.descriptor_shape(), &c_target) {
                return Err(Error::Type(
                    c"c is not unidirectionally broadcastable to [shapeA[0], shapeB[1]]".to_owned(),
                ));
            }
        }

        // Step 13: Let |desc| be the result of creating an MLOperandDescriptor given |a|'s dataType and « |shapeA|[0], |shapeB|[1] ».
        let out_dtype_enum = match a_dtype {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            _ => MLOperandDataType::Float32,
        };
        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: output_shape.clone(),
        };

        // Step 14: *Make graph connections* — create the operator/output relationship per spec.
        // 14.1: Let |output| be the result of creating an MLOperand given this and |desc|.
        // (Implementation: create backend operand id first, then DOM operand is created at the end.)

        // Prepare input backend ids (14.4 & 14.5)
        let mut input_ids = vec![
            a.id()
                .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?,
            b.id()
                .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?,
        ];
        if let Some(c_op) = options.c.as_ref() {
            input_ids.push(c_op.id().unwrap());
        }

        // Create backend operand for the output now (backend id required to record the operator).
        let rust_operand =
            self.create_rust_operand(a_dtype, output_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Record operator attributes including alpha/beta and transpose flags.
        let mut attributes = serde_json::json!({
            "alpha": alpha,
            "beta": beta,
            "aTranspose": options.aTranspose,
            "bTranspose": options.bTranspose,
        });
        attributes["hasBias"] = serde_json::json!(options.c.is_some());

        // Optional label (operator metadata)
        let label = {
            let l = options.parent.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // 14.2–14.6: create operator record, set inputs and output (operator metadata persisted in GraphInfo.operations).
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "gemm".to_string(),
                input_operands: input_ids,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("gemm", attributes),
                label,
            });
        }

        // Step 15: Return |output| — create the DOM-level MLOperand for the output and return it.
        let operand = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-tile>
    fn Tile(
        &self,
        input: &MLOperand,
        repetitions: Vec<u32>,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If MLGraphBuilder/validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        let input_shape = input.descriptor_shape();

        // Step 3: If |repetitions|'s list/size is not equal to |input|'s MLOperand/rank, then throw a TypeError.
        if repetitions.len() != input_shape.len() {
            return Err(Error::Type(
                c"repetitions size must match input rank".to_owned(),
            ));
        }

        // Step 4: If |repetitions|'s values contain 0's, then throw a TypeError.
        if repetitions.contains(&0) {
            return Err(Error::Type(
                c"repetitions values must all be greater than 0".to_owned(),
            ));
        }

        // Step 5: Let |outputShape| be a copy of |input|'s MLOperand/shape.
        // Step 6: For each |index| in the range 0 to |outputShape|'s list/size, exclusive:
        // Step 6.1: Set |outputShape|[|index|] to |outputShape|[|index|] * |repetitions|[|index|].
        // Note: the multiplication loop is delegated to rustnn::shape_inference::infer_tile_shape.
        let output_shape =
            match rustnn::shape_inference::infer_tile_shape(&input_shape, &repetitions) {
                Ok(shape) => shape,
                Err(e) => return Err(Error::Type(cformat!("{e}"))),
            };

        // Step 7: Let |outputDescriptor| be the result of creating an MLOperandDescriptor
        // given |input|'s MLOperand/dataType and |outputShape|.
        let out_dtype_str = input.descriptor_data_type();
        let out_dtype_enum = match out_dtype_str {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            "int32" => MLOperandDataType::Int32,
            "uint32" => MLOperandDataType::Uint32,
            "int64" => MLOperandDataType::Int64,
            "uint64" => MLOperandDataType::Uint64,
            "int8" => MLOperandDataType::Int8,
            "uint8" => MLOperandDataType::Uint8,
            _ => MLOperandDataType::Float32,
        };
        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: output_shape.clone(),
        };

        // Step 8: *Make graph connections:*
        // Step 8.1: Let |output| be the result of creating an MLOperand given |outputDescriptor|.
        // Step 8.2: Let |operator| be an operator for the "tile" operation, given |options|.
        // Step 8.3: Set |operator|'s [=operator/input=] to |input|.
        // Step 8.4: Set |operator|'s [=operator/output=] to |output|.
        // Note: Implementation allocates backend operand ids before creating the DOM operand
        // so the operation record can point to concrete input/output ids.

        // Step 8.3: Set |operator|'s [=operator/input=] to |input|.
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

        // Step 8.4: Set |operator|'s [=operator/output=] to |output|.
        let rust_operand = self.create_rust_operand(
            out_dtype_str,
            output_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 8.2: Let |operator| be an operator for the "tile" operation, given |options|.
        let attributes = serde_json::json!({
            "repetitions": repetitions,
        });

        let label = {
            let l = options.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        // Step 8.4: Set |operator|'s operator/output to |output|.
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "tile".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("tile", attributes),
                label,
            });
        }

        // Step 8.1: Let |output| be the result of creating an MLOperand given |outputDescriptor|.
        let operand = copy_an_mloperand(input, Some(&desc), Some(output_id), can_gc);

        // Step 9: Return |output|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-elu>
    fn Elu(
        &self,
        input: &MLOperand,
        options: &MLEluOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-elu)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. Set |options|.{{MLEluOptions/alpha}} to the result of [=casting=] |options|.{{MLEluOptions/alpha}} to |input|'s [=MLOperand/dataType=].
        let alpha = cast_number_to_data_type(*options.alpha, input_data_type);

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "elu" operation, given |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "elu".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("elu", serde_json::json!({ "alpha": alpha })),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-gelu>
    fn Gelu(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-gelu)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "gelu" operation given |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "gelu".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("gelu", serde_json::json!({})),
                label: Self::label_from_operator_options(options),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-hardsigmoid>
    fn HardSigmoid(
        &self,
        input: &MLOperand,
        options: &MLHardSigmoidOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-hardSigmoid)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. Set |options|.{{MLHardSigmoidOptions/alpha}} to the result of [=casting=] |options|.{{MLHardSigmoidOptions/alpha}} to |input|'s [=MLOperand/dataType=].
        let alpha = cast_number_to_data_type(*options.alpha, input_data_type);

        // 1. Set |options|.{{MLHardSigmoidOptions/beta}} to the result of [=casting=] |options|.{{MLHardSigmoidOptions/beta}} to |input|'s [=MLOperand/dataType=].
        let beta = cast_number_to_data_type(*options.beta, input_data_type);

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "hardSigmoid" operation, given |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "hardSigmoid".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "hardSigmoid",
                    serde_json::json!({ "alpha": alpha, "beta": beta }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-hardswish>
    fn HardSwish(
        &self,
        input: &MLOperand,
        options: &MLHardSigmoidOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-hardSwish)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "hardSwish" operation, given |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "hardSwish".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("hardSwish", serde_json::json!({})),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-leakyrelu>
    fn LeakyRelu(
        &self,
        input: &MLOperand,
        options: &MLLeakyReluOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-leakyRelu)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. Set |options|.{{MLLeakyReluOptions/alpha}} to the result of [=casting=] |options|.{{MLLeakyReluOptions/alpha}} to |input|'s [=MLOperand/dataType=].
        let alpha = cast_number_to_data_type(*options.alpha, input_data_type);

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "leakyRelu" operation, given |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "leakyRelu".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "leakyRelu",
                    serde_json::json!({ "alpha": alpha }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-linear>
    fn Linear(
        &self,
        input: &MLOperand,
        options: &MLLinearOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-linear)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. Set |options|.{{MLLinearOptions/alpha}} to the result of [=casting=] |options|.{{MLLinearOptions/alpha}} to |input|'s [=MLOperand/dataType=].
        let alpha = cast_number_to_data_type(*options.alpha, input_data_type);

        // 1. Set |options|.{{MLLinearOptions/beta}} to the result of [=casting=] |options|.{{MLLinearOptions/beta}} to |input|'s [=MLOperand/dataType=].
        let beta = cast_number_to_data_type(*options.beta, input_data_type);

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "linear" operation, given |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "linear".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "linear",
                    serde_json::json!({ "alpha": alpha, "beta": beta }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-sigmoid>
    fn Sigmoid(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-sigmoid)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // Step 4: *Make graph connections:*
        // Step 4.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        // Note: implementation allocates backend output id before creating the DOM operand
        // so operator metadata can point to concrete input/output ids.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 4.2: Let |operator| be an [=operator=] for the "sigmoid" operation, given |options|.
        // Step 4.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 4.4: Set |operator|'s [=operator/input=] to |input|.
        // Step 4.5: Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "sigmoid".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("sigmoid", serde_json::json!({})),
                label: Self::label_from_operator_options(options),
            });
        }

        // Step 4.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let output = copy_an_mloperand(input, None, Some(output_id), can_gc);

        // Step 5: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-softplus>
    fn Softplus(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-softplus)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // Step 4: *Make graph connections:*
        // Step 4.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 4.2: Let |operator| be an [=operator=] for the "softplus" operation and |options|.
        // Step 4.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 4.4: Set |operator|'s [=operator/input=] to |input|.
        // Step 4.5: Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "softplus".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("softplus", serde_json::json!({})),
                label: Self::label_from_operator_options(options),
            });
        }

        // Step 4.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let output = copy_an_mloperand(input, None, Some(output_id), can_gc);

        // Step 5: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-softsign>
    fn Softsign(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-softsign)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // Step 4: *Make graph connections:*
        // Step 4.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 4.2: Let |operator| be an [=operator=] for the "softsign" operation and |options|.
        // Step 4.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 4.4: Set |operator|'s [=operator/input=] to |input|.
        // Step 4.5: Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "softsign".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("softsign", serde_json::json!({})),
                label: Self::label_from_operator_options(options),
            });
        }

        // Step 4.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let output = copy_an_mloperand(input, None, Some(output_id), can_gc);

        // Step 5: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reverse>
    fn Reverse(
        &self,
        input: &MLOperand,
        options: &MLReverseOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-reverse)), then [=exception/throw=] a {{TypeError}}.
        // Note: tensor-limits-reverse allows any data type, so this check is always satisfied.

        // Step 4: Let |inputRank| be |input|'s [=MLOperand/rank=].
        let input_rank = input.descriptor_shape().len();

        // Step 5: If |axes| is not given, then let |axes| be [=the range=] 0 to |inputRank|, exclusive.
        // Step 6: Otherwise, if |axes| contains duplicate values, or if any of its elements is not in [=the range=] 0 to |inputRank|, exclusive, then return failure.
        let axes = if let Some(ref supplied_axes) = options.axes {
            let mut seen = std::collections::HashSet::new();
            for &axis in supplied_axes.iter() {
                let axis_usize = axis as usize;
                if axis_usize >= input_rank || !seen.insert(axis) {
                    return Err(Error::Type(c"invalid axes".to_owned()));
                }
            }
            supplied_axes.clone()
        } else {
            (0..input_rank as u32).collect()
        };

        // Step 7: *Make graph connections:*
        // Step 7.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_data_type = input.descriptor_data_type();
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 7.2: Let |operator| be an [=operator=] for the "reverse" operation and |options|.
        // Step 7.3: Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // Step 7.4: Set |operator|'s [=operator/input=] to |input|.
        // Step 7.5: Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "reverse".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "reverse",
                    serde_json::json!({ "axes": axes }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // Step 7.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let output = copy_an_mloperand(input, None, Some(output_id), can_gc);

        // Step 8: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-equal>
    fn Equal(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "equal", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("equal", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-greater>
    fn Greater(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "greater", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("greater", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-greaterorequal>
    fn GreaterOrEqual(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "greaterOrEqual", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("greaterOrEqual", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-lesser>
    fn Lesser(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "lesser", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("lesser", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-lesserorequal>
    fn LesserOrEqual(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "lesserOrEqual", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("lesserOrEqual", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-notequal>
    fn NotEqual(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "notEqual", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("notEqual", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-logicaland>
    fn LogicalAnd(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "logicalAnd", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("logicalAnd", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-logicalor>
    fn LogicalOr(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "logicalOr", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("logicalOr", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-logicalxor>
    fn LogicalXor(
        &self,
        a: &MLOperand,
        b: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "logicalXor", |a|, |b|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("logicalXor", a, Some(b), options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-logicalnot>
    fn LogicalNot(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "logicalNot", |a|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("logicalNot", input, None, options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-isnan>
    fn IsNaN(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "isNaN", |a|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("isNaN", input, None, options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-isinfinite>
    fn IsInfinite(
        &self,
        input: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating an element-wise logical operation given "isInfinite", |a|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_an_element_wise_logical_operation("isInfinite", input, None, options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reshape>
    fn Reshape(
        &self,
        input: &MLOperand,
        new_shape: Vec<u32>,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Validate the reshape sizes against input shape.
        rustnn::shape_inference::validate_reshape(input.descriptor_shape(), &new_shape)
            .map_err(|e| Error::Type(cformat!("{e}")))?;

        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: new_shape.clone(),
        };

        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, new_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 4: Make graph connections for the "reshape" operator.
        self.push_unary_operation(
            "reshape",
            input_id,
            output_id,
            serde_json::json!({"newShape": new_shape}),
            Self::label_from_operator_options(options),
        );

        Ok(copy_an_mloperand(
            input,
            Some(&desc),
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-expand>
    fn Expand(
        &self,
        input: &MLOperand,
        new_shape: Vec<u32>,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Let |outputShape| be the result of inferring expanded output shape.
        let output_shape =
            rustnn::shape_inference::infer_expand_shape(input.descriptor_shape(), &new_shape)
                .map_err(|e| Error::Type(cformat!("{e}")))?;
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);
        // Step 4: Make graph connections for the "expand" operator.
        self.push_unary_operation(
            "expand",
            input_id,
            output_id,
            serde_json::json!({"newShape": new_shape}),
            Self::label_from_operator_options(options),
        );
        Ok(copy_an_mloperand(
            input,
            Some(&desc),
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-slice>
    fn Slice(
        &self,
        input: &MLOperand,
        starts: Vec<u32>,
        sizes: Vec<u32>,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Let |outputShape| be the result of slice-shape validation/inference.
        let output_shape =
            rustnn::shape_inference::infer_slice_shape(input.descriptor_shape(), &starts, &sizes)
                .map_err(|e| Error::Type(cformat!("{e}")))?;
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);
        // Step 4: Make graph connections for the "slice" operator.
        self.push_unary_operation(
            "slice",
            input_id,
            output_id,
            serde_json::json!({"starts": starts, "sizes": sizes}),
            Self::label_from_operator_options(options),
        );
        Ok(copy_an_mloperand(
            input,
            Some(&desc),
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-gather>
    fn Gather(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        options: &MLGatherOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) || !self.validate_operand_ref(indices) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Validate indices dataType.
        if !["int32", "uint32", "int64"].contains(&indices.descriptor_data_type()) {
            return Err(Error::Type(
                c"unsupported indices dataType for gather".to_owned(),
            ));
        }

        // Steps 4-8: Validate axis against input rank.
        let input_shape = input.descriptor_shape();
        let input_rank = input_shape.len();
        let axis = options.axis;
        if (axis as usize) >= input_rank {
            return Err(Error::Type(c"axis out of range".to_owned()));
        }

        // Steps 9-17: Derive output shape.
        let output_shape = rustnn::shape_inference::infer_gather_shape(
            &input_shape,
            indices.descriptor_shape(),
            axis,
        )
        .map_err(|e| Error::Type(cformat!("{e}")))?;
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let indices_id = indices
            .id()
            .ok_or_else(|| Error::Type(c"indices operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);
        self.push_binary_operation(
            "gather",
            vec![input_id, indices_id],
            output_id,
            serde_json::json!({"axis": axis}),
            Self::label_from_operator_options(&options.parent),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-gatherelements>
    fn GatherElements(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        options: &MLGatherOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) || !self.validate_operand_ref(indices) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Validate indices dataType.
        if !["int32", "uint32", "int64"].contains(&indices.descriptor_data_type()) {
            return Err(Error::Type(
                c"unsupported indices dataType for gatherElements".to_owned(),
            ));
        }

        // Step 4: Validate ranks.
        let input_shape = input.descriptor_shape();
        let indices_shape = indices.descriptor_shape();
        if input_shape.is_empty() || indices_shape.is_empty() {
            return Err(Error::Type(
                c"input and indices must have rank >= 1".to_owned(),
            ));
        }
        if input_shape.len() != indices_shape.len() {
            return Err(Error::Type(
                c"indices must have same rank as input".to_owned(),
            ));
        }

        // Step 5-8: Validate axis and expected indices shape.
        let axis = options.axis as usize;
        if axis >= input_shape.len() {
            return Err(Error::Type(c"axis out of range".to_owned()));
        }
        let mut indices_shape_expected = input_shape.clone();
        indices_shape_expected[axis] = indices_shape[axis];
        if indices_shape != &indices_shape_expected {
            return Err(Error::Type(
                c"indices shape does not match expected gatherElements shape".to_owned(),
            ));
        }

        // Output shape follows the spec-computed shape, which equals indices shape here.
        let output_shape = indices.descriptor_shape().clone();
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let indices_id = indices
            .id()
            .ok_or_else(|| Error::Type(c"indices operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);
        self.push_binary_operation(
            "gatherElements",
            vec![input_id, indices_id],
            output_id,
            serde_json::json!({"axis": options.axis}),
            Self::label_from_operator_options(&options.parent),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-gathernd>
    fn GatherND(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) || !self.validate_operand_ref(indices) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Validate indices dataType.
        if !["int32", "uint32", "int64"].contains(&indices.descriptor_data_type()) {
            return Err(Error::Type(
                c"unsupported indices dataType for gatherND".to_owned(),
            ));
        }

        // Step 4: Validate ranks.
        let input_shape = input.descriptor_shape();
        if input_shape.is_empty() {
            return Err(Error::Type(c"input must have rank >= 1".to_owned()));
        }
        let indices_shape = indices.descriptor_shape();
        if indices_shape.is_empty() {
            return Err(Error::Type(c"indices must have rank >= 1".to_owned()));
        }

        // Steps 5-13: Derive output shape components.
        let k = indices_shape[indices_shape.len() - 1] as usize;
        if k > input_shape.len() {
            return Err(Error::Type(
                c"indices last dimension out of range".to_owned(),
            ));
        }
        let mut output_shape = indices_shape[..indices_shape.len() - 1].to_vec();
        output_shape.extend_from_slice(&input_shape[k..]);
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let indices_id = indices
            .id()
            .ok_or_else(|| Error::Type(c"indices operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);
        self.push_binary_operation(
            "gatherND",
            vec![input_id, indices_id],
            output_id,
            serde_json::json!({}),
            Self::label_from_operator_options(options),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-pad>
    fn Pad(
        &self,
        input: &MLOperand,
        beginning_padding: Vec<u32>,
        ending_padding: Vec<u32>,
        options: &MLPadOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If |beginningPadding|'s and |endingPadding|'s sizes are not both equal to |input|'s rank, then throw a TypeError.
        let input_shape = input.descriptor_shape();
        let input_rank = input_shape.len();
        if beginning_padding.len() != input_rank || ending_padding.len() != input_rank {
            return Err(Error::Type(
                c"beginningPadding and endingPadding sizes must both equal input rank".to_owned(),
            ));
        }

        // Step 4: Let |desc| be a copy of |input|.[[descriptor]].
        // Step 5: Let |outputShape| be a copy of |input|'s shape.
        let mut output_shape = input_shape.clone();

        // Step 6: For each index, validate reflection mode constraints and apply beginning/ending padding.
        let mode = match options.mode {
            MLPaddingMode::Constant => "constant",
            MLPaddingMode::Edge => "edge",
            MLPaddingMode::Reflection => "reflection",
        };
        for index in 0..input_rank {
            if mode == "reflection" {
                if beginning_padding[index] >= output_shape[index] {
                    return Err(Error::Type(
                        c"beginningPadding[index] must be less than input dimension in reflection mode".to_owned(),
                    ));
                }
                if ending_padding[index] >= output_shape[index] {
                    return Err(Error::Type(
                        c"endingPadding[index] must be less than input dimension in reflection mode"
                            .to_owned(),
                    ));
                }
            }

            output_shape[index] = output_shape[index]
                .checked_add(beginning_padding[index])
                .and_then(|value| value.checked_add(ending_padding[index]))
                .ok_or_else(|| Error::Type(c"invalid output shape".to_owned()))?;
        }

        // Step 7: If any item in |outputShape| is not a valid dimension, then throw a TypeError.
        let output_desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(input.descriptor_data_type()),
            shape: output_shape.clone(),
        };
        if !check_dimensions(&output_desc) {
            return Err(Error::Type(c"invalid output shape".to_owned()));
        }

        // Step 8: Set |options|.value to the result of casting it to |input|'s data type.
        // Note: value casting is represented in operator attributes for backend consumption.
        let out_dtype = input.descriptor_data_type();
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        let mut padding = beginning_padding.clone();
        padding.extend_from_slice(&ending_padding);
        let value = cast_number_to_data_type(options.value, out_dtype);
        let value = match out_dtype {
            "float32" | "float16" => serde_json::json!(value),
            "int8" => serde_json::json!(value as i8),
            "uint8" => serde_json::json!(value as u8),
            "int32" => serde_json::json!(value as i32),
            "uint32" => serde_json::json!(value as u32),
            "int64" => serde_json::json!(value as i64),
            "uint64" => serde_json::json!(value as u64),
            _ => serde_json::json!(value),
        };

        self.push_unary_operation(
            "pad",
            input_id,
            output_id,
            serde_json::json!({
                "beginningPadding": beginning_padding,
                "endingPadding": ending_padding,
                "padding": padding,
                "mode": mode,
                "value": value,
            }),
            Self::label_from_operator_options(&options.parent),
        );

        // Step 9: Return |output|.
        Ok(copy_an_mloperand(
            input,
            Some(&desc),
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-softmax>
    fn Softmax(
        &self,
        input: &MLOperand,
        options: &MLSoftmaxOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-softmax)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. If |axis| is greater than or equal to |input|'s [=MLOperand/rank=], then [=exception/throw=] a {{TypeError}}.
        let axis = options.axis;
        if (axis as usize) >= input.descriptor_shape().len() {
            return Err(Error::Type(c"axis out of range".to_owned()));
        }

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "softmax" operation, given |axis| and |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "softmax".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "softmax",
                    serde_json::json!({ "axis": axis }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducesum>
    fn ReduceSum(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16", "int32", "uint32", "int64", "uint64"];
        self.create_reduction_operation(
            "reduceSum",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducemean>
    fn ReduceMean(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16"];
        self.create_reduction_operation(
            "reduceMean",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducemax>
    fn ReduceMax(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        self.create_reduction_operation("reduceMax", input, options, None, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducemin>
    fn ReduceMin(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        self.create_reduction_operation("reduceMin", input, options, None, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reduceproduct>
    fn ReduceProduct(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16", "int32", "uint32", "int64", "uint64"];
        self.create_reduction_operation(
            "reduceProduct",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducel1>
    fn ReduceL1(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16", "int32", "uint32", "int64", "uint64"];
        self.create_reduction_operation(
            "reduceL1",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducel2>
    fn ReduceL2(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16"];
        self.create_reduction_operation(
            "reduceL2",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducelogsum>
    fn ReduceLogSum(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16"];
        self.create_reduction_operation(
            "reduceLogSum",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducelogsumexp>
    fn ReduceLogSumExp(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16"];
        self.create_reduction_operation(
            "reduceLogSumExp",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-reducesumsquare>
    fn ReduceSumSquare(
        &self,
        input: &MLOperand,
        options: &MLReduceOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        let allowed_data_types = ["float32", "float16", "int32", "uint32", "int64", "uint64"];
        self.create_reduction_operation(
            "reduceSumSquare",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-quantizelinear>
    fn QuantizeLinear(
        &self,
        input: &MLOperand,
        scale: &MLOperand,
        zero_point: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        self.create_quantize_or_dequantize_linear_operation(
            "quantizeLinear",
            input,
            scale,
            zero_point,
            true,
            options,
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-dequantizelinear>
    fn DequantizeLinear(
        &self,
        input: &MLOperand,
        scale: &MLOperand,
        zero_point: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        self.create_quantize_or_dequantize_linear_operation(
            "dequantizeLinear",
            input,
            scale,
            zero_point,
            false,
            options,
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-scatterelements>
    fn ScatterElements(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        updates: &MLOperand,
        options: &MLGatherOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        self.create_scatter_elements_operation(
            input,
            indices,
            updates,
            options.axis as i32,
            &options.parent,
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-scatternd>
    fn ScatterND(
        &self,
        input: &MLOperand,
        indices: &MLOperand,
        updates: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        self.create_scatter_nd_operation(input, indices, updates, options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-cumulativesum>
    fn CumulativeSum(
        &self,
        input: &MLOperand,
        axis: u32,
        options: &MLCumulativeSumOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // 1. If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // 1. If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // 1. If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-cumulativesum)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if !["float32", "float16", "int32", "uint32", "int64", "uint64"].contains(&input_data_type)
        {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // 1. If |axis| is greater than or equal to |input|'s [=MLOperand/rank=], then [=exception/throw=] a {{TypeError}}.
        if (axis as usize) >= input.descriptor_shape().len() {
            return Err(Error::Type(c"axis out of range".to_owned()));
        }

        // 1. *Make graph connections:*
        // 1. Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_shape = input.descriptor_shape();
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // 1. Let |operator| be an [=operator=] for the "cumulativeSum" operation and |options|.
        // 1. Set |output|.{{MLOperand/[[operator]]}} to |operator|.
        // 1. Set |operator|'s [=operator/input=] to |input|.
        // 1. Set |operator|'s [=operator/output=] to |output|.
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "cumulativeSum".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "cumulativeSum",
                    serde_json::json!({
                        "axis": axis,
                        "exclusive": options.exclusive,
                        "reversed": options.reversed,
                    }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // 1. Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-averagepool2d>
    fn AveragePool2d(
        &self,
        input: &MLOperand,
        options: &MLPool2dOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating a pool2d operation given "averagePool2d", |input|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        let allowed_data_types = ["float32", "float16"];
        self.create_a_pooling_operation(
            "averagePool2d",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-l2pool2d>
    fn L2Pool2d(
        &self,
        input: &MLOperand,
        options: &MLPool2dOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating a pool2d operation given "l2Pool2d", |input|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        let allowed_data_types = ["float32", "float16"];
        self.create_a_pooling_operation(
            "l2Pool2d",
            input,
            options,
            Some(&allowed_data_types),
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-maxpool2d>
    fn MaxPool2d(
        &self,
        input: &MLOperand,
        options: &MLPool2dOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating a pool2d operation given "maxPool2d", |input|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_a_pooling_operation("maxPool2d", input, options, None, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-convtranspose2d>
    fn ConvTranspose2d(
        &self,
        input: &MLOperand,
        filter: &MLOperand,
        options: &MLConvTranspose2dOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: Let |output| be the result of creating a convTranspose2d operation given |input|, |filter|, and |options|.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |output|.
        self.create_conv_transpose2d_operation(input, filter, options, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-instancenormalization>
    fn InstanceNormalization(
        &self,
        input: &MLOperand,
        options: &MLInstanceNormalizationOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and any of |input|, |options|.scale (if it exists), and |options|.bias (if it exists) returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        if let Some(scale) = options.scale.as_ref() {
            if !self.validate_operand(scale) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }
        }
        if let Some(bias) = options.bias.as_ref() {
            if !self.validate_operand(bias) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-instanceNormalization)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        // Step 4: If |input|'s [=MLOperand/rank=] is not its [=/allowed rank=], then [=exception/throw=] a {{TypeError}}.
        let input_shape = input.descriptor_shape();
        if input_shape.len() != 4 {
            return Err(Error::Type(c"input must be a 4-D tensor".to_owned()));
        }

        // Step 5: Set |options|.epsilon to the result of casting |options|.epsilon to |input|'s [=MLOperand/dataType=].
        let epsilon = cast_number_to_data_type(*options.epsilon, input_data_type);

        // Step 6: Let |axis| be 1 if |options|.layout is "nchw", and 3 otherwise.
        let axis = if options.layout == MLInputOperandLayout::Nchw {
            1usize
        } else {
            3usize
        };

        // Step 7/8: Validate optional |scale| and |bias| dataType and shape.
        if let Some(scale) = options.scale.as_ref() {
            if scale.descriptor_data_type() != input_data_type {
                return Err(Error::Type(
                    c"scale must have same dataType as input".to_owned(),
                ));
            }
            if scale.descriptor_shape().len() != 1 ||
                scale.descriptor_shape()[0] != input_shape[axis]
            {
                return Err(Error::Type(c"invalid scale shape".to_owned()));
            }
        }
        if let Some(bias) = options.bias.as_ref() {
            if bias.descriptor_data_type() != input_data_type {
                return Err(Error::Type(
                    c"bias must have same dataType as input".to_owned(),
                ));
            }
            if bias.descriptor_shape().len() != 1 || bias.descriptor_shape()[0] != input_shape[axis]
            {
                return Err(Error::Type(c"invalid bias shape".to_owned()));
            }
        }

        // Step 9: *Make graph connections:*
        // Step 9.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 9.2: Let |operator| be an [=operator=] for the "instanceNormalization" operation, given |options|.
        // Step 9.3/9.4/9.5/9.6/9.7: Set output/operator links and optional extra inputs.
        let mut input_operands = vec![input_id];
        if let Some(scale) = options.scale.as_ref() {
            input_operands.push(
                scale
                    .id()
                    .ok_or_else(|| Error::Type(c"scale operand has no backend id".to_owned()))?,
            );
        }
        if let Some(bias) = options.bias.as_ref() {
            input_operands.push(
                bias.id()
                    .ok_or_else(|| Error::Type(c"bias operand has no backend id".to_owned()))?,
            );
        }

        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "instanceNormalization".to_string(),
                input_operands,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "instanceNormalization",
                    serde_json::json!({
                        "epsilon": epsilon,
                        "layout": if options.layout == MLInputOperandLayout::Nchw { "nchw" } else { "nhwc" },
                        "hasScale": options.scale.is_some(),
                        "hasBias": options.bias.is_some(),
                    }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // Step 10: Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-layernormalization>
    fn LayerNormalization(
        &self,
        input: &MLOperand,
        options: &MLLayerNormalizationOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and any of |input|, |options|.scale (if it exists), and |options|.bias (if it exists) returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }
        if let Some(scale) = options.scale.as_ref() {
            if !self.validate_operand(scale) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }
        }
        if let Some(bias) = options.bias.as_ref() {
            if !self.validate_operand(bias) {
                return Err(Error::Type(c"invalid operand".to_owned()));
            }
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-layerNormalization)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if input_data_type != "float32" && input_data_type != "float16" {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        let input_shape = input.descriptor_shape();
        let input_rank = input_shape.len();

        // Step 4/5: Resolve and validate |axes|.
        let axes = if let Some(ref explicit_axes) = options.axes {
            let mut seen = std::collections::HashSet::new();
            for &axis in explicit_axes.iter() {
                if (axis as usize) >= input_rank || !seen.insert(axis) {
                    return Err(Error::Type(c"invalid axes".to_owned()));
                }
            }
            explicit_axes.clone()
        } else if input_rank > 1 {
            (1..input_rank as u32).collect()
        } else {
            Vec::new()
        };

        // Step 6: Set |options|.epsilon to the result of casting |options|.epsilon to |input|'s [=MLOperand/dataType=].
        let epsilon = cast_number_to_data_type(*options.epsilon, input_data_type);

        // Step 7/8: Validate optional |scale| and |bias| rank and type.
        if let Some(scale) = options.scale.as_ref() {
            if scale.descriptor_data_type() != input_data_type {
                return Err(Error::Type(
                    c"scale must have same dataType as input".to_owned(),
                ));
            }
            if scale.descriptor_shape().len() != axes.len() {
                return Err(Error::Type(c"invalid scale rank".to_owned()));
            }
        }
        if let Some(bias) = options.bias.as_ref() {
            if bias.descriptor_data_type() != input_data_type {
                return Err(Error::Type(
                    c"bias must have same dataType as input".to_owned(),
                ));
            }
            if bias.descriptor_shape().len() != axes.len() {
                return Err(Error::Type(c"invalid bias rank".to_owned()));
            }
        }

        // Step 9: Validate each axis maps to matching dimensions in scale/bias when present.
        for (index, &axis) in axes.iter().enumerate() {
            let axis_index = axis as usize;
            if axis_index >= input_rank {
                return Err(Error::Type(c"axis out of range".to_owned()));
            }
            let size = input_shape[axis_index];
            if let Some(scale) = options.scale.as_ref() {
                if scale.descriptor_shape()[index] != size {
                    return Err(Error::Type(c"invalid scale shape".to_owned()));
                }
            }
            if let Some(bias) = options.bias.as_ref() {
                if bias.descriptor_shape()[index] != size {
                    return Err(Error::Type(c"invalid bias shape".to_owned()));
                }
            }
        }

        // Step 10: *Make graph connections:*
        // Step 10.1: Let |output| be the result of [=copying an MLOperand=] given |input|.
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand = self.create_rust_operand(
            input_data_type,
            input_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 10.2/10.3/10.4/10.5/10.6/10.7: Record operator metadata and optional extra inputs.
        let mut input_operands = vec![input_id];
        if let Some(scale) = options.scale.as_ref() {
            input_operands.push(
                scale
                    .id()
                    .ok_or_else(|| Error::Type(c"scale operand has no backend id".to_owned()))?,
            );
        }
        if let Some(bias) = options.bias.as_ref() {
            input_operands.push(
                bias.id()
                    .ok_or_else(|| Error::Type(c"bias operand has no backend id".to_owned()))?,
            );
        }

        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "layerNormalization".to_string(),
                input_operands,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "layerNormalization",
                    serde_json::json!({
                        "axes": axes,
                        "epsilon": epsilon,
                        "hasScale": options.scale.is_some(),
                        "hasBias": options.bias.is_some(),
                    }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        // Step 11: Return |output|.
        Ok(copy_an_mloperand(input, None, Some(output_id), can_gc))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-prelu>
    fn Prelu(
        &self,
        input: &MLOperand,
        slope: &MLOperand,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }
        if !self.validate_operand_ref(input) || !self.validate_operand_ref(slope) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: Validate dataTypes per tensor limits table.
        let input_data_type = input.descriptor_data_type();
        if !["float32", "float16", "int64", "int32", "int8"].contains(&input_data_type) {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }
        if slope.descriptor_data_type() != input_data_type {
            return Err(Error::Type(
                c"slope must have same dataType as input".to_owned(),
            ));
        }

        // Step 4: Validate bidirectional broadcastability and infer output shape.
        let output_shape = rustnn::shape_inference::infer_prelu_shape(
            input.descriptor_shape(),
            slope.descriptor_shape(),
        )
        .map_err(|e| Error::Type(cformat!("{e}")))?;
        let out_dtype = input_data_type;
        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(out_dtype),
            shape: output_shape.clone(),
        };
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let slope_id = slope
            .id()
            .ok_or_else(|| Error::Type(c"slope operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(out_dtype, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);
        self.push_binary_operation(
            "prelu",
            vec![input_id, slope_id],
            output_id,
            serde_json::json!({}),
            Self::label_from_operator_options(options),
        );
        Ok(create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        ))
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-resample2d>
    fn Resample2d(
        &self,
        input: &MLOperand,
        options: &MLResample2dOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If [=this=] [=MLGraphBuilder/can not build=], then [=exception/throw=] an "{{InvalidStateError}}" {{DOMException}}.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If [=MLGraphBuilder/validating operand=] with [=this=] and |input| returns false, then [=exception/throw=] a {{TypeError}}.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        // Step 3: If |input|'s [=MLOperand/dataType=] is not one of its [=/allowed data types=] (according to [this table](#tensor-limits-resample2d)), then [=exception/throw=] a {{TypeError}}.
        let input_data_type = input.descriptor_data_type();
        if !["float32", "float16", "uint8", "int8"].contains(&input_data_type) {
            return Err(Error::Type(c"unsupported input dataType".to_owned()));
        }

        let input_shape = input.descriptor_shape();
        if input_shape.len() != 4 {
            return Err(Error::Type(c"input must be a 4-D tensor".to_owned()));
        }

        // Step 5: Resolve and validate |scales|.
        let scales: Vec<f32> = if let Some(ref provided_scales) = options.scales {
            if provided_scales.len() != 2 || provided_scales.iter().any(|value| **value <= 0.0) {
                return Err(Error::Type(c"invalid scales".to_owned()));
            }
            provided_scales.iter().map(|value| **value).collect()
        } else {
            vec![1.0, 1.0]
        };

        // Step 6: Validate optional |sizes|.
        if let Some(ref sizes) = options.sizes {
            if sizes.len() != 2 || sizes.iter().any(|&value| value == 0) {
                return Err(Error::Type(c"invalid sizes".to_owned()));
            }
        }

        // Step 7: Resolve and validate |axes|.
        let axes: Vec<u32> = if let Some(ref provided_axes) = options.axes {
            let mut seen = std::collections::HashSet::new();
            if provided_axes
                .iter()
                .any(|&axis| (axis as usize) >= input_shape.len() || !seen.insert(axis))
            {
                return Err(Error::Type(c"invalid axes".to_owned()));
            }
            provided_axes.clone()
        } else {
            vec![2, 3]
        };
        if axes.len() != 2 {
            return Err(Error::Type(c"axes must have length 2".to_owned()));
        }

        // Step 8: Calculate the output shape.
        let mut output_shape = input_shape.clone();
        for index in 0..axes.len() {
            let axis = axes[index] as usize;
            let size = if let Some(ref sizes) = options.sizes {
                sizes[index]
            } else {
                ((input_shape[axis] as f32) * scales[index]).floor() as u32
            };

            if size == 0 {
                return Err(Error::Type(c"invalid output dimension".to_owned()));
            }

            output_shape[axis] = size;
        }

        let desc = MLOperandDescriptor {
            dataType: Self::data_type_enum_from_str(input_data_type),
            shape: output_shape.clone(),
        };

        // Step 9: *Make graph connections:*
        // Step 9.1: Let |output| be the result of [=creating an MLOperand=] given [=this=] and |desc|.
        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;
        let rust_operand =
            self.create_rust_operand(input_data_type, output_shape, OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Step 9.2/9.3/9.4/9.5: Record operator metadata and links.
        let mode = match options.mode {
            MLInterpolationMode::Nearest_neighbor => "nearest-neighbor",
            MLInterpolationMode::Linear => "linear",
        };
        if let Some(ref mut graph_info) = self.graph_info.borrow_mut().as_mut() {
            graph_info.operations.push(Operation {
                op_type: "resample2d".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes(
                    "resample2d",
                    serde_json::json!({
                        "mode": mode,
                        "axes": axes,
                        "scales": scales,
                        "sizes": options.sizes.clone(),
                    }),
                ),
                label: Self::label_from_operator_options(&options.parent),
            });
        }

        let output = create_an_mloperand(
            self,
            Some(&desc),
            None,
            None,
            false,
            false,
            Some(output_id),
            can_gc,
        );

        // Step 10: Return |output|.
        Ok(output)
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-split-splits>
    fn Split(
        &self,
        input: &MLOperand,
        splits: u32,
        options: &MLSplitOptions,
        can_gc: CanGc,
    ) -> Fallible<Vec<DomRoot<MLOperand>>> {
        // Step 1: Let |outputs| be the result of creating a split operation with count form.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |outputs|.
        self.create_split_operation(
            input,
            rustnn::shape_inference::SplitSpec::Count(splits),
            options.axis,
            &options.parent,
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-split-splitsequence>
    fn Split_(
        &self,
        input: &MLOperand,
        splits: Vec<u32>,
        options: &MLSplitOptions,
        can_gc: CanGc,
    ) -> Fallible<Vec<DomRoot<MLOperand>>> {
        // Step 1: Let |outputs| be the result of creating a split operation with explicit split sizes.
        // Step 1.1: If that throws an error, then rethrow the error.

        // Step 2: Return |outputs|.
        self.create_split_operation(
            input,
            rustnn::shape_inference::SplitSpec::Sizes(splits),
            options.axis,
            &options.parent,
            can_gc,
        )
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraphbuilder-transpose>
    fn Transpose(
        &self,
        input: &MLOperand,
        options: &MLTransposeOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If MLGraphBuilder/validating operand with this and |input| returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type(c"invalid operand".to_owned()));
        }

        let input_shape = input.descriptor_shape();
        let input_rank = input_shape.len();

        // Step 3: If |options|.permutation does not exist, then let |options|.permutation be
        // the reversed sequence of all indices for |input|'s shape.
        // Step 4: Otherwise validate permutation shape/range/uniqueness constraints.
        let permutation: Vec<u32> = match options.permutation.as_ref() {
            Some(permutation) => permutation.clone(),
            None => (0..input_rank as u32).rev().collect(),
        };

        // Step 4.1: If |permutation|'s size is not equal to |input|'s rank, then throw a TypeError.
        // Step 4.2: If any value is out of range [0, |input|'s rank), then throw a TypeError.
        // Step 4.3: If |permutation| contains duplicate values, then throw a TypeError.
        let output_shape = match rustnn::shape_inference::infer_transpose_shape(
            &input_shape,
            Some(permutation.as_slice()),
        ) {
            Ok(shape) => shape,
            Err(e) => return Err(Error::Type(cformat!("{e}"))),
        };

        // Step 5.1: Let |output| be the result of copying an MLOperand given |input|.
        let out_dtype_str = input.descriptor_data_type();
        let out_dtype_enum = match out_dtype_str {
            "float32" => MLOperandDataType::Float32,
            "float16" => MLOperandDataType::Float16,
            "int32" => MLOperandDataType::Int32,
            "uint32" => MLOperandDataType::Uint32,
            "int64" => MLOperandDataType::Int64,
            "uint64" => MLOperandDataType::Uint64,
            "int8" => MLOperandDataType::Int8,
            "uint8" => MLOperandDataType::Uint8,
            _ => MLOperandDataType::Float32,
        };
        let desc = MLOperandDescriptor {
            dataType: out_dtype_enum,
            shape: output_shape.clone(),
        };

        let input_id = input
            .id()
            .ok_or_else(|| Error::Type(c"input operand has no backend id".to_owned()))?;

        let rust_operand = self.create_rust_operand(
            out_dtype_str,
            output_shape.clone(),
            OperandKind::Output,
            None,
        );
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Note: the implementation allocates the backend output id before Step 5.1 so
        // graph-connection metadata can reference the output operand during operator recording.

        // Step 5.2: Let |operator| be an operator for the "transpose" operation, given |options|.
        // Step 5.3-5.5: Record operator input/output connections.
        let attributes = serde_json::json!({
            "permutation": permutation,
        });

        let label = {
            let l = options.parent.label.clone();
            if l.is_empty() {
                None
            } else {
                Some(l.clone().to_string())
            }
        };

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "transpose".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes: Self::operator_attributes("transpose", attributes),
                label,
            });
        }

        // Step 5.1: Let |output| be the result of copying an MLOperand given |input|.
        let operand = copy_an_mloperand(input, Some(&desc), Some(output_id), can_gc);

        // Step 6: Return |output|.
        Ok(operand)
    }
}
