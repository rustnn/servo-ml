use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use dom_struct::dom_struct;
use js::rust::HandleObject;
use rustnn::graph::{DataType, GraphInfo, Operand, OperandDescriptor, OperandKind, Operation};
use script_bindings::codegen::GenericBindings::NavigatorBinding::NavigatorMethods;
use script_bindings::codegen::GenericBindings::WindowBinding::WindowMethods;
use script_bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;
use script_bindings::record::Record;
use script_bindings::str::USVString;
use webnn_traits::WebNNMsg;

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLArgMinMaxOptions, MLBatchNormalizationOptions, MLClampOptions, MLConv2dFilterOperandLayout,
    MLConv2dOptions, MLGemmOptions, MLGraphBuilderMethods, MLInputOperandLayout, MLOperandDataType,
    MLOperandDescriptor, MLOperatorOptions,
};
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlgraph::MLGraph;
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
        // Builder can build iff it still owns a GraphInfo and the context is not lost.
        self.graph_info.borrow().is_some() && !self.context().is_lost()
    }

    fn validate_operand(&self, operand: &DomRoot<MLOperand>) -> bool {
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
        let id = self.next_operand_id.get();
        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operands.push(operand);
            if add_to_inputs {
                gi.input_operands.push(id);
            }
        }
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
        let desc = OperandDescriptor {
            data_type: rust_data_type,
            shape: rustnn::graph::to_dimension_vector(&shape),
            pending_permutation: Vec::new(),
        };
        Operand {
            descriptor: desc,
            kind,
            name,
        }
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
            return Err(Error::Type("invalid operand".to_owned()));
        }

        // Step 4: If |axis| is greater than or equal to |input|'s rank, then throw a TypeError.
        let in_shape = input.descriptor_shape();
        if (axis as usize) >= in_shape.len() {
            return Err(Error::Type("axis out of range".to_owned()));
        }

        // Step 5: Validate |options|.outputDataType is allowed (int32 or int64 for argMin/argMax).
        let out_dtype_str = options.outputDataType.as_str();
        if out_dtype_str != "int32" && out_dtype_str != "int64" {
            return Err(Error::Type(
                "outputDataType must be 'int32' or 'int64'".to_owned(),
            ));
        }

        // Step 6: If input.shape[axis] is greater than outputDataType's max value, then throw.
        let axis_dim = in_shape[axis as usize] as u128;
        match out_dtype_str {
            "int32" => {
                if axis_dim > (i32::MAX as u128) {
                    return Err(Error::Type("dimension too large for int32".to_owned()));
                }
            },
            "int64" => {
                if axis_dim > (i64::MAX as u128) {
                    return Err(Error::Type("dimension too large for int64".to_owned()));
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
            Err(e) => return Err(Error::Type(e.to_string())),
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
            None => return Err(Error::Type("input operand has no backend id".to_owned())),
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
                attributes,
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

impl MLGraphBuilder {
    /// Element-wise binary op helper implementations: add, sub, mul, div, max, min, pow
    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary>
    fn binary_elementwise_op(
        &self,
        op_name: &str,
        a: &MLOperand,
        b: &MLOperand,
        options_label: Option<String>,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: [Assert]: |op| is one of "add", "sub", "mul", "div", "max", "min", "pow".
        // Step 2: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 3: If validating operand with this and any of |a| and |b| returns false, then throw a TypeError.
        if !self.validate_operand_ref(a) || !self.validate_operand_ref(b) {
            return Err(Error::Type("invalid operand".to_owned()));
        }

        // Step 4: If |a|'s dataType is not equal to |b|'s dataType, then throw a TypeError.
        // (We enforce this invariant here; the spec permits implementation-defined promotions.)
        let a_dtype = a.descriptor_data_type();
        if a_dtype != b.descriptor_data_type() {
            return Err(Error::Type("input dataType must match".to_owned()));
        }

        // Step 5: Let |outputShape| be the result of bidirectionally broadcasting |a|.shape and |b|.shape.
        //         If that returns failure, then throw a TypeError.
        let out_shape = match rustnn::shape_inference::broadcast_shapes(
            a.descriptor_shape(),
            b.descriptor_shape(),
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(e.to_string())),
        };

        // Step 6: Let |descriptor| be the result of creating an MLOperandDescriptor given |a|'s dataType and |outputShape|.
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

        // Step 7: Make graph connections — create backend operand, operator record and return output.
        let a_id = a
            .id()
            .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?;
        let b_id = b
            .id()
            .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?;

        let rust_operand =
            self.create_rust_operand(a_dtype, out_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        let mut attributes = serde_json::json!({});
        let label = options_label
            .map(|s| if s.is_empty() { None } else { Some(s) })
            .flatten();

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: op_name.to_string(),
                input_operands: vec![a_id, b_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes,
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
}

impl MLGraphBuilderMethods<crate::DomTypeHolder> for MLGraphBuilder {
    fn Constructor(
        global: &GlobalScope,
        proto: Option<HandleObject>,
        can_gc: CanGc,
        context: &MLContext,
    ) -> DomRoot<MLGraphBuilder> {
        MLGraphBuilder::new(context, global, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-input>
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
            return Err(Error::Type("name is empty".to_owned()));
        }

        // Step 3: If any MLOperand in this graph's computational graph/inputs has [[name]] == |name|, then throw a TypeError.
        if let Some(ref gi) = self.graph_info.borrow().as_ref() {
            for &input_id in gi.input_operands.iter() {
                if let Some(op) = gi.operands.get(input_id as usize) {
                    if let Some(op_name) = &op.name {
                        if op_name.as_str() == name.str().as_ref() {
                            return Err(Error::Type("duplicate input name".to_owned()));
                        }
                    }
                }
            }
        }

        // Step 4: If MLOperandDescriptor/checking dimensions given |descriptor| returns false, then throw a TypeError.
        if !check_dimensions(descriptor) {
            return Err(Error::Type("invalid operand descriptor".to_owned()));
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

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-constant-buffer>
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
            return Err(Error::Type("invalid operand descriptor".to_owned()));
        }

        // Step 3: If validating buffer with descriptor given |buffer| and |descriptor| returns false, then throw a TypeError.
        // Step 3: TODO — validate the provided buffer against |descriptor| (byteLength, element type and shape).
        // TODO (spec: #api-mlgraphbuilder-constant-buffer): implement buffer validation (byte length, data type, shape).

        // Step 4: *Make graph connections:*
        // Step 4.1: Let |operand| be the result of creating an MLOperand given this and |descriptor|.
        // Step 4.2: Let |bytes| be the result of getting a copy of the bytes held by the buffer source given |buffer|.
        //         TODO — copy |bytes| from |buffer| and convert/validate them according to |descriptor| (byteLength, element type, shape).
        //         TODO (spec: #api-mlgraphbuilder-constant-buffer): implement buffer validation and copying.
        // Step 4.3: Add |operand| to this graph's computational graph/constants with |bytes| as value (persist bytes in GraphInfo).
        //         TODO — store |bytes| in `graph_info.constant_operand_ids_to_handles` and
        //         `graph_info.id_to_constant_tensor_operand_map` so Build() can materialize the constant.

        // Transfer the buffer contents into a backend tensor so that Build()
        // can later look it up by tensor id (the same mechanism used by
        // `Constant_(tensor)` and the Python bindings).  We get the id by
        // asking the context to allocate a constant tensor; the helper will
        // send a single message to the manager that creates and initializes the
        // storage.  The id is then recorded in the graph info's
        // `id_to_constant_tensor_operand_map` instead of stuffing the bytes into
        // `constant_operand_ids_to_handles`.
        let bytes: Vec<u8> = match buffer {
            ArrayBufferViewOrArrayBuffer::ArrayBufferView(view) => view.to_vec(),
            ArrayBufferViewOrArrayBuffer::ArrayBuffer(buf) => buf.to_vec(),
        };

        // ask context for a tensor id and queue the backend allocation
        let tensor_id = self.context().allocate_constant_tensor_for_builder(bytes);

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

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-constant-tensor>
    fn Constant_(&self, tensor: &MLTensor, can_gc: CanGc) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If tensor.[[context]] is not this.[[context]], throw a TypeError.
        if tensor.context() != self.context() {
            return Err(Error::Type(
                "tensor is not owned by this builder's context".to_owned(),
            ));
        }

        // Step 2: If |tensor|.[[isDestroyed]] is true, then throw a TypeError.
        if tensor.is_destroyed() {
            return Err(Error::Type("tensor is destroyed".to_owned()));
        }

        // Step 3: If |tensor|.[[isConstant]] is false, then throw a TypeError.
        if !tensor.is_constant() {
            return Err(Error::Type("tensor is not constant".to_owned()));
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

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-build>
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
            p.reject_error(Error::Type("outputs is empty".to_owned()), can_gc);
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
                    p.reject_error(Error::Type("operand name is empty".to_owned()), can_gc);
                    return p;
                }

                // Duplicate check for outputs.
                if !seen_output_names.insert(name.as_ref().to_string()) {
                    let p = Promise::new(global, can_gc);
                    p.reject_error(Error::Type("duplicate output name".to_owned()), can_gc);
                    return p;
                }

                // Check collision with any existing input name recorded in GraphInfo.
                for &input_id in gi.input_operands.iter() {
                    if let Some(op) = gi.operands.get(input_id as usize) {
                        if let Some(op_name) = &op.name {
                            if op_name.as_str() == name.as_ref() {
                                let p = Promise::new(global, can_gc);
                                p.reject_error(
                                    Error::Type("output name conflicts with input".to_owned()),
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
                    p.reject_error(Error::Type("invalid operand".to_owned()), can_gc);
                    return p;
                }

                // Step 4.3: If |operand| is in this graph's input operands or constants, then reject.
                if operand.is_input() || operand.is_constant() {
                    let p = Promise::new(global, can_gc);
                    p.reject_error(
                        Error::Type("operand cannot be an input or constant".to_owned()),
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

        // record the graph info + promise on the context; the actual DOM
        // `MLGraph` object will be created later in the compile callback.
        let p = Promise::new(global, can_gc);
        self.context()
            .register_build(graph_id, graph_info.clone(), p.clone());

        // send compile request to the manager; include the ML persistent
        // callback and the graph id so the manager can notify us when the
        // compilation finishes.  The promise will be resolved asynchronously
        // by the callback handler.  We clone the GraphInfo for the message.
        let cb = self
            .global()
            .as_window()
            .Navigator()
            .Ml()
            .get_or_setup_callback(global);
        let _ = self.global().webnn_sender().send(WebNNMsg::Compile(
            cb,
            graph_id,
            self.context().context_id(),
            graph_info.clone(),
        ));

        // Step 7/8: Return the promise without resolving it yet. It will be
        // resolved by the compile callback when compilation completes.
        p
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-argminmax>
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

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-argminmax>
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

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-where>
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
            return Err(Error::Type("invalid operand".to_owned()));
        }

        // Step 3: Validate data types per spec
        if condition.descriptor_data_type() != "uint8" {
            return Err(Error::Type(
                "condition must have dataType 'uint8'".to_owned(),
            ));
        }
        if true_value.descriptor_data_type() != false_value.descriptor_data_type() {
            return Err(Error::Type(
                "trueValue and falseValue must have the same dataType".to_owned(),
            ));
        }

        // Infer output shape using rustnn shape inference helper
        let output_shape = match rustnn::shape_inference::infer_where_shape(
            condition.descriptor_shape(),
            true_value.descriptor_shape(),
            false_value.descriptor_shape(),
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(e.to_string())),
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
        dataType: MLOperandDataType,
        options: &MLOperatorOptions,
        can_gc: CanGc,
    ) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If validating operand with this and input returns false, then throw a TypeError.
        if !self.validate_operand_ref(input) {
            return Err(Error::Type("invalid operand".to_owned()));
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
        let out_dtype_str = match dataType {
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
            dataType: dataType,
            shape: in_shape.clone(),
        };

        // Ensure the input has a backend operand id.
        let input_id = match input.id() {
            Some(i) => i,
            None => return Err(Error::Type("input operand has no backend id".to_owned())),
        };

        // Create backend operand for the output now (backend id required to record the operator).
        let rust_operand =
            self.create_rust_operand(out_dtype_str, in_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Build operation attributes and optional label (recording operator metadata).
        let mut attributes = serde_json::json!({ "dataType": out_dtype_str });

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
                attributes,
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
            return Err(Error::Type("invalid operand".to_owned()));
        }

        // Step 3: Let minValue be the options.minValue if given, or Infinity otherwise.
        // Step 4: Set options.minValue to the result of casting minValue to input's dataType.
        // Step 5: Let maxValue be the options.maxValue if given, or -Infinity otherwise.
        // Step 6: Set options.maxValue to the result of casting maxValue to input's dataType.
        // Step 7: If options.minValue is greater than options.maxValue, then throw a TypeError.

        let in_dtype = input.descriptor_data_type();
        let min_opt = options.minValue.as_ref().map(|v| **v);
        let max_opt = options.maxValue.as_ref().map(|v| **v);

        // Perform cast-to-input-dtype for comparison/recording.
        let cast_min = |v: f64| -> serde_json::Value {
            match in_dtype {
                "float32" | "float16" => serde_json::json!(v),
                "int8" => serde_json::json!(v as i8),
                "uint8" => serde_json::json!(v as u8),
                "int32" => serde_json::json!(v as i32),
                "uint32" => serde_json::json!(v as u32),
                "int64" => serde_json::json!(v as i64),
                "uint64" => serde_json::json!(v as u64),
                _ => serde_json::json!(v),
            }
        };

        // Validate min/max ordering after casting to input dtype semantics.
        if let (Some(min_v), Some(max_v)) = (min_opt, max_opt) {
            if (in_dtype == "float32" || in_dtype == "float16") {
                let min_c = min_v as f64;
                let max_c = max_v as f64;
                if min_c > max_c {
                    return Err(Error::Type(
                        "minValue must not be greater than maxValue".to_owned(),
                    ));
                }
            } else {
                let min_c = min_v as i128;
                let max_c = max_v as i128;
                if min_c > max_c {
                    return Err(Error::Type(
                        "minValue must not be greater than maxValue".to_owned(),
                    ));
                }
            }
        }

        // Make graph connections per the spec: output has same descriptor as input.
        let in_shape = input.descriptor_shape();
        let out_dtype_enum = match in_dtype {
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
            shape: in_shape.clone(),
        };

        // Ensure input has backend id.
        let input_id = match input.id() {
            Some(i) => i,
            None => return Err(Error::Type("input operand has no backend id".to_owned())),
        };
        let rust_operand =
            self.create_rust_operand(in_dtype, in_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Build attributes: record existence flags and casted values when present.
        let mut attributes = serde_json::json!({
            "hasMinValue": min_opt.is_some(),
            "hasMaxValue": max_opt.is_some(),
        });
        if let Some(mv) = min_opt {
            attributes["minValue"] = cast_min(mv);
        }
        if let Some(mv) = max_opt {
            attributes["maxValue"] = cast_min(mv);
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

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "clamp".to_string(),
                input_operands: vec![input_id],
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes,
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
            return Err(Error::Type("inputs is empty".to_owned()));
        }
        for inp in inputs.iter() {
            if !self.validate_operand(inp) {
                return Err(Error::Type("invalid operand".to_owned()));
            }
        }

        // Step 4: Let first be inputs[0].
        let first = inputs.get(0).expect("inputs non-empty; checked above");
        let first_shape = first.descriptor_shape();
        // Step 5: If axis is >= first.rank, then throw.
        if (axis as usize) >= first_shape.len() {
            return Err(Error::Type("axis out of range".to_owned()));
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
                    "input dataType must match first input".to_owned(),
                ));
            }
            if in_shape.len() != first_shape.len() {
                return Err(Error::Type("input rank must match first input".to_owned()));
            }

            for dim in 0..in_shape.len() {
                if dim != (axis as usize) {
                    if in_shape[dim] != first_shape[dim] {
                        return Err(Error::Type(
                            "input shapes must match on all dims except axis".to_owned(),
                        ));
                    }
                } else {
                    // Sum sizes on axis and check validity (no overflow / non-zero).
                    let size_sum = (desc.shape[dim] as u128)
                        .checked_add(in_shape[dim] as u128)
                        .ok_or_else(|| Error::Type("dimension size overflow".to_owned()))?;
                    if size_sum == 0 || size_sum > (u32::MAX as u128) {
                        return Err(Error::Type(
                            "invalid concatenated dimension size".to_owned(),
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
        let mut attributes = serde_json::json!({ "axis": axis });
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
                .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?;
            input_ids.push(id);
        }

        if let Some(ref mut gi) = self.graph_info.borrow_mut().as_mut() {
            gi.operations.push(Operation {
                op_type: "concat".to_string(),
                input_operands: input_ids,
                output_operand: Some(output_id),
                output_operands: Vec::new(),
                attributes,
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
            return Err(Error::Type("invalid operand".to_owned()));
        }
        if options.bias.is_some() {
            if !self.validate_operand_ref(options.bias.as_ref().unwrap()) {
                return Err(Error::Type("invalid operand".to_owned()));
            }
        }

        // Step 3: Input's dataType must be floating-point per spec.
        let in_dtype = input.descriptor_data_type();
        if in_dtype != "float32" && in_dtype != "float16" {
            return Err(Error::Type(
                "input dataType must be 'float32' or 'float16'".to_owned(),
            ));
        }

        // Step 4: Input must be 4-D.
        let in_shape = input.descriptor_shape();
        if in_shape.len() != 4 {
            return Err(Error::Type("input must be a 4-D tensor".to_owned()));
        }

        // Step 5: Filter must be 4-D.
        let filter_shape = filter.descriptor_shape();
        if filter_shape.len() != 4 {
            return Err(Error::Type("filter must be a 4-D tensor".to_owned()));
        }

        // Step 6: Filter's dataType must match input's dataType.
        if filter.descriptor_data_type() != in_dtype {
            return Err(Error::Type(
                "filter must have same dataType as input".to_owned(),
            ));
        }

        // Steps for options: apply defaults and validate lengths/values.
        let pads = match &options.padding {
            Some(p) if !p.is_empty() => p.clone(),
            _ => vec![0u32, 0u32, 0u32, 0u32],
        };
        if pads.len() != 4 {
            return Err(Error::Type("padding must be length 4".to_owned()));
        }

        let strides = match &options.strides {
            Some(s) if !s.is_empty() => s.clone(),
            _ => vec![1u32, 1u32],
        };
        if strides.len() != 2 {
            return Err(Error::Type("strides must be length 2".to_owned()));
        }
        if strides[0] < 1 || strides[1] < 1 {
            return Err(Error::Type("strides must be >= 1".to_owned()));
        }

        let dilations = match &options.dilations {
            Some(d) if !d.is_empty() => d.clone(),
            _ => vec![1u32, 1u32],
        };
        if dilations.len() != 2 {
            return Err(Error::Type("dilations must be length 2".to_owned()));
        }
        if dilations[0] < 1 || dilations[1] < 1 {
            return Err(Error::Type("dilations must be >= 1".to_owned()));
        }

        let groups = options.groups;
        if groups == 0 {
            return Err(Error::Type("groups must be >= 1".to_owned()));
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
            return Err(Error::Type("inputChannels % groups must be 0".to_owned()));
        }
        if (input_channels / groups) != filter_input_channels {
            return Err(Error::Type(
                "inputChannels / groups must equal filterInputChannels".to_owned(),
            ));
        }

        // If bias exists validate shape and dtype.
        if let Some(b) = options.bias.as_ref() {
            if b.descriptor_shape().len() != 1 {
                return Err(Error::Type("bias must be a 1-D tensor".to_owned()));
            }
            if b.descriptor_shape()[0] != output_channels {
                return Err(Error::Type(
                    "bias size must equal the filter output channels".to_owned(),
                ));
            }
            if b.descriptor_data_type() != in_dtype {
                return Err(Error::Type(
                    "bias must have same dataType as input".to_owned(),
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
            Err(e) => return Err(Error::Type(e.to_string())),
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
            "pads": pads,
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
                .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?,
        );
        input_ids.push(
            filter
                .id()
                .ok_or_else(|| Error::Type("filter operand has no backend id".to_owned()))?,
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
                attributes,
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
            return Err(Error::Type("invalid operand".to_owned()));
        }
        if options.scale.is_some() {
            if !self.validate_operand_ref(options.scale.as_ref().unwrap()) {
                return Err(Error::Type("invalid operand".to_owned()));
            }
        }
        if options.bias.is_some() {
            if !self.validate_operand_ref(options.bias.as_ref().unwrap()) {
                return Err(Error::Type("invalid operand".to_owned()));
            }
        }

        // Step 3: If input’s dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        let in_dtype = input.descriptor_data_type();
        if in_dtype != "float32" && in_dtype != "float16" {
            return Err(Error::Type(
                "input dataType must be 'float32' or 'float16'".to_owned(),
            ));
        }

        // Step 4: If options.axis is not in the range 0 to input’s rank, exclusive, then throw a TypeError.
        let in_shape = input.descriptor_shape();
        let axis = options.axis as usize;
        if axis >= in_shape.len() {
            return Err(Error::Type("axis out of range".to_owned()));
        }

        // Step 5: If mean’s dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        if mean.descriptor_data_type() != in_dtype {
            return Err(Error::Type(
                "mean must have same dataType as input".to_owned(),
            ));
        }

        // Step 6: If mean’s shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if mean.descriptor_shape().len() != 1 {
            return Err(Error::Type("mean must be a 1-D tensor".to_owned()));
        }
        if mean.descriptor_shape()[0] != in_shape[axis] {
            return Err(Error::Type(
                "mean size must equal the size of the input dimension denoted by axis".to_owned(),
            ));
        }

        // Step 7: If variance’s dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        if variance.descriptor_data_type() != in_dtype {
            return Err(Error::Type(
                "variance must have same dataType as input".to_owned(),
            ));
        }

        // Step 8: If variance’s shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if variance.descriptor_shape().len() != 1 {
            return Err(Error::Type("variance must be a 1-D tensor".to_owned()));
        }
        if variance.descriptor_shape()[0] != in_shape[axis] {
            return Err(Error::Type(
                "variance size must equal the size of the input dimension denoted by axis"
                    .to_owned(),
            ));
        }

        // Step 9: Set options.epsilon to the result of casting options.epsilon to input’s dataType.
        // `options.epsilon` is a `Finite<f64>` wrapper from bindings; dereference to `f64` for JSON.
        let epsilon = *options.epsilon;

        // Step 10.1: If options.scale exists and its dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        // Step 10.2: If options.scale exists and its shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if let Some(s) = options.scale.as_ref() {
            if s.descriptor_data_type() != in_dtype {
                return Err(Error::Type(
                    "scale must have same dataType as input".to_owned(),
                ));
            }
            if s.descriptor_shape().len() != 1 {
                return Err(Error::Type("scale must be a 1-D tensor".to_owned()));
            }
            if s.descriptor_shape()[0] != in_shape[axis] {
                return Err(Error::Type(
                    "scale size must equal the size of the input dimension denoted by axis"
                        .to_owned(),
                ));
            }
        }

        // Step 11.1: If options.bias exists and its dataType is not one of its allowed data types (according to this table), then throw a TypeError.
        // Step 11.2: If options.bias exists and its shape is not equal to « input’s shape[options.axis] », then throw a TypeError.
        if let Some(b) = options.bias.as_ref() {
            if b.descriptor_data_type() != in_dtype {
                return Err(Error::Type(
                    "bias must have same dataType as input".to_owned(),
                ));
            }
            if b.descriptor_shape().len() != 1 {
                return Err(Error::Type("bias must be a 1-D tensor".to_owned()));
            }
            if b.descriptor_shape()[0] != in_shape[axis] {
                return Err(Error::Type(
                    "bias size must equal the size of the input dimension denoted by axis"
                        .to_owned(),
                ));
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
            Err(e) => return Err(Error::Type(e.to_string())),
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
        let mut attributes = serde_json::json!({
            "epsilon": epsilon,
            "axis": options.axis,
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
                    None => return Err(Error::Type("input operand has no backend id".to_owned())),
                },
                match mean.id() {
                    Some(i) => i,
                    None => return Err(Error::Type("mean operand has no backend id".to_owned())),
                },
                match variance.id() {
                    Some(i) => i,
                    None => {
                        return Err(Error::Type("variance operand has no backend id".to_owned()));
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
                attributes,
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

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary>
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
        self.binary_elementwise_op("add", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary>
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
        self.binary_elementwise_op("sub", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary>
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
        self.binary_elementwise_op("mul", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-binary>
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
        self.binary_elementwise_op("div", a, b, Some(label), can_gc)
    }

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
        self.binary_elementwise_op("max", a, b, Some(label), can_gc)
    }

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
        self.binary_elementwise_op("min", a, b, Some(label), can_gc)
    }

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
        self.binary_elementwise_op("pow", a, b, Some(label), can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-matmul>
    fn Matmul(&self, a: &MLOperand, b: &MLOperand, can_gc: CanGc) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If this can not build, then throw an "InvalidStateError" DOMException.
        if !self.can_build() {
            return Err(Error::InvalidState(None));
        }

        // Step 2: If MLGraphBuilder/validating operand with this and any of |a| and |b| returns false, then throw a TypeError.
        if !self.validate_operand_ref(a) || !self.validate_operand_ref(b) {
            return Err(Error::Type("invalid operand".to_owned()));
        }

        // Step 3: If the MLOperand/dataType of any of |a| or |b| is not one of its allowed data types, then throw a TypeError.
        // (This implementation requires matching data types; promotion is not performed.)
        let a_dtype = a.descriptor_data_type();
        if a_dtype != b.descriptor_data_type() {
            return Err(Error::Type("input dataType must match".to_owned()));
        }

        // Step 4 (substeps 4.1–4.11): Calculate the output shape. The rustnn helper validates ranks, transposes
        // and inner-dimension compatibility, broadcasts batch shapes and appends spatial dims.
        let output_shape = match rustnn::shape_inference::infer_matmul_shape(
            &a.descriptor_shape(),
            &b.descriptor_shape(),
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(e.to_string())),
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
            .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?;
        let b_id = b
            .id()
            .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?;

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
                attributes: serde_json::json!({}),
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

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-gemm>
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
            return Err(Error::Type("invalid operand".to_owned()));
        }

        // NOTE (early check): we do a quick ownership check for options.c here, but
        // full spec Step 12 (broadcastability + dtype checks for `c`) is performed
        // at the location in the spec order (see TODO below).
        if options.c.is_some() {
            if !self.validate_operand_ref(options.c.as_ref().unwrap()) {
                return Err(Error::Type("invalid operand".to_owned()));
            }
        }

        // Step 3: If the MLOperand/dataType of any of |a| or |b| is not one of its allowed data types, then throw a TypeError.
        let a_dtype = a.descriptor_data_type();
        // enforce matching data types (implementation-defined promotion not supported here)
        if a_dtype != b.descriptor_data_type() {
            return Err(Error::Type("input dataType must match".to_owned()));
        }

        // Steps 4, 7–11: Shape/rank/transposition validations and inner-dimension checks.
        // Step 4: If the MLOperand/rank of any of |a| or |b| is not its allowed rank, then throw a TypeError.
        // Step 7–11: clone shapes, apply transposes, and verify shapeA[1] == shapeB[0].
        // Implementation delegates these checks to `rustnn::shape_inference::infer_gemm_shape`.
        let output_shape = match rustnn::shape_inference::infer_gemm_shape(
            &a.descriptor_shape(),
            &b.descriptor_shape(),
            options.aTranspose,
            options.bTranspose,
        ) {
            Ok(s) => s,
            Err(e) => return Err(Error::Type(e.to_string())),
        };

        // Step 12 (spec order): If |options|.c exists, then validate broadcastability and dtype per spec.
        // NOTE: current implementation only validates `c` belongs to this builder (ownership).
        // TODO: implement unidirectional broadcastability check to « shapeA[0], shapeB[1] » and
        // verify `c`'s dataType is allowed per the tensor-limits table.
        if options.c.is_some() {
            // ownership already validated above; remaining checks are TODO.
        }

        // Step 5 & 6 (spec order — casting alpha/beta): Set options.alpha/options.beta to the result of casting
        // to |a|'s dataType.
        // NOTE: the implementation currently records `*options.alpha`/`*options.beta` in the operator
        // attributes without performing an explicit cast to `a`'s data type. TODO: implement cast per spec.

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
                .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?,
            b.id()
                .ok_or_else(|| Error::Type("input operand has no backend id".to_owned()))?,
        ];
        if let Some(c_op) = options.c.as_ref() {
            input_ids.push(c_op.id().unwrap());
        }

        // Create backend operand for the output now (backend id required to record the operator).
        let rust_operand =
            self.create_rust_operand(a_dtype, output_shape.clone(), OperandKind::Output, None);
        let output_id = self.push_operand_to_graph(rust_operand, false);

        // Record operator attributes including alpha/beta and transpose flags (Steps 5/6 recorded here; casting TODO).
        let mut attributes = serde_json::json!({
            "alpha": *options.alpha,
            "beta": *options.beta,
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
                attributes,
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
}
