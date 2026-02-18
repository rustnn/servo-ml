use std::rc::Rc;
use std::cell::Cell;

use dom_struct::dom_struct;
use js::rust::HandleObject;
use script_bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLGraphBuilderMethods, MLOperandDescriptor,
};
use crate::dom::bindings::error::{Error, Fallible};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::globalscope::GlobalScope;
use std::collections::HashMap;
use rustnn::graph::{
    GraphInfo, Operand, OperandDescriptor, OperandKind,
    DataType,
};
use crate::dom::promise::Promise;
use crate::dom::MLContext;
use crate::dom::webnn::mlgraph::MLGraph;
use crate::dom::MLTensor;
use crate::dom::webnn::mloperand::MLOperand;
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
            shape,
            pending_permutation: Vec::new(),
        };
        Operand { descriptor: desc, kind, name }
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
) -> DomRoot<MLOperand> {
    // Step 1: Let |realm| be |builder|'s relevant realm.
    // Step 2: Let |operand| be a new MLOperand in |realm|.
    let global = &builder.global();
    let operand = if let Some(t) = tensor {
        MLOperand::new_from_tensor(builder, global, t, name, is_input, is_constant, operand_id, CanGc::note())
    } else if let Some(desc) = descriptor {
        MLOperand::new(builder, global, desc, name, is_input, is_constant, operand_id, CanGc::note())
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
            CanGc::note(),
        );
    };

    // Step 3: Set |operand|.[[builder]] to |builder| and set |operand|.[[descriptor]] to |desc|.
    // (Handled by the MLOperand constructors above.)

    // Step 4: Return |operand|.
    operand
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
    fn Input(&self, name: DOMString, descriptor: &MLOperandDescriptor) -> Fallible<DomRoot<MLOperand>> {
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
        let operand = create_an_mloperand(self, Some(descriptor), None, Some(name.clone()), true, false, Some(id));

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

        // Current implementation: create a rustnn constant operand (no bytes yet) and link the DOM operand to it.
        let _ = buffer;

        // Create rustnn operand descriptor and operand id.
        let rust_operand = self.create_rust_operand(
            descriptor.dataType.as_str(),
            descriptor.shape.clone(),
            OperandKind::Constant,
            None,
        );
        let id = self.push_operand_to_graph(rust_operand, false);
        // Step 4.3: TODO — persist the constant bytes for operand id |id| into GraphInfo so
        // Build() can materialize the constant; populate
        // `graph_info.constant_operand_ids_to_handles` and
        // `graph_info.id_to_constant_tensor_operand_map`.

        // Step 4.1: Let |operand| be the result of creating an MLOperand given this and |descriptor|.
        let operand = create_an_mloperand(self, Some(descriptor), None, None, false, true, Some(id));
        // Step 4.2: DOM operands are no longer kept by the builder; backend bookkeeping
        // for the operand (id and bytes) must be persisted in `graph_info`.
        // Step 5: Return |operand|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-constant-tensor>
    fn Constant_(&self, tensor: &MLTensor) -> Fallible<DomRoot<MLOperand>> {
        // Step 1: If tensor.[[context]] is not this.[[context]], throw a TypeError.
        if tensor.context() != self.context() {
            return Err(Error::Type("tensor is not owned by this builder's context".to_owned()));
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
        // Step 5.3: TODO — persist the bytes of |tensor| in the builder's GraphInfo so the
        // runtime can materialize the constant during Build().
        // TODO (spec: #api-mlgraphbuilder-constant-tensor): record tensor bytes in
        // `graph_info.constant_operand_ids_to_handles` and `id_to_constant_tensor_operand_map`. 

        // Step 5.1: Let |operand| be the result of creating an MLOperand given this and |tensor|.[[descriptor]].
        let operand = create_an_mloperand(self, None, Some(tensor), None, false, true, Some(id));
        // Step 5.2: Set |operand|.[[constantTensor]] to |tensor| (TODO: MLOperand currently does not store a reference).
        // Step 5.3: DOM operands are not stored on the builder; persist tensor bytes in `graph_info`
        // so that Build() can materialize the constant.
        // Step 6: Return |operand|.
        Ok(operand)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-build>
    fn Build(&self, outputs: Vec<DomRoot<MLOperand>>) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: If this can not build, then return a new promise in |realm| rejected with an InvalidStateError.
        if !self.can_build() {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(Error::InvalidState(None), CanGc::note());
            return p;
        }

        // Step 3: If |outputs| is empty, then return a new promise in |realm| rejected with a TypeError.
        if outputs.is_empty() {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(Error::Type("outputs is empty".to_owned()), CanGc::note());
            return p;
        }

        // Step 4: For each |operand| of |outputs|, run the per-operand validations from the spec.
        for operand in outputs.iter() {
            // Step 4.1: If |name| is empty, then return a rejected promise with a TypeError.
            // TODO (spec: #api-mlgraphbuilder-build): the current binding accepts sequence<MLOperand>
            // (no names). Implement MLNamedOperands support to validate empty names when needed.

            // Step 4.2: If MLGraphBuilder/validating operand given |this| and |operand| returns false, then reject.
            if !self.validate_operand(operand) {
                let p = Promise::new(global, CanGc::note());
                p.reject_error(Error::Type("invalid operand".to_owned()), CanGc::note());
                return p;
            }

            // Step 4.3: If |operand| is in this graph's input operands or constants, then reject.
            if operand.is_input() || operand.is_constant() {
                let p = Promise::new(global, CanGc::note());
                p.reject_error(
                    Error::Type("operand cannot be an input or constant".to_owned()),
                    CanGc::note(),
                );
                return p;
            }

            // Step 4.4: If |operand|.[[constantTensor]] exists and |operand|.[[constantTensor]].[[isDestroyed]] is true, then reject.
            // TODO (spec: #api-mlgraphbuilder-build): MLOperand currently does not keep a reference to an
            // associated constant MLTensor. Add tracking or a helper so this validation can be implemented.
        }

        // Step 5: Let |graph| be a new MLGraph and associate it with this.[[context]].
        // Step 6: Set this.[[hasBuilt]] to true.
        //
        // Implementation note: `graph_info == None` represents [[hasBuilt]]; Build() moves GraphInfo.
        let graph_info = self
            .graph_info
            .borrow_mut()
            .take()
            .expect("can_build() ensured graph_info is Some");
        let graph = MLGraph::new_with_info(&self.context(), graph_info, global, CanGc::note());

        // Step 7: Convert the builder's computational graph into an implementation-defined format
        // and enqueue initialization on the ML timeline. This is an async timeline task per the spec.
        // Step 7: TODO — queue ML timeline initialization and do not resolve the returned promise here.
        // TODO (spec: #api-mlgraphbuilder-build): implement ML timeline graph initialization which
        // must perform preprocessing on the MLContext/[[timeline]] and resolve/reject the promise.

        // Step 8: Return |promise| (timeline task will resolve/reject it asynchronously).
        let p = Promise::new(global, CanGc::note());
        p
    }
}
