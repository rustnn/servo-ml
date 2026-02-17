use std::rc::Rc;
use std::cell::Cell;

use dom_struct::dom_struct;
use js::rust::HandleObject;
use script_bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLGraphBuilderMethods, MLGraphMethods, MLOperandDescriptor, MLOperandMethods,
};
use crate::dom::bindings::error::{Error, throw_dom_exception};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::globalscope::GlobalScope;
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
    has_built: Cell<bool>,

    /// <https://webmachinelearning.github.io/webnn/#computational-graph-input>
    inputs: DomRefCell<Vec<DomRoot<MLOperand>>>,

    /// <https://webmachinelearning.github.io/webnn/#computational-graph-constants>
    constants: DomRefCell<Vec<DomRoot<MLOperand>>>,
}

impl MLGraphBuilder {
    pub(crate) fn new_inherited(context: &MLContext) -> MLGraphBuilder {
        MLGraphBuilder {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            has_built: Cell::new(false),
            inputs: DomRefCell::new(Vec::new()),
            constants: DomRefCell::new(Vec::new()),
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
        !self.has_built.get() && !self.context().is_lost()
    }

    fn validate_operand(&self, operand: &DomRoot<MLOperand>) -> bool {
        operand.builder() == Dom::from_ref(self)
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
    fn Input(&self, name: DOMString, descriptor: &MLOperandDescriptor) -> DomRoot<MLOperand> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(cx, &self.global(), Error::InvalidState(None), CanGc::note());
            // Return a placeholder (exception pending will take precedence in JS).
            return MLOperand::new(
                self,
                &self.global(),
                descriptor,
                Some(name.clone()),
                true,
                false,
                CanGc::note(),
            );
        }

        // Step 2: If any operand in this graph's input operands has a name equal to |name|, throw a TypeError.
        for existing in self.inputs.borrow().iter() {
            if let Some(n) = existing.name() {
                if n == name {
                    let cx = GlobalScope::get_cx();
                    throw_dom_exception(
                        cx,
                        &self.global(),
                        Error::Type("duplicate input name".to_owned()),
                        CanGc::note(),
                    );
                    return MLOperand::new(
                        self,
                        &self.global(),
                        descriptor,
                        Some(name.clone()),
                        true,
                        false,
                        CanGc::note(),
                    );
                }
            }
        }

        // Step 3: Create |operand| and add it to this graph's input operands (see https://webmachinelearning.github.io/webnn/#computational-graph-input).
        let operand = MLOperand::new(
            self,
            &self.global(),
            descriptor,
            Some(name.clone()),
            true,
            false,
            CanGc::note(),
        );
        self.inputs.borrow_mut().push(operand.clone());

        // Step 4: Return |operand|.
        operand
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-constant-buffer>
    fn Constant(
        &self,
        descriptor: &MLOperandDescriptor,
        buffer: ArrayBufferViewOrArrayBuffer,
    ) -> DomRoot<MLOperand> {
        // Step 1: If this can not build, then throw an InvalidStateError.
        if !self.can_build() {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(
                cx,
                &self.global(),
                Error::InvalidState(None),
                CanGc::note(),
            );
            return MLOperand::new(
                self,
                &self.global(),
                descriptor,
                None,
                false,
                true,
                CanGc::note(),
            );
        }

        // Step 2: If MLOperandDescriptor/checking dimensions given |descriptor| returns false, then throw a TypeError.
        // TODO: implement MLOperandDescriptor/checking dimensions validation.

        // Step 3: If validating buffer with descriptor given |buffer| and |descriptor| returns false, then throw a TypeError.
        // TODO: implement buffer validation according to the spec (byte length, data type, shape).

        // Step 4: Make graph connections:
        //   1. Let |operand| be the result of creating an MLOperand given this and |descriptor|.
        //   2. Let |bytes| be the result of getting a copy of the bytes held by the buffer source given |buffer|.
        //   3. Add |operand| to this graph's computational graph/constants with |bytes| as value.
        // TODO: copy bytes from |buffer| and add to the computational graph constants (not implemented yet).

        // Current implementation: create the operand and add it to the builder's constants list as a placeholder.
        let _ = buffer;
        let operand = MLOperand::new(
            self,
            &self.global(),
            descriptor,
            None,
            false,
            true,
            CanGc::note(),
        );
        self.constants.borrow_mut().push(operand.clone());
        operand
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlgraphbuilder-constant-tensor>
    fn Constant_(&self, tensor: &MLTensor) -> DomRoot<MLOperand> {
        // Step 1: If tensor.[[context]] is not this.[[context]], throw a TypeError.
        if tensor.context() != self.context() {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(
                cx,
                &self.global(),
                Error::Type("tensor is not owned by this builder's context".to_owned()),
                CanGc::note(),
            );
            return MLOperand::new_from_tensor(
                self,
                &self.global(),
                tensor,
                None,
                false,
                true,
                CanGc::note(),
            );
        }

        // Step 2: If this can not build, throw an InvalidStateError.
        if !self.can_build() {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(
                cx,
                &self.global(),
                Error::InvalidState(None),
                CanGc::note(),
            );
            return MLOperand::new_from_tensor(
                self,
                &self.global(),
                tensor,
                None,
                false,
                true,
                CanGc::note(),
            );
        }

        // Step 3: Create operand and add it to constants (tensor retained by operand in spec).
        let operand = MLOperand::new_from_tensor(
            self,
            &self.global(),
            tensor,
            None,
            false,
            true,
            CanGc::note(),
        );
        self.constants.borrow_mut().push(operand.clone());
        operand
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
        let graph = MLGraph::new(&self.context(), global, CanGc::note());

        // Step 6: Set this.[[hasBuilt]] to true.
        self.has_built.set(true);

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
