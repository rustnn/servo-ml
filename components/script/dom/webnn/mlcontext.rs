use std::rc::Rc;

use dom_struct::dom_struct;
use script_bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLContextLostInfo, MLContextMethods, MLOpSupportLimits, MLOperandDescriptor, MLPowerPreference,
    MLTensorDescriptor,
};
use crate::dom::bindings::error::{Error, throw_dom_exception};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mltensor::MLTensor;
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-mlcontext>
pub(crate) struct MLContext {
    reflector_: Reflector,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-contexttype-slot>
    context_type: String,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-powerpreference-slot>
    power_preference: MLPowerPreference,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated-slot>
    accelerated: bool,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-lost-slot>
    #[conditional_malloc_size_of]
    lost: Rc<Promise>,
}

impl MLContext {
    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    pub(crate) fn new_inherited(
        accelerated: bool,
        power_preference: MLPowerPreference,
        lost: Rc<Promise>,
    ) -> MLContext {
        // Note: implements the ML "To create a context" constructor steps that initialize
        // the context's internal slots; mapping is not 1:1 with the spec algorithm.
        // Step 1.1: Let |context| be a new MLContext in |realm| (constructor value).
        // Step 1.2: Set |context|.[[contextType]] to "default".
        // Step 1.3: Set |context|.[[powerPreference]] to the provided value.
        // Step 1.4: Set |context|.[[accelerated]] to the provided `accelerated` value.
        MLContext {
            reflector_: Reflector::new(),
            context_type: "default".into(),
            power_preference,
            accelerated,
            lost,
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    pub(crate) fn new(
        global: &GlobalScope,
        accelerated: bool,
        power_preference: MLPowerPreference,
        can_gc: CanGc,
    ) -> DomRoot<MLContext> {
        // Step 1.6: Set |context|.[[lost]] to a new promise in |realm|.
        let lost_promise = Promise::new(global, can_gc);
        let ctx = reflect_dom_object(
            Box::new(MLContext::new_inherited(
                accelerated,
                power_preference,
                lost_promise.clone(),
            )),
            global,
            can_gc,
        );

        ctx
    }

    /// <https://webmachinelearning.github.io/webnn/#mlcontext-is-lost>
    /// Helper: return true when this context is lost (spec helper #mlcontext-is-lost).
    pub(crate) fn is_lost(&self) -> bool {
        // Step 1: Return this.[[lost]] is fulfilled.
        self.lost.is_fulfilled()
    }

    /// Run the MLContext/lose steps for this context.
    ///
    /// Implements the spec algorithm that:
    /// 1) resolves this.[[lost]] with an MLContextLostInfo, and
    /// 2) runs the destroy steps for all associated graphs and tensors.
    ///
    /// Per the user's instruction we implement the promise-resolution portion
    /// now and leave the object-destruction loops as TODOs.
    pub(crate) fn lose(&self, message: Option<DOMString>) {
        // Step 1: Let |info| be a new MLContextLostInfo.
        let info = MLContextLostInfo { message };

        // Step 2: Resolve this.[[lost]] with |info|.
        (&*self.lost).resolve_native(&info, CanGc::note());

        // Step 3: For each MLGraph where graph.[[context]] == this, run MLGraph/destroy() steps.
        // TODO: enumerate and destroy associated MLGraph objects (not yet implemented).

        // Step 4: For each MLTensor where tensor.[[context]] == this, run MLTensor/destroy() steps.
        // TODO: enumerate and destroy associated MLTensor objects (not yet implemented).
    }
}

impl MLContextMethods<crate::DomTypeHolder> for MLContext {
    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated>
    fn Accelerated(&self) -> bool {
        // Step 1: Return this.[[accelerated]].
        self.accelerated
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-createtensor>
    fn CreateTensor(&self, descriptor: &MLTensorDescriptor) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm.
        // Note: the realm is represented by `global` in this implementation.

        // Step 3: If |this| is lost, return a new promise in |realm| rejected with an InvalidStateError.
        if self.is_lost() {
            // Step 3: create and return the rejected promise in |realm|.
            let p = Promise::new(global, CanGc::note());
            p.reject_error(Error::InvalidState(None), CanGc::note());

            return p;
        }

        // Step 4: Let |tensor| be the result of creating an MLTensor given |this| and |descriptor|.
        let tensor = MLTensor::new(self, global, descriptor, CanGc::note());

        // Step 5: Let |promise| be a new promise in |realm|.
        let p = Promise::new(global, CanGc::note());

        // Step 6: Enqueue the following steps to this.[[timeline]]:
        //     1. Run these steps, but abort when this is lost:
        //         1. Create |tensor|.[[data]] given |descriptor| and initialize all bytes to zeros.
        //         1. If that fails, then queue an ML task with |global| to reject |promise| with an "UnknownError" {{DOMException}}, and abort these steps.
        //         1. Otherwise, queue an ML task with |global| to resolve |promise| with |tensor|.
        //     1. [=/If aborted=], then queue an ML task with |global| to reject |promise| with an "InvalidStateError" {{DOMException}}.
        // TODO (spec: #api-mlcontext-createtensor): implement ML timeline task queuing and the zero-initialization/allocation
        // of tensor.[[data]]; that timeline task must resolve or reject |promise| asynchronously. Do NOT resolve |promise| here.

        // Step 7: Return |promise|.
        p
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-readtensor>
    fn ReadTensor(&self, tensor: &MLTensor) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm (represented by `global`).

        // Step 3: If |tensor|.[[context]] is not |this|, return a rejected promise with a TypeError.
        if tensor.context() != Dom::from_ref(self) {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(
                Error::Type("tensor is not owned by this context".to_owned()),
                CanGc::note(),
            );
            return p;
        }

        // Step 4: If |tensor|.[[isDestroyed]] is true, return a rejected promise with a TypeError.
        if tensor.is_destroyed() {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(
                Error::Type("MLTensor is destroyed".to_owned()),
                CanGc::note(),
            );
            return p;
        }

        // Step 5: If |tensor|.[[descriptor]]..readable is false, return a rejected promise with a TypeError.
        if !tensor.readable() {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(
                Error::Type("tensor is not readable".to_owned()),
                CanGc::note(),
            );
            return p;
        }

        // Step 6: Let |promise| be a new promise in |realm| and append it to tensor.[[pendingPromises]].
        let p = Promise::new(global, CanGc::note());
        tensor.append_pending_promise(p.clone());

        // Step 7: Enqueue the following steps to |tensor|.[[context]]'s [[timeline]]:
        //     1. Run these steps, but abort when this is lost:
        //         1. Let |bytes| be a byte sequence containing a copy of |tensor|.[[data]].
        //         1. If that fails, then queue an ML task with |global| to run these steps:
        //             1. Remove |promise| from |tensor|.[[pendingPromises]].
        //             1. Reject |promise| with an "UnknownError" {{DOMException}}, and abort these steps.
        //         1. Otherwise, queue an ML task with |global| to run these steps:
        //             1. Remove |promise| from |tensor|.[[pendingPromises]].
        //             1. Let |buffer| be the result of ArrayBuffer/creating an {{ArrayBuffer}} from |bytes| in |realm|.
        //             1. Resolve |promise| with |buffer|.
        //     1. [=/If aborted=], then queue an ML task with |global| to reject |promise| with an "InvalidStateError" {{DOMException}}.
        // TODO (spec: #api-mlcontext-readtensor): implement ML timeline task queuing, copying of tensor.[[data]], ArrayBuffer creation, and promise resolution; do NOT resolve |promise| here.

        // Step 8: Return |promise|.
        p
    }

    /// BYOB overload: readTensor(tensor, outputData)
    fn ReadTensor_(
        &self,
        tensor: &MLTensor,
        output_data: ArrayBufferViewOrArrayBuffer,
    ) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm (represented by `global`).

        // Step 3: If |tensor|.[[context]] is not |this|, then return a new promise in |realm| rejected with a TypeError.
        if tensor.context() != Dom::from_ref(self) {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(
                Error::Type(
                    "tensor is not owned by this context".to_owned(),
                ),
                CanGc::note(),
            );
            return p;
        }

        // Step 4: If |tensor|.[[isDestroyed]] is true, then return a new promise in |realm| rejected with a TypeError.
        if tensor.is_destroyed() {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(
                Error::Type("MLTensor is destroyed".to_owned()),
                CanGc::note(),
            );
            return p;
        }

        // Step 5: If |tensor|.[[descriptor]].{{MLTensorDescriptor/readable}} is false, then return a new promise in |realm| rejected with a TypeError.
        if !tensor.readable() {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(
                Error::Type("tensor is not readable".to_owned()),
                CanGc::note(),
            );
            return p;
        }

        // Step 6: If validating buffer with descriptor given |outputData| and |tensor|.[[descriptor]] returns false,
        // then return a new promise in |realm| rejected with a TypeError.
        // TODO (spec: #api-mlcontext-readtensor-byob): implement `validating buffer with descriptor` and reject when invalid.
        // For now we *do not* perform validation here — the BYOB validation and timeline copy are TODOs.
        let _ = output_data; // keep variable referenced until validation/usage is implemented

        // Step 7: Let |promise| be a new promise in |realm|.
        let p = Promise::new(global, CanGc::note());

        // Step 8: Append |promise| to |tensor|.[[pendingPromises]].
        tensor.append_pending_promise(p.clone());

        // Step 9: Enqueue the following steps to |tensor|.[[context]]'s [[timeline]]:
        //     1. Run these steps, but abort when this is lost:
        //         1. Let |bytes| be a byte sequence containing a copy of |tensor|.[[data]].
        //         1. If that fails, then queue an ML task with |global| to run these steps:
        //             1. Remove |promise| from |tensor|.[[pendingPromises]].
        //             1. Reject |promise| with an "UnknownError" {{DOMException}}, and abort these steps.
        //         1. Otherwise, queue an ML task with |global| to run these steps:
        //             1. Remove |promise| from |tensor|.[[pendingPromises]].
        //             1. If |outputData| is BufferSource/detached, then reject |promise| with a {{TypeError}}, and abort these steps.
        //             1. ArrayBuffer/Write |bytes| to |outputData|.
        //             1. Resolve |promise| with {{undefined}}.
        //     1. [=/If aborted=], then queue an ML task with |global| to reject |promise| with an "InvalidStateError" {{DOMException}}.
        // TODO (spec: #api-mlcontext-readtensor-byob): queue ML timeline task that performs the copy, handles detached buffers,
        // removes |promise| from tensor.[[pendingPromises]], and resolves or rejects |promise| appropriately.

        // Step 10: Return |promise|.
        p
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-createconstanttensor>
    fn CreateConstantTensor(
        &self,
        descriptor: &MLOperandDescriptor,
        input_data: ArrayBufferViewOrArrayBuffer,
    ) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm.

        // Step 3: If |this| is lost, return a new promise in |realm| rejected with an InvalidStateError.
        if self.is_lost() {
            let p = Promise::new(global, CanGc::note());
            p.reject_error(Error::InvalidState(None), CanGc::note());
            return p;
        }

        // Step 4: If MLOperandDescriptor/checking dimensions given |descriptor| returns false, then return a new promise in |realm| rejected with a {{TypeError}}.
        // Step 5: If validating buffer with descriptor given |inputData| and |descriptor| returns false, then return a new promise in |realm| rejected with a {{TypeError}}.
        // Step 6: Let |bytes| be the result of getting a copy of the bytes held by the buffer source given |inputData|.
        // Step 7: [=Assert=]: |bytes|'s [=byte sequence/length=] is equal to |descriptor|'s [=MLOperandDescriptor/byte length=].
        // TODO (spec: #api-mlcontext-createconstanttensor): implement descriptor-dimension checks, buffer validation, copy-to-|bytes| and the length assertion.
        let _ = input_data; // validation + timeline copy are TODOs.

        // Step 8: Let |tensor| be the result of creating a constant MLTensor given |this| and |descriptor|.
        let tensor = MLTensor::new_constant(self, global, descriptor, CanGc::note());

        // Step 9: Let |promise| be a new promise in |realm|.
        let p = Promise::new(global, CanGc::note());

        // Step 10: Enqueue the following steps to this.[[timeline]]:
        //     1. Run these steps, but abort when this is lost:
        //         1. Create |tensor|.[[data]] given |descriptor|.
        //         1. If that fails, then queue an ML task with |global| to reject |promise| with an "UnknownError" {{DOMException}}, and abort these steps.
        //         1. Copy |bytes| to |tensor|.[[data]].
        //         1. If that fails, then queue an ML task with |global| to reject |promise| with an "UnknownError" {{DOMException}}, and abort these steps.
        //         1. Otherwise, queue an ML task with |global| to resolve |promise| with |tensor|.
        //     1. [=/If aborted=], then queue an ML task with |global| to reject |promise| with an "InvalidStateError" {{DOMException}}.
        // TODO (spec: #api-mlcontext-createconstanttensor): queue ML timeline task that creates tensor data, copies |bytes| and resolves/rejects |promise| asynchronously. Do NOT resolve |promise| here.

        // Step 11: Return |promise|.
        p
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-writetensor>
    fn WriteTensor(&self, tensor: &MLTensor, input_data: ArrayBufferViewOrArrayBuffer) {
        // Step 1: Let |global| be this's relevant global object.

        // Step 2: Let |realm| be this's relevant realm.

        // Step 3: If |tensor|.[[context]] is not |this|, then throw a TypeError.
        if tensor.context() != Dom::from_ref(self) {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(
                cx,
                &self.global(),
                Error::Type("tensor is not owned by this context".to_owned()),
                CanGc::note(),
            );
            return;
        }

        // Step 4: If |tensor|.[[isDestroyed]] is true, then throw a TypeError.
        if tensor.is_destroyed() {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(
                cx,
                &self.global(),
                Error::Type("MLTensor is destroyed".to_owned()),
                CanGc::note(),
            );
            return;
        }

        // Step 5: If |tensor|.[[descriptor]].{{MLTensorDescriptor/writable}} is false, then throw a TypeError.
        if !tensor.writable() {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(
                cx,
                &self.global(),
                Error::Type("tensor is not writable".to_owned()),
                CanGc::note(),
            );
            return;
        }

        // Step 6: If validating buffer with descriptor given |inputData| and |tensor|.[[descriptor]] returns false, then throw a {{TypeError}}.
        // Step 7: Let |bytes| be the result of getting a copy of the bytes held by the buffer source given |inputData|.
        // Step 8: [=Assert=]: |bytes|'s [=byte sequence/length=] is equal to |tensor|.[[descriptor]]'s [=MLOperandDescriptor/byte length=].
        // Step 9: Enqueue the following steps to |tensor|.[[context]]'s [[timeline]]:
        //     1. Run these steps, but abort when this is lost:
        //         1. Copy |bytes| to |tensor|.[[data]].
        // TODO (spec: #api-mlcontext-writetensor): perform buffer validation, obtain |bytes|, assert lengths, and queue timeline copy.

        // Step 8: Return undefined.
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-opsupportlimits>
    fn OpSupportLimits(&self) -> MLOpSupportLimits {
        // Step 1: Return this implementation's supported operation limits.
        // Minimal implementation: return the default limits (placeholder).
        Default::default()
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-destroy>
    fn Destroy(&self) {
        // Step 1: If this is lost, then abort these steps.
        if self.is_lost() {
            return;
        }

        // Step 2: Run the steps to MLContext/lose this with an implementation-defined message.
        // Per spec this is a direct call into the MLContext/lose abstract operation.
        // The remaining destroy-of-associated-objects logic is TODO.
        self.lose(Some(DOMString::from("destroyed")));

        // TODO: queue or perform any additional destroy bookkeeping required by the implementation.
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-lost>
    fn Lost(&self) -> Rc<Promise> {
        // Step 1: Return this.[[lost]].
        self.lost.clone()
    }
}
