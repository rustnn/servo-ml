use std::rc::Rc;

use dom_struct::dom_struct;
use profile_traits::generic_callback::GenericCallback;
use script_bindings::codegen::GenericBindings::NavigatorBinding::NavigatorMethods;
use script_bindings::codegen::GenericBindings::WindowBinding::WindowMethods;
use script_bindings::codegen::GenericUnionTypes::ArrayBufferViewOrArrayBuffer;
use webnn_traits::{ContextId, WebNNMsg};

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLContextLostInfo, MLContextMethods, MLOpSupportLimits, MLOperandDescriptor, MLPowerPreference,
    MLTensorDescriptor,
};
use crate::dom::bindings::error::{Error, throw_dom_exception};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::bindings::trace::HashMapTracedValues;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::ml::ML;
use crate::dom::webnn::mltensor::MLTensor;
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-mlcontext>
pub(crate) struct MLContext {
    reflector_: Reflector,

    /// Unique identifier for this context (pipeline + counter).
    #[no_trace]
    context_id: ContextId,

    /// Per-context tensor id counter.
    next_tensor_id: crate::dom::bindings::trace::NoTrace<std::cell::Cell<u32>>,

    /// Map of pending tensors (tensor_id -> MLTensor) waiting for backend allocation.
    pending_tensors: DomRefCell<HashMapTracedValues<u32, Dom<MLTensor>>>,

    /// Map of promises (tensor_id -> Promise) for create-tensor requests. The create-tensor
    /// promise belongs to the context per spec.
    #[conditional_malloc_size_of]
    pending_tensor_promises: DomRefCell<HashMapTracedValues<u32, Rc<Promise>>>,

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
        context_id: ContextId,
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
            context_id,
            next_tensor_id: crate::dom::bindings::trace::NoTrace(std::cell::Cell::new(1)),
            pending_tensors: Default::default(),
            pending_tensor_promises: Default::default(),
            context_type: "default".into(),
            power_preference,
            accelerated,
            lost,
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    pub(crate) fn new(
        global: &GlobalScope,
        context_id: ContextId,
        accelerated: bool,
        power_preference: MLPowerPreference,
        can_gc: CanGc,
    ) -> DomRoot<MLContext> {
        // Step 1.6: Set |context|.[[lost]] to a new promise in |realm|.
        let lost_promise = Promise::new(global, can_gc);
        let ctx = reflect_dom_object(
            Box::new(MLContext::new_inherited(
                context_id,
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
    pub(crate) fn lose(&self, message: Option<DOMString>, can_gc: CanGc) {
        // Step 1: Let |info| be a new MLContextLostInfo.
        let info = MLContextLostInfo { message };

        // Step 2: Resolve this.[[lost]] with |info|.
        (&*self.lost).resolve_native(&info, can_gc);

        // Step 3: For each MLGraph where graph.[[context]] == this, run MLGraph/destroy() steps.
        // TODO: enumerate and destroy associated MLGraph objects (not yet implemented).

        // Step 4: For each MLTensor where tensor.[[context]] == this, run MLTensor/destroy() steps.
        // TODO: enumerate and destroy associated MLTensor objects (not yet implemented).
    }

    /// Called when the backend replies to a create-tensor request.
    ///
    /// Implementation / spec-mapped steps:
    /// 1. Let |tensor| be the `MLTensor` removed from this.[[pendingTensors]] keyed by `tensor_id`.
    ///    - If no such tensor exists, log a warning and return.
    /// 2. Let |promise| be the Promise removed from this.[[pendingTensorPromises]] keyed by `tensor_id`.
    ///    - If no such promise exists, log a warning and return.
    /// 3. If `result` is success, resolve |promise| with |tensor|. Otherwise reject |promise| with an "Operation" error.
    ///
    /// Notes: this operation implements the script-side portion of the WebNN create-tensor
    /// callback flow (the manager/backend performs allocation and invokes the persisted
    /// ML callback with a ContextMessage that routes here).
    pub(crate) fn create_tensor_callback(
        &self,
        tensor_id: u32,
        result: Result<(), ()>,
        can_gc: CanGc,
    ) {
        let tensor = {
            let mut pending = self.pending_tensors.borrow_mut();
            match pending.remove(&tensor_id) {
                Some(dom_tensor) => dom_tensor.as_rooted(),
                None => {
                    warn!("create_tensor_callback: unknown tensor id {}", tensor_id);
                    return;
                },
            }
        };

        let promise = {
            let mut pending = self.pending_tensor_promises.borrow_mut();
            match pending.remove(&tensor_id) {
                Some(p) => p,
                None => {
                    warn!(
                        "create_tensor_callback: missing pending promise for tensor {}",
                        tensor_id
                    );
                    return;
                },
            }
        };

        match result {
            Ok(()) => promise.resolve_native(&tensor, can_gc),
            Err(_) => promise.reject_error(Error::Operation(None), can_gc),
        }
    }
}

impl MLContextMethods<crate::DomTypeHolder> for MLContext {
    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated>
    fn Accelerated(&self) -> bool {
        // Step 1: Return this.[[accelerated]].
        self.accelerated
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-createtensor>
    fn CreateTensor(&self, descriptor: &MLTensorDescriptor, can_gc: CanGc) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm.
        // Note: the realm is represented by `global` in this implementation.

        // Step 3: If |this| is lost, return a new promise in |realm| rejected with an InvalidStateError.
        if self.is_lost() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::InvalidState(None), can_gc);
            return p;
        }

        // Implementation detail (pre‑Step 4): validate |descriptor| and compute element/byte lengths.
        // The spec's "creating an MLTensor" step implicitly covers descriptor validation; this
        // implementation performs explicit validation before constructing the DOM `MLTensor`.
        if descriptor.shape.iter().any(|&d| d == 0) {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::Type("invalid tensor descriptor".to_owned()), can_gc);
            return p;
        }

        let mut element_length: u128 = 1;
        for &dim in descriptor.shape.iter() {
            element_length = match element_length.checked_mul(dim as u128) {
                Some(v) => v,
                None => {
                    let p = Promise::new(global, can_gc);
                    p.reject_error(
                        Error::Type("tensor descriptor too large".to_owned()),
                        can_gc,
                    );
                    return p;
                },
            };
        }
        let element_size: u128 = match &*descriptor.dataType.str() {
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
            Some(v) if v <= (usize::MAX as u128) => v as usize,
            _ => {
                let p = Promise::new(global, can_gc);
                p.reject_error(
                    Error::Type("tensor descriptor too large".to_owned()),
                    can_gc,
                );
                return p;
            },
        };

        // Step 4: Let |tensor| be the result of creating an MLTensor given |this| and |descriptor|.
        let tensor = MLTensor::new(self, global, descriptor, can_gc);

        // Step 5: Let |promise| be a new promise in |realm|.
        let p = Promise::new(global, can_gc);

        // Implementation detail: assign a context-local tensor id and record DOM-side pending state.
        // (Bookkeeping so the promise can be resolved by `create_tensor_callback` when the backend replies.)
        let id = self.next_tensor_id.0.get();
        self.next_tensor_id.0.set(id.wrapping_add(1));
        tensor.set_tensor_id(id);
        // Per-spec the create-tensor promise is stored on the context (not on the tensor).
        self.pending_tensors
            .borrow_mut()
            .insert(id, Dom::from_ref(&*tensor));
        self.pending_tensor_promises
            .borrow_mut()
            .insert(id, p.clone());

        // Implementation detail: ensure ML-level persistent callback exists for manager/backend replies.
        let ml_dom = global.as_window().Navigator().Ml();
        let cb = ml_dom.get_or_setup_callback(global);

        // Step 6: Enqueue the following steps to this.[[timeline]] (spec):
        // 6.1 Create |tensor|.[[data]] given |descriptor| and initialize all bytes to zeros.
        // 6.2 If that fails -> queue an ML task with |global| to reject |promise| with an "UnknownError" and abort.
        // 6.3 Otherwise -> queue an ML task with |global| to resolve |promise| with |tensor|.
        // 6.4 [=/If aborted=] -> queue an ML task with |global| to reject |promise| with an "InvalidStateError".
        // Discrepancy: timeline enqueue is implemented by sending `WebNNMsg::CreateTensor` to the WebNN manager; the
        // manager allocates backend storage and invokes the persisted ML callback that routes to `create_tensor_callback`.
        if let Err(e) = self.global().webnn_sender().send(WebNNMsg::CreateTensor(
            cb,
            self.context_id,
            id,
            byte_length,
        )) {
            error!("WebNN CreateTensor send failed ({:?})", e);
            // If sending fails we must clean up the pending DOM-side state and reject
            // the promise with an Operation error (implementation-defined behavior).
            self.pending_tensors.borrow_mut().remove(&id);
            p.reject_error(Error::Operation(None), can_gc);
            return p;
        }

        // Step 7: Return |promise|.
        p
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-readtensor>
    fn ReadTensor(&self, tensor: &MLTensor, can_gc: CanGc) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm (represented by `global`).

        // Step 3: If |tensor|.[[context]] is not |this|, return a rejected promise with a TypeError.
        if tensor.context() != Dom::from_ref(self) {
            let p = Promise::new(global, can_gc);
            p.reject_error(
                Error::Type("tensor is not owned by this context".to_owned()),
                can_gc,
            );
            return p;
        }

        // Step 4: If |tensor|.[[isDestroyed]] is true, return a rejected promise with a TypeError.
        if tensor.is_destroyed() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::Type("MLTensor is destroyed".to_owned()), can_gc);
            return p;
        }

        // Step 5: If |tensor|.[[descriptor]]..readable is false, return a rejected promise with a TypeError.
        if !tensor.readable() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::Type("tensor is not readable".to_owned()), can_gc);
            return p;
        }

        // Step 6: Let |promise| be a new promise in |realm| and append it to tensor.[[pendingPromises]].
        let p = Promise::new(global, can_gc);
        tensor.append_pending_promise(p.clone());

        // Step 7: Enqueue timeline steps to |tensor|.[[context]]'s [[timeline]]:
        // 7.1 Run these steps, abort when this is lost.
        // 7.1.1 Let |bytes| be a byte sequence containing a copy of |tensor|.[[data]].
        // 7.1.2 If that fails -> queue an ML task with |global| to remove |promise| from |tensor|.[[pendingPromises]] and reject |promise| with an "UnknownError".
        // 7.1.3 Otherwise -> queue an ML task with |global| to remove |promise| from |tensor|.[[pendingPromises]], create an ArrayBuffer from |bytes| in |realm|, and resolve |promise| with that buffer.
        // 7.2 [=/If aborted=] -> queue an ML task with |global| to reject |promise| with an "InvalidStateError".
        // Implementation note: ML timeline task queuing, copying of |tensor|.[[data]] and ArrayBuffer creation are TODO; do NOT resolve |promise| here.

        // Step 8: Return |promise|.
        p
    }

    /// BYOB overload: readTensor(tensor, outputData)
    fn ReadTensor_(
        &self,
        tensor: &MLTensor,
        output_data: ArrayBufferViewOrArrayBuffer,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm (represented by `global`).

        // Step 3: If |tensor|.[[context]] is not |this|, then return a new promise in |realm| rejected with a TypeError.
        if tensor.context() != Dom::from_ref(self) {
            let p = Promise::new(global, can_gc);
            p.reject_error(
                Error::Type("tensor is not owned by this context".to_owned()),
                can_gc,
            );
            return p;
        }

        // Step 4: If |tensor|.[[isDestroyed]] is true, then return a new promise in |realm| rejected with a TypeError.
        if tensor.is_destroyed() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::Type("MLTensor is destroyed".to_owned()), can_gc);
            return p;
        }

        // Step 5: If |tensor|.[[descriptor]].{{MLTensorDescriptor/readable}} is false, then return a new promise in |realm| rejected with a TypeError.
        if !tensor.readable() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::Type("tensor is not readable".to_owned()), can_gc);
            return p;
        }

        // Step 6: If validating buffer with descriptor given |outputData| and |tensor|.[[descriptor]] returns false,
        // then return a new promise in |realm| rejected with a TypeError.
        // TODO (spec: #api-mlcontext-readtensor-byob): implement `validating buffer with descriptor` and reject when invalid.
        // For now we *do not* perform validation here — the BYOB validation and timeline copy are TODOs.
        let _ = output_data; // keep variable referenced until validation/usage is implemented

        // Step 7: Let |promise| be a new promise in |realm|.
        let p = Promise::new(global, can_gc);

        // Step 8: Append |promise| to |tensor|.[[pendingPromises]].
        tensor.append_pending_promise(p.clone());

        // Step 9: Enqueue timeline steps to |tensor|.[[context]]'s [[timeline]]:
        // 9.1 Run these steps, abort when this is lost.
        // 9.1.1 Let |bytes| be a byte sequence containing a copy of |tensor|.[[data]].
        // 9.1.2 If that fails -> queue an ML task with |global| to remove |promise| from |tensor|.[[pendingPromises]] and reject |promise| with an "UnknownError".
        // 9.1.3 Otherwise -> queue an ML task with |global| to remove |promise| from |tensor|.[[pendingPromises]]; if |outputData| is detached then reject |promise| with a TypeError and abort; otherwise write |bytes| to |outputData| and resolve |promise| with undefined.
        // 9.2 [=/If aborted=] -> queue an ML task with |global| to reject |promise| with an "InvalidStateError".
        // Implementation note: BYOB timeline copy/validation is TODO; do NOT resolve |promise| here.

        // Step 10: Return |promise|.
        p
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-createconstanttensor>
    fn CreateConstantTensor(
        &self,
        descriptor: &MLOperandDescriptor,
        input_data: ArrayBufferViewOrArrayBuffer,
        can_gc: CanGc,
    ) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm.

        // Step 3: If |this| is lost, return a new promise in |realm| rejected with an InvalidStateError.
        if self.is_lost() {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::InvalidState(None), can_gc);
            return p;
        }

        // Step 4: If MLOperandDescriptor/checking dimensions given |descriptor| returns false, then return a new promise in |realm| rejected with a {{TypeError}}.
        if !crate::dom::webnn::check_dimensions(descriptor) {
            let p = Promise::new(global, can_gc);
            p.reject_error(Error::Type("invalid operand descriptor".to_owned()), can_gc);
            return p;
        }

        // Step 5: If validating buffer with descriptor given |inputData| and |descriptor| returns false, then return a new promise in |realm| rejected with a {{TypeError}}.
        // Step 6: Let |bytes| be the result of getting a copy of the bytes held by the buffer source given |inputData|.
        // Step 7: [=Assert=]: |bytes|'s [=byte sequence/length=] is equal to |descriptor|'s [=MLOperandDescriptor/byte length=].
        // TODO (spec: #api-mlcontext-createconstanttensor): implement buffer validation, copy-to-|bytes| and the length assertion.
        let _ = input_data; // buffer validation + timeline copy are TODOs.

        // Step 8: Let |tensor| be the result of creating a constant MLTensor given |this| and |descriptor|.
        let tensor = MLTensor::new_constant(self, global, descriptor, can_gc);

        // Step 9: Let |promise| be a new promise in |realm|.
        let p = Promise::new(global, can_gc);

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
    fn WriteTensor(
        &self,
        tensor: &MLTensor,
        input_data: ArrayBufferViewOrArrayBuffer,
        can_gc: CanGc,
    ) {
        // Step 1: Let |global| be this's relevant global object.

        // Step 2: Let |realm| be this's relevant realm.

        // Step 3: If |tensor|.[[context]] is not |this|, then throw a TypeError.
        if tensor.context() != Dom::from_ref(self) {
            let cx = GlobalScope::get_cx();
            throw_dom_exception(
                cx,
                &self.global(),
                Error::Type("tensor is not owned by this context".to_owned()),
                can_gc,
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
                can_gc,
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
                can_gc,
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
    fn Destroy(&self, can_gc: CanGc) {
        // Step 1: If this is lost, then abort these steps.
        if self.is_lost() {
            return;
        }

        // Inform backend/manager that this context is being destroyed.
        if let Err(e) = self
            .global()
            .webnn_sender()
            .send(WebNNMsg::DestroyContext(self.context_id))
        {
            error!("WebNN DestroyContext send failed ({:?})", e);
        }

        // Step 2: Run the steps to MLContext/lose this with an implementation-defined message.
        // Per spec this is a direct call into the MLContext/lose abstract operation.
        // The remaining destroy-of-associated-objects logic is TODO.
        self.lose(Some(DOMString::from("destroyed")), can_gc);

        // TODO: queue or perform any additional destroy bookkeeping required by the implementation.
    }

    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-lost>
    fn Lost(&self) -> Rc<Promise> {
        // Step 1: Return this.[[lost]].
        self.lost.clone()
    }
}
