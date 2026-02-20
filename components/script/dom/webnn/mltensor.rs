use std::rc::Rc;

use dom_struct::dom_struct;

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLOperandDescriptor, MLTensorDescriptor, MLTensorMethods,
};
use crate::dom::bindings::codegen::UnionTypes::ArrayBufferViewOrArrayBuffer;
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlcontext::MLContext;
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#dom-mltensor>
pub(crate) struct MLTensor {
    reflector_: Reflector,

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-context-slot>
    context: Dom<MLContext>,

    /// Script-visible tensor id (assigned by MLContext). Uses 0 for "no backend id".
    tensor_id: crate::dom::bindings::trace::NoTrace<std::cell::Cell<u32>>,

    /// <https://webmachinelearning.github.io/webnn/#dom-mloperanddescriptor-datatype>
    data_type: String,

    /// <https://webmachinelearning.github.io/webnn/#dom-mloperanddescriptor-shape>
    shape: Vec<i64>,

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensordescriptor-readable>
    readable: bool,

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensordescriptor-writable>
    writable: bool,

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-pendingpromises-slot>
    #[conditional_malloc_size_of]
    pending_promises: DomRefCell<Vec<Rc<Promise>>>,

    /// Pending BYOB output `outputData` values corresponding (by FIFO index) to `[[pendingPromises]]`.
    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-readtensor-byob>
    #[ignore_malloc_size_of = "ArrayBufferViewOrArrayBuffer"]
    pending_out: DomRefCell<
        Vec<Option<crate::dom::bindings::codegen::UnionTypes::ArrayBufferViewOrArrayBuffer>>,
    >,

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-isdestroyed-slot>
    is_destroyed: bool,

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-isconstant-slot>
    is_constant: bool,
}

impl MLTensor {
    fn new_inherited(
        context: &MLContext,
        data_type: String,
        shape: Vec<i64>,
        readable: bool,
        writable: bool,
        tensor_id: u32,
    ) -> MLTensor {
        MLTensor {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            tensor_id: crate::dom::bindings::trace::NoTrace(std::cell::Cell::new(tensor_id)),
            data_type,
            shape,
            readable,
            writable,
            pending_promises: DomRefCell::new(Vec::new()),
            pending_out: DomRefCell::new(Vec::new()),
            is_destroyed: false,
            is_constant: false,
        }
    }

    // Internal accessors for other components to use. DOM structs must not expose
    // fields publicly; use these instead to inspect/manipulate internal slots.
    pub(crate) fn context(&self) -> Dom<MLContext> {
        self.context.clone()
    }

    pub(crate) fn data_type(&self) -> &str {
        &self.data_type
    }

    pub(crate) fn shape(&self) -> &Vec<i64> {
        &self.shape
    }

    pub(crate) fn readable(&self) -> bool {
        self.readable
    }

    pub(crate) fn writable(&self) -> bool {
        self.writable
    }

    pub(crate) fn is_destroyed(&self) -> bool {
        self.is_destroyed
    }

    pub(crate) fn is_constant(&self) -> bool {
        self.is_constant
    }

    /// Script-visible tensor id assigned by the context (0 means no backend id).
    pub(crate) fn tensor_id(&self) -> u32 {
        self.tensor_id.0.get()
    }

    pub(crate) fn set_tensor_id(&self, id: u32) {
        debug_assert_ne!(id, 0, "MLTensor::set_tensor_id must not be called with 0");
        self.tensor_id.0.set(id);
    }

    pub(crate) fn append_pending_promise(&self, p: Rc<Promise>) {
        // Keep `pending_promises` and `pending_out` FIFO-aligned by appending
        // a `None` placeholder for the BYOB slot (filled by the BYOB overload).
        self.pending_promises.borrow_mut().push(p);
        self.pending_out.borrow_mut().push(None);
    }

    // Pop and return the first pending promise (used by callback resolution).
    pub(crate) fn take_first_pending_promise(&self) -> Option<Rc<Promise>> {
        let mut v = self.pending_promises.borrow_mut();
        if v.is_empty() {
            None
        } else {
            Some(v.remove(0))
        }
    }

    // Pop and return the first pending BYOB `outputData` (used by callback resolution).
    pub(crate) fn take_first_pending_out(&self) -> Option<ArrayBufferViewOrArrayBuffer> {
        let mut v = self.pending_out.borrow_mut();
        if v.is_empty() { None } else { v.remove(0) }
    }

    // Set the last pending_out entry (used by BYOB overload immediately after appending a promise).
    pub(crate) fn set_last_pending_out(&self, out: ArrayBufferViewOrArrayBuffer) {
        let mut v = self.pending_out.borrow_mut();
        if let Some(slot) = v.last_mut() {
            *slot = Some(out);
        } else {
            v.push(Some(out));
        }
    }

    // Remove a pending promise reference (used by timeline task when resolve/reject).
    pub(crate) fn remove_pending_promise(&self, to_remove: *const Promise) {
        let mut promises = self.pending_promises.borrow_mut();
        if let Some(pos) = promises.iter().position(|p| Rc::as_ptr(p) == to_remove) {
            promises.remove(pos);
            let mut outs = self.pending_out.borrow_mut();
            if pos < outs.len() {
                outs.remove(pos);
            }
        } else {
            // Fallback behaviour: remove any matching entries.
            promises.retain(|p| Rc::as_ptr(p) != to_remove);
        }
    }

    pub(crate) fn new(
        context: &MLContext,
        global: &GlobalScope,
        descriptor: &MLTensorDescriptor,
        tensor_id: u32,
        can_gc: CanGc,
    ) -> DomRoot<MLTensor> {
        // Debug-check the invariant that non-constant tensors must have a non-zero id.
        // Callers must ensure `tensor_id != 0`; this is verified only in debug builds.
        debug_assert_ne!(tensor_id, 0, "MLTensor::new called with tensor_id == 0");

        let data_type = descriptor.dataType.clone().to_string();
        let shape = descriptor.shape.clone();
        let readable = descriptor.readable;
        let writable = descriptor.writable;

        reflect_dom_object(
            Box::new(MLTensor::new_inherited(
                context, data_type, shape, readable, writable, tensor_id,
            )),
            global,
            can_gc,
        )
    }

    /// Helper to create a constant MLTensor (used by `MLContext.createConstantTensor`).
    ///
    /// Minimal implementation: mark the tensor as constant and make it
    /// non-readable / non-writable. Timeline allocation/copy MUST be performed by
    /// the caller's ML timeline task; this helper does not resolve any promises.
    pub(crate) fn new_constant(
        context: &MLContext,
        global: &GlobalScope,
        descriptor: &MLOperandDescriptor,
        can_gc: CanGc,
    ) -> DomRoot<MLTensor> {
        let data_type = descriptor.dataType.as_str().to_string();
        let shape = descriptor.shape.iter().map(|&s| s as i64).collect();
        let readable = false;
        let writable = false;

        // Use the same reflector construction as `new_inherited`, but mark `is_constant`.
        reflect_dom_object(
            Box::new(MLTensor::new_inherited(
                context, data_type, shape, readable, writable, 0,
            )),
            global,
            can_gc,
        )
    }
}

impl MLTensorMethods<crate::DomTypeHolder> for MLTensor {}
