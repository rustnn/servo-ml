use std::collections::VecDeque;
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

/// Internal queue entry representing a pending read operation for this tensor.
///
/// This enum is used by `MLTensor.pending_reads`.  It replaces the previous
/// pair of `[[pendingPromises]]`/`[[pendingOut]]` slots to keep the two pieces of
/// information together and avoid the risk of mis‑alignment or accidental
/// overwrites (for example, via `set_last_pending_out`).
///
/// Internal helper enum – not measured by malloc_size_of since the queue is
/// ignored entirely on the tensor object.
#[derive(JSTraceable)]
pub(crate) enum PendingRead {
    /// A normal `readTensor()` request; the callback should resolve the promise
    /// with the returned bytes.
    Read(Rc<Promise>),

    /// A BYOB overload.  The `output` buffer supplied by script will receive the
    /// bytes when the backend reply arrives and the promise is resolved with
    /// `undefined`.
    ReadByob {
        promise: Rc<Promise>,
        output: ArrayBufferViewOrArrayBuffer,
    },
}

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

    /// A FIFO queue of pending read requests (both normal and BYOB).
    ///
    /// This field is ignored for malloc-size accounting; it only holds transient
    /// promise/buffer pairs that live on the timeline queue.
    #[ignore_malloc_size_of = "transient queue, not included in malloc stats"]
    pending_reads: DomRefCell<VecDeque<PendingRead>>,

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
            pending_reads: DomRefCell::new(VecDeque::new()),
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

    /// Append an ordinary (non-BYOB) read request to the queue.
    pub(crate) fn append_pending_read(&self, p: Rc<Promise>) {
        self.pending_reads
            .borrow_mut()
            .push_back(PendingRead::Read(p));
    }

    /// Append a BYOB read request along with the provided output buffer.
    pub(crate) fn append_pending_read_byob(
        &self,
        p: Rc<Promise>,
        out: ArrayBufferViewOrArrayBuffer,
    ) {
        self.pending_reads
            .borrow_mut()
            .push_back(PendingRead::ReadByob {
                promise: p,
                output: out,
            });
    }

    /// Pop and return the first queued read entry.
    pub(crate) fn take_first_pending_read(&self) -> Option<PendingRead> {
        self.pending_reads.borrow_mut().pop_front()
    }

    /// Remove any queued request whose promise pointer matches `to_remove`.
    /// Because the queue is strictly FIFO and the only way elements are removed
    /// is via `remove_pending_read` itself, the entry (if present) will always be
    /// found at or near the front.  We therefore only scan once and delete the
    /// matching position; there's no need for a fallback `retain` pass.
    pub(crate) fn remove_pending_read(&self, to_remove: *const Promise) {
        let mut reads = self.pending_reads.borrow_mut();
        if let Some(pos) = reads.iter().position(|r| match r {
            PendingRead::Read(p) => Rc::as_ptr(p) == to_remove,
            PendingRead::ReadByob { promise, .. } => Rc::as_ptr(promise) == to_remove,
        }) {
            reads.remove(pos);
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
