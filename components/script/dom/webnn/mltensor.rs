use std::rc::Rc;

use dom_struct::dom_struct;

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLOperandDescriptor, MLTensorDescriptor, MLTensorMethods,
};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlcontext::MLContext;
use crate::script_runtime::CanGc;

#[dom_struct]
/// Minimal MLTensor required for `MLContext.createTensor()` (expand later).
pub(crate) struct MLTensor {
    reflector_: Reflector,

    /// <https://webmachinelearning.github.io/webnn/#dom-mltensor-context-slot>
    context: Dom<MLContext>,

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
    ) -> MLTensor {
        MLTensor {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            data_type,
            shape,
            readable,
            writable,
            pending_promises: DomRefCell::new(Vec::new()),
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

    pub(crate) fn append_pending_promise(&self, p: Rc<Promise>) {
        self.pending_promises.borrow_mut().push(p);
    }

    // Remove a pending promise reference (used by timeline task when resolve/reject).
    pub(crate) fn remove_pending_promise(&self, to_remove: *const Promise) {
        let mut v = self.pending_promises.borrow_mut();
        v.retain(|p| Rc::as_ptr(p) != to_remove);
    }

    pub(crate) fn new(
        context: &MLContext,
        global: &GlobalScope,
        descriptor: &MLTensorDescriptor,
        can_gc: CanGc,
    ) -> DomRoot<MLTensor> {
        let data_type = descriptor.dataType.clone().to_string();
        let shape = descriptor.shape.clone();
        let readable = descriptor.readable;
        let writable = descriptor.writable;

        reflect_dom_object(
            Box::new(MLTensor::new_inherited(
                context, data_type, shape, readable, writable,
            )),
            global,
            can_gc,
        )
    }

    /// Create a constant MLTensor (used by `MLContext.createConstantTensor`).
    ///
    /// Minimal implementation: mark the tensor as constant and make it
    /// non-readable / non-writable. The ML timeline copy/allocation is left
    /// as a TODO and must not resolve any promises here.
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

        // Use the same reflector construction as `new`, but mark `is_constant`.
        reflect_dom_object(
            Box::new(MLTensor {
                reflector_: Reflector::new(),
                context: Dom::from_ref(context),
                data_type,
                shape,
                readable,
                writable,
                pending_promises: DomRefCell::new(Vec::new()),
                is_destroyed: false,
                is_constant: true,
            }),
            global,
            can_gc,
        )
    }
}

impl MLTensorMethods<crate::DomTypeHolder> for MLTensor {}
