use std::cell::Cell;

use dom_struct::dom_struct;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::MLContextMethods;
use crate::dom::bindings::reflector::{Reflector, reflect_dom_object};
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::script_runtime::CanGc;

#[dom_struct]
/// Minimal `MLContext` implementation — only `accelerated` getter is provided for now.
pub(crate) struct MLContext {
    reflector_: Reflector,
    accelerated: Cell<bool>,
}

impl MLContext {
    pub(crate) fn new_inherited() -> MLContext {
        MLContext { reflector_: Reflector::new(), accelerated: Cell::new(true) }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<MLContext> {
        reflect_dom_object(Box::new(MLContext::new_inherited()), global, can_gc)
    }
}

impl MLContextMethods<crate::DomTypeHolder> for MLContext {
    /// Return whether the context is accelerated. (TODO placeholder)
    fn Accelerated(&self) -> bool {
        todo!("MLContext::Accelerated is not implemented yet");
    }
}
