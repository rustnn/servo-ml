use std::cell::Cell;

use dom_struct::dom_struct;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::MLContextMethods;
use crate::dom::bindings::reflector::{Reflector, reflect_dom_object};
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-mlcontext>
/// Minimal `MLContext` implementation — only `accelerated` getter is provided for now.
pub(crate) struct MLContext {
    reflector_: Reflector,
    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated-slot>
    accelerated: Cell<bool>,
}

impl MLContext {
    pub(crate) fn new_inherited() -> MLContext {
        MLContext {
            reflector_: Reflector::new(),
            accelerated: Cell::new(true),
        }
    }

    pub(crate) fn new_inherited_with_accelerated(accelerated: bool) -> MLContext {
        MLContext {
            reflector_: Reflector::new(),
            accelerated: Cell::new(accelerated),
        }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<MLContext> {
        reflect_dom_object(Box::new(MLContext::new_inherited()), global, can_gc)
    }

    pub(crate) fn new_with_accelerated(
        global: &GlobalScope,
        accelerated: bool,
        can_gc: CanGc,
    ) -> DomRoot<MLContext> {
        reflect_dom_object(
            Box::new(MLContext::new_inherited_with_accelerated(accelerated)),
            global,
            can_gc,
        )
    }
}

impl MLContextMethods<crate::DomTypeHolder> for MLContext {
    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated>
    fn Accelerated(&self) -> bool {
        // Step 1: Return this.[[accelerated]].
        self.accelerated.get()
    }
}
