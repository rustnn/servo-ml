use std::cell::Cell;

use dom_struct::dom_struct;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::MLGraphMethods;
use crate::dom::bindings::reflector::{Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::globalscope::GlobalScope;
use crate::dom::webnn::mlcontext::MLContext;
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-mlgraph>
pub(crate) struct MLGraph {
    reflector_: Reflector,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraph-context-slot>
    context: Dom<MLContext>,

    /// <https://webmachinelearning.github.io/webnn/#dom-mlgraph-isdestroyed-slot>
    is_destroyed: Cell<bool>,
}

impl MLGraph {
    pub(crate) fn new_inherited(context: &MLContext) -> MLGraph {
        MLGraph {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            is_destroyed: Cell::new(false),
        }
    }

    pub(crate) fn new(
        context: &MLContext,
        global: &GlobalScope,
        can_gc: CanGc,
    ) -> DomRoot<MLGraph> {
        reflect_dom_object(Box::new(MLGraph::new_inherited(context)), global, can_gc)
    }

    pub(crate) fn context(&self) -> Dom<MLContext> {
        self.context.clone()
    }

    pub(crate) fn is_destroyed(&self) -> bool {
        self.is_destroyed.get()
    }
}

impl MLGraphMethods<crate::DomTypeHolder> for MLGraph {
    /// <https://webmachinelearning.github.io/webnn/#api-mlgraph-destroy>
    fn Destroy(&self) {
        // Step 1: If [=this=].[[isDestroyed]] is true, then abort these steps.
        if self.is_destroyed() {
            return;
        }

        // Step 2: Set [=this=].[[isDestroyed]] to true.
        self.is_destroyed.set(true);

        // TODO (spec: #api-mlgraph-destroy): release platform resources associated with the graph.
    }
}
