use std::rc::Rc;

use dom_struct::dom_struct;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLContextOptions, MLMethods};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-ml>
pub(crate) struct ML {
    reflector_: Reflector,
}

impl ML {
    pub(crate) fn new_inherited() -> ML {
        ML {
            reflector_: Reflector::new(),
        }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<ML> {
        reflect_dom_object(Box::new(ML::new_inherited()), global, can_gc)
    }
}

impl MLMethods<crate::DomTypeHolder> for ML {
    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    fn CreateContext(&self, options: &MLContextOptions) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        // Step 2: Let |realm| be this's relevant realm.
        // Step 3: If |global|'s associated Document is not allowed to use the webnn feature,
        //         then return a new promise in |realm| rejected with a "SecurityError" DOMException.
        // Note: In this implementation `self.global()` provides the relevant global/realm; the
        //       spec's realm/global steps do not map 1:1 to Rust call sites.

        // Step 4: Let |promise| be a new promise in |realm|.
        let p = Promise::new(&self.global(), CanGc::note());

        // Step 5.1: Let |context| be the result of creating a context given |realm| and |options|.
        // Step 5.2: Queue an ML task with |global| to resolve |promise| with |context|.
        // TODO: implement ML task queuing, backend/device-selection and promise rendezvous (spec: #api-ml-createcontext).

        // Step 6: Return |promise|.
        p
    }
}
