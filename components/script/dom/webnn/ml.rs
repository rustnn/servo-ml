use std::cell::Cell;
use std::rc::Rc;

use dom_struct::dom_struct;
use webnn_traits::{ContextId, WebNNMsg};

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLContextOptions, MLMethods};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::trace::HashMapTracedValues;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mlcontext::MLContext;
use crate::script_runtime::CanGc;

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-ml>
pub(crate) struct ML {
    reflector_: Reflector,

    /// Map of active contexts created via this ML object.
    contexts: DomRefCell<HashMapTracedValues<ContextId, Dom<MLContext>>>,

    /// Per-GlobalScope counter used to create ContextId.counter values.
    next_context_counter: Cell<u32>,
}

impl ML {
    pub(crate) fn new_inherited() -> ML {
        ML {
            reflector_: Reflector::new(),
            contexts: Default::default(),
            next_context_counter: Cell::new(1),
        }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<ML> {
        reflect_dom_object(Box::new(ML::new_inherited()), global, can_gc)
    }

    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    pub(crate) fn create_context(
        &self,
        options: &MLContextOptions,
        can_gc: CanGc,
    ) -> DomRoot<MLContext> {
        // Create a new ContextId (pipeline-scoped counter).
        let pipeline_id = self.global().pipeline_id();
        let counter = self.next_context_counter.get();
        self.next_context_counter.set(counter.wrapping_add(1));
        let ctx_id = ContextId {
            pipeline_id,
            counter,
        };

        // Create the DOM-side MLContext with the new id.
        let ctx = MLContext::new(
            &self.global(),
            ctx_id,
            options.accelerated,
            options.powerPreference,
            can_gc,
        );

        // Store the context in the ML map.
        self.contexts
            .borrow_mut()
            .insert(ctx_id, Dom::from_ref(&*ctx));

        // Inform the WebNN backend/manager about the new context.
        if let Err(e) = self
            .global()
            .webnn_sender()
            .send(WebNNMsg::NewContext(ctx_id))
        {
            error!("WebNN NewContext send failed ({:?})", e);
        }

        ctx
    }
}

impl MLMethods<crate::DomTypeHolder> for ML {
    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    fn CreateContext(&self, options: &MLContextOptions, can_gc: CanGc) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        // Step 2: Let |realm| be this's relevant realm.
        // Step 3: If |global|'s associated Document is not allowed to use the webnn feature,
        //         then return a new promise in |realm| rejected with a "SecurityError" DOMException.
        // Note: In this implementation `self.global()` provides the relevant global/realm; the
        //       spec's realm/global steps do not map 1:1 to Rust call sites.

        // Step 4: Let |promise| be a new promise in |realm|.
        let p = Promise::new(&self.global(), can_gc);

        // Step 5.1: Let |context| be the result of creating a context given |realm| and |options|.
        // Note: the spec runs the context-creation and promise-resolution in-parallel. This
        // helper implements the context-creation steps synchronously on the event loop; the
        // timeline start remains an implementation-defined, in-parallel operation (TODO).
        let context = self.create_context(options, can_gc);

        // Step 5.1 (cont): the constructor already set [[lost]] to a new Promise in the realm.

        // Step 5.2: Resolve |promise| with |context| immediately (synchronous rendezvous).
        p.resolve_native(&context, can_gc);

        // Step 5.3: TODO — start the MLContext's timeline in an implementation-defined manner
        // TODO (spec: #api-ml-createcontext): queue ML task to start timeline / select backend.

        // Step 6: Return |promise|.
        p
    }
}
