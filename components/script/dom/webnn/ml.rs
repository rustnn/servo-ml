/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::RefCell;
use std::rc::Rc;

use dom_struct::dom_struct;
use profile_traits::generic_callback::GenericCallback;
use webnn_traits::{ContextId, ContextMessage, GraphId, WebNNMsg};

use crate::dom::bindings::cell::DomRefCell;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLContextOptions, MLMethods};
use crate::dom::bindings::refcounted::Trusted;
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

    /// Persistent GenericCallback used to route backend context-level replies into ML.
    #[no_trace]
    create_tensor_cb: RefCell<Option<GenericCallback<ContextMessage>>>,
}

impl ML {
    pub(crate) fn new_inherited() -> ML {
        ML {
            reflector_: Reflector::new(),
            contexts: Default::default(),
            create_tensor_cb: RefCell::new(None),
        }
    }

    /// Ensure a persistent ML-level callback exists that receives `ContextMessage`s
    /// from the manager and routes them onto the script thread.
    ///
    /// Implementation steps (IDBFactory-style persistent callback pattern):
    /// 1. If `self.create_tensor_cb` is already set, return it.
    /// 2. Otherwise, construct a `GenericCallback<ContextMessage>` that:
    ///    a. Accepts the IPC result wrapper and extracts the `ContextMessage` payload.
    ///    b. For `CreateTensorResult`, queues an ML task that calls into `ML::create_tensor_callback`.
    /// 3. Store the created callback on `self` and return it.
    pub(crate) fn get_or_setup_callback(
        &self,
        global: &GlobalScope,
    ) -> GenericCallback<ContextMessage> {
        if let Some(cb) = self.create_tensor_cb.borrow().as_ref().cloned() {
            return cb;
        }

        let trusted_ml: Trusted<ML> = Trusted::new(self);
        let task_source = global.task_manager().ml_task_source().to_sendable();
        let cb = GenericCallback::new(global.time_profiler_chan().clone(), move |message| {
            let trusted_ml = trusted_ml.clone();
            let Ok(ctx_msg) = message else {
                return;
            };
            match ctx_msg {
                ContextMessage::CreateTensorResult(ctx_id, tensor_id, backend_result) => {
                    task_source.queue(task!(set_request_result_to_ml: move || {
                        let ml = trusted_ml.root();
                        // Route the create-tensor reply to ML which will forward to the correct MLContext.
                        ml.create_tensor_callback(ctx_id, tensor_id, backend_result, CanGc::note());
                    }));
                },

                ContextMessage::ReadTensorResult(ctx_id, tensor_id, backend_result) => {
                    task_source.queue(task!(set_read_result_to_ml: move || {
                        let ml = trusted_ml.root();
                        // Route the read-tensor reply to ML which will forward to the correct MLContext.
                        ml.read_tensor_callback(ctx_id, tensor_id, backend_result, CanGc::note());
                    }));
                },

                ContextMessage::CompileResult(ctx_id, graph_id) => {
                    task_source.queue(task!(compile_result_to_ml: move || {
                        let ml = trusted_ml.root();
                        ml.compile_callback(ctx_id, graph_id, CanGc::note());
                    }));
                },
            }
        })
        .expect("Could not create ML context persistent callback");

        self.create_tensor_cb.borrow_mut().replace(cb.clone());
        cb
    }

    /// Route a context-level callback (from the manager) to the correct MLContext.
    ///
    /// Implementation: look up the `MLContext` in `self.contexts` and invoke
    /// `MLContext::create_tensor_callback` to perform the promise resolution on the
    /// context's timeline/task source.
    pub(crate) fn create_tensor_callback(
        &self,
        context_id: ContextId,
        tensor_id: u32,
        result: Result<(), ()>,
        can_gc: CanGc,
    ) {
        let maybe_ctx = {
            let contexts = self.contexts.borrow();
            contexts.get(&context_id).cloned()
        };
        let Some(ctx) = maybe_ctx else {
            warn!("create_tensor_callback: unknown context {:?}", context_id);
            return;
        };
        ctx.create_tensor_callback(tensor_id, result, can_gc);
    }

    /// Route a read-tensor backend reply to the corresponding MLContext/MLTensor.
    pub(crate) fn read_tensor_callback(
        &self,
        context_id: ContextId,
        tensor_id: u32,
        result: Result<Vec<u8>, ()>,
        can_gc: CanGc,
    ) {
        let maybe_ctx = {
            let contexts = self.contexts.borrow();
            contexts.get(&context_id).cloned()
        };
        let Some(ctx) = maybe_ctx else {
            warn!("read_tensor_callback: unknown context {:?}", context_id);
            return;
        };
        ctx.read_tensor_callback(tensor_id, result, can_gc);
    }

    /// Route a compile-complete notification to the correct MLContext so it can
    /// resolve any promises queued by `MLGraphBuilder.build()`.  The backend
    /// only returns the graph identifier; script code no longer retains or
    /// receives the `GraphInfo` during compilation.
    pub(crate) fn compile_callback(&self, context_id: ContextId, graph_id: GraphId, can_gc: CanGc) {
        let maybe_ctx = {
            let contexts = self.contexts.borrow();
            contexts.get(&context_id).cloned()
        };
        let Some(ctx) = maybe_ctx else {
            warn!("compile_callback: unknown context {:?}", context_id);
            return;
        };
        ctx.compile_callback(graph_id, can_gc);
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<ML> {
        let ml = reflect_dom_object(Box::new(ML::new_inherited()), global, can_gc);

        ml
    }

    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    pub(crate) fn create_context(
        &self,
        options: &MLContextOptions,
        can_gc: CanGc,
    ) -> DomRoot<MLContext> {
        // Ensure ML has a persistent context-level callback set up (lazily).
        // This follows the IDBFactory "response listener" pattern: create
        // a single persistent callback and reuse it for all contexts.
        let _cb = self.get_or_setup_callback(&self.global());

        // Create a new ContextId.  The type itself handles cross-thread
        // uniqueness (see shared/webnn/src/lib.rs) so we don't need a
        // per-Global counter anymore.
        let ctx_id = ContextId::new();

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
