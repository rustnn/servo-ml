use std::rc::Rc;

use dom_struct::dom_struct;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLContextMethods, MLPowerPreference, MLTensorDescriptor,
};
use crate::dom::bindings::reflector::{DomGlobal, Reflector, reflect_dom_object};
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::dom::webnn::mltensor::MLTensor;
use crate::script_runtime::CanGc;
#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-mlcontext>
pub(crate) struct MLContext {
    reflector_: Reflector,

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
    /// Note: implements the ML "To create a context" constructor steps that initialize
    /// the context's internal slots; mapping is not 1:1 with the spec algorithm.
    pub(crate) fn new_inherited(
        accelerated: bool,
        power_preference: MLPowerPreference,
        lost: Rc<Promise>,
    ) -> MLContext {
        // Step 1.1: Let |context| be a new MLContext in |realm| (constructor value).
        // Step 1.2: Set |context|.[[contextType]] to "default".
        // Step 1.3: Set |context|.[[powerPreference]] to the provided value.
        // Step 1.4: Set |context|.[[accelerated]] to the provided `accelerated` value.
        MLContext {
            reflector_: Reflector::new(),
            context_type: "default".into(),
            power_preference,
            accelerated,
            lost,
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#api-ml-createcontext>
    pub(crate) fn new(
        global: &GlobalScope,
        accelerated: bool,
        power_preference: MLPowerPreference,
        can_gc: CanGc,
    ) -> DomRoot<MLContext> {
        // Step 1.6: Set |context|.[[lost]] to a new promise in |realm|.
        let lost_promise = Promise::new(global, can_gc);
        let ctx = reflect_dom_object(
            Box::new(MLContext::new_inherited(
                accelerated,
                power_preference,
                lost_promise.clone(),
            )),
            global,
            can_gc,
        );

        ctx
    }
}

impl MLContextMethods<crate::DomTypeHolder> for MLContext {
    /// <https://webmachinelearning.github.io/webnn/#dom-mlcontext-accelerated>
    fn Accelerated(&self) -> bool {
        // Step 1: Return this.[[accelerated]].
        self.accelerated
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mlcontext-createtensor>
    fn CreateTensor(&self, descriptor: &MLTensorDescriptor) -> Rc<Promise> {
        // Step 1: Let |global| be this's relevant global object.
        let global = &self.global();

        // Step 2: Let |realm| be this's relevant realm.
        // Note: the realm is represented by `global` in this implementation.

        // Step 3: If |this| is lost, return a new promise in |realm| rejected with an InvalidStateError.
        if self.lost.is_fulfilled() {
            // Step 3: create and return the rejected promise in |realm|.
            let p = Promise::new(global, CanGc::note());
            p.reject_error(
                crate::dom::bindings::error::Error::InvalidState(None),
                CanGc::note(),
            );

            return p;
        }

        // Step 4: Let |tensor| be the result of creating an MLTensor given |this| and |descriptor|.
        let tensor = MLTensor::new(self, global, descriptor, CanGc::note());

        // Step 5: Let |promise| be a new promise in |realm|.
        let p = Promise::new(global, CanGc::note());

        // Step 6: TODO — enqueue timeline steps to allocate and zero-initialize |tensor|.[[data]].
        // TODO (spec: #api-mlcontext-createtensor): implement ML timeline task queuing and
        // the zero-initialization/allocation of tensor.[[data]]; that task must resolve or
        // reject |promise| asynchronously. Do NOT resolve |promise| here.

        // Step 7: Return |promise|.
        p
    }
}
