use std::rc::Rc;

use dom_struct::dom_struct;

use crate::dom::bindings::codegen::Bindings::WebNNBinding::{MLMethods, MLContextOptions};
use crate::dom::bindings::reflector::{Reflector, reflect_dom_object};
use crate::dom::bindings::root::DomRoot;
use crate::dom::globalscope::GlobalScope;
use crate::dom::promise::Promise;
use crate::script_runtime::CanGc;

#[dom_struct]
/// `ML` (WebNN) — exposed as `navigator.ml`.
/// See: https://w3c.github.io/webnn/
pub(crate) struct ML {
    reflector_: Reflector,
}

impl ML {
    pub(crate) fn new_inherited() -> ML {
        ML { reflector_: Reflector::new() }
    }

    pub(crate) fn new(global: &GlobalScope, can_gc: CanGc) -> DomRoot<ML> {
        reflect_dom_object(Box::new(ML::new_inherited()), global, can_gc)
    }
}

impl MLMethods<crate::DomTypeHolder> for ML {
    /// <https://w3c.github.io/webnn/#dom-ml-createcontext>
    fn CreateContext(&self, _options: &MLContextOptions) -> Rc<Promise> {
        todo!("ML::CreateContext is not implemented yet");
    }
}
