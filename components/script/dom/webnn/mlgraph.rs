use std::cell::Cell;

use dom_struct::dom_struct;
use rustnn::graph::GraphInfo;

use crate::dom::bindings::cell::DomRefCell;
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

    /// Implementation-defined graph representation (rustnn `GraphInfo`) owned by the MLGraph.
    #[no_trace]
    #[ignore_malloc_size_of = "rustnn::GraphInfo is external; skip malloc-size accounting"]
    graph_info: DomRefCell<GraphInfo>,
}

impl MLGraph {
    pub(crate) fn new_inherited(context: &MLContext) -> MLGraph {
        // Preserve the original default (empty) constructor used by the JS-facing
        // `new MLGraph()` binding. Graphs created from a builder should use
        // `new_inherited_with_info` / `new_with_info` below.
        MLGraph {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            is_destroyed: Cell::new(false),
            graph_info: DomRefCell::new(GraphInfo {
                operands: Vec::new(),
                input_operands: Vec::new(),
                output_operands: Vec::new(),
                operations: Vec::new(),
                constant_operand_ids_to_handles: std::collections::HashMap::new(),
                id_to_constant_tensor_operand_map: std::collections::HashMap::new(),
                quantized: false,
            }),
        }
    }

    pub(crate) fn new_with_info(
        context: &MLContext,
        graph_info: GraphInfo,
        global: &GlobalScope,
        can_gc: CanGc,
    ) -> DomRoot<MLGraph> {
        reflect_dom_object(
            Box::new(MLGraph::new_inherited_with_info(context, graph_info)),
            global,
            can_gc,
        )
    }

    fn new_inherited_with_info(context: &MLContext, graph_info: GraphInfo) -> MLGraph {
        MLGraph {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            is_destroyed: Cell::new(false),
            graph_info: DomRefCell::new(graph_info),
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

    /// Accessor for the owned GraphInfo.
    pub(crate) fn graph_info(&self) -> DomRefCell<GraphInfo> {
        self.graph_info.clone()
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
