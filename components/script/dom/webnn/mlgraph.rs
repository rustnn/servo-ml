/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

use std::cell::Cell;

use dom_struct::dom_struct;
use rustnn::graph::GraphInfo;
use webnn_traits::GraphId;

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

    /// Identifier supplied by the builder.  Zero is used for graphs created
    /// directly via `new MLGraph()`.
    #[no_trace]
    graph_id: GraphId,

    /// Implementation-defined graph representation (rustnn `GraphInfo`) owned by the MLGraph.
    #[no_trace]
    #[ignore_malloc_size_of = "rustnn::GraphInfo is external; skip malloc-size accounting"]
    graph_info: DomRefCell<GraphInfo>,
}

impl MLGraph {
    /// Internal helper: create a graph with supplied id and info.
    fn new_inherited(context: &MLContext, graph_id: GraphId, graph_info: GraphInfo) -> MLGraph {
        MLGraph {
            reflector_: Reflector::new(),
            context: Dom::from_ref(context),
            is_destroyed: Cell::new(false),
            graph_id,
            graph_info: DomRefCell::new(graph_info),
        }
    }

    pub(crate) fn new(
        context: &MLContext,
        graph_id: GraphId,
        graph_info: GraphInfo,
        global: &GlobalScope,
        can_gc: CanGc,
    ) -> DomRoot<MLGraph> {
        reflect_dom_object(
            Box::new(MLGraph::new_inherited(context, graph_id, graph_info)),
            global,
            can_gc,
        )
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

    /// Retrieve this graph's identifier.
    pub(crate) fn graph_id(&self) -> GraphId {
        self.graph_id
    }
}

impl MLGraphMethods<crate::DomTypeHolder> for MLGraph {
    /// <https://webmachinelearning.github.io/webnn/#api-mlgraph-destroy>
    fn Destroy(&self, _can_gc: CanGc) {
        // Step 1: If [=this=].[[isDestroyed]] is true, then abort these steps.
        if self.is_destroyed() {
            return;
        }

        // Step 2: Set [=this=].[[isDestroyed]] to true.
        self.is_destroyed.set(true);

        // TODO (spec: #api-mlgraph-destroy): release platform resources associated with the graph.
    }
}
