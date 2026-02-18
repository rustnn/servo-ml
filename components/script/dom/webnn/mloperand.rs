use dom_struct::dom_struct;
use js::rust::MutableHandleValue;

use crate::dom::MLTensor;
use crate::dom::bindings::codegen::Bindings::WebNNBinding::{
    MLOperandDataType, MLOperandDescriptor, MLOperandMethods,
};
use crate::dom::bindings::reflector::{Reflector, reflect_dom_object};
use crate::dom::bindings::root::{Dom, DomRoot};
use crate::dom::bindings::str::DOMString;
use crate::dom::bindings::utils::to_frozen_array;
use crate::dom::globalscope::GlobalScope;
use crate::dom::webnn::mlgraphbuilder::MLGraphBuilder;
use crate::script_runtime::{CanGc, JSContext};

#[dom_struct]
/// <https://webmachinelearning.github.io/webnn/#api-mloperand>
pub(crate) struct MLOperand {
    reflector_: Reflector,

    /// <https://webmachinelearning.github.io/webnn/#dom-mloperand-builder-slot>
    builder: Dom<MLGraphBuilder>,

    /// Backend operand id linking this DOM operand to the rustnn `Operand`.
    operand_id: Option<u32>,

    /// <https://webmachinelearning.github.io/webnn/#dom-mloperand-descriptor-slot>
    descriptor_data_type: String,
    descriptor_shape: Vec<u32>,

    /// <https://webmachinelearning.github.io/webnn/#dom-mloperand-name-slot>
    name: Option<DOMString>,

    /// Whether this operand was created as an input. (spec: input flag)
    is_input: bool,

    /// Whether this operand was created as a constant. (spec: constant flag)
    is_constant: bool,
}

impl MLOperand {
    pub(crate) fn new_inherited(
        builder: &MLGraphBuilder,
        operand_id: Option<u32>,
        descriptor_data_type: String,
        descriptor_shape: Vec<u32>,
        name: Option<DOMString>,
        is_input: bool,
        is_constant: bool,
    ) -> MLOperand {
        MLOperand {
            reflector_: Reflector::new(),
            builder: Dom::from_ref(builder),
            operand_id,
            descriptor_data_type,
            descriptor_shape,
            name,
            is_input,
            is_constant,
        }
    }

    pub(crate) fn new(
        builder: &MLGraphBuilder,
        global: &GlobalScope,
        descriptor: &MLOperandDescriptor,
        name: Option<DOMString>,
        is_input: bool,
        is_constant: bool,
        operand_id: Option<u32>,
        can_gc: CanGc,
    ) -> DomRoot<MLOperand> {
        // Extract minimal descriptor fields needed by the DOM-facing attributes.
        let data_type = descriptor.dataType.as_str().to_string();
        let shape = descriptor.shape.clone();

        reflect_dom_object(
            Box::new(MLOperand::new_inherited(
                builder,
                operand_id,
                data_type,
                shape,
                name,
                is_input,
                is_constant,
            )),
            global,
            can_gc,
        )
    }

    /// Create an operand from an existing `MLTensor` (used by `MLGraphBuilder.constant(tensor)`).
    pub(crate) fn new_from_tensor(
        builder: &MLGraphBuilder,
        global: &GlobalScope,
        tensor: &MLTensor,
        name: Option<DOMString>,
        is_input: bool,
        is_constant: bool,
        operand_id: Option<u32>,
        can_gc: CanGc,
    ) -> DomRoot<MLOperand> {
        let data_type = tensor.data_type().to_string();
        let shape = tensor.shape().iter().map(|&s| s as u32).collect();

        reflect_dom_object(
            Box::new(MLOperand::new_inherited(
                builder,
                operand_id,
                data_type,
                shape,
                name,
                is_input,
                is_constant,
            )),
            global,
            can_gc,
        )
    }

    // Internal accessor used by MLGraphBuilder algorithms (spec: "validate operand").
    pub(crate) fn builder(&self) -> Dom<MLGraphBuilder> {
        self.builder.clone()
    }

    /// Return the backend operand id associated with this DOM operand (if any).
    pub(crate) fn id(&self) -> Option<u32> {
        self.operand_id
    }

    // Minimal internal accessors for descriptor-backed attributes.
    pub(crate) fn descriptor_data_type(&self) -> &str {
        &self.descriptor_data_type
    }

    pub(crate) fn descriptor_shape(&self) -> &Vec<u32> {
        &self.descriptor_shape
    }

    pub(crate) fn is_input(&self) -> bool {
        self.is_input
    }

    pub(crate) fn is_constant(&self) -> bool {
        self.is_constant
    }

    pub(crate) fn name(&self) -> Option<DOMString> {
        self.name.clone()
    }
}

impl MLOperandMethods<crate::DomTypeHolder> for MLOperand {
    /// <https://webmachinelearning.github.io/webnn/#api-mloperand>
    fn DataType(&self) -> MLOperandDataType {
        use crate::dom::bindings::codegen::Bindings::WebNNBinding::MLOperandDataType as B;
        match self.descriptor_data_type.as_str() {
            "float32" => B::Float32,
            "float16" => B::Float16,
            "int32" => B::Int32,
            "uint32" => B::Uint32,
            "int64" => B::Int64,
            "uint64" => B::Uint64,
            "int8" => B::Int8,
            "uint8" => B::Uint8,
            _ => B::Float32,
        }
    }

    /// <https://webmachinelearning.github.io/webnn/#api-mloperand>
    fn Shape(&self, cx: JSContext, retval: MutableHandleValue) {
        // Return the descriptor shape as a FrozenArray-like JS array per WebIDL.
        to_frozen_array(self.descriptor_shape.as_slice(), cx, retval, CanGc::note());
    }
}
