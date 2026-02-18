// Source: Web Neural Network API (WebNN)
// Direct source: https://github.com/webmachinelearning/webnn

[SecureContext, Exposed=(Window, Worker)]
interface mixin NavigatorML {
  [SameObject] readonly attribute ML ml;
};
Navigator includes NavigatorML;
WorkerNavigator includes NavigatorML;

[SecureContext, Exposed=(Window, Worker)]
interface ML {
  Promise<MLContext> createContext(optional MLContextOptions options = {});
};

[SecureContext, Exposed=(Window, Worker)]
interface MLContext {
  Promise<MLTensor> createTensor(MLTensorDescriptor descriptor);
  Promise<MLTensor> createConstantTensor(MLOperandDescriptor descriptor, /*[AllowShared]*/ BufferSource inputData);

  Promise<ArrayBuffer> readTensor(MLTensor tensor);
  Promise<undefined> readTensor(MLTensor tensor, /*[AllowShared]*/ BufferSource outputData);

  undefined writeTensor(MLTensor tensor, /*[AllowShared]*/ BufferSource inputData);

  MLOpSupportLimits opSupportLimits();

  undefined destroy();

  readonly attribute boolean accelerated;
  readonly attribute Promise<MLContextLostInfo> lost;
};

// Minimal types/dictionaries needed by MLContext methods

dictionary MLContextLostInfo {
  DOMString message;
};

enum MLOperandDataType {
  "float32",
  "float16",
  "int32",
  "uint32",
  "int64",
  "uint64",
  "int8",
  "uint8"
};

dictionary MLOperandDescriptor {
  required MLOperandDataType dataType;
  required sequence<[EnforceRange] unsigned long> shape;
};

// Generic operator options used by several ops (label etc.)
dictionary MLOperatorOptions {
  USVString label = "";
};

// Options for argMin/argMax
dictionary MLArgMinMaxOptions : MLOperatorOptions {
  boolean keepDimensions = false;
  MLOperandDataType outputDataType = "int32";
};

dictionary MLRankRange {
  unsigned long min;
  unsigned long max;
};

typedef sequence<MLOperandDataType> MLDataTypeList;

dictionary MLTensorLimits {
  MLDataTypeList dataTypes;
  MLRankRange rankRange;
};

dictionary MLOpSupportLimits {
  MLDataTypeList preferredInputLayout;
  [EnforceRange] unsigned long long maxTensorByteLength;
  MLTensorLimits input;
  MLTensorLimits constant;
  MLTensorLimits output;
};



// Minimal MLTensor descriptor and interface needed for createTensor()
// `dataType` and `shape` are required per the spec's expectations.
dictionary MLTensorDescriptor {
  required DOMString dataType;
  required sequence<long long> shape;
  boolean readable = false;
  boolean writable = false;
};

[SecureContext, Exposed=(Window, Worker)]
interface MLTensor {
  // Minimal placeholder; more members will be implemented later.
};

enum MLPowerPreference {
  "default",
  "high-performance",
  "low-power"
};

dictionary MLContextOptions {
  MLPowerPreference powerPreference = "default";
  boolean accelerated = true;
};

// Minimal graph/builder/operand interfaces (implementation is intentionally
// minimal in this repository; timeline/compilation steps are TODOs).

[SecureContext, Exposed=(Window, Worker)]
interface MLOperand {
  readonly attribute MLOperandDataType dataType;
  /* FrozenArray<unsigned long> */ readonly attribute any shape;
};

[SecureContext, Exposed=(Window, Worker)]
interface MLGraphBuilder {
  constructor(MLContext context);

  [Throws] MLOperand input(DOMString name, MLOperandDescriptor descriptor);
  [Throws] MLOperand constant(MLOperandDescriptor descriptor, /*[AllowShared]*/ BufferSource buffer);
  [Throws] MLOperand constant(MLTensor tensor);

  Promise<MLGraph> build(sequence<MLOperand> outputs);
};

partial interface MLGraphBuilder {
  [Throws] MLOperand argMin(MLOperand input, [EnforceRange] unsigned long axis,
                   optional MLArgMinMaxOptions options = {});
  [Throws] MLOperand argMax(MLOperand input, [EnforceRange] unsigned long axis,
                   optional MLArgMinMaxOptions options = {});
  [Throws] MLOperand where(MLOperand condition,
                  MLOperand trueValue,
                  MLOperand falseValue,
                  optional MLOperatorOptions options = {});
};

[SecureContext, Exposed=(Window, Worker)]
interface MLGraph {
  undefined destroy();
};
