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

typedef record<USVString, MLTensor> MLNamedTensors;

// Named operands used for graph builder outputs (spec uses {name: operand} records).
typedef record<USVString, MLOperand> MLNamedOperands;

[SecureContext, Exposed=(Window, Worker)]
interface MLContext {
  Promise<MLTensor> createTensor(MLTensorDescriptor descriptor);
  Promise<MLTensor> createConstantTensor(MLOperandDescriptor descriptor, /*[AllowShared]*/ BufferSource inputData);

  Promise<ArrayBuffer> readTensor(MLTensor tensor);
  Promise<undefined> readTensor(MLTensor tensor, /*[AllowShared]*/ BufferSource outputData);

  [Throws] undefined writeTensor(MLTensor tensor, /*[AllowShared]*/ BufferSource inputData);

  MLOpSupportLimits opSupportLimits();

  undefined destroy();

  [Throws] undefined dispatch(MLGraph graph, MLNamedTensors inputs, MLNamedTensors outputs);

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

// Options for batchNormalization
dictionary MLBatchNormalizationOptions : MLOperatorOptions {
  MLOperand scale;
  MLOperand bias;
  [EnforceRange] unsigned long axis = 1;
  double epsilon = 1e-5;
};

// Options for clamp()
dictionary MLClampOptions : MLOperatorOptions {
  double minValue;
  double maxValue;
};

// Options for conv2d

// Layout enums used by conv2d options and other ops in the spec.
enum MLInputOperandLayout {
  "nchw",
  "nhwc"
};

enum MLConv2dFilterOperandLayout {
  "oihw",
  "hwio",
  "ohwi",
  "ihwo"
};

dictionary MLConv2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> padding;
  sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations;
  [EnforceRange] unsigned long groups = 1;
  MLInputOperandLayout inputLayout = "nchw";
  MLConv2dFilterOperandLayout filterLayout = "oihw";
  MLOperand bias;
};

// Options for GEMM (General Matrix Multiplication)
dictionary MLGemmOptions : MLOperatorOptions {
  MLOperand c;
  double alpha = 1.0;
  double beta = 1.0;
  boolean aTranspose = false;
  boolean bTranspose = false;
};

dictionary MLTransposeOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> permutation;
};

dictionary MLTriangularOptions : MLOperatorOptions {
  boolean upper = true;
  [EnforceRange] long diagonal = 0;
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

// Support limits for operators with a single tensor input and a single tensor
// output.  The spec uses this for many elementwise ops and for `cast`.
dictionary MLSingleInputSupportLimits {
  MLTensorLimits input;
  MLTensorLimits output;
};

dictionary MLOpSupportLimits {
  MLDataTypeList preferredInputLayout;
  [EnforceRange] unsigned long long maxTensorByteLength;
  MLTensorLimits input;
  MLTensorLimits constant;
  MLTensorLimits output;
  // Per-operator support limit members; tests expect `cast` to exist.
  MLSingleInputSupportLimits cast;
  MLSingleInputSupportLimits transpose;
  MLSingleInputSupportLimits triangular;
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

  // The spec requires a record mapping output names to operands.
  Promise<MLGraph> build(MLNamedOperands outputs);
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
  [Throws] MLOperand batchNormalization(MLOperand input, MLOperand mean, MLOperand variance,
                                       optional MLBatchNormalizationOptions options = {});
  [Throws] MLOperand cast(MLOperand input, MLOperandDataType dataType, optional MLOperatorOptions options = {});
  [Throws] MLOperand clamp(MLOperand input, optional MLClampOptions options = {});
  [Throws] MLOperand triangular(MLOperand input, optional MLTriangularOptions options = {});
  [Throws] MLOperand concat(sequence<MLOperand> inputs, [EnforceRange] unsigned long axis, optional MLOperatorOptions options = {});
  [Throws] MLOperand conv2d(MLOperand input, MLOperand filter, optional MLConv2dOptions options = {});
};

partial interface MLGraphBuilder {
  [Throws] MLOperand add(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand sub(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand mul(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand div(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand max(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand min(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand pow(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand matmul(MLOperand a, MLOperand b);
  [Throws] MLOperand gemm(MLOperand a, MLOperand b, optional MLGemmOptions options = {});
  [Throws] MLOperand transpose(MLOperand input, optional MLTransposeOptions options = {});
};

[SecureContext, Exposed=(Window, Worker)]
interface MLGraph {
  undefined destroy();
};
