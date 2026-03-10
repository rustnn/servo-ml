/* This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/. */

// https://webmachinelearning.github.io/webnn/#idl-index

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

typedef unrestricted double MLNumber;

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

dictionary MLReduceOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> axes;
  boolean keepDimensions = false;
};

dictionary MLGatherOptions : MLOperatorOptions {
  [EnforceRange] unsigned long axis = 0;
};

dictionary MLSplitOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> splits;
  [EnforceRange] unsigned long axis = 0;
};

enum MLPaddingMode {
  "constant",
  "edge",
  "reflection"
};

dictionary MLPadOptions : MLOperatorOptions {
  MLPaddingMode mode = "constant";
  MLNumber value = 0;
};

dictionary MLEluOptions : MLOperatorOptions {
  double alpha = 1.0;
};

dictionary MLLeakyReluOptions : MLOperatorOptions {
  double alpha = 0.01;
};

dictionary MLHardSigmoidOptions : MLOperatorOptions {
  double alpha = 0.2;
  double beta = 0.5;
};

dictionary MLLinearOptions : MLOperatorOptions {
  double alpha = 1.0;
  double beta = 0.0;
};

dictionary MLReverseOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> axes;
};

dictionary MLInstanceNormalizationOptions : MLOperatorOptions {
  MLOperand scale;
  MLOperand bias;
  double epsilon = 1e-5;
  MLInputOperandLayout layout = "nchw";
};

dictionary MLLayerNormalizationOptions : MLOperatorOptions {
  MLOperand scale;
  MLOperand bias;
  sequence<[EnforceRange] unsigned long> axes;
  double epsilon = 1e-5;
};

enum MLInterpolationMode {
  "nearest-neighbor",
  "linear"
};

dictionary MLResample2dOptions : MLOperatorOptions {
  MLInterpolationMode mode = "nearest-neighbor";
  sequence<float> scales;
  sequence<[EnforceRange] unsigned long> sizes;
  sequence<[EnforceRange] unsigned long> axes;
};

dictionary MLSoftmaxOptions : MLOperatorOptions {
  long axis = 1;
};

dictionary MLCumulativeSumOptions : MLOperatorOptions {
  boolean exclusive = false;
  boolean reversed = false;
};

enum MLRoundingType {
  "floor",
  "ceil"
};

dictionary MLPool2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> windowDimensions;
  sequence<[EnforceRange] unsigned long> padding;
  sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations;
  MLInputOperandLayout layout = "nchw";
  MLRoundingType outputShapeRounding = "floor";
  sequence<[EnforceRange] unsigned long> outputSizes;
};

enum MLConvTranspose2dFilterOperandLayout {
  "iohw",
  "hwoi",
  "ohwi",
  "oihw"
};

dictionary MLConvTranspose2dOptions : MLOperatorOptions {
  sequence<[EnforceRange] unsigned long> padding;
  sequence<[EnforceRange] unsigned long> strides;
  sequence<[EnforceRange] unsigned long> dilations;
  sequence<[EnforceRange] unsigned long> outputPadding;
  sequence<[EnforceRange] unsigned long> outputSizes;
  [EnforceRange] unsigned long groups = 1;
  MLInputOperandLayout inputLayout = "nchw";
  MLConvTranspose2dFilterOperandLayout filterLayout = "iohw";
  MLOperand bias;
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

dictionary MLBinarySupportLimits {
  MLTensorLimits a;
  MLTensorLimits b;
  MLTensorLimits output;
};

dictionary MLBatchNormalizationSupportLimits {
  MLTensorLimits input;
  MLTensorLimits mean;
  MLTensorLimits variance;
  MLTensorLimits scale;
  MLTensorLimits bias;
  MLTensorLimits output;
};

dictionary MLConcatSupportLimits {
  MLTensorLimits inputs;
  MLTensorLimits output;
};

dictionary MLConv2dSupportLimits {
  MLTensorLimits input;
  MLTensorLimits filter;
  MLTensorLimits bias;
  MLTensorLimits output;
};

dictionary MLQuantizeDequantizeLinearSupportLimits {
  MLTensorLimits input;
  MLTensorLimits scale;
  MLTensorLimits zeroPoint;
  MLTensorLimits output;
};

dictionary MLGemmSupportLimits {
  MLTensorLimits a;
  MLTensorLimits b;
  MLTensorLimits c;
  MLTensorLimits output;
};

dictionary MLNormalizationSupportLimits {
  MLTensorLimits input;
  MLTensorLimits scale;
  MLTensorLimits bias;
  MLTensorLimits output;
};

dictionary MLGatherSupportLimits {
  MLTensorLimits input;
  MLTensorLimits indices;
  MLTensorLimits output;
};

dictionary MLPreluSupportLimits {
  MLTensorLimits input;
  MLTensorLimits slope;
  MLTensorLimits output;
};

dictionary MLScatterSupportLimits {
  MLTensorLimits input;
  MLTensorLimits indices;
  MLTensorLimits updates;
  MLTensorLimits output;
};

dictionary MLSplitSupportLimits {
  MLTensorLimits input;
  MLTensorLimits outputs;
};

dictionary MLWhereSupportLimits {
  MLTensorLimits condition;
  MLTensorLimits trueValue;
  MLTensorLimits falseValue;
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
  MLSingleInputSupportLimits pad;
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
  [Throws] MLOperand abs(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand ceil(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand cos(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand erf(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand exp(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand floor(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand identity(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand log(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand neg(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand reciprocal(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand roundEven(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand sin(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand sign(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand sqrt(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand tan(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand tanh(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand add(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand sub(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand mul(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand div(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand max(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand min(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand pow(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand matmul(MLOperand a, MLOperand b);
  [Throws] MLOperand gemm(MLOperand a, MLOperand b, optional MLGemmOptions options = {});
  [Throws] MLOperand tile(MLOperand input, sequence<unsigned long> repetitions,
                          optional MLOperatorOptions options = {});
  [Throws] MLOperand transpose(MLOperand input, optional MLTransposeOptions options = {});
  [Throws] MLOperand averagePool2d(MLOperand input, optional MLPool2dOptions options = {});
  [Throws] MLOperand l2Pool2d(MLOperand input, optional MLPool2dOptions options = {});
  [Throws] MLOperand maxPool2d(MLOperand input, optional MLPool2dOptions options = {});
  [Throws] MLOperand convTranspose2d(MLOperand input, MLOperand filter,
                                     optional MLConvTranspose2dOptions options = {});

  [Throws] MLOperand cumulativeSum(MLOperand input, [EnforceRange] unsigned long axis,
                                   optional MLCumulativeSumOptions options = {});

  [Throws] MLOperand elu(MLOperand input, optional MLEluOptions options = {});
  [Throws] MLOperand gelu(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand hardSigmoid(MLOperand input, optional MLHardSigmoidOptions options = {});
  [Throws] MLOperand hardSwish(MLOperand input, optional MLHardSigmoidOptions options = {});
  [Throws] MLOperand leakyRelu(MLOperand input, optional MLLeakyReluOptions options = {});
  [Throws] MLOperand linear(MLOperand input, optional MLLinearOptions options = {});
  [Throws] MLOperand sigmoid(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand softplus(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand softsign(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand reverse(MLOperand input, optional MLReverseOptions options = {});

  [Throws] MLOperand equal(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand greater(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand greaterOrEqual(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand lesser(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand lesserOrEqual(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand notEqual(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});

  [Throws] MLOperand logicalAnd(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand logicalOr(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand logicalXor(MLOperand a, MLOperand b, optional MLOperatorOptions options = {});
  [Throws] MLOperand logicalNot(MLOperand input, optional MLOperatorOptions options = {});

  [Throws] MLOperand isNaN(MLOperand input, optional MLOperatorOptions options = {});
  [Throws] MLOperand isInfinite(MLOperand input, optional MLOperatorOptions options = {});

  [Throws] MLOperand gather(MLOperand input, MLOperand indices, optional MLGatherOptions options = {});
  [Throws] MLOperand gatherElements(MLOperand input, MLOperand indices,
                                    optional MLGatherOptions options = {});
  [Throws] MLOperand gatherND(MLOperand input, MLOperand indices, optional MLOperatorOptions options = {});

  [Throws] MLOperand scatterElements(MLOperand input, MLOperand indices, MLOperand updates,
                                     optional MLGatherOptions options = {});
  [Throws] MLOperand scatterND(MLOperand input, MLOperand indices, MLOperand updates,
                               optional MLOperatorOptions options = {});

  [Throws] MLOperand reshape(MLOperand input, sequence<[EnforceRange] unsigned long> newShape,
                             optional MLOperatorOptions options = {});
  [Throws] MLOperand expand(MLOperand input, sequence<[EnforceRange] unsigned long> newShape,
                            optional MLOperatorOptions options = {});

  [Throws] MLOperand slice(MLOperand input,
                           sequence<[EnforceRange] unsigned long> starts,
                           sequence<[EnforceRange] unsigned long> sizes,
                           optional MLOperatorOptions options = {});

  [Throws] MLOperand pad(MLOperand input,
                         sequence<[EnforceRange] unsigned long> beginningPadding,
                         sequence<[EnforceRange] unsigned long> endingPadding,
                         optional MLPadOptions options = {});

  [Throws] MLOperand softmax(MLOperand input, optional MLSoftmaxOptions options = {});
  [Throws] sequence<MLOperand> split(MLOperand input, [EnforceRange] unsigned long splits,
                                     optional MLSplitOptions options = {});
  [Throws] sequence<MLOperand> split(MLOperand input,
                                     sequence<[EnforceRange] unsigned long> splits,
                                     optional MLSplitOptions options = {});

  [Throws] MLOperand reduceL1(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceL2(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceLogSum(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceLogSumExp(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceMax(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceMean(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceMin(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceProduct(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceSumSquare(MLOperand input, optional MLReduceOptions options = {});
  [Throws] MLOperand reduceSum(MLOperand input, optional MLReduceOptions options = {});

  [Throws] MLOperand quantizeLinear(MLOperand input, MLOperand scale, MLOperand zeroPoint,
                                    optional MLOperatorOptions options = {});
  [Throws] MLOperand dequantizeLinear(MLOperand input, MLOperand scale, MLOperand zeroPoint,
                                      optional MLOperatorOptions options = {});

  [Throws] MLOperand instanceNormalization(MLOperand input, optional MLInstanceNormalizationOptions options = {});
  [Throws] MLOperand layerNormalization(MLOperand input, optional MLLayerNormalizationOptions options = {});
  [Throws] MLOperand prelu(MLOperand input, MLOperand slope, optional MLOperatorOptions options = {});
  [Throws] MLOperand resample2d(MLOperand input,
                                optional MLResample2dOptions options = {});
};

partial dictionary MLOpSupportLimits {
  MLSingleInputSupportLimits argMin;
  MLSingleInputSupportLimits argMax;
  MLBinarySupportLimits add;
  MLSingleInputSupportLimits averagePool2d;
  MLBatchNormalizationSupportLimits batchNormalization;
  MLSingleInputSupportLimits clamp;
  MLConcatSupportLimits concat;
  MLConv2dSupportLimits conv2d;
  MLConv2dSupportLimits convTranspose2d;
  MLSingleInputSupportLimits cumulativeSum;
  MLQuantizeDequantizeLinearSupportLimits dequantizeLinear;
  MLBinarySupportLimits div;
  MLSingleInputSupportLimits elu;
  MLBinarySupportLimits equal;
  MLSingleInputSupportLimits expand;
  MLGatherSupportLimits gather;
  MLGatherSupportLimits gatherElements;
  MLGatherSupportLimits gatherND;
  MLSingleInputSupportLimits gelu;
  MLGemmSupportLimits gemm;
  MLBinarySupportLimits greater;
  MLBinarySupportLimits greaterOrEqual;
  MLSingleInputSupportLimits hardSigmoid;
  MLSingleInputSupportLimits hardSwish;
  MLSingleInputSupportLimits isInfinite;
  MLSingleInputSupportLimits isNaN;
  MLNormalizationSupportLimits instanceNormalization;
  MLNormalizationSupportLimits layerNormalization;
  MLSingleInputSupportLimits l2Pool2d;
  MLSingleInputSupportLimits leakyRelu;
  MLBinarySupportLimits lesser;
  MLBinarySupportLimits lesserOrEqual;
  MLSingleInputSupportLimits linear;
  MLSingleInputSupportLimits logicalNot;
  MLBinarySupportLimits logicalAnd;
  MLBinarySupportLimits logicalOr;
  MLBinarySupportLimits logicalXor;
  MLBinarySupportLimits matmul;
  MLSingleInputSupportLimits maxPool2d;
  MLBinarySupportLimits max;
  MLBinarySupportLimits min;
  MLBinarySupportLimits mul;
  MLBinarySupportLimits notEqual;
  MLPreluSupportLimits prelu;
  MLBinarySupportLimits pow;
  MLQuantizeDequantizeLinearSupportLimits quantizeLinear;
  MLSingleInputSupportLimits reduceL1;
  MLSingleInputSupportLimits reduceL2;
  MLSingleInputSupportLimits reduceLogSum;
  MLSingleInputSupportLimits reduceLogSumExp;
  MLSingleInputSupportLimits reduceMax;
  MLSingleInputSupportLimits reduceMean;
  MLSingleInputSupportLimits reduceMin;
  MLSingleInputSupportLimits reduceProduct;
  MLSingleInputSupportLimits reduceSum;
  MLSingleInputSupportLimits reduceSumSquare;
  MLSingleInputSupportLimits resample2d;
  MLSingleInputSupportLimits reshape;
  MLSingleInputSupportLimits reverse;
  MLScatterSupportLimits scatterElements;
  MLScatterSupportLimits scatterND;
  MLSingleInputSupportLimits sigmoid;
  MLSingleInputSupportLimits slice;
  MLSingleInputSupportLimits softmax;
  MLSingleInputSupportLimits softplus;
  MLSingleInputSupportLimits softsign;
  MLSplitSupportLimits split;
  MLBinarySupportLimits sub;
  MLWhereSupportLimits where;

  MLSingleInputSupportLimits abs;
  MLSingleInputSupportLimits ceil;
  MLSingleInputSupportLimits cos;
  MLSingleInputSupportLimits erf;
  MLSingleInputSupportLimits exp;
  MLSingleInputSupportLimits floor;
  MLSingleInputSupportLimits identity;
  MLSingleInputSupportLimits log;
  MLSingleInputSupportLimits neg;
  MLSingleInputSupportLimits reciprocal;
  MLSingleInputSupportLimits roundEven;
  MLSingleInputSupportLimits sin;
  MLSingleInputSupportLimits sign;
  MLSingleInputSupportLimits sqrt;
  MLSingleInputSupportLimits tan;
  MLSingleInputSupportLimits tanh;
  MLSingleInputSupportLimits tile;
};

[SecureContext, Exposed=(Window, Worker)]
interface MLGraph {
  undefined destroy();
};
