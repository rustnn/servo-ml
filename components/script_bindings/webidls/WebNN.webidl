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
