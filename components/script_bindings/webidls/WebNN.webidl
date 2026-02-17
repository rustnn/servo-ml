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

  readonly attribute boolean accelerated;
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
