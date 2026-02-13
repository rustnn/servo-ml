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
  readonly attribute boolean accelerated;
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
