import cupy as cp
print(cp.__version__)
print(cp.cuda.runtime.runtimeGetVersion())
print(cp.cuda.runtime.driverGetVersion())
print(cp.cuda.runtime.getDeviceCount())
print(cp.cuda.Device(0).compute_capability)