#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>

/*
This defines a TVM PackedFunc named "kernel" in the compiled module.

FlashInfer-Bench TVM-FFI builder loads the module and does:
    getattr(mod, "kernel")
so we must export a symbol/function called "kernel".

We then forward the call to the global registry function registered in binding.py:
    @register_func("flashinfer.kernel")
*/
TVM_DLL_EXPORT_TYPED void kernel(TVMValue* args, int* type_codes, int num_args,
                                 TVMRetValueHandle ret) {
  auto f = tvm::runtime::Registry::Get("flashinfer.kernel");
  if (f == nullptr) {
    TVMAPISetLastError("Global function flashinfer.kernel not found");
    return;
  }

  tvm::runtime::TVMArgs tvm_args(args, type_codes, num_args);
  tvm::runtime::TVMRetValue tvm_ret;
  (*f)(tvm_args, &tvm_ret);

  // Copy return value into provided ret handle
  *reinterpret_cast<tvm::runtime::TVMRetValue*>(ret) = tvm_ret;
}