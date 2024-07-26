#include <Python.h>
#include <vector>

PyObject *module;
static std::vector<PyMethodDef> methods;

void AddPyMethodDefs(std::vector<PyMethodDef> &vector, PyMethodDef *methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

PyObject *hello(PyObject *) {
  printf("[Info] hello world.\n");
  Py_RETURN_NONE;
}
static PyMethodDef TorchDpuHello[] = {
    {"_dpu_hello", (PyCFunction)hello, METH_NOARGS, nullptr},
    {nullptr, nullptr, 0, nullptr}};

extern "C" PyObject *initModule() {
  AddPyMethodDefs(methods, TorchDpuHello);
  static struct PyModuleDef torchdpu_module = {
      PyModuleDef_HEAD_INIT, "torch_dpu._C", nullptr, -1, methods.data()};
  module = PyModule_Create(&torchdpu_module);
  return module;
}

PyMODINIT_FUNC PyInit__C(void) { return initModule(); }
