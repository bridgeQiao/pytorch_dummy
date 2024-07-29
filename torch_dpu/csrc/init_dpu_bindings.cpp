#include <cstdio>
#include <pybind11/pybind11.h>

namespace py = pybind11;

void print_hello() {
    printf("Hello\n");
}

PYBIND11_MODULE(_C, m) {
    m.def("pirnt_hello", &print_hello, "A function that prints 'hello'");
}
