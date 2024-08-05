#!/bin/bash

CDIR="$(cd "$(dirname "$0")" ; pwd -P)"
python_execute="$1"
source_yaml="$CDIR/torch_dpu/csrc/aten/dpu_native_functions.yaml"

echo ${python_execute} -m codegen.gen_backend_stubs  \
  --output_dir="torch_dpu/csrc/aten" \
  --source_yaml="$source_yaml"

  ${python_execute} -m codegen.gen_backend_stubs  \
    --output_dir="torch_dpu/csrc/aten" \
    --source_yaml="$source_yaml"
