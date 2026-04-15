@echo off
setlocal

set GLSLC=..\.claude\tools\glslangValidator.exe

echo Compiling shaders...

%GLSLC% -V double_floats.comp -o double_floats.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: double_floats.comp
    exit /b 1
)

%GLSLC% -V gate_scan.comp -o gate_scan.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: gate_scan.comp
    exit /b 1
)

%GLSLC% -V accumulate.comp -o accumulate.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: accumulate.comp
    exit /b 1
)

%GLSLC% -V rmsnorm.comp -o rmsnorm.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: rmsnorm.comp
    exit /b 1
)

%GLSLC% -V rmsnorm_copy.comp -o rmsnorm_copy.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: rmsnorm_copy.comp
    exit /b 1
)

%GLSLC% -V rmsnorm_batch.comp -o rmsnorm_batch.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: rmsnorm_batch.comp
    exit /b 1
)

%GLSLC% -V rmsnorm_copy_batch.comp -o rmsnorm_copy_batch.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: rmsnorm_copy_batch.comp
    exit /b 1
)

%GLSLC% -V kv_cache_store.comp -o kv_cache_store.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: kv_cache_store.comp
    exit /b 1
)

%GLSLC% -V kv_cache_store_batch.comp -o kv_cache_store_batch.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: kv_cache_store_batch.comp
    exit /b 1
)

%GLSLC% -V rope_batch.comp -o rope_batch.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: rope_batch.comp
    exit /b 1
)

%GLSLC% -V attn_scores_prefill.comp -o attn_scores_prefill.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: attn_scores_prefill.comp
    exit /b 1
)

%GLSLC% -V softmax_prefill.comp -o softmax_prefill.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: softmax_prefill.comp
    exit /b 1
)

%GLSLC% -V attn_value_prefill.comp -o attn_value_prefill.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: attn_value_prefill.comp
    exit /b 1
)

%GLSLC% -V matvec_f16.comp -o matvec_f16.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: matvec_f16.comp
    exit /b 1
)

%GLSLC% -V matvec_q4_0.comp -o matvec_q4_0.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: matvec_q4_0.comp
    exit /b 1
)

%GLSLC% -V matvec_q8_0.comp -o matvec_q8_0.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: matvec_q8_0.comp
    exit /b 1
)

%GLSLC% -V matmul_f16.comp -o matmul_f16.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: matmul_f16.comp
    exit /b 1
)

%GLSLC% -V matmul_q8_0.comp -o matmul_q8_0.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: matmul_q8_0.comp
    exit /b 1
)

%GLSLC% -V matvec_q4_k.comp -o matvec_q4_k.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: matvec_q4_k.comp
    exit /b 1
)

%GLSLC% -V matvec_q6_k.comp -o matvec_q6_k.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: matvec_q6_k.comp
    exit /b 1
)

%GLSLC% -V qk_norm.comp -o qk_norm.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: qk_norm.comp
    exit /b 1
)

%GLSLC% -V rope.comp -o rope.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: rope.comp
    exit /b 1
)

%GLSLC% -V attn_scores.comp -o attn_scores.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: attn_scores.comp
    exit /b 1
)

%GLSLC% -V softmax.comp -o softmax.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: softmax.comp
    exit /b 1
)

%GLSLC% -V attn_value.comp -o attn_value.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: attn_value.comp
    exit /b 1
)

%GLSLC% -V gelu_mul.comp -o gelu_mul.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: gelu_mul.comp
    exit /b 1
)

%GLSLC% -V vec_add.comp -o vec_add.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: vec_add.comp
    exit /b 1
)

%GLSLC% -V embed_lookup_f16.comp -o embed_lookup_f16.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: embed_lookup_f16.comp
    exit /b 1
)

%GLSLC% -V embed_lookup_q8.comp -o embed_lookup_q8.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: embed_lookup_q8.comp
    exit /b 1
)

%GLSLC% -V embed_lookup_batch_f16.comp -o embed_lookup_batch_f16.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: embed_lookup_batch_f16.comp
    exit /b 1
)

%GLSLC% -V embed_lookup_batch_q8.comp -o embed_lookup_batch_q8.spv
if %ERRORLEVEL% NEQ 0 (
    echo FAILED: embed_lookup_batch_q8.comp
    exit /b 1
)

echo Done.