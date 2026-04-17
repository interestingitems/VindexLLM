{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Embeddings;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.Math,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.GGUFReader,
  VindexLLM.Compute,
  VindexLLM.LayerNorm,
  VindexLLM.Attention,
  VindexLLM.Tokenizer,
  VindexLLM.ChatTemplate,
  VindexLLM.Shaders,
  VindexLLM.FFNWeights;

type

  // Push-constant structs that mirror the ones declared in
  // VindexLLM.Inference — we can't re-use those directly without pulling
  // in the whole inference unit, so we redeclare the shape here. Must stay
  // binary-compatible with the SPIR-V shader layouts.

  { TVdxEmbGeluMulPush }
  TVdxEmbGeluMulPush = record
    Count: UInt32;
  end;

  { TVdxEmbVecAddPush }
  TVdxEmbVecAddPush = record
    Count: UInt32;
  end;

  { TVdxEmbBatchPush }
  TVdxEmbBatchPush = record
    DimParam: UInt32;    // hidden_dim/2 for F16, hidden_dim for Q8_0
    EmbedScale: Single;
    NumTokens: UInt32;
  end;

  { TVdxEmbeddingsEvent }
  TVdxEmbeddingsEvent = (
    eeLoadStart,
    eeLoadEnd,
    eeUnloadStart,
    eeUnloadEnd,
    eeEmbedStart,
    eeEmbedEnd
  );

  { TVdxEmbeddingsEventCallback }
  TVdxEmbeddingsEventCallback = reference to procedure(
    const AEvent: TVdxEmbeddingsEvent;
    const AUserData: Pointer);

  { TVdxEmbeddings — loads an embedding-style transformer (EmbeddingGemma)
    and produces L2-normalized vectors for arbitrary text. Mirrors the
    subsystem layout of TVdxInference but runs a single forward pass
    that ends in mean-pooling + normalization instead of unembedding. }
  TVdxEmbeddings = class(TVdxErrorsObject)
  private
    // Subsystem objects — same set as TVdxInference, independent instances
    FReader: TVdxGGUFReader;
    FCompute: TVdxVulkanCompute;
    FNorm: TVdxLayerNorm;
    FAttn: TVdxAttention;
    FVindex: TVdxFFNWeights;
    FTokenizer: TVdxTokenizer;

    FEventCallback: TVdxCallback<TVdxEmbeddingsEventCallback>;

    FModelLoaded: Boolean;

    // Model config (read from GGUF metadata)
    FArchitecture: string;
    FNumLayers: UInt32;
    FHiddenDim: UInt32;
    FFFNWidth: UInt32;
    FNumQHeads: UInt32;
    FNumKVHeads: UInt32;
    FHeadDim: UInt32;
    FVocabSize: Integer;
    FMaxSeqLen: UInt32;
    FEmbeddingDim: Integer;

    // Embedding table (mmap'd from GGUF) — reused across all Embed() calls
    FEmbedPtr: PByte;
    FEmbedScale: Single;
    FEmbedType: TVdxGGMLType;

    // Per-layer weights
    FAttnWeights: array of TVdxAttnLayerWeights;
    FNormWeights: array of TVdxNormLayerWeights;
    FUpWeights: array of TVdxGpuBuffer;
    FWeightType: TVdxGGMLType;

    // Final output norm applied to last-layer hidden states before pooling
    FOutputNormGpu: TVdxGpuBuffer;

    // Post-pooling sentence-transformers dense projections.
    // EmbeddingGemma: mean_pool → Dense(768→3072) → Dense(3072→768) → L2.
    // Weights live on CPU — applied once per Embed() call, trivial cost
    // vs the GPU forward pass. If the GGUF doesn't include these tensors
    // FProjectionsLoaded stays False and Embed() returns the normalized
    // mean-pooled vector (still usable, just not the canonical output).
    FDense1Weights: array of Single;   // [Dense1Out x Dense1In] row-major
    FDense2Weights: array of Single;   // [Dense2Out x Dense2In] row-major
    FDense1In: Integer;
    FDense1Out: Integer;
    FDense2In: Integer;
    FDense2Out: Integer;
    FProjectionsLoaded: Boolean;

    // Batched prefill matrix buffers — one row per input token
    FResidualMat: TVdxGpuBuffer;
    FWorkMat: TVdxGpuBuffer;
    FQMat: TVdxGpuBuffer;
    FKMat: TVdxGpuBuffer;
    FVMat: TVdxGpuBuffer;
    FAttnOutMatBuf: TVdxGpuBuffer;
    FGateMat: TVdxGpuBuffer;
    FUpMatBuf: TVdxGpuBuffer;
    FFFNOutMat: TVdxGpuBuffer;

    // Vulkan pipelines — identical layouts to TVdxInference's equivalents
    FGeluMulShader: VkShaderModule;
    FGeluMulBundle: TVdxComputePipelineBundle;
    FGeluMulDescLayout: VkDescriptorSetLayout;
    FGeluMulDescPool: VkDescriptorPool;
    FGeluMulDescSet: VkDescriptorSet;

    FVecAddShader: VkShaderModule;
    FVecAddBundle: TVdxComputePipelineBundle;
    FVecAddDescLayout: VkDescriptorSetLayout;
    FVecAddDescPool: VkDescriptorPool;
    FVecAddAttnDescSet: VkDescriptorSet;
    FVecAddFFNDescSet: VkDescriptorSet;

    // Batched embedding lookup — F16, Q8_0, Q4_0 variants
    FEmbedBatchF16Shader: VkShaderModule;
    FEmbedBatchQ8Shader: VkShaderModule;
    FEmbedBatchQ4Shader: VkShaderModule;
    FEmbedBatchF16Bundle: TVdxComputePipelineBundle;
    FEmbedBatchQ8Bundle: TVdxComputePipelineBundle;
    FEmbedBatchQ4Bundle: TVdxComputePipelineBundle;
    FEmbedBatchDescLayout: VkDescriptorSetLayout;
    FEmbedBatchDescPool: VkDescriptorPool;
    FEmbedBatchDescSet: VkDescriptorSet;
    FTokenIdsGpu: TVdxGpuBuffer;
    FEmbedGpu: TVdxGpuBuffer;

    // Private helpers — same patterns as TVdxInference
    function  UploadNormWeight(const ATensorName: string;
      const ACount: UInt32): TVdxGpuBuffer;
    function  UploadWeightTensor(const ATensorName: string): TVdxGpuBuffer;
    procedure EmbedTokensBatch(const ATokenIds: TArray<Integer>;
      const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
    procedure RunLayerForwardBatch(const ALayer: Integer;
      const ANumTokens: UInt32);
    procedure FireEvent(const AEvent: TVdxEmbeddingsEvent);

    // Pooling + normalize helpers (CPU-side)
    function  MeanPool(const AHiddenMat: array of Single;
      const ANumTokens: Integer): TArray<Single>;
    procedure L2Normalize(var AVec: TArray<Single>);

    // Post-pooling projection helpers.
    // TryLoadDenseProjections — tries a short list of common tensor-name
    // conventions used by llama.cpp's sentence-transformers GGUF exports
    // (dense_2.weight / dense_3.weight etc.). Returns True if both
    // projections were found and loaded, False otherwise. Non-fatal — an
    // absent projection just means Embed() returns pooled hidden states.
    function  TryLoadDenseProjections(): Boolean;
    // ApplyLinear — CPU matmul for a single vector against a [Out x In]
    // row-major weight matrix. Output[o] = sum_i Weight[o, i] * In[i].
    procedure ApplyLinear(const AIn: array of Single;
      const AWeights: array of Single; const AInDim, AOutDim: Integer;
      out AOut: TArray<Single>);

  public
    constructor Create(); override;
    destructor Destroy(); override;

    function  LoadModel(const AGGUFPath: string): Boolean;
    procedure UnloadModel();
    function  IsLoaded(): Boolean;

    // The main entrypoint — produces a normalized embedding vector.
    // AIsQuery selects the EmbeddingGemma task prefix:
    //   True  → query-side retrieval ('task: search result | query: ')
    //   False → document-side storage ('title: none | text: ')
    // Returns a TArray<Single> of length FEmbeddingDim, unit-norm.
    function  Embed(const AText: string; const AIsQuery: Boolean):
      TArray<Single>;

    procedure SetEmbeddingsEventCallback(
      const ACallback: TVdxEmbeddingsEventCallback;
      const AUserData: Pointer);

    // Model info
    function  GetEmbeddingDim(): Integer;
    function  GetMaxSeqLen(): Integer;
    function  GetArchitecture(): string;

    // Utility — cosine similarity between two unit vectors.
    // For normalized vectors this is just the dot product.
    class function CosineSimilarity(const AVecA, AVecB: TArray<Single>):
      Single; static;
  end;

implementation

{ TVdxEmbeddings }

constructor TVdxEmbeddings.Create();
begin
  inherited Create();
  FErrors := TVdxErrors.Create();
  FReader := nil;
  FCompute := nil;
  FNorm := nil;
  FAttn := nil;
  FVindex := nil;
  FTokenizer := nil;
  FModelLoaded := False;
  FArchitecture := '';
  FEmbedPtr := nil;
  FEmbedScale := 0.0;
  FEmbeddingDim := 0;
  FProjectionsLoaded := False;
  FDense1In := 0;
  FDense1Out := 0;
  FDense2In := 0;
  FDense2Out := 0;
end;

destructor TVdxEmbeddings.Destroy();
begin
  if FModelLoaded then
    UnloadModel();
  FreeAndNil(FErrors);
  inherited Destroy();
end;

function TVdxEmbeddings.IsLoaded(): Boolean;
begin
  Result := FModelLoaded;
end;

function TVdxEmbeddings.GetEmbeddingDim(): Integer;
begin
  Result := FEmbeddingDim;
end;

function TVdxEmbeddings.GetMaxSeqLen(): Integer;
begin
  Result := Integer(FMaxSeqLen);
end;

function TVdxEmbeddings.GetArchitecture(): string;
begin
  Result := FArchitecture;
end;

procedure TVdxEmbeddings.SetEmbeddingsEventCallback(
  const ACallback: TVdxEmbeddingsEventCallback;
  const AUserData: Pointer);
begin
  FEventCallback.Callback := ACallback;
  FEventCallback.UserData := AUserData;
end;

procedure TVdxEmbeddings.FireEvent(const AEvent: TVdxEmbeddingsEvent);
begin
  if FEventCallback.IsAssigned() then
    FEventCallback.Callback(AEvent, FEventCallback.UserData);
end;

// ---------------------------------------------------------------------------
// UploadNormWeight — reads an F32 norm tensor by name, uploads to a GPU
// host-visible storage buffer. Same pattern as TVdxInference.
// ---------------------------------------------------------------------------
function TVdxEmbeddings.UploadNormWeight(const ATensorName: string;
  const ACount: UInt32): TVdxGpuBuffer;
var
  LPtr: Pointer;
  LData: array of Single;
begin
  LPtr := FReader.GetTensorDataPtr(ATensorName);
  SetLength(LData, ACount);
  Move(LPtr^, LData[0], ACount * SizeOf(Single));
  Result := FCompute.CreateGpuBuffer(
    UInt64(ACount) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  FCompute.UploadToBuffer(Result, @LData[0],
    UInt64(ACount) * SizeOf(Single));
end;

// ---------------------------------------------------------------------------
// UploadWeightTensor — reads any quantized tensor by name, stages it, then
// copies into a device-local storage buffer. Same pattern as TVdxInference.
// ---------------------------------------------------------------------------
function TVdxEmbeddings.UploadWeightTensor(
  const ATensorName: string): TVdxGpuBuffer;
var
  LInfo: TVdxGGUFTensorInfo;
  LPtr: Pointer;
  LSize: UInt64;
  LStaging: TVdxGpuBuffer;
begin
  LInfo := FReader.GetTensorInfo(ATensorName);
  LPtr := FReader.GetTensorDataPtr(ATensorName);
  LSize := VdxGGMLTensorBytes(LInfo.TensorType,
    LInfo.Dimensions[0], LInfo.Dimensions[1]);
  if LSize = 0 then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Unsupported tensor type for %s: %s',
      [ATensorName, VdxGGMLTypeName(LInfo.TensorType)]);
    FillChar(Result, SizeOf(Result), 0);
    Exit;
  end;
  LStaging := FCompute.CreateGpuBuffer(LSize,
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
  try
    FCompute.UploadToBuffer(LStaging, LPtr, LSize);
    Result := FCompute.CreateGpuBuffer(LSize,
      VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or
      VK_BUFFER_USAGE_TRANSFER_DST_BIT,
      VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    FCompute.CopyBuffer(LStaging, Result, LSize);
  finally
    FCompute.DestroyGpuBuffer(LStaging);
  end;
end;

// ---------------------------------------------------------------------------
// LoadModel — mirrors TVdxInference.LoadModel but validates for the
// 'gemma-embedding' architecture and skips KV cache / sampler / unembedding
// pipeline setup (none of which an embedding pass needs). Uses the same
// tensor naming and metadata key convention as gemma3 since EmbeddingGemma
// is built on the Gemma 3 backbone.
// ---------------------------------------------------------------------------
function TVdxEmbeddings.LoadModel(const AGGUFPath: string): Boolean;
var
  LLayer: Integer;
  LSpvData: TBytes;
  LQInfo: TVdxGGUFTensorInfo;
  LModelMax: UInt32;
begin
  Result := False;
  FErrors.Clear();
  FireEvent(eeLoadStart);

  if FModelLoaded then
  begin
    FErrors.Add(esFatal, 'LOAD',
      'Model already loaded — call UnloadModel() first');
    Exit;
  end;

  // --- Create subsystem objects ---
  FReader := TVdxGGUFReader.Create();
  FCompute := TVdxVulkanCompute.Create();
  FNorm := TVdxLayerNorm.Create();
  FAttn := TVdxAttention.Create();
  FVindex := TVdxFFNWeights.Create();
  FTokenizer := TVdxTokenizer.Create();

  FReader.SetStatusCallback(FStatusCallback.Callback,
    FStatusCallback.UserData);
  FCompute.SetStatusCallback(FStatusCallback.Callback,
    FStatusCallback.UserData);
  FNorm.SetStatusCallback(FStatusCallback.Callback,
    FStatusCallback.UserData);

  // --- Open GGUF ---
  if not FReader.Open(AGGUFPath) then
  begin
    FErrors.Add(esFatal, 'LOAD', 'Failed to open GGUF: %s', [AGGUFPath]);
    Exit;
  end;

  // --- Detect and validate architecture ---
  FArchitecture := TVdxChatTemplate.DetectArchitecture(FReader);
  Status('Architecture: %s', [FArchitecture]);

  if FArchitecture <> 'gemma-embedding' then
  begin
    FErrors.Add(esFatal, 'ARCH',
      'Expected gemma-embedding architecture, got "%s". ' +
      'TVdxEmbeddings currently supports EmbeddingGemma only.',
      [FArchitecture]);
    Exit;
  end;

  // --- Read model config (same key scheme as gemma3, different prefix) ---
  FNumLayers := FReader.GetMetadataUInt32(
    FArchitecture + '.block_count');
  FHiddenDim := FReader.GetMetadataUInt32(
    FArchitecture + '.embedding_length');
  FFFNWidth := FReader.GetMetadataUInt32(
    FArchitecture + '.feed_forward_length');
  FNumQHeads := FReader.GetMetadataUInt32(
    FArchitecture + '.attention.head_count');
  FNumKVHeads := FReader.GetMetadataUInt32(
    FArchitecture + '.attention.head_count_kv');

  LQInfo := FReader.GetTensorInfo('blk.0.attn_q.weight');
  FHeadDim := UInt32(LQInfo.Dimensions[1]) div FNumQHeads;

  if FReader.HasMetadata(FArchitecture + '.context_length') then
    LModelMax := FReader.GetMetadataUInt32(
      FArchitecture + '.context_length')
  else
    LModelMax := 2048;
  FMaxSeqLen := LModelMax;

  FWeightType := LQInfo.TensorType;
  FEmbedType := FReader.GetTensorInfo('token_embd.weight').TensorType;

  Status('Config: layers=%d hidden=%d ffn=%d heads=%d/%d head_dim=%d ctx=%d',
    [FNumLayers, FHiddenDim, FFFNWidth, FNumQHeads, FNumKVHeads,
     FHeadDim, FMaxSeqLen]);
  Status('Weight type: %s, embedding type: %s',
    [VdxGGMLTypeName(FWeightType), VdxGGMLTypeName(FEmbedType)]);

  // Output dim = hidden dim for EmbeddingGemma (no separate projection).
  // Caller can truncate via MRL afterward if needed (v2).
  FEmbeddingDim := Integer(FHiddenDim);

  // --- Init Vulkan + subsystems ---
  FCompute.Init();
  FNorm.Init(FCompute);
  // Reuse TVdxAttention even though we don't need KV cache for embeddings.
  // MaxSeqLen = context length since one forward pass processes the whole
  // input as a prefill batch with no decode step afterward.
  FAttn.Init(FCompute, FHiddenDim, FNumQHeads, FNumKVHeads,
    FHeadDim, FNumLayers, FMaxSeqLen, FFFNWidth);
  if not FVindex.BuildFromGGUF(FReader) then
  begin
    FErrors.Add(esFatal, 'LOAD', 'Failed to build FFN weight index from GGUF');
    Exit;
  end;

  // --- Load tokenizer ---
  if not FTokenizer.LoadFromGGUF(FReader) then
  begin
    FErrors.Add(esFatal, 'LOAD', 'Failed to load tokenizer from GGUF');
    Exit;
  end;
  FVocabSize := FTokenizer.GetVocabSize();
  Status('Tokenizer loaded: %d tokens, BOS=%d, EOS=%d',
    [FVocabSize, FTokenizer.GetBosId(), FTokenizer.GetEosId()]);

  // --- Upload weights ---
  Status('Uploading weights to GPU...');
  FVindex.UploadAll(FCompute);

  Status('  Attention weights (%d layers)...', [FNumLayers]);
  SetLength(FAttnWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    FAttn.UploadAttnWeights(FReader, LLayer, FAttnWeights[LLayer]);

  Status('  FFN up weights (%d layers)...', [FNumLayers]);
  SetLength(FUpWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    FUpWeights[LLayer] := UploadWeightTensor(
      Format('blk.%d.ffn_up.weight', [LLayer]));

  Status('  Norm weights (%d layers)...', [FNumLayers]);
  SetLength(FNormWeights, FNumLayers);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
  begin
    FNormWeights[LLayer].AttnNormGpu := UploadNormWeight(
      Format('blk.%d.attn_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].PostAttnNormGpu := UploadNormWeight(
      Format('blk.%d.post_attention_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].FFNNormGpu := UploadNormWeight(
      Format('blk.%d.ffn_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].PostFFNNormGpu := UploadNormWeight(
      Format('blk.%d.post_ffw_norm.weight', [LLayer]), FHiddenDim);
    FNormWeights[LLayer].QNormGpu := UploadNormWeight(
      Format('blk.%d.attn_q_norm.weight', [LLayer]), FHeadDim);
    FNormWeights[LLayer].KNormGpu := UploadNormWeight(
      Format('blk.%d.attn_k_norm.weight', [LLayer]), FHeadDim);
  end;

  FOutputNormGpu := UploadNormWeight('output_norm.weight', FHiddenDim);

  // Embedding table — needed on GPU for batched lookup
  FEmbedGpu := UploadWeightTensor('token_embd.weight');

  if FErrors.HasFatal() then
    Exit;

  // --- Batch matrix buffers (one row per token) ---
  Status('Allocating matrix buffers...');

  FResidualMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
    // Host-visible so we can map it directly for CPU-side mean pooling
    // at the end of Embed(). The GPU still accesses it as a storage buffer
    // during the forward pass — on NVIDIA this is slower than device-local
    // but fine for a single forward pass per call.
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  FWorkMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FQMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FNumQHeads * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FKMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FNumKVHeads * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FVMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FNumKVHeads * FHeadDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FAttnOutMatBuf := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FGateMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FFFNWidth * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FUpMatBuf := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FFFNWidth * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  FFFNOutMat := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  // --- Pipelines: gelu_mul, vec_add, batched embed lookup ---

  // gelu_mul — operates on gate/up matrices
  LSpvData := VdxLoadShader('GELU_MUL');
  FGeluMulShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FGeluMulDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FGeluMulBundle := FCompute.CreateComputePipelineWithPush(
    FGeluMulShader, 'main', FGeluMulDescLayout,
    SizeOf(TVdxEmbGeluMulPush));

  // vec_add — operates on residual + attn/ffn output matrices
  LSpvData := VdxLoadShader('VEC_ADD');
  FVecAddShader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  FVecAddDescLayout := FCompute.CreateStorageDescriptorSetLayout(2);
  FVecAddBundle := FCompute.CreateComputePipelineWithPush(
    FVecAddShader, 'main', FVecAddDescLayout,
    SizeOf(TVdxEmbVecAddPush));

  // Combined descriptor pool for the three per-batch elementwise sets
  FVecAddDescPool := FCompute.CreateDescriptorPoolForStorage(3, 6);
  FVecAddAttnDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualMat, FAttnOutMatBuf]);
  FVecAddFFNDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FVecAddDescPool, FVecAddDescLayout, [FResidualMat, FFFNOutMat]);
  FGeluMulDescPool := FCompute.CreateDescriptorPoolForStorage(1, 2);
  FGeluMulDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FGeluMulDescPool, FGeluMulDescLayout, [FGateMat, FUpMatBuf]);

  // Batched embed lookup — 3 quant variants, rebinding per call
  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_F16');
  FEmbedBatchF16Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_Q8');
  FEmbedBatchQ8Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));
  LSpvData := VdxLoadShader('EMBED_LOOKUP_BATCH_Q4_0');
  FEmbedBatchQ4Shader := FCompute.CreateShaderModule(
    @LSpvData[0], NativeUInt(Length(LSpvData)));

  FEmbedBatchDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);
  FEmbedBatchF16Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchF16Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbBatchPush));
  FEmbedBatchQ8Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchQ8Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbBatchPush));
  FEmbedBatchQ4Bundle := FCompute.CreateComputePipelineWithPush(
    FEmbedBatchQ4Shader, 'main', FEmbedBatchDescLayout,
    SizeOf(TVdxEmbBatchPush));

  FTokenIdsGpu := FCompute.CreateGpuBuffer(
    UInt64(FMaxSeqLen) * SizeOf(UInt32),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or
    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

  FEmbedBatchDescPool := FCompute.CreateDescriptorPoolForStorage(1, 3);
  FEmbedBatchDescSet := FCompute.AllocateDescriptorSetForBuffers(
    FEmbedBatchDescPool, FEmbedBatchDescLayout,
    [FEmbedGpu, FResidualMat, FTokenIdsGpu]);

  // Compute the embedding scale factor the same way TVdxInference does
  FEmbedScale := Sqrt(Single(FHiddenDim));
  FEmbedPtr := PByte(FReader.GetTensorDataPtr('token_embd.weight'));

  // Post-pooling dense projections — tries common tensor-name conventions.
  // On success FEmbeddingDim gets updated to match the final projection's
  // output dim (typically 768). On miss FEmbeddingDim stays at FHiddenDim
  // and Embed() returns the normalized pooled hidden states directly.
  Status('Looking for post-pooling dense projections...');
  FProjectionsLoaded := TryLoadDenseProjections();

  FModelLoaded := True;
  Result := True;
  FireEvent(eeLoadEnd);
  Status('Embedding model loaded successfully');
end;

// ---------------------------------------------------------------------------
// EmbedTokensBatch — uploads token IDs and dispatches the batched embed
// lookup shader. Mirrors TVdxInference.EmbedTokensBatch.
// ---------------------------------------------------------------------------
procedure TVdxEmbeddings.EmbedTokensBatch(const ATokenIds: TArray<Integer>;
  const ANumTokens: Integer; const AOutputBuf: TVdxGpuBuffer);
var
  LPush: TVdxEmbBatchPush;
  LIds: array of UInt32;
  LI: Integer;
begin
  SetLength(LIds, ANumTokens);
  for LI := 0 to ANumTokens - 1 do
    LIds[LI] := UInt32(ATokenIds[LI]);
  FCompute.UploadToBuffer(FTokenIdsGpu, @LIds[0],
    UInt64(ANumTokens) * SizeOf(UInt32));

  FCompute.UpdateDescriptorSetBuffers(FEmbedBatchDescSet,
    [FEmbedGpu, AOutputBuf, FTokenIdsGpu]);

  LPush.EmbedScale := FEmbedScale;
  LPush.NumTokens := UInt32(ANumTokens);

  if FEmbedType = gtQ4_0 then
  begin
    LPush.DimParam := FHiddenDim;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchQ4Bundle.Pipeline, FEmbedBatchQ4Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush), UInt32(ANumTokens));
  end
  else if FEmbedType = gtQ8_0 then
  begin
    LPush.DimParam := FHiddenDim;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchQ8Bundle.Pipeline, FEmbedBatchQ8Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush), UInt32(ANumTokens));
  end
  else
  begin
    LPush.DimParam := FHiddenDim div 2;
    FCompute.DispatchComputeWithPush(
      FEmbedBatchF16Bundle.Pipeline, FEmbedBatchF16Bundle.PipelineLayout,
      FEmbedBatchDescSet, @LPush, SizeOf(LPush),
      (FHiddenDim div 2 + 255) div 256, UInt32(ANumTokens));
  end;
  FCompute.BatchBarrier();
end;

// ---------------------------------------------------------------------------
// RunLayerForwardBatch — same body as TVdxInference.RunLayerForwardBatch
// except AStartPos is always 0 (embedding models don't track position
// across calls) and theta dispatch follows the same sliding/full layer
// pattern as Gemma 3. NOTE: if EmbeddingGemma uses a different sliding
// pattern this may need adjustment — flag for verification after first run.
// ---------------------------------------------------------------------------
procedure TVdxEmbeddings.RunLayerForwardBatch(const ALayer: Integer;
  const ANumTokens: UInt32);
var
  LTheta: Single;
  LGeluPush: TVdxEmbGeluMulPush;
  LVecAddPush: TVdxEmbVecAddPush;
begin
  // === Attention branch ===
  FNorm.ApplyCopyBatch(FResidualMat, FNormWeights[ALayer].AttnNormGpu,
    FWorkMat, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  if ALayer mod 6 = 5 then
    LTheta := 1000000.0
  else
    LTheta := 10000.0;

  FAttn.ForwardBatch(FWorkMat, FAttnWeights[ALayer],
    FNormWeights[ALayer].QNormGpu, FNormWeights[ALayer].KNormGpu,
    ALayer, ANumTokens, 0, LTheta,
    FQMat, FKMat, FVMat, FAttnOutMatBuf, True);

  FNorm.ApplyBatch(FAttnOutMatBuf,
    FNormWeights[ALayer].PostAttnNormGpu, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  LVecAddPush.Count := ANumTokens * FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddAttnDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (ANumTokens * FHiddenDim + 255) div 256);
  FCompute.BatchBarrier();

  // === FFN branch ===
  FNorm.ApplyCopyBatch(FResidualMat, FNormWeights[ALayer].FFNNormGpu,
    FWorkMat, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  FAttn.BatchMatMul(FVindex.GetLayer(ALayer).GateGpuBuffer,
    FWorkMat, FGateMat, FHiddenDim, FFFNWidth, ANumTokens, FWeightType);
  FAttn.BatchMatMul(FUpWeights[ALayer],
    FWorkMat, FUpMatBuf, FHiddenDim, FFFNWidth, ANumTokens, FWeightType);
  FCompute.BatchBarrier();

  LGeluPush.Count := ANumTokens * FFFNWidth;
  FCompute.DispatchComputeWithPush(
    FGeluMulBundle.Pipeline, FGeluMulBundle.PipelineLayout,
    FGeluMulDescSet, @LGeluPush, SizeOf(LGeluPush),
    (ANumTokens * FFFNWidth + 255) div 256);
  FCompute.BatchBarrier();

  FAttn.BatchMatMul(FVindex.GetLayer(ALayer).DownGpuBuffer,
    FGateMat, FFFNOutMat, FFFNWidth, FHiddenDim, ANumTokens, FWeightType);
  FCompute.BatchBarrier();

  FNorm.ApplyBatch(FFFNOutMat,
    FNormWeights[ALayer].PostFFNNormGpu, FHiddenDim, ANumTokens);
  FCompute.BatchBarrier();

  LVecAddPush.Count := ANumTokens * FHiddenDim;
  FCompute.DispatchComputeWithPush(
    FVecAddBundle.Pipeline, FVecAddBundle.PipelineLayout,
    FVecAddFFNDescSet, @LVecAddPush, SizeOf(LVecAddPush),
    (ANumTokens * FHiddenDim + 255) div 256);
  FCompute.BatchBarrier();
end;

// ---------------------------------------------------------------------------
// MeanPool — averages hidden states across all tokens to produce a single
// sentence-level vector. Works on a flat [NumTokens * FHiddenDim] buffer.
// ---------------------------------------------------------------------------
function TVdxEmbeddings.MeanPool(const AHiddenMat: array of Single;
  const ANumTokens: Integer): TArray<Single>;
var
  LI: Integer;
  LT: Integer;
  LSum: Single;
  LInv: Single;
begin
  SetLength(Result, FHiddenDim);
  if ANumTokens <= 0 then
    Exit;

  LInv := 1.0 / Single(ANumTokens);
  // For each output dim, average across all token positions.
  for LI := 0 to Integer(FHiddenDim) - 1 do
  begin
    LSum := 0.0;
    for LT := 0 to ANumTokens - 1 do
      LSum := LSum + AHiddenMat[LT * Integer(FHiddenDim) + LI];
    Result[LI] := LSum * LInv;
  end;
end;

// ---------------------------------------------------------------------------
// L2Normalize — scales the vector to unit length. Cosine similarity between
// two unit vectors is just the dot product, so this is done here once.
// ---------------------------------------------------------------------------
procedure TVdxEmbeddings.L2Normalize(var AVec: TArray<Single>);
var
  LI: Integer;
  LSumSq: Single;
  LInvNorm: Single;
begin
  LSumSq := 0.0;
  for LI := 0 to High(AVec) do
    LSumSq := LSumSq + AVec[LI] * AVec[LI];

  // Guard against zero vector — leave as-is rather than divide by zero.
  if LSumSq <= 1E-20 then
    Exit;

  LInvNorm := 1.0 / Sqrt(LSumSq);
  for LI := 0 to High(AVec) do
    AVec[LI] := AVec[LI] * LInvNorm;
end;

// ---------------------------------------------------------------------------
// CosineSimilarity — dot product for pre-normalized unit vectors.
// Callers must have embedded both vectors via Embed() (which L2-normalizes).
// ---------------------------------------------------------------------------
class function TVdxEmbeddings.CosineSimilarity(
  const AVecA, AVecB: TArray<Single>): Single;
var
  LI: Integer;
  LDot: Single;
begin
  Result := 0.0;
  if (Length(AVecA) = 0) or (Length(AVecA) <> Length(AVecB)) then
    Exit;

  LDot := 0.0;
  for LI := 0 to High(AVecA) do
    LDot := LDot + AVecA[LI] * AVecB[LI];
  Result := LDot;
end;

// ---------------------------------------------------------------------------
// ApplyLinear — CPU matmul y = W * x, where W is [AOutDim x AInDim]
// row-major. Output[o] = sum over i of W[o, i] * In[i]. Bias-free, as
// the EmbeddingGemma dense projections have no bias.
// ---------------------------------------------------------------------------
procedure TVdxEmbeddings.ApplyLinear(const AIn: array of Single;
  const AWeights: array of Single; const AInDim, AOutDim: Integer;
  out AOut: TArray<Single>);
var
  LO: Integer;
  LI: Integer;
  LAcc: Single;
  LRowBase: Integer;
begin
  SetLength(AOut, AOutDim);
  for LO := 0 to AOutDim - 1 do
  begin
    LRowBase := LO * AInDim;
    LAcc := 0.0;
    for LI := 0 to AInDim - 1 do
      LAcc := LAcc + AWeights[LRowBase + LI] * AIn[LI];
    AOut[LO] := LAcc;
  end;
end;

// ---------------------------------------------------------------------------
// TryLoadDenseProjections — attempt to load the two post-pooling dense
// projections from the GGUF. The sentence-transformers module structure
// maps to different tensor-name conventions depending on which llama.cpp
// converter was used. We try the common ones and stop at the first
// pair that exists. Returns False if no pair is found — not fatal, just
// means the final output is the normalized pooled hidden state.
// ---------------------------------------------------------------------------
function TVdxEmbeddings.TryLoadDenseProjections(): Boolean;
const
  // Candidate tensor-name pairs ordered by likelihood for llama.cpp's
  // EmbeddingGemma GGUF conversion. First match wins.
  CNamePairs: array[0..3, 0..1] of string = (
    ('dense_2.weight',        'dense_3.weight'),
    ('2_Dense.weight',        '3_Dense.weight'),
    ('pooler.dense_1.weight', 'pooler.dense_2.weight'),
    ('dense.0.weight',        'dense.1.weight')
  );
var
  LPairIdx: Integer;
  LName1: string;
  LName2: string;
  LInfo1: TVdxGGUFTensorInfo;
  LInfo2: TVdxGGUFTensorInfo;
  LPtr1: Pointer;
  LPtr2: Pointer;
  LCount1: Integer;
  LCount2: Integer;
begin
  Result := False;

  for LPairIdx := 0 to High(CNamePairs) do
  begin
    LName1 := CNamePairs[LPairIdx, 0];
    LName2 := CNamePairs[LPairIdx, 1];
    if FReader.HasTensor(LName1) and FReader.HasTensor(LName2) then
    begin
      LInfo1 := FReader.GetTensorInfo(LName1);
      LInfo2 := FReader.GetTensorInfo(LName2);

      // Only F32 is supported here — the projections are small and the
      // code path is non-hot. If the GGUF has them in F16 or quantized
      // form we'd need conversion; deferring that until we see it.
      if (LInfo1.TensorType <> gtF32) or (LInfo2.TensorType <> gtF32) then
      begin
        Status('  Dense projections found (%s, %s) but not F32 (%s, %s) '
             + '— skipping for now',
          [LName1, LName2,
           VdxGGMLTypeName(LInfo1.TensorType),
           VdxGGMLTypeName(LInfo2.TensorType)]);
        Exit;
      end;

      // GGUF stores 2D tensors as [in_dim, out_dim] (llama.cpp convention)
      // so our row-major [out_dim, in_dim] view uses Dimensions[0] as
      // in_dim and Dimensions[1] as out_dim.
      FDense1In  := Integer(LInfo1.Dimensions[0]);
      FDense1Out := Integer(LInfo1.Dimensions[1]);
      FDense2In  := Integer(LInfo2.Dimensions[0]);
      FDense2Out := Integer(LInfo2.Dimensions[1]);

      LCount1 := FDense1In * FDense1Out;
      LCount2 := FDense2In * FDense2Out;

      LPtr1 := FReader.GetTensorDataPtr(LName1);
      LPtr2 := FReader.GetTensorDataPtr(LName2);

      SetLength(FDense1Weights, LCount1);
      SetLength(FDense2Weights, LCount2);
      Move(LPtr1^, FDense1Weights[0], LCount1 * SizeOf(Single));
      Move(LPtr2^, FDense2Weights[0], LCount2 * SizeOf(Single));

      Status('  Dense projections loaded: %s [%d->%d], %s [%d->%d]',
        [LName1, FDense1In, FDense1Out,
         LName2, FDense2In, FDense2Out]);
      FEmbeddingDim := FDense2Out;
      Result := True;
      Exit;
    end;
  end;

  Status('  Dense projections NOT found — tried %d name pair(s). ' +
    'Embed() will return normalized pooled hidden states without ' +
    'the final 768->3072->768 projection.', [Length(CNamePairs)]);
end;

// ---------------------------------------------------------------------------
// Embed — the main public entrypoint. Applies the architecture-appropriate
// task prefix, tokenizes (with BOS), runs one batched forward pass through
// all layers, applies the final output norm, downloads the token-wise
// hidden states, mean-pools across tokens, L2-normalizes, and returns
// the unit vector. Length of the returned array == FEmbeddingDim.
// ---------------------------------------------------------------------------
function TVdxEmbeddings.Embed(const AText: string;
  const AIsQuery: Boolean): TArray<Single>;
var
  LPrefixed: string;
  LTokens: TArray<Integer>;
  LWithBos: TArray<Integer>;
  LNumTokens: Integer;
  LLayer: Integer;
  LHiddenMat: array of Single;
  LMatBytes: UInt64;
  LDense1Out: TArray<Single>;
begin
  SetLength(Result, 0);

  if not FModelLoaded then
  begin
    FErrors.Add(esError, 'EMBED', 'Model not loaded');
    Exit;
  end;

  FireEvent(eeEmbedStart);

  // 1. Apply EmbeddingGemma task prefix via the shared template dispatch.
  LPrefixed := TVdxChatTemplate.FormatEmbedding(
    FArchitecture, AText, AIsQuery);

  // 2. Tokenize. Encode() does not add BOS — we prepend it ourselves so
  //    the model sees the start-of-sequence marker the sentence-transformers
  //    pipeline would have added.
  LTokens := FTokenizer.Encode(LPrefixed, False);
  SetLength(LWithBos, Length(LTokens) + 1);
  LWithBos[0] := FTokenizer.GetBosId();
  if Length(LTokens) > 0 then
    Move(LTokens[0], LWithBos[1], Length(LTokens) * SizeOf(Integer));

  // 3. Clamp to max context so we don't overrun the batch matrix buffers.
  LNumTokens := Length(LWithBos);
  if LNumTokens > Integer(FMaxSeqLen) then
  begin
    LNumTokens := Integer(FMaxSeqLen);
    SetLength(LWithBos, LNumTokens);
  end;

  Status('Embedding %d tokens (query=%s)',
    [LNumTokens, BoolToStr(AIsQuery, True)]);

  // 4. Run the forward pass in one GPU batch.
  FCompute.BeginBatch();
  EmbedTokensBatch(LWithBos, LNumTokens, FResidualMat);
  for LLayer := 0 to Integer(FNumLayers) - 1 do
    RunLayerForwardBatch(LLayer, UInt32(LNumTokens));

  // 5. Apply final output norm in place on the residual matrix.
  FNorm.ApplyBatch(FResidualMat, FOutputNormGpu,
    FHiddenDim, UInt32(LNumTokens));
  FCompute.BatchBarrier();
  FCompute.EndBatch();

  // 6. Download the token-wise hidden states to CPU for pooling.
  LMatBytes := UInt64(LNumTokens) * FHiddenDim * SizeOf(Single);
  SetLength(LHiddenMat, LNumTokens * Integer(FHiddenDim));
  FCompute.DownloadFromBuffer(FResidualMat, @LHiddenMat[0], LMatBytes);

  // 7. Mean-pool across tokens → single FHiddenDim vector.
  Result := MeanPool(LHiddenMat, LNumTokens);

  // 7a. Post-pooling dense projections (768 -> 3072 -> 768). Only runs if
  //     the GGUF included the sentence-transformers dense modules. If not,
  //     we proceed straight to L2 normalize on the pooled hidden state.
  if FProjectionsLoaded then
  begin
    ApplyLinear(Result, FDense1Weights, FDense1In, FDense1Out, LDense1Out);
    ApplyLinear(LDense1Out, FDense2Weights, FDense2In, FDense2Out, Result);
  end;

  // 8. L2 normalize so cosine similarity is a straight dot product.
  L2Normalize(Result);

  FireEvent(eeEmbedEnd);
end;

// ---------------------------------------------------------------------------
// UnloadModel — releases all GPU resources and subsystem objects.
// Mirrors TVdxInference.UnloadModel, minus the KV cache and unembedding
// resources we never created.
// ---------------------------------------------------------------------------
procedure TVdxEmbeddings.UnloadModel();
var
  LI: Integer;
begin
  if not FModelLoaded then
    Exit;
  FireEvent(eeUnloadStart);

  // Matrix buffers
  if Assigned(FCompute) then
  begin
    FCompute.DestroyGpuBuffer(FResidualMat);
    FCompute.DestroyGpuBuffer(FWorkMat);
    FCompute.DestroyGpuBuffer(FQMat);
    FCompute.DestroyGpuBuffer(FKMat);
    FCompute.DestroyGpuBuffer(FVMat);
    FCompute.DestroyGpuBuffer(FAttnOutMatBuf);
    FCompute.DestroyGpuBuffer(FGateMat);
    FCompute.DestroyGpuBuffer(FUpMatBuf);
    FCompute.DestroyGpuBuffer(FFFNOutMat);
    FCompute.DestroyGpuBuffer(FTokenIdsGpu);
    FCompute.DestroyGpuBuffer(FEmbedGpu);

    // Norm weights
    for LI := 0 to High(FNormWeights) do
    begin
      FCompute.DestroyGpuBuffer(FNormWeights[LI].AttnNormGpu);
      FCompute.DestroyGpuBuffer(FNormWeights[LI].PostAttnNormGpu);
      FCompute.DestroyGpuBuffer(FNormWeights[LI].FFNNormGpu);
      FCompute.DestroyGpuBuffer(FNormWeights[LI].PostFFNNormGpu);
      FCompute.DestroyGpuBuffer(FNormWeights[LI].QNormGpu);
      FCompute.DestroyGpuBuffer(FNormWeights[LI].KNormGpu);
    end;
    SetLength(FNormWeights, 0);

    // FFN up weights
    for LI := 0 to High(FUpWeights) do
      FCompute.DestroyGpuBuffer(FUpWeights[LI]);
    SetLength(FUpWeights, 0);

    FCompute.DestroyGpuBuffer(FOutputNormGpu);

    // Attention weights — TVdxAttention-owned tensors stored in per-layer
    // records, freed here alongside the rest.
    for LI := 0 to High(FAttnWeights) do
    begin
      FCompute.DestroyGpuBuffer(FAttnWeights[LI].QWeightGpu);
      FCompute.DestroyGpuBuffer(FAttnWeights[LI].KWeightGpu);
      FCompute.DestroyGpuBuffer(FAttnWeights[LI].VWeightGpu);
      FCompute.DestroyGpuBuffer(FAttnWeights[LI].OWeightGpu);
    end;
    SetLength(FAttnWeights, 0);

    // Pipelines + descriptor pools
    FCompute.DestroyComputePipelineBundle(FGeluMulBundle);
    FCompute.DestroyShaderModuleHandle(FGeluMulShader);
    FCompute.DestroyDescriptorPoolHandle(FGeluMulDescPool);
    FCompute.DestroyDescriptorSetLayoutHandle(FGeluMulDescLayout);

    FCompute.DestroyComputePipelineBundle(FVecAddBundle);
    FCompute.DestroyShaderModuleHandle(FVecAddShader);
    FCompute.DestroyDescriptorPoolHandle(FVecAddDescPool);
    FCompute.DestroyDescriptorSetLayoutHandle(FVecAddDescLayout);

    FCompute.DestroyComputePipelineBundle(FEmbedBatchF16Bundle);
    FCompute.DestroyComputePipelineBundle(FEmbedBatchQ8Bundle);
    FCompute.DestroyComputePipelineBundle(FEmbedBatchQ4Bundle);
    FCompute.DestroyShaderModuleHandle(FEmbedBatchF16Shader);
    FCompute.DestroyShaderModuleHandle(FEmbedBatchQ8Shader);
    FCompute.DestroyShaderModuleHandle(FEmbedBatchQ4Shader);
    FCompute.DestroyDescriptorPoolHandle(FEmbedBatchDescPool);
    FCompute.DestroyDescriptorSetLayoutHandle(FEmbedBatchDescLayout);
  end;

  // Subsystems
  FreeAndNil(FTokenizer);
  FreeAndNil(FVindex);
  FreeAndNil(FAttn);
  FreeAndNil(FNorm);
  FreeAndNil(FCompute);
  FreeAndNil(FReader);

  FModelLoaded := False;
  FArchitecture := '';
  FEmbedPtr := nil;
  FEmbeddingDim := 0;
  SetLength(FDense1Weights, 0);
  SetLength(FDense2Weights, 0);
  FProjectionsLoaded := False;
  FDense1In := 0;
  FDense1Out := 0;
  FDense2In := 0;
  FDense2Out := 0;
  FireEvent(eeUnloadEnd);
end;

end.
