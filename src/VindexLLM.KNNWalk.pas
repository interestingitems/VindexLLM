{===============================================================================
  VindexLLM™ - Graph-Walk LLM Inference Engine

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.KNNWalk;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.VirtualBuffer,
  VindexLLM.VulkanCompute,
  VindexLLM.Vindex;

type

  // Push constant layouts matching the GLSL shaders
  TVdxGateScanPush = record
    HiddenDimHalf: UInt32;
  end;

  TVdxAccumPush = record
    HiddenDimHalf: UInt32;
    TopK: UInt32;
  end;

  { TVdxKNNWalk }
  TVdxKNNWalk = class(TVdxStatusObject)
  private
    FCompute: TVdxVulkanCompute;
    FTopK: Integer;
    FHiddenDim: UInt64;
    FFFNWidth: UInt64;

    // GPU buffers (persistent)
    FResidualGpu: TVdxGpuBuffer;      // F32 x HiddenDim, host-visible coherent
    FScoresGpu: TVdxGpuBuffer;        // F32 x FFNWidth, host-visible coherent
    FDownPackGpu: TVdxGpuBuffer;      // F16 x K x HiddenDim, device-local
    FIndicesGpu: TVdxGpuBuffer;       // UInt32 x K, host-visible coherent
    FActivationsGpu: TVdxGpuBuffer;   // F32 x K, host-visible coherent

    // Persistent mapped pointers (set once at init)
    FResidualMapped: Pointer;
    FScoresMapped: Pointer;
    FIndicesMapped: Pointer;
    FActivationsMapped: Pointer;

    // CPU workspace
    FScores: TVdxVirtualBuffer<Single>;
    FDownPack: TVdxVirtualBuffer<Byte>;

    // Shader modules
    FGateScanShader: VkShaderModule;
    FAccumShader: VkShaderModule;

    // Pipelines
    FGateScanBundle: TVdxComputePipelineBundle;
    FGateScanDescLayout: VkDescriptorSetLayout;

    FAccumBundle: TVdxComputePipelineBundle;
    FAccumDescLayout: VkDescriptorSetLayout;

    // Internal
    procedure FindTopK(const ACount: Integer; const AK: Integer;
      out ATopIndices: TArray<UInt32>; out ATopValues: TArray<Single>);
    procedure PackDownVectors(const ALayer: TVdxFFNLayerView;
      const AIndices: TArray<UInt32>; const ACount: Integer);

  public
    constructor Create(); override;
    destructor Destroy(); override;

    procedure Init(const ACompute: TVdxVulkanCompute;
      const AHiddenDim: UInt64; const AFFNWidth: UInt64;
      const ATopK: Integer = 128);
    procedure Cleanup();

    // Set/get the residual vector (F32, HiddenDim elements)
    procedure SetResidual(const AData: Pointer);
    procedure GetResidual(const AData: Pointer);

    // Execute one FFN walk step for a layer
    procedure WalkLayer(const ALayer: TVdxFFNLayerView);

    // Accessors
    function GetTopK(): Integer;
    function GetHiddenDim(): UInt64;
    function GetFFNWidth(): UInt64;
  end;

implementation

uses
  System.IOUtils;

// ============================================================================
//  TVdxKNNWalk — Construction / Destruction
// ============================================================================

constructor TVdxKNNWalk.Create();
begin
  inherited;

  FCompute := nil;
  FTopK := 0;
  FHiddenDim := 0;
  FFFNWidth := 0;
  FResidualMapped := nil;
  FScoresMapped := nil;
  FIndicesMapped := nil;
  FActivationsMapped := nil;
  FScores := nil;
  FDownPack := nil;
  FGateScanShader := VK_NULL_HANDLE;
  FAccumShader := VK_NULL_HANDLE;
  FillChar(FResidualGpu, SizeOf(FResidualGpu), 0);
  FillChar(FScoresGpu, SizeOf(FScoresGpu), 0);
  FillChar(FDownPackGpu, SizeOf(FDownPackGpu), 0);
  FillChar(FIndicesGpu, SizeOf(FIndicesGpu), 0);
  FillChar(FActivationsGpu, SizeOf(FActivationsGpu), 0);
  FillChar(FGateScanBundle, SizeOf(FGateScanBundle), 0);
  FillChar(FAccumBundle, SizeOf(FAccumBundle), 0);
  FGateScanDescLayout := VK_NULL_HANDLE;
  FAccumDescLayout := VK_NULL_HANDLE;
end;

destructor TVdxKNNWalk.Destroy();
begin
  Cleanup();
  inherited;
end;

// ============================================================================
//  TVdxKNNWalk — Init / Cleanup
// ============================================================================

procedure TVdxKNNWalk.Init(const ACompute: TVdxVulkanCompute;
  const AHiddenDim: UInt64; const AFFNWidth: UInt64;
  const ATopK: Integer);
var
  LSpvPath: string;
  LSpvBytes: TBytes;
  LDownPackSize: UInt64;
begin
  FCompute := ACompute;
  FTopK := ATopK;
  FHiddenDim := AHiddenDim;
  FFFNWidth := AFFNWidth;

  Status('KNNWalk: Init (hidden=%d, ffn=%d, topK=%d)', [FHiddenDim, FFFNWidth, FTopK]);

  // GPU buffers — host-visible coherent (persistently mapped)
  FResidualGpu := FCompute.CreateGpuBuffer(
    FHiddenDim * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  FResidualMapped := FCompute.MapBufferPersistent(FResidualGpu);

  FScoresGpu := FCompute.CreateGpuBuffer(
    FFFNWidth * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  FScoresMapped := FCompute.MapBufferPersistent(FScoresGpu);

  FIndicesGpu := FCompute.CreateGpuBuffer(
    UInt64(FTopK) * SizeOf(UInt32),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  FIndicesMapped := FCompute.MapBufferPersistent(FIndicesGpu);

  FActivationsGpu := FCompute.CreateGpuBuffer(
    UInt64(FTopK) * SizeOf(Single),
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  );
  FActivationsMapped := FCompute.MapBufferPersistent(FActivationsGpu);

  // GPU buffer — device-local for packed down vectors (uploaded via staging each step)
  LDownPackSize := UInt64(FTopK) * FHiddenDim * 2;  // F16 = 2 bytes per element
  FDownPackGpu := FCompute.CreateGpuBuffer(
    LDownPackSize,
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT or VK_BUFFER_USAGE_TRANSFER_DST_BIT,
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
  );

  // CPU workspace
  FScores := TVdxVirtualBuffer<Single>.Create(FFFNWidth);
  FDownPack := TVdxVirtualBuffer<Byte>.Create(UInt64(FTopK) * FHiddenDim * 2);

  // Load shaders from .spv files
  LSpvPath := TPath.Combine(
    TPath.GetDirectoryName(ParamStr(0)),
    '..\shaders\gate_scan.spv'
  );
  LSpvPath := TPath.GetFullPath(LSpvPath);
  TVdxUtils.FailIf(not TFile.Exists(LSpvPath), 'gate_scan.spv not found: %s', [LSpvPath]);
  LSpvBytes := TFile.ReadAllBytes(LSpvPath);
  FGateScanShader := FCompute.CreateShaderModule(@LSpvBytes[0], NativeUInt(Length(LSpvBytes)));

  LSpvPath := TPath.Combine(
    TPath.GetDirectoryName(ParamStr(0)),
    '..\shaders\accumulate.spv'
  );
  LSpvPath := TPath.GetFullPath(LSpvPath);
  TVdxUtils.FailIf(not TFile.Exists(LSpvPath), 'accumulate.spv not found: %s', [LSpvPath]);
  LSpvBytes := TFile.ReadAllBytes(LSpvPath);
  FAccumShader := FCompute.CreateShaderModule(@LSpvBytes[0], NativeUInt(Length(LSpvBytes)));

  // Descriptor layouts
  // Gate scan: binding 0=gate, 1=residual, 2=scores
  FGateScanDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);

  // Accumulate: binding 0=down_pack, 1=activations, 2=residual
  FAccumDescLayout := FCompute.CreateStorageDescriptorSetLayout(3);

  // Pipelines with push constants
  FGateScanBundle := FCompute.CreateComputePipelineWithPush(
    FGateScanShader, 'main', FGateScanDescLayout, SizeOf(TVdxGateScanPush)
  );

  FAccumBundle := FCompute.CreateComputePipelineWithPush(
    FAccumShader, 'main', FAccumDescLayout, SizeOf(TVdxAccumPush)
  );

  Status('KNNWalk: Ready');
end;

procedure TVdxKNNWalk.Cleanup();
begin
  if FCompute = nil then
    Exit;

  // Free CPU workspace
  FreeAndNil(FScores);
  FreeAndNil(FDownPack);

  // Destroy pipelines
  FCompute.DestroyComputePipelineBundle(FGateScanBundle);
  FCompute.DestroyComputePipelineBundle(FAccumBundle);

  // Destroy descriptor layouts
  FCompute.DestroyDescriptorSetLayoutHandle(FGateScanDescLayout);
  FCompute.DestroyDescriptorSetLayoutHandle(FAccumDescLayout);
  FGateScanDescLayout := VK_NULL_HANDLE;
  FAccumDescLayout := VK_NULL_HANDLE;

  // Destroy shaders
  FCompute.DestroyShaderModuleHandle(FGateScanShader);
  FCompute.DestroyShaderModuleHandle(FAccumShader);
  FGateScanShader := VK_NULL_HANDLE;
  FAccumShader := VK_NULL_HANDLE;

  // Unmap and destroy GPU buffers
  if FResidualMapped <> nil then
    FCompute.UnmapBuffer(FResidualGpu);
  FCompute.DestroyGpuBuffer(FResidualGpu);
  FResidualMapped := nil;

  if FScoresMapped <> nil then
    FCompute.UnmapBuffer(FScoresGpu);
  FCompute.DestroyGpuBuffer(FScoresGpu);
  FScoresMapped := nil;

  if FIndicesMapped <> nil then
    FCompute.UnmapBuffer(FIndicesGpu);
  FCompute.DestroyGpuBuffer(FIndicesGpu);
  FIndicesMapped := nil;

  if FActivationsMapped <> nil then
    FCompute.UnmapBuffer(FActivationsGpu);
  FCompute.DestroyGpuBuffer(FActivationsGpu);
  FActivationsMapped := nil;

  FCompute.DestroyGpuBuffer(FDownPackGpu);

  FCompute := nil;
end;

// ============================================================================
//  TVdxKNNWalk — Residual Access
// ============================================================================

procedure TVdxKNNWalk.SetResidual(const AData: Pointer);
begin
  Move(AData^, FResidualMapped^, FHiddenDim * SizeOf(Single));
end;

procedure TVdxKNNWalk.GetResidual(const AData: Pointer);
begin
  Move(FResidualMapped^, AData^, FHiddenDim * SizeOf(Single));
end;

// ============================================================================
//  TVdxKNNWalk — Internal: CPU Top-K Selection
// ============================================================================

procedure TVdxKNNWalk.FindTopK(const ACount: Integer; const AK: Integer;
  out ATopIndices: TArray<UInt32>; out ATopValues: TArray<Single>);
var
  LI: Integer;
  LJ: Integer;
  LMinIdx: Integer;
  LMinVal: Single;
  LScore: Single;
begin
  // Simple selection: maintain a min-heap of K largest scores.
  // For 10240 elements and K=128 this is fast enough (~microseconds).
  SetLength(ATopIndices, AK);
  SetLength(ATopValues, AK);

  // Fill initial K slots
  for LI := 0 to AK - 1 do
  begin
    ATopIndices[LI] := UInt32(LI);
    ATopValues[LI] := FScores[LI];
  end;

  // Find current minimum in the top-K set
  LMinIdx := 0;
  LMinVal := ATopValues[0];
  for LI := 1 to AK - 1 do
  begin
    if ATopValues[LI] < LMinVal then
    begin
      LMinVal := ATopValues[LI];
      LMinIdx := LI;
    end;
  end;

  // Scan remaining elements, replace minimum when we find larger
  for LI := AK to ACount - 1 do
  begin
    LScore := FScores[LI];
    if LScore > LMinVal then
    begin
      ATopIndices[LMinIdx] := UInt32(LI);
      ATopValues[LMinIdx] := LScore;

      // Find new minimum
      LMinIdx := 0;
      LMinVal := ATopValues[0];
      for LJ := 1 to AK - 1 do
      begin
        if ATopValues[LJ] < LMinVal then
        begin
          LMinVal := ATopValues[LJ];
          LMinIdx := LJ;
        end;
      end;
    end;
  end;
end;

// ============================================================================
//  TVdxKNNWalk — Internal: Pack Down Vectors from mmap'd GGUF
// ============================================================================

procedure TVdxKNNWalk.PackDownVectors(const ALayer: TVdxFFNLayerView;
  const AIndices: TArray<UInt32>; const ACount: Integer);
var
  LI: Integer;
  LFeatureIdx: UInt64;
  LSrcOffset: UInt64;
  LDstOffset: UInt64;
  LVectorBytes: UInt64;
begin
  // Each down vector for feature k starts at k * HiddenDim * 2 bytes (F16)
  // in the mmap'd GGUF. We pack K winners contiguously into FDownPack.
  LVectorBytes := FHiddenDim * 2;  // F16 = 2 bytes per element

  for LI := 0 to ACount - 1 do
  begin
    LFeatureIdx := AIndices[LI];
    LSrcOffset := LFeatureIdx * LVectorBytes;
    LDstOffset := UInt64(LI) * LVectorBytes;

    // Copy from mmap'd GGUF down tensor into packed CPU buffer
    Move(
      Pointer(UIntPtr(ALayer.DownPtr) + UIntPtr(LSrcOffset))^,
      Pointer(UIntPtr(FDownPack.Memory) + UIntPtr(LDstOffset))^,
      LVectorBytes
    );
  end;
end;

// ============================================================================
//  TVdxKNNWalk — WalkLayer: Gate Scan → Top-K → Accumulate
// ============================================================================

procedure TVdxKNNWalk.WalkLayer(const ALayer: TVdxFFNLayerView);
var
  LGateScanPush: TVdxGateScanPush;
  LAccumPush: TVdxAccumPush;
  LDescPool: VkDescriptorPool;
  LGateScanDescSet: VkDescriptorSet;
  LAccumDescSet: VkDescriptorSet;
  LTopIndices: TArray<UInt32>;
  LTopValues: TArray<Single>;
  LStaging: TVdxGpuBuffer;
  LDownPackSize: UInt64;
begin
  // Create descriptor pool for this layer (2 sets, 6 total storage buffers)
  LDescPool := FCompute.CreateDescriptorPoolForStorage(2, 6);
  try
    // === Step 1: Gate Scan (GPU) ===
    // Bind: gate buffer (device-local from Vindex) + residual + scores
    LGateScanDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FGateScanDescLayout,
      [ALayer.GateGpuBuffer, FResidualGpu, FScoresGpu]
    );

    LGateScanPush.HiddenDimHalf := UInt32(FHiddenDim div 2);

    FCompute.DispatchComputeWithPush(
      FGateScanBundle.Pipeline,
      FGateScanBundle.PipelineLayout,
      LGateScanDescSet,
      @LGateScanPush,
      SizeOf(LGateScanPush),
      UInt32((FFFNWidth + 255) div 256)  // ceil(10240/256) = 40 workgroups
    );

    // === Step 2: Download Scores → CPU Top-K ===
    // Scores buffer is host-visible coherent + fence completed, just read mapped ptr
    FScores.CopyFrom(FScoresMapped, FFFNWidth * SizeOf(Single));

    FindTopK(Integer(FFFNWidth), FTopK, LTopIndices, LTopValues);

    // === Step 3: Pack Down Vectors + Upload to GPU ===
    PackDownVectors(ALayer, LTopIndices, FTopK);

    LDownPackSize := UInt64(FTopK) * FHiddenDim * 2;

    // Upload via staging buffer
    LStaging := FCompute.CreateGpuBuffer(
      LDownPackSize,
      VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT or VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );
    try
      FCompute.UploadToBuffer(LStaging, FDownPack.Memory, LDownPackSize);
      FCompute.CopyBuffer(LStaging, FDownPackGpu, LDownPackSize);
    finally
      FCompute.DestroyGpuBuffer(LStaging);
    end;

    // Write top-K indices and activations to mapped GPU buffers
    Move(LTopIndices[0], FIndicesMapped^, FTopK * SizeOf(UInt32));
    Move(LTopValues[0], FActivationsMapped^, FTopK * SizeOf(Single));

    // === Step 4: Accumulate (GPU) ===
    // Bind: down_pack (device-local) + activations + residual
    LAccumDescSet := FCompute.AllocateDescriptorSetForBuffers(
      LDescPool, FAccumDescLayout,
      [FDownPackGpu, FActivationsGpu, FResidualGpu]
    );

    LAccumPush.HiddenDimHalf := UInt32(FHiddenDim div 2);
    LAccumPush.TopK := UInt32(FTopK);

    FCompute.DispatchComputeWithPush(
      FAccumBundle.Pipeline,
      FAccumBundle.PipelineLayout,
      LAccumDescSet,
      @LAccumPush,
      SizeOf(LAccumPush),
      UInt32((FHiddenDim + 255) div 256)  // ceil(2560/256) = 10 workgroups
    );

  finally
    // Destroy pool (frees all descriptor sets allocated from it)
    FCompute.DestroyDescriptorPoolHandle(LDescPool);
  end;
end;

// ============================================================================
//  TVdxKNNWalk — Accessors
// ============================================================================

function TVdxKNNWalk.GetTopK(): Integer;
begin
  Result := FTopK;
end;

function TVdxKNNWalk.GetHiddenDim(): UInt64;
begin
  Result := FHiddenDim;
end;

function TVdxKNNWalk.GetFFNWidth(): UInt64;
begin
  Result := FFFNWidth;
end;

end.
