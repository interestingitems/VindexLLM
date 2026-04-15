{===============================================================================
  VindexLLM� - Liberating LLM inference

  Copyright � 2026-present tinyBigGAMES� LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit UVdxTestbed;

interface

procedure RunVdxTestbed();

implementation

uses
  WinAPI.Windows,
  System.SysUtils,
  System.IOUtils,
  System.Math,
  VindexLLM.Utils,
  VindexLLM.Vulkan,
  VindexLLM.Compute,
  VindexLLM.TurboQuant,
  VindexLLM.Inference,
  VindexLLM.Sampler;

procedure StatusCallback(const AText: string; const AUserData: Pointer);
begin
  TVdxUtils.PrintLn(AText);
end;

function PrintToken(const AToken: string; const AUserData: Pointer): Boolean;
begin
  Write(AToken);
  Result := (GetAsyncKeyState(VK_ESCAPE) and $8000) = 0;
end;

procedure PrintStats(const AStats: PVdxInferenceStats);
begin
  TVdxUtils.PrintLn();
  TVdxUtils.PrintLn('Prefill:    %d tokens in %.0fms (%.1f tok/s)', [
    AStats.PrefillTokens, AStats.PrefillTimeMs, AStats.PrefillTokPerSec]);
  TVdxUtils.PrintLn('Generation: %d tokens in %.0fms (%.1f tok/s)', [
    AStats.GeneratedTokens, AStats.GenerationTimeMs, AStats.GenerationTokPerSec]);
  TVdxUtils.PrintLn('TTFT: %.0fms | Total: %.0fms | Stop: %s', [
    AStats.TimeToFirstTokenMs, AStats.TotalTimeMs,
    CVdxStopReasons[AStats.StopReason]]);
end;

procedure Test01();
const
  CPrompt =
  '''
    Explain the differences between these three sorting algorithms: bubble sort,
    merge sort, and quicksort. For each one, describe how it works step by step,
    give the best-case and worst-case time complexity using big-O notation,
    explain when you would choose it over the others, and provide a real-world
    analogy that helps illustrate the concept. Also discuss whether each
    algorithm is stable or unstable, and what that means in practice.
  ''';
var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
begin
  TVdxUtils.Pause();
  LInference := TVdxInference.Create();
  try
    LInference.SetStatusCallback(StatusCallback, nil);
    //LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.f16.gguf');
    LInference.LoadModel('C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf');
    LInference.SetTokenCallback(PrintToken, nil);

    LConfig := TVdxSampler.DefaultConfig();
    LConfig.Temperature := 0.7;
    LConfig.TopK := 40;
    LConfig.Seed := 42;  // deterministic
    LInference.SetSamplerConfig(LConfig);

    //LInference.Generate(CPrompt);
    //LInference.Generate('how to make kno3?');
    //LInference.Generate('what is the capital of france?');
    LInference.Generate('who are you?');
    PrintStats(LInference.GetStats());
    LInference.UnloadModel();
  finally
    LInference.Free();
  end;
end;

procedure Test02_Sampling();
const
  CModel = 'C:\Dev\LLM\GGUF\gemma-3-4b-it-null-space-abliterated.Q8_0.gguf';
  CPrompt = 'what is the capital of france?';
var
  LInference: TVdxInference;
  LConfig: TVdxSamplerConfig;
begin
  TVdxUtils.Pause();
  LInference := TVdxInference.Create();
  try
    LInference.SetStatusCallback(StatusCallback, nil);
    LInference.LoadModel(CModel);
    LInference.SetTokenCallback(PrintToken, nil);

    // --- Run 1: Greedy (default) — must produce "Paris" ---
    TVdxUtils.PrintLn('=== Run 1: Greedy (Temperature=0) ===');
    LInference.SetSamplerConfig(TVdxSampler.DefaultConfig());
    LInference.Generate(CPrompt, 32);
    PrintStats(LInference.GetStats());
    TVdxUtils.PrintLn('');

    // --- Run 2: Sampling, Seed=42 ---
    TVdxUtils.PrintLn('=== Run 2: Temp=0.7, TopK=40, Seed=42 ===');
    LConfig := TVdxSampler.DefaultConfig();
    LConfig.Temperature := 0.7;
    LConfig.TopK := 40;
    LConfig.Seed := 42;
    LInference.SetSamplerConfig(LConfig);
    LInference.Generate(CPrompt, 64);
    PrintStats(LInference.GetStats());
    TVdxUtils.PrintLn('');

    // --- Run 3: Same seed — must match Run 2 ---
    TVdxUtils.PrintLn('=== Run 3: Same config (determinism check) ===');
    LInference.SetSamplerConfig(LConfig);
    LInference.Generate(CPrompt, 64);
    PrintStats(LInference.GetStats());
    TVdxUtils.PrintLn('');

    // --- Run 4: Full pipeline ---
    TVdxUtils.PrintLn('=== Run 4: Temp=0.8, TopK=40, TopP=0.95, MinP=0.05, RepPen=1.2 ===');
    LConfig := TVdxSampler.DefaultConfig();
    LConfig.Temperature := 0.8;
    LConfig.TopK := 40;
    LConfig.TopP := 0.95;
    LConfig.MinP := 0.05;
    LConfig.RepeatPenalty := 1.2;
    LConfig.RepeatWindow := 64;
    LConfig.Seed := 42;
    LInference.SetSamplerConfig(LConfig);
    LInference.Generate(CPrompt, 128);
    PrintStats(LInference.GetStats());
    TVdxUtils.PrintLn('');

    // --- Run 5: Greedy + repetition penalty ---
    TVdxUtils.PrintLn('=== Run 5: Greedy + RepeatPenalty=1.2 ===');
    LConfig := TVdxSampler.DefaultConfig();
    LConfig.RepeatPenalty := 1.2;
    LConfig.RepeatWindow := 64;
    LInference.SetSamplerConfig(LConfig);
    LInference.Generate(CPrompt, 64);
    PrintStats(LInference.GetStats());

    LInference.UnloadModel();
  finally
    LInference.Free();
  end;
end;

procedure RunVdxTestbed();
var
  LIndex: Integer;
begin
  try
    LIndex := 2;

    case LIndex of
      1: Test01();
      2: Test02_Sampling();
    end;
  except
    on E: Exception do
    begin
      TVdxUtils.PrintLn('');
      TVdxUtils.PrintLn(COLOR_RED + 'EXCEPTION: %s', [E.Message]);
    end;
  end;

  if TVdxUtils.RunFromIDE() then
    TVdxUtils.Pause();
end;

end.
