{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.ChatTemplate;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  VindexLLM.Utils,
  VindexLLM.GGUFReader;

type
  { TVdxChatTemplate }
  TVdxChatTemplate = class(TVdxBaseObject)
  public
    // Detect model family from GGUF metadata key 'general.architecture'
    // Returns lowercase architecture string (e.g. 'gemma3', 'llama')
    // Returns 'unknown' if the key is not present
    class function DetectArchitecture(
      const AReader: TVdxGGUFReader): string;

    // Format a user prompt into the model's expected chat template
    // BOS is NOT included — caller must prepend BOS during tokenization
    // Falls back to ChatML (OpenAI standard) for unknown architectures
    class function FormatPrompt(
      const AArchitecture: string;
      const APrompt: string): string;

    // Format input text for an embedding model by prepending the
    // architecture-appropriate task prefix.
    // AIsQuery = True  → query prefix (retrieval search text)
    // AIsQuery = False → document prefix (text being stored / indexed)
    // BOS is NOT included — caller prepends BOS during tokenization.
    // Unknown architectures return AText unchanged (no prefix).
    class function FormatEmbedding(
      const AArchitecture: string;
      const AText: string;
      const AIsQuery: Boolean): string;
  end;

implementation

{ TVdxChatTemplate }

class function TVdxChatTemplate.DetectArchitecture(
  const AReader: TVdxGGUFReader): string;
begin
  if AReader.HasMetadata('general.architecture') then
    Result := LowerCase(AReader.GetMetadataString('general.architecture'))
  else
    Result := 'unknown';
end;

class function TVdxChatTemplate.FormatPrompt(
  const AArchitecture: string;
  const APrompt: string): string;
begin
  // Gemma 3 instruction-tuned chat template
  // Special tokens are encoded as text — the BPE tokenizer resolves them
  if AArchitecture = 'gemma3' then
  begin
    Result :=
      '<start_of_turn>user' + #10 +
      APrompt + '<end_of_turn>' + #10 +
      '<start_of_turn>model' + #10;
  end
  else
  begin
    // ChatML (OpenAI standard) — widely supported default
    Result :=
      '<|im_start|>user' + #10 +
      APrompt + '<|im_end|>' + #10 +
      '<|im_start|>assistant' + #10;
  end;
end;

class function TVdxChatTemplate.FormatEmbedding(
  const AArchitecture: string;
  const AText: string;
  const AIsQuery: Boolean): string;
begin
  // EmbeddingGemma task prefixes come from the model's
  // config_sentence_transformers.json — 'Retrieval-query' and
  // 'Retrieval-document' presets. They are NOT stored in the GGUF, so we
  // dispatch by architecture the same way FormatPrompt does for chat.
  // Source: https://huggingface.co/google/embeddinggemma-300m
  if AArchitecture = 'gemma-embedding' then
  begin
    if AIsQuery then
      Result := 'task: search result | query: ' + AText
    else
      Result := 'title: none | text: ' + AText;
  end
  else
  begin
    // Unknown embedding architecture — pass text through unchanged rather
    // than inject a prefix the model wasn't trained to recognize.
    Result := AText;
  end;
end;

end.
