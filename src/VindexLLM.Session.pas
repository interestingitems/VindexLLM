{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Session;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.IOUtils,
  System.Generics.Collections,
  VindexLLM.Utils,
  VindexLLM.Inference,
  VindexLLM.Memory,
  VindexLLM.Embeddings,
  VindexLLM.Sampler,
  VindexLLM.ChatTemplate;


const
  // BM25 hits on rebuild retrieval
  CVdxSessionFTS5TopK = 5;

  // Cosine hits on rebuild retrieval (if embedder attached)
  CVdxSessionVectorTopK = 5;

  // Recent turns preserved on rebuild for conversational flow
  CVdxSessionRecentTurns = 3;

  // Max unique retrieved turns after dedup (FTS5 + vector merged)
  CVdxSessionMergeTopK = 5;

type

  { TVdxRetrievalConfig — controls per-turn RAG retrieval behavior.
    When Enabled, Chat() searches the memory DB for relevant chunks
    and facts and injects them as context in the user prompt. }
  TVdxRetrievalConfig = record
    Enabled: Boolean;       // master switch (default True)
    TopK: Integer;          // max retrieved items per Chat() call (default 3)
  end;

  { TVdxSession — high-level inference facade that ties together
    TVdxInference, TVdxMemory, and TVdxEmbeddings. The caller creates
    one instance, loads a model, and has a multi-turn conversation via
    Chat(). Turn logging, prompt formatting, context rebuild, and
    retrieval assembly happen behind the scenes. }
  TVdxSession = class(TVdxErrorsObject)
  private
    // Owned subsystems — created/destroyed by TVdxSession
    FInference: TVdxInference;
    FMemory: TVdxMemory;
    FEmbedder: TVdxEmbeddings;     // nil when no embedder path supplied

    // Configuration
    FSystemPrompt: string;
    FTurnIndex: Integer;           // tracks turn count for prompt assembly
    FLastUserMessage: string;      // raw user text for rebuild search queries
    FRetrievalConfig: TVdxRetrievalConfig;


    // Caller-facing callbacks — stored here, forwarded to FInference
    FTokenCallback: TVdxCallback<TVdxTokenCallback>;
    FCancelCallback: TVdxCallback<TVdxCancelCallback>;

    // Shared retrieval helper — searches FTS5 + optional vector,
    // deduplicates by TurnId, returns up to ATopK merged results.
    function RetrieveContext(const AQuery: string;
      const ATopK: Integer): TArray<TVdxMemoryTurn>;

    // Formats retrieved turns into a labeled context block for prompt
    // injection. Chunks and facts become "Reference information".
    function FormatRetrievedContext(
      const ATurns: TArray<TVdxMemoryTurn>): string;

    // Prompt formatting — builds Gemma 3 chat template strings.
    // AContext is a pre-formatted context block (or '' for none).
    function FormatPrompt(const AUserMessage: string;
      const AContext: string): string;

    // Rebuild handler — installed on FInference.SetRebuildCallback.
    // Queries FMemory for relevant context and assembles a replacement
    // prompt from system + retrieved + recent + current user message.
    function HandleRebuild(const APosition: UInt32;
      const AMaxContext: UInt32; const APrompt: string): string;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // --- Lifecycle ---
    function LoadModel(
      const AModelPath: string;
      const AMemoryDbPath: string;
      const AEmbedderPath: string;
      const AMaxContext: Integer = 2048;
      const ARebuildAt: Integer = -1
    ): Boolean;

    procedure UnloadModel();
    function IsLoaded(): Boolean;


    // --- Configuration ---
    procedure SetSystemPrompt(const APrompt: string);
    procedure SetSamplerConfig(const AConfig: TVdxSamplerConfig);
    procedure SetRetrievalConfig(const AConfig: TVdxRetrievalConfig);
    class function DefaultRetrievalConfig(): TVdxRetrievalConfig; static;
    procedure SetTokenCallback(const ACallback: TVdxTokenCallback;
      const AUserData: Pointer);
    procedure SetCancelCallback(const ACallback: TVdxCancelCallback;
      const AUserData: Pointer);

    // --- Conversation ---
    function Chat(const AUserMessage: string;
      const AMaxTokens: Integer = 256): string;

    // Reset conversation state — clears the KV cache, purges all turns
    // from memory, and resets the turn index. The next Chat() call
    // behaves as a fresh session (BOS, system prompt injected again).
    // Memory DB schema stays intact; only data is cleared.
    procedure ClearHistory();

    // --- Knowledge ---
    function AddDocument(const ASource: string; const ATitle: string;
      const AText: string; const AChunkTokens: Integer = 512;
      const AOverlapTokens: Integer = 64;
      const APinned: Boolean = False): Int64;
    function AddFact(const AText: string;
      const APinned: Boolean = True): Int64;

    // --- Info ---
    function GetStats(): PVdxInferenceStats;
    function GetTurnCount(): Integer;
  end;

implementation


{ TVdxSession }

constructor TVdxSession.Create();
begin
  inherited Create();
  FErrors := TVdxErrors.Create();
  FInference := nil;
  FMemory := nil;
  FEmbedder := nil;
  FSystemPrompt := '';
  FTurnIndex := 0;
  FLastUserMessage := '';
  FRetrievalConfig := DefaultRetrievalConfig();
  FTokenCallback := Default(TVdxCallback<TVdxTokenCallback>);
  FCancelCallback := Default(TVdxCallback<TVdxCancelCallback>);
end;

destructor TVdxSession.Destroy();
begin
  UnloadModel();
  FreeAndNil(FErrors);
  inherited Destroy();
end;

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

function TVdxSession.LoadModel(const AModelPath: string;
  const AMemoryDbPath: string; const AEmbedderPath: string;
  const AMaxContext: Integer; const ARebuildAt: Integer): Boolean;
var
  LLoaded: Boolean;
  LOpened: Boolean;
  LI: Integer;
  LItems: TList<TVdxError>;
begin
  Result := False;

  // Reset if already loaded
  if IsLoaded() then
    UnloadModel();

  FErrors.Clear();

  // Validate inputs
  if not TFile.Exists(AModelPath) then
  begin
    FErrors.Add(esError, 'SESSION', 'Model file not found: %s',
      [AModelPath]);
    Exit;
  end;

  if AMemoryDbPath.Trim().IsEmpty() then
  begin
    FErrors.Add(esError, 'SESSION', 'Memory DB path must not be empty');
    Exit;
  end;

  // --- Create and load inference engine ---
  FInference := TVdxInference.Create();
  LLoaded := FInference.LoadModel(AModelPath, AMaxContext, ARebuildAt);

  // Copy any inference errors/warnings into our error list
  LItems := FInference.GetErrors().GetItems();
  for LI := 0 to LItems.Count - 1 do
    FErrors.Add(LItems[LI].Severity, LItems[LI].Code, LItems[LI].Message);

  if not LLoaded then
  begin
    FreeAndNil(FInference);
    Exit;
  end;

  // --- Create and open memory DB ---
  FMemory := TVdxMemory.Create();
  LOpened := FMemory.OpenSession(AMemoryDbPath);
  if not LOpened then
  begin
    FErrors.Add(esError, 'SESSION', 'Failed to open memory DB: %s',
      [AMemoryDbPath]);
    FInference.UnloadModel();
    FreeAndNil(FInference);
    FreeAndNil(FMemory);
    Exit;
  end;

  // --- Optionally create and load embedder ---
  if AEmbedderPath.Trim() <> '' then
  begin
    if not TFile.Exists(AEmbedderPath) then
    begin
      FErrors.Add(esWarning, 'SESSION',
        'Embedder file not found, continuing without vector search: %s',
        [AEmbedderPath]);
    end
    else
    begin
      FEmbedder := TVdxEmbeddings.Create();
      LLoaded := FEmbedder.LoadModel(AEmbedderPath);

      // Copy embedder errors/warnings
      LItems := FEmbedder.GetErrors().GetItems();
      for LI := 0 to LItems.Count - 1 do
        FErrors.Add(LItems[LI].Severity, LItems[LI].Code,
          LItems[LI].Message);

      if not LLoaded then
      begin
        // Non-fatal — continue without vector search
        FErrors.Add(esWarning, 'SESSION',
          'Embedder failed to load, continuing without vector search');
        FreeAndNil(FEmbedder);
      end
      else
      begin
        FMemory.AttachEmbeddings(FEmbedder);
      end;
    end;
  end;

  // --- Forward pre-set callbacks ---
  if FTokenCallback.IsAssigned() then
    FInference.SetTokenCallback(FTokenCallback.Callback,
      FTokenCallback.UserData);

  if FCancelCallback.IsAssigned() then
    FInference.SetCancelCallback(FCancelCallback.Callback,
      FCancelCallback.UserData);

  // --- Install rebuild callback ---
  FInference.SetRebuildCallback(
    function(const APosition: UInt32; const AMaxCtx: UInt32;
      const APrompt: string; const AUserData: Pointer): string
    begin
      Result := HandleRebuild(APosition, AMaxCtx, APrompt);
    end,
    nil);

  FTurnIndex := 0;
  Result := True;
end;


procedure TVdxSession.UnloadModel();
begin
  if Assigned(FEmbedder) then
  begin
    if Assigned(FMemory) then
      FMemory.DetachEmbeddings();
    FEmbedder.UnloadModel();
    FreeAndNil(FEmbedder);
  end;

  if Assigned(FMemory) then
  begin
    FMemory.CloseSession();
    FreeAndNil(FMemory);
  end;

  if Assigned(FInference) then
  begin
    FInference.UnloadModel();
    FreeAndNil(FInference);
  end;

  FTurnIndex := 0;
end;

function TVdxSession.IsLoaded(): Boolean;
begin
  Result := Assigned(FInference) and Assigned(FMemory);
end;


// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

procedure TVdxSession.SetSystemPrompt(const APrompt: string);
begin
  FSystemPrompt := APrompt;
end;

procedure TVdxSession.SetSamplerConfig(const AConfig: TVdxSamplerConfig);
begin
  if Assigned(FInference) then
    FInference.SetSamplerConfig(AConfig);
end;

procedure TVdxSession.SetRetrievalConfig(const AConfig: TVdxRetrievalConfig);
begin
  FRetrievalConfig := AConfig;
end;

class function TVdxSession.DefaultRetrievalConfig(): TVdxRetrievalConfig;
begin
  Result.Enabled := True;
  Result.TopK := 3;
end;

procedure TVdxSession.SetTokenCallback(const ACallback: TVdxTokenCallback;
  const AUserData: Pointer);
begin
  FTokenCallback.Callback := ACallback;
  FTokenCallback.UserData := AUserData;
  if Assigned(FInference) then
    FInference.SetTokenCallback(ACallback, AUserData);
end;

procedure TVdxSession.SetCancelCallback(const ACallback: TVdxCancelCallback;
  const AUserData: Pointer);
begin
  FCancelCallback.Callback := ACallback;
  FCancelCallback.UserData := AUserData;
  if Assigned(FInference) then
    FInference.SetCancelCallback(ACallback, AUserData);
end;


// ---------------------------------------------------------------------------
// Conversation
// ---------------------------------------------------------------------------

function TVdxSession.Chat(const AUserMessage: string;
  const AMaxTokens: Integer): string;
var
  LPrompt: string;
  LResponse: string;
  LContext: string;
  LRetrieved: TArray<TVdxMemoryTurn>;
begin
  Result := '';

  if not IsLoaded() then
    Exit;

  // 1. Store raw user text for rebuild search queries
  FLastUserMessage := AUserMessage;

  // 2. Log user turn to memory
  FMemory.AppendTurn(CVdxMemRoleUser, AUserMessage, 0);

  // 3. Per-turn RAG retrieval — search for relevant chunks and facts
  LContext := '';
  if FRetrievalConfig.Enabled and (FRetrievalConfig.TopK > 0) then
  begin
    LRetrieved := RetrieveContext(AUserMessage, FRetrievalConfig.TopK);

    if Length(LRetrieved) > 0 then
      LContext := FormatRetrievedContext(LRetrieved);
  end;

  // 4. Assemble prompt (system + context + user)
  LPrompt := FormatPrompt(AUserMessage, LContext);

  // 5. Generate assistant response
  LResponse := FInference.Generate(LPrompt, AMaxTokens);

  // 6. Log assistant turn to memory
  FMemory.AppendTurn(CVdxMemRoleAssistant, LResponse, 0);

  // 7. Track turn count (user + assistant = 2 per Chat call)
  Inc(FTurnIndex, 2);

  Result := LResponse;
end;

procedure TVdxSession.ClearHistory();
begin
  if not IsLoaded() then
    Exit;

  FInference.ResetKVCache();
  FMemory.PurgeAll();
  FTurnIndex := 0;
  FLastUserMessage := '';
end;

// ---------------------------------------------------------------------------
// Private — prompt formatting
// ---------------------------------------------------------------------------

function TVdxSession.FormatPrompt(const AUserMessage: string;
  const AContext: string): string;
var
  LContent: string;
begin
  // Build the content to place inside the user turn.
  // Layout: [system prompt] [context block] [user message]
  LContent := '';

  // System prompt on first turn only
  if (FInference.GetKVCachePosition() = 0) and (FSystemPrompt <> '') then
    LContent := FSystemPrompt + #10 + #10;

  // Injected RAG context (if any)
  if AContext <> '' then
    LContent := LContent + AContext + #10 + #10;

  // User message
  LContent := LContent + AUserMessage;

  // Wrap in Gemma 3 chat template:
  //   <start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n
  Result := TVdxChatTemplate.FormatPrompt('gemma3', LContent);

  // For continuation turns, prepend newline to bridge after previous
  // <end_of_turn> token that is already in the KV cache.
  if FInference.GetKVCachePosition() > 0 then
    Result := #10 + Result;
end;

// ---------------------------------------------------------------------------
// Private — shared retrieval
// ---------------------------------------------------------------------------

function TVdxSession.RetrieveContext(const AQuery: string;
  const ATopK: Integer): TArray<TVdxMemoryTurn>;
var
  LFTS5Hits: TArray<TVdxMemoryTurn>;
  LVectorHits: TArray<TVdxMemoryTurn>;
  LSeenIds: TDictionary<Int64, Boolean>;
  LMergeCount: Integer;
  LI: Integer;
begin
  // Search FTS5 (BM25 keyword ranking)
  try
    LFTS5Hits := FMemory.SearchFTS5(AQuery, ATopK);
  except
    SetLength(LFTS5Hits, 0);
  end;

  // Search vector (cosine similarity) if embedder attached
  if Assigned(FEmbedder) and FEmbedder.IsLoaded() then
  begin
    try
      LVectorHits := FMemory.SearchVector(AQuery, ATopK);
    except
      SetLength(LVectorHits, 0);
    end;
  end
  else
    SetLength(LVectorHits, 0);

  // Merge and deduplicate by TurnId — FTS5 hits first, vector fills
  LSeenIds := TDictionary<Int64, Boolean>.Create();
  try
    LMergeCount := 0;
    SetLength(Result, ATopK);

    for LI := 0 to High(LFTS5Hits) do
    begin
      if LMergeCount >= ATopK then
        Break;
      if not LSeenIds.ContainsKey(LFTS5Hits[LI].TurnId) then
      begin
        LSeenIds.Add(LFTS5Hits[LI].TurnId, True);
        Result[LMergeCount] := LFTS5Hits[LI];
        Inc(LMergeCount);
      end;
    end;

    for LI := 0 to High(LVectorHits) do
    begin
      if LMergeCount >= ATopK then
        Break;
      if not LSeenIds.ContainsKey(LVectorHits[LI].TurnId) then
      begin
        LSeenIds.Add(LVectorHits[LI].TurnId, True);
        Result[LMergeCount] := LVectorHits[LI];
        Inc(LMergeCount);
      end;
    end;

    SetLength(Result, LMergeCount);
  finally
    LSeenIds.Free();
  end;
end;

function TVdxSession.FormatRetrievedContext(
  const ATurns: TArray<TVdxMemoryTurn>): string;
var
  LI: Integer;
begin
  if Length(ATurns) = 0 then
  begin
    Result := '';
    Exit;
  end;

  Result := 'Reference information:';
  for LI := 0 to High(ATurns) do
    Result := Result + #10 + '- ' + ATurns[LI].Text;
end;

// ---------------------------------------------------------------------------
// Private — rebuild handler
// ---------------------------------------------------------------------------

function TVdxSession.HandleRebuild(const APosition: UInt32;
  const AMaxContext: UInt32; const APrompt: string): string;
var
  LRetrieved: TArray<TVdxMemoryTurn>;
  LResult: string;
  LI: Integer;
begin
  Result := '';
  if not IsLoaded() then
    Exit;

  // --- 1. Retrieve relevant context via RAG ---
  LRetrieved := RetrieveContext(FLastUserMessage, CVdxSessionMergeTopK);

  // --- 2. Assemble lean replacement prompt ---
  // Single user turn: system prompt + retrieved context + current question
  LResult := '<start_of_turn>user' + #10;

  if FSystemPrompt <> '' then
    LResult := LResult + FSystemPrompt + #10 + #10;

  if Length(LRetrieved) > 0 then
  begin
    LResult := LResult + 'Reference information:' + #10;
    for LI := 0 to High(LRetrieved) do
      LResult := LResult + '- ' + LRetrieved[LI].Text + #10;
    LResult := LResult + #10;
  end;

  LResult := LResult + FLastUserMessage;
  LResult := LResult + '<end_of_turn>' + #10;

  // Open model turn for generation
  LResult := LResult + '<start_of_turn>model' + #10;

  Result := LResult;
end;


// ---------------------------------------------------------------------------
// Knowledge
// ---------------------------------------------------------------------------

function TVdxSession.AddDocument(const ASource: string;
  const ATitle: string; const AText: string;
  const AChunkTokens: Integer; const AOverlapTokens: Integer;
  const APinned: Boolean): Int64;
begin
  if Assigned(FMemory) then
    Result := FMemory.AddDocument(ASource, ATitle, AText, AChunkTokens,
      AOverlapTokens, APinned)
  else
    Result := -1;
end;

function TVdxSession.AddFact(const AText: string;
  const APinned: Boolean): Int64;
begin
  if Assigned(FMemory) then
    Result := FMemory.AddFact(AText, APinned)
  else
    Result := -1;
end;

// ---------------------------------------------------------------------------
// Info
// ---------------------------------------------------------------------------

function TVdxSession.GetStats(): PVdxInferenceStats;
begin
  if Assigned(FInference) then
    Result := FInference.GetStats()
  else
    Result := nil;
end;

function TVdxSession.GetTurnCount(): Integer;
begin
  if Assigned(FMemory) then
    Result := FMemory.GetTurnCount()
  else
    Result := 0;
end;

end.
