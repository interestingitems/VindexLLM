{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Memory;

{$I VindexLLM.Defines.inc}

interface

uses
  System.SysUtils,
  System.Classes,
  System.DateUtils,
  System.Generics.Collections,
  System.Generics.Defaults,
  Data.DB,
  FireDAC.Stan.Intf,
  FireDAC.Stan.Option,
  FireDAC.Stan.Error,
  FireDAC.Stan.Def,
  FireDAC.Stan.Async,
  FireDAC.DApt,
  FireDAC.Phys,
  FireDAC.Phys.SQLite,
  FireDAC.Phys.SQLiteWrapper,
  FireDAC.Phys.SQLiteWrapper.Stat,
  FireDAC.Comp.Client,
  VindexLLM.Utils,
  VindexLLM.Embeddings;

const
  // Rebuild threshold: when current KV position >= 75% of max context,
  // the generation loop should trigger a rebuild cycle (FTS5 retrieval
  // + recent turns + new user turn). Integration is a separate task —
  // this unit only exposes the constant for consumers to use.
  CVdxMemRebuildThreshold: Single = 0.75;

  // Recent-turns window preserved on rebuild for conversational flow.
  CVdxMemKeepRecentTurns: Integer = 3;

  // Default Top-K for BM25-ranked FTS5 retrieval.
  CVdxMemFTS5TopK: Integer = 5;

  // Turns shorter than this token count are not worth indexing
  // meaningfully. Enforced at query-time by the caller, not in schema.
  CVdxMemMinTurnTokens: Integer = 2;

  // Standard role strings for the turns table. Kept here so callers
  // don't sprinkle magic strings across the codebase.
  CVdxMemRoleUser:      string = 'user';
  CVdxMemRoleAssistant: string = 'assistant';
  CVdxMemRoleSystem:    string = 'system';

type

  { TVdxMemoryTurn — a single conversational turn as stored in the
    long-term memory DB. Returned by reads and searches. Score and
    CosineScore carry relevance signals from the two search APIs:
      * Score       — BM25 rank from SearchFTS5 (lower = more relevant)
      * CosineScore — cosine similarity from SearchVector (higher = more relevant)
    Both default to zero on non-search reads. Keeping them as separate
    fields avoids the footgun of a single field whose meaning flips
    depending on which search method produced it. }
  TVdxMemoryTurn = record
    TurnId      : Int64;
    TurnIndex   : Integer;
    Role        : string;
    Text        : string;
    TokenCount  : Integer;
    CreatedAt   : Int64;    // unix epoch seconds
    Score       : Single;   // BM25; only set by SearchFTS5  (lower = better)
    CosineScore : Single;   // cosine; only set by SearchVector (higher = better)
  end;

  { TVdxMemory — SQLite + FTS5-backed turn journal for one conversation
    session. One DB file per session. Append-only from the caller's
    view; the FTS5 index stays in sync via triggers declared in the
    schema. Static FireDAC SQLite linkage — no sqlite3.dll deployment. }
  TVdxMemory = class(TVdxBaseObject)
  private
    FLink: TFDPhysSQLiteDriverLink;
    FConn: TFDConnection;
    FDbPath: string;
    FNextTurnIndex: Integer;

    // Optional embedder binding — non-owning reference supplied via
    // AttachEmbeddings. When non-nil, AppendTurn embeds turn text inline
    // and SearchVector is enabled. FEmbedderDim is cached at attach
    // time to validate BLOB byte-lengths at search time without a round
    // trip to the embedder every call.
    FEmbedder: TVdxEmbeddings;
    FEmbedderDim: Integer;

    // Schema bootstrap — runs DDL idempotently on every open.
    procedure CreateSchemaIfNeeded();

    // Populate FNextTurnIndex from MAX(turn_index)+1, or 0 if empty.
    procedure LoadNextTurnIndex();

    // Strip FTS5 operator characters from user input so a raw prompt
    // never becomes an FTS5 syntax error. Returns the sanitized query
    // with tokens joined by ' OR ' for recall-oriented retrieval;
    // returns '' if nothing usable remained after stripping.
    function  SanitizeFTSQuery(const AQuery: string): string;

    // Shared record-reader used by every result-shape method.
    function  ReadTurnFromQuery(const AQuery: TFDQuery): TVdxMemoryTurn;

    // Schema migration — runs ALTER TABLE for columns added after v1.
    // Idempotent: introspects current schema via PRAGMA before altering.
    procedure MigrateSchema();

    // Embedding vector serialization. Vectors are stored as raw
    // little-endian F32 bytes — no header, no dim marker. Dim is
    // validated at read-time by comparing byte length to the attached
    // embedder's expected size.
    function  SingleArrayToBytes(const AVec: TArray<Single>): TBytes;
    function  BytesToSingleArray(const ABytes: TBytes): TArray<Single>;

    // Cosine similarity via dot product. Both inputs MUST be
    // L2-normalized — this function does not renormalize. For
    // normalized vectors cosine equals the dot product.
    function  CosineDot(const AVecA, AVecB: TArray<Single>): Single;

  public
    constructor Create(); override;
    destructor Destroy(); override;

    // Session lifecycle
    function  OpenSession(const ADbPath: string): Boolean;
    procedure CloseSession();
    function  IsOpen(): Boolean;

    // Metadata (session_meta table)
    procedure SetMeta(const AKey: string; const AValue: string);
    function  GetMeta(const AKey: string): string;

    // Embedder binding — optional. When an embedder is attached,
    // AppendTurn embeds new turns inline and SearchVector becomes
    // available. The reference is non-owning: the caller retains
    // ownership of the TVdxEmbeddings instance and must call
    // DetachEmbeddings (or destroy this TVdxMemory) before freeing
    // the embedder. Attach requires a loaded embedder and raises
    // otherwise; Detach is idempotent.
    procedure AttachEmbeddings(const AEmbedder: TVdxEmbeddings);
    procedure DetachEmbeddings();

    // Turn CRUD
    function  AppendTurn(const ARole: string; const AText: string;
      const ATokenCount: Integer): Int64;
    function  GetTurn(const ATurnId: Int64): TVdxMemoryTurn;
    function  GetRecentTurns(const ACount: Integer): TArray<TVdxMemoryTurn>;
    function  GetTurnCount(): Integer;

    // Retrieval
    function  SearchFTS5(const AQuery: string;
      const ATopK: Integer): TArray<TVdxMemoryTurn>;

    // Brute-force cosine search over stored embeddings. Requires an
    // attached and loaded embedder (raises otherwise). O(N) per call;
    // fine under ~10k turns. Returns up to ATopK turns sorted by
    // cosine similarity DESC (higher = more relevant). The CosineScore
    // field on each returned record holds the raw cosine value.
    // Raises if any stored embedding's byte length disagrees with the
    // current embedder's dim — indicates embedder-swap mid-DB, which
    // should never happen and is treated as a bug rather than handled
    // silently.
    function  SearchVector(const AQuery: string;
      const ATopK: Integer): TArray<TVdxMemoryTurn>;

    // Utility
    function  GetDbPath(): string;
  end;

implementation

const
  CVdxDDLTurns =
    'CREATE TABLE IF NOT EXISTS turns (' + sLineBreak +
    '  turn_id     INTEGER PRIMARY KEY AUTOINCREMENT,' + sLineBreak +
    '  turn_index  INTEGER NOT NULL,' + sLineBreak +
    '  role        TEXT    NOT NULL,' + sLineBreak +
    '  text        TEXT    NOT NULL,' + sLineBreak +
    '  token_count INTEGER NOT NULL,' + sLineBreak +
    '  created_at  INTEGER NOT NULL' + sLineBreak +
    ')';

  CVdxDDLTurnsIndex =
    'CREATE INDEX IF NOT EXISTS idx_turns_index ON turns(turn_index)';

  CVdxDDLMeta =
    'CREATE TABLE IF NOT EXISTS session_meta (' + sLineBreak +
    '  key   TEXT PRIMARY KEY,' + sLineBreak +
    '  value TEXT' + sLineBreak +
    ')';

  // External-content FTS5 table — references turns.text rather than
  // duplicating it, saving roughly half the disk footprint.
  CVdxDDLFTS =
    'CREATE VIRTUAL TABLE IF NOT EXISTS turns_fts USING fts5(' + sLineBreak +
    '  text,' + sLineBreak +
    '  content=''turns'',' + sLineBreak +
    '  content_rowid=''turn_id''' + sLineBreak +
    ')';

  CVdxDDLTrigInsert =
    'CREATE TRIGGER IF NOT EXISTS turns_ai AFTER INSERT ON turns BEGIN' + sLineBreak +
    '  INSERT INTO turns_fts(rowid, text) VALUES (new.turn_id, new.text);' + sLineBreak +
    'END';

  CVdxDDLTrigDelete =
    'CREATE TRIGGER IF NOT EXISTS turns_ad AFTER DELETE ON turns BEGIN' + sLineBreak +
    '  INSERT INTO turns_fts(turns_fts, rowid, text)' + sLineBreak +
    '    VALUES(''delete'', old.turn_id, old.text);' + sLineBreak +
    'END';

  CVdxDDLTrigUpdate =
    'CREATE TRIGGER IF NOT EXISTS turns_au AFTER UPDATE ON turns BEGIN' + sLineBreak +
    '  INSERT INTO turns_fts(turns_fts, rowid, text)' + sLineBreak +
    '    VALUES(''delete'', old.turn_id, old.text);' + sLineBreak +
    '  INSERT INTO turns_fts(rowid, text) VALUES (new.turn_id, new.text);' + sLineBreak +
    'END';

  // Characters that would be interpreted as FTS5 syntax if passed raw.
  // Stripped from user input before it reaches MATCH.
  CVdxFTSOperatorSet: TSysCharSet = ['"', '*', '(', ')', ':', '-', '+', '^'];

constructor TVdxMemory.Create();
begin
  inherited Create();
  FLink := nil;
  FConn := nil;
  FDbPath := '';
  FNextTurnIndex := 0;
  FEmbedder := nil;
  FEmbedderDim := 0;
end;

destructor TVdxMemory.Destroy();
begin
  CloseSession();
  inherited Destroy();
end;

function TVdxMemory.IsOpen(): Boolean;
begin
  Result := (FConn <> nil) and FConn.Connected;
end;

function TVdxMemory.GetDbPath(): string;
begin
  Result := FDbPath;
end;

function TVdxMemory.OpenSession(const ADbPath: string): Boolean;
begin
  Result := False;

  // Idempotent re-open: silently close an existing session first.
  if IsOpen() then
    CloseSession();

  FDbPath := ADbPath;

  try
    FLink := TFDPhysSQLiteDriverLink.Create(nil);
    FLink.EngineLinkage := slStatic;

    FConn := TFDConnection.Create(nil);
    FConn.DriverName := 'SQLite';
    FConn.Params.Values['Database'] := ADbPath;
    FConn.LoginPrompt := False;
    FConn.Open();

    CreateSchemaIfNeeded();
    LoadNextTurnIndex();

    Result := True;
  except
    // On any failure during open, clean up partial state so the object
    // is not left half-constructed. Re-raise so the caller sees why.
    CloseSession();
    raise;
  end;
end;

procedure TVdxMemory.CloseSession();
begin
  if FConn <> nil then
  begin
    if FConn.Connected then
      FConn.Close();
    FConn.Free();
    FConn := nil;
  end;
  if FLink <> nil then
  begin
    FLink.Free();
    FLink := nil;
  end;
  FNextTurnIndex := 0;
end;

procedure TVdxMemory.CreateSchemaIfNeeded();
begin
  // All DDL uses IF NOT EXISTS, so this is safe to run on every open.
  FConn.ExecSQL(CVdxDDLTurns);
  FConn.ExecSQL(CVdxDDLTurnsIndex);
  FConn.ExecSQL(CVdxDDLMeta);
  FConn.ExecSQL(CVdxDDLFTS);
  FConn.ExecSQL(CVdxDDLTrigInsert);
  FConn.ExecSQL(CVdxDDLTrigDelete);
  FConn.ExecSQL(CVdxDDLTrigUpdate);

  // Migrations for columns added after v1 (e.g. embedding BLOB in 2.5b).
  // Kept separate from DDL above because ALTER TABLE is not idempotent
  // in SQLite — MigrateSchema introspects first, then alters if needed.
  MigrateSchema();
end;

procedure TVdxMemory.MigrateSchema();
var
  LQuery: TFDQuery;
  LHasEmbedding: Boolean;
begin
  // Probe current columns via SQLite introspection. PRAGMA table_info
  // returns one row per column with the column name in the 'name' field.
  LHasEmbedding := False;
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'PRAGMA table_info(turns)';
    LQuery.Open();
    while not LQuery.Eof do
    begin
      if SameText(LQuery.FieldByName('name').AsString, 'embedding') then
      begin
        LHasEmbedding := True;
        Break;
      end;
      LQuery.Next();
    end;
    LQuery.Close();
  finally
    LQuery.Free();
  end;

  // Add embedding column if missing. SQLite's ALTER TABLE ADD COLUMN
  // is not idempotent (it errors if the column already exists), hence
  // the introspection guard above. Default NULL for all existing rows.
  if not LHasEmbedding then
    FConn.ExecSQL('ALTER TABLE turns ADD COLUMN embedding BLOB');
end;

procedure TVdxMemory.LoadNextTurnIndex();
var
  LQuery: TFDQuery;
begin
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    // COALESCE returns -1 when the table is empty, so +1 yields 0.
    LQuery.SQL.Text := 'SELECT COALESCE(MAX(turn_index), -1) + 1 FROM turns';
    LQuery.Open();
    FNextTurnIndex := LQuery.Fields[0].AsInteger;
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

procedure TVdxMemory.SetMeta(const AKey: string; const AValue: string);
var
  LQuery: TFDQuery;
begin
  if not IsOpen() then
    raise Exception.Create('TVdxMemory.SetMeta: session not open');

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'INSERT INTO session_meta(key, value) VALUES (:k, :v) ' +
      'ON CONFLICT(key) DO UPDATE SET value = excluded.value';
    LQuery.ParamByName('k').AsString := AKey;
    LQuery.ParamByName('v').AsString := AValue;
    LQuery.ExecSQL();
  finally
    LQuery.Free();
  end;
end;

function TVdxMemory.GetMeta(const AKey: string): string;
var
  LQuery: TFDQuery;
begin
  Result := '';
  if not IsOpen() then
    raise Exception.Create('TVdxMemory.GetMeta: session not open');

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'SELECT value FROM session_meta WHERE key = :k';
    LQuery.ParamByName('k').AsString := AKey;
    LQuery.Open();
    if not LQuery.Eof then
      Result := LQuery.Fields[0].AsString;
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

procedure TVdxMemory.AttachEmbeddings(const AEmbedder: TVdxEmbeddings);
begin
  // Strict contract: non-nil, already loaded. Catches setup-order bugs
  // loudly at bind time rather than letting them surface later inside
  // a Generate() loop. A per-use re-check still happens in AppendTurn
  // and SearchVector to cover the case where the caller unloads the
  // embedder after attach.
  if AEmbedder = nil then
    raise Exception.Create(
      'TVdxMemory.AttachEmbeddings: embedder is nil');
  if not AEmbedder.IsLoaded() then
    raise Exception.Create(
      'TVdxMemory.AttachEmbeddings: embedder is not loaded');

  FEmbedder := AEmbedder;
  FEmbedderDim := AEmbedder.GetEmbeddingDim();
end;

procedure TVdxMemory.DetachEmbeddings();
begin
  // Idempotent — safe to call with nothing attached. Clears the cached
  // dim so any stale BLOB length checks (should we ever reattach a
  // different embedder) start fresh.
  FEmbedder := nil;
  FEmbedderDim := 0;
end;

function TVdxMemory.AppendTurn(const ARole: string; const AText: string;
  const ATokenCount: Integer): Int64;
var
  LQuery: TFDQuery;
  LNowUnix: Int64;
  LVec: TArray<Single>;
  LBytes: TBytes;
  LStream: TBytesStream;
begin
  Result := 0;
  if not IsOpen() then
    raise Exception.Create('TVdxMemory.AppendTurn: session not open');

  // Per-use guard: if an embedder was attached but has since been
  // unloaded, surface that explicitly. Silent fallback to NULL would
  // hide a real bug in caller setup — they should DetachEmbeddings
  // before unloading the embedder model.
  if (FEmbedder <> nil) and (not FEmbedder.IsLoaded()) then
    raise Exception.Create(
      'TVdxMemory.AppendTurn: attached embedder is no longer loaded — ' +
      'call DetachEmbeddings before unloading the embedder');

  // UTC unix seconds — matches schema contract documented in TASK-memory-v1.
  LNowUnix := DateTimeToUnix(Now(), False);

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'INSERT INTO turns ' +
      '  (turn_index, role, text, token_count, created_at, embedding) ' +
      'VALUES (:idx, :role, :txt, :tok, :ts, :emb)';
    LQuery.ParamByName('idx').AsInteger := FNextTurnIndex;
    LQuery.ParamByName('role').AsString := ARole;
    LQuery.ParamByName('txt').AsString  := AText;
    LQuery.ParamByName('tok').AsInteger := ATokenCount;
    LQuery.ParamByName('ts').AsLargeInt := LNowUnix;

    if FEmbedder <> nil then
    begin
      // AIsQuery=False selects EmbeddingGemma's document-side task
      // prefix ('title: none | text: ...'), which is the correct side
      // for stored content. Query-side prefix is used only when the
      // caller later searches via SearchVector.
      //
      // Strict exception propagation: if Embed() raises, the INSERT
      // is abandoned and the turn is NOT saved. This is deliberate —
      // silently saving a turn with NULL embedding on an embed failure
      // would mask a real problem in the embedder.
      LVec := FEmbedder.Embed(AText, False);
      LBytes := SingleArrayToBytes(LVec);
      // TFDParam.AsBytes is an indexed property (for array params), not
      // a scalar setter — so we bind via a TBytesStream + LoadFromStream
      // with an explicit ftBlob. Stream is stack-scoped by try/finally.
      LStream := TBytesStream.Create(LBytes);
      try
        LQuery.ParamByName('emb').LoadFromStream(LStream, ftBlob);
      finally
        LStream.Free();
      end;
    end
    else
    begin
      // No embedder attached — store SQL NULL in the embedding column.
      // Row is still fully searchable via FTS5; only SearchVector
      // will skip it.
      LQuery.ParamByName('emb').DataType := ftBlob;
      LQuery.ParamByName('emb').Clear();
    end;

    LQuery.ExecSQL();

    // SQLite returns the AUTOINCREMENT rowid for the just-inserted row.
    LQuery.SQL.Text := 'SELECT last_insert_rowid()';
    LQuery.Open();
    Result := LQuery.Fields[0].AsLargeInt;
    LQuery.Close();

    Inc(FNextTurnIndex);
  finally
    LQuery.Free();
  end;
end;

function TVdxMemory.ReadTurnFromQuery(
  const AQuery: TFDQuery): TVdxMemoryTurn;
begin
  // Explicit field init — record locals are not zero-initialized in Delphi.
  Result.TurnId      := AQuery.FieldByName('turn_id').AsLargeInt;
  Result.TurnIndex   := AQuery.FieldByName('turn_index').AsInteger;
  Result.Role        := AQuery.FieldByName('role').AsString;
  Result.Text        := AQuery.FieldByName('text').AsString;
  Result.TokenCount  := AQuery.FieldByName('token_count').AsInteger;
  Result.CreatedAt   := AQuery.FieldByName('created_at').AsLargeInt;
  Result.Score       := 0.0;
  Result.CosineScore := 0.0;
end;

function TVdxMemory.SingleArrayToBytes(
  const AVec: TArray<Single>): TBytes;
var
  LLen: Integer;
begin
  // Raw memcpy — platform is little-endian F32, matches the on-disk
  // byte order we want for BLOB storage.
  LLen := Length(AVec) * SizeOf(Single);
  SetLength(Result, LLen);
  if LLen > 0 then
    Move(AVec[0], Result[0], LLen);
end;

function TVdxMemory.BytesToSingleArray(
  const ABytes: TBytes): TArray<Single>;
var
  LCount: Integer;
begin
  // Integer-divide to silently drop any trailing partial element —
  // callers should never pass a non-multiple-of-4 buffer, but rejecting
  // at this layer would force every read site to special-case it.
  LCount := Length(ABytes) div SizeOf(Single);
  SetLength(Result, LCount);
  if LCount > 0 then
    Move(ABytes[0], Result[0], LCount * SizeOf(Single));
end;

function TVdxMemory.CosineDot(const AVecA, AVecB: TArray<Single>): Single;
var
  LI: Integer;
  LSum: Single;
  LN: Integer;
begin
  // Both inputs must be the same length AND L2-normalized by the caller.
  // For unit vectors dot product equals cosine similarity exactly, so
  // we skip the norm division.
  LSum := 0.0;
  LN := Length(AVecA);
  if Length(AVecB) < LN then
    LN := Length(AVecB);
  for LI := 0 to LN - 1 do
    LSum := LSum + AVecA[LI] * AVecB[LI];
  Result := LSum;
end;

function TVdxMemory.GetTurn(const ATurnId: Int64): TVdxMemoryTurn;
var
  LQuery: TFDQuery;
begin
  // Zero-init so the caller sees an empty record if the row isn't found.
  Result.TurnId      := 0;
  Result.TurnIndex   := 0;
  Result.Role        := '';
  Result.Text        := '';
  Result.TokenCount  := 0;
  Result.CreatedAt   := 0;
  Result.Score       := 0.0;
  Result.CosineScore := 0.0;

  if not IsOpen() then
    raise Exception.Create('TVdxMemory.GetTurn: session not open');

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'SELECT turn_id, turn_index, role, text, token_count, created_at ' +
      'FROM turns WHERE turn_id = :id';
    LQuery.ParamByName('id').AsLargeInt := ATurnId;
    LQuery.Open();
    if not LQuery.Eof then
      Result := ReadTurnFromQuery(LQuery);
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

function TVdxMemory.GetTurnCount(): Integer;
var
  LQuery: TFDQuery;
begin
  Result := 0;
  if not IsOpen() then
    raise Exception.Create('TVdxMemory.GetTurnCount: session not open');

  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text := 'SELECT COUNT(*) FROM turns';
    LQuery.Open();
    Result := LQuery.Fields[0].AsInteger;
    LQuery.Close();
  finally
    LQuery.Free();
  end;
end;

function TVdxMemory.GetRecentTurns(
  const ACount: Integer): TArray<TVdxMemoryTurn>;
var
  LQuery: TFDQuery;
  LList: TList<TVdxMemoryTurn>;
  LI: Integer;
begin
  Result := nil;
  if not IsOpen() then
    raise Exception.Create('TVdxMemory.GetRecentTurns: session not open');
  if ACount <= 0 then
    Exit;

  LList := TList<TVdxMemoryTurn>.Create();
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    // Pull the tail of the journal (DESC, LIMIT), then reverse in-memory
    // so the caller receives chronological order across the window:
    // oldest -> newest within the returned slice.
    LQuery.SQL.Text :=
      'SELECT turn_id, turn_index, role, text, token_count, created_at ' +
      'FROM turns ORDER BY turn_index DESC LIMIT :lim';
    LQuery.ParamByName('lim').AsInteger := ACount;
    LQuery.Open();
    while not LQuery.Eof do
    begin
      LList.Add(ReadTurnFromQuery(LQuery));
      LQuery.Next();
    end;
    LQuery.Close();

    SetLength(Result, LList.Count);
    for LI := 0 to LList.Count - 1 do
      Result[LI] := LList[LList.Count - 1 - LI];
  finally
    LQuery.Free();
    LList.Free();
  end;
end;

function TVdxMemory.SanitizeFTSQuery(const AQuery: string): string;
var
  LCleaned: string;
  LI: Integer;
  LCh: Char;
  LTokens: TArray<string>;
  LOut: TStringBuilder;
  LTok: string;
begin
  // Phase 1 — replace every FTS5 operator character with a space.
  // Build into a like-sized buffer to avoid quadratic concatenation.
  SetLength(LCleaned, Length(AQuery));
  for LI := 1 to Length(AQuery) do
  begin
    LCh := AQuery[LI];
    if CharInSet(LCh, CVdxFTSOperatorSet) then
      LCleaned[LI] := ' '
    else
      LCleaned[LI] := LCh;
  end;

  // Phase 2 — split on any whitespace, drop empties, rejoin with ' OR '.
  // OR semantics favour recall for conversational retrieval; BM25 then
  // ranks the candidate set so the best hits surface first.
  LTokens := LCleaned.Split([' ', #9, #10, #13],
    TStringSplitOptions.ExcludeEmpty);

  if Length(LTokens) = 0 then
    Exit('');

  LOut := TStringBuilder.Create();
  try
    for LI := 0 to High(LTokens) do
    begin
      LTok := Trim(LTokens[LI]);
      if LTok = '' then
        Continue;
      if LOut.Length > 0 then
        LOut.Append(' OR ');
      LOut.Append(LTok);
    end;
    Result := LOut.ToString();
  finally
    LOut.Free();
  end;
end;

function TVdxMemory.SearchFTS5(const AQuery: string;
  const ATopK: Integer): TArray<TVdxMemoryTurn>;
var
  LQuery: TFDQuery;
  LList: TList<TVdxMemoryTurn>;
  LMatch: string;
  LTurn: TVdxMemoryTurn;
begin
  Result := nil;
  if not IsOpen() then
    raise Exception.Create('TVdxMemory.SearchFTS5: session not open');
  if ATopK <= 0 then
    Exit;

  LMatch := SanitizeFTSQuery(AQuery);
  // All operator chars / whitespace — no usable tokens left. Return
  // empty rather than let FTS5 throw a syntax error on empty MATCH.
  if LMatch = '' then
    Exit;

  LList := TList<TVdxMemoryTurn>.Create();
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    // bm25() returns a scalar where lower = more relevant. ORDER BY
    // ascending so the best matches come first in the result set.
    LQuery.SQL.Text :=
      'SELECT turns.turn_id, turns.turn_index, turns.role, turns.text, ' +
      '       turns.token_count, turns.created_at, ' +
      '       bm25(turns_fts) AS score ' +
      'FROM turns_fts ' +
      'JOIN turns ON turns.turn_id = turns_fts.rowid ' +
      'WHERE turns_fts MATCH :q ' +
      'ORDER BY bm25(turns_fts) ' +
      'LIMIT :lim';
    LQuery.ParamByName('q').AsString    := LMatch;
    LQuery.ParamByName('lim').AsInteger := ATopK;
    LQuery.Open();

    while not LQuery.Eof do
    begin
      LTurn := ReadTurnFromQuery(LQuery);
      LTurn.Score := Single(LQuery.FieldByName('score').AsFloat);
      LList.Add(LTurn);
      LQuery.Next();
    end;
    LQuery.Close();

    Result := LList.ToArray();
  finally
    LQuery.Free();
    LList.Free();
  end;
end;

function TVdxMemory.SearchVector(const AQuery: string;
  const ATopK: Integer): TArray<TVdxMemoryTurn>;
var
  LQuery: TFDQuery;
  LList: TList<TVdxMemoryTurn>;
  LQueryVec: TArray<Single>;
  LBlob: TBytes;
  LRowVec: TArray<Single>;
  LTurn: TVdxMemoryTurn;
  LExpectedLen: Integer;
  LFieldStream: TStream;
  LI: Integer;
  LCount: Integer;
begin
  Result := nil;

  // Full guard set — strict at use time, mirrors AttachEmbeddings.
  // Checked separately so the error message tells the caller exactly
  // which precondition failed.
  if not IsOpen() then
    raise Exception.Create('TVdxMemory.SearchVector: session not open');
  if FEmbedder = nil then
    raise Exception.Create(
      'TVdxMemory.SearchVector: no embedder attached — ' +
      'call AttachEmbeddings first');
  if not FEmbedder.IsLoaded() then
    raise Exception.Create(
      'TVdxMemory.SearchVector: attached embedder is no longer loaded — ' +
      'call DetachEmbeddings before unloading the embedder');

  // Mirror SearchFTS5 semantics — a non-positive top-K returns empty
  // rather than raising. Keeps the two search APIs interchangeable
  // from a caller's perspective.
  if ATopK <= 0 then
    Exit;

  // Encode query with the query-side task prefix. EmbeddingGemma uses
  // asymmetric encoding — query vectors MUST be produced with
  // AIsQuery=True to score correctly against stored document-side
  // vectors (AppendTurn uses False).
  LQueryVec := FEmbedder.Embed(AQuery, True);
  LExpectedLen := FEmbedderDim * SizeOf(Single);

  LList := TList<TVdxMemoryTurn>.Create();
  LQuery := TFDQuery.Create(nil);
  try
    LQuery.Connection := FConn;
    LQuery.SQL.Text :=
      'SELECT turn_id, turn_index, role, text, token_count, created_at, ' +
      '       embedding ' +
      'FROM turns ' +
      'WHERE embedding IS NOT NULL';
    LQuery.Open();

    while not LQuery.Eof do
    begin
      // Extract the BLOB bytes via a read-only blob stream. FireDAC
      // exposes BLOB fields as streams rather than via a direct byte-
      // array getter.
      LFieldStream := LQuery.CreateBlobStream(
        LQuery.FieldByName('embedding'), bmRead);
      try
        SetLength(LBlob, LFieldStream.Size);
        if LFieldStream.Size > 0 then
          LFieldStream.ReadBuffer(LBlob[0], LFieldStream.Size);
      finally
        LFieldStream.Free();
      end;

      // Dim validation — a byte-length mismatch means someone changed
      // embedders between AppendTurn calls. Raise loudly; silently
      // producing garbage similarity scores would be far worse than
      // surfacing the bug to the caller.
      if Length(LBlob) <> LExpectedLen then
        raise Exception.CreateFmt(
          'TVdxMemory.SearchVector: embedding byte length mismatch ' +
          '(got %d, expected %d for dim %d) — embedder dim changed mid-DB',
          [Length(LBlob), LExpectedLen, FEmbedderDim]);

      LRowVec := BytesToSingleArray(LBlob);

      LTurn := ReadTurnFromQuery(LQuery);
      LTurn.CosineScore := CosineDot(LQueryVec, LRowVec);
      LList.Add(LTurn);
      LQuery.Next();
    end;
    LQuery.Close();

    // Sort DESC on CosineScore — best match first. Inline comparer via
    // TComparer.Construct avoids hoisting a named compare function
    // for this one-off use.
    LList.Sort(TComparer<TVdxMemoryTurn>.Construct(
      function(const ALeft, ARight: TVdxMemoryTurn): Integer
      begin
        if ALeft.CosineScore > ARight.CosineScore then
          Result := -1
        else if ALeft.CosineScore < ARight.CosineScore then
          Result := 1
        else
          Result := 0;
      end));

    // Truncate to the requested top-K. If the DB has fewer matching
    // rows than ATopK, we return them all (Length(Result) <= ATopK).
    LCount := LList.Count;
    if LCount > ATopK then
      LCount := ATopK;
    SetLength(Result, LCount);
    for LI := 0 to LCount - 1 do
      Result[LI] := LList[LI];
  finally
    LQuery.Free();
    LList.Free();
  end;
end;

end.
