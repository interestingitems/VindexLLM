{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.TokenWriter;

{$I VindexLLM.Defines.inc}

interface

uses
  VindexLLM.Utils;

type

  { TVdxTokenWriter }
  TVdxTokenWriter = class(TVdxBaseObject)
  private
    FMaxWidth: Integer;      // configurable line width (columns)
    FColumn: Integer;        // current column position (0-based)
    FWordBuffer: string;     // accumulates the current in-progress word
    FWordStartCol: Integer;  // column where the current word started
  protected
    // Override these in subclasses for target-specific I/O
    procedure DoWrite(const AText: string); virtual; abstract;
    procedure DoEraseBack(const ACount: Integer); virtual; abstract;
    procedure DoNewLine(); virtual; abstract;
    // Flush the pending word buffer via DoWrite, then clear it
    procedure FlushWord();
  public
    constructor Create(); override;
    // Main entry point — call from the token callback
    procedure Write(const AToken: string);
    // Clear state for a new generation run
    procedure Reset();
    // Line width in columns (default 80)
    property MaxWidth: Integer read FMaxWidth write FMaxWidth;
  end;

  { TVdxConsoleTokenWriter }
  TVdxConsoleTokenWriter = class(TVdxTokenWriter)
  protected
    procedure DoWrite(const AText: string); override;
    procedure DoEraseBack(const ACount: Integer); override;
    procedure DoNewLine(); override;
  end;

implementation

uses
  System.SysUtils;

{ TVdxTokenWriter }
constructor TVdxTokenWriter.Create();
begin
  inherited Create();
  FMaxWidth := 80;
  FColumn := 0;
  FWordBuffer := '';
  FWordStartCol := 0;
end;

procedure TVdxTokenWriter.FlushWord();
begin
  // Characters were already emitted one-by-one via DoWrite(LCh) in Write().
  // Just clear the buffer — no re-emit needed.
  FWordBuffer := '';
end;

procedure TVdxTokenWriter.Reset();
begin
  FColumn := 0;
  FWordBuffer := '';
  FWordStartCol := 0;
end;

procedure TVdxTokenWriter.Write(const AToken: string);
var
  LI: Integer;
  LCh: Char;
begin
  for LI := 1 to Length(AToken) do
  begin
    LCh := AToken[LI];

    // --- Newline: flush current word, move to next line ---
    if LCh = #10 then
    begin
      FlushWord();
      DoNewLine();
      FColumn := 0;
      FWordStartCol := 0;
      Continue;
    end;

    // --- Space: flush current word, emit the space ---
    if LCh = ' ' then
    begin
      FlushWord();
      DoWrite(' ');
      FColumn := FColumn + 1;
      // Next word will start at this column
      FWordStartCol := FColumn;
      Continue;
    end;

    // --- Printable character: accumulate in word buffer ---
    FWordBuffer := FWordBuffer + LCh;
    DoWrite(LCh);
    FColumn := FColumn + 1;

    // Check if we've hit the margin mid-word
    if FColumn >= FMaxWidth then
    begin
      // Edge case: if the single word is longer than the full line width,
      // just let it overflow — don't loop infinitely trying to wrap it.
      if Length(FWordBuffer) >= FMaxWidth then
        Continue;

      // Backtrack: erase the partial word we've been printing on this line
      DoEraseBack(Length(FWordBuffer));

      // Move to a fresh line and re-emit the whole word there
      DoNewLine();
      FColumn := 0;
      DoWrite(FWordBuffer);
      FColumn := Length(FWordBuffer);
      FWordStartCol := 0;
    end;
  end;
end;

{ TVdxConsoleTokenWriter }
procedure TVdxConsoleTokenWriter.DoWrite(const AText: string);
begin
  TVdxUtils.Print(AText);
end;

procedure TVdxConsoleTokenWriter.DoEraseBack(const ACount: Integer);
var
  LI: Integer;
begin
  // Backspace over the characters, overwrite with spaces, backspace again
  for LI := 1 to ACount do
    TVdxUtils.Print(#8);
  for LI := 1 to ACount do
    TVdxUtils.Print(' ');
  for LI := 1 to ACount do
    TVdxUtils.Print(#8);
end;

procedure TVdxConsoleTokenWriter.DoNewLine();
begin
  TVdxUtils.PrintLn();
end;

end.
