{===============================================================================
  VindexLLM™ - Liberating LLM inference

  Copyright © 2026-present tinyBigGAMES™ LLC
  All Rights Reserved.

  https://vindexllm.com

  See LICENSE for license information
===============================================================================}

unit VindexLLM.Resources;

{$I VindexLLM.Defines.inc}

interface

resourcestring

  //--------------------------------------------------------------------------
  // Severity Names
  //--------------------------------------------------------------------------
  RSSeverityHint    = 'Hint';
  RSSeverityWarning = 'Warning';
  RSSeverityError   = 'Error';
  RSSeverityFatal   = 'Fatal';
  RSSeverityNote    = 'Note';
  RSSeverityUnknown = 'Unknown';

  //--------------------------------------------------------------------------
  // Error Format Strings
  //--------------------------------------------------------------------------
  RSErrorFormatSimple              = '%s %s: %s';
  RSErrorFormatWithLocation        = '%s: %s %s: %s';
  RSErrorFormatRelatedSimple       = '  %s: %s';
  RSErrorFormatRelatedWithLocation = '  %s: %s: %s';

  //--------------------------------------------------------------------------
  // Fatal / I/O Messages
  //--------------------------------------------------------------------------
  RSFatalFileNotFound  = 'File not found: ''%s''';
  RSFatalFileReadError = 'Cannot read file ''%s'': %s';
  RSFatalInternalError = 'Internal error: %s';

  //--------------------------------------------------------------------------
  // Memory (TVdxMemory) Messages
  //--------------------------------------------------------------------------
  RSMemSessionNotOpen   = 'Memory session not open';
  RSMemDbPathEmpty      = 'Memory database path is empty';
  RSMemEmbedderNil      = 'Embedder is nil';
  RSMemEmbedderNotLoaded = 'Embedder is not loaded';
  RSMemEmbedderDetached = 'Attached embedder is no longer loaded — call DetachEmbeddings before unloading';
  RSMemNoEmbedder       = 'No embedder attached — call AttachEmbeddings first';
  RSMemEmbeddingMismatch = 'Embedding byte length mismatch (got %d, expected %d for dim %d)';
  RSMemWhereEmpty       = 'Empty WHERE clause — use PurgeAll instead';
  RSMemChunkInvalid     = 'AChunkTokens must be > 0';
  RSMemOverlapInvalid   = 'AOverlapTokens must be < AChunkTokens';

implementation

end.
