program VdxTestbed;

{$APPTYPE CONSOLE}

{$R *.res}

uses
  System.SysUtils,
  UVdxTestbed in 'UVdxTestbed.pas',
  VindexLLM.Config in '..\src\VindexLLM.Config.pas',
  VindexLLM.GGUFReader in '..\src\VindexLLM.GGUFReader.pas',
  VindexLLM.Resources in '..\src\VindexLLM.Resources.pas',
  VindexLLM.TOML in '..\src\VindexLLM.TOML.pas',
  VindexLLM.Utils in '..\src\VindexLLM.Utils.pas',
  VindexLLM.VulkanCompute in '..\src\VindexLLM.VulkanCompute.pas';

begin
  RunVdxTestbed();
end.
