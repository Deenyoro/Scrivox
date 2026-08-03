; Inno Setup script for Scrivox (Windows installer).
; Built in CI after PyInstaller produces a onedir build in dist\<variant>.
; One script serves all three variants; pass /DVariant=Lite|Regular|Full and
; /DMyAppVersion=x.y.z on the ISCC command line.
;
; Besides placing the files and shortcuts, the installer registers Scrivox in
; the places integrations look for it (App Paths + the uninstall entry with
; InstallLocation), so tools like SimpleReliableRecorder auto-detect an
; installed Scrivox without any manual path setup.

#ifndef MyAppVersion
  #define MyAppVersion "0.0.0"
#endif
#ifndef Variant
  #define Variant "Regular"
#endif

#define MyAppPublisher "Deenyoro"
#define MyAppExe "Scrivox.exe"

#if Variant == "Lite"
  #define MyAppName "Scrivox Lite"
  #define MyDirName "Scrivox-Lite"
  ; AppId is an opaque identifier, not a real GUID. It is how Windows and Inno
  ; Setup recognise an existing install for upgrades/uninstall, so it must
  ; NEVER change across releases. Each variant has its own so they can coexist.
  #define MyAppId "{{4C7E9B2D-1F6A-4E83-B5D0-SCRIVOXLIT001}"
#elif Variant == "Full"
  #define MyAppName "Scrivox Full"
  #define MyDirName "Scrivox-Full"
  #define MyAppId "{{4C7E9B2D-1F6A-4E83-B5D0-SCRIVOXFUL001}"
#else
  #define MyAppName "Scrivox"
  #define MyDirName "Scrivox"
  #define MyAppId "{{4C7E9B2D-1F6A-4E83-B5D0-SCRIVOXREG001}"
#endif

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyDirName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=installer_out
OutputBaseFilename={#MyDirName}-{#MyAppVersion}-win64-setup
; lzma2/fast: the Full variant carries ~2 GB of CUDA DLLs and models; max
; compression there costs tens of CI minutes for a few percent of size.
Compression=lzma2/fast
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExe}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "..\dist\{#MyDirName}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Registry]
; App Paths makes "Scrivox" runnable from Run/cmd and is the first registry
; location integrations probe. HKA = HKLM for admin installs, HKCU otherwise.
Root: HKA; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExe}"; ValueType: string; ValueData: "{app}\{#MyAppExe}"; Flags: uninsdeletekey
Root: HKA; Subkey: "SOFTWARE\Microsoft\Windows\CurrentVersion\App Paths\{#MyAppExe}"; ValueType: string; ValueName: "Path"; ValueData: "{app}"; Flags: uninsdeletekey

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExe}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Nothing beyond what was installed: .env, config and models the user added
; next to the exe survive an uninstall on purpose.
