"""Build only the new voice helper; no package installs or tracked-file edits."""
import plistlib
from pathlib import Path
import subprocess

root=Path(__file__).resolve().parents[1]
bundle=root/'artifacts/demo-build/FLGround Voice.app'
contents=bundle/'Contents'
executable=contents/'MacOS/FLGroundVoice'
executable.parent.mkdir(parents=True,exist_ok=True)
with (contents/'Info.plist').open('wb') as stream:
    plistlib.dump({'CFBundleIdentifier':'local.flground.voice','CFBundleName':'FLGround Voice',
                  'CFBundleExecutable':'FLGroundVoice','CFBundlePackageType':'APPL',
                  'CFBundleVersion':'1','LSUIElement':True,
                  'NSMicrophoneUsageDescription':'Transcribe your spoken FL Studio request locally on this Mac.',
                  'NSSpeechRecognitionUsageDescription':'Use on-device speech recognition to fill the request field.'},stream)
cache=root/'artifacts/demo-build/module-cache'
cache.mkdir(exist_ok=True)
subprocess.run(['swiftc',str(root/'demo_app/Speech.swift'),'-o',str(executable),
                '-module-cache-path',str(cache),
                '-Xlinker','-sectcreate','-Xlinker','__TEXT','-Xlinker','__info_plist',
                '-Xlinker',str(contents/'Info.plist'),'-framework','AVFoundation','-framework','Speech'],check=True)
subprocess.run(['codesign','--force','--sign','-',str(bundle)],check=True)
print(f'Built local voice helper: {executable}')
