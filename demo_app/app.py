"""Native, standalone voice-to-control frontend. No changes to the harness required."""
import argparse
import io
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import AppKit as A
import objc
from Foundation import NSObject, NSTimer, NSData
from PIL import Image, ImageDraw
from flground.contracts import Point
from flground.overlay_macos import marker_frame


def label(text,x,y,w,h,size=14,color=None):
    field=A.NSTextField.labelWithString_(text)
    field.setFrame_(((x,y),(w,h)))
    field.setFont_(A.NSFont.systemFontOfSize_(size))
    field.setTextColor_(color or A.NSColor.labelColor())
    return field


def button(title,x,y,w,action,target):
    control=A.NSButton.alloc().initWithFrame_(((x,y),(w,36)))
    control.setTitle_(title)
    control.setBezelStyle_(A.NSBezelStyleRounded)
    control.setTarget_(target);control.setAction_(action)
    return control


class Demo(NSObject):
    def applicationDidFinishLaunching_(self, notification):
        self.events=queue.Queue();self.process=None;self.generation=0;self.mode='idle'
        self.report=None;self.marker=None;self.marker_deadline=0
        self.window=A.NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            ((0,0),(640,480)),A.NSWindowStyleMaskTitled|A.NSWindowStyleMaskClosable|A.NSWindowStyleMaskMiniaturizable,
            A.NSBackingStoreBuffered,False)
        self.window.setTitle_('FLGround · Ask your workspace')
        self.window.setReleasedWhenClosed_(False)
        self.window.setBackgroundColor_(A.NSColor.colorWithCalibratedRed_green_blue_alpha_(.075,.085,.105,1))
        self.window.setAppearance_(A.NSAppearance.appearanceNamed_(A.NSAppearanceNameDarkAqua))
        view=self.window.contentView()
        for field in [label('ASK YOUR WORKSPACE',28,424,580,22,12,A.NSColor.systemTealColor()),
                      label('Find your next control.',28,376,580,40,29),
                      label('Speak or type a request. We’ll point it out in FL Studio.',28,343,580,24),
                      label('LOCAL GEMMA 4  •  LOCAL VOICE  •  NO AUTO-CLICKS',28,311,580,22,11,A.NSColor.secondaryLabelColor())]:
            view.addSubview_(field)
        self.input=A.NSTextField.alloc().initWithFrame_(((28,251),(584,44)))
        self.input.setFont_(A.NSFont.systemFontOfSize_(18))
        self.input.setPlaceholderString_('Where is the Drums channel button?')
        view.addSubview_(self.input)
        self.mic=button('●  Speak',28,201,120,'speak:',self);view.addSubview_(self.mic)
        self.find=button('Show me  →',452,201,160,'find:',self);view.addSubview_(self.find)
        self.cancel=button('Cancel',324,201,120,'cancel:',self);view.addSubview_(self.cancel)
        self.status=label('Ready. Keep FL Studio on one display.',28,141,584,48,14)
        view.addSubview_(self.status)
        self.details=label('Audio stays on this Mac. Screenshots go only to your local llama-server.\nReview the transcript, then Show me. You have 5 seconds to switch windows.',28,79,584,48,12,A.NSColor.secondaryLabelColor())
        view.addSubview_(self.details)
        self.report_button=button('Open last report',28,23,170,'report:',self)
        self.report_button.setEnabled_(False);view.addSubview_(self.report_button)
        self.strategy=A.NSPopUpButton.alloc().initWithFrame_pullsDown_(((372,24),(240,32)),False)
        self.strategy.addItemsWithTitles_(['Panel harness','Baseline','Panel + verification'])
        view.addSubview_(self.strategy)
        menu=A.NSMenu.alloc().initWithTitle_('FLGround')
        for title,action,key in [('Ask your workspace','show:',''),('Cancel request','cancel:',''),('Quit','quit:','q')]:
            item=A.NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title,action,key)
            item.setTarget_(self);menu.addItem_(item)
        self.tray=A.NSStatusBar.systemStatusBar().statusItemWithLength_(A.NSVariableStatusItemLength)
        self.tray.button().setTitle_('FL ◎');self.tray.setMenu_(menu)
        self.timer=NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(.1,self,'tick:',None,True)
        self.window.center();self.show_(None)

    def show_(self,sender):
        self.window.makeKeyAndOrderFront_(None)
        A.NSApplication.sharedApplication().activateIgnoringOtherApps_(True)

    def statusText_(self,text):
        self.status.setStringValue_(text)
        self.tray.button().setToolTip_(text)

    @objc.python_method
    def spawn(self,args,mode):
        self.generation+=1;token=self.generation;self.mode=mode
        self.find.setEnabled_(False);self.strategy.setEnabled_(False)
        logs=ROOT/'artifacts/demo-build/logs'
        logs.mkdir(parents=True,exist_ok=True)
        if mode=='voice':
            from voice_process import VoiceProcess
            self.process=VoiceProcess(args[0],logs,OPTIONS.endpoint)
            log_path=self.process.log
        else:
            log_path=logs/f'{mode}-{time.time_ns()}.log'
            with log_path.open('w') as error_log:
                self.process=subprocess.Popen(args,cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                              stderr=error_log,text=True,bufsize=1)
        print(f'{mode} helper log: {log_path}',flush=True)
        process=self.process
        def read():
            for line in process.stdout:
                try:self.events.put((token,json.loads(line)))
                except json.JSONDecodeError:pass
            code=process.wait()
            self.events.put((token,{'event':'ended','code':code,'log':str(log_path)}))
        threading.Thread(target=read,daemon=True).start()

    def speak_(self,sender):
        if self.mode=='voice':
            if self.process and self.process.poll() is None:
                try:self.process.stdin.write('\n');self.process.stdin.flush()
                except (BrokenPipeError,OSError):pass
            self.mic.setEnabled_(False);self.statusText_('Finishing local transcription…');return
        if self.mode!='idle':return
        helper=ROOT/'artifacts/demo-build/FLGround Voice.app/Contents/MacOS/FLGroundVoice'
        if not helper.exists():
            self.statusText_('Voice helper is not built. Run .venv/bin/python demo_app/build.py, or type a request.');return
        self.input.setStringValue_('')
        self.statusText_('Preparing local microphone… Allow access if macOS asks.')
        self.mic.setTitle_('■  Stop')
        self.spawn([str(helper)],'voice')

    def find_(self,sender):
        if self.mode!='idle':return
        instruction=self.input.stringValue().strip()
        if not instruction or len(instruction)>500:
            self.statusText_('Enter a request between 1 and 500 characters.');return
        self.clear_marker()
        strategy=['panel','baseline','panel_verify'][self.strategy.indexOfSelectedItem()]
        self.statusText_('Switch to FL Studio now. Capture starts in 5 seconds.')
        self.mic.setEnabled_(False)
        self.spawn([sys.executable,str(ROOT/'demo_app/backend.py'),'--instruction',instruction,
                    '--strategy',strategy,'--endpoint',OPTIONS.endpoint,'--coordinates',OPTIONS.coordinates],'ground')
        self.window.orderOut_(None)
        A.NSApplication.sharedApplication().hide_(None)

    @objc.python_method
    def reset(self):
        self.mode='idle';self.find.setEnabled_(True);self.strategy.setEnabled_(True)
        self.mic.setEnabled_(True);self.mic.setTitle_('●  Speak');self.tray.button().setTitle_('FL ◎')

    def cancel_(self,sender):
        self.generation+=1
        if self.process and self.process.poll() is None:self.process.terminate()
        self.clear_marker();self.reset();self.statusText_('Cancelled. No click was performed.')
        # The local server may finish an already submitted request after cancellation.

    def quit_(self,sender):
        self.cancel_(sender)
        A.NSApplication.sharedApplication().terminate_(None)

    def report_(self,sender):
        if self.report:
            from Foundation import NSURL
            A.NSWorkspace.sharedWorkspace().openURL_(NSURL.fileURLWithPath_(self.report))

    @objc.python_method
    def clear_marker(self):
        if self.marker:
            self.marker.orderOut_(None);self.marker.close();self.marker=None

    @objc.python_method
    def present_marker(self,data):
        front=A.NSWorkspace.sharedWorkspace().frontmostApplication()
        if not front or not front.localizedName().lower().startswith('fl studio'):
            self.statusText_('FL Studio is no longer in front. Open the report or try again.');self.show_(None);return
        screens=A.NSScreen.screens()
        size=data['desktop_size']
        if len(screens)!=1 or tuple(screens[0].frame().size)!=tuple(size):
            self.statusText_('Display geometry changed. Please try again.');self.show_(None);return
        frame=marker_frame(Point(**data['point']),tuple(data['screenshot_size']),tuple(size))
        image=Image.new('RGBA',(160,160));draw=ImageDraw.Draw(image)
        for color,width in [('black',10),('#00eeff',5)]:
            draw.ellipse((36,36,124,124),outline=color,width=width)
            for line in [(80,12,80,46),(80,114,80,148),(12,80,46,80),(114,80,148,80)]:draw.line(line,fill=color,width=width)
        buffer=io.BytesIO();image.save(buffer,format='PNG');raw=buffer.getvalue()
        native=A.NSImage.alloc().initWithData_(NSData.dataWithBytes_length_(raw,len(raw)))
        panel=A.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(frame,
            A.NSWindowStyleMaskBorderless|A.NSWindowStyleMaskNonactivatingPanel,A.NSBackingStoreBuffered,False)
        panel.setReleasedWhenClosed_(False);panel.setOpaque_(False);panel.setHasShadow_(False)
        panel.setBackgroundColor_(A.NSColor.clearColor());panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False);panel.setLevel_(A.NSScreenSaverWindowLevel)
        panel.setCollectionBehavior_(A.NSWindowCollectionBehaviorCanJoinAllSpaces|A.NSWindowCollectionBehaviorFullScreenAuxiliary)
        view=A.NSImageView.alloc().initWithFrame_(((0,0),(80,80)))
        view.setImage_(native);view.setImageScaling_(A.NSImageScaleAxesIndependently)
        panel.setContentView_(view);panel.orderFrontRegardless()
        self.marker=panel;self.marker_deadline=time.monotonic()+10
        self.tray.button().setTitle_('FL ◉')

    def tick_(self,timer):
        if self.marker and time.monotonic()>self.marker_deadline:
            self.clear_marker();self.tray.button().setTitle_('FL ◎')
        while not self.events.empty():
            token,data=self.events.get()
            if token!=self.generation:continue
            event=data.get('event')
            if event in ('partial','transcript'):
                self.input.setStringValue_(data.get('text',''))
                if event=='transcript':
                    self.reset();self.statusText_('Review what I heard, then choose Show me.' if data.get('text') else 'No speech heard. Try again or type.')
            elif event=='listening':self.statusText_('Listening locally… Press Stop when finished (20-second limit).')
            elif event=='countdown':self.tray.button().setTitle_(f'FL {data["seconds"]}')
            elif event=='working':self.tray.button().setTitle_('FL …');self.statusText_(data['message'])
            elif event=='result':
                self.report=data['report'];self.report_button.setEnabled_(True);self.reset()
                if data['point']:
                    self.statusText_(f'Located in {data["seconds"]}s · {data["calls"]} model calls. Marker lasts 10 seconds.')
                    self.present_marker(data)
                else:
                    self.statusText_('I couldn’t identify that control. Try its visible name or open the report.');self.show_(None)
            elif event=='error':
                self.reset();self.statusText_(data['message']);self.show_(None)
            elif event=='ended' and self.mode!='idle':
                self.reset();self.statusText_(f"Helper exited ({data.get('code')}). See terminal log path.");self.show_(None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8080/v1')
    parser.add_argument('--coordinates',choices=['pixels','normalized_1000','normalized_1'],default='normalized_1000')
    OPTIONS=parser.parse_args()
    application=A.NSApplication.sharedApplication()
    application.setActivationPolicy_(A.NSApplicationActivationPolicyRegular)
    delegate=Demo.alloc().init()
    application.setDelegate_(delegate)
    application.run()
