"""Background Option+Z voice demo with a microphone-driven notch HUD."""
import argparse
from pathlib import Path
import math
import sys
import time

import AppKit as A
import Quartz as Q
import objc
from Foundation import NSTimer
import app as window_app
from app import Demo, ROOT
from hotkey_state import HoldShortcut


class WaveView(A.NSView):
    def initWithFrame_(self,frame):
        self=objc.super(WaveView,self).initWithFrame_(frame)
        if self is not None:
            self.level=0.0;self.message='Hold ⌥Z to ask';self.phase=0.0
        return self

    def drawRect_(self,rect):
        A.NSColor.colorWithCalibratedWhite_alpha_(.04,.96).setFill()
        A.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(self.bounds(),18,18).fill()
        attrs={A.NSFontAttributeName:A.NSFont.systemFontOfSize_(12),
               A.NSForegroundColorAttributeName:A.NSColor.whiteColor()}
        from Foundation import NSString
        NSString.stringWithString_(self.message[:90]).drawInRect_withAttributes_(((12,37),(296,18)),attrs)
        A.NSColor.systemTealColor().setFill()
        for index in range(34):
            amplitude=self.level*(.35+.65*abs(math.sin(index*.65+self.phase)))
            height=3+23*amplitude
            A.NSBezierPath.bezierPathWithRoundedRect_xRadius_yRadius_(
                ((14+index*8.6,18-height/2),(4,height)),2,2).fill()


class HotkeyDemo(Demo):
    def applicationDidFinishLaunching_(self,notification):
        self.hud=None;self.hud_until=0;self.shortcut=HoldShortcut();self.release_pending=False
        objc.super(HotkeyDemo,self).applicationDidFinishLaunching_(notification)
        self.window.orderOut_(None)
        self.tray.button().setTitle_('⌥Z')
        # Default event tap consumes the shortcut so FL Studio doesn't also receive Option+Z.
        mask=sum(1<<event for event in (Q.kCGEventKeyDown,Q.kCGEventKeyUp,Q.kCGEventFlagsChanged))
        self.tap_callback=self.handle_key
        self.tap=Q.CGEventTapCreate(Q.kCGSessionEventTap,Q.kCGHeadInsertEventTap,
                                  Q.kCGEventTapOptionDefault,mask,self.tap_callback,None)
        if self.tap is None:
            self.toast('Enable Accessibility / Input Monitoring; restart.',duration=15)
            print('Allow the launching app in System Settings → Privacy & Security → Accessibility and Input Monitoring, then restart.',flush=True)
        else:
            self.tap_source=Q.CFMachPortCreateRunLoopSource(None,self.tap,0)
            Q.CFRunLoopAddSource(Q.CFRunLoopGetMain(),self.tap_source,Q.kCFRunLoopCommonModes)
            Q.CGEventTapEnable(self.tap,True)
            self.toast('Hold ⌥Z · speak · release to locate',duration=5)

    @objc.python_method
    def handle_key(self,proxy,event_type,event,refcon):
        if event_type in (Q.kCGEventTapDisabledByTimeout,Q.kCGEventTapDisabledByUserInput):
            self.events.put((self.generation,{'event':'hotkey_cancel'}))
            self.shortcut.held=False;Q.CGEventTapEnable(self.tap,True)
            return event
        flags=Q.CGEventGetFlags(event)
        key=Q.CGEventGetIntegerValueField(event,Q.kCGKeyboardEventKeycode)
        kind={Q.kCGEventKeyDown:'down',Q.kCGEventKeyUp:'up',Q.kCGEventFlagsChanged:'flags'}.get(event_type)
        action=self.shortcut.event(kind,key,bool(flags & Q.kCGEventFlagMaskAlternate),
                                   bool(flags & (Q.kCGEventFlagMaskCommand|Q.kCGEventFlagMaskControl|Q.kCGEventFlagMaskShift)))
        if action=='cancel' and self.mode=='idle' and not self.marker:return event
        if action and action!='swallow':
            self.events.put((self.generation,{'event':'hotkey_'+action}))
        if action and event_type!=Q.kCGEventFlagsChanged:return None
        return event

    @objc.python_method
    def toast(self,text,duration=3):
        if self.hud is None:
            screen=A.NSScreen.mainScreen()
            frame=screen.frame();safe=screen.safeAreaInsets().top if hasattr(screen,'safeAreaInsets') else 0
            rect=((frame.origin.x+(frame.size.width-320)/2,frame.origin.y+frame.size.height-safe-68),(320,64))
            self.hud=A.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(rect,
                A.NSWindowStyleMaskBorderless|A.NSWindowStyleMaskNonactivatingPanel,A.NSBackingStoreBuffered,False)
            self.hud.setReleasedWhenClosed_(False);self.hud.setOpaque_(False)
            self.hud.setBackgroundColor_(A.NSColor.clearColor());self.hud.setIgnoresMouseEvents_(True)
            self.hud.setHidesOnDeactivate_(False);self.hud.setLevel_(A.NSScreenSaverWindowLevel)
            self.hud.setCollectionBehavior_(A.NSWindowCollectionBehaviorCanJoinAllSpaces|A.NSWindowCollectionBehaviorFullScreenAuxiliary)
            self.wave=WaveView.alloc().initWithFrame_(((0,0),(320,64)))
            self.hud.setContentView_(self.wave)
        self.wave.message=text;self.wave.setNeedsDisplay_(True)
        self.hud.orderFrontRegardless();self.hud_until=time.monotonic()+duration

    def show_(self,sender):
        # Override every inherited window-opening path; no foreground window in this UX.
        self.toast(self.status.stringValue() if hasattr(self,'status') else 'Hold ⌥Z to ask',duration=5)

    def statusText_(self,text):
        objc.super(HotkeyDemo,self).statusText_(text)
        if self.mode!='ground':self.toast(text,duration=5)

    def cancel_(self,sender):
        self.release_pending=False
        if hasattr(self,'waiting_text'):del self.waiting_text
        objc.super(HotkeyDemo,self).cancel_(sender)

    @objc.python_method
    def submit(self,text):
        if not text.strip():self.toast('No speech heard. Hold ⌥Z and try again.');return
        self.input.setStringValue_(text)
        # Hide all demo graphics before capture; no focus switch or countdown required.
        self.clear_marker()
        if self.hud:self.hud.orderOut_(None)
        self.hud_until=0
        self.mic.setEnabled_(False)
        self.spawn([sys.executable,str(ROOT/'demo_app/backend.py'),'--instruction='+text,
                    '--strategy','panel','--delay','0','--endpoint',window_app.OPTIONS.endpoint,
                    '--coordinates',window_app.OPTIONS.coordinates],'ground')
        self.tray.button().setTitle_('FL …')

    def tick_(self,timer):
        # Transform hotkey/voice events, then let the existing controller handle result/marker lifecycle.
        pending=[]
        while not self.events.empty():
            token,data=self.events.get()
            event=data.get('event','')
            if token!=self.generation and not event.startswith('hotkey_'):continue
            if event=='hotkey_start':
                if self.mode!='idle':self.cancel_(None)
                self.clear_marker();self.release_pending=False
                self.speak_(None)
                self.toast('Preparing microphone…',duration=30)
            elif event=='hotkey_release':
                if self.mode=='voice':
                    self.release_pending=True
                    self.speak_(None)
                    self.toast('Transcribing locally…',duration=30)
            elif event=='hotkey_cancel':self.cancel_(None)
            elif event=='level':
                if self.hud:self.wave.level=max(0,min(1,float(data.get('value',0))))
            elif event=='listening':self.toast('Listening · release ⌥Z to locate',duration=30)
            elif event=='partial':self.input.setStringValue_(data.get('text',''))
            elif event=='transcript':
                text=data.get('text','');submit=self.release_pending
                self.reset()
                if submit:
                    self.release_pending=False;self.submit(text)
                else:
                    self.input.setStringValue_(text)
                    self.toast('Recording limit reached. Release ⌥Z to submit.',duration=15)
                    self.waiting_text=text
            elif event=='working':
                self.toast('Finding: '+self.input.stringValue(),duration=180)
                pending.append((token,data))
            else:pending.append((token,data))
        # If max-duration ended recording before key-up, submit its completed transcript once released.
        if hasattr(self,'waiting_text') and not self.shortcut.held:
            text=self.waiting_text;del self.waiting_text
            self.submit(text)
        for item in pending:self.events.put(item)
        objc.super(HotkeyDemo,self).tick_(timer)
        if self.hud:
            self.wave.phase+=.4
            if self.mode!='voice':self.wave.level*=.8
            self.wave.setNeedsDisplay_(True)
            if self.hud_until and time.monotonic()>self.hud_until:self.hud.orderOut_(None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint',default='http://127.0.0.1:8080/v1')
    parser.add_argument('--coordinates',choices=['pixels','normalized_1000','normalized_1'],default='normalized_1000')
    window_app.OPTIONS=parser.parse_args()
    application=A.NSApplication.sharedApplication()
    application.setActivationPolicy_(A.NSApplicationActivationPolicyAccessory)
    delegate=HotkeyDemo.alloc().init();application.setDelegate_(delegate);application.run()
