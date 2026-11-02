import Foundation
import AVFoundation
import Speech

func emit(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value), let text = String(data: data, encoding: .utf8) {
        print(text)
        fflush(stdout)
    }
}

final class Recorder {
    let engine = AVAudioEngine()
    let request = SFSpeechAudioBufferRecognitionRequest()
    var task: SFSpeechRecognitionTask?
    var done = false
    var stopping = false
    var transcript = ""
    var lastLevel = Date.distantPast

    func finish(_ error: String? = nil) {
        if done { return }
        done = true
        if engine.isRunning { engine.stop(); engine.inputNode.removeTap(onBus: 0) }
        request.endAudio()
        task?.cancel()
        if let error = error { emit(["event":"error", "message":error]) }
        else { emit(["event":"transcript", "text":transcript]) }
        exit(error == nil ? 0 : 1)
    }

    func stop() {
        if done || stopping { return }
        stopping = true
        if engine.isRunning { engine.stop(); engine.inputNode.removeTap(onBus: 0) }
        request.endAudio()
        DispatchQueue.main.asyncAfter(deadline: .now()+2) { self.finish() }
    }

    func start() {
        if done || stopping { finish(); return }
        guard let recognizer = SFSpeechRecognizer(locale: Locale(identifier: "en-US")),
              recognizer.isAvailable, recognizer.supportsOnDeviceRecognition else {
            finish("Local speech recognition is unavailable. Enable/download English dictation in macOS settings, or type your request."); return
        }
        request.requiresOnDeviceRecognition = true
        request.shouldReportPartialResults = true
        request.contextualStrings = ["FL Studio", "Channel Rack", "Piano roll", "Drums", "Snare", "Mixer", "volume", "pan"]
        let node = engine.inputNode
        let format = node.outputFormat(forBus: 0)
        guard format.sampleRate > 0, format.channelCount > 0 else {
            finish("No microphone input is available."); return
        }
        node.installTap(onBus: 0, bufferSize: 1024, format: format) { buffer, _ in
            self.request.append(buffer)
            if Date().timeIntervalSince(self.lastLevel) > 0.05, let samples = buffer.floatChannelData?[0] {
                self.lastLevel = Date()
                let count = Int(buffer.frameLength)
                if count > 0 {
                    var sum: Float = 0
                    for index in 0..<count { sum += samples[index] * samples[index] }
                    let level = min(1, sqrt(sum / Float(count)) * 10)
                    DispatchQueue.main.async { if !self.done { emit(["event":"level", "value":level]) } }
                }
            }
        }
        task = recognizer.recognitionTask(with: request) { result, error in
            DispatchQueue.main.async {
                if self.done { return }
                if let result = result {
                    self.transcript = result.bestTranscription.formattedString
                    emit(["event":"partial", "text":self.transcript])
                    if result.isFinal { self.finish(); return }
                }
                if error != nil {
                    if self.stopping && !self.transcript.isEmpty { self.finish() }
                    else { self.finish("Speech recognition stopped. Try again or type your request.") }
                }
            }
        }
        do { engine.prepare(); try engine.start() }
        catch { finish("Could not start microphone capture."); return }
        emit(["event":"listening"])
        DispatchQueue.main.asyncAfter(deadline: .now()+20) { self.stop() }
    }
}

// Direct subprocess launches must carry privacy descriptions in the Mach-O image.
for key in ["NSSpeechRecognitionUsageDescription", "NSMicrophoneUsageDescription"] {
    guard let description = Bundle.main.object(forInfoDictionaryKey: key) as? String,
          !description.isEmpty else {
        emit(["event":"error", "message":"Voice helper is missing privacy metadata. Rebuild with python3 demo_app/build.py and restart."])
        exit(1)
    }
}
if CommandLine.arguments.contains("--check-metadata") {
    emit(["event":"metadata_ok"])
    exit(0)
}
let recorder = Recorder()
DispatchQueue.global().async {
    _ = readLine()
    DispatchQueue.main.async { recorder.stop() }
}
SFSpeechRecognizer.requestAuthorization { status in
    guard status == .authorized else {
        DispatchQueue.main.async { recorder.finish("Allow Speech Recognition in System Settings, or type your request.") }
        return
    }
    AVCaptureDevice.requestAccess(for: .audio) { allowed in
        DispatchQueue.main.async {
            if allowed { recorder.start() }
            else { recorder.finish("Allow Microphone access in System Settings, or type your request.") }
        }
    }
}
RunLoop.main.run()
