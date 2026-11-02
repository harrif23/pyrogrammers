import Foundation
import AVFoundation

func argument(_ flag: String) -> String? {
    guard let i = CommandLine.arguments.firstIndex(of: flag), i+1 < CommandLine.arguments.count else { return nil }
    return CommandLine.arguments[i+1]
}
if let path = argument("--events-file") { freopen(path, "a", stdout) }
if let path = argument("--log-file") { freopen(path, "a", stderr) }
func emit(_ value: [String: Any]) {
    if let data = try? JSONSerialization.data(withJSONObject: value), let text = String(data:data, encoding:.utf8) {
        print(text); fflush(stdout)
    }
}
if CommandLine.arguments.contains("--check-metadata") {
    guard Bundle.main.object(forInfoDictionaryKey:"NSMicrophoneUsageDescription") != nil else { exit(1) }
    emit(["event":"metadata_ok"]); exit(0)
}
var recorder: AVAudioRecorder?
var finished = false
func stop() {
    if finished { return }; finished = true
    guard let current = recorder else { emit(["event":"error","message":"Recording stopped before microphone was ready."]); exit(1) }
    current.stop()
    emit(["event":"audio","path":current.url.path])
    exit(0)
}
AVCaptureDevice.requestAccess(for:.audio) { allowed in
    DispatchQueue.main.async {
        guard allowed, let path = argument("--audio-file") else {
            emit(["event":"error","message":"Allow microphone access and restart recording."]); exit(1)
        }
        do {
            let settings: [String:Any] = [AVFormatIDKey:kAudioFormatLinearPCM, AVSampleRateKey:16000,
                AVNumberOfChannelsKey:1, AVLinearPCMBitDepthKey:16, AVLinearPCMIsFloatKey:false,
                AVLinearPCMIsBigEndianKey:false]
            recorder = try AVAudioRecorder(url:URL(fileURLWithPath:path),settings:settings)
            recorder!.isMeteringEnabled = true
            guard recorder!.record() else { throw NSError(domain:"capture",code:1) }
            emit(["event":"listening"])
            Timer.scheduledTimer(withTimeInterval:0.05,repeats:true) { _ in
                recorder?.updateMeters()
                emit(["event":"level","value":min(1,pow(10,Double(recorder?.averagePower(forChannel:0) ?? -80)/20)*5)])
                if let path = argument("--stop-file"), FileManager.default.fileExists(atPath:path) { stop() }
            }
            DispatchQueue.main.asyncAfter(deadline:.now()+20) { stop() }
        } catch { emit(["event":"error","message":"Could not record microphone audio."]); exit(1) }
    }
}
RunLoop.main.run()
