# openWakeWord Android

Kotlin/Android port of [openWakeWord](https://github.com/dscripka/openWakeWord) — on-device wake word detection powered by ONNX Runtime.

Runs the full 3-stage inference pipeline (mel spectrogram → Google speech embedding → wake word classifier) entirely on-device. No network, no API key, no cloud dependency.

## Quick start

### 1. Add the dependency

**JitPack** (recommended):

```gradle
// settings.gradle.kts
dependencyResolutionManagement {
    repositories {
        maven { url = uri("https://jitpack.io") }
    }
}

// build.gradle.kts (app)
dependencies {
    implementation("com.github.msnilsen:openwakeword-android:0.1.0")
}
```

### 2. Use it

```kotlin
import com.openwakeword.OpenWakeWord

// Build a detector
val detector = OpenWakeWord.Builder(context)
    .setModel(OpenWakeWord.BuiltInModel.HEY_JARVIS)
    .setThreshold(0.5f)
    .build()

// Start listening (callback on main thread)
detector.start { score ->
    Log.d("WakeWord", "Detected! score=$score")
}

// Stop / release when done
detector.stop()
detector.release()
```

Requires `RECORD_AUDIO` permission — the library handles AudioRecord internally.

## Built-in models

| Model | Wake phrase | Asset size |
|-------|-----------|------------|
| `HEY_JARVIS` | "Hey Jarvis" | 1.3 MB |
| `ALEXA` | "Alexa" | 854 KB |
| `HEY_MYCROFT` | "Hey Mycroft" | 858 KB |

Core preprocessing models (mel spectrogram + embedding) add ~2.4 MB, shared across all wake word models.

## Custom models

You can train custom wake word models using the [openWakeWord training notebook](https://colab.research.google.com/drive/1q1oe2zOyZp7UsB3jJiQ1IFn8z5YfjwEb) and load them:

```kotlin
// From app assets
OpenWakeWord.Builder(context)
    .setModelAsset("my_custom_model.onnx")
    .build()

// From raw bytes
OpenWakeWord.Builder(context)
    .setModelBytes(modelByteArray)
    .build()
```

Custom models must be ONNX format, trained with the openWakeWord pipeline (input shape `[1, 16, 96]`, output shape `[1, 1]`).

## API reference

### `OpenWakeWord.Builder`

| Method | Description |
|--------|-------------|
| `setModel(BuiltInModel)` | Use a bundled pre-trained model (default: `HEY_JARVIS`) |
| `setModelAsset(String)` | Use a custom ONNX model from app assets |
| `setModelBytes(ByteArray)` | Use a custom ONNX model from raw bytes |
| `setThreshold(Float)` | Detection threshold 0.01–0.99 (default: 0.5). Lower = more sensitive. |
| `setDebounceMs(Long)` | Min ms between detections (default: 2000) |
| `build()` | Create the detector instance |

### `OpenWakeWord`

| Method | Description |
|--------|-------------|
| `start(OnDetectionListener)` | Start listening. Callback fires on main thread. |
| `stop()` | Stop listening. Can restart with `start()`. |
| `release()` | Release all resources. Instance cannot be reused. |

## How it works

This is a faithful Kotlin port of the [openWakeWord Python pipeline](https://github.com/dscripka/openWakeWord). The same ONNX models are used:

1. **Mel spectrogram** — `melspectrogram.onnx` converts 16 kHz PCM audio into mel-frequency spectrograms
2. **Speech embedding** — `embedding_model.onnx` (Google's [speech_embedding](https://tfhub.dev/google/speech_embedding/1)) converts spectrograms into 96-dimensional feature vectors
3. **Wake word classifier** — A small model outputs a 0–1 confidence score per 80 ms audio frame

Audio is captured via Android's `AudioRecord` at 16 kHz mono and processed on a dedicated background thread.

## Requirements

- Android API 26+ (Android 8.0)
- `RECORD_AUDIO` permission

## License

Code: [Apache License 2.0](LICENSE)

Pre-trained wake word models are from [openWakeWord](https://github.com/dscripka/openWakeWord) and licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

The speech embedding model is from [Google](https://tfhub.dev/google/speech_embedding/1) under [Apache License 2.0](https://opensource.org/licenses/Apache-2.0).
