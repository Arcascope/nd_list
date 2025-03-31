import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:mic_stream/mic_stream.dart';
import 'package:nd_list/nd_list.dart';
import 'package:permission_handler/permission_handler.dart';

const int sampleRate = 48000; // Sample rate for the microphone
const int bufferSeconds = 30; // Buffer size in seconds
const int nFFT = 4096; // FFT size

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Mic Spectrogram',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const SpectrogramScreen(),
    );
  }
}

class SpectrogramScreen extends StatefulWidget {
  const SpectrogramScreen({super.key});

  @override
  _SpectrogramScreenState createState() => _SpectrogramScreenState();
}

class _SpectrogramScreenState extends State<SpectrogramScreen> {
  StreamSubscription<Uint8List>? _micStream;
  final NDList<double> _audioBuffer = NDList.filled(
      [sampleRate * bufferSeconds],
      0.0); // bufferSeconds seconds of audio at 48kHz
  final NDList<double> _spectrogramBuffer =
      NDList.filled([bufferSeconds, nFFT ~/ 2 + 1], 0.0);
  bool _isRecording = false;
  int _spectrogramIndex = 0;
  int _audioBufferIndex = 0;
  NDList<double>? _currentSpectrogram;
  Timer? _uiUpdateTimer;
  int _updateCounter = 0;

  @override
  void dispose() {
    _micStream?.cancel();
    _uiUpdateTimer?.cancel();
    super.dispose();
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      _micStream?.cancel();
      _uiUpdateTimer?.cancel();
      setState(() {
        _isRecording = false;
      });
      return;
    }

    final isMac = Platform.isMacOS;

    if (isMac || await Permission.microphone.request().isGranted) {
      final micStream = await MicStream.microphone(
        audioFormat: AudioFormat.ENCODING_PCM_16BIT,
      );
      if (micStream == null) {
        debugPrint('MicStream is not supported on this platform.');
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content:
                  Text('Microphone access is not supported on this platform')),
        );
        return;
      }

      // Set up a timer to update the UI at a reasonable rate (5 times per second)
      _uiUpdateTimer = Timer.periodic(const Duration(milliseconds: 200), (_) {
        if (_currentSpectrogram != null) {
          setState(() {
            // This triggers a rebuild with the current spectrogram
          });
        }
      });

      _micStream = micStream.listen((data) {
        try {
          // Convert Uint8List to doubles and append to buffer
          final samples = Int16List.view(data.buffer)
              .map((e) => e.toDouble() / (1 << 15)) // Normalize to [-1.0, 1.0]
              .toList();

          // Update the audio buffer
          final endIndex = _audioBufferIndex + samples.length;
          if (endIndex <= _audioBuffer.shape[0]) {
            _audioBuffer['$_audioBufferIndex:$endIndex'] =
                NDList.from<double>(samples);
          } else {
            // Handle wrap-around
            final splitIndex = _audioBuffer.shape[0] - _audioBufferIndex;
            _audioBuffer['$_audioBufferIndex:${_audioBuffer.shape[0]}'] =
                NDList.from<double>(samples.sublist(0, splitIndex));
            _audioBuffer['0:${samples.length - splitIndex}'] =
                NDList.from<double>(samples.sublist(splitIndex));
          }
          _audioBufferIndex = endIndex % _audioBuffer.shape[0];

          // Only process FFT periodically to avoid overloading
          _updateCounter++;
          if (_updateCounter % 4 != 0) return; // Process every 4th chunk

          // Compute spectrogram if enough data is available
          if (_audioBufferIndex >= nFFT || _audioBufferIndex == 0) {
            final startIndex =
                (_audioBufferIndex - nFFT) % _audioBuffer.shape[0];
            NDList<double> fftInput;
            if (startIndex >= 0) {
              fftInput = _audioBuffer['$startIndex:${startIndex + nFFT}'];
            } else {
              // Handle wrap-around for FFT input
              fftInput = NDList.stacked([
                _audioBuffer[
                    '${_audioBuffer.shape[0] + startIndex}:${_audioBuffer.shape[0]}'],
                _audioBuffer['0:${nFFT + startIndex}']
              ]);
            }

            final spectrogram = fftInput
                .spectrogram(nFFT, hopLength: nFFT ~/ 2)
                .slice(0, nFFT ~/ 2 + 1, axis: 1);

            // Update the spectrogram buffer
            _spectrogramBuffer[_spectrogramIndex] = spectrogram;
            _spectrogramIndex =
                (_spectrogramIndex + 1) % _spectrogramBuffer.shape[0];

            // Update the reference to current spectrogram - UI update happens in timer
            _currentSpectrogram = _spectrogramBuffer;
            debugPrint('Updated spectrogram buffer index: $_spectrogramIndex');
          }
        } catch (e, stackTrace) {
          debugPrint('Error in micStream.listen: $e');
          debugPrint('Stack trace: $stackTrace');
        }
      }, onError: (error) {
        debugPrint('Error from mic stream: $error');
      });

      setState(() {
        _isRecording = true;
      });
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Microphone permission denied')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Mic Spectrogram')),
      body: Column(
        children: [
          Expanded(
            child: _currentSpectrogram == null
                ? const Center(child: Text('No data'))
                : CustomPaint(
                    painter: SpectrogramPainter(_currentSpectrogram!),
                    child: Container(),
                  ),
          ),
          ElevatedButton(
            onPressed: _toggleRecording,
            child: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
          ),
        ],
      ),
    );
  }
}

class SpectrogramPainter extends CustomPainter {
  final NDList<double> spectrogram;

  SpectrogramPainter(this.spectrogram);

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint();
    final rows = spectrogram.shape[0];
    final cols = spectrogram.shape[1];
    final cellWidth = size.width / cols;
    final cellHeight = size.height / rows;

    // Draw fewer cells for better performance
    final skipFactor =
        (cols > 500) ? 2 : 1; // Skip cells for large spectrograms

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j += skipFactor) {
        final value = spectrogram[[i, j]].item ?? 0.0;
        // Use log scale to better visualize intensity
        final scaledValue = value > 0 ? (0.3 * (1 + log(value))) : 0.0;
        final intensity = scaledValue.clamp(0.0, 1.0);
        paint.color = Color.fromRGBO(0, 0, (intensity * 255).toInt(), 1.0);
        canvas.drawRect(
          Rect.fromLTWH(j * cellWidth, i * cellHeight, cellWidth * skipFactor,
              cellHeight),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant SpectrogramPainter oldDelegate) {
    // Only repaint if the spectrogram data has changed
    return true; // For now, but you could implement comparison
  }
}
