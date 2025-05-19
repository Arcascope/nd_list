import 'dart:async';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:mic_stream/mic_stream.dart';
import 'package:nd_list/nd_list.dart';
import 'package:permission_handler/permission_handler.dart';
import 'dart:ui' as ui;

const int sampleRate = 48000; // Sample rate for the microphone
const int bufferSeconds = 30; // Buffer size in seconds for spectrogram display
const int nFFT = 1024; // Reduced FFT size (was 4096)
const int hopLength = nFFT ~/ 2; // Hop length for FFT
const double redrawIntervalSeconds = 1; // Redraw every second
const int downsampleFactor = 2; // Average every 4 samples into 1

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
  final List<double> _audioBuffer = <double>[]; // Dynamic buffer
  final List<double> _downsampleBuffer = <double>[]; // Buffer for downsampling

  final NDList<double> _spectrogramBuffer =
      NDList.filled([bufferSeconds, nFFT ~/ 2 + 1], 0.0);
  bool _isRecording = false;
  int _spectrogramIndex = 0;
  NDList<double>? _currentSpectrogram;
  Timer? _uiUpdateTimer;
  double _secondsUntilRedraw = redrawIntervalSeconds;
  int _sampleCount = 0; // For downsampling

  @override
  void dispose() {
    _micStream?.cancel();
    _uiUpdateTimer?.cancel();
    super.dispose();
  }

  // New method for downsampling audio
  List<double> _downsampleAudio(List<double> samples) {
    final result = <double>[];

    // Add samples to temporary buffer
    _downsampleBuffer.addAll(samples);

    // Process complete blocks of downsampleFactor
    while (_downsampleBuffer.length >= downsampleFactor) {
      // Average the next downsampleFactor samples
      double sum = 0;
      for (int i = 0; i < downsampleFactor; i++) {
        sum += _downsampleBuffer[i];
      }
      result.add(sum / downsampleFactor);

      // Remove processed samples
      _downsampleBuffer.removeRange(0, downsampleFactor);
    }

    return result;
  }

  void _startRecording() async {
    if (_isRecording) return;

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

      // Clear the buffers
      _audioBuffer.clear();
      _downsampleBuffer.clear();
      _sampleCount = 0;

      // Set up a timer to update the UI at a set interval
      _secondsUntilRedraw = redrawIntervalSeconds;
      const tickerUpdateDuration = Duration(milliseconds: 100);
      _uiUpdateTimer = Timer.periodic(tickerUpdateDuration, (timer) {
        if (mounted) {
          setState(() {
            _secondsUntilRedraw = (redrawIntervalSeconds -
                    (tickerUpdateDuration * timer.tick).inMilliseconds /
                        1000.0) %
                redrawIntervalSeconds;
            // Force redraw regardless of countdown
            if (_currentSpectrogram != null && _secondsUntilRedraw <= 0) {
              // Force rebuild by creating a new painter instance
              _currentSpectrogram = _spectrogramBuffer;
            }
          });
        }
      });

      _micStream = micStream.listen((data) {
        try {
          // Convert Uint8List to doubles (normalized)
          final samples = Int16List.view(data.buffer)
              .map((e) => e.toDouble() / (1 << 15))
              .toList();

          // Downsample the audio
          final downsampledAudio = _downsampleAudio(samples);

          // Add downsampled samples to the buffer
          _audioBuffer.addAll(downsampledAudio);

          // Process FFT when we have enough samples
          while (_audioBuffer.length >= nFFT) {
            _processFFT();

            // Remove processed samples (hop length) from the buffer
            _audioBuffer.removeRange(0, hopLength);
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

  void _stopRecording() {
    if (!_isRecording) return;

    _micStream?.cancel();
    _uiUpdateTimer?.cancel();
    setState(() {
      _isRecording = false;
      _secondsUntilRedraw = redrawIntervalSeconds;
    });
  }

  void _processFFT() {
    if (_audioBuffer.length < nFFT) return;

    // Get a block of audio samples for FFT
    List<double> fftInput = _audioBuffer.sublist(0, nFFT);

    // Apply window function to reduce spectral leakage
    for (int i = 0; i < nFFT; i++) {
      // Hann window
      double window = 0.5 * (1 - cos((2 * pi * i) / (nFFT - 1)));
      fftInput[i] *= window;
    }

    // Compute the spectrogram
    final spectrogram = NDList.from<double>(fftInput)
        .spectrogram(nFFT, hopLength: hopLength)
        .slice(0, nFFT ~/ 2 + 1, axis: 1);

    // Update the spectrogram buffer
    _spectrogramBuffer[_spectrogramIndex] = spectrogram;

    // Create a new spectrogram with the updated row information
    _currentSpectrogram = _spectrogramBuffer;
    debugPrint('Processed FFT: Row $_spectrogramIndex updated');

    // Update index for next time
    _spectrogramIndex = (_spectrogramIndex + 1) % _spectrogramBuffer.shape[0];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Mic Spectrogram'),
        actions: [
          Center(
            child: Padding(
              padding: const EdgeInsets.only(right: 16.0),
              child: Text(
                'Next redraw: $_secondsUntilRedraw s',
                style: const TextStyle(fontWeight: FontWeight.bold),
              ),
            ),
          ),
        ],
      ),
      body: Column(
        children: [
          Expanded(
            child: _currentSpectrogram == null
                ? const Center(child: Text('No data - Press Start Recording'))
                : CustomPaint(
                    painter: SpectrogramPainter(_currentSpectrogram!),
                    child: Container(),
                  ),
          ),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton(
                  onPressed: _isRecording ? null : _startRecording,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.green,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 12),
                  ),
                  child: const Text('Start Recording'),
                ),
                ElevatedButton(
                  onPressed: _isRecording ? _stopRecording : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    padding: const EdgeInsets.symmetric(
                        horizontal: 24, vertical: 12),
                  ),
                  child: const Text('Stop Recording'),
                ),
              ],
            ),
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
    final rows = spectrogram.shape[0];
    final cols = spectrogram.shape[1];
    final cellWidth = size.width / cols;
    final cellHeight = size.height / rows;
    final paint = Paint();

    // Use a higher skip factor for better performance
    final skipFactor = (cols > 256) ? 8 : 4;

    // Calculate max value for normalization
    double specMax = 0.0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j += skipFactor) {
        final value = spectrogram[[i, j]].item ?? 0.0;
        if (value > specMax) specMax = value;
      }
    }

    // Avoid division by zero
    specMax = specMax > 0 ? specMax : 1.0;

    // Draw directly to canvas (no async operations)
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j += skipFactor) {
        final value = spectrogram[[i, j]].item ?? 0.0;
        final scaledValue = value / specMax; // Normalize to [0, 1]
        final intensity = scaledValue.clamp(0.0, 1.0);

        // Use a log-based color scale for better visibility of low energies
        final logIntensity = intensity > 0
            ? (0.3 + 0.7 * log(1 + 9 * intensity) / log(10)).clamp(0.0, 1.0)
            : 0.0;

        paint.color = Color.lerp(Colors.black, Colors.blue, logIntensity)!;

        canvas.drawRect(
          Rect.fromLTWH(j * cellWidth, i * cellHeight, cellWidth * skipFactor,
              cellHeight),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true; // Always repaint
  }
}
