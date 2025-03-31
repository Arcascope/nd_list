import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:mic_stream/mic_stream.dart';
import 'package:nd_list/nd_list.dart';
import 'package:permission_handler/permission_handler.dart';

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
  List<double> _audioBuffer = [];
  bool _isRecording = false;
  NDList<double>? _spectrogram;

  @override
  void dispose() {
    _micStream?.cancel();
    super.dispose();
  }

  Future<void> _toggleRecording() async {
    if (_isRecording) {
      _micStream?.cancel();
      setState(() {
        _isRecording = false;
      });
      return;
    }

    if (await Permission.microphone.request().isGranted) {
      final micStream = await MicStream.microphone(
        audioFormat: AudioFormat.ENCODING_PCM_16BIT,
      );
      if (micStream == null) {
        debugPrint('MicStream is not supported on this platform.');
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Microphone access is not supported on macOS')),
        );
        return;
      }

      _micStream = micStream.listen((data) {
        // Convert Uint8List to doubles and append to buffer
        final samples = Int16List.view(data.buffer)
            .map((e) => e.toDouble() / 32768.0)
            .toList();
        _audioBuffer.addAll(samples);

        // Process buffer when it reaches ~0.5 seconds of data
        final sampleRate = 44100; // Default sample rate
        final nFFT = sampleRate ~/ 2; // ~0.5 seconds
        if (_audioBuffer.length >= nFFT) {
          final ndList = NDList.from<double>(_audioBuffer.sublist(0, nFFT));
          final spectrogram = ndList.spectrogram(nFFT);
          setState(() {
            _spectrogram = spectrogram;
          });
          _audioBuffer.removeRange(0, nFFT);
        }
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
            child: _spectrogram == null
                ? const Center(child: Text('No data'))
                : CustomPaint(
                    painter: SpectrogramPainter(_spectrogram!),
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

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        final value = spectrogram[i][j].item ?? 0.0;
        final color = Color.lerp(Colors.black, Colors.red, value)!;
        paint.color = color;
        canvas.drawRect(
          Rect.fromLTWH(j * cellWidth, i * cellHeight, cellWidth, cellHeight),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
