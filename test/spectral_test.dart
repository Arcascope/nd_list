import 'dart:io';
import 'dart:math';
import 'package:nd_list/nd_list.dart';
import 'package:test/test.dart';

void main() {
  group('Test pure cosines', () {
    test('Single cosine', () {
      final nPeriods = 4;
      final nSamples = nPeriods * 16;
      NDList<double> time = NDList.from<double>([
        for (int i = 0; i < nSamples; i++)
          i.toDouble() / nSamples * nPeriods * 2 * pi
      ]);

      NDList<double> signal = time.map((t) => cos(t));

      final fft = signal.fft().map((e) => e.abs());

      final frequencies =
          List.generate(nSamples, (i) => i.toDouble() / nSamples);

      // Create data for SyncFusion chart
      final List<Map<String, double>> fftData = List.generate(nSamples,
          (i) => {'frequency': frequencies[i], 'magnitude': fft.list[i]});

      // Note: You can't directly show the chart in a test
      // Instead, you could save the widget for later use in a Flutter app
      // or export the data to a file
      print(
          'FFT data prepared for visualization with ${fftData.length} points');
      // If you need to visualize this in a Flutter app, use:
      // FFTChartWidget(fftData: fftData, title: 'FFT Magnitude Spectrum')

      // Alternatively, you could save the data to a CSV file
      final csvData =
          fftData.map((d) => '${d["frequency"]},${d["magnitude"]}').join('\n');
      File('fft_data.csv').writeAsStringSync('frequency,magnitude\n$csvData');

      expect(fft.shape, equals([nSamples]));
      print(fft);

      final fft_over_1 = fft.list.where((e) => e > 1.0).length;
      expect(fft_over_1, lessThan(3));
      expect(fft_over_1, greaterThan(0));

      final nFFT = 32;
      final specgram = signal.spectrogram(nFFT);

      expect(specgram.shape, equals([nSamples - nFFT + 1, nFFT]));
    });
  });
  group('FFT of simple signals', () {
    test('DC signal (constant)', () {
      final signal = NDList.from<double>(List.filled(64, 1.0));
      final fft = signal.fft();

      // DC component should be the sum of all values, rest should be zero
      expect(fft.list[0].real, closeTo(64.0, 1e-10));
      for (int i = 1; i < fft.list.length; i++) {
        expect(fft.list[i].abs(), lessThan(1e-10));
      }
    });

    test('Pure cosine at fundamental frequency', () {
      final n = 64;
      final signal = NDList.from<double>(
          [for (int i = 0; i < n; i++) cos(2 * pi * i / n)]);

      final fft = signal.fft();

      // For cosine, we expect peaks at frequencies 1 and n-1 (symmetric)
      expect(fft.list[1].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[n - 1].abs(), closeTo(n / 2, 0.1));

      // Other frequencies should have negligible magnitude
      for (int i = 2; i < n ~/ 2; i++) {
        expect(fft.list[i].abs(), lessThan(0.1));
      }
    });

    test('Pure sine at fundamental frequency', () {
      final n = 64;
      final signal = NDList.from<double>(
          [for (int i = 0; i < n; i++) sin(2 * pi * i / n)]);

      final fft = signal.fft();

      // For sine, we expect peaks at frequencies 1 and n-1 with imaginary component
      expect(fft.list[1].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[n - 1].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[1].argument().abs(), closeTo(pi / 2, 0.1)); // Phase check
    });

    test('Higher harmonic cosines', () {
      final n = 64;
      for (int harmonic = 2; harmonic <= 8; harmonic++) {
        final signal = NDList.from<double>(
            [for (int i = 0; i < n; i++) cos(2 * pi * harmonic * i / n)]);

        final fft = signal.fft();

        // Check peak at expected harmonic frequency
        expect(fft.list[harmonic].abs(), closeTo(n / 2, 0.1),
            reason: 'Failed for harmonic $harmonic');
        expect(fft.list[n - harmonic].abs(), closeTo(n / 2, 0.1),
            reason: 'Failed for harmonic $harmonic');
      }
    });

    test('Sum of cosines at different harmonics', () {
      final n = 64;
      final signal = NDList.from<double>([
        for (int i = 0; i < n; i++)
          cos(2 * pi * 2 * i / n) + 0.5 * cos(2 * pi * 4 * i / n)
      ]);

      final fft = signal.fft();

      // Check peaks at harmonics 2 and 4
      expect(fft.list[2].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[4].abs(), closeTo(n / 4, 0.1)); // Half amplitude
      expect(fft.list[n - 2].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[n - 4].abs(), closeTo(n / 4, 0.1));
    });

    test('Mixed cosine and sine signals', () {
      final n = 64;
      final signal = NDList.from<double>([
        for (int i = 0; i < n; i++)
          cos(2 * pi * 3 * i / n) + sin(2 * pi * 5 * i / n)
      ]);

      final fft = signal.fft();

      // Check peaks at harmonics 3 and 5
      expect(fft.list[3].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[5].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[n - 3].abs(), closeTo(n / 2, 0.1));
      expect(fft.list[n - 5].abs(), closeTo(n / 2, 0.1));

      // Check phase: cosine component should be real, sine should be imaginary
      expect(
          fft.list[3].argument(), lessThan(0.1)); // Near zero phase for cosine
      expect((fft.list[5].argument().abs() - pi / 2).abs(),
          lessThan(0.1)); // Near pi/2 phase for sine
    });
  });

  group('Spectrogram tests', () {
    test('Spectrogram of cosine signal', () {
      final n = 128;
      final nFFT = 32;
      final hopLength = 4;

      final signal = NDList.from<double>(
          [for (int i = 0; i < n; i++) cos(2 * pi * 4 * i / n)]);

      final spectrogram = signal.spectrogram(nFFT, hopLength: hopLength);
      final spectrogramData = spectrogram
          .toIteratedList()
          .map((row) => row.map((e) => '$e').join(','))
          .join('\n');
      File('cosine_chirp_spectrogram.csv').writeAsStringSync(spectrogramData);

      // Check dimensions
      expect(spectrogram.shape, equals([1 + (n - nFFT) ~/ hopLength, nFFT]));

      // Check magnitude peaks at frequency bin 4 and n-4
      for (int t = 0; t < spectrogram.shape[0]; t++) {
        expect(
            spectrogram[[t, 4]].item!, greaterThan(spectrogram[[t, 3]].item!));
        expect(
            spectrogram[[t, 4]].item!, greaterThan(spectrogram[[t, 5]].item!));
      }
    });

    test('Spectrogram of frequency chirp', () {
      final n = 256;
      final nFFT = 64;
      final hopLength = 16;

      // Create a linear chirp (frequency increases linearly with time)
      final freq =
          NDList.from<double>([for (int i = 0; i < n; i++) 1 + 10 * i / n]);
      final signal = NDList.from<double>(
          [for (int i = 0; i < n; i++) cos(2 * pi * freq[i].item! * i / n)]);

      final spectrogram = signal.spectrogram(nFFT, hopLength: hopLength);

      // save signal and spectrogram to CSV for visualization
      final signalData = signal.toFlattenedList();
      final signalCsv = List.generate(
              signalData.length, (i) => '${freq[i].item},${signalData[i]}')
          .join('\n');
      File('chirp_signal.csv')
          .writeAsStringSync('frequency,amplitude\n$signalCsv');

      // spectrogram data is a 2D array shape (M, N), save it to CSV with M rows and N columns
      final spectrogramData = spectrogram
          .toIteratedList()
          .map((row) => row.map((e) => '$e').join(','))
          .join('\n');
      File('chirp_spectrogram.csv').writeAsStringSync(spectrogramData);

      // Simply check dimensions for this complex signal
      expect(spectrogram.shape, equals([1 + (n - nFFT) ~/ hopLength, nFFT]));

      // In a chirp, the peak frequency should increase over time
      // We'll check this by comparing peak positions in first and last time frames
      var firstFramePeak = 0;
      var firstFrameMax = 0.0;
      for (int f = 0; f < nFFT ~/ 2; f++) {
        if (spectrogram[[0, f]].item! > firstFrameMax) {
          firstFrameMax = spectrogram[[0, f]].item!;
          firstFramePeak = f;
        }
      }

      var lastFramePeak = 0;
      var lastFrameMax = 0.0;
      for (int f = 0; f < nFFT ~/ 2; f++) {
        if (spectrogram[[spectrogram.shape[0] - 1, f]].item! > lastFrameMax) {
          lastFrameMax = spectrogram[[spectrogram.shape[0] - 1, f]].item!;
          lastFramePeak = f;
        }
      }

      // Last frame should have higher peak frequency than first frame
      expect(lastFramePeak, greaterThan(firstFramePeak));
    });
  });
}
