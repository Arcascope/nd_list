import 'dart:math';
import 'package:nd_list/nd_list.dart';
import 'package:test/test.dart';

void main() {
  group('Test pure cosines', () {
    test('Single cosine', () {
      final nSamples = 256;
      final nPeriods = 3;
      NDList<double> time = NDList.from([
        for (int i = 0; i < nSamples; i++)
          i.toDouble() / nSamples * nPeriods * 2 * pi
      ]);

      NDList<double> signal = time.map((t) => cos(t));

      final nFFT = 32;
      final fft = signal.fft().map((e) => e.abs());

      expect(fft.shape, equals([nSamples]));
      // we expect the fft to be concentrated around one specific frequency
      // expect(fft, matcher)

      final specgram = signal.spectrogram(nFFT);

      expect(specgram.shape, equals([nSamples - nFFT + 1, nFFT]));
    });
  });
}
