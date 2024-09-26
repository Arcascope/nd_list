import 'dart:math';
import 'package:nd_list/nd_list.dart';
import 'package:test/test.dart';

void main() {
  group('Test pure cosines', () {
    test('Single cosine', () {
      final nSamples = 64;
      final nPeriods = 3;
      NDList<double> time = NDList.from([
        for (int i = 0; i < nSamples; i++)
          i.toDouble() / nSamples * nPeriods * 2 * pi
      ]);

      NDList<double> signal = time.map((t) => cos(t));

      final fft = signal.fft().map((e) => e.abs());

      expect(fft.shape, equals([nSamples]));
      print(fft);
      // we expect the fft to be concentrated around one specific frequency
      // expect(fft, matcher)

      final nFFT = 32;
      final specgram = signal.spectrogram(nFFT);

      expect(specgram.shape, equals([nSamples - nFFT + 1, nFFT]));
    });
  });
}
