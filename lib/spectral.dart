import 'dart:math';
import 'package:complex/complex.dart';
import 'package:nd_list/nd_list.dart';

/// For an input NDList<double> computes a spectrogram. The spectrogram is taken as follows:
/// 1. If the input is 1D, this is just a sliding FFT
/// 2. If the input is 2D, this is a sliding FFT along the columns (axis 1), meaning each array[[:, i]] is taken as input
/// 3. If the input is 3D, this a stacked spectrogram along axis 2, meaning each array[:, :, i] is taken as input to the previous case.
///
/// The pattern continues into higher dimensions, where the last axis is taken as the input to the previous case.

extension SpectralAnalysis on NDList<double> {
  /// Computes the Fast Fourier Transform of the NDList<double>

  List<Complex> _fft(List<Complex> x) {
    int N = x.length;
    if (N <= 1) return x;

    // Divide
    var even = List<Complex>.generate(N ~/ 2, (i) => x[2 * i]);
    var odd = List<Complex>.generate(N ~/ 2, (i) => x[2 * i + 1]);

    // Conquer
    var fftEven = _fft(even);
    var fftOdd = _fft(odd);

    // Combine
    var result = List<Complex>.filled(N, Complex(0.0));
    for (int k = 0; k < N ~/ 2; k++) {
      var t = Complex.polar(1.0, -2 * pi * k / N) * fftOdd[k];
      result[k] = fftEven[k] + t;
      result[k + N ~/ 2] = fftEven[k] - t;
    }
    return result;
  }

  NDList<Complex> fft() {
    if (shape.length != 1) {
      throw ArgumentError('rfft is only implemented for 1D arrays.');
    }

    var complexInput = map((e) => Complex(e, 0)).toFlattenedList();
    var complexOutput = _fft(complexInput);

    // var realOutput = complexOutput.map((e) => e.abs()).toList();
    return NDList.from(complexOutput);
  }

  NDList<double> spectrogram(int nFFT) {
    if (!is1D) {
      return rolling(nFFT, axis: -1)
          .reduce((a) => a.spectrogram(nFFT))
          .cemented();
    }

    return reshape([-1])
        .rolling(nFFT, axis: 0)
        .reduce((e) => e.fft().map((e) => e.abs()))
        .cemented();
  }
}
