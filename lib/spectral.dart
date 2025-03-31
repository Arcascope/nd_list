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
  /// Calculates the twiddle factor for a given index and length.
  Complex fftPair(int k, int N) {
    final angle = -2 * pi * k / N;
    return Complex.polar(1, angle);
  }

  /// The Radix-2 FFT algorithm for real-valued data.
  List<Complex> splitRadixFFT(List<double> data) {
    final N = data.length;

    // Base cases
    if (N == 1) {
      return [Complex(data[0], 0)];
    }

    // Check if N is a power of 2
    if ((N & (N - 1)) != 0) {
      throw ArgumentError('FFT length must be a power of 2');
    }

    // Split into even and odd indices
    List<double> even = List<double>.filled(N ~/ 2, 0);
    List<double> odd = List<double>.filled(N ~/ 2, 0);

    for (int i = 0; i < N ~/ 2; i++) {
      even[i] = data[2 * i];
      odd[i] = data[2 * i + 1];
    }

    // Recursively compute FFTs of even and odd parts
    List<Complex> evenFFT = splitRadixFFT(even);
    List<Complex> oddFFT = splitRadixFFT(odd);

    // Combine results
    List<Complex> result = List<Complex>.filled(N, Complex.zero);
    for (int k = 0; k < N ~/ 2; k++) {
      Complex t = fftPair(k, N) * oddFFT[k];

      result[k] = evenFFT[k] + t;
      result[k + N ~/ 2] = evenFFT[k] - t;
    }

    return result;
  }

  NDList<Complex> fft() {
    var complexOutput = splitRadixFFT(list);

    return NDList.from(complexOutput);
  }

  NDList<double> spectrogram(int nFFT, {int hopLength = 1}) {
    if (!is1D) {
      return rolling(nFFT, axis: -1)
          .reduce((a) => a.spectrogram(nFFT, hopLength: hopLength))
          .cemented();
    }

    return reshape([-1])
        .rolling(nFFT, step: hopLength, axis: 0)
        .reduce((e) => e.fft().map((e) => e.abs() * e.abs()))
        .cemented();
  }
}
