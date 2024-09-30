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
  Complex twiddle(int k, int N) {
    final angle = -2 * pi * k / N;
    return Complex.polar(1, angle);
  }

  /// The Radix-2 split-radix FFT algorithm for real-valued data.
  List<Complex> splitRadixFFT(List<double> data) {
    final N = data.length;
    if (N <= 1) {
      return [Complex(data[0], 0)];
    } else if (N == 2) {
      // Handle N == 2 case separately
      final e = data[0];
      final o = data[1];
      return [Complex(e + o, 0), Complex(e - o, 0)];
    } else if (N == 4) {
      // Handle N == 4 case separately
      final e0 = data[0];
      final e1 = data[1];
      final o0 = data[2];
      final o1 = data[3];

      final t0 = twiddle(0, 4);
      final t1 = twiddle(1, 4);

      return [
        Complex(e0 + e1 + o0 + o1, 0),
        Complex(e0 - e1, 0) + t1 * Complex(0, o0 - o1),
        Complex(e0 + e1 - o0 - o1, 0),
        Complex(e0 - e1, 0) - t1 * Complex(0, o0 - o1),
      ];
    }

    // Split into even and odd indices
    final even = data.sublist(0, N ~/ 2);
    final odd = data.sublist(N ~/ 2);

    // Recursively compute FFTs of even and odd parts
    final evenFFT = splitRadixFFT(even);
    final oddFFT = splitRadixFFT(odd);

    // Combine results
    final result = List.generate(N, (i) => Complex(0, 0));
    for (int k = 0; k < N ~/ 4; k++) {
      final t = twiddle(k, N);
      final e = evenFFT[k];
      final o = oddFFT[k];
      final o1 = oddFFT[N ~/ 4 - k - 1].conjugate();
      result[k] = e + t * o;
      result[k + N ~/ 4] = e - t * o;
      result[k + N ~/ 2] = e + t * o1;
      result[k + 3 * N ~/ 4] = e - t * o1;
    }

    return result;
  }

  /// Computes the Fast Fourier Transform of the NDList<double>
  List<Complex> _fft(List<double> x, {bool isReal = true}) {
    int N = x.length;

    final z = List<Complex>.generate(N, (index) => Complex(x[index], 0));
    if (N <= 1) return z;

    // Cooley-Tukey FFT algorithm optimized for real input
    List<Complex> even =
        _fft([for (int i = 0; i < N ~/ 2; i++) x[2 * i]], isReal: false);
    List<Complex> odd =
        _fft([for (int i = 0; i < N ~/ 2; i++) x[2 * i + 1]], isReal: false);

    List<Complex> result = List<Complex>.filled(N, Complex.zero);
    for (int k = 0; k < N ~/ 2; k++) {
      Complex t = Complex.polar(1.0, -2 * pi * k / N) * odd[k];
      result[k] = even[k] + t;
      result[k + N ~/ 2] = even[k] - t;
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
