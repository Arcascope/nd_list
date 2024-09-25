import 'dart:math';

import 'package:nd_list/nd_list.dart';
import 'package:test/test.dart';

void main() {
  group('Rolling indices', () {
    test('Window size 1, axis 0', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ]);
      int windowSize = 1;
      int axis = 0;

      final xRolling = x.rolling(windowSize, axis: axis);

      expect(xRolling.slices.length, equals(3));

      for (int sliceIndex = 0;
          sliceIndex < xRolling.slices.length;
          sliceIndex++) {
        expect(xRolling.slices[sliceIndex].parentIndices, [
          x.shape[0] * sliceIndex,
          x.shape[0] * sliceIndex + 1,
          x.shape[0] * sliceIndex + 2
        ]);
      }
    });

    test('Window size 2, axis 0', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ]);
      int windowSize = 2;
      int axis = 0;

      final xRolling = x.rolling(windowSize, axis: axis);

      expect(xRolling.slices.length, equals(2));

      for (int sliceIndex = 0;
          sliceIndex < xRolling.slices.length;
          sliceIndex++) {
        expect(xRolling.slices[sliceIndex].parentIndices, [
          x.shape[0] * sliceIndex,
          x.shape[0] * sliceIndex + 1,
          x.shape[0] * sliceIndex + 2,
          x.shape[0] * sliceIndex + 3,
          x.shape[0] * sliceIndex + 4,
          x.shape[0] * sliceIndex + 5,
        ]);
      }
    });

    test('Window size 3, axis 0', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ]);
      int windowSize = 3;
      int axis = 0;

      final xRolling = x.rolling(windowSize, axis: axis);

      expect(xRolling.slices.length, equals(1));

      for (int sliceIndex = 0;
          sliceIndex < xRolling.slices.length;
          sliceIndex++) {
        expect(xRolling.slices[sliceIndex].parentIndices, [
          x.shape[0] * sliceIndex,
          x.shape[0] * sliceIndex + 1,
          x.shape[0] * sliceIndex + 2,
          x.shape[0] * sliceIndex + 3,
          x.shape[0] * sliceIndex + 4,
          x.shape[0] * sliceIndex + 5,
          x.shape[0] * sliceIndex + 6,
          x.shape[0] * sliceIndex + 7,
          x.shape[0] * sliceIndex + 8,
        ]);
      }
    });

    test('Window size 1, axis 1', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ]);
      int windowSize = 1;
      int axis = 1;

      final xRolling = x.rolling(windowSize, axis: axis);

      expect(xRolling.slices.length, equals(3));

      for (int sliceIndex = 0;
          sliceIndex < xRolling.slices.length;
          sliceIndex++) {
        expect(xRolling.slices[sliceIndex].parentIndices, [
          sliceIndex,
          sliceIndex + 3,
          sliceIndex + 6,
        ]);
      }
    });

    test('Window size 2, axis 1', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ]);
      int windowSize = 2;
      int axis = 1;

      final xRolling = x.rolling(windowSize, axis: axis);

      expect(xRolling.slices.length, equals(2));

      for (int sliceIndex = 0;
          sliceIndex < xRolling.slices.length;
          sliceIndex++) {
        expect(xRolling.slices[sliceIndex].parentIndices, [
          sliceIndex,
          sliceIndex + 1,
          sliceIndex + 3,
          sliceIndex + 4,
          sliceIndex + 6,
          sliceIndex + 7,
        ]);
      }
    });

    test('Window size 3, axis 1', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0]
      ]);
      int windowSize = 3;
      int axis = 1;

      final xRolling = x.rolling(windowSize, axis: axis);

      expect(xRolling.slices.length, equals(1));

      for (int sliceIndex = 0;
          sliceIndex < xRolling.slices.length;
          sliceIndex++) {
        expect(xRolling.slices[sliceIndex].parentIndices, [
          sliceIndex,
          sliceIndex + 1,
          sliceIndex + 2,
          sliceIndex + 3,
          sliceIndex + 4,
          sliceIndex + 5,
          sliceIndex + 6,
          sliceIndex + 7,
          sliceIndex + 8,
        ]);
      }
    });

    test('Window size 3, axis 2', () {
      final shape = [3, 4, 5];
      final arr3D = NDList.from<double>([
        for (int i = 0; i < shape.reduce((a, b) => a * b); i++) i.toDouble()
      ]).reshape(shape);

      // use an "identity" rolling window to create a 1D vector-of-vectors that is just the sliding windows themselves along axis 0
      final NDList<NDList<double>> xRolling =
          arr3D.rolling(3, axis: 2).reduce((a) => a);

      expect(xRolling.shape, equals([3]));

      for (int sliceIndex = 0; sliceIndex < xRolling.length; sliceIndex++) {
        expect(xRolling[sliceIndex].item!,
            equals(arr3D[[':', ':', '$sliceIndex:${sliceIndex + 3}']]));
      }
    });
  });
}
