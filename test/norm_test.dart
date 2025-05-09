import 'package:test/test.dart';
import 'package:nd_list/nd_list.dart';
import 'dart:math';

void main() {
  group('NDList norm()', () {
    // Norm across all dimensions
    test('Infinity norm of flat NDList', () {
      final nd = NDList.from<double>([0.1, -7.0, 2.5]);
      expect(nd.norm(order: double.infinity).list[0], 7.0);
    });

    test('L1 norm of flat NDList', () {
      final nd = NDList.from<double>([-1.0, 2.0, -3.0]);
      expect(nd.norm(order: 1).list[0], 6.0);
    });

    test('L2 norm of flat NDList', () {
      final nd = NDList.from<double>([3.0, 4.0]);
      expect(nd.norm().list[0], closeTo(5.0, 1e-9));
    });

    test('Infinity norm of 2D NDList', () {
      final nd = NDList.from<double>([
        [0.1, -7.0],
        [2.5, 3.0],
      ]);
      expect(nd.norm(order: double.infinity).list[0], 7.0);
    });

    test('L1 norm of 2D NDList', () {
      final nd = NDList.from<double>([
        [1.0, -2.0],
        [3.0, 4.0],
      ]);
      expect(nd.norm(order: 1).list[0], 10.0);
    });

    test('L2 norm of 2D NDList', () {
      final nd = NDList.from<double>([
        [3.0, 4.0],
        [5.0, 12.0],
      ]);
      expect(nd.norm().list[0], closeTo(13.928388, 1e-4));
    });

    test('Infinity norm of 3D NDList', () {
      final nd = NDList.from<double>([
        [
          [0.1, -7.0],
          [2.5, 3.0],
        ],
        [
          [5.0, -1.0],
          [2.0, 3.0],
        ],
      ]);
      expect(nd.norm(order: double.infinity).list[0], 7.0);
    });

    test('L1 norm of 3D NDList', () {
      final nd = NDList.from<double>([
        [
          [1.0, -2.0],
          [3.0, 4.0],
        ],
        [
          [5.0, -6.0],
          [7.0, 8.0],
        ],
      ]);
      expect(nd.norm(order: 1).list[0], closeTo(36.0, 1e-9));
    });


    test('L2 norm of 3D NDList', () {
      final nd = NDList.from<double>([
        [
          [1.0, -1.0],
          [3.0, 2.0],
        ],
        [
          [5.0, -1.0],
          [2.0, 3.0],
        ],
      ]);
      expect(nd.norm().list[0], closeTo(7.34846922, 1e-4));
    });

    // Two dimensions, test individual axis
    test('L1 norm along axis 1 of 2D NDList', () {
      final nd = NDList.from<double>([
        [1.0, -2.0],
        [3.0, 4.0],
      ]);
      final result = nd.norm(axis: 1, order: 1).list;
      expect(result.length, 2);
      expect(result[0], closeTo(3.0, 1e-9));
      expect(result[1], closeTo(7.0, 1e-9));
    });
    test('L1 norm along axis 0 of 2D NDList', () {
      final nd = NDList.from<double>([
        [1.0, -2.0],
        [3.0, 4.0],
      ]);
      final result = nd.norm(axis: 0, order: 1).list;
      expect(result.length, 2);
      expect(result[0], closeTo(4.0, 1e-9));
      expect(result[1], closeTo(6.0, 1e-9));
    });

    test('L2 norm along axis 0 of 2D NDList', () {
      final nd = NDList.from<double>([
        [3.0, 5.0],
        [4.0, 12.0],
      ]);
      final result = nd.norm(axis: 0).list;
      expect(result.length, 2);
      expect(result[0], closeTo(5.0, 1e-9));
      expect(result[1], closeTo(13.0, 1e-9));
    });

    test('L2 norm along axis 1 of 2D NDList', () {
      final nd = NDList.from<double>([
        [3.0, 4.0],
        [5.0, 12.0],
      ]);
      final result = nd.norm(axis: 1).list;
      expect(result.length, 2);
      expect(result[0], closeTo(5.0, 1e-9));
      expect(result[1], closeTo(13.0, 1e-9));
    });

    // Three dimensions
    test('L1 norm of 3D NDList', () {
      final nd = NDList.from<double>([
        [
          [1.0, -2.0],
          [3.0, 4.0],
        ],
        [
          [5.0, -6.0],
          [7.0, 8.0],
        ],
      ]);
      final result = nd.norm(order: 1).list;
      expect(result.length, 1);
      expect(result[0], closeTo(36.0, 1e-9));

    });

    test('L1 norm along axis 0 of 3D NDList', () {
      final data = [
        [
          [1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0],
          [9.0, 10.0, 11.0, 12.0],
        ],
        [
          [13.0, 14.0, 15.0, 16.0],
          [17.0, 18.0, 19.0, 20.0],
          [21.0, 22.0, 23.0, 24.0],
        ]
      ];
      final nd = NDList.from<double>(data);
      final result = nd.norm(axis: 0, order: 1);
      expect(result.shape, equals([3, 4]));
      expect(result[0].squeeze(), equals(NDList.from<double>([14.0, 16.0, 18.0, 20.0])));
      expect(result[1].squeeze(), equals(NDList.from<double>([22.0, 24.0, 26.0, 28.0])));
      expect(result[2].squeeze(), equals(NDList.from<double>([30.0, 32.0, 34.0, 36.0])));
    });

    test('L1 norm along axis 1 of 3D NDList', () {
      final data = [
        [
          [1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0],
          [9.0, 10.0, 11.0, 12.0],
        ],
        [
          [13.0, 14.0, 15.0, 16.0],
          [17.0, 18.0, 19.0, 20.0],
          [21.0, 22.0, 23.0, 24.0],
        ]
      ];
      final nd = NDList.from<double>(data);
      final result = nd.norm(axis: 1, order: 1);
      expect(result.shape, equals([2, 4]));
      expect(result[0].squeeze(), equals(NDList.from<double>([15.0, 18.0, 21.0, 24.0])));
      expect(result[1].squeeze(), equals(NDList.from<double>([51.0, 54.0, 57.0, 60.0])));
    });

    test('L1 norm along axis 2 of 3D NDList', () {
      final data = [
        [
          [1.0, 2.0, 3.0, 4.0],
          [5.0, 6.0, 7.0, 8.0],
          [9.0, 10.0, 11.0, 12.0],
        ],
        [
          [13.0, 14.0, 15.0, 16.0],
          [17.0, 18.0, 19.0, 20.0],
          [21.0, 22.0, 23.0, 24.0],
        ]
      ];
      final nd = NDList.from<double>(data);
      final result = nd.norm(axis: 2, order: 1);
      expect(result.shape, equals([2, 3]));
      expect(result[0].squeeze(), equals(NDList.from<double>([10.0, 26.0, 42.0])));
      expect(result[1].squeeze(), equals(NDList.from<double>([58.0, 74.0, 90.0])));
    });
  });
}
