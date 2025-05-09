import 'package:test/test.dart';
import 'package:nd_list/nd_list.dart';
import 'dart:math';

void main() {
  group('NDList diff()', () {
    // Test for 1D NDList
    test('1D NDList diff', () {
      final nd = NDList.from<double>([1.0, 2.0, 4.0, 7.0]);
      final result = nd.diff();
      expect(result.list, [1.0, 2.0, 3.0]);
    });

  // Test for 2D NDList
  test('2D NDList diff', () {
    final nd = NDList.from<double>([
      [1.0, 2.0, 4.0],
      [7.0, 11.0, 16.0],
    ]);
    final result = nd.diff();
    expect(result.shape, [2, 2]);
    expect(result[0].squeeze(), equals(NDList.from([1.0, 2.0])));
    expect(result[1].squeeze(), equals(NDList.from([4.0, 5.0])));
  });

  test('2D NDList diff axis 0', () {
    final nd = NDList.from<double>([
      [1.0, 2.0, 4.0],
      [7.0, 11.0, 16.0],
    ]);
    final result = nd.diff(axis: 0);
    expect(result.shape, [1, 3]);
    expect(result.squeeze(), equals(NDList.from([6.0, 9.0, 12.0])));
  });

  test('2D NDList diff axis 1', () {
    final nd = NDList.from<double>([
      [1.0, 2.0, 4.0],
      [7.0, 11.0, 16.0],
    ]);
    final result = nd.diff(axis: 1);
    expect(result.shape, [2, 2]);
    expect(result[0].squeeze(), equals(NDList.from([1.0, 2.0])));
    expect(result[1].squeeze(), equals(NDList.from([4.0, 5.0])));
  });

  // Test for 3D NDList
  test('3D NDList diff axis 0', () {
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
    final result = nd.diff(axis: 0);
    expect(result.shape, [1, 3, 4]);
    expect(result.squeeze(), equals(NDList.from([
      [12.0, 12.0, 12.0, 12.0],
      [12.0, 12.0, 12.0, 12.0],
      [12.0, 12.0, 12.0, 12.0],
    ]).squeeze()));
  });
  test('3D NDList diff axis 1', () {
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
    final result = nd.diff(axis: 1);
    expect(result.shape, [2, 2, 4]);
    expect(result.squeeze(), equals(NDList.from([
      [[4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0]],
       [[4.0, 4.0, 4.0, 4.0],
        [4.0, 4.0, 4.0, 4.0]]
    ])));
  });

  test('3D NDList diff axis 2', () {
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
    final result = nd.diff(axis: 2);
    expect(result.shape, [2, 3, 3]);
    expect(result.squeeze(), equals(NDList.from([
      [[1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]],

       [[1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0]] 
   ])));
  });
  });
}