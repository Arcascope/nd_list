import 'dart:math';

import 'package:nd_list/nd_list.dart';
import 'package:test/test.dart';

void main() {
  group('NDList []=', () {
    test('Test can assign 1 element of a 1d NDList', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final ndList = NDList.from<double>(data);

      ndList[0] = 5.0;
      expect(ndList[0].item!, equals(5.0));

      ndList[1] = 6.0;
      expect(ndList[1].item!, equals(6.0));

      ndList[-1] = 7.0;
      expect(ndList[3].item!, equals(7.0));
    });
    test('Test can assign a slice of a 1d NDList', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final ndList = NDList.from<double>(data);

      ndList['1:3'] = 5.0;
      expect(ndList[0].item!, equals(data[0]));
      expect(ndList[3].item!, equals(data[3]));
      expect(ndList[1].item!, equals(5.0));
      expect(ndList[2].item!, equals(5.0));
    });
    test('Test can assign an axis-0 element of a 2d NDList', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      final editToBe = NDList.from<double>([5.0, 6.0, 7.0]);
      ndList[0] = editToBe;

      expect(ndList[0], equals(editToBe));
    });
    test('Test can assign an axis-0 slice of a 2d NDList', () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndList = NDList.from<double>(data);
      final editSlice = NDList.filled([2, 3], -99.0);

      ndList['1:3'] = editSlice;

      expect(ndList[0], equals(NDList.from<double>(data[0])));
      expect(ndList[1], equals(editSlice[0]));
      expect(ndList[2], equals(editSlice[1]));
      expect(ndList[3], equals(NDList.from<double>(data[3])));
    });
    test('Test can assign an axis-1 element of a 2d NDList', () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndList = NDList.from<double>(data);
      const fillValue = -99.0;
      final editSlice = NDList.filled([4, 1], fillValue);

      ndList[[':', 1]] = editSlice;

      for (int row = 0; row < ndList.shape[0]; row++) {
        expect(ndList[row][0].item, equals(data[row][0]),
            reason: "Element 0 of row $row should not have changed");
        expect(ndList[row][1].item, equals(fillValue),
            reason: "Element 1 of row $row should have been set to $fillValue");
        expect(ndList[row][2].item, equals(data[row][2]),
            reason: "Element 2 of row $row should not have changed");
      }
    });
  });

  group('Cementing', () {
    /// This is a fairly important operation. It allows us to decompose a transformation into a sequence of smaller transformations on blocks, and then stack the blocks together with .cemented().
    test('1x1s', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      final subNDs = <NDList<double>>[];
      for (int i = 0; i < data.length; i++) {
        for (int j = 0; j < data[i].length; j++) {
          subNDs.add(NDList.from<double>([data[i][j]]));
        }
      }

      final ndOfNDs = NDList.from<NDList<double>>(subNDs);

      expect(ndOfNDs.shape, equals([data.length * data[0].length]));

      final cemented = ndOfNDs.reshape([2, 3]).cemented();

      // note: this works a little strangely for a cement of 1x1s
      expect(cemented.shape, equals([2, 3, 1]));
      expect(cemented.squeeze(), equals(ndList));
    });

    test('3x2 of (4,)s', () {
      const nRows = 3;
      const nCols = 2;
      final filler = NDList.from<double>([1.0, 2.0, 3.0, 4.0]);
      expect(filler.shape, [4]);
      final ndLists = [
        for (int i = 0; i < nRows; i++) [for (int j = 0; j < nCols; j++) filler]
      ];

      final ndOfNDs = NDList.from<NDList<double>>(ndLists);

      expect(ndOfNDs.shape, equals([nRows, nCols]));

      for (var i = 0; i < nRows; i++) {
        for (var j = 0; j < nCols; j++) {
          // NOTE! This is a 1x1 NDList; the element _happens_ to be an NDList with shape [4], but it's not correct to think of ndOfNDs[[i, j]] as the same as it's only element. This is a more complicated example of the Dart difference between 1, [1], and NDList.from<int>([1]).
          expect(ndOfNDs[[i, j]].shape, equals([1]));

          // since ndOfNDs[[i, j]] has shape [1], we can use .item to get its contents.
          // this is the [4]-shaped NDList `filler`
          expect(ndOfNDs[[i, j]].item!.shape, equals(filler.shape));
        }
      }

      final cemented = ndOfNDs.cemented();

      expect(cemented.shape, equals([nRows, nCols, 4]));
    });
  });

  group('NDList<double> indexing', () {
    test('==', () {
      final data = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      final ndList = NDList.from<double>(data);
      final ndList0 = NDList.from<double>(data[0]);
      final ndList0Second = NDList.from<double>([1.0, 2.0]);

      // check they equal themselves
      expect(ndList, equals(ndList));
      expect(ndList0, equals(ndList0));
      expect(ndList0Second, equals(ndList0));

      // and do not equal others
      expect(ndList0 == ndList, isFalse);
    });

    test('1d Indexing with int', () {
      final data = [91.0, 92.0, 94.0];
      final ndList = NDList.from<double>(data);

      expect(ndList.shape, [data.length]);
      for (var i = 0; i < 3; i++) {
        expect(ndList[i].shape, equals([1]));
        expect(ndList[i].item, equals(data[i]));
      }
    });

    test('2d Indexing with int, axis 0', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);
      final ndList0 = NDList.from<double>(data[0]);

      expect(ndList[0], equals(ndList0));
    });
    test('2d Indexing with int, axis 1', () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndList = NDList.from<double>(data);

      final ndList1 = ndList[[':', 1]];

      final expectedData = [
        [1.0],
        [4.0],
        [7.0],
        [10.0]
      ];

      expect(ndList1, equals(NDList.from<double>(expectedData)));
    });

    test('2d Indexing with List<int>', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      for (int i = 0; i < 2; i++) {
        for (var j = 0; j < 3; j++) {
          expect(ndList[[i, j]].item, equals(data[i][j]));
        }
      }
    });

    test('.item', () {
      final data = [1.0];
      final ndList = NDList.from<double>(data);

      expect(ndList.item, equals(1.0));

      final data2 = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      final ndList2 = NDList.from<double>(data2);

      expect(ndList2.item, isNull);
    });

    test('Create NDList from List<double>', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final ndList = NDList.from<double>(data);

      expect(ndList.shape, equals([4]));
      expect(ndList[0].item, equals(1.0));
      expect(ndList[1].item, equals(2.0));
      expect(ndList[2].item, equals(3.0));
      expect(ndList[3].item, equals(4.0));
    });

    test('Negative indices (1D)', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final ndList = NDList.from<double>(data);

      // negative indices count from the end and loop
      expect(ndList[-1].item, equals(4.0));
      expect(ndList[-2].item, equals(3.0));
      expect(ndList[-3].item, equals(2.0));
      expect(ndList[-4].item, equals(1.0));
      expect(ndList[-5].item, equals(4.0));
      expect(ndList[-6].item, equals(3.0));
      expect(ndList[-7].item, equals(2.0));
      expect(ndList[-8].item, equals(1.0));
    });

    test('Create NDList from nested List<double>', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      expect(ndList.shape, equals([2, 3]));

      // 1d data
      final length0 = data[0].length;
      final ndList0 = NDList.from<double>(data[0]);
      expect(ndList0.shape, equals([length0]));
      expect(ndList[0], equals(ndList0));
      expect(ndList0[0].item, equals(data[0][0]));
      expect(ndList0[1].item, equals(data[0][1]));
      expect(ndList[0][0].item, equals(data[0][0]));
      expect(ndList[0][1].item, equals(data[0][1]));

      final ndList1 = NDList.from<double>(data[1]);
      expect(ndList[1], equals(ndList1));
      expect(ndList1[0].item, equals(data[1][0]));
      expect(ndList1[1].item, equals(data[1][1]));
      expect(ndList[1][0].item, equals(data[1][0]));
      expect(ndList[1][1].item, equals(data[1][1]));

      // trivially 2-dim [1, N]
      final ndList0Wrapped = NDList.from<double>([data[0]]);
      expect(ndList0Wrapped.shape, equals([1, length0]));
      expect(ndList0Wrapped[0], equals(ndList0));
      expect(ndList0Wrapped[0][0].item, equals(1.0));
      expect(ndList0Wrapped[0][1].item, equals(2.0));
    });

    test('zeros: 2D', () {
      final shape = [3, 2];
      final ndList = NumNDList.zeros<double>(shape);

      expect(ndList.shape, equals(shape));
      for (var i = 0; i < shape[0]; i++) {
        expect(ndList[i].shape, equals([2]));
        for (var j = 0; j < shape[1]; j++) {
          expect(ndList[i][j].item, equals(0.0));
        }
      }
    });

    test('zeros: high D', () {
      final shape = [1, 2, 3, 4, 5, 6];
      final ndList = NumNDList.zeros<double>(shape);

      expect(ndList.shape, equals(shape));
      final ndListFlat = ndList.flatten();
      for (var i = 0; i < ndListFlat.length; i++) {
        expect(ndListFlat[i].item, equals(0.0));
      }
    });

    test('zerosLike', () {
      final data1x4 = [1.0, 2.0, 3.0, 4.0];
      final ndList1x4 = NDList.from<double>(data1x4);
      final data2x3 = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList2x3 = NDList.from<double>(data2x3);

      final zerosLike1x4 = NumNDList.zerosLike<double>(ndList1x4);
      final zerosLike2x3 = NumNDList.zerosLike<double>(ndList2x3);

      expect(zerosLike1x4.shape, equals([4]));
      expect(zerosLike1x4[0].item.runtimeType, double);
      for (var i = 0; i < 4; i++) {
        expect(zerosLike1x4[i].item, equals(0.0));
      }

      expect(zerosLike2x3.shape, equals([2, 3]));

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 3; j++) {
          expect(zerosLike2x3[i][j].item, equals(0.0));
        }
      }
    });

    test('Slicing once along axis 0, both end points given', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
      ];
      final ndList = NDList.from<double>(data);
      final sliced02 = ndList['0:2'];
      final sliced13 = ndList['1:3'];
      final sliced24 = ndList['2:4'];

      final sliceShape = [2, 3];

      expect(sliced02.shape, equals(sliceShape));
      expect(sliced02, equals(NDList.from<double>(data.sublist(0, 2))));
      expect(sliced13.shape, equals(sliceShape));
      expect(sliced13, equals(NDList.from<double>(data.sublist(1, 3))));
      expect(sliced24.shape, equals(sliceShape));
      expect(sliced24, equals(NDList.from<double>(data.sublist(2, 4))));
    });

    test('Slicing once along axis 0, only one point given', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
      ];
      final ndList = NDList.from<double>(data);
      final slicedTil2 = ndList[':2'];
      final slicedFrom2 = ndList['2:'];

      final sliceShape = [2, 3];

      expect(slicedTil2.shape, equals(sliceShape));
      expect(slicedFrom2.shape, equals(sliceShape));

      expect(slicedTil2, equals(NDList.from<double>(data.sublist(0, 2))));
      expect(
          slicedFrom2,
          equals(NDList.from<double>(data.sublist(
            2,
          ))));
    });

    test('Shape of 3D NDList slice', () {
      final testND = NDList.filled([2, 4, 3], 0.0);

      // axis 0
      final axis0Slice = testND[[':1', ':', ':']];
      expect(axis0Slice.shape, equals([1, 4, 3]), reason: 'axis 0 slice shape');

      // axis 1
      final axis1Slice = testND[[':', ':1', ':']];
      expect(axis1Slice.shape, equals([2, 1, 3]), reason: 'axis 1 slice shape');

      // axis 2
      final axis2Slice = testND[[':', ':', ':1']];
      expect(axis2Slice.shape, equals([2, 4, 1]), reason: 'axis 2 slice shape');

      // final testSlice = testND[[':', '1:3']];
      // expect(testSlice.shape, equals([2, 2, 3]));
      // final testSlice2 = testND[[':2', ':1']];
      // expect(testSlice2.shape, equals([2, 1, 3]));
      // final iteratedSlice = testSlice[[':', ':', ':1']];
      // expect(iteratedSlice.shape, equals([2, 2, 1]));
    });

    test('Slicing once along axis 1', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
      ];

      final expectedSliceData = [
        [1.0, 2.0],
        [4.0, 5.0],
        [7.0, 8.0],
        [10.0, 11.0]
      ];
      final ndList = NDList.from<double>(data);
      final sliceShape = [4, 2];

      // python-style slice, formatted as a string
      final slicedTil2Py = ndList[[':', ':2']];
      expect(slicedTil2Py.shape, equals(sliceShape),
          reason: 'python-style slice has wrong shape');
      expect(slicedTil2Py, NDList.from<double>(expectedSliceData),
          reason: 'python-style slice has wrong data');

      // explicitly calling the .slice method
      final slicedTil2 = ndList.slice(0, 2, axis: 1);
      expect(slicedTil2.shape, equals(sliceShape),
          reason: 'explicit slice has wrong shape');
      expect(slicedTil2, NDList.from<double>(expectedSliceData),
          reason: 'explicit slice has wrong data');
    });

    test('Sum', () {
      final data1 = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      final data2 = [
        [2.0, 3.0],
        [4.0, 5.0]
      ];
      final ndList1 = NDList.from<double>(data1);
      final ndList2 = NDList.from<double>(data2);

      final sum = ndList1 + ndList2;

      expect(sum.shape, equals([2, 2]));

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
          expect(sum[i][j].item, equals(data1[i][j] + data2[i][j]));
        }
      }
    });
    test('Product', () {
      final data1 = [
        [1.0, 2.0],
        [3.0, 4.0]
      ];
      final data2 = [
        [2.0, 3.0],
        [4.0, 5.0]
      ];
      final ndList1 = NDList.from<double>(data1);
      final ndList2 = NDList.from<double>(data2);

      final sum = ndList1 * ndList2;

      expect(sum.shape, equals([2, 2]));

      for (var i = 0; i < 2; i++) {
        for (var j = 0; j < 2; j++) {
          expect(sum[i][j].item, equals(data1[i][j] * data2[i][j]));
        }
      }
    });

    test('Test .cement() for 1x1s', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      final subNDs = <NDList<double>>[];
      for (int i = 0; i < data.length; i++) {
        for (int j = 0; j < data[i].length; j++) {
          subNDs.add(NDList.from<double>([data[i][j]]));
        }
      }

      final ndOfNDs = NDList.from<NDList<double>>(subNDs);

      expect(ndOfNDs.shape, equals([data.length * data[0].length]));

      final cemented = ndOfNDs.reshape([2, 3]).cemented();

      // note: this works a little strangely for a cement of 1x1s
      expect(cemented.shape, equals([2, 3, 1]));
      expect(cemented.reshape([2, 3]), equals(ndList));
    });

    test('Test .cement() for 1x2s', () {
      final ndLists = [
        for (int i = 0; i < 3 * 2; i++) NDList.from<double>([1.0, 2.0])
      ];

      final ndOfNDs = NDList.from<NDList<double>>(ndLists);

      final cemented = ndOfNDs.reshape([3, 2]).cemented();

      expect(cemented.shape, equals([3, 2, 2]));
    });

    test('Test int indexing. Throw error only if out of bounds', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);
      // single int input
      expect(ndList[0], equals(NDList.from<double>(data[0])));
      expect(ndList[1], equals(NDList.from<double>(data[1])));
      expect(() => ndList[2], throwsRangeError);
      expect(ndList[-1], equals(NDList.from<double>(data[1])));
      expect(ndList[-2], equals(NDList.from<double>(data[0])));
    });
  });
}
