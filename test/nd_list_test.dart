import 'dart:math';

import 'package:nd_list/nd_list.dart';
import 'package:test/test.dart';

void main() {
  group('NDIndexResult', () {
    test('Test resolving', () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final ndList = NDList.from<double>(data);

      final result = NDIndexResult.from(ndList);
      final result2 = result.resolveStep([1, 2, 3], [3]);
      expect(result2.evaluate(), equals(NDList.from<double>([2.0, 3.0, 4.0])));
      expect(result2.parentIndices, [1, 2, 3]);
      expect(result2.parent, equals(ndList));
      expect(result2.shape, equals([3]));

      final result3 = result2.resolveStep([1, 2], [2]);
      expect(result3.parentIndices, equals([2, 3]));
      expect(result3.parent, equals(ndList));
      expect(result3.shape, equals([2]));
      expect(result3.evaluate(), equals(NDList.from<double>([3.0, 4.0])));
    });

    test('Resolving 3D', () {
      final ndList = NumNDList.zeros([2, 3, 4]);

      final result = NDIndexResult.from(ndList);
      // manually set the indices for slice [:, 1]
      // NOTE! Comments are in here intentionally,
      //  it helps keep track of the indices
      final sliceIndex = [
        // [
        //  0, 1, 2, 3,
        4, 5, 6, 7,
        //  8, 9, 10, 11,
        // ], [
        // ]
        //  12, 13, 14, 15,
        16, 17, 18, 19
        //  20, 21, 22, 23,
      ];

      final result2 = result.resolveStep(sliceIndex, [2, 4]);
      expect(result2.parentIndices, sliceIndex);
      expect(result2.parent, equals(ndList));
      expect(result2.shape, equals([2, 4]));

      final subSlice = [1, 5]; // => [5, 17] aka [:, 1, 1]
      final result3 = result2.resolveStep(subSlice, [2, 1]);
      expect(result3.parentIndices, equals([5, 17]));
      expect(result3.parent, equals(ndList));
      expect(result3.shape, equals([2, 1]));
    });

    test('Evaluate 3D from resolve', () {
      final ndList = NumNDList.zeros<double>([2, 3, 4]);

      final result = NDIndexResult.from(ndList);

      final valuesAdded = <double>[];
      for (int i = 0; i < ndList.shape[0]; i++) {
        for (int j = 0; j < ndList.shape[1]; j++) {
          for (int k = 0; k < ndList.shape[2]; k++) {
            final newValue = i * 100.0 + j * 10 + k;
            ndList[[i, j, k]] = newValue;
            valuesAdded.add(newValue);
          }
        }
      }

      final result2 = result.resolveStep(range(6), [2, 3]);
      final ndResult = result2.evaluate();

      expect(ndResult.shape, equals([2, 3]));
      expect(ndResult.flatten(),
          equals(NDList.from<double>(valuesAdded.sublist(0, 6))));
    });
  });

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

      expect(ndList, NDList.from<double>([5.0, 6.0, 3.0, 7.0]));
      print(ndList);
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

    test('Test can assign a slice of a 1d NDList, where end index is negative',
        () {
      final data = [1.0, 2.0, 3.0, 4.0];
      final ndList = NDList.from<double>(data);

      ndList['1:-1'] = 5.0;
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

      expect(ndList[0], equals(editToBe.reshape([1, 3])));
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

      expect(ndList[0].flatten(), equals(NDList.from<double>(data[0])));
      expect(ndList[1], equals(editSlice[0]));
      expect(ndList[2], equals(editSlice[1]));
      expect(ndList[3].flatten(), equals(NDList.from<double>(data[3])));
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
        expect(ndList[[row, 0]].item, equals(data[row][0]),
            reason: "Element 0 of row $row should not have changed");
        expect(ndList[[row, 1]].item, equals(fillValue),
            reason: "Element 1 of row $row should have been set to $fillValue");
        expect(ndList[[row, 2]].item, equals(data[row][2]),
            reason: "Element 2 of row $row should not have changed");
      }
    });

    test('Test can assign axis-1 slice of a 2d NDList<double> with 0d (double)',
        () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndData = NDList.from<double>(data);

      ndData[[':', '1:']] = 99.0;
      for (int i = 0; i < ndData.shape[0]; i++) {
        expect(ndData[[i, 0]].item, equals(data[i][0]));
        expect(ndData[[i, 1]].item, equals(99.0));
        expect(ndData[[i, 2]].item, equals(99.0));
      }
    });
    test(
        'Test can assign axis-1 slice with last index negative of a 2d NDList<double> with 0d (double)',
        () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndData = NDList.from<double>(data);

      final fillValue = 99.0;
      ndData[[':', ':-1']] = fillValue;
      for (int i = 0; i < ndData.shape[0]; i++) {
        expect(ndData[[i, 0]].item, equals(fillValue), reason: "Row $i, 0");
        expect(ndData[[i, 1]].item, equals(fillValue), reason: "Row $i, 1");
        expect(ndData[[i, 2]].item, equals(data[i].last), reason: "Row $i, 2");
      }
    });

    test('Test can assign axis-0 slice of a 2d NDList with 1d data', () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndData = NDList.from<double>(data);

      ndData[['3:']] = NDList.filled([3], 99.0);
      for (int i = 0; i < ndData.shape[1]; i++) {
        expect(ndData[[0, i]].item, data[0][i]);
        expect(ndData[[1, i]].item, data[1][i]);
        expect(ndData[[2, i]].item, data[2][i]);
        expect(ndData[[3, i]].item, 99);
      }
    });

    test('Test can assign axis-1 slice of a 2d NDList with 1d data', () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndData = NDList.from<double>(data);

      ndData[[':', '2:']] = NDList.filled([4], 99.0);
      for (int i = 0; i < ndData.shape[0]; i++) {
        expect(ndData[[i, 0]].item, equals(data[i][0]));
        expect(ndData[[i, 1]].item, equals(data[i][1]));
        expect(ndData[[i, 2]].item, equals(99.0));
      }
      ndData[[':', '1:2']] = NDList.filled([4, 1], 111.0);
      for (int i = 0; i < ndData.shape[0]; i++) {
        expect(ndData[[i, 0]].item, equals(data[i][0]));
        expect(ndData[[i, 1]].item, equals(111.0));
        expect(ndData[[i, 2]].item, equals(99.0));
      }
      ndData[[':', ':1']] = NDList.filled([4, 1], -888.0);
      for (int i = 0; i < ndData.shape[0]; i++) {
        expect(ndData[[i, 0]].item, equals(-888.0));
        expect(ndData[[i, 1]].item, equals(111.0));
        expect(ndData[[i, 2]].item, equals(99.0));
      }
    });

    test('Test can assign axis-1 slice of a 2d NDList with 2d data', () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndData = NDList.from<double>(data);

      ndData[[':', '1:']] = NDList.filled([4, 2], 99.0);
      for (int i = 0; i < ndData.shape[0]; i++) {
        expect(ndData[[i, 0]].item, equals(data[i][0]));
        expect(ndData[[i, 1]].item, equals(99.0));
        expect(ndData[[i, 2]].item, equals(99.0));
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
          // NOTE! This is a 1x1 NDList<NDList<double>>;
          // the element _happens_ to be an NDList with shape [4], but it's not correct to think of ndOfNDs[[i, j]] as the same as it's only element.
          // This is a more complicated example of the Dart difference between 1, [1], and NDList.from<int>([1]).
          expect(ndOfNDs[[i, j]].shape, equals([1, 1]));

          // since ndOfNDs[[i, j]] has shape [1], we can use .item to get its contents.
          // this is the [4]-shaped NDList `filler`
          expect(ndOfNDs[[i, j]].item!.shape, equals(filler.shape));
        }
      }

      final cemented = ndOfNDs.cemented();

      expect(cemented.shape, equals([nRows, nCols, 4]));
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
  });

  group('Basic operators', () {
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
  });
  group('NDList<double> indexing', () {
    /// How does shape change on indexing?
    ///
    /// dim = 1, shape (n, )
    /// arr[i] shape (1, )
    ///
    /// dim = 2, shape (n, m)
    /// arr[i] shape (1, m)
    /// arr[:, j] shape (n, 1)
    ///
    /// dim = 3, shape (n, m, p)
    /// arr[i] shape (1, m, p)
    /// arr[:, j] shape (n, 1, p)
    /// arr[:, :, k] shape (n, m, 1)
    ///
    ///
    test('1d Indexing with int', () {
      final data = [91.0, 92.0, 94.0];
      final ndList = NDList.from<double>(data);

      expect(ndList.shape, [data.length]);
      for (var i = 0; i < 3; i++) {
        expect(ndList[i].shape, equals([1]));
        expect(ndList[i].item, equals(data[i]));
      }
    });

    test('Shape of axis-j index, j=0,1,2 : 3D', () {
      final array = NumNDList.zeros<double>([3, 4, 2]);

      for (var i = 0; i < array.shape[0]; i++) {
        final slice = array[i];
        expect(slice.shape, equals([1, 4, 2]));
      }

      for (var i = 0; i < array.shape[1]; i++) {
        final slice = array[[':', i]];
        expect(slice.shape, equals([3, 1, 2]));
      }

      for (var i = 0; i < array.shape[2]; i++) {
        final slice = array.slice(i, i + 1, axis: 2);
        expect(slice.shape, equals([3, 4, 1]));
      }
    });
    test('Shape of axis-j index, j=0,1 : 2D', () {
      final array = NumNDList.zeros<double>([3, 4]);

      for (var i = 0; i < 3; i++) {
        final slice = array[i];
        expect(slice.shape, equals([1, 4]));
      }

      for (var i = 0; i < 4; i++) {
        final slice = array[[':', i]];
        expect(slice.shape, equals([3, 1]));
      }
    });

    test('2d Indexing with int, axis 0', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);
      final ndList0 = NDList.from<double>([data[0]]);

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
          print(ndList[[i, j]]);
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
      final ndList0 = NDList.from<double>([data[0]]);
      expect(ndList0.shape, equals([1, length0]));
      expect(ndList[0], equals(ndList0));
      expect(ndList0[[0, 0]].item, equals(data[0][0]));
      expect(ndList0[[0, 1]].item, equals(data[0][1]));
      expect(ndList[[0, 0]].item, equals(data[0][0]));
      expect(ndList[[0, 1]].item, equals(data[0][1]));

      // 1d data
      final length1 = data[1].length;
      final ndList1 = NDList.from<double>([data[0]]);
      expect(ndList1.shape, equals([1, length1]));
      expect(ndList[0], equals(ndList1));
      expect(ndList1[[0, 0]].item, equals(data[0][0]));
      expect(ndList1[[0, 1]].item, equals(data[0][1]));
      expect(ndList[[0, 0]].item, equals(data[0][0]));
      expect(ndList[[0, 1]].item, equals(data[0][1]));
    });

    test('zeros: 2D', () {
      final shape = [3, 2];
      final ndList = NumNDList.zeros<double>(shape);

      expect(ndList.shape, equals(shape));
      for (var i = 0; i < shape[0]; i++) {
        expect(ndList[i].shape, equals([1, 2]));
        for (var j = 0; j < shape[1]; j++) {
          expect(ndList[[i, j]].item, equals(0.0));
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
          expect(zerosLike2x3[[i, j]].item, equals(0.0));
        }
      }
    });
  });

  group('Slicing tests', () {
    test('2d slice length 1, axis 1', () {
      final data = [
        [0.0, 1.0, 2.0],
        [3.0, 4.0, 5.0],
        [6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0]
      ];

      final ndList = NDList.from<double>(data);

      final ndList1 = ndList.slice(1, 2, axis: 1);

      final expectedData = [
        [1.0],
        [4.0],
        [7.0],
        [10.0]
      ];
      final expectedSlice = NDList.from<double>(expectedData);

      expect(ndList1.shape, equals(expectedSlice.shape));

      expect(ndList1, equals(expectedSlice));
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

      // // axis 1
      final axis1Slice = testND[[':', ':1']];
      expect(axis1Slice.shape, equals([2, 1, 3]),
          reason: 'axis 1 slice shape, start-axis slice');
      final testSlice2 = testND[[':2', ':1']]; // redundant :2, equiv to above
      expect(testSlice2.shape, equals([2, 1, 3]),
          reason: 'axis 1 slice shape, redundant :2');
      final testSlice = testND[[':', '1:3']];
      expect(testSlice.shape, equals([2, 2, 3]),
          reason: 'axis 1 slice shape, mid-axis slice');

      // axis 2
      final axis2Slice = testND[[':', ':', ':1']];
      expect(axis2Slice.shape, equals([2, 4, 1]),
          reason: 'axis 2 slice shape, start-axis slice');
      final iteratedSlice = testSlice[[':', ':', '1:3']];
      expect(iteratedSlice.shape, equals([2, 2, 2]),
          reason: 'axis 2 slice shape, mid-axis slice');
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

    // if (start == end) {...}
    test('start == end (Empty slice)', () {
      // 1 d
      final nd1d = NDList.from<double>([1.0, 2.0, 3.0, 4.0]);
      for (int i = 0; i < nd1d.shape[0]; i++) {
        final empty1d = nd1d.slice(i, i);
        expect(empty1d.count, 0);
        expect(empty1d.shape, [0]);
      }

      // 2 d
      final nd2d = NDList.from<double>([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ]);

      for (int axis = 0; axis < nd2d.nDims; axis++) {
        for (int i = 0; i < nd2d.shape[axis]; i++) {
          final slice = nd2d.slice(i, i, axis: axis);
          expect(slice.count, 0);
          expect(slice.shape, [0]);
        }
      }

      // 3 d
      final nd3d = NumNDList.zeros<double>([2, 3, 4]);

      for (int axis = 0; axis < nd3d.nDims; axis++) {
        for (int i = 0; i < nd3d.shape[axis]; i++) {
          final slice = nd3d.slice(i, i, axis: axis);
          expect(slice.count, 0);
          expect(slice.shape, [0]);
        }
      }
    });

    // if (axis > 0 && priorResult.shape.contains(1)) {...}
    test('axis > 0 && priorResult.shape[0] == 1', () {
      final ndList = NumNDList.zeros([1, 2, 3]);

      final sliceAxis1 = ndList.slice(0, 1, axis: 1);
      final sliceAxis2 = ndList.slice(0, 1, axis: 2);

      expect(sliceAxis1.shape, equals([1, 1, 3]),
          reason: "Axis 1 slice has the wrong shape");
      expect(sliceAxis2.shape, equals([1, 2, 1]),
          reason: "Axis 2 slice has the wrong shape");
    });

    test('axis > 0 && priorResult.shape[1] == 1', () {
      final ndList = NumNDList.zeros([2, 1, 3]);

      final sliceAxis1 = ndList.slice(0, 1, axis: 1);
      final sliceAxis2 = ndList.slice(0, 1, axis: 2);

      expect(sliceAxis1.shape, equals([2, 1, 3]),
          reason: "Axis 1 slice has the wrong shape");
      expect(sliceAxis2.shape, equals([2, 1, 1]),
          reason: "Axis 2 slice has the wrong shape");
    });

    test('axis > 0 && priorResult.shape[2] == 1', () {
      final ndList = NumNDList.zeros([2, 3, 1]);

      final sliceAxis1 = ndList.slice(0, 1, axis: 1);
      final sliceAxis2 = ndList.slice(0, 1, axis: 2);

      expect(sliceAxis1.shape, equals([2, 1, 1]),
          reason: "Axis 1 slice has the wrong shape");
      expect(sliceAxis2.shape, equals([2, 3, 1]),
          reason: "Axis 2 slice has the wrong shape");
    });

    // if (axis > priorResult.shape.length - 1) {...}
    test('Axis too large', () {
      final ndList = NumNDList.zeros([2, 3, 4]);

      expect(() => ndList.slice(0, 1, axis: 3), throwsArgumentError);
      expect(() => ndList.slice(0, 1, axis: 4), throwsArgumentError);
    });

    // if (start == 0 && end == shape[axis]) {...}
    // this is tested elsewhere via full slice tests, but just to do an really thorough check in here.
    test('Full slice, 3D', () {
      final ndList = NumNDList.zeros([2, 3, 4]);

      final sliceStrings = <String>[];

      // test we can use full-slice syntax on 3D list, every axis
      for (var i = 0; i < ndList.nDims; i++) {
        final fullSlice = ndList.slice(0, ndList.shape[i], axis: i);
        expect(fullSlice.shape, equals([2, 3, 4]));
        expect(fullSlice, equals(ndList));

        // [] => [':'] => [':', ':'] => [':', ':', ':']
        sliceStrings.add(':');
        final fullSlice2 = ndList[sliceStrings];
        expect(fullSlice2.shape, equals([2, 3, 4]));
        expect(fullSlice2, equals(ndList));
      }
    });
  });

  group('Arithmetic', () {
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
          expect(sum[[i, j]].item, equals(data1[i][j] + data2[i][j]));
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
          expect(sum[[i, j]].item, equals(data1[i][j] * data2[i][j]));
        }
      }
    });
  });

  group('Transpose', () {
    test('Transpose 2D', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ];
      final ndList = NDList.from<double>(data);

      final transposed = ndList.transpose();

      expect(transposed.shape, equals([3, 2]));

      for (var i = 0; i < 3; i++) {
        for (var j = 0; j < 2; j++) {
          expect(transposed[[i, j]].item, equals(data[j][i]));
        }
      }
    });
  });
  group('Test errors', () {
    test('Test int indexing. Throw error only if out of bounds', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);
      // single int input
      expect(ndList[0], equals(NDList.from<double>([data[0]])));
      expect(ndList[1], equals(NDList.from<double>([data[1]])));
      expect(() => ndList[2], throwsRangeError);
      expect(ndList[-1], equals(NDList.from<double>([data[1]])));
      expect(ndList[-2], equals(NDList.from<double>([data[0]])));
    });
  });

  // Test min max abs, etc.
  group('Test min max abs', () {
    test('Test min', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      expect(ndList.min(), equals(1.0));
    });

    test('Test max', () {
      final data = [
        [1.0, 2.0, 4.0],
        [3.0, 4.0, 16.0]
      ];
      final ndList = NDList.from<double>(data);

      expect(ndList.max(), equals(16.0));
    });

    test('Test abs', () {
      final data = [
        [-1.0, -2.0, -4.0],
        [-3.0, -4.0, -16.0]
      ];
      final ndList = NDList.from<double>(data);

      expect(ndList.abs().sum().item!, equals(30.0));
      expect(ndList.abs()[[0, 0]].item, equals(1.0));
      expect(ndList.abs()[[1, 2]].item, equals(16.0));
    });
  });

  //Test slicesAlongAxis
  group('slicesAlongAxis', () {
    test('slices along axis 0 of 2D array', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ];
      final nd = NDList.from<double>(data);
      final slices = NDList.slicesAlongAxis(nd, 0);
      expect(slices.length, 3);
      expect(slices[0], equals(NDList.from<double>([1.0, 4.0])));
      expect(slices[1], equals(NDList.from<double>([2.0, 5.0])));
      expect(slices[2], equals(NDList.from<double>([3.0, 6.0])));
    });

    test('slices along axis 1 of 2D array', () {
      final data = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ];
      final nd = NDList.from<double>(data);
      final slices = NDList.slicesAlongAxis(nd, 1);
      expect(slices.length, 2);
      expect(slices[0], equals(NDList.from<double>([1.0, 2.0, 3.0])));
      expect(slices[1], equals(NDList.from<double>([4.0, 5.0, 6.0])));
    });

    test('slices along axis 0 of 3D array', () {
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
      final slices = NDList.slicesAlongAxis(nd, 0);
      expect(slices.length, 12);
      expect(slices[0], equals(NDList.from<double>([1.0, 13.0])));
      expect(slices[1], equals(NDList.from<double>([2.0, 14.0])));
      expect(slices[2], equals(NDList.from<double>([3.0, 15.0])));
      expect(slices[3], equals(NDList.from<double>([4.0, 16.0])));
      expect(slices[4], equals(NDList.from<double>([5.0, 17.0])));
      expect(slices[5], equals(NDList.from<double>([6.0, 18.0])));
      expect(slices[6], equals(NDList.from<double>([7.0, 19.0])));
      expect(slices[7], equals(NDList.from<double>([8.0, 20.0])));
      expect(slices[8], equals(NDList.from<double>([9.0, 21.0])));
      expect(slices[9], equals(NDList.from<double>([10.0, 22.0])));
      expect(slices[10], equals(NDList.from<double>([11.0, 23.0])));
      expect(slices[11], equals(NDList.from<double>([12.0, 24.0])));
    });
    test('slices along axis 1 of 3D array', () {
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
      final slices = NDList.slicesAlongAxis(nd, 1);
      expect(slices.length, 8);
      expect(slices[0], equals(NDList.from<double>([1.0, 5.0, 9.0])));
      expect(slices[1], equals(NDList.from<double>([2.0, 6.0, 10.0])));
      expect(slices[2], equals(NDList.from<double>([3.0, 7.0, 11.0])));
      expect(slices[3], equals(NDList.from<double>([4.0, 8.0, 12.0])));
      expect(slices[4], equals(NDList.from<double>([13.0, 17.0, 21.0])));
      expect(slices[5], equals(NDList.from<double>([14.0, 18.0, 22.0])));
      expect(slices[6], equals(NDList.from<double>([15.0, 19.0, 23.0])));
      expect(slices[7], equals(NDList.from<double>([16.0, 20.0, 24.0])));
    });

    test('slices along axis 2 of 3D array', () {
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
      final slices = NDList.slicesAlongAxis(nd, 2);
      expect(slices.length, 6);
      expect(slices[0], equals(NDList.from<double>([1.0, 2.0, 3.0, 4.0])));
      expect(slices[1], equals(NDList.from<double>([5.0, 6.0, 7.0, 8.0])));
      expect(slices[2], equals(NDList.from<double>([9.0, 10.0, 11.0, 12.0])));
      expect(slices[3], equals(NDList.from<double>([13.0, 14.0, 15.0, 16.0])));
      expect(slices[4], equals(NDList.from<double>([17.0, 18.0, 19.0, 20.0])));
      expect(slices[5], equals(NDList.from<double>([21.0, 22.0, 23.0, 24.0])));
    });

    test('throws on invalid axis', () {
      final nd = NDList.from<double>([1.0, 2.0, 3.0]);
      expect(() => NDList.slicesAlongAxis(nd, -1), throwsArgumentError);
      expect(() => NDList.slicesAlongAxis(nd, 1), throwsArgumentError);
    });
  });

  // Test reduceAlongAxis
  group('Test reduceAlongAxis', () {
    test('sum along axis 0 of 3D array', () {
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
      final reduced = nd.reduceAlongAxis((slice) {
        return slice.sum().item!;
      }, axis: 0);
      expect(reduced.shape, equals([3, 4]));
      expect(reduced[0].squeeze(),
          equals(NDList.from<double>([14.0, 16.0, 18.0, 20.0])));
      expect(reduced[1].squeeze(),
          equals(NDList.from<double>([22.0, 24.0, 26.0, 28.0])));
      expect(reduced[2].squeeze(),
          equals(NDList.from<double>([30.0, 32.0, 34.0, 36.0])));
    });

    test('sum along axis 1 of 3D array', () {
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
      final reduced = nd.reduceAlongAxis((slice) {
        return slice.sum().item!;
      }, axis: 1);
      expect(reduced.shape, equals([2, 4]));
      expect(reduced[0].squeeze(),
          equals(NDList.from<double>([15.0, 18.0, 21.0, 24.0])));
      expect(reduced[1].squeeze(),
          equals(NDList.from<double>([51.0, 54.0, 57.0, 60.0])));
    });
    test('sum along axis 2 of 3D array', () {
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
      final reduced = nd.reduceAlongAxis((slice) {
        return slice.sum().item!;
      }, axis: 2);
      expect(reduced.shape, equals([2, 3]));
      expect(reduced[0].squeeze(),
          equals(NDList.from<double>([10.0, 26.0, 42.0])));
      expect(reduced[1].squeeze(),
          equals(NDList.from<double>([58.0, 74.0, 90.0])));
    });
  });
}
