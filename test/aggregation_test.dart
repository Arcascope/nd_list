import 'package:nd_list/nd_list.dart';
import 'package:test/test.dart';

void main() {
  group('Summation of full arrays', () {
    test('Sum of (3,) array', () {
      NDList<double> x = NDList.from([1.0, 2.0, 3.0]);
      double sum = x.sum();
      expect(sum, equals(6.0));
    });
    test('Sum of (3, 1) array', () {
      NDList<double> x = NDList.from([
        [1.0],
        [2.0],
        [3.0]
      ]);
      double sum = x.sum();
      expect(sum, equals(6.0));
    });
    test('Sum of (1, 3) array', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0]
      ]);
      double sum = x.sum();
      expect(sum, equals(6.0));
    });

    test('Sum of (2, 3) array', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ]);
      double sum = x.sum();
      expect(sum, equals(21.0));
    });
  });

  group('Summation along a single axis', () {
    test('Sum of (3,) array along axis 0', () {
      NDList<double> x = NDList.from([1.0, 2.0, 3.0]);
      NDList<double> sum = x.sumAlong(axis: 0);
      expect(sum.shape, equals([3]));
      expect(sum, equals(x));
    });

    test('Sum of (3, 1) array along axis 0', () {
      NDList<double> x = NDList.from([
        [1.0],
        [2.0],
        [3.0]
      ]);
      NDList<double> sum = x.sumAlong(axis: 0);
      expect(sum.shape, equals([1, 1]));
      expect(sum.toFlattenedList(), equals([6.0]));
    });

    test('Sum of (1, 3) array along axis 0', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0]
      ]);
      NDList<double> sum = x.sumAlong(axis: 0);
      expect(sum.shape, equals([1, 1]));
      expect(sum.toFlattenedList(), equals([6.0]));
    });

    test('Sum of (2, 3) array along axis 0', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ]);
      NDList<double> sum = x.sumAlong(axis: 0);
      expect(sum.shape, equals([1, 3]));
      expect(sum.toFlattenedList(), equals([5.0, 7.0, 9.0]));
    });

    test('Sum of (2, 3) array along axis 1', () {
      NDList<double> x = NDList.from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
      ]);
      NDList<double> sum = x.sumAlong(axis: 1);
      expect(sum.shape, equals([2, 1]));
      expect(sum.toFlattenedList(), equals([6.0, 15.0]));
    });
  });

  group('Quantiles and median', () {
    test('Quantiles of (3,) array', () {
      NDList<double> x = NDList.from([1.0, 2.0, 3.0]);
      double quantile = x.quantile(0.0);
      expect(quantile, equals(1.0));

      // median, calculated by a different method
      quantile = x.quantile(0.5);
      expect(quantile, equals(2.0));

      // 0.5 quantile is split into 5 equal parts here
      // hence the 0.1 quantile is 1/5 of the way between 0th and 1st element
      quantile = x.quantile(0.1);
      expect(quantile, equals(1.2));

      // half way between 1st and 2nd element
      quantile = x.quantile(0.25);
      expect(quantile, equals(1.5));

      quantile = x.quantile(0.75);
      expect(quantile, equals(2.5));
    });

    test('50% quantile equals median', () {
      NDList<double> x = NDList.from([1.0, 2.0, 3.0]);
      double quantile = x.quantile(0.5);
      double median = x.median();
      expect(quantile, equals(median));
    });
  });
}
