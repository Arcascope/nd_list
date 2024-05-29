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
import 'package:tflite_flutter/tflite_flutter.dart';

List<int> unsqueezeShape(List<int> shape, int axis) {
  if (axis < 0) {
    axis += shape.length + 1;
  }
  return [...shape.sublist(0, axis), 1, ...shape.sublist(axis)];
}

List<int> squeezeShape(List<int> shape) {
  return shape.where((element) => element != 1).toList();
}

int getLinearIndex(List<int> shape, List<int> index) {
  if (shape.length != index.length) {
    throw ArgumentError('Shape and index must have the same length');
  }

  int linearIndex = 0;
  int size = 1;

  for (int i = shape.length - 1; i >= 0; i--) {
    linearIndex += index[i] * size;
    size *= shape[i];
  }

  return linearIndex;
}

List<int> range(int count) {
  return List.generate(count, (index) => index);
}

/// Provides a lazy index result that can be resolved to a concrete NDList using `.evaluate()`.
///
/// This lets us quickly and reliably reduce complex indices to simples pieces, and also lets us build the []= algorithm on top of the [] algorithm.
class NDIndexResult<X> {
  NDList<X> parent;
  List<int> parentIndices;
  List<int> shape;

  NDIndexResult(this.parent, this.parentIndices, this.shape);

  /// Used when processing an index via intermediate steps.
  ///
  /// We got `this` from an index, so it references a subtensor of `parent`. When we do the next step in the index, we are putting that index on the subtensor. This method resolves the indices on the subtensor to the indices on the parent tensor.
  NDIndexResult<X> resolveStep(List<int> subtensorIndices, List<int> newShape) {
    print("Resolving step: new shape $newShape");
    print("finding indices: $subtensorIndices from $parentIndices");
    final newParentIndices = [for (int i in subtensorIndices) parentIndices[i]];
    return NDIndexResult(parent, newParentIndices, newShape);
  }

  NDList<X> evaluate() {
    return NDList._([for (int i in parentIndices) parent._list[i]], shape);
  }

  static NDIndexResult<Y> from<Y>(NDList<Y> ndList) {
    return NDIndexResult(ndList, range(ndList.count), ndList.shape);
  }
}

typedef EnumeratedSliceResult<X> = (List<int>, NDList<X>);

/// Wrapper on multi-dimensional lists to provide easier indexing and slicing.
/// This class is inspired by NumPy's ndarray.
///
/// Example:
/// ```dart
/// final data = [
///  [1.0, 2.0, 3.0],
/// [4.0, 5.0, 6.0],
/// [7.0, 8.0, 9.0]
/// ];
/// final ndList = NDList.from<double>(data);
///
/// final sliced = ndList[['1:3', '0:2']];
/// ```
/// That is, you can use Python-like slice syntax to access elements.
/// In the end, `sliced` would represent `[[4.0, 5.0], [7.0, 8.0]]`.
class NDList<X> {
  final List<X> _list = [];
  final List<int> _shape = [];

  List toIteratedList() {
    return _list.reshape(shape);
  }

  @override
  String toString() {
    return toIteratedList().toString();
  }

  NDList.empty() {
    _shape.add(0);
  }

  NDList.filled(List<int> shape, X fill) {
    _list.addAll(List<X>.filled(NDList._product(shape), fill));
    _shape.addAll(shape);
  }

  static NDList<X> stacked<X>(List<NDList<X>> ndLists, {int axis = 0}) {
    return NDList.from<NDList<X>>(ndLists).cemented();
  }

  static NDList<E> from<E>(List multiList) {
    if (multiList.isEmpty) {
      return NDList.empty();
    }
    var shape = [multiList.length];
    var list = multiList;
    while (list[0] is List) {
      final raggedElementIndex =
          list.indexWhere((element) => element.length != list[0].length);
      if (raggedElementIndex != -1) {
        throw ArgumentError(
            'Ragged array detected! First ragged index: $raggedElementIndex, which has ${list[raggedElementIndex].length} elements, but the 0th element has ${list[0].length}');
      }
      shape.add(list[0].length);
      list = list.expand((element) => element).toList();
    }

    if (_product(shape) != list.length) {
      throw ArgumentError('Ragged array detected!.');
    }

    final typedList = list.whereType<E>().toList();

    if (_product(shape) == typedList.length) {
      return NDList._(typedList, shape);
    } else {
      throw ArgumentError('Invalid list');
    }
  }

  NDList._(List<X> list, List<int> shape, {X? fill}) {
    if (_product(shape) != list.length && fill == null) {
      throw ArgumentError('Shape does not match the length of the list');
    }
    _list.addAll(list);
    if (fill != null && _product(shape) > list.length) {
      _list.addAll(List<X>.filled(_product(shape) - list.length, fill));
    }
    _shape.addAll(shape);
  }

  static int _product(List<int> list) {
    return list.fold(1, (a, b) => a * b);
  }

  /// By default, if nd ~ [1, 2] and we call nd[0], it actually returns a wrapped list, [1].
  /// To get 1 itself, call .item (and check this is not null)
  X? get item => _list.length == 1 ? _list[0] : null;

  int get count => _list.length;
  int get length => _shape.isNotEmpty ? _shape[0] : -1;

  List<int> get shape => _shape;
  int get nDims => _shape.length;

  /// This method checks if the shapes are equal element-wise.
  ///
  /// Checking `shape == other.shape` does not work because lists are not equal based on element-wise comparison.
  bool _shapeMatches(NDList other) {
    if (shape.length != other.shape.length) {
      return false;
    }
    for (var i = 0; i < shape.length; i++) {
      if (shape[i] != other.shape[i]) {
        return false;
      }
    }
    return true;
  }

  NDList<Y> map<Y>(Y Function(X) f) {
    return NDList._(_list.map(f).toList(), _shape);
  }

  static (int, int?)? _parseSlice(String slice) {
    try {
      if (slice.isEmpty) {
        return (0, 0);
      }
      // ':' => parts == ['', ''] => start = 0, end = _shape[0]
      // '1:' => parts == ['1', ''] => start = 1, end = _shape[0]
      // ':2' => parts == ['', '2'] => start = 0, end = 2
      final parts = slice.split(':');
      final start = parts[0].isEmpty ? 0 : int.parse(parts[0]);
      final end = parts[1].isEmpty ? null : int.parse(parts[1]);
      return (start, end);
    } catch (e) {
      return null;
    }
  }

  NDList<X> operator [](index) {
    if (_list.isEmpty) {
      throw ArgumentError('Empty NDList, cannot index.');
    }
    return _compoundIndexWithEnumeration(shape, index).evaluate();
  }

  void operator []=(index, value) {
    if (_shape.isEmpty) {
      throw ArgumentError('Cannot index an empty NDList');
    }

    // interpret X as a [1] shaped NDList
    if (value is X) {
      this[index] = NDList.from<X>([value]);
      return;
    }
    // if we made it this far, then value is not an X.
    // (possibly we're inside the recursive call above)
    if (value is! NDList<X>) {
      throw ArgumentError(
          'Invalid value type $value, must be ${_list.first.runtimeType}');
    }

    // this gives a subtensor whose elements can be modified
    // and are the same objects as in this._list
    // So, when we edit elements of this sub-tensor we are modifying the original too.
    final indicesAndSliceToEdit =
        this._compoundIndexWithEnumeration(shape, index);
    final editIndices = indicesAndSliceToEdit.parentIndices;
    final sliceToEdit = indicesAndSliceToEdit.evaluate();
    final paddedValueShape =
        _padShape(shape: value.shape, toMatch: sliceToEdit.shape);
    value = value.reshape(paddedValueShape);
    final sizeDivisor = _sizeDivisor(sliceToEdit.shape, value.shape);
    if (sizeDivisor.any((element) => element < 1)) {
      throw ArgumentError(
          '[]= error: Shape of indexed subtensor ${sliceToEdit.shape} and RHS ${value.shape} are incompatible. Each RHS shape dimension must evenly divide the LHS.');
    }

    // now, we can iterate over the elements of the value and assign them to the slice
    final repeatedValue = NDList.filled(sizeDivisor, value).cemented();
    for (var i = 0; i < editIndices.length; i++) {
      _list[editIndices[i]] = repeatedValue._list[i];
    }
  }

  /// Sometimes shapes are "incompatible" for trivial reasons, like one is [4, 1] and the other is [4]. This figures out the simplest way to make them compatible, if one exists.
  ///
  /// Examples:
  /// 1. _padShape([4], [4, 1]) => [4, 1]
  /// 2. _padShape([4, 1], [4]) => ERROR
  /// 3. _padShape([2, 3], [1, 2, 1, 3]) => [1, 2, 1, 3]
  static List<int> _padShape(
      {required List<int> shape, required List<int> toMatch}) {
    if (shape.length > toMatch.length) {
      throw ArgumentError('shape must have no more dimensions than toMatch');
    }
    final matches = <int>[];
    final paddedShape = List.filled(toMatch.length, 1);
    for (int dim in shape) {
      // important to start searching _after_ the last one
      // eg: [1, 4, 1, 4].indexOf(4) => 1, but we want [1, 3] instead of [1, 1]
      final nextIndex = toMatch.indexWhere(
          (e) => e % dim == 0, matches.isEmpty ? 0 : matches.last + 1);
      if (nextIndex == -1) {
        throw ArgumentError('shape must be a subset of toMatch');
      }
      paddedShape[nextIndex] = dim;
      matches.add(nextIndex);
    }

    return paddedShape;
  }

  static const int _divisorSizeError = -999;

  /// Returns the integer division of the shapes provided.
  ///
  /// We can detect things about the relative shape based on the values returned:
  /// - If the shapes are not divisible, the error value `_divisorSizeError` is returned.
  /// - If `shape2` is larger than `shape1` in some dimension, the returned shape divisor is 0.
  ///
  /// This is useful for stacking together numerous NDLists, each with shape `shape2`, to make a larger NDList with shape `shape1`.
  ///
  /// Example:
  /// If `tensor1` has shape `[8, 4]`, we can view this as `2 x 2` grid of `[4, 2]` shaped subtensors. In this cases, `_sizeDivisor([8, 4], [4, 2])` would return `[2, 2]`.
  static List<int> _sizeDivisor(List<int> shape1, List<int> shape2) {
    if (shape1.length != shape2.length) {
      try {
        shape2 = _padShape(shape: shape2, toMatch: shape1);
      } catch (e) {
        throw ArgumentError(
            'Shapes must have the same length, up to adding 1s');
      }
    }
    // if any dimension is not divisible, record that error
    return [
      for (var i = 0; i < shape1.length; i++)
        (shape1[i] % shape2[i] == 0)
            ? shape1[i] ~/ shape2[i]
            : _divisorSizeError
    ];
  }

  /// Returns the expected result from any accepted index, as well as the indices on _list that correspond to its elements.
  NDIndexResult<X> _compoundIndexWithEnumeration(List<int> shape, index) {
    NDIndexResult<X> priorResult = NDIndexResult.from(this);
    if (index is List) {
      return _listIndex<X>(priorResult, index);
    } else if (index is int) {
      return _intIndex(priorResult, index);
    } else if (index is String) {
      return _stringIndex(priorResult, index, 0);
    } else {
      throw ArgumentError('Invalid index');
    }
  }

  static NDIndexResult<Y> _stringIndex<Y>(
      NDIndexResult<Y> priorResult, String index, int axis) {
    print('string index: "$index"');
    try {
      // is it just an int in string format?
      // .parse throws if cannot be parsed as an int
      return _intIndexWithAxis(priorResult, int.parse(index), axis);
    } catch (e) {
      // just move on, it's not an int
    }
    final parsed = _parseSlice(index);
    if (parsed == null) {
      throw ArgumentError('Invalid slice');
    }
    return _slice(priorResult, parsed.$1, parsed.$2 ?? priorResult.shape[axis],
        axis: axis);
  }

  /// This method is used to index the NDList with a list of valid indices, i.e. ints and formatted slice strings.
  static NDIndexResult<X> _listIndex<X>(
      NDIndexResult<X> priorResult, List index) {
    print('list index: $index');
    for (var i = 0; i < index.length; i++) {
      if (index[i] is String) {
        print('string index: "${index[i]}"');
        priorResult = _stringIndex(priorResult, index[i], i);
      } else if (index[i] is int) {
        print("int index: ${index[i]}");
        priorResult = _intIndexWithAxis(priorResult, index[i], i);
      } else {
        throw ArgumentError(
            'Invalid index, "${index[i]}" in position $i is not an int or a string.');
      }
    }
    return priorResult;
  }

  static NDIndexResult<X> _intIndex<X>(
      NDIndexResult<X> priorResult, int index) {
    print('int index: $index');
    if (priorResult.shape.isEmpty) {
      throw ArgumentError('Cannot index an empty NDList');
    }

    while (index < 0) {
      // -1 => priorResult.shape[0] - 1 (aka last element)
      // -2 => second last element, etc.
      index += priorResult.shape[0];
    }
    // error handling
    if (index >= priorResult.shape[0]) {
      throw RangeError(
          'Index out of bounds: index $index is out of bounds for axis with size ${priorResult.shape[0]}');
    }
    // return the appropriate axis-0 slice
    if (priorResult.shape.length == 1) {
      return priorResult.resolveStep([index], [1]);
    }
    final returnShape = [1, ...priorResult.shape.sublist(1)];
    final subLength = _product(returnShape);
    final listIndex = List.generate(subLength, (i) => index * subLength + i);
    return priorResult.resolveStep(listIndex, returnShape);
  }

  /// This builds on the base case of an axis-0 int index, and allows for indexing on any axis.
  static NDIndexResult<X> _intIndexWithAxis<X>(
      NDIndexResult<X> priorResult, int index, int axis) {
    print('int index with axis: $index, axis $axis');
    return _slice(priorResult, index, index + 1, axis: axis);
  }

  NDList<X> slice(int start, int end, {int axis = 0}) {
    return _slice(NDIndexResult.from(this), start, end, axis: axis).evaluate();
  }

  static NDIndexResult<Y> _slice<Y>(
      NDIndexResult<Y> priorResult, int start, int end,
      {required int axis}) {
    // if (start < 0) {
    //   start += priorResult.shape[axis];
    // }
    // if (end < 0) {
    //   end += priorResult.shape[axis];
    // }
    // if (end < start) {
    //   return _slice(priorResult, end, start, axis: axis);
    // }
    if (end == start) {
      return priorResult.resolveStep([], []);
    }

    if (axis > 0 && priorResult.shape.contains(1)) {
      final indexOf1 = priorResult.shape.indexOf(1);
      final reducedShape = [
        for (var i in range(priorResult.shape.length)) priorResult.shape[i]
      ];
      reducedShape.removeAt(indexOf1);
      final newAxis = (axis > indexOf1) ? axis - 1 : axis;
      final newPrior = NDIndexResult(
          priorResult.parent, priorResult.parentIndices, reducedShape);
      final reducedSlice = _slice(newPrior, start, end, axis: newAxis);
      reducedSlice.shape.insert(indexOf1, 1);
      return reducedSlice;
    }
    if (axis > priorResult.shape.length - 1) {
      throw ArgumentError(
          'Invalid axis $axis for ${priorResult.shape.length}-D list with shape ${priorResult.shape}');
    }

    if (start == 0 && end == priorResult.shape[axis]) {
      return priorResult;
    }

    if (axis == 0) {
      // final singleSliceShape = priorResult.shape.sublist(1);
      // final sliceStep = _product(singleSliceShape);
      final sliceEnd = end > priorResult.shape[0] ? priorResult.shape[0] : end;
      // trivial slice
      final sliceLength = sliceEnd - start;
      final listIndices = [
        for (int i = start; i < sliceEnd; i++)
          ..._intIndex(priorResult, i).parentIndices
      ];

      final sliceShape = [sliceLength, ...priorResult.shape.sublist(1)];
      return NDIndexResult(priorResult.parent, listIndices, sliceShape);
    }

    final axis0 = [
      for (int i = 0; i < priorResult.shape[0]; i++) _intIndex(priorResult, i)
    ];

    final axis0Slices = axis0.map((e) => _slice(e, start, end, axis: axis));

    final resolvedIndices = axis0Slices.expand((e) => e.parentIndices).toList();
    final resolvedShape = [
      axis0Slices.length,
      ...axis0Slices.first.shape.sublist(1)
    ];

    return priorResult.resolveStep(resolvedIndices, resolvedShape);

    // final subtensorIndices = _enumerateSubtensors(priorResult.shape, axis);
    // final indicesAndSubTensors = subtensorIndices
    //     .map((e) => _listIndex(priorResult, e))
    //     .map((subtensorResult) {
    //   return _slice(subtensorResult, start, end, axis: axis - 1);
    // }).toList();

    // final shapeBeforeAxis = priorResult.shape.sublist(0, axis);

    // final resolvedIndices = indicesAndSubTensors.expand((e) => e.parentIndices);

    // return priorResult.resolveStep(resolvedIndices.toList(),
    //     shapeBeforeAxis + indicesAndSubTensors.first.shape);
  }

  /// !! (Remember we index the axes from 0.)
  /// This function returns the cartesian product of [0, 1, ..., _shape[i]-1] for each `i < belowAxis`.
  ///
  /// Q: Why do we want this function?
  /// You can view many operations on NDList as being done on a "flattened" version of the NDList.
  ///
  /// Suppose `this` has shape [2, 3, 4, 5] and we want to do something on axis 2, which has size 4.
  /// We can view this as an operation to each element of a [2, 3] NDList, where each element is a [4, 5] NDList.
  static List<List<int>> _enumerateSubtensors(List<int> shape, int belowAxis) {
    // [[0], [1], [2], ...] for each axis
    // eg shape == [2, 4, 3],
    // [[0], [1]]
    // [[0], [1], [2], [3]]
    // [[0], [1], [2]]
    final axisEnums = [
      for (int shapeIndex = 0; shapeIndex < belowAxis; shapeIndex++)
        [
          for (int i = 0; i < shape[shapeIndex]; i++) [i]
        ]
    ];

    // now take the cartesian product of axisEnums
    // eg   [[0], [1]]
    //    x [[0], [1], [2], [3]]
    //    x [[0], [1], [2]]

    final enumerated =
        axisEnums.fold<List<List<int>>>([[]], (previousValue, element) {
      return [
        for (var i = 0; i < previousValue.length; i++)
          for (var j = 0; j < element.length; j++)
            [...previousValue[i], ...element[j]]
      ];
    });

    return enumerated;
  }

  NDList<X> reshape(List<int> newShape) {
    if (newShape.where((element) => element == 0).isNotEmpty) {
      if (_list.isEmpty) return NDList._([], newShape);
      throw ArgumentError('New shape cannot have a dimension of 0');
    }
    final positiveDims = newShape.where((element) => element < 1).toList();
    if (positiveDims.length > 1) {
      throw ArgumentError('Only one dimension can be -1');
    }
    final nSpecified = _product(positiveDims);
    if (count % nSpecified != 0) {
      throw ArgumentError('New shape must have the same number of elements');
    }

    final otherAxis = count ~/ nSpecified;

    return NDList._(_list, newShape.map((e) => e < 1 ? otherAxis : e).toList());
  }

  NDList<X> flatten() {
    return NDList._(_list, [count]);
  }

  @override
  bool operator ==(Object other) {
    if (other is NDList) {
      // check if the shape and elements match
      if (!_shapeMatches(other)) {
        return false;
      }
      for (var i = 0; i < count; i++) {
        if (this._list[i] != other._list[i]) {
          return false;
        }
      }
      return true;
    }
    return false;
  }

  @override
  int get hashCode => _list.hashCode ^ _shape.hashCode;
}

/// Provides a number of useful extensions in the typical use case of NDList with numbers. This includes methods like `zeros`, `ones`, and element-wise operations.
///
/// In Dart, the `num` abstract class unifies `int` and `double`, so we work with each separately.
extension NumNDList on NDList {
  /// At time of writing, `X extends num` means that `X` is either an `int` or a `double`. Thus, we can just check if `X` is an `int` and return `0` or `0.0` accordingly.
  static X zero<X extends num>() {
    return (X is int) ? 0 as X : 0.0 as X;
  }

  /// Returns appropriate 1 for X's type. See docstring on `.zero<X>()` in this extension.
  static X one<X extends num>() {
    return (X is int) ? 1 as X : 1.0 as X;
  }

  /// Creates a new NDList with the provided shape and filled with zeros of the specified type.
  ///
  /// Thus, if you want a `NDList<int>` of shape `[2, 3]` filled with the integer `0`, you would call `NumNDList.zeros<int>([2, 3])`.
  ///
  /// If we call with `NumNDList.zeros<double>([2, 3])`, we would get a `NDList<double>` filled with `0.0` instead.
  static NDList<X> zeros<X extends num>(List<int> shape) {
    return NDList.filled(shape, NumNDList.zero());
  }

  /// Creates a new NDList with the same shape as the provided NDList and filled with zeros of the specified type.
  static NDList<X> zerosLike<X extends num>(NDList other) {
    return NumNDList.zeros(other.shape);
  }

  /// Creates a new NDList with the provided shape and filled with ones of the specified type.
  static NDList<X> ones<X extends num>(List<int> shape) {
    return NDList.filled(shape, NumNDList.one());
  }

  /// Creates a new NDList with the same shape as the provided NDList and filled with ones of the specified type.
  static NDList<X> onesLike<X extends num>(NDList other) {
    return NumNDList.ones<X>(other.shape);
  }
}

extension ArithmeticNDList<X extends num> on NDList<X> {
  NDList<X> zipWith(NDList<X> other, X Function(X, X) f) {
    if (!_shapeMatches(other)) {
      throw ArgumentError('Shapes do not match');
    }
    final result = NumNDList.zerosLike<X>(this);
    for (var i = 0; i < count; i++) {
      result._list[i] = f(_list[i], other._list[i]);
    }
    return result;
  }

  /// Element-wise addition of two NDLists.
  NDList<X> operator +(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 + p1) as X));
  }

  operator -(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 - p1) as X));
  }

  operator *(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 * p1) as X));
  }

  operator /(NDList<X> other) {
    return this.zipWith(other, ((p0, p1) => (p0 / p1) as X));
  }
}

extension MultiLinear<X> on NDList<NDList<X>> {
  NDList<X> flatten() {
    return NDList.from<X>(_list.expand((element) => element._list).toList());
  }

  /// When the elements are also NDList with the same shape and inner type X, they can form building blocks to a larger NDList<X> by basically "forgetting" the separations, cementing the vectors together into a big tensor.
  ///
  /// Suppose:
  /// * this `NDList` has `.shape == [a0, .... aM]` and
  /// * every elment is an `NDList<X>` with fixed shape `[b0, ... bN]`,
  ///
  /// then this method returns a new `NDList<X>` with shape `[a0, ... aN, b0, ... bN]`.
  ///
  /// Example to keep in mind: An matrix can be thought of as a grid of 1x1 matrices, but we can just "erase" the division between those 1x1s. More generally, this is basically viewing a matrix as a set of equal-sized submatrices. For 3D tensors, think wooden blocks being cemented together to form a larger block.
  ///
  /// Example:
  /// ```
  /// [[[1], [2], [3]],
  ///  [[4], [5], [6]]]
  /// ```
  ///
  /// turns into
  /// ```
  /// [[1, 2, 3],
  ///  [4, 5, 6]]
  /// ```
  /// using `.cemented()` followed by `.squeeze()` to remove the trivial dimension, giving shape `[2, 3]` instead of `[2, 3, 1]` that `.cemented()` returns.
  NDList<X> cemented() {
    if (_list.isEmpty) {
      return NDList<X>.empty();
    }
    final cementedShape = [..._shape, ..._list[0].shape];
    final cementedList = _list.expand((element) => element._list).toList();
    return NDList._(cementedList, cementedShape);
  }
}

extension Squeezing<X> on NDList<X> {
  NDList<X> squeeze() {
    if (nDims < 2) {
      return this;
    }

    return reshape(squeezeShape(_shape));
  }
}
