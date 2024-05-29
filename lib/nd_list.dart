import "package:tflite_flutter/tflite_flutter.dart";

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
    // // pretty print the array
    // if (_shape.isEmpty) {
    //   return '[]';
    // }
    // if (_shape.length == 1) {
    //   return _list.toString();
    // }
    // return '[${[
    //   for (var i = 0; i < _shape[0]; i++) this[i].toString()
    // ].join('\n ')}]\n';
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
      // shape.insert(0, list[0].length);
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

  (int, int)? _parseSlice(String slice) {
    try {
      if (slice.isEmpty) {
        return (0, 0);
      }
      // ':' => parts == ['', ''] => start = 0, end = _shape[0]
      // '1:' => parts == ['1', ''] => start = 1, end = _shape[0]
      // ':2' => parts == ['', '2'] => start = 0, end = 2
      final parts = slice.split(':');
      final start = parts[0].isEmpty ? 0 : int.parse(parts[0]);
      final end = parts[1].isEmpty ? _shape[0] : int.parse(parts[1]);
      return (start, end);
    } catch (e) {
      return null;
    }
  }

  NDList<X> operator [](index) {
    return _compoundIndexWithEnumeration(index).$2;
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
    final indicesAndSliceToEdit = this._compoundIndexWithEnumeration(index);
    final sliceToEdit = indicesAndSliceToEdit.$2;
    final editIndices = indicesAndSliceToEdit.$1;
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
    for (var i = 0; i < repeatedValue.count; i++) {
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
  EnumeratedSliceResult<X> _compoundIndexWithEnumeration(index) {
    if (_list.isEmpty) {
      throw ArgumentError('Empty NDList, cannot index.');
    }
    if (index is List) {
      return _listIndex(index);
    } else if (index is int) {
      return _intIndex(index);
    } else if (index is String) {
      return _stringIndex(index, 0);
    } else {
      throw ArgumentError('Invalid index');
    }
  }

  EnumeratedSliceResult<X> _stringIndex(String index, int axis) {
    try {
      // is it just an int in string format?
      // .parse throws if cannot be parsed as an int
      return this._intIndexWithAxis(int.parse(index), axis);
    } catch (e) {
      // just move on, it's not an int
    }
    final parsed = _parseSlice(index);
    if (parsed == null) {
      throw ArgumentError('Invalid slice');
    }
    return this._slice(parsed.$1, parsed.$2, axis: axis);
  }

  /// This method is used to index the NDList with a list of valid indices, i.e. ints and formatted slice strings.
  EnumeratedSliceResult<X> _listIndex(List index) {
    if (index.length == 1 && index[0] is int) {
      return this._intIndex(index[0]);
    } else if (index.length == 1 && index[0] is String) {
      return this._stringIndex(index[0], 0);
    }
    var _listIndex = <int>[];
    var sliced = this;
    late EnumeratedSliceResult<X> nextSlicing;
    for (var i = 0; i < index.length; i++) {
      if (index[i] is String) {
        print("string index: ${index[i]}");
        nextSlicing = sliced._stringIndex(index[i], i);
      } else if (index[i] is int) {
        print("int index: ${index[i]}");
        nextSlicing = sliced._intIndexWithAxis(index[i], i);
      } else {
        throw ArgumentError(
            'Invalid index, "${index[i]}" in position $i is not an int or a string.');
      }
      sliced = nextSlicing.$2;
      if (_listIndex.isEmpty) {
        _listIndex = nextSlicing.$1;
        continue;
      }
      _listIndex = [for (int i in nextSlicing.$1) _listIndex[i]];
    }
    return (_listIndex, sliced);
  }

  EnumeratedSliceResult<X> _intIndex(int index) {
    if (_shape.isEmpty) {
      throw ArgumentError('Cannot index an empty NDList');
    }
    while (index < 0) {
      // -1 => _shape[0] - 1 (aka last element)
      // -2 => second last element, etc.
      index += _shape[0];
    }
    // error handling
    if (index >= _shape[0]) {
      throw RangeError(
          'Index out of bounds: index $index is out of bounds for axis with size ${_shape[0]}');
    }
    // return the appropriate axis-0 slice
    if (_shape.length == 1) {
      return ([index], NDList._([_list[index]], [1]));
    }
    final returnShape = _shape.sublist(1);
    final subLength = _product(returnShape);
    final theSlice = NDList._(
        _list.sublist(index * subLength, (index + 1) * subLength), returnShape);
    final _listIndex = List.generate(subLength, (i) => index * subLength + i);
    return (_listIndex, theSlice);
  }

  /// This builds on the base case of an axis-0 int index, and allows for indexing on any axis.
  EnumeratedSliceResult<X> _intIndexWithAxis(int index, int axis) {
    return _slice(index, index + 1, axis: axis);
    if (_shape.isEmpty) {
      throw ArgumentError('Cannot index an empty NDList');
    }
    if (axis == 0) {
      return _intIndex(index);
    }
    if (axis < 0 || axis >= _shape.length) {
      throw ArgumentError('Invalid axis $axis for shape $_shape');
    }
    final shapeAfter = _shape.sublist(axis + 1);
    final shapeBefore = _shape.sublist(0, axis);
    final returnShape = [...shapeBefore, ...shapeAfter];

    // figure out the starting point for our indices
    final firstIndex = index * _product(shapeAfter);

    // The indices appear in blocks of consecutive values, corresponding to fully enumerating the values between the start and end of a single off-axis slice
    final sliceStep = _shape[axis] * _product(shapeAfter);

    final singleAxisElements = List.generate(
        _product(shapeAfter), (innerIdx) => firstIndex + innerIdx);
    final indicesOnThisList = [
      for (int i = 0; i < _product(shapeBefore); i++)
        ...singleAxisElements.map((e) => e + i * sliceStep)
    ];

    print(indicesOnThisList);

    return (
      indicesOnThisList,
      NDList._([for (int i in indicesOnThisList) _list[i]], returnShape)
    );
  }

  NDList<X> slice(int start, int end, {int axis = 0}) {
    return _slice(start, end, axis: axis).$2;
  }

  EnumeratedSliceResult<X> _slice(int start, int end, {int axis = 0}) {
    if (start < 0) {
      start += _shape[axis];
    }
    if (end < 0) {
      end += _shape[axis];
    }
    if (end < start) {
      return this._slice(end, start, axis: axis);
    }
    if (end == start) {
      return ([], NDList._([], [0]));
    }
    if (_shape.length == 1) {
      final sliceEnd = end > _shape[0] ? _shape[0] : end;
      final sliceLength = sliceEnd - start;
      return (
        List.generate(sliceLength, (i) => start + i),
        NDList._(_list.sublist(start, sliceEnd), [sliceLength])
      );
    }

    if (axis > _shape.length - 1) {
      throw ArgumentError(
          'Invalid axis $axis for ${_shape.length}D list with shape $_shape');
    }

    if (axis == 0) {
      final sliceStep = _product(_shape.sublist(1));
      final sliceEnd = end > _shape[0] ? _shape[0] : end;
      final sliceLength = sliceEnd - start;
      final _listIndices =
          List.generate(sliceLength * sliceStep, (i) => start * sliceStep + i);
      return (
        _listIndices,
        // we know (_shape.length >= 2) since checked == 1 above
        NDList._([for (int i in _listIndices) _list[i]],
            [sliceLength, ..._shape.sublist(1)])
      );
    }

    // now, build a NDList<NDList<X>>, where each element has the same shape
    // Then we will use .cemented() to get an NDList<X> with the expected shape
    final subtensorIndices = _enumerateSubtensors(axis);
    final indicesAndSubTensors = subtensorIndices
        .map((e) => this._compoundIndexWithEnumeration(e))
        .map((indicesAndSubtensor) {
      final (indices, subtensor) = indicesAndSubtensor;
      // The indices we get back here are actually in reference to the subtensor.
      final (sliceIdx, subSlice) = subtensor._slice(start, end, axis: axis - 1);
      // To convert these back to indices on `this._list` we now map sliceIdx[i] to indices[slideIdx[i]], since `indices` tells us which `this._list` elements went into the subtensor..
      final listIndices = sliceIdx.map((i) => indices[i]).toList();
      return (listIndices, subSlice);
    }).toList();

    final subTensors = indicesAndSubTensors.map((e) => e.$2).toList();
    final indicesOnThisList =
        indicesAndSubTensors.expand((e) => e.$1).toSet().toList()..sort();

    final shapeBeforeAxis = _shape.sublist(0, axis);
    return (
      indicesOnThisList,
      NDList.from<NDList<X>>(subTensors).reshape(shapeBeforeAxis).cemented()
    );
  }

  /// !! (Remember we index the axes from 0.)
  /// This function returns the cartesian product of [0, 1, ..., _shape[i]-1] for each `i < belowAxis`.
  ///
  /// Q: Why do we want this function?
  /// You can view many operations on NDList as being done on a "flattened" version of the NDList.
  ///
  /// Suppose `this` has shape [2, 3, 4, 5] and we want to do something on axis 2, which has size 4.
  /// We can view this as an operation to each element of a [2, 3] NDList, where each element is a [4, 5] NDList.
  List<List<int>> _enumerateSubtensors(int belowAxis) {
    // [[0], [1], [2], ...] for each axis
    // eg shape == [2, 4, 3],
    // [[0], [1]]
    // [[0], [1], [2], [3]]
    // [[0], [1], [2]]
    final axisEnums = [
      for (int shapeIndex = 0; shapeIndex < belowAxis; shapeIndex++)
        [
          for (int i = 0; i < _shape[shapeIndex]; i++) [i]
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

    final removed1s = _shape.where((element) => element != 1).toList();
    return reshape(removed1s);
  }
}
