
import 'dart:math';

import 'package:collection/collection.dart';

class MathUtil {
  static double dotProduct(List<double> a, List<double> b) {
    assert(a.length == b.length);
    double sum = 0.0;
    for (final (index, val) in a.indexed) {
      sum += val * b[index];
    }
    return sum;
  }

  static double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }

  static List<double> softmax(List<double> x) {
    final xMax = x.max;
    final expX = x.map((val) => exp(val - xMax));
    final expXSum = expX.reduce((a, b) => a + b);
    final result = expX.map((val) => val / expXSum).toList();
    return result;
  }

  static int argMax(List<double> x) {
    final maxVal = x.max;
    return x.indexOf(maxVal);
  }

}