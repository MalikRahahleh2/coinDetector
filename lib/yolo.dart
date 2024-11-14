import 'dart:math' as math;
import 'package:collection/collection.dart';

import 'math_util.dart';

class BoundingBox<T extends num> {
  T x1;
  T y1;
  T x2;
  T y2;

  BoundingBox(this.x1, this.y1, this.x2, this.y2);

  double iou(BoundingBox other) {
    final xA = math.max(x1, other.x1);
    final yA = math.max(y1, other.y1);
    final xB = math.min(x2, other.x2);
    final yB = math.min(y2, other.y2);

    final intersectionArea = math.max(0, xB - xA) * math.max(0, yB - yA);

    final area = (x2 - x1) * (y2 - y1);
    final otherArea = (other.x2 - other.x1) * (other.y2 - other.y1);

    final iou = intersectionArea / (area + otherArea - intersectionArea);
    return iou;
  }
}


class YOLODetection {
  BoundingBox<double> boundingBox;
  double confidence;
  String label;

  YOLODetection(
      {required this.boundingBox, required this.confidence, required this.label});
}

class YOLOV6Detection extends YOLODetection {
  int imageIndex;

  YOLOV6Detection({required this.imageIndex, required super.boundingBox, required super.confidence, required super.label});
}

class YOLO {

  static Iterable<YOLODetection> filterPerClass(
      Iterable<YOLODetection> detections, int maxResultsPerClass) sync* {
    Map<String, int> counters = {};

    final pq = PriorityQueue<YOLODetection>(
          (lhs, rhs) =>
          (rhs.confidence)
              .compareTo(lhs.confidence),
    )
      ..addAll(detections);
    while (pq.isNotEmpty) {
      YOLODetection detection = pq.removeFirst();
      String label = detection.label;

      if (counters[label] == null) {
        counters[label] = 1;
      } else {
        int count = counters[label]!;
        if (count >= maxResultsPerClass) {
          continue;
        } else {
          counters[label] = count + 1;
        }
      }
      yield detection;
    }
  }

  static double calculateIoU(BoundingBox box1, BoundingBox box2) {
    return box1.iou(box2);
  }

  static List<YOLODetection> nonMaxSuppression(List<YOLODetection> detections,
      {int limit = 10, double threshold = 0.5}) {
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));

    final selectedDetections = <YOLODetection>[];

    while (detections.isNotEmpty && selectedDetections.length < limit) {
      final currentDetection = detections.removeAt(0);
      selectedDetections.add(currentDetection);

      detections.removeWhere((detection) =>
      calculateIoU(currentDetection.boundingBox, detection.boundingBox) >
          threshold);
    }

    return selectedDetections;
  }

}

class YOLOv2 {
  static Iterable<YOLODetection> getValidDetections(List<double> output,
      {
        required int numClasses,
        required int numBoxesPerBlock,
        required double threshold,
        required int blockSize,
        required int gridSize,
        required List<double> anchors,
        required List<String> classNames,
        required int inputSize,
      }) sync* {

    var offset = 0;
    for (int y = 0; y < gridSize; ++y) {
      for (int x = 0; x < gridSize; ++x) {
        for (int b = 0; b < numBoxesPerBlock; ++b) {
          final (lx, ly, lw, lh, aw, ah) = (output[offset++], output[offset++], output[offset++], output[offset++], anchors[2 * b + 0], anchors[2 * b + 1]);
          final confidence = MathUtil.sigmoid(output[offset++]);

          final classes = output.sublist(offset, offset + numClasses);
          offset += numClasses;

          final confidences = MathUtil.softmax(classes);

          int detectedClass = -1;
          double maxClassConfidence = 0;
          for (int c = 0; c < numClasses; ++c) {
            if (confidences[c] > maxClassConfidence) {
              detectedClass = c;
              maxClassConfidence = confidences[c];
            }
          }

          final double confidenceInClass = maxClassConfidence * confidence;
          if (confidenceInClass > threshold) {
            final String label = classNames[detectedClass];
            final double cx = (x + MathUtil.sigmoid(lx));
            final double cy = (y + MathUtil.sigmoid(ly));
            final double hw = (math.exp(lw) * aw) / 2;
            final double hh = (math.exp(lh) * ah) / 2;
            final scale = blockSize / inputSize;
            final double x1 = math.max(0, (cx - hw) * scale);
            final double y1 = math.max(0, (cy - hh) * scale);
            final double x2 = math.min(1, (cx + hw) * scale);
            final double y2 = math.min(1, (cy + hh) * scale);
            final boundingBox = BoundingBox(x1, y1, x2, y2);
            yield YOLODetection(boundingBox: boundingBox, confidence: confidenceInClass, label: label);
          }
        }
      }
    }
  }

}

class YOLOv8 {

  static Iterable<YOLODetection> getValidDetections(List<double> output,
      {
        required double confidenceThreshold,
        required double classThreshold,
        required List<String> classNames,
        required int inputSize,
      }) sync* {
    final numClasses = classNames.length;
    if (output.length % numClasses != 0) {
      throw ArgumentError("${output.length} should be ${numClasses}xN shaped");
    }
    final numBoxes = output.length ~/ (5 + numClasses);
    final maxValues = List<double>.filled(numBoxes, 0);
    final maxIndices = List<int>.filled(numBoxes, 0);

    // shape is (5 + numClasses, numBoxes)
    int offset = 5 * numBoxes;
    final confidences = output.sublist(numBoxes * 4, numBoxes * 5);
    // final maxConfidence = confidences.max;
    for (int classIndex = 0; classIndex < numClasses; ++classIndex) {
      for (int boxIndex = 1; boxIndex < numBoxes; ++boxIndex) {
        if (confidences[boxIndex] <= confidenceThreshold) {
          continue;
        }
        final double value = output[offset++];
        if (value > maxValues[boxIndex]) {
          maxIndices[boxIndex] = classIndex;
          maxValues[boxIndex] = value;
        }
      }
    }
    for (var (boxIndex, classScore) in maxValues.indexed) {
      if (confidences[boxIndex] <= confidenceThreshold) {
        continue;
      }
      if (classScore <= classThreshold) {
        continue;
      }

      int offset = boxIndex;
      final cx = output[offset];
      offset += numBoxes;
      final cy = output[offset];
      offset += numBoxes;
      final hw = output[offset] / 2;
      offset += numBoxes;
      final hh = output[offset] / 2;
      final boundingBox = BoundingBox(
          (cx - hw) * inputSize, (cy - hh) * inputSize,
          (cx + hw) * inputSize, (cy + hh) * inputSize);
      // TODO : do we need both confidence and classScore?
      final confidence = confidences[boxIndex];
      final classIndex = maxIndices[boxIndex];

      final detection = YOLODetection(
          boundingBox: boundingBox,
          confidence: confidence * classScore,
          label: classNames[classIndex]);
      yield detection;
    }
  }

}

class YOLOv6Seg {
  static Iterable<YOLODetection> getValidDetections(List<double> output,
      {
        required double confidenceThreshold,
        required double classThreshold,
        required List<String> classNames,
        required int inputSize,
        int numImages = 1,
      }) sync* {
    // predictionShape is [numImages, 8400, 5 + numClasses]
    // confsShape is for segmentation and is [numImages, 8400, 33]
    final numBoxes = output.length ~/ (classNames.length + 5);
    final numClasses = classNames.length;

    // pred_candidates = torch.logical_and(prediction[..., 4] > conf_thres, torch.max(prediction[..., 5: 5 + num_classes], axis=-1)[0] > conf_thres)  # candidates
    int offset = 0;
    final stride = (5 + numClasses);
    for (var imageIndex = 0; imageIndex < numImages; ++imageIndex) {
      for (var boxIndex = 0; boxIndex < numBoxes; ++boxIndex) {
        final nextOffset = offset + stride;
        final cx = output[offset++],
            cy = output[offset++],
            w = output[offset++],
            h = output[offset++];
        final objectConfidence = output[offset++];

        if (!(objectConfidence > confidenceThreshold)) {
          offset = nextOffset;
          continue;
        }

        var maxClassConfidence = double.negativeInfinity;
        var maxClassIndex = -1;
        for (int classIndex = 0; classIndex < numClasses; ++classIndex) {
          final classConfidence = output[offset++];
          if (classConfidence > maxClassConfidence) {
            maxClassConfidence = classConfidence;
            maxClassIndex = classIndex;
          }
        }

        offset = nextOffset;
        if (!(maxClassConfidence > confidenceThreshold)) {
          continue;
        }

        // no multilabel (otherwise we need to calculate confidence per class and keep all those with confidence > confidenceThreshold
        // x[:, 5: 5 + num_classes] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        final confidence = objectConfidence * maxClassConfidence;
        // box = xywh2xyxy(x[:, :4])
        final hw = w / 2,
            hh = h / 2;
        final boundingBox = BoundingBox((cx - hw)/inputSize, (cy - hh)/inputSize, (cx + hw)/inputSize, (cy + hh)/inputSize);
        final label = classNames[maxClassIndex];

        // segconf = x[:, 5 + num_classes: ]
        // TODO : support segmentation too
        yield YOLOV6Detection(boundingBox: boundingBox,
            confidence: confidence,
            label: label,
            imageIndex: imageIndex);
      }
    }
  }
}
