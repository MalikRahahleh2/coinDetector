import 'package:demo_camera/yolo.dart';
import 'package:demo_camera/math_util.dart';

extension BoundingBoxCalculations on BoundingBox<double> {

  
  bool widthInside(BoundingBox other) {
    return (x1 >= other.x1) && (x2 <= other.x2);
  }
  bool heightInside(BoundingBox other) {
    return (y1 >= other.y1) && (y2 <= other.y2);
  }
  bool isInside(BoundingBox other) {
    return widthInside(other) && heightInside(other);
  }

  BoundingBox<double> scale(double sx, double sy) {
    return BoundingBox<double>(x1 * sx, y1 * sy, x2 * sx, y2 * sy);
  }
  double get width => (x2 - x1);
  double get height => (y2 - y1);
  double get aspectRatio => width / height;
  double get area => width * height;
}

enum NumaObjectDetecState {
  noObject,
  objectOutsideROI, // MoveCameraToObject
  objectTooLarge, // MoveCameraAway
  objectTooSmall, // MoveCameraCloser
  objectTooNarrow, // TiltCameraInFrontOfObject
  objectPlacedWell, // ReadyToScan
}

enum NumaObjectType {
  coin,
  banknote
}

class NumaObjectDetectConfig {
  final double minIoU;
  final double coinAllowedDeviationFromSquare;
  const NumaObjectDetectConfig({
    this.minIoU = 0.2,
    this.coinAllowedDeviationFromSquare = 0.5,
  });
}

class NumaObjectDetectLogic {
  static NumaObjectDetecState getState(List<YOLODetection> detections, BoundingBox<double> roi, NumaObjectType objectType, NumaObjectDetectConfig config) {
    detections = detections.where((d) => d.label == objectType.name).toList();
    if (detections.isEmpty) {
      return NumaObjectDetecState.noObject;
    }
    final ious = detections.map((d) => d.boundingBox.iou(roi)).toList();
    final index = MathUtil.argMax(ious);
    final objectBox = detections[index].boundingBox;
    final iou = ious[index];
    if (objectBox.isInside(roi)) {
      if (iou < config.minIoU) {
        return NumaObjectDetecState.objectTooSmall;
      }
      if ((objectType == NumaObjectType.coin) && ((1 - objectBox.aspectRatio).abs() > config.coinAllowedDeviationFromSquare)) {
        return NumaObjectDetecState.objectTooNarrow;
      }
      return NumaObjectDetecState.objectPlacedWell;
    }
    else if (roi.widthInside(objectBox) || roi.heightInside(objectBox)) {
      return NumaObjectDetecState.objectTooLarge;
    }
    else {
      return NumaObjectDetecState.objectOutsideROI;
    }
  }
}