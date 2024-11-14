import 'dart:async';
import 'dart:ui' as ui;
import 'package:demo_camera/object_detect_logic.dart';
import 'package:demo_camera/yolo.dart';
import 'package:flutter/material.dart';

import 'object_detect.dart';
import 'time_util.dart';


class YOLODetectionOverlay extends StatefulWidget {
  final Stream<TimedResult<YOLOResult>> _detectionStream;
  const YOLODetectionOverlay(this._detectionStream, {super.key});

  @override
  State<StatefulWidget> createState()  => YOLODetectionOverlayState();
}

class YOLODetectionOverlayState extends State<YOLODetectionOverlay> {
  YOLOResult? lastResult;
  DateTime lastUpdate = DateTime.now();
  double fps = 0.0;
  StreamSubscription<TimedResult<YOLOResult>>? _subscription;

  @override
  void initState() {
    super.initState();
    _subscription = widget._detectionStream.listen((timedResult) {
      setState(() {
        var now = DateTime.now();
        var usSinceLastUpdate = now.difference( lastUpdate ).inMicroseconds;
        fps = 1.0e6 / (usSinceLastUpdate.toDouble());
        lastResult = timedResult.result;
        lastUpdate = DateTime.now();
      });
    });
  }

  @override
  void dispose() {
    super.dispose();
    if (_subscription != null) {
      _subscription!.cancel();
      _subscription = null;
    }
  }

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: YOLODetectionPainter(lastResult, fps),
      child: Container(),
    );
  }

}


class YOLODetectionPainter extends CustomPainter {
  final YOLOResult? result;
  final double fps;

  const YOLODetectionPainter(this.result, this.fps);

  static ui.Rect boundingBoxToRect(BoundingBox<double> box) {
    return ui.Rect.fromLTRB(box.x1, box.y1, box.x2, box.y2);
  }

  static void drawDashedLine({
    required Canvas canvas,
    required Offset p1,
    required Offset p2,
    required List<double> pattern,
    required Paint paint,
  }) {
    // thanks to https://stackoverflow.com/questions/74911485
    assert(pattern.length.isEven);
    final distance = (p2 - p1).distance;
    final normalizedPattern = pattern.map((width) => width / distance).toList();
    final points = <Offset>[];
    double t = 0;
    int i = 0;
    while (t < 1) {
      points.add(Offset.lerp(p1, p2, t)!);
      t += normalizedPattern[i++];  // dashWidth
      points.add(Offset.lerp(p1, p2, t.clamp(0, 1))!);
      t += normalizedPattern[i++];  // dashSpace
      i %= normalizedPattern.length;
    }
    canvas.drawPoints(ui.PointMode.lines, points, paint);
  }

  static void drawDashedPolyline({
    required Canvas canvas,
    required List<Offset> points,
    required bool isClosed,
    required List<double> pattern,
    required Paint paint,
  }) {
    final n = points.length - (isClosed? 0 : 1);
    final finalI = n - 1;
    for (int i = 0; i < n; ++i) {
      final p1 = points[i];
      final p2 = (i == finalI) ? points[0] : points[i + 1];
      drawDashedLine(canvas: canvas, p1: p1, p2: p2, pattern: pattern, paint: paint);
    }
  }

  static void drawDashedRect({
    required Canvas canvas,
    required ui.Rect rect,
    required List<double> pattern,
    required Paint paint,
  }) {
    drawDashedPolyline(canvas: canvas, points: [rect.topLeft, rect.topRight, rect.bottomRight, rect.bottomLeft], isClosed: true, pattern: pattern, paint: paint);
  }

  static const config = NumaObjectDetectConfig();

  @override
  void paint(Canvas canvas, Size size) {
    final result = this.result;
    if (result == null) {
      return;
    }
    final paint = Paint()
      ..color = Colors.blue
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    final goodOutlinePaint = Paint()
      ..color = Colors.green
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    final badOutlinePaint = Paint()
      ..color = Colors.red
      ..strokeWidth = 2
      ..style = PaintingStyle.stroke;
    final paintFill = Paint()
      ..color = Colors.blue
      ..strokeWidth = 2
      ..style = PaintingStyle.fill;
    final w = size.width;
    final h = size.height;
    const textStyle = TextStyle(
      color: Colors.black,
      fontSize: 16,
    );

    var fpsLabelPainter = TextPainter(
      text: TextSpan(
        text: "FPS: ${fps.toStringAsFixed(2)}",
        style: textStyle,
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    );
    fpsLabelPainter.layout(
      minWidth: 0,
      maxWidth: 100,
    );

    final outlineBox = BoundingBox<double>(0.1, 0.05, 0.9, 0.95);

    final rectLabel = ui.Rect.fromLTRB(0, 0, fpsLabelPainter.width, fpsLabelPainter.height);
    canvas.drawRect(rectLabel, paintFill);
    fpsLabelPainter.paint(canvas, const Offset(0, 0));

    for (final detection in result.detections) {
      final b = detection.boundingBox;

      final rect = ui.Rect.fromLTRB(b.x1*w , b.y1*h , b.x2*w, b.y2*h);
      canvas.drawRect(rect, paint);
      final textSpan = TextSpan(
        text: "${detection.label} ${(detection.confidence*100).toStringAsFixed(1)}%",
        style: textStyle,
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
        maxLines: 1,
      );
      textPainter.layout(
        minWidth: 0,
        maxWidth: (b.x2 - b.x1)*w,

      );

      final rectLabel = ui.Rect.fromLTRB(b.x1*w , b.y2*h , b.x1*w + textPainter.width, b.y2*h + textPainter.height);
      canvas.drawRect(rectLabel, paintFill);
      textPainter.paint(canvas, Offset(b.x1*w , b.y2*h));
    }


    var message = "Unknown state";
    var goodDetection = false;
    var stopLookingForOtherObjectTypes = false;
    for (final objectType in [NumaObjectType.coin, NumaObjectType.banknote]) {
      final state = NumaObjectDetectLogic.getState(
          result.detections, outlineBox, objectType, config);
      stopLookingForOtherObjectTypes = true;
      switch (state) {
        case NumaObjectDetecState.objectPlacedWell:
          message = "Ready to scan";
          goodDetection = true;
        case NumaObjectDetecState.noObject:
          message = "Place the item flat against a dark background";
          stopLookingForOtherObjectTypes = false;
        case NumaObjectDetecState.objectOutsideROI:
          message = "Move the camera in front of the item";
        case NumaObjectDetecState.objectTooLarge:
          message = "Move the camera away from the item";
        case NumaObjectDetecState.objectTooSmall:
          message = "Move the camera closer to the item";
        case NumaObjectDetecState.objectTooNarrow:
          message = "Move the camera in front of the item";
      }
      if (stopLookingForOtherObjectTypes) {
        break;
      }
    }
    final outlineRect = boundingBoxToRect(outlineBox.scale(w, h));
    drawDashedRect(canvas: canvas, rect: outlineRect, pattern: [5, 5], paint: goodDetection ? goodOutlinePaint : badOutlinePaint);

    var messagePainter = TextPainter(
      text: TextSpan(
        text: message,
        style: textStyle,
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    );
    messagePainter.layout();
    final messageOffset = ui.Offset(0, h - messagePainter.height);
    final rectMessage = ui.Rect.fromPoints(messageOffset, messagePainter.size.bottomRight(messageOffset));
    canvas.drawRect(rectMessage, paintFill);
    messagePainter.paint(canvas, messageOffset);

  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
