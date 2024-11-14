import 'dart:async';
import 'dart:ui' as ui;

import 'package:flutter/material.dart';

import 'classify.dart';
import 'time_util.dart';

class MobileClipOverlay extends StatefulWidget {
  final Stream<TimedResult<MobileClipResult>> _stream;
  const MobileClipOverlay(this._stream, {super.key});

  @override
  State<StatefulWidget> createState()  => MobileClipOverlayState();

}

class MobileClipOverlayState extends State<MobileClipOverlay> {
  MobileClipResult? lastResult;
  DateTime lastUpdate = DateTime.now();
  double fps = 0.0;

  StreamSubscription<TimedResult<MobileClipResult>>? _subscription;

  @override
  void initState() {
    super.initState();
    _subscription= widget._stream.listen((timedResult) {
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
      painter: MobileClipOverlayPainter(lastResult, fps),
      child: Container(),
    );
  }
}

class MobileClipOverlayPainter extends CustomPainter {
  final MobileClipResult? result;
  final double fps;

  const MobileClipOverlayPainter(this.result, this.fps);

  @override
  void paint(Canvas canvas, Size size) {
    if (result == null) {
      return;
    }

    final paintFill = Paint()
      ..color = Colors.deepPurple
      ..strokeWidth = 2
      ..style = PaintingStyle.fill;
    final w = size.width;
    final h = size.height;
    const textStyle = TextStyle(
      color: Colors.white,
      fontSize: 16,
    );

    var labelPainter = TextPainter(
      text: TextSpan(
        text: "FPS: ${fps.toStringAsFixed(2)} | ${result!.label} (${(result!.confidence * 100.0).toStringAsFixed(1)}%)",
        style: textStyle,
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    );
    labelPainter.layout(
      minWidth: 0,
      maxWidth: w,
    );
    final offset = Offset(w - labelPainter.width, h - labelPainter.height);
    final rectLabel = ui.Rect.fromLTRB(offset.dx, offset.dy, offset.dx + labelPainter.width, offset.dy + labelPainter.height);
    canvas.drawRect(rectLabel, paintFill);
    labelPainter.paint(canvas, offset);

  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) {
    return true;
  }
}
