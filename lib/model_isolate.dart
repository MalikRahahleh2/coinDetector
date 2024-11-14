
import 'dart:isolate';
import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as image_lib;
import 'package:camera/camera.dart';

import 'time_util.dart';
import 'image_util.dart';

/// Bundles data to pass between Isolate
class IsolateInferenceData<ModelInit, ModelResult> {
  ModelInit init;
  CameraImage cameraImage;
  SendPort responsePort;
  ModelFactory<ModelInit, ModelResult> factory;
  int rotation;

  IsolateInferenceData({
    required this.init,
    required this.cameraImage,
    required this.responsePort,
    required this.rotation,
    required this.factory,
  });
}


abstract class ModelBase<ModelResult> {
  ModelResult run(image_lib.Image image, DebugTimerStack? dt);
}

abstract class ModelFactory<ModelInit, ModelResult> {
  ModelBase<ModelResult> getInstance(ModelInit init);
}


/// Manages separate Isolate instance for inference
class ModelIsolate<ModelInit, ModelResult> {
  static const String debugName = "IsolateInference";

  ModelInit? _init;
  Isolate? _isolate;
  final ReceivePort _receivePort = ReceivePort();
  SendPort? _sendPort;
  ModelFactory<ModelInit, ModelResult>? _factory;
  bool _started = false;

  SendPort? get sendPort => _sendPort;
  Isolate? get isolate => _isolate;

  Future<void> start(ModelFactory<ModelInit, ModelResult> factory, ModelInit init) async {
    if (_started) {
      throw Exception("Already started");
    }
    _init = init;
    _factory = factory;
    _isolate = await Isolate.spawn(
      entryPoint<ModelResult>,
      _receivePort.sendPort,
      debugName: debugName,
    );

    // the first message we get from the isolate is its send port
    _sendPort = await _receivePort.first;
    _started = true;
  }

  void stop() {
    if (!_started) {
      throw Exception("Not started");
    }
    _isolate?.kill();
    _receivePort.close();
    _started = false;
  }

  Future<TimedResult<ModelResult>?> run(CameraImage cameraImage, int rotation) async {
    if (!_started) {
      return null;
    }
    final responsePort = ReceivePort();
    sendPort!.send(IsolateInferenceData<ModelInit, ModelResult>(
        cameraImage: cameraImage,
        rotation: rotation,
        init: _init!,
        responsePort: responsePort.sendPort,
        factory: _factory!,
    ));
    return (await responsePort.first) as TimedResult<ModelResult>;
    // return null;
  }

  static void entryPoint<ModelResult>(SendPort sendPort) async {
    final port = ReceivePort();
    sendPort.send(port.sendPort);

    await for (final IsolateInferenceData isolateData in port) {

      // final model = isolateData.factory(isolateData.init) as ModelBase<ModelResult>;
      final model = isolateData.factory.getInstance(isolateData.init) as ModelBase<ModelResult>;
      // await detector.initialize(interpreterAddress: isolateData.interpreterAddress, classNames: isolateData.classNames);
      final dt = DebugTimerStack();
      try {
        dt.next("Convert image");
        final int width = isolateData.cameraImage.width;
        final int height = isolateData.cameraImage.height;
        int shorter = math.min(width, height);
        // NOTE : this should work well across platforms and resolution presets.
        int scale = (shorter / 270).round();
        var image = ImageUtil.convertCameraImage(isolateData.cameraImage, scale);

        if (isolateData.rotation != 0) {
          dt.next("Rotate image");
          image = image_lib.copyRotate(image, angle: isolateData.rotation);
        }

        // debugPrintThrottled(dt.timingsMs.map((x) => "${x.path.join('/')}: ${x.ms}ms").join("\n"));
        final result = model.run(image, dt..push("Model"));
        dt.pop();
        isolateData.responsePort.send(TimedResult(result: result, timingsMs: dt.timingsMs));
        // debugPrintThrottled("--");
        // debugPrintThrottled(dt.timingsMs.map((x) => "${x.path.join('/')}: ${x.ms}ms").join("\n"));
      }
      finally {
        dt.end();
      }
      
    }
  }
}
