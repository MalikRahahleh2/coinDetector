
import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:camera/camera.dart';
import 'package:coin_detector/math_util.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'asset_util.dart';
import 'camera_processor.dart';
import 'model_isolate.dart';
import 'tflite_util.dart';
import 'time_util.dart';

class MobileClipResult {
  String label;
  double confidence;

  MobileClipResult({required this.label, required this.confidence});
}

class MobileClipInitData {
  int address;
  Map<String, List<double>> labelEmbeddings;

  MobileClipInitData({required this.address, required this.labelEmbeddings});
}

class MobileClipFactory extends ModelFactory<MobileClipInitData, MobileClipResult> {
  @override
  ModelBase<MobileClipResult> getInstance(MobileClipInitData init) {
    // ONNX Runtime
    // final session = OrtSession.fromAddress(init.address);
    // final instance = MobileClipClassifier(session: session);

    // TFLite
    final interpreter = Interpreter.fromAddress(init.address);
    final instance = MobileClipClassifier(interpreter: interpreter, labelEmbeddings: init.labelEmbeddings);

    return instance;
  }
}

class MobileClipClassifier extends ModelBase<MobileClipResult> {
  // ONNX Runtime
  // OrtSession session;
  // MobileClipClassifier({required this.session});

  // TFLite
  final Interpreter interpreter;
  final Map<String, List<double>> labelEmbeddings;
  MobileClipClassifier({required this.interpreter, required this.labelEmbeddings});

  // for mobileclip_s0
  static const int inputSize = 256;


  MobileClipResult getResult(List<double> imageEmbedding) {
    final labels = labelEmbeddings.keys.toList();
    final similarities = List<double>.filled(labels.length, 0.0);
    for (final (index, label) in labels.indexed) {
      similarities[index] = MathUtil.dotProduct(labelEmbeddings[label]!, imageEmbedding);
    }
    final probabilities = MathUtil.softmax(similarities.map((val) => val * 100.0).toList());
    final prediction = MathUtil.argMax(similarities);
    return MobileClipResult(label: labels[prediction], confidence: probabilities[prediction]);
  }

  @override
  MobileClipResult run(image_lib.Image image, DebugTimerStack? dt) {
    // final shape = [image.height, image.width];
    // final imageTensor = OrtValueTensor.createTensorWithDataList(image.data, shape);
    // final imageInputName = session.inputNames[0];
    // final inputs = {imageInputName: imageTensor};
    // final runOptions = OrtRunOptions();
    // // NOTE : we run synchronously because we are already in an isolate
    // // TODO : resize, center-crop and normalize (0 .. 255 -> 0.0 .. 1.0) the image
    // final outputs = session.run(runOptions, inputs);
    // imageTensor.release();
    // runOptions.release();
    //
    // // TODO : add text query vectors here and compute dot product
    // for (var element in outputs) {
    //   element?.release();
    // }
    dt?.next("Resize");
    final imageResized = image_lib.copyResizeCropSquare(image, size: inputSize);

    dt?.next("ToTensor");
    final input = TfliteUtil.toTensor(image: image, order: ShapeOrder.chw);

    final output = TfliteUtil.runImagesThoughModel(
        interpreter, [input], dt?..push("Inference"))[0];
    dt?.pop();

    dt?.next("Postprocess");
    var result = getResult(output.list as TypedDataList<double>);
    dt?.end();

    return result;

  }
}

class MobileClipFrameClassifier extends ModelFrameProcessor<MobileClipResult> {
  bool _initialized = false;

  // late OrtSession _session;
  late Interpreter _interpreter;
  final _isolate = ModelIsolate<MobileClipInitData, MobileClipResult>();
  // Older versions of tflite_flutter prepends this with 'assets/'
  // static const modelPath = 'assets/models/vision_model.tflite';
  // static const modelPath = 'assets/models/vision_model_int8_dynamic.tflite';
  static const modelPath = 'assets/models/vision_model_int8_weights_only.tflite';
  static const labelEmbeddingsPath = 'assets/models/vision_model_label_embeddings.json';
  static final _isAvailable = AssetUtil.assetExists(modelPath);
  static get isAvailableFuture => _isAvailable;

  @override
  Future<void> initialize() async {
    if (_initialized) {
      throw Exception("Already initialized; call release() first");
    }

    // ONNX Runtime
    // OrtEnv.instance.init();
    // final sessionOptions = OrtSessionOptions();
    // const assetFileName = 'assets/models/vision_model_uint8.onnx';
    // final rawAssetFile = await rootBundle.load(assetFileName);
    // final bytes = rawAssetFile.buffer.asUint8List();
    // _session = OrtSession.fromBuffer(bytes, sessionOptions);

    // await _isolate.start(MobileClipFactory(), MobileClipInitData(address: _session.address));

    // TFLite
    _interpreter = await TfliteUtil.loadInterpreter(modelPath);
    String labelEmbeddingsJson = await rootBundle.loadString(labelEmbeddingsPath);
    Map<String, dynamic> labelEmbeddingsUntyped = json.decode(labelEmbeddingsJson);
    Map<String, List<double>> labelEmbeddings = labelEmbeddingsUntyped.map((key, value) => MapEntry(
          key,
          List<double>.from(value)
      ),
    );

    await _isolate.start(
      MobileClipFactory(),
      MobileClipInitData(address: _interpreter.address, labelEmbeddings: labelEmbeddings)
    );

    _initialized = true;
  }

  @override
  Future<TimedResult<MobileClipResult>?> run(CameraImage cameraImage, int orientation) {
    return _isolate.run(cameraImage, orientation);
  }

  @override
  bool isReady() {
    return _initialized;
  }

  @override
  void release() {
    if (!_initialized) {
      throw Exception("Not initialized; call initialize() first");
    }
    // _session.release();
    // OrtEnv.instance.release();
    _isolate.stop();
    _interpreter.close();

    _initialized = false;
  }
}
