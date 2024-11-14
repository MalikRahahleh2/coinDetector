import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:collection/collection.dart';
import 'package:camera/camera.dart';
import 'package:demo_camera/image_util.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:image/image.dart' as image_lib;
import 'package:tflite_flutter/tflite_flutter.dart';

import 'asset_util.dart';
import 'camera_processor.dart';
import 'model_isolate.dart';
import 'time_util.dart';
import 'tflite_util.dart';
import 'yolo.dart';

class YOLORawOutputs {
  List<double> detections;
  List<double>? segmentations;
  YOLORawOutputs({ required this.detections, required this.segmentations });
}

abstract class YOLOObjectDetector<ConfigT>  extends ModelBase<YOLOResult> {

  late Interpreter _interpreter;
  late List<String> _classNames;
  late int _inputSize;
  bool _isReady = false;

  int get inputSize => _inputSize;
  String get modelPath;
  String get classNamesPath;
  ConfigT get config;

  YOLOObjectDetector();
  YOLOObjectDetector.fromConfig(ConfigT config);

  void initializeFrom({
    required int interpreterAddress,
    required List<String> classNames,
  }) {
    if (_isReady) {
      throw StateError("Already initialized; call close() first");
    }
    _interpreter = Interpreter.fromAddress(interpreterAddress);
    final inputTensors = _interpreter.getInputTensors();
    _inputSize = inputTensors[0].shape[1];
    _classNames = classNames;
    _isReady = true;
  }

  // Loads interpreter from asset
  Future<void> initialize() async {
    if (_isReady) {
      throw StateError("Already initialized; call close() first");
    }
    _interpreter = await TfliteUtil.loadInterpreter(modelPath);
    final inputTensors = _interpreter.getInputTensors();
    _inputSize = inputTensors[0].shape[1];
    final contents = await rootBundle.loadString(classNamesPath);
    _classNames = LineSplitter.split(contents).toList();
    _isReady = true;
  }


  void close() {
    ensureReady();
    _interpreter.close();
    _classNames.clear();
    _isReady = false;
  }

  List<YOLODetection> postprocess(List<double> outputs);

  static BoundingBox<double> transformBoundingBox(BoundingBox<double> bbox, Transform transform) {
    final c = transform.transform([bbox.x1, bbox.y1, bbox.x2, bbox.y2]);
    return BoundingBox<double>(c[0], c[1], c[2], c[3]);
  }

  bool _outputIndicesSet = false;
  int _detectionsOutputIndex = -1;
  int _segmentationsOutputIndex = -1;

  YOLORawOutputs mapOutputs(List<ShapedDataList<num>> outputs) {
    if (_outputIndicesSet) {
      return YOLORawOutputs(
        detections: outputs[_detectionsOutputIndex].list as TypedDataList<double>,
        segmentations: _segmentationsOutputIndex == -1 ? null : outputs[_segmentationsOutputIndex].list as TypedDataList<double>,
      );
    }
    final numClasses = _classNames.length;
    final possibleNumBoxes = [2125, 8400, 8500];
    const batchSize = 1;
    for (final (index, output) in outputs.indexed) {
      if (possibleNumBoxes.any((numBoxes) => output.shape.equals([batchSize, numBoxes, numClasses + 5]))) {
        _detectionsOutputIndex = index;
        continue;
      }
      if (possibleNumBoxes.any((numBoxes) => output.shape.equals([batchSize, numBoxes, 33]))) {
        // TODO : figure out how this output should be called
        continue;
      }
      if ((output.shape.length == 4) && (output.shape[0] == batchSize) && (output.shape[3] == 32) && (output.shape[1] == output.shape[2])) {
        _segmentationsOutputIndex = index;
        continue;
      }
    }
    if (_detectionsOutputIndex == -1) {
      throw UnsupportedError("None of the YOLO outputs are shaped like the detections matrix (batchSize, numBoxes, numClasses + 5)");
    }
    _outputIndicesSet = true;
    return mapOutputs(outputs);
  }


  /// Gets the interpreter instance
  @override
  YOLOResult run(image_lib.Image image, DebugTimerStack? dt) {
    if (!_isReady) {
      return YOLOResult(detections: []);
    }
    dt?.next("Resize");
    final transformedImage = ImageUtil.letterbox(image: image, size: inputSize);

    dt?.next("Preprocess");
    final input = TfliteUtil.toTensor(
        image: transformedImage.image, order: ShapeOrder.hwc);

    final outputs = TfliteUtil.runImagesThoughModel(
        _interpreter, [input], dt?..push("Inference"));
    dt?.pop();

    final mappedOutputs = mapOutputs(outputs);

    dt?.next("Postprocess");
    final detections = postprocess(mappedOutputs.detections);
    dt?.mark();

    final rescaledDetections = detections.map((d) => YOLODetection(boundingBox: transformBoundingBox(d.boundingBox, transformedImage.transform), confidence: d.confidence, label: d.label)).toList();

    return YOLOResult(detections: rescaledDetections);
  }

  void ensureReady() {
    if (_isReady) {
      return;
    }
    throw StateError("Not initialized; call initialize() first");
  }

  List<String> get classNames => _classNames.toList();
  int get address {
    ensureReady();
    return _interpreter.address;
  }

  bool get isReady => _isReady;

}


class YOLOFrameProcessor<ConfigT> extends ModelFrameProcessor<YOLOResult> {
  final _detectorIsolate = ModelIsolate<YOLOInitData<ConfigT>, YOLOResult>();
  final YOLOObjectDetector _detector;
  final YOLOFactory<ConfigT> _factory;
  bool _initialized = false;

  YOLOFrameProcessor({required YOLOObjectDetector detector, required YOLOFactory<ConfigT> factory}) : _detector = detector, _factory = factory;

  @override
  Future<void> initialize() async {
    if (_initialized) {
      throw Exception("Already initialized; call release() first");
    }
    await _detector.initialize();
    final initData = YOLOInitData<ConfigT>(
      interpreterAddress: _detector.address,
      classNames: _detector.classNames,
      config: _detector.config,
    );
    await _detectorIsolate.start(_factory, initData);
    _initialized = true;
  }

  @override
  bool isReady() {
    return _initialized && _detector.isReady;
  }

  @override
  Future<TimedResult<YOLOResult>?> run(CameraImage cameraImage, int orientation) {
    return _detectorIsolate.run(cameraImage, orientation);
  }

  @override
  void release() {
    if (!_initialized) {
      throw Exception("Not initialized. Call start() first");
    }
    _detectorIsolate.stop();
    _detector.close();
    _initialized = false;
  }

}


class YOLOResult {
  List<YOLODetection> detections;
  YOLOResult({required this.detections});
}

// typedef YOLOFactory<ConfigT> = YOLOObjectDetector<ConfigT> Function(ConfigT config);

class YOLOInitData<ConfigT> {
  int interpreterAddress;
  List<String> classNames;
  ConfigT config;

  YOLOInitData(
      {required this.interpreterAddress, required this.classNames, required this.config});
}

abstract class YOLOFactory<ConfigT> extends ModelFactory<YOLOInitData<ConfigT>, YOLOResult> {

}

class YOLOv2Factory extends YOLOFactory<YOLOv2Config> {
  @override
  getInstance(YOLOInitData init) {
    // TODO: implement getInstance
    var instance = YOLOv2ObjectDetector.fromConfig(init.config);
    instance.initializeFrom(interpreterAddress: init.interpreterAddress, classNames: init.classNames);
    return instance;
  }
}

class YOLOv2Config {
  static const defaultAnchors = [0.57273,0.677385,1.87446,2.06253,3.33843,5.47434,7.88282,3.52778,9.77052,9.16828];
  final int blockSize;
  final double threshold;
  final int numBoxesPerBlock;
  final int numResultsPerClass;
  final List<double> anchors;
  final int numThreads;
  YOLOv2Config({
    this.threshold = 0.4,
    this.numBoxesPerBlock = 5,
    this.anchors = defaultAnchors,
    this.blockSize = 32,
    this.numResultsPerClass = 5,
    this.numThreads = 4,
  });
}

class YOLOv2ObjectDetector extends YOLOObjectDetector<YOLOv2Config> {
  static const _modelPath = "assets/models/yolov2_tiny.tflite";
  static const _classNamesPath = "assets/models/yolo_classes.txt";

  YOLOv2Config _config;

  @override
  get modelPath => _modelPath;
  @override
  get classNamesPath => _classNamesPath;
  @override
  get config => _config;

  static final _isAvailable = AssetUtil.assetExists(_modelPath);
  static get isAvailableFuture => _isAvailable;

  // see also https://github.com/onnx/models/blob/main/vision/object_detection_segmentation/tiny-yolov2/README.md
  late int gridSize;

  @override
  YOLOv2ObjectDetector() : this.fromConfig(YOLOv2Config());
  @override
  YOLOv2ObjectDetector.fromConfig(YOLOv2Config config) :
      _config = config,
      super() {
    gridSize = (_inputSize ~/ config.blockSize);
  }


  List<YOLODetection> postprocess(List<double> output) {
      final numResultsPerClass = _config.numResultsPerClass;

      final validDetections = YOLOv2.getValidDetections(
          output,
          numClasses: _classNames.length,
          numBoxesPerBlock: _config.numBoxesPerBlock,
          threshold: _config.threshold,
          blockSize: _config.blockSize,
          gridSize: gridSize,
          anchors: _config.anchors,
          classNames: _classNames,
          inputSize: _inputSize,
      );
      final classCapDetections = YOLO.filterPerClass(validDetections, numResultsPerClass).toList();
      final nmsDetections = YOLO.nonMaxSuppression(classCapDetections);
      return nmsDetections;
  }

}

class YOLOv8Config {
  final double classThreshold;
  final double confidenceThreshold;
  final int maxResultsPerClass;
  YOLOv8Config({
    this.classThreshold = 0.4,
    this.confidenceThreshold = 0.45,
    this.maxResultsPerClass = 10,
  });
}

class YOLOv8Factory extends YOLOFactory<YOLOv8Config> {
  @override
  getInstance(YOLOInitData init) {
    var instance = YOLOv8ObjectDetector.fromConfig(init.config);
    instance.initializeFrom(interpreterAddress: init.interpreterAddress, classNames: init.classNames);
    return instance;
  }
}

class YOLOv8ObjectDetector  extends YOLOObjectDetector<YOLOv8Config> {
  static const _modelPath = "assets/models/yolov8n_numa.tflite";
  static const _classNamesPath = "assets/models/yolo_numa_classes.txt";
  final YOLOv8Config _config;

  @override
  String get modelPath => _modelPath;
  @override
  String get classNamesPath => _classNamesPath;
  @override
  YOLOv8Config get config => _config;

  static final _isAvailable = AssetUtil.assetExists(_modelPath);
  static get isAvailableFuture => _isAvailable;

  @override
  YOLOv8ObjectDetector() : this.fromConfig(YOLOv8Config());
  @override
  YOLOv8ObjectDetector.fromConfig(YOLOv8Config config) : _config = config;

  List<YOLODetection> postprocess(List<double> output) {
    final maxResultsPerClass = _config.maxResultsPerClass;

    final validDetections = YOLOv8.getValidDetections(
        output,
        inputSize: _inputSize,
        classNames: _classNames,
        classThreshold: _config.classThreshold,
        confidenceThreshold: _config.confidenceThreshold,
    ).toList();
    final filteredDetections = YOLO.filterPerClass(validDetections, maxResultsPerClass).toList();
    return YOLO.nonMaxSuppression(filteredDetections);
  }


}


class YOLOv6SegConfig {
  final double classThreshold;
  final double confidenceThreshold;
  final int maxResultsPerClass;
  YOLOv6SegConfig({
    this.classThreshold = 0.45,
    this.confidenceThreshold = 0.45,
    this.maxResultsPerClass = 10,
  });
}


class YOLOv6SegFactory extends YOLOFactory<YOLOv6SegConfig> {
  @override
  getInstance(YOLOInitData init) {
    var instance = YOLOv6SegObjectDetector.fromConfig(init.config);
    instance.initializeFrom(interpreterAddress: init.interpreterAddress,
        classNames: init.classNames);
    return instance;
  }
}

class YOLOv6SegObjectDetector  extends YOLOObjectDetector<YOLOv6SegConfig> {
  // static const _modelPath = "assets/models/yolov6lite_s_int8_tf-2.17.1.tflite";
  // static const _modelPath = "assets/models/yolov6s_float16-tf-2.17.1.tflite";
  static const _modelPath = "assets/models/best_ckpt_lite-oct-29-exp2_float16.tflite";

  // static const _classNamesPath = "assets/models/yolo_classes.txt";
  static const _classNamesPath = "assets/models/yolo_numa_classes.txt";

  final YOLOv6SegConfig _config;

  @override
  String get modelPath => _modelPath;
  @override
  String get classNamesPath => _classNamesPath;
  @override
  YOLOv6SegConfig get config => _config;

  static final _isAvailable = AssetUtil.assetExists(_modelPath);
  static get isAvailableFuture => _isAvailable;

  @override
  YOLOv6SegObjectDetector() : this.fromConfig(YOLOv6SegConfig());
  @override
  YOLOv6SegObjectDetector.fromConfig(YOLOv6SegConfig config) : _config = config;

  // bool _outputIndicesSet = false;
  // int _detectionsOutputIndex = -1;
  //
  // void setOutputIndices(List<List<double>> outputs) {
  //   if (_outputIndicesSet) {
  //     return;
  //   }
  //   final numClasses = _classNames.length;
  //   final detectionsOutputSize = (numClasses) * 5
  //   for (final output in outputs) {
  //     final n = output.length;
  //     if ()
  //   }
  // }

  // List<double> getDetectionsOutput(outputs) {
  //   setOutputIndices();
  //   return outputs[_detectionsOutputIndex];
  // }

  List<YOLODetection> postprocess(List<double> output) {
    final maxResultsPerClass = _config.maxResultsPerClass;

    final validDetections = YOLOv6Seg.getValidDetections(
        output,
        confidenceThreshold: _config.confidenceThreshold,
        classThreshold: _config.classThreshold,
        classNames: _classNames,
        inputSize: _inputSize,
        numImages: 1,
    ).toList();
    final filteredDetections = YOLO.filterPerClass(validDetections, maxResultsPerClass).toList();
    return YOLO.nonMaxSuppression(filteredDetections);
  }


}

