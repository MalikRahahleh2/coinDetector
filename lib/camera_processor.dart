
import 'dart:async';

import 'package:camera/camera.dart';

import 'time_util.dart';

// Creation order:
// App -> CameraProcessor -> FrameProcessor
// FrameProcessor -> Model (but never runs it) , then InitData
// FrameProcessor -> ModelIsolate (with InitData) -> Factory (with InitData) -> Model
abstract class ModelFrameProcessor<ResultT> {
  Future<void> initialize();
  Future<TimedResult<ResultT>?> run(CameraImage cameraImage, int orientation);
  bool isReady();
  void release();
}

class ModelCameraProcessor<ResultT>  {
  bool _currentlyProcessing = false;
  final StreamController<TimedResult<ResultT>> _resultStream;
  final ModelFrameProcessor<ResultT> _frameProcessor;
  bool _initialized = false;
  bool _initializing = false;
  bool _destroying = false;

  ModelCameraProcessor(this._resultStream, this._frameProcessor);

  Future<void> initialize() async {
    if (_initialized || _initializing) {
      throw Exception("Already initialized. Call release() first.");
    }
    _initializing = true;
    await _frameProcessor.initialize();
    _initialized = true;
    _initializing = false;
  }

  Future<void> processFrame(CameraImage cameraImage, int orientation) async {
    if (_initializing || !_initialized || _currentlyProcessing || _destroying || !_frameProcessor.isReady()) {
      return;
    }
    _currentlyProcessing = true;
    final result = await _frameProcessor.run(cameraImage, orientation);
    if (_resultStream.hasListener && result != null) {
      _resultStream.sink.add(result);
    }
    // await Future.delayed(const Duration(milliseconds: 10));
    _currentlyProcessing = false;
  }

  get isWorking => _currentlyProcessing;

  void release() {
    if (!_initialized) {
      throw Exception("Not initialized. Call initialize() first.");
    }
    _destroying = true;
    _frameProcessor.release();
    _initialized = false;
  }

}