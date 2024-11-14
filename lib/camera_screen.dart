import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

import 'camera_processor.dart';
import 'classify.dart';
import 'classify_overlay.dart';
import 'object_detect.dart';
import 'object_detect_overlay.dart';
import 'time_util.dart';
import 'timings_widget.dart';

class CameraScreen extends StatefulWidget {
  @override
  CameraScreenState createState() => CameraScreenState();

  const CameraScreen({super.key});
}

class CameraScreenState extends State<CameraScreen> {
  final _resultStreamObjectDetect =
      StreamController<TimedResult<YOLOResult>>.broadcast();
  final _resultStreamClassify =
      StreamController<TimedResult<MobileClipResult>>.broadcast();
  final List<ModelCameraProcessor<Object>> _cameraProcessors = [];

  late CameraController _cameraController;
  bool _cameraReady = false;
  bool _initialized = false;
  bool _initializing = false;
  bool _yoloEnabled = false;
  bool _yoloAvailable = false;
  bool _clipEnabled = false;
  bool _clipAvailable = false;
  ResolutionPreset _resolutionPreset =
      Platform.isIOS ? ResolutionPreset.medium : ResolutionPreset.low;

  Future<void> _initializeCameraController() async {
    final cameras = await availableCameras();
    final backCamera = cameras.firstWhere(
        (element) => element.lensDirection == CameraLensDirection.back);
    _cameraController = CameraController(backCamera, _resolutionPreset,
        enableAudio: false, imageFormatGroup: ImageFormatGroup.bgra8888);
    await _initializeCameraRetry();
    setState(() {
      _cameraReady = true;
    });
  }

  Future<void> _initializeCameraRetry() async {
    while (true) {
      try {
        if (Platform.isAndroid) {
          PermissionStatus status = await Permission.camera.status;
          while (!status.isGranted) {
            status = await Permission.camera.request();
            if (!status.isGranted) {
              await Future.delayed(const Duration(seconds: 2));
            }
          }
        }
        SystemChrome.setPreferredOrientations([
          DeviceOrientation.portraitUp,
          DeviceOrientation.portraitDown,
        ]);
        return await _cameraController.initialize();
      } catch (exc) {
        await Future.delayed(const Duration(seconds: 2));
      }
    }
  }

  Future<void> updateProcessors() async {
    if (!_initialized) {
      return;
    }
    final cameraProcessorsLocal =
        List<ModelCameraProcessor<Object>>.from(_cameraProcessors);

    // remove before uninitializing
    setState(() {
      _cameraProcessors.clear();
    });

    while (cameraProcessorsLocal.any((processor) => processor.isWorking)) {
      await Future.delayed(Durations.medium1);
    }
    for (final processor in cameraProcessorsLocal) {
      processor.release();
    }
    cameraProcessorsLocal.clear();

    if (_yoloAvailable && _yoloEnabled) {
      cameraProcessorsLocal.add(ModelCameraProcessor<YOLOResult>(
          _resultStreamObjectDetect,
          YOLOFrameProcessor<YOLOv6SegConfig>(
              factory: YOLOv6SegFactory(),
              detector: YOLOv6SegObjectDetector())));
    }
    if (_clipAvailable && _clipEnabled) {
      cameraProcessorsLocal.add(ModelCameraProcessor<MobileClipResult>(
          _resultStreamClassify, MobileClipFrameClassifier()));
    }
    //
    for (final processor in cameraProcessorsLocal) {
      await processor.initialize();
    }

    setState(() {
      _cameraProcessors.addAll(cameraProcessorsLocal);
    });
  }

  Future<void> _initializeDetector() async {
    assert(_cameraReady);

    _cameraController.startImageStream((CameraImage cameraImage) async {
      if (!_cameraReady || !_initialized) {
        // during camera reset
        return;
      }
      final orientation = _cameraController.description.sensorOrientation;
      for (final processor in _cameraProcessors) {
        processor.processFrame(cameraImage, orientation);
      }
    });
  }

  Future<void> initialize() async {
    if (_initializing) {
      return;
    }
    if (_initialized) {
      throw Exception("Already initialized. Call release() first.");
    }
    _initializing = true;

    final yoloAvailable = await YOLOv6SegObjectDetector.isAvailableFuture;
    final clipAvailable = await MobileClipFrameClassifier.isAvailableFuture;
    setState(() {
      _yoloAvailable = yoloAvailable;
      _clipAvailable = clipAvailable;
    });

    await _initializeCameraController();
    await _initializeDetector();

    setState(() {
      _initialized = true;
    });
    await updateProcessors();
    _initializing = false;
  }

  Future<void> release() async {
    if (!_initialized) {
      throw Exception("Not initialized. Call initialize() first.");
    }

    setState(() {
      _initializing = false;
      _initialized = false;
      _yoloAvailable = false;
      _clipAvailable = false;
      _cameraReady = false;
    });

    _cameraController.stopImageStream();
    await _cameraController.dispose();

    for (final processor in _cameraProcessors) {
      processor.release();
    }

    setState(() {
      _cameraProcessors.clear();
    });
  }

  @override
  void initState() {
    super.initState();
    initialize();
  }

  @override
  void dispose() {
    _cameraController.dispose();

    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(
            title: const Text('Vision camera demo'),
            actions: (<Widget>[
                  PopupMenuButton<ResolutionPreset>(
                    initialValue: _resolutionPreset,
                    onSelected: (ResolutionPreset value) async {
                      await release();
                      setState(() {
                        _resolutionPreset = value;
                        initialize();
                      });
                    },
                    itemBuilder: (BuildContext context) =>
                        <PopupMenuEntry<ResolutionPreset>>[
                      const PopupMenuItem<ResolutionPreset>(
                        value: ResolutionPreset.low,
                        child: Text('Low'),
                      ),
                      const PopupMenuItem<ResolutionPreset>(
                        value: ResolutionPreset.medium,
                        child: Text('Medium'),
                      ),
                      const PopupMenuItem<ResolutionPreset>(
                        value: ResolutionPreset.high,
                        child: Text('High'),
                      ),
                      const PopupMenuItem<ResolutionPreset>(
                        value: ResolutionPreset.veryHigh,
                        child: Text('Very High'),
                      ),
                      const PopupMenuItem<ResolutionPreset>(
                        value: ResolutionPreset.ultraHigh,
                        child: Text('Ultra High'),
                      ),
                      const PopupMenuItem<ResolutionPreset>(
                        value: ResolutionPreset.max,
                        child: Text('Max'),
                      ),
                    ],
                  )
                ] +
                (_yoloAvailable
                    ? <Widget>[
                        IconButton(
                            onPressed: () {
                              setState(() {
                                _yoloEnabled = !_yoloEnabled;
                                updateProcessors();
                              });
                            },
                            isSelected: _yoloEnabled,
                            icon: _yoloEnabled
                                ? const Icon(Icons.rectangle)
                                : const Icon(Icons.rectangle_outlined))
                      ]
                    : <Widget>[]) +
                (_clipAvailable
                    ? <Widget>[
                        IconButton(
                            onPressed: () {
                              setState(() {
                                _clipEnabled = !_clipEnabled;
                                updateProcessors();
                              });
                            },
                            isSelected: _clipEnabled,
                            icon: _clipEnabled
                                ? const Icon(Icons.question_answer)
                                : const Icon(Icons.question_answer_outlined))
                      ]
                    : <Widget>[]))),
        body: _cameraReady
            ? Column(
                children: <Widget>[
                      Stack(
                          children: [
                                AspectRatio(
                                    aspectRatio: 2 / 2,
                                    child: CameraPreview(_cameraController)),
                              ] +
                              (_yoloAvailable && _yoloEnabled
                                  ? [
                                      AspectRatio(
                                          aspectRatio: 2 / 2,
                                          child: YOLODetectionOverlay(
                                              _resultStreamObjectDetect.stream))
                                    ]
                                  : []) +
                              (_clipAvailable && _clipEnabled
                                  ? [
                                      AspectRatio(
                                          aspectRatio: 2 / 2,
                                          child: MobileClipOverlay(
                                              _resultStreamClassify.stream))
                                    ]
                                  : [])),
                    ] +
                    (_yoloAvailable && _yoloEnabled
                        ? <Widget>[
                            StreamTimingsWidget(
                                _resultStreamObjectDetect.stream)
                          ]
                        : []) +
                    (_clipAvailable && _clipEnabled
                        ? <Widget>[
                            StreamTimingsWidget(_resultStreamClassify.stream)
                          ]
                        : []))
            : const Center(child: CircularProgressIndicator()));
  } // build
}
