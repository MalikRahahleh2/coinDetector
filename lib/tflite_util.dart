import 'dart:io';
import 'dart:math';
import 'dart:typed_data';
import 'package:collection/collection.dart';
import 'package:demo_camera/image_util.dart';
import 'package:demo_camera/math_util.dart';
import 'package:demo_camera/time_util.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
// import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as image_lib;

class TensorImageData {
  final ByteBuffer data;
  final List<int> shape;
  final TensorType type;

  const TensorImageData(
      {required this.data, required this.shape, required this.type});
}

class ShapedDataList<T> {
  TypedDataList<T> list;
  List<int> shape;
  ShapedDataList({required this.list, required this.shape});
}

//
// extension IntListMixin on List<int> {
//   get product => fold(1, (result, el) => result * el);
//   get sum => fold(0, (result, el) => result + el);
// }
//
// extension ListElementCount on List<int> {
//   get numEl => shape.product;
// }
enum ShapeOrder {
  chw,
  hwc,
  cwh,
  whc,
}

class TfliteUtil {
  // static final imageProcessor = ImageProcessorBuilder()
  //     .add(ResizeOp(inputSize, inputSize, ResizeMethod.NEAREST_NEIGHBOUR))
  //     .add(NormalizeOp(0.0, 255.0))   // 0 .. 255 -> 0.0 .. 1.0
  //     .build();


  static ByteBuffer concatBuffers(Iterable<ByteBuffer> buffers) {
    final totalLengthInBytes = buffers.fold(0, (sum, buffer) => sum + buffer.lengthInBytes);
    final dataList = Uint8List(totalLengthInBytes);
    int offset = 0;
    for (final buffer in buffers) {
      final end = offset + buffer.lengthInBytes;
      dataList.setRange(offset, end, buffer.asUint8List());
      offset = end;
    }
    return dataList.buffer;
  }

  static TensorImageData toTensor({required image_lib.Image image, required ShapeOrder order}) {
    var shape = [image.width, image.height, image.numChannels];
    const type = TensorType.float32;

    if (order == ShapeOrder.whc || order == ShapeOrder.cwh) {
      image = ImageUtil.transpose(image);
      shape = [shape[1], shape[0], shape[2]];
    }
    if (order == ShapeOrder.hwc || order == ShapeOrder.whc) {
       image = image.convert(format: image_lib.Format.float32);
       return TensorImageData(data: image.buffer, shape: shape, type: type);
    }
    // channel-first
    final buffer = concatBuffers(ImageUtil.splitChannels(image).map((channel) => channel.convert(format: image_lib.Format.float32).buffer));
    shape = [shape[2], shape[0], shape[1]];
    return TensorImageData(data: buffer, shape: shape, type: TensorType.float32);
  }

  static final delegateOptionsByOS = {
    // "ios": [() => [GpuDelegate()]],
    // "android": [() => [GpuDelegateV2()]],
  };

  static Future<Interpreter> loadInterpreter(String assetPath) async {
    final delegateOptions = List<List<Delegate> Function()>.from(delegateOptionsByOS[Platform.operatingSystem] ?? []);
    delegateOptions.add(() => [XNNPackDelegate()]);
    delegateOptions.add(() => []);


    for (final makeDelegates in delegateOptions) {
      final options = InterpreterOptions();
      options.threads = -1;
      final delegates = makeDelegates();

      for (final delegate in delegates) {
        options.addDelegate(delegate);
      }

      try {
        final interpreter = await Interpreter.fromAsset(
          assetPath,
          options: options,
        );
        interpreter.allocateTensors();

        return interpreter;
      }
      catch(e) {
        if (delegates.isNotEmpty) {
          continue;
        }
        rethrow;
      }
    }
    throw Exception("Could not load interpreter");
  }
  // static final tensorToImageTypeMap = {
  //   TensorType.float32: image_lib.Format.float32,
  //   TensorType.boolean: image_lib.Format.uint1,
  //   TensorType.uint8: image_lib.Format.uint8,
  // }
  static TypedDataList<num> createBuffer(int count, TensorType type) {
    // TODO : consider supporting Float32x4List type
    switch(type) {
      case TensorType.float32:
        return Float32List(count);
      case TensorType.uint8:
        return Uint8List(count);
      default:

        // NOTE : while BoolList is a space-efficient implementation, I am not
        // sure how tflite_flutter packs the TensorType.bool format. It's most
        // likely a Uint8, but not sure.
        throw ArgumentError("TensorType not supported");
    }
  }



  // static List<List<double>> runImagesThoughModel(final Interpreter interpreter, List<TensorImage> inputs) {
  static List<ShapedDataList<num>> runImagesThoughModel(final Interpreter interpreter, List<TensorImageData> inputs, DebugTimerStack? dt) {
    dt?.next("Prep");
    final inputTensors = interpreter.getInputTensors();
    assert(inputTensors.length == 1);
    for (final (index, expectedInput) in inputTensors.indexed) {
      // assert(expectedInput.shape.slice(1).equals(inputs[index].tensorBuffer.getShape()));
      // assert(expectedInput.type == inputs[index].tensorBuffer.getDataType());
      final input = inputs[index];
      assert(expectedInput.shape.slice(1).equals(input.shape));
      assert(expectedInput.type == input.type);
    }
    // final outputs = interpreter.getOutputTensors().map((value) => TensorBuffer.createFixedSize(value.shape, value.type)).toList();
    final outputTensors = interpreter.getOutputTensors();
    final outputs = outputTensors.map((value) {
      if (value.type == TensorType.float32) {
        final buffer = Float32List(value.numElements());
        return ShapedDataList<double>(list: buffer, shape: value.shape);
      }
      if (value.type == TensorType.uint8) {
        final buffer = Uint8List(value.numElements());
        return ShapedDataList<int>(list: buffer, shape: value.shape);
      }
      if (value.type == TensorType.int8) {
        final buffer = Int8List(value.numElements());
        return ShapedDataList<int>(list: buffer, shape: value.shape);
      }
      throw UnsupportedError("Unsupported output tensor type");
    }).toList();
    final inputList = inputs.map((input) => input.data).toList(growable: false);
    final outputMap = outputs.map((value) => value.list.buffer).toList(growable: false).asMap();

    dt?.next("Run");
    // interpreter.runForMultipleInputs(inputs.map((input) => input.buffer).toList(), outputs.map((value) => value.buffer).toList().asMap());
    interpreter.runForMultipleInputs(inputList, outputMap);

    dt?.mark();

    return outputs;
  }


}