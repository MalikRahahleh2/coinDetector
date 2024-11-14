import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as image_lib;
import 'package:camera/camera.dart';


class Transform {
  double a, b, c, d, e, f;

  Transform(this.a, this.b, this.c, this.d, this.e, this.f);
  Transform.identity() : a = 1.0, b = 0.0, c = 0.0, d = 0.0, e = 1.0, f = 0.0;

  transform(List<double> xy) {
    final result = List<double>.filled(xy.length, 0.0);
    for (int offset = 0; offset < xy.length; offset += 2) {

      final x = xy[offset], y = xy[offset + 1];
      result[offset] = a * x + b * y + c;
      result[offset + 1] = d * x + e * y + f;
    }
    return result;
  }

  Transform inverse() {
    final det = a * e - b * d;
    return Transform(e / det, -b / det, (b * f - c * e) / det, -d / det, a / det, (c * d - a * f) / det);
  }

  Transform mul(Transform o) {
    return Transform(
        a * o.a + b * o.d,
        a * o.b + b * o.e,
        a * o.c + b * o.f + c,
        d * o.a + e * o.d,
        d * o.b + e * o.e,
        d * o.c + e * o.f + f
    );
  }

}

class TransformedImage {
  Transform transform;
  image_lib.Image image;
  TransformedImage({required this.transform, required this.image});
}

/// ImageUtils
class ImageUtil {

  /// Converts a [CameraImage] in YUV420 format to [image_lib.Image] in RGB format
  static image_lib.Image convertCameraImage(CameraImage cameraImage, [int scale = 1]) {
    if (cameraImage.format.group == ImageFormatGroup.yuv420) {
      return convertYUV420ToImage(cameraImage, scale);
    }
    if (cameraImage.format.group == ImageFormatGroup.bgra8888) {
      return convertBGRA8888ToImage(cameraImage);
    }
    throw ArgumentError("Invalid format.group for cameraImage");
  }


  static Uint8List subsampleImageBytes(Uint8List sourceImageBytes, int width, int height, int scale) {
    // thanks to https://gemini.google.com/app/4a6e55e32560fdb5
    // Validate input parameters
    if (scale <= 1) {
      throw ArgumentError('Invalid scale or image dimensions for subsampling.');
    }

    // Calculate dimensions of the subsampled image
    int newWidth = width ~/ scale;
    int newHeight = height ~/ scale;

    // Create a new Uint8List to store the subsampled image
    Uint8List result = Uint8List(newWidth * newHeight * 4);

    // Iterate through the source image and copy every `scale` pixel to the subsampled image
    int targetIndex = 0;
    for (int y = 0; y < height; y += scale) {
      final rowIndex = y * width;
      int sourceIndex = rowIndex;
      for (int x = 0; x < width; x += scale) {
        result[targetIndex++] = sourceImageBytes[sourceIndex++];
        result[targetIndex++] = sourceImageBytes[sourceIndex++];
        result[targetIndex++] = sourceImageBytes[sourceIndex++];
        result[targetIndex++] = sourceImageBytes[sourceIndex++];
      }
    }

    return result;
  }

  /// Converts a [CameraImage] in BGRA888 format to [image_lib.Image] in RGB format
  static image_lib.Image convertBGRA8888ToImage(CameraImage cameraImage, [int scale = 1]) {
    var imageBytes = cameraImage.planes[0].bytes;
    final width = cameraImage.width;
    final height = cameraImage.height;
    if (scale != 1) {
      // TODO : this is nice, but even nicer will be to be able to center-crop
      //        the image during this step, so we don't do any resizing later.
      imageBytes = subsampleImageBytes(imageBytes, width, height, scale);
    }

    image_lib.Image img = image_lib.Image.fromBytes(
        width: width,
        height: height,
        bytes: imageBytes.buffer,
        numChannels: 4,
        order: image_lib.ChannelOrder.bgra,
    );
    return img;
  }

  /// Converts a [CameraImage] in YUV420 format to [image_lib.Image] in RGB format
  static image_lib.Image convertYUV420ToImage(CameraImage cameraImage, [int scale = 1]) {
    final width = cameraImage.width;
    final height = cameraImage.height;
    final targetWidth = (width / scale).floor();
    final targetHeight = (height / scale).floor();

    final uvRowStride = cameraImage.planes[1].bytesPerRow;
    final uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 1;  // guessing 1 for iOS

    final image = image_lib.Image(width: targetWidth, height: targetHeight);

    final uvXOffsets = List<int>.filled(targetWidth, 0).indexed.map((pair) => uvPixelStride * (pair.$1 / 2).floor()).toList(growable: false);

    final yPlaneBytes = cameraImage.planes[0].bytes;
    final uPlaneBytes = cameraImage.planes[1].bytes;
    final vPlaneBytes = cameraImage.planes[2].bytes;

    for (int ty = 0, sy = 0 ; sy < height; ++ty, sy += scale) {
      final yOffset = sy * width;
      final uvYOffset = uvRowStride * (sy / 2).floor();
      for (int tx = 0, sx = 0; sx < height; ++tx, sx += scale) {
        final int uvIndex = uvYOffset + uvXOffsets[tx];
        final int yIndex = yOffset + sx;

        final y = yPlaneBytes[yIndex];
        final u = uPlaneBytes[uvIndex];
        final v = vPlaneBytes[uvIndex];
        final rgb = yuv2rgb(y, u, v);
        image.setPixel(tx, ty, rgb);
      }
    }
    return image;
  }
  // // TODO : not sure this works properly
  // /// Converts a [CameraImage] in YUV420 format to [Float32List] in [c,h,w] 0..1 RGB format
  // static Float32List convertYUV420ToTensor(CameraImage cameraImage, [int scale = 1]) {
  //   final width = cameraImage.width;
  //   final height = cameraImage.height;
  //
  //   // The Y-plane is guaranteed not to be interleaved with the U/V planes (in particular, pixel stride is always 1 in yPlane.getPixelStride()).
  //   //The U/V planes are guaranteed to have the same row stride and pixel stride (in particular, uPlane.getRowStride() == vPlane.getRowStride() and uPlane.getPixelStride() == vPlane.getPixelStride(); ).
  //   final uvRowStride = cameraImage.planes[1].bytesPerRow;
  //   final uvPixelStride = cameraImage.planes[1].bytesPerPixel ?? 1;  // guessing 1 for iOS
  //
  //   final image = List<double>.filled(3 * width * height, 0);
  //
  //   final int cr = 0, cg = width * height, cb = 2 * cg;
  //   int row = 0;
  //
  //   for (int h = 0; h < height; h += scale) {
  //     row = h * width;
  //     for (int w = 0; w < width; w += scale) {
  //       final int uvIndex =
  //           uvPixelStride * (w / 2).floor() + uvRowStride * (h / 2).floor();
  //       final int yIndex = row + w;
  //
  //
  //       // based on https://fourcc.org/fccyvrgb.php
  //       // TODO : make this more efficient + implement per platform
  //       final y = cameraImage.planes[0].bytes[yIndex];
  //       final u = cameraImage.planes[1].bytes[uvIndex];
  //       final v = cameraImage.planes[2].bytes[uvIndex];
  //       final r = (1.164 * (y - 16)                     + 1.596 * (v - 128)) / 256 - 0.5;
  //       final g = (1.164 * (y - 16) - 0.391 * (u - 128) - 0.813 * (v - 128)) / 256 - 0.5;
  //       final b = (1.164 * (y - 16) + 2.018 * (u - 128)                    ) / 256 - 0.5;
  //
  //       image[cr + row + w] = r;
  //       image[cg + row + w] = g;
  //       image[cb + row + w] = b;
  //
  //     }
  //   }
  //   return Float32List.fromList(image);
  // }

  /// Convert a single YUV pixel to RGB
  // static int yuv2rgb(int y, int u, int v) {
  static  image_lib.Color yuv2rgb(int y, int u, int v) {
    // TODO : this seems to be wrong, as
    // Convert yuv pixel to rgb
    int r = (y + v * 1436 / 1024 - 179).round();
    int g = (y - u * 46549 / 131072 + 44 - v * 93604 / 131072 + 91).round();
    int b = (y + u * 1814 / 1024 - 227).round();

    // Clipping RGB values to be inside boundaries [ 0 , 255 ]
    r = r.clamp(0, 255);
    g = g.clamp(0, 255);
    b = b.clamp(0, 255);

    return image_lib.ColorInt32.rgb(r, g, b);
    // return 0xff000000 |
    // ((b << 16) & 0xff0000) |
    // ((g << 8) & 0xff00) |
    // (r & 0xff);
  }


  static image_lib.Image transpose(image_lib.Image image) {
    return image_lib.flip(image_lib.copyRotate(image, angle: 90.0), direction: image_lib.FlipDirection.horizontal);
  }

  // static TensorImage preprocess(image_lib.Image image, int size) {
  //   TensorImage tensorImage = TensorImage.fromImage(image);
  //   final input = imageProcessor.process(tensorImage);
  //   return input;
  // }
  static List<image_lib.Image> splitChannels(image_lib.Image image) {
    final imageChannels = List<image_lib.Image>.empty(growable: true);
    for (final channel in [image_lib.Channel.red, image_lib.Channel.green, image_lib.Channel.blue]) {
      final imageChannel = image_lib.Image(width: image.width, height: image.height, numChannels: 1);
      image_lib.copyImageChannels(imageChannel, from: image, red: channel);
      imageChannels.add(imageChannel.convert(format: image_lib.Format.float32));
    }
    return imageChannels;
  }


  static TransformedImage letterbox({required image_lib.Image image, required int size, image_lib.Color? backgroundColor}) {
    final s = size / math.max(image.width, image.height);
    final w = math.min(size, (s * image.width).round());
    final h = math.min(size, (s * image.height).round());

    final resizedImage = image_lib.copyResize(
      image,
      width: size,
      height: size,
      backgroundColor: backgroundColor,
      maintainAspect: true,
    );


    // transforms from relative coordinates in the output image to relative coordinates in the input image
    final fromRelativeCoordsTransform = Transform(size * 1.0, 0, 0, 0, size * 1.0, 0);
    final translateTransform = Transform(s, 0, math.max(0, (size - w) / 2), 0, s, math.max(0, (size - h) / 2));
    final toRelativeCoordsTransform = Transform(1/image.width, 0, 0, 0, 1/image.height, 0);
    final transform = toRelativeCoordsTransform.mul(translateTransform.inverse()).mul(fromRelativeCoordsTransform);

    return TransformedImage(transform: transform, image: resizedImage);
  }
}