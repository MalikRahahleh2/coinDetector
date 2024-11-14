import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class AssetUtil {
  static Future<bool> assetExists(String path) async {
    if (kDebugMode) {
      debugPrint("Looking for asset: $path");
    }
    try {
      final data = await rootBundle.load(path);
      if (kDebugMode) {
        debugPrint("Asset found: $path");
      }
      return true;
    }
    on FlutterError {
      if (kDebugMode) {
        debugPrint("Asset not found: $path");
      }
      return false;
    }
  }
}