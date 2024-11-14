import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';

import 'time_util.dart';

class TimingsWidget extends StatelessWidget {
  final Timings timingsMs;
  const TimingsWidget({super.key, required this.timingsMs});

  Widget buildCell(String name, int value, int total, Color color)  => Flexible(
    flex: max(1, value),
    fit: FlexFit.tight,
    child: Container(
      color: color,
      child: Text('${value}ms (${(100 * value/total).round().toString()}%) $name', overflow: TextOverflow.clip,maxLines: 1,softWrap: false,),
    ),
  );

  @override
  Widget build(BuildContext context) {
    const colors = [Colors.blue, Colors.purple, Colors.pink, Colors.red, Colors.orange, Colors.yellow];
    final total = timingsMs.fold(0, (total, timing) => total + max(1, timing.ms));
    final cells = timingsMs.indexed.map((pair) => buildCell(pair.$2.path[pair.$2.path.length - 1], pair.$2.ms, total, colors[pair.$1 % colors.length])).toList();
    return SizedBox(height: 36, width: null, child:
    // Row(
      // children: [
        Flex(
            direction: Axis.horizontal,
            children: [
              Expanded(child: Flex(direction: Axis.horizontal,clipBehavior: Clip.hardEdge, children: cells)),
              SizedBox(
                  width: 100,
                  child: Text('${total}ms', textAlign: TextAlign.right,),

              )
            ],
        )

      // ],
    // )
    );
  }
}

class StreamTimingsWidget<T> extends StatefulWidget {
  final Stream<TimedResult<T>> _stream;
  const StreamTimingsWidget(this._stream, {super.key});

  @override
  State<StatefulWidget> createState()  => StreamTimingsWidgetState();
}


class StreamTimingsWidgetState<T> extends State<StreamTimingsWidget<T>> {
  Timings? lastTimingsMs;
  StreamSubscription<TimedResult<T>>? _subscription;

  @override
  void initState() {
    super.initState();
    _subscription = widget._stream.listen((timedResult) {
      setState(() {
        lastTimingsMs = timedResult.timingsMs;
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
  Widget build(BuildContext context) => lastTimingsMs == null ? const SizedBox.shrink() : TimingsWidget(timingsMs: lastTimingsMs!);
}