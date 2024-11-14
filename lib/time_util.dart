
class Timing {
  List<String> path;
  int ms;
  Timing({required this.path, required this.ms});
}

typedef Timings = List<Timing>;

class TimedResult<T> {
  Timings timingsMs;
  T result;
  TimedResult({required this.timingsMs, required this.result});
}


class DebugTimer {
  final stopwatch = Stopwatch();
  String? name;
  List<String> path;
  Timings timingsMs;

  void next(String? name) {
    if (!stopwatch.isRunning) {
      stopwatch.start();
    }
    else {
      mark();
    }
    this.name = name;
  }

  void mark() {
    if (!stopwatch.isRunning) {
      return;
    }
    final name = this.name;
    if (name != null) {
      timingsMs.add(Timing(path: path.followedBy([name]).toList(), ms: stopwatch.elapsedMilliseconds));
    }
    stopwatch.reset();
  }

  void end() {
    mark();
    stopwatch.stop();
  }

  DebugTimer([List<String>? path, Timings? timingsMs]) : path = path ?? [], timingsMs = timingsMs ?? Timings.empty(growable: true);
}

class DebugTimerStack {
  DebugTimer _state = DebugTimer();
  final List<DebugTimer> _stack = [];

  void push([String? prefix]) {
    _state.next(null);
    _stack.add(_state);
    _state = DebugTimer(prefix != null ? _state.path.followedBy([prefix]).toList() : _state.path, _state.timingsMs);
  }

  void pop() {
    _state = _stack.removeLast();
  }

  void next(String name) => _state.next(name);
  void mark() => _state.mark();
  void end() => _state.end();
  Timings get timingsMs => _state.timingsMs;

}