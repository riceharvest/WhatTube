// AudioWorkletProcessor for low-latency 16 kHz Float32 PCM extraction

class PCMWorkletProcessor extends AudioWorkletProcessor {
  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0 && input[0].length > 0) {
      // Transfer mono channel data (Float32Array)
      const channelData = input[0];
      // Clone buffer to avoid detachment issues during transfer
      this.port.postMessage(new Float32Array(channelData));
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PCMWorkletProcessor);
