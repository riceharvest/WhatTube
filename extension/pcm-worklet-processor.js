// AudioWorkletProcessor for low-latency 16 kHz Float32 PCM extraction with quantum batching

class PCMWorkletProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 2048; // 128ms @ 16kHz (~8 frames/sec instead of 125 frames/sec)
    this.buffer = new Float32Array(this.bufferSize);
    this.bufferIndex = 0;
  }

  process(inputs, outputs, parameters) {
    const input = inputs[0];
    if (input && input.length > 0 && input[0].length > 0) {
      const channelData = input[0];
      const channelLength = channelData.length;

      let sourceIndex = 0;
      while (sourceIndex < channelLength) {
        const remainingSpace = this.bufferSize - this.bufferIndex;
        const toCopy = Math.min(channelLength - sourceIndex, remainingSpace);

        this.buffer.set(channelData.subarray(sourceIndex, sourceIndex + toCopy), this.bufferIndex);
        this.bufferIndex += toCopy;
        sourceIndex += toCopy;

        if (this.bufferIndex >= this.bufferSize) {
          this.port.postMessage(new Float32Array(this.buffer));
          this.bufferIndex = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor("pcm-worklet-processor", PCMWorkletProcessor);
