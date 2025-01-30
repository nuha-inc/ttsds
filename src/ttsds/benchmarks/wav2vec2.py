class Wav2Vec2Benchmark(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(
            name="Wav2Vec2",
            category=BenchmarkCategory.OVERALL,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="Wav2Vec2 embeddings",
            version="1.0.0",
            supported_devices=[DeviceSupport.CPU, DeviceSupport.GPU],
            **kwargs,
        )
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self.device = "cpu" 