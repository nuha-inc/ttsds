class AllosaurusBenchmark(Benchmark):
    def __init__(self, **kwargs):
        super().__init__(
            name="Allosaurus",
            category=BenchmarkCategory.PHONETICS,
            dimension=BenchmarkDimension.N_DIMENSIONAL,
            description="Allosaurus phoneme recognition",
            version="1.0.0",
            supported_devices=[DeviceSupport.CPU],
            **kwargs,
        )
        self.model = None 