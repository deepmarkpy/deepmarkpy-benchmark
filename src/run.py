from benchmark import Benchmark
import json

benchmark = Benchmark()


attacks = ['AdditiveNoiseAttack']

model_name = 'AudioSealModel'

filepaths = ['test.wav']

benchmark.show_available_plugins()

results = benchmark.run(model_name=model_name, attack_types=attacks, filepaths=filepaths)

mean = benchmark.compute_mean_accuracy(results)

print(json.dumps(results, indent=2))
print(json.dumps(mean, indent=2))