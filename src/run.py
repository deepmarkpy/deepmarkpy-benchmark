from benchmark import Benchmark

benchmark = Benchmark()


attacks = ['SameModelAttack']

model_name = 'AudioSealModel'

filepaths = ['test.wav']

benchmark.show_available_plugins()

results = benchmark.run(model_name=model_name, attack_types=attacks, filepaths=filepaths)

print(results)