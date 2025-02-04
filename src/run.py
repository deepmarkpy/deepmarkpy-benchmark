from benchmark import Benchmark

benchmark = Benchmark()


attacks = ['AdditiveNoiseAttack', 'VAEAttack']

model_name = 'AudioSealModel'

filepaths = ['test.wav']

results = benchmark.run(model_name=model_name, attack_types=attacks, filepaths=filepaths)

print(results)