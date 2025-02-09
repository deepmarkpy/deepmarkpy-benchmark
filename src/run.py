import argparse
import json
import os
from benchmark import Benchmark

def main():

    benchmark = Benchmark()
    
    models, attacks, valid_args = benchmark.get_available_args()
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="Run DeepMark Benchmark CLI")

    # Add model and attack selection
    parser.add_argument("--wav_files_dir", type=str, help="Path to the directory containing .wav files.", required=True)
    parser.add_argument("--wm_model", type=str, choices=models, required=True, help="Watermarking model to use.")
    parser.add_argument("--attack_types", type=str, nargs='*', choices=attacks, default=None, metavar="ATTACK", help="List of attacks to apply. Allowed values: " + ", ".join(attacks))
    
    # Dynamically add configuration parameters from the available plugins
    for arg, default_value in valid_args.items():
        if isinstance(default_value, bool):
            parser.add_argument(f"--{arg}", action='store_true', help=f"Enable {arg} (default: {default_value})")
        else:
            parser.add_argument(f"--{arg}", type=type(default_value), default=default_value, help=f"Set {arg} (default: {default_value})")
    
    # Parse command-line arguments
    args = parser.parse_args()
    
    # Convert parsed arguments to a dictionary
    args_dict = vars(args)
    
    filepaths = [os.path.join(args.wav_files_dir, f) for f in os.listdir(args.wav_files_dir) if f.endswith(".wav")]
    
    # Run the benchmark
    results = benchmark.run(
        filepaths=filepaths,
        **args_dict
    )
    
    # Save results to JSON
    with open("benchmark_results.json", "w") as fp:
        json.dump(results, fp, indent=4)
    
    print("Benchmark completed. Results saved to benchmark_results.json")

    with open("benchmark_stats.json", "w") as fp:
        json.dump(benchmark.compute_mean_accuracy(results), fp, indent=4)

    print("Stats saved to benchmark_stats.json")

if __name__ == "__main__":
    main()