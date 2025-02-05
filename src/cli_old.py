import os
import json
import argparse
from benchmark import Benchmark

def main():
    parser = argparse.ArgumentParser(description="Run watermarking benchmark with various attacks.")
    parser.add_argument("--wav_files_dir", type=str, help="Path to the directory containing .wav files.")
    parser.add_argument("--model_name", type=str, choices=["AudioSeal", "WavMark", "SilentCipher"], required=True,
                        help="Name of the model to benchmark ('AudioSeal', 'WavMark', or 'SilentCipher').")
    parser.add_argument("--attacks", type=str, nargs="+", default=None, help="List of attacks to apply. Defaults to all attacks.")
    parser.add_argument("--collusion_size", type=int, default=5, help="Number of watermarked signals to use in a collusion attack (default: 5).")
    parser.add_argument("--cents", type=float, default=5.0, help="Pitch shift in cents for pitch shifting attack (default: 5).")
    parser.add_argument("--stretch_rate", type=float, default=1.4, help="Stretch rate for time stretch attack (default: 1.4).")
    parser.add_argument("--inverted_stretch_rate", type=float, default=2.0, help="Inverted stretch rate for inverted time stretch attack (default: 2.0).")
    parser.add_argument("--noise_level", type=float, default=0.003, help="Noise level for additive noise attack (default: 0.001).")
    parser.add_argument("--zero_cross_pause_length", type=int, default=20, help="Pause length in samples for zero-cross insert attack (default: 20).")
    parser.add_argument("--zero_cross_min_distance", type=float, default=1.0, help="Minimum distance between zero-cross pauses in seconds (default: 1.0).")
    parser.add_argument("--cut_max_sequence_length", type=int, default=50, help="Maximum sequence length for cut samples attack (default: 50).")
    parser.add_argument("--cut_num_sequences", type=int, default=20, help="Number of sequences to cut in the cut samples attack (default: 20).")
    parser.add_argument("--cut_duration", type=float, default=0.5, help="Duration over which cuts occur in the cut samples attack (default: 0.5 seconds).")
    parser.add_argument("--num_flips", type=int, default=20, help="Number of sample flips in the flip samples attack (default: 20).")
    parser.add_argument("--flip_duration", type=float, default=0.5, help="Duration over which flips occur in the flip samples attack (default: 0.5 seconds).")
    parser.add_argument("--wavelet", type=str, default="db1", help="Wavelet type for wavelet-based denoising attack (default: db1).")
    parser.add_argument("--wt_mode", type=str, default="soft", choices=["soft", "hard"], help="Wavelet thresholding mode for wavelet-based denoising (default: soft).")
    parser.add_argument("--output_file", type=str, default="result.json", help="Path to save the benchmark results (default: result.json).")

    args = parser.parse_args()

    wav_files = [os.path.join(args.wav_files_dir, f) for f in os.listdir(args.wav_files_dir) if f.endswith(".wav")]

    sampling_rate = 44100 if args.model_name == "SilentCipher" else 16000

    benchmark = Benchmark()

    results = benchmark.run(
        wav_files,
        model_name=args.model_name,
        attack_types=args.attacks,
        sampling_rate=sampling_rate,
        collusion_size=args.collusion_size,
        cents=args.cents,
        stretch_rate=args.stretch_rate,
        inverted_stretch_rate=args.inverted_stretch_rate,
        noise_level=args.noise_level,
        zero_cross_pause_length=args.zero_cross_pause_length,
        zero_cross_min_distance=args.zero_cross_min_distance,
        cut_max_sequence_length=args.cut_max_sequence_length,
        cut_num_sequences=args.cut_num_sequences,
        cut_duration=args.cut_duration,
        num_flips=args.num_flips,
        flip_duration=args.flip_duration,
        wavelet=args.wavelet,
        wt_mode=args.wt_mode,
    )

    metrics = benchmark.compute_mean_accuracy(results)

    print("Mean Accuracy for Each Attack:")
    for attack, mean_accuracy in metrics.items():
        print(f"{attack}: {mean_accuracy:.2f}")

    with open(args.output_file, "w") as fp:
        json.dump(results, fp)

    print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()