import numpy as np
import time

# Constants
PHI = (1 + np.sqrt(5)) / 2  # Golden Ratio
PI = np.pi
C = 3.14159  # Example constant
DOUBLE_PI = 2 * np.pi

# Function to run do_kernel_test (placeholder)
def do_kernel_test(seed):
    # Simulating a test run with some calculations
    np.random.seed(seed)
    return np.random.rand() * seed  

# Function to run tests
def multi_run_until_2pi():
    speedup_measurements = []
    seed = 42  # Starting seed
    speedup_ratio = 0
    phase_checks = []

    while speedup_ratio < DOUBLE_PI:
        start_time = time.time()
        # Placeholding for the kernel test method
        result = do_kernel_test(seed)
        end_time = time.time()

        elapsed_time = end_time - start_time
        speedup_measurement = result / elapsed_time
        speedup_measurements.append(speedup_measurement)
        speedup_ratio = speedup_measurement  # Adjust according to your speedup calculation logic

        # Track phase checks
        phase_checks.append((seed, speedup_measurement))
        print(f"Phase check for seed {seed}: {speedup_measurement}")

        # Increment seed for next iteration
        seed += 1

    print(f"Final speedup ratio: {speedup_ratio:.4f}")
    print("Phase checks:")
    for phase in phase_checks:
        print(f"Seed: {phase[0]}, Speedup: {phase[1]:.4f}")

if __name__ == '__main__':
    multi_run_until_2pi()