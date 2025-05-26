### **Executable Plan: Performance and Scalability**

#### **Task 4.1: Performance Profiling and Bottleneck Analysis**

**Objective:** To quantitatively identify performance hotspots in the training loop before attempting optimizations. The audit notes that the project currently lacks profiling, making this the essential first step[cite: 134].

**Steps:**

1.  **Select Profiling Tools:**
    * Use a standard Python profiler like `cProfile` for function-level timing or a sampling profiler like `py-spy` for a lower-overhead view of a running process.

2.  **Execute a Profiled Training Run:**
    * Run a standard training session for a fixed number of steps (e.g., one full epoch of 2048 steps) with the profiler attached.
    * Example command: `python -m cProfile -o train.prof keisei/train.py --total-timesteps=2048`.

3.  **Analyze Profiling Results:**
    * Visualize the profiler's output using a tool like `snakeviz` to create an interactive flame graph.
    * Identify the functions where the most time is spent. The audit hypothesizes that the Shogi engine's move generation (`get_legal_moves`) is a likely bottleneck[cite: 426, 427]. The analysis should confirm this or identify other hotspots, such as data processing or logging.

4.  **Document Findings:**
    * Create a brief report summarizing the findings. This data will justify the specific parallelization strategy chosen in the next task. For instance, if the environment step is the bottleneck, parallel data collection is the correct solution.

---

#### **Task 4.2: Implement Parallel Experience Collection**

**Objective:** To decouple CPU-bound environment interaction from GPU-bound learning, thereby improving overall training throughput and GPU utilization[cite: 131, 132]. The audit proposes two primary implementation paths[cite: 333, 339].

**Option A: Custom `multiprocessing` Implementation**

This approach offers maximum control but requires more manual implementation.

1.  **Create a `SelfPlayWorker` Process:**
    * Define a new class, `SelfPlayWorker`, that inherits from `multiprocessing.Process`.
    * Its `run()` method will contain a loop that instantiates its own `ShogiGame` and agent network, collects trajectories of experience (e.g., one full episode), and puts the collected data onto a shared `multiprocessing.Queue`.

2.  **Modify the `Trainer` to Manage Workers:**
    * In the `Trainer` class, create and manage a pool of `SelfPlayWorker` processes.
    * The `Trainer`'s main loop will no longer step the environment directly. Instead, it will fetch completed trajectories from the shared queue to fill its `ExperienceBuffer`.
    * This allows the `Trainer`'s main process to focus on running the PPO learning updates on the GPU as soon as enough data is available.

3.  **Implement Model Synchronization:**
    * The workers need the most recent policy network to generate useful data.
    * Periodically, the `Trainer` must send its agent's updated `state_dict` to all worker processes, which will then load the new weights into their local copy of the agent.

**Option B: Gymnasium Interface and Vectorized Environments**

This approach leverages existing, well-tested libraries for parallelism.

1.  **Create a Gymnasium-Compliant Wrapper:**
    * Create a new class that wraps `ShogiGame` and implements the standard Gymnasium API (`reset`, `step`, `observation_space`, `action_space`)[cite: 341]. This makes the custom Shogi engine compatible with the broader RL ecosystem.

2.  **Integrate a `VecEnv` Wrapper:**
    * Replace the single environment instance in the `Trainer` with a vectorized environment wrapper like Gymnasium's `AsyncVectorEnv`.
    * This wrapper automatically handles the creation of multiple subprocesses, each with its own game instance.

3.  **Adapt the Training Loop for Batched Operations:**
    * Modify the `Trainer`'s loop to work with the `VecEnv` interface. Calls to `vec_env.step()` will now return a batch of observations, rewards, and "done" flags from all parallel environments at once. This simplifies the training loop, as the complexity of inter-process communication is handled by the library.

---

#### **Task 4.3: Enhance CI for a Parallel System**

**Objective:** To expand the Continuous Integration pipeline to validate the more complex, parallelized codebase.

**Steps:**

1.  **Create a Parallelism Smoke Test:**
    * Add a new, dedicated test that runs a very short training session (e.g., 100 timesteps) using the parallel environment system with at least two worker processes.
    * The test's purpose is not to check for convergence but to ensure that the parallel system can initialize, run without deadlocking, and terminate cleanly.

2.  **Update the CI Workflow:**
    * Add a new job to the CI configuration (e.g., in the `.github/workflows/ci.yml` file).
    * This job will run the new smoke test. It can be marked as a "slow" or "integration" test and configured to run only on merges to the main branch to keep pull request checks fast.