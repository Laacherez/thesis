import os
import itertools
import yaml

# You need to check what simulation_config.yaml holds.
# This code is meant to run locally.


# PLEASE check setup.py before anything.

# -------------------- utilities --------------------


def set_blas_threads(n_threads: int | None):
    """
    basic linear algebra subprograms.
    blas are alredy implemented in numpy/scipy. we only restrict np/sp to a single thread per calculation to
    avoid artificial slowing down. example :
    8 simu processes with 8 blas = 64 threads, whereas 8 simu with 1 blas gives 8 thread.
    i also think blas is a funny name.
    """
    if n_threads is None:
        return
    s = str(int(max(1, n_threads)))
    os.environ.setdefault("OMP_NUM_THREADS", s)
    os.environ.setdefault("MKL_NUM_THREADS", s)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", s)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", s)


# ----------------- batch runner --------------------


class SimulationBatchRunner:
    """
    I only used classes here to lern a little bit more about them, but I didn't like it so much.
    I think I will try again with another project where it makes more sense, but I am not sure yet.
    """

    def __init__(self, config_file):
        """Initialise the runner with the configuration."""
        self.config_file = config_file
        self.config = self.load_config(config_file)

        self.base_output_dir = self.config["base_output_dir"]
        self.log_dir = self.config["log_dir"]
        self.param_dir = self.config["param_dir"]

        os.makedirs(self.base_output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.param_dir, exist_ok=True)

        self.save_every = self.config.get("save_every", 1000)
        self.max_workers = self.config.get("max_workers", None)

        self.use_convergence = self.config.get("use_convergence", True)
        self.tol = self.config.get("tol", 5e-3)
        self.check_every = self.config.get("check_every", 5)
        self.patience = self.config.get("patience", 3)
        self.min_trajectories = self.config.get("min_trajectories", 20)
        self.max_trajectories = self.config.get(
            "max_trajectories", 100000
        )  # You need to stop the simulation at some point.

    def load_config(self, config_file):
        """Load the configuration from simulation_config.yaml."""
        with open(config_file, "r") as file:
            return yaml.safe_load(file)

    def generate_param_file(self, output_file, **params):
        """Generate the parameter file for the simulation."""
        with open(output_file, "w") as f:
            for key, value in params.items():
                f.write(f"{key}={value}\n")

    def get_output_subdir(self, **params):
        """
        Generate a directory path for output based on key params.
        """
        key_order = [
            "mu11",
            "mu12",
            "mu22",
            "kappa",
            "alpha",
            "l",
            "n_tasks",
            "trajectory_per_task",
            "use_convergence",
            "tol",
            "check_every",
            "patience",
            "min_trajectories",
            "max_trajectories",
            "save_every",
        ]
        parts = []
        for k in key_order:
            if k in params and params[k] is not None:
                parts.append(f"{k}_{params[k]}")
        param_str = "_".join(parts)
        return os.path.join(self.base_output_dir, param_str)

    def run_simulation(
        self, mu11, mu12, mu22, kappa, alpha, l, n_tasks, trajectory_per_task
    ):
        """Run the simulation for a specific parameter set."""
        from simulator_save_every import SinusoidalTrapSimulator

        # workers
        effective_max_workers = self.max_workers
        if effective_max_workers is None:
            effective_max_workers = min(int(os.cpu_count() or 1), int(n_tasks))

        params = {
            "mu11": mu11,
            "mu12": mu12,
            "mu22": mu22,
            "kappa": kappa,
            "alpha": alpha,
            "l": l,
            "n_tasks": n_tasks,
            "trajectory_per_task": trajectory_per_task,
            "use_convergence": self.use_convergence,
            "tol": self.tol,
            "check_every": self.check_every,
            "patience": self.patience,
            "min_trajectories": self.min_trajectories,
            "max_trajectories": self.max_trajectories,
            "save_every": self.save_every,
            "max_workers": effective_max_workers,
        }

        output_subdir = self.get_output_subdir(**params)

        param_file = os.path.join(
            self.param_dir,
            f"params_mu_{mu11}_{mu12}_{mu22}__k_{kappa}__a_{alpha}__l_{l}__tasks_{n_tasks}.txt",
        )
        self.generate_param_file(param_file, **params)

        # simulator
        simulator = SinusoidalTrapSimulator(
            mu11,
            mu12,
            mu22,
            kappa,
            alpha,
            l,
            output_dir=self.base_output_dir,
            N_trajectories=trajectory_per_task,  # use_convergence=False
            n_tasks=n_tasks,
            output_subfile=output_subdir,
            save_every=self.save_every,
            use_convergence=self.use_convergence,
            tol=self.tol,
            check_every=self.check_every,
            patience=self.patience,
            min_trajectories=self.min_trajectories,
            max_trajectories=self.max_trajectories,
        )

        simulator.run_all_tasks(max_workers=effective_max_workers)

    def run_all_simulations(self):
        """Run all simulations based on the config parameters."""
        # Backwards compatibility
        mu11_list = self.config["mu11"]
        mu12_list = self.config["mu12"]
        mu22_list = self.config["mu22"]
        k_list = self.config.get("ks", self.config.get("k", []))
        alpha_list = self.config.get("alphas", self.config.get("alpha", []))
        l_list = self.config.get("ls", self.config.get("l", []))
        n_tasks_list = self.config.get("n_taskss", self.config.get("n_tasks", []))
        traj_per_task_list = self.config.get(
            "trajectory_per_tasks", self.config.get("trajectory_per_task", [])
        )

        for (
            mu11,
            mu12,
            mu22,
            kappa,
            alpha,
            l,
            n_tasks,
            trajectory_per_task,
        ) in itertools.product(
            mu11_list,
            mu12_list,
            mu22_list,
            k_list,
            alpha_list,
            l_list,
            n_tasks_list,
            traj_per_task_list,
        ):
            self.run_simulation(
                mu11, mu12, mu22, kappa, alpha, l, n_tasks, trajectory_per_task
            )
            print(
                "Simulation completed for:",
                f"mu=({mu11},{mu12},{mu22})",
                f"kappa={kappa}",
                f"alpha={alpha}",
                f"l={l}",
                f"n_tasks={n_tasks}",
                f"trajectories/task={trajectory_per_task}",
                f"convergence={self.use_convergence} tol={self.tol}",
            )


# -------------------- main --------------------

if __name__ == "__main__":

    CONFIG_PATH = "simulation_config.yaml"
    with open(CONFIG_PATH, "r") as _f:
        _cfg = yaml.safe_load(_f)

    set_blas_threads(_cfg.get("blas_threads", 1))

    runner = SimulationBatchRunner(CONFIG_PATH)
    runner.run_all_simulations()
