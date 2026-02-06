# Discovery of Hidden Tumor Dynamics via Neural UDEs

To be completely transparent from the start : this project marks a personal milestone. While I have spent time studying the mathematics behind **Universal Differential Equations (UDEs)**, i.e. the interplay between differential operators and universal approximators, this is my very first venture into implementing them in code.

I am stepping out of my pure math comfort zone here. My goal was to see if I could take the theoretical rigour I know and translate it into a working AI architecture that doesn't just fit data, but actually **discovers biological laws**.

## The Problem

In oncology we rarely know everything. We might know how a tumor grows naturally (Gompertz law) but we often struggle to model exactly how a specific patient reacts to a new chemotherapy drug.

* **Pure Physics models** are too rigid. If the parameters change they fail
* **Pure Deep Learning** is too flexible. It fits the training data perfectly but fails to predict the future (due to overfitting)

## The Solution : Hybrid Modeling

This project implements a **Neural Universal Differential Equation**.
Instead of forcing a model to learn everything from scratch, I give it a "First Guess" based on known biology (a biased Gompertz model). Then I embed a Neural Network **inside** the differential equation to learn only what is missing, the drug's pharmacodynamics.

## Experiments & Results

I designed a series of benchmarks to test if this hybrid approach could beat a standard Black Box Neural ODE.

* **Experiment 1 : Reconstruction** Can the model fix a biased physical prior (I gave it a wrong growth rate) and recover the true trajectory ? 
* **Experiment 2 : Physics Recovery** By looking at the weights of the trained network can we extract the mathematical term for the drug's effect ?
* **Experiment 3 : The Forecasting Trap** I hid a specific dose (Day 70) from the training set.
* **Experiment 4 : Zero-Shot Generalization** I tested the model on a completely new dosing schedule it had never seen.

### Visual Proof : Generalization 

![Generalization Benchmark](docs/exp05_generalization_test.png)
*(Fig : The Hybrid UDE adapting to unseen protocols where Pure AI fails)*

## How to Reproduce (Docker)

Since this is my first project involving this tech stack, I wanted to make sure it was reproducible for everyone not just on my machine. I have containerized the entire environment (Julia 1.10, SciML ecosystem, Plots).

**Prerequisites:** Just [Docker Desktop](https://www.docker.com/products/docker-desktop/).

1. **Build and Start:**
```bash
docker-compose up --build -d

```

2. **Run an Experiment:**
To run the generalization test (Experiment 6), simply type:
```bash
docker-compose exec science julia scripts/05_generalization.jl

```


3. **View Results:**
Check the `docs/` folder. The generated plots (like `exp06_generalization.png`) will appear there automatically.

## Repository Structure

* `src/` : Contains the core logic.
* `models.jl` : The Lux.jl neural network definition (using `swish` and `tanh` to avoid stiffness).
* `physics.jl` : The Ground Truth dynamics and the Biased Priors.


* `scripts/` : The numbered experiments corresponding to the report.
* `Project.toml` : The Julia dependency environment.

## Tech Stack & Acknowledgments

I chose **Julia** over Python for this project because of the **DifferentialEquations.jl** ecosystem. When mixing Neural Nets with ODEs the equations become "stiff" (numerically unstable). Julia’s `Vern7` solver and `ForwardDiff` (automatic differentiation) were essential to making this work without crashing. Also I wanted to learn how to code in Julia, it's my first project with this programming language.

* **SciML ecosystem** (Chris Rackauckas et al.) for the incredible tooling.
* **Lux.jl** for the explicit parameter handling in neural networks.

---

*If you find any non-idiomatic code patterns, please remember : I'm a mathematician learning to speak AI for the first time ! Feedback is always welcome.*
