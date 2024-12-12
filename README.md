# Mean Field Game EA Project

This code simulates the McKean-Vlasov process obtained after solving the Mean Field Game equation.

A more complete overview about the derivation of the solution can be found on the project report.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)

## Installation

Follow these steps to install the project.

### Prerequisites

- Python 3.7+
- pip (Python package installer)

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/EA-Modelisation.git

   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

This code simulates a multistep Mean Field Game Solution.

### Example

The execution of the code is pretty straightforward.

The Jupyter Notebook **main.ipynb** contains the functions **plot_onestep** and **plot_Nsteps** imported from the Python files.

**plot_onestep** plots the McKean-Vlasov drift and the distribution of the agents' value after the step.
**plot_Nsteps** plots 3 graphs:

- 5 agents' trajectories with and without mutual holdings
- The distribution of agents' final equity and of the provision
- The smoothed out density associated with these distributions
