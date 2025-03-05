# QMCS Project

## Overview
Code for paper Fidelity Aware Multipath Routing for Multipartite State Distribution in Quantum Networks
The QMCS (Quantum 'Multi' Carlo Simulations) project is designed to provide tools and libraries for performing Monte Carlo simulations of a quantum network, specifically for mulitparite GHZ state distribution. These simulations are used a simplified network model to study quantum systems and their properties. 

## Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/qmcs.git
    ```
2. Navigate to the project directory:
    ```bash
    cd qmcs
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
The package uses NetSquid to calculate the fidelity of the distributed GHZ states.
Hhowever, Netsquid is not set as a requirement and if not installed, an approximate fidelty value used instead.

## Usage
To run a basic simulation, use the following command:
```bash
python example_script.py
```
or checkout the examples notebook


## Contact
For any questions or inquiries, please contact evan.sutcliffe.20 (ɑt) ucl.ac.uk