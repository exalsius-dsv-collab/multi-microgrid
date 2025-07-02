# Multi-Microgrid Vessim Test Environment

> **Note**: This README was created with assistance from an LLM.

A comprehensive Vessim simulation environment that models three interconnected microgrids across different geographical locations to demonstrate distributed renewable energy systems with intelligent load balancing and dynamic computing load management.

## Overview

This test environment simulates a distributed energy system consisting of three microgrids located in Berlin (Germany), San Francisco (USA), and Sydney (Australia). Each microgrid operates with computing systems, solar generators, and energy storage, coordinated through a global load balancing controller that optimizes energy consumption based on renewable energy availability and battery state of charge.

## System Architecture

### Core Components

- **3 Microgrids**: Each with computing systems, solar generators, and energy storage
- **Global Load Balancing Controller**: Distributes computing load across microgrids based on renewable energy availability
- **Solar Power Generation**: Uses real-world solar data from Solcast 2022 dataset
- **Energy Storage**: SimpleBattery components for each microgrid
- **Dynamic Computing Loads**: AdjustableSignal-based computing nodes that can be controlled in real-time
- **Comprehensive Monitoring**: Data collection and analysis with modular visualization

## Microgrid Details

### Berlin Microgrid (Germany)

- **Location**: Berlin, Germany
- **Computing Load**: 900W total (3 computing nodes)
  - Server 1: 400W
  - Server 2: 300W
  - Server 3: 200W
- **Solar Generation**: 3 solar panels (7.5kW total capacity)
  - Solar Panel 1: 3.0kW
  - Solar Panel 2: 2.5kW
  - Solar Panel 3: 2.0kW
- **Energy Storage**: 3.5kWh SimpleBattery
  - Initial SOC: 65%
  - Minimum SOC: 10%

### San Francisco Microgrid (USA)

- **Location**: San Francisco, USA
- **Computing Load**: 1,400W total (4 computing nodes)
  - Server 1: 500W
  - Server 2: 400W
  - Server 3: 300W
  - Server 4: 200W
- **Solar Generation**: 3 solar panels (9.0kW total capacity)
  - Solar Panel 1: 3.5kW
  - Solar Panel 2: 3.0kW
  - Solar Panel 3: 2.5kW
- **Energy Storage**: 4.5kWh SimpleBattery
  - Initial SOC: 75%
  - Minimum SOC: 15%

### Sydney Microgrid (Australia)

- **Location**: Sydney, Australia
- **Computing Load**: 750W total (3 computing nodes)
  - Server 1: 350W
  - Server 2: 250W
  - Server 3: 150W
- **Solar Generation**: 3 solar panels (10.5kW total capacity)
  - Solar Panel 1: 4.0kW
  - Solar Panel 2: 3.5kW
  - Solar Panel 3: 3.0kW
- **Energy Storage**: 5.5kWh SimpleBattery
  - Initial SOC: 70%
  - Minimum SOC: 10%

## Load Balancing Controller

### MultiMicrogridLoadBalancer

The `MultiMicrogridLoadBalancer` controller implements intelligent load distribution across microgrids based on renewable energy availability and storage capacity.

#### Key Features:

- **Real-time Monitoring**: Tracks solar power generation and battery state of charge across all microgrids
- **Availability Scoring**: Calculates availability score for each microgrid (weighted 70% solar power, 30% battery SOC)
- **Dynamic Load Adjustment**: Scales computing load from 20% to 150% of base consumption
- **Cross-Microgrid Coordination**: Global controller that manages all microgrids simultaneously

#### Load Balancing Algorithm:

1. **Solar Power Assessment**: Normalizes solar generation to 0-1 scale (based on 5kW maximum)
2. **Battery State Evaluation**: Incorporates current battery state of charge
3. **Availability Calculation**: Combines solar (70%) and battery (30%) factors
4. **Load Distribution**: Adjusts computing loads proportionally across microgrids

## Technical Implementation

### Custom Signal System

The simulation uses a custom `AdjustableSignal` class to enable dynamic load control:

```python
class AdjustableSignal(vs.Signal):
    def __init__(self, initial_value: float):
        super().__init__()
        self._value = initial_value

    def now(self, at=None):
        return self._value

    def set_value(self, value: float):
        self._value = value
```

This allows the load balancer to dynamically modify computing loads based on microgrid power availability.

### Controller Architecture

The global load balancer is attached to the Berlin microgrid but coordinates all microgrids through a shared config dictionary.

## Simulation Configuration

### Environment Setup

- **Simulation Start**: June 15, 2022 00:00:00
- **Duration**: 48 hours
- **Step Size**: 300 seconds (5 minutes)
- **Data Source**: Solcast 2022 Global Dataset
- **Load Balancing Frequency**: Every simulation step (5 minutes)

## Analysis and Visualization

### Modular Analysis Structure

The simulation provides separate analysis cells for each microgrid and a composite system view:

#### Individual Microgrid Analysis

Each microgrid has its own dedicated analysis cell with:

- **Computing Load**: Real-time power consumption
- **Battery State of Charge**: Storage utilization over time
- **Solar vs Computing Load**: Renewable energy matching
- **Grid Power**: Grid dependency and energy exchange
- **Summary Statistics**: Key performance metrics

#### Composite System Analysis

Comprehensive system-wide analysis including:

- **Load Distribution**: Computing load across all microgrids
- **Battery Utilization Comparison**: Storage performance across locations
- **Grid Dependency Comparison**: Grid power consumption patterns
- **Renewable Energy Usage**: Percentage comparison across microgrids
- **Load Balancing Effectiveness**: Controller performance metrics

### Key Metrics

#### Performance Indicators:

- **Solar-to-Load Ratio**: How well solar generation matches computing demand
- **Grid Dependency**: Energy drawn from the grid (kWh)
- **Renewable Energy Export**: Excess energy fed back to grid (kWh)
- **Battery Utilization**: Dynamic usage patterns (standard deviation)
- **Load Variability**: Coefficient of variation in computing loads
- **Peak Solar Utilization**: Maximum solar power relative to average load

#### System-wide Metrics:

- **Total Grid Energy Drawn**: Combined grid dependency across all microgrids
- **Total Renewable Energy Exported**: Combined renewable energy export
- **Average Solar-to-Load Ratio**: System-wide renewable energy utilization
- **Net Grid Dependency**: Overall grid reliance after renewable exports

## Simulation Outputs

### Data Files

The simulation generates three CSV files in the `results/` directory:

1. **`berlin-microgrid.csv`**: Berlin microgrid data
2. **`san_francisco-microgrid.csv`**: San Francisco microgrid data
3. **`sydney-microgrid.csv`**: Sydney microgrid data

### Data Columns

Each CSV file contains:

- **Timestamp**: Simulation time (datetime)
- **p_delta**: Grid power consumption/production (W)
- **e_delta**: Energy delta (Wh)
- **storage.soc**: Battery state of charge
- **Solar columns**: Individual solar panel outputs (berlin_solar_1, berlin_solar_2, etc.)
- **Server columns**: Individual computing node loads (berlin_server_1, berlin_server_2, etc.)

## Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

### Running the Simulation

```bash
python multi-microgrid.py
```

### Expected Output

- Console output showing simulation progress
- CSV files with detailed results in `results/` directory
- Interactive plots for each microgrid (if running in Jupyter)
- Comprehensive system analysis and comparison charts
- Detailed load balancing effectiveness metrics

### Jupyter Notebook Structure

The simulation is organized into modular cells:

1. **Setup and Configuration**: Environment and microgrid setup
2. **Simulation Execution**: 48-hour simulation run
3. **Berlin Microgrid Analysis**: Individual Berlin performance
4. **San Francisco Microgrid Analysis**: Individual SF performance
5. **Sydney Microgrid Analysis**: Individual Sydney performance
6. **Composite System Analysis**: System-wide performance and metrics

## Extensibility

The system is designed for easy extension:

- **Add New Microgrids**: Follow the existing pattern with location-specific solar data
- **Modify Controller Logic**: Implement different optimization strategies
- **Integrate Additional Sources**: Add wind, hydro, or other renewable sources
- **Implement Advanced Storage**: Replace SimpleBattery with more sophisticated models
- **Add Real-time APIs**: Integrate with actual power management systems

## Future Enhancements

- **Predictive Control**: Machine learning-based load forecasting
- **Real-time Integration**: API connections to actual microgrid systems
- **Advanced Storage Models**: More sophisticated battery and storage modeling
- **Weather Integration**: Real-time weather data for improved solar forecasting
- **Carbon Intensity**: Integration with grid carbon intensity data for carbon-aware computing
