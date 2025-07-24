# %% [markdown]
"""
# Vessim Multi-Microgrid Simulation
A test environment with three microgrids across different locations:
- Berlin, Germany
- San Francisco, USA  
- Sydney, Australia

Each microgrid contains:
- Computing system actors with adjustable load
- Multiple storage nodes (SimpleBattery)
- Solar power generators using solcast2022_global data
- Load balancing controller for optimal energy distribution
"""

# %%
import vessim as vs
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List
import numpy as np
import nest_asyncio
nest_asyncio.apply()


# %%
class MultiMicrogridLoadBalancer(vs.Controller):
    """
    Load balancing controller that distributes computing load across microgrids
    based on solar availability and stored energy levels.
    """

    def __init__(self, microgrids: list, computing_nodes: Dict[str, List]):
        super().__init__(microgrids)
        self.computing_nodes = computing_nodes
        self.base_loads = {}

        # Store original power consumption for each computing node
        for mg_name, nodes in computing_nodes.items():
            self.base_loads[mg_name] = {}
            for node in nodes:
                # Get the current power consumption from the actor's signal
                self.base_loads[mg_name][node.name] = abs(node.signal.now())

    def step(self, time, microgrid_states: dict) -> None:
        """
        Balance load across microgrids based on:
        1. Solar power availability
        2. Battery state of charge
        3. Grid power consumption
        """

        # Calculate availability score for each microgrid
        availability_scores = {}
        
        for mg_name, mg_state in microgrid_states.items():
            # Get solar power from actors
            solar_power = 0
            for actor_name, actor_data in mg_state["state"].items():
                if 'solar' in actor_name.lower():
                    solar_power += actor_data.get('p', 0)

            # Get battery state of charge from storage
            battery_soc = 0
            if 'storage' in mg_state["state"]:
                battery_soc = mg_state["state"]["storage"].get('soc', 0)

            # Calculate availability score (higher is better)
            # Solar power contributes 70%, battery SOC contributes 30%
            availability_score = (solar_power / 5000) * 0.7 + battery_soc * 0.3
            availability_scores[mg_name] = max(0, availability_score)

        # Normalize scores
        total_score = sum(availability_scores.values())
        if total_score > 0:
            normalized_scores = {k: v / total_score for k, v in availability_scores.items()}
        else:
            # If no availability, distribute evenly
            normalized_scores = {k: 1.0 / len(availability_scores) for k in availability_scores.keys()}

        # Adjust computing loads based on availability
        for mg_name, nodes in self.computing_nodes.items():
            target_load_factor = normalized_scores.get(mg_name, 0.33)

            for node in nodes:
                base_load = self.base_loads[mg_name][node.name]
                # Scale load based on availability (0.2 min load to 1.5x base max load)
                new_load = base_load * (0.2 + 1.3 * target_load_factor)
                # Update the actor's signal with the new load (negative for consumption)
                node.signal.set_value(-new_load)


# %%
# Create the simulation environment
environment = vs.Environment(sim_start="2022-06-15 00:00:00")

# Define computing nodes for each microgrid
computing_nodes = {
    "berlin": [
        vs.Actor(name="berlin_server_1", signal=vs.ConstantSignal(-400)),
        vs.Actor(name="berlin_server_2", signal=vs.ConstantSignal(-300)),
        vs.Actor(name="berlin_server_3", signal=vs.ConstantSignal(-200))
    ],
    "san_francisco": [
        vs.Actor(name="san_francisco_server_1", signal=vs.ConstantSignal(-500)),
        vs.Actor(name="san_francisco_server_2", signal=vs.ConstantSignal(-400)),
        vs.Actor(name="san_francisco_server_3", signal=vs.ConstantSignal(-300)),
        vs.Actor(name="san_francisco_server_4", signal=vs.ConstantSignal(-200))
    ],
    "sydney": [
        vs.Actor(name="sydney_server_1", signal=vs.ConstantSignal(-350)),
        vs.Actor(name="sydney_server_2", signal=vs.ConstantSignal(-250)),
        vs.Actor(name="sydney_server_3", signal=vs.ConstantSignal(-150))
    ]
}

# Create microgrids
berlin = environment.add_microgrid(
    actors=[
        *computing_nodes["berlin"],
        vs.Actor(name="berlin_solar_1", signal=vs.Trace.load("solcast2022_global", column="Berlin", params={"scale": 3000})),
        vs.Actor(name="berlin_solar_2", signal=vs.Trace.load("solcast2022_global", column="Berlin", params={"scale": 2500})),
        vs.Actor(name="berlin_solar_3", signal=vs.Trace.load("solcast2022_global", column="Berlin", params={"scale": 2000}))
    ],
    storage=vs.SimpleBattery(capacity=3500, initial_soc=0.65, min_soc=0.1),
    name="berlin"
)

san_francisco = environment.add_microgrid(
    actors=[
        *computing_nodes["san_francisco"],
        vs.Actor(name="san_francisco_solar_1", signal=vs.Trace.load("solcast2022_global", column="San Francisco", params={"scale": 3500})),
        vs.Actor(name="san_francisco_solar_2", signal=vs.Trace.load("solcast2022_global", column="San Francisco", params={"scale": 3000})),
        vs.Actor(name="san_francisco_solar_3", signal=vs.Trace.load("solcast2022_global", column="San Francisco", params={"scale": 2500}))
    ],
    storage=vs.SimpleBattery(capacity=4500, initial_soc=0.75, min_soc=0.15),
    name="san_francisco"
)

sydney = environment.add_microgrid(
    actors=[
        *computing_nodes["sydney"],
        vs.Actor(name="sydney_solar_1", signal=vs.Trace.load("solcast2022_global", column="Sydney", params={"scale": 4000})),
        vs.Actor(name="sydney_solar_2", signal=vs.Trace.load("solcast2022_global", column="Sydney", params={"scale": 3500})),
        vs.Actor(name="sydney_solar_3", signal=vs.Trace.load("solcast2022_global", column="Sydney", params={"scale": 3000}))
    ],
    storage=vs.SimpleBattery(capacity=5500, initial_soc=0.7, min_soc=0.1),
    name="sydney"
)

# Create controllers
monitor = vs.Monitor([berlin, san_francisco, sydney], outdir="results")
load_balancer = MultiMicrogridLoadBalancer([berlin, san_francisco, sydney], computing_nodes)

# Add controllers to environment
environment.add_controller(monitor)
environment.add_controller(load_balancer)

# %%
# Run the simulation for 48 hours to see patterns across different time zones
print("Starting multi-microgrid simulation...")
environment.run(until=48 * 3600)  # 48 hours
print("Simulation completed!")


# %%
# Load and analyze results
def load_microgrid_results(microgrid_name: str) -> pd.DataFrame:
    """Load results for a specific microgrid"""
    filename = f"results/{microgrid_name}-microgrid.csv"
    df = pd.read_csv(filename, parse_dates=[0], index_col=0)
    return df


# Load all results
berlin_df = load_microgrid_results("berlin")
sf_df = load_microgrid_results("san_francisco") 
sydney_df = load_microgrid_results("sydney")

print("Berlin Microgrid Results:")
print(berlin_df.head())
print("\nSan Francisco Microgrid Results:")
print(sf_df.head())
print("\nSydney Microgrid Results:")
print(sydney_df.head())

# %%
# Berlin Microgrid Analysis
print("=== Berlin Microgrid Analysis ===")

# Calculate Berlin-specific metrics
berlin_computing_total = berlin_df.filter(like='server').sum(axis=1)
berlin_solar_total = berlin_df.filter(like='solar').sum(axis=1)
berlin_effectiveness = berlin_solar_total / (-berlin_computing_total)

# Create Berlin-specific visualization
fig_berlin = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Berlin Computing Load', 'Berlin Battery State of Charge',
        'Berlin Solar vs Computing Load', 'Berlin Grid Power'
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# 1. Computing Load
fig_berlin.add_trace(
    go.Scatter(x=berlin_df.index, y=-berlin_computing_total,
               name="Computing Load", line=dict(color="blue", width=2)),
    row=1, col=1
)

# 2. Battery State of Charge
fig_berlin.add_trace(
    go.Scatter(x=berlin_df.index, y=berlin_df['storage.soc'],
               name="Battery SOC", line=dict(color="green")),
    row=1, col=2
)

# 3. Solar vs Computing Load
fig_berlin.add_trace(
    go.Scatter(x=berlin_df.index, y=berlin_solar_total,
               name="Solar Generation", line=dict(color="orange")),
    row=2, col=1
)
fig_berlin.add_trace(
    go.Scatter(x=berlin_df.index, y=-berlin_computing_total,
               name="Computing Load", line=dict(color="blue")),
    row=2, col=1
)

# 4. Grid Power
fig_berlin.add_trace(
    go.Scatter(x=berlin_df.index, y=berlin_df['p_delta'],
               name="Grid Power", line=dict(color="red"), fill='tonexty'),
    row=2, col=2
)

fig_berlin.update_layout(
    title="Berlin Microgrid Performance Analysis",
    height=600,
    showlegend=True
)

fig_berlin.update_yaxes(title_text="Power (W)", row=1, col=1)
fig_berlin.update_yaxes(title_text="State of Charge", row=1, col=2)
fig_berlin.update_yaxes(title_text="Power (W)", row=2, col=1)
fig_berlin.update_yaxes(title_text="Grid Power (W)", row=2, col=2)

fig_berlin.show()

# Berlin summary statistics
print(f"Berlin Average Computing Load: {-berlin_computing_total.mean():.1f} W")
print(f"Berlin Average Solar Generation: {berlin_solar_total.mean():.1f} W")
print(f"Berlin Solar-to-Load Ratio: {berlin_effectiveness.mean():.2f}")
print(f"Berlin Battery Utilization (std): {berlin_df['storage.soc'].std():.3f}")

# %%
# San Francisco Microgrid Analysis
print("=== San Francisco Microgrid Analysis ===")

# Calculate San Francisco-specific metrics
sf_computing_total = sf_df.filter(like='server').sum(axis=1)
sf_solar_total = sf_df.filter(like='solar').sum(axis=1)
sf_effectiveness = sf_solar_total / (-sf_computing_total)

# Create San Francisco-specific visualization
fig_sf = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'San Francisco Computing Load', 'San Francisco Battery State of Charge',
        'San Francisco Solar vs Computing Load', 'San Francisco Grid Power'
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# 1. Computing Load
fig_sf.add_trace(
    go.Scatter(x=sf_df.index, y=-sf_computing_total,
               name="Computing Load", line=dict(color="red", width=2)),
    row=1, col=1
)

# 2. Battery State of Charge
fig_sf.add_trace(
    go.Scatter(x=sf_df.index, y=sf_df['storage.soc'],
               name="Battery SOC", line=dict(color="green")),
    row=1, col=2
)

# 3. Solar vs Computing Load
fig_sf.add_trace(
    go.Scatter(x=sf_df.index, y=sf_solar_total,
               name="Solar Generation", line=dict(color="orange")),
    row=2, col=1
)
fig_sf.add_trace(
    go.Scatter(x=sf_df.index, y=-sf_computing_total,
               name="Computing Load", line=dict(color="red")),
    row=2, col=1
)

# 4. Grid Power
fig_sf.add_trace(
    go.Scatter(x=sf_df.index, y=sf_df['p_delta'],
               name="Grid Power", line=dict(color="purple"), fill='tonexty'),
    row=2, col=2
)

fig_sf.update_layout(
    title="San Francisco Microgrid Performance Analysis",
    height=600,
    showlegend=True
)

fig_sf.update_yaxes(title_text="Power (W)", row=1, col=1)
fig_sf.update_yaxes(title_text="State of Charge", row=1, col=2)
fig_sf.update_yaxes(title_text="Power (W)", row=2, col=1)
fig_sf.update_yaxes(title_text="Grid Power (W)", row=2, col=2)

fig_sf.show()

# San Francisco summary statistics
print(f"San Francisco Average Computing Load: {-sf_computing_total.mean():.1f} W")
print(f"San Francisco Average Solar Generation: {sf_solar_total.mean():.1f} W")
print(f"San Francisco Solar-to-Load Ratio: {sf_effectiveness.mean():.2f}")
print(f"San Francisco Battery Utilization (std): {sf_df['storage.soc'].std():.3f}")

# %%
# Sydney Microgrid Analysis
print("=== Sydney Microgrid Analysis ===")

# Calculate Sydney-specific metrics
sydney_computing_total = sydney_df.filter(like='server').sum(axis=1)
sydney_solar_total = sydney_df.filter(like='solar').sum(axis=1)
sydney_effectiveness = sydney_solar_total / (-sydney_computing_total)

# Create Sydney-specific visualization
fig_sydney = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Sydney Computing Load', 'Sydney Battery State of Charge',
        'Sydney Solar vs Computing Load', 'Sydney Grid Power'
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# 1. Computing Load
fig_sydney.add_trace(
    go.Scatter(x=sydney_df.index, y=-sydney_computing_total,
               name="Computing Load", line=dict(color="green", width=2)),
    row=1, col=1
)

# 2. Battery State of Charge
fig_sydney.add_trace(
    go.Scatter(x=sydney_df.index, y=sydney_df['storage.soc'],
               name="Battery SOC", line=dict(color="darkgreen")),
    row=1, col=2
)

# 3. Solar vs Computing Load
fig_sydney.add_trace(
    go.Scatter(x=sydney_df.index, y=sydney_solar_total,
               name="Solar Generation", line=dict(color="orange")),
    row=2, col=1
)
fig_sydney.add_trace(
    go.Scatter(x=sydney_df.index, y=-sydney_computing_total,
               name="Computing Load", line=dict(color="green")),
    row=2, col=1
)

# 4. Grid Power
fig_sydney.add_trace(
    go.Scatter(x=sydney_df.index, y=sydney_df['p_delta'],
               name="Grid Power", line=dict(color="brown"), fill='tonexty'),
    row=2, col=2
)

fig_sydney.update_layout(
    title="Sydney Microgrid Performance Analysis",
    height=600,
    showlegend=True
)

fig_sydney.update_yaxes(title_text="Power (W)", row=1, col=1)
fig_sydney.update_yaxes(title_text="State of Charge", row=1, col=2)
fig_sydney.update_yaxes(title_text="Power (W)", row=2, col=1)
fig_sydney.update_yaxes(title_text="Grid Power (W)", row=2, col=2)

fig_sydney.show()

# Sydney summary statistics
print(f"Sydney Average Computing Load: {-sydney_computing_total.mean():.1f} W")
print(f"Sydney Average Solar Generation: {sydney_solar_total.mean():.1f} W")
print(f"Sydney Solar-to-Load Ratio: {sydney_effectiveness.mean():.2f}")
print(f"Sydney Battery Utilization (std): {sydney_df['storage.soc'].std():.3f}")

# %%
# Composite Multi-Microgrid System Analysis
print("=== Composite Multi-Microgrid System Analysis ===")


# Calculate composite metrics
def calculate_summary_stats(df: pd.DataFrame, location: str) -> Dict:
    """Calculate summary statistics for a microgrid"""
    solar_columns = [col for col in df.columns if 'solar' in col.lower()]
    total_solar = df[solar_columns].sum(axis=1) if solar_columns else pd.Series(0,
                                                                                index=df.index)

    return {
        "location": location,
        "total_solar_energy": total_solar.sum() * 300 / 3600,
        # Convert to kWh (300s steps)
        "avg_solar_power": total_solar.mean(),
        "max_solar_power": total_solar.max(),
        "total_grid_energy": df['p_delta'].sum() * 300 / 3600,  # Convert to kWh
        "avg_grid_power": df['p_delta'].mean(),
        "renewable_percentage": (total_solar.sum() / (
                    total_solar.sum() - df['p_delta'].sum())) * 100 if (
                                                                                   total_solar.sum() -
                                                                                   df[
                                                                                       'p_delta'].sum()) > 0 else 0
    }


# Calculate statistics for all microgrids
stats = [
    calculate_summary_stats(berlin_df, "Berlin"),
    calculate_summary_stats(sf_df, "San Francisco"),
    calculate_summary_stats(sydney_df, "Sydney")
]

# Display summary table
summary_df = pd.DataFrame(stats)
print("\n=== Multi-Microgrid Simulation Summary ===")
print(summary_df.round(2))

# Create composite visualization
fig_composite = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        'Load Distribution Across Microgrids', 'Battery Utilization Comparison',
        'Grid Dependency Comparison', 'Renewable Energy Usage'
    ),
    specs=[[{"secondary_y": False}, {"secondary_y": False}],
           [{"secondary_y": False}, {"secondary_y": False}]]
)

# 1. Load Distribution
fig_composite.add_trace(
    go.Scatter(x=berlin_df.index, y=-berlin_computing_total,
               name="Berlin", line=dict(color="blue", width=2)),
    row=1, col=1
)
fig_composite.add_trace(
    go.Scatter(x=sf_df.index, y=-sf_computing_total,
               name="San Francisco", line=dict(color="red", width=2)),
    row=1, col=1
)
fig_composite.add_trace(
    go.Scatter(x=sydney_df.index, y=-sydney_computing_total,
               name="Sydney", line=dict(color="green", width=2)),
    row=1, col=1
)

# 2. Battery Utilization
fig_composite.add_trace(
    go.Scatter(x=berlin_df.index, y=berlin_df['storage.soc'],
               name="Berlin Battery", line=dict(color="blue")),
    row=1, col=2
)
fig_composite.add_trace(
    go.Scatter(x=sf_df.index, y=sf_df['storage.soc'],
               name="SF Battery", line=dict(color="red")),
    row=1, col=2
)
fig_composite.add_trace(
    go.Scatter(x=sydney_df.index, y=sydney_df['storage.soc'],
               name="Sydney Battery", line=dict(color="green")),
    row=1, col=2
)

# 3. Grid Dependency
fig_composite.add_trace(
    go.Scatter(x=berlin_df.index, y=berlin_df['p_delta'],
               name="Berlin Grid", line=dict(color="blue"), fill='tonexty'),
    row=2, col=1
)
fig_composite.add_trace(
    go.Scatter(x=sf_df.index, y=sf_df['p_delta'],
               name="SF Grid", line=dict(color="red"), fill='tonexty'),
    row=2, col=1
)
fig_composite.add_trace(
    go.Scatter(x=sydney_df.index, y=sydney_df['p_delta'],
               name="Sydney Grid", line=dict(color="green"), fill='tonexty'),
    row=2, col=1
)

# 4. Renewable Energy Usage Comparison
locations = ["Berlin", "San Francisco", "Sydney"]
renewable_percentages = [stats[i]["renewable_percentage"] for i in range(3)]

fig_composite.add_trace(go.Bar(
    x=locations,
    y=renewable_percentages,
    text=[f"{p:.1f}%" for p in renewable_percentages],
    textposition='auto',
    marker_color=['#1f77b4', '#ff7f0e', '#2ca02c'],
    name="Renewable %"
), row=2, col=2)

fig_composite.update_layout(
    title="Composite Multi-Microgrid System Performance",
    height=800,
    showlegend=True
)

fig_composite.update_yaxes(title_text="Computing Load (W)", row=1, col=1)
fig_composite.update_yaxes(title_text="Battery SOC", row=1, col=2)
fig_composite.update_yaxes(title_text="Grid Power (W)", row=2, col=1)
fig_composite.update_yaxes(title_text="Renewable Energy %", row=2, col=2)

fig_composite.show()


# Calculate and display load balancing effectiveness metrics
def calculate_load_balancing_metrics(berlin_df: pd.DataFrame, sf_df: pd.DataFrame,
                                     sydney_df: pd.DataFrame):
    """Calculate metrics that demonstrate load balancing effectiveness"""

    # Calculate solar generation totals
    berlin_solar = berlin_df.filter(like='solar').sum(axis=1)
    sf_solar = sf_df.filter(like='solar').sum(axis=1)
    sydney_solar = sydney_df.filter(like='solar').sum(axis=1)

    # Calculate computing loads
    berlin_load = -(berlin_df.filter(like='server').sum(axis=1))
    sf_load = -(sf_df.filter(like='server').sum(axis=1))
    sydney_load = -(sydney_df.filter(like='server').sum(axis=1))

    # Calculate grid power (negative = drawing from grid)
    berlin_grid = berlin_df['p_delta']
    sf_grid = sf_df['p_delta']
    sydney_grid = sydney_df['p_delta']

    metrics = {
        "Berlin": {
            "avg_computing_load": berlin_load.mean(),
            "avg_solar_generation": berlin_solar.mean(),
            "solar_load_ratio": berlin_solar.mean() / berlin_load.mean(),
            "grid_dependency": (berlin_grid[berlin_grid < 0].sum() * 300 / 3600),
            # kWh drawn from grid
            "renewable_energy_exported": (
                        berlin_grid[berlin_grid > 0].sum() * 300 / 3600),  # kWh exported
            "battery_utilization": berlin_df['storage.soc'].std(),
            # Higher std = more dynamic usage
            "peak_solar_utilization": berlin_solar.max() / berlin_load.mean(),
            "load_variability": berlin_load.std() / berlin_load.mean()
            # Coefficient of variation
        },
        "San Francisco": {
            "avg_computing_load": sf_load.mean(),
            "avg_solar_generation": sf_solar.mean(),
            "solar_load_ratio": sf_solar.mean() / sf_load.mean(),
            "grid_dependency": (sf_grid[sf_grid < 0].sum() * 300 / 3600),
            "renewable_energy_exported": (sf_grid[sf_grid > 0].sum() * 300 / 3600),
            "battery_utilization": sf_df['storage.soc'].std(),
            "peak_solar_utilization": sf_solar.max() / sf_load.mean(),
            "load_variability": sf_load.std() / sf_load.mean()
        },
        "Sydney": {
            "avg_computing_load": sydney_load.mean(),
            "avg_solar_generation": sydney_solar.mean(),
            "solar_load_ratio": sydney_solar.mean() / sydney_load.mean(),
            "grid_dependency": (sydney_grid[sydney_grid < 0].sum() * 300 / 3600),
            "renewable_energy_exported": (
                        sydney_grid[sydney_grid > 0].sum() * 300 / 3600),
            "battery_utilization": sydney_df['storage.soc'].std(),
            "peak_solar_utilization": sydney_solar.max() / sydney_load.mean(),
            "load_variability": sydney_load.std() / sydney_load.mean()
        }
    }

    return metrics


# Calculate and display metrics
load_balancing_metrics = calculate_load_balancing_metrics(berlin_df, sf_df, sydney_df)

print("\n=== Load Balancing Controller Effectiveness Metrics ===")
print("=" * 80)

for location, metrics in load_balancing_metrics.items():
    print(f"\n{location} Microgrid:")
    print(f"  • Average Computing Load: {metrics['avg_computing_load']:.1f} W")
    print(f"  • Average Solar Generation: {metrics['avg_solar_generation']:.1f} W")
    print(f"  • Solar-to-Load Ratio: {metrics['solar_load_ratio']:.2f}")
    print(f"  • Grid Energy Drawn: {metrics['grid_dependency']:.1f} kWh")
    print(
        f"  • Renewable Energy Exported: {metrics['renewable_energy_exported']:.1f} kWh")
    print(f"  • Battery Utilization (std): {metrics['battery_utilization']:.3f}")
    print(f"  • Peak Solar Utilization: {metrics['peak_solar_utilization']:.2f}")
    print(f"  • Load Variability: {metrics['load_variability']:.3f}")

# Calculate overall system metrics
total_grid_dependency = sum(m['grid_dependency'] for m in load_balancing_metrics.values())
total_renewable_export = sum(
    m['renewable_energy_exported'] for m in load_balancing_metrics.values())
avg_solar_ratio = np.mean(
    [m['solar_load_ratio'] for m in load_balancing_metrics.values()])

print(f"\n=== Overall System Performance ===")
print(f"Total Grid Energy Drawn: {total_grid_dependency:.1f} kWh")
print(f"Total Renewable Energy Exported: {total_renewable_export:.1f} kWh")
print(f"Average Solar-to-Load Ratio: {avg_solar_ratio:.2f}")
print(f"Net Grid Dependency: {total_grid_dependency - total_renewable_export:.1f} kWh")

print(f"\nSimulation completed successfully!")
print(
    f"Results saved to: results/berlin-microgrid.csv, results/san_francisco-microgrid.csv, results/sydney-microgrid.csv")
