"""
Bus Overcrowding Prediction System
Data Simulator — Synthetic Dataset Generator
Reva University Mini Project | ISE Dept.

10 real BMTC Bengaluru routes with actual stop names.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

NUM_TRIPS    = 15000
BUS_CAPACITY = 60
OUTPUT_FILE  = "bus_occupancy_data.csv"

ROUTES = {
    "500D": {
        "name": "Hebbal to Silk Board",
        "stops": [
            "Hebbala (Canara Bank)", "Manyata Tech Park", "Veerannapalya",
            "Kalyan Nagar", "C V Raman Nagar", "Tin Factory",
            "K. R. Puram Rlwy Station", "Marathahalli Bridge (ORR)",
            "Kadubeesanahalli", "Bellandur", "Iblur",
            "Central Silk Board (BTM)", "Central Silk Board (Hosur RD)"
        ],
        "base_demand": 1.4,
    },
    "335E": {
        "name": "Majestic to Whitefield",
        "stops": [
            "Majestic (KBS)", "Corporation", "Domlur",
            "Indiranagar 100 FT RD (6th Mai)", "Marathahalli (Jn. Vartur and O)",
            "Kundalahalli", "ITPL", "BSNL Office (Whitefield RD)"
        ],
        "base_demand": 1.2,
    },
    "401": {
        "name": "Kempegowda Bus Station to Yelahanka",
        "stops": [
            "Majestic (KBS)", "Mehkri Circle", "Hebbala (Canara Bank)",
            "Kogilu Cross", "Yelahanka Old Town", "Yelahanka"
        ],
        "base_demand": 0.9,
    },
    "250D": {
        "name": "Banashankari to Rajajinagar",
        "stops": [
            "Banashankari Bus Station", "Jayanagar Water Tank",
            "South End Circle", "Lalbagh West Gate",
            "K. R. Market", "City Market", "Rajajinagar/Navaranga"
        ],
        "base_demand": 1.0,
    },
    "G5": {
        "name": "Kempegowda Bus Station to Sarjapur",
        "stops": [
            "Majestic (KBS)", "K. R. Market", "Lalbagh West Gate",
            "Koramangala 1st Block", "Koramangala (SONY World)",
            "Ejipura", "Kaggadasapura Jn (Big Bazaar)",
            "Sarjapur Road", "Sarjapur"
        ],
        "base_demand": 0.85,
    },
    "201R": {
        "name": "Hebbal to Electronic City",
        "stops": [
            "Hebbala (Canara Bank)", "Manyata Tech Park",
            "K. R. Puram Rlwy Station", "Marathahalli Bridge (ORR)",
            "Central Silk Board (ORR)", "Electronic City (Hosur Rd)",
            "Electronic City"
        ],
        "base_demand": 1.3,
    },
    "356F": {
        "name": "Yeshwanthpur to BTM Layout",
        "stops": [
            "R M C (Yeshwanthpura)", "Rajajinagar/Navaranga",
            "Bhashyam Circle (Rajajinagar)", "Majestic (KBS)",
            "South End Circle", "BTM Check Post",
            "Kuvempu Nagar (BTM Layout)", "HSR Layout"
        ],
        "base_demand": 1.1,
    },
    "600K": {
        "name": "Koramangala to Indiranagar",
        "stops": [
            "Koramangala (80 FT RD)", "Koramangala (SONY World)",
            "Ejipura", "Domlur",
            "Indiranagar 100 FT RD (6th Mai)",
            "Indiranagar 100 FT RD (13th Ma)", "Binnamangala"
        ],
        "base_demand": 1.15,
    },
    "V1": {
        "name": "Banashankari to Marathahalli",
        "stops": [
            "Banashankari Bus Station", "Jayanagar Water Tank",
            "Central Silk Board (BTM)", "HSR Layout (BDA Complex)",
            "HSR Layout", "Bellandur",
            "Marathahalli (Jn. Vartur and O)", "Marathahalli Bridge (ORR)"
        ],
        "base_demand": 1.2,
    },
    "C9": {
        "name": "Majestic to HSR Layout",
        "stops": [
            "Majestic (KBS)", "Lalbagh West Gate", "K. R. Market",
            "Koramangala 1st Block", "Koramangala (80 FT RD)",
            "Central Silk Board (BTM)", "HSR Layout (BDA Complex)",
            "HSR (Depot-25)"
        ],
        "base_demand": 1.1,
    },
}

STOP_POPULARITY = {
    "Majestic (KBS)": 3.0,
    "Central Silk Board (Hosur RD)": 2.8,
    "Central Silk Board (BTM)": 2.8,
    "Central Silk Board (ORR)": 2.7,
    "Hebbala (Canara Bank)": 2.5,
    "Marathahalli Bridge (ORR)": 2.3,
    "K. R. Puram Rlwy Station": 2.2,
    "Manyata Tech Park": 2.0,
    "ITPL": 2.0,
    "Electronic City (Hosur Rd)": 1.9,
    "Electronic City": 1.9,
    "Indiranagar 100 FT RD (6th Mai)": 1.8,
    "Banashankari Bus Station": 1.7,
    "K. R. Market": 1.6,
    "Koramangala (SONY World)": 1.5,
    "Koramangala 1st Block": 1.5,
    "HSR Layout": 1.4,
    "BTM Check Post": 1.3,
    "Bellandur": 1.3,
    "Kadubeesanahalli": 1.2,
    "Rajajinagar/Navaranga": 1.2,
    "R M C (Yeshwanthpura)": 1.2,
    "Kundalahalli": 1.1,
    "Domlur": 1.1,
    "Jayanagar Water Tank": 1.0,
    "South End Circle": 1.0,
}
DEFAULT_STOP_WEIGHT = 0.7

def get_hour_demand_multiplier(hour, is_weekend):
    if is_weekend:
        profile = {0:0.05,1:0.03,2:0.02,3:0.02,4:0.05,5:0.15,6:0.30,7:0.40,
                   8:0.50,9:0.60,10:0.70,11:0.80,12:0.85,13:0.80,14:0.75,
                   15:0.70,16:0.72,17:0.75,18:0.70,19:0.60,20:0.45,21:0.30,22:0.15,23:0.08}
    else:
        profile = {0:0.04,1:0.02,2:0.02,3:0.03,4:0.08,5:0.20,6:0.45,7:0.75,
                   8:1.00,9:0.90,10:0.55,11:0.45,12:0.60,13:0.55,14:0.50,
                   15:0.65,16:0.80,17:1.00,18:0.95,19:0.75,20:0.50,21:0.30,22:0.15,23:0.07}
    return profile.get(hour, 0.5)

def get_stop_weight(stop_name):
    return STOP_POPULARITY.get(stop_name, DEFAULT_STOP_WEIGHT)

def simulate_passengers(route_id, stop_name, hour, is_weekend, current_occupancy, stop_index, total_stops):
    route = ROUTES[route_id]
    demand = route["base_demand"]
    hour_mult = get_hour_demand_multiplier(hour, is_weekend)
    stop_weight = get_stop_weight(stop_name)
    available = max(0, BUS_CAPACITY - current_occupancy)
    base_board = demand * hour_mult * stop_weight * BUS_CAPACITY * 0.25
    passengers_in = int(np.random.poisson(max(0, base_board)))
    passengers_in = min(passengers_in, available)
    if stop_index == 0:
        passengers_in = min(int(passengers_in * 1.5), BUS_CAPACITY)
    progress = stop_index / max(total_stops - 1, 1)
    alight_rate = 0.15 + (progress * 0.35)
    passengers_out = int(np.random.binomial(
        n=max(0, current_occupancy),
        p=min(alight_rate + np.random.uniform(-0.05, 0.05), 1.0)
    ))
    if stop_index == total_stops - 1:
        passengers_out = current_occupancy + passengers_in
    return passengers_in, passengers_out

def classify_occupancy(occupancy, capacity):
    ratio = occupancy / capacity
    if ratio < 0.30: return "Low"
    elif ratio <= 0.80: return "Medium"
    else: return "High"

def simulate_dataset(num_trips):
    records = []
    route_ids = list(ROUTES.keys())
    start_date = datetime(2024, 1, 1)
    date_range = [start_date + timedelta(days=i) for i in range(90)]
    trips_done = 0
    while trips_done < num_trips:
        route_id = random.choice(route_ids)
        route = ROUTES[route_id]
        stops = route["stops"]
        date = random.choice(date_range)
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5
        day_name = date.strftime("%A")
        hour = random.randint(5, 22)
        if random.random() > get_hour_demand_multiplier(hour, is_weekend) * 0.9:
            continue
        trip_id = f"TRIP_{trips_done + 1:05d}"
        current_occupancy = 0
        for stop_index, stop_name in enumerate(stops):
            passengers_in, passengers_out = simulate_passengers(
                route_id, stop_name, hour, is_weekend,
                current_occupancy, stop_index, len(stops)
            )
            current_occupancy = max(0, current_occupancy + passengers_in - passengers_out)
            occupancy_label = classify_occupancy(current_occupancy, BUS_CAPACITY)
            occupancy_pct = round((current_occupancy / BUS_CAPACITY) * 100, 1)
            hour_sin = round(np.sin(2 * np.pi * hour / 24), 4)
            hour_cos = round(np.cos(2 * np.pi * hour / 24), 4)
            dow_sin = round(np.sin(2 * np.pi * day_of_week / 7), 4)
            dow_cos = round(np.cos(2 * np.pi * day_of_week / 7), 4)
            records.append({
                "trip_id": trip_id, "date": date.strftime("%Y-%m-%d"),
                "route_id": route_id, "route_name": route["name"],
                "stop_name": stop_name, "stop_index": stop_index,
                "total_stops": len(stops), "hour": hour,
                "day_of_week": day_of_week, "day_name": day_name,
                "is_weekend": int(is_weekend),
                "passengers_in": passengers_in, "passengers_out": passengers_out,
                "occupancy_count": current_occupancy, "bus_capacity": BUS_CAPACITY,
                "occupancy_pct": occupancy_pct, "occupancy_label": occupancy_label,
                "stop_popularity": get_stop_weight(stop_name),
                "hour_sin": hour_sin, "hour_cos": hour_cos,
                "dow_sin": dow_sin, "dow_cos": dow_cos,
            })
        trips_done += 1
        if trips_done % 1000 == 0:
            print(f"  Simulated {trips_done}/{num_trips} trips...")
    return pd.DataFrame(records)

def main():
    print("=" * 55)
    print("  Bus Overcrowding Prediction System")
    print(f"  {len(ROUTES)} Real BMTC Bengaluru Routes")
    print("=" * 55)
    print(f"\nGenerating {NUM_TRIPS} synthetic bus trips...")
    df = simulate_dataset(NUM_TRIPS)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset saved: {OUTPUT_FILE}")
    print(f"Total records: {len(df):,}")
    print("\nOccupancy distribution:")
    for label, count in df["occupancy_label"].value_counts().items():
        print(f"  {label}: {count:,} ({count/len(df)*100:.1f}%)")
    print("\nRoutes simulated:")
    for rid, r in ROUTES.items():
        print(f"  {rid}: {r['name']}")
    print("\nNext: run feature_engineering.py")

if __name__ == "__main__":
    main()