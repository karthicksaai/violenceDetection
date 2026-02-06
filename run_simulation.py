import time
import os
import sys
from orchestration.engine import OrchestrationEngine

# Ensure we can import modules
sys.path.append(os.getcwd())

def run_simulation():
    print("🚀 Initializing Event-Driven Surveillance System...")
    
    # 1. Initialize Engine
    engine = OrchestrationEngine('orchestration/cameras_network.json')
    print(f"✅ Loaded Camera Network: {len(engine.cameras)} sensors configured.")
    print("   Nodes:", list(engine.cameras.keys()))
    print("   Topology:", engine.graph)
    print("-" * 50)

    # 2. Simulate Normal State
    print("\n🕐 T=0: Normal Monitoring. No active events.")
    print(f"   Active Feeds: {engine.get_active_feeds()}")
    time.sleep(1)

    # 3. Trigger VIOLENCE on Camera 1
    print("\n🚨 T=2: VIOLENCE DETECTED at [cam-01] (Zone: Entrance_Gate)")
    print("   -> Simulating Detection Alert...")
    
    # Simulate the event coming from the CV model
    engine.handle_detection(camera_id='cam-01', label='Violence', confidence=0.88)
    
    print(f"\n   Active Feeds: {engine.get_active_feeds()}")
    print("   (Note how cam-01 is now active naturally)")

    # 4. Watch Orchestration happen
    print("\n🤖 T=2.1: ORCHESTRATION ENGINE REACTION")
    # The engine should have automatically looked up neighbors of cam-01
    # Graph: cam-01 -> cam-02
    
    active_feeds = engine.get_active_feeds()
    if 'cam-02' in active_feeds:
        print("   ✅ SUCCESS: System predicted path and activated [cam-02]!")
    else:
        print("   ❌ FAIL: System did not activate downstream camera.")

    print(f"   Current Active Cluster: {active_feeds}")

    # 5. Keep running to show "Live" status
    try:
        print("\n🎥 System Live. Press Ctrl+C to stop.")
        while True:
            # Here we would run the CV loop on active cameras
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped.")

if __name__ == "__main__":
    run_simulation()
