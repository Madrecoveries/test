"""
mission_offboard.py
Offboard-control mission controller using MAVSDK-Python.

Requirements:
    pip install mavsdk asyncio numpy opencv-python apriltag-python   # apriltag/vision optional

Behavior:
    - Connects to vehicle (default UDP SITL: udp://:14540)
    - Sets home if not preconfigured
    - Arms & takeoff via high-level Action (to be safe)
    - Starts Offboard and sends velocity setpoints to:
        * fly to TARGET_WAYPOINT (simple P-control on horizontal error)
        * hover
        * return to home using offboard velocity control
        * when near home, attempt precision landing by descending under offboard control
    - If offboard fails or battery becomes critically low, falls back to RTL or LAND via Action
Notes:
    - Offboard requires frequent setpoint streaming (>= 2 Hz). This script streams at 10 Hz.
    - Tune gains, speeds, tolerances to your vehicle and mission.
    - Test thoroughly in PX4 SITL before using on real hardware.
"""

import asyncio
import math
import time
from typing import Optional, Tuple

from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from mavsdk import (Telemetry, Action)

import numpy as np

# ------------- CONFIG -------------
SYSTEM_ADDRESS = "udp://:14540"   # change to serial:///dev/ttyACM0:921600 or udp://<ip>:14550 for real FC
TARGET_WAYPOINT = (-37.810000, 144.960000, 10.0)  # lat, lon, altitude (m) - configure
HOME_COORD: Optional[Tuple[float, float, float]] = None  # set to None to auto-capture on connect
TAKEOFF_ALT = 10.0            # meters (used for Action.takeoff)
CRUISE_SPEED = 4.0           # m/s max commanded horizontal speed
OFFBOARD_STREAM_HZ = 10.0    # Hz to stream offboard setpoints (>= 2 Hz)
GOTO_TOLERANCE_M = 2.5       # consider waypoint reached within this horizontal distance
LOW_BATTERY_THRESHOLD = 20   # percent -> fallback
CRITICAL_BATTERY_THRESHOLD = 12  # percent -> immediate RTL/LAND
RTL_ALTITUDE = 12.0          # RTL altitude set on FC for safe return
MAX_OFFBOARD_DURATION = 300  # seconds; safety timeout for offboard mode
PRECISION_DESCEND_RATE = 0.5 # m/s downward while searching for marker (positive down in VelocityNedYaw)
MIN_LANDING_ALT = 0.5        # if below this, perform FC land
# Horizontal position control gains (simple proportional controller)
KP_HORIZONTAL = 0.8          # higher -> more aggressive
# ----------------------------------

# Dummy vision detector placeholder (replace with camera+AprilTag)
async def vision_marker_detector() -> Tuple[bool, Optional[float], Optional[float], Optional[float]]:
    """
    Should return (found, tx, ty, tz)
    - tx, ty, tz are camera-frame translations in meters (or NED offsets),
      or (lat, lon, alt) if your integration computes GPS from camera observations.
    Placeholder returns not found.
    """
    await asyncio.sleep(0.01)
    return (False, None, None, None)

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

async def wait_for_connection(drone: System, timeout: float = 30.0):
    start = time.time()
    async for state in drone.core.connection_state():
        if state.is_connected:
            print(f"[INFO] Connected to drone (UUID: {state.uuid})")
            return
        if time.time() - start > timeout:
            raise RuntimeError("Connection timeout")

async def get_current_position(drone: System):
    async for pos in drone.telemetry.position():
        return (pos.latitude_deg, pos.longitude_deg, pos.relative_altitude_m)
    return None

async def set_home_if_none(drone: System):
    global HOME_COORD
    if HOME_COORD is not None:
        print("[INFO] Using configured HOME_COORD")
        return HOME_COORD
    print("[INFO] Waiting for global position fix to set HOME...")
    # wait until FC reports global position ok
    async for health in drone.telemetry.health():
        if health.is_global_position_ok:
            break
    pos = await get_current_position(drone)
    if pos is None:
        raise RuntimeError("Unable to read vehicle position to set home")
    # pos is (lat, lon, rel_alt)
    HOME_COORD = (pos[0], pos[1], pos[2])
    print(f"[INFO] Home set to lat={HOME_COORD[0]:.7f}, lon={HOME_COORD[1]:.7f}, rel_alt={HOME_COORD[2]:.2f}")
    return HOME_COORD

async def arm_and_takeoff(drone: System, altitude_m: float):
    print("[INFO] Arming and taking off (Action.takeoff)")
    await drone.action.set_takeoff_altitude(altitude_m)
    await drone.action.arm()
    # give a moment before takeoff
    await asyncio.sleep(0.5)
    await drone.action.takeoff()
    # wait until relative altitude near target
    reached = False
    start = time.time()
    while time.time() - start < 30:
        async for pos in drone.telemetry.position():
            cur_rel = pos.relative_altitude_m
            print(f"[TAKEOFF] relative alt: {cur_rel:.2f} m")
            if cur_rel >= altitude_m * 0.85:
                reached = True
            break
        if reached:
            break
        await asyncio.sleep(0.5)
    if not reached:
        print("[WARN] Takeoff: target altitude not reached within timeout")

async def start_offboard_and_stream(drone: System):
    # To enter Offboard we must send an initial setpoint first.
    print("[INFO] Preparing initial offboard setpoint (zero velocity)")
    try:
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        await drone.offboard.start()
        print("[INFO] Offboard started")
    except OffboardError as e:
        print(f"[ERROR] Failed to start Offboard: {e}")
        raise

async def stop_offboard_safe(drone: System):
    try:
        await drone.offboard.stop()
        print("[INFO] Offboard stopped")
    except OffboardError as e:
        print(f"[WARN] Could not stop offboard cleanly: {e}")

async def check_battery_and_handle(drone: System):
    async for bat in drone.telemetry.battery():
        if bat.remaining_percent is None:
            return
        percent = bat.remaining_percent * 100.0
        if percent <= CRITICAL_BATTERY_THRESHOLD:
            print(f"[CRITICAL] Battery {percent:.1f}% - immediate RTL")
            await drone.action.set_return_to_launch_altitude(RTL_ALTITUDE)
            await drone.action.return_to_launch()
        elif percent <= LOW_BATTERY_THRESHOLD:
            print(f"[WARN] Battery {percent:.1f}% - initiating safe return")
            await drone.action.set_return_to_launch_altitude(RTL_ALTITUDE)
            await drone.action.return_to_launch()
        break

async def offboard_go_to_waypoint(drone: System, target_lat: float, target_lon: float, target_alt_rel: float):
    """
    Proportional controller in local horizontal plane using velocity setpoints.
    target_alt_rel is relative altitude (m) we want to hold during transit.
    """
    print(f"[INFO] Offboard-guided transit to lat={target_lat:.7f}, lon={target_lon:.7f}, alt_rel={target_alt_rel:.2f}")
    start_time = time.time()
    last_stream = 0.0
    timeout = MAX_OFFBOARD_DURATION
    while True:
        # battery check / emergency fallback
        await check_battery_and_handle(drone)

        pos = await get_current_position(drone)
        if pos is None:
            await asyncio.sleep(1.0 / OFFBOARD_STREAM_HZ)
            continue
        lat, lon, cur_alt_rel = pos
        horiz_dist = haversine_m(lat, lon, target_lat, target_lon)
        # compute bearing to target (for yaw) and unit vector in NED local approximation
        # small-angle flat-earth approx: convert lat/lon deltas to meters
        d_lat = (target_lat - lat) * (math.pi/180.0) * 6371000.0
        d_lon = (target_lon - lon) * (math.pi/180.0) * 6371000.0 * math.cos(math.radians(lat))
        # desired velocity from P-controller
        vx_cmd = KP_HORIZONTAL * d_lat
        vy_cmd = KP_HORIZONTAL * d_lon
        # constrain to cruise speed
        speed = math.hypot(vx_cmd, vy_cmd)
        if speed > CRUISE_SPEED:
            scale = CRUISE_SPEED / speed
            vx_cmd *= scale
            vy_cmd *= scale

        # altitude error -> vertical velocity (NED: down positive)
        alt_err = cur_alt_rel - target_alt_rel  # positive if above desired
        # we want to reduce alt_err -> if above desired, command positive down velocity (NED)
        vz_cmd = KP_HORIZONTAL * alt_err
        # limit descent/ascent
        vz_cmd = max(min(vz_cmd, 1.0), -1.0)  # limit to +/-1 m/s mellow rate

        # yaw - face direction of travel (degrees)
        yaw_deg = 0.0
        if horiz_dist > 0.5:
            yaw_deg = math.degrees(math.atan2(vy_cmd, vx_cmd)) if (vx_cmd != 0 or vy_cmd != 0) else 0.0

        # send setpoint at stream rate
        now = time.time()
        if now - last_stream >= 1.0 / OFFBOARD_STREAM_HZ:
            try:
                # VelocityNedYaw: vx (north m/s), vy (east m/s), vz (down m/s), yaw_deg
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx_cmd, vy_cmd, vz_cmd, yaw_deg))
            except OffboardError as e:
                print(f"[ERROR] Offboard setpoint failed: {e} - attempting safe fallback")
                # fallback: stop offboard and trigger RTL
                await stop_offboard_safe(drone)
                await drone.action.set_return_to_launch_altitude(RTL_ALTITUDE)
                await drone.action.return_to_launch()
                return False
            last_stream = now

        print(f"[OFFBOARD] dist={horiz_dist:.1f}m vx={vx_cmd:.2f} vy={vy_cmd:.2f} vz={vz_cmd:.2f} yaw={yaw_deg:.1f}")

        if horiz_dist <= GOTO_TOLERANCE_M:
            print("[INFO] Reached waypoint (within tolerance) - commanding hover (zero velocity).")
            # hover briefly by sending zeros for a short period
            for _ in range(int(OFFBOARD_STREAM_HZ * 2)):
                await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, yaw_deg))
                await asyncio.sleep(1.0 / OFFBOARD_STREAM_HZ)
            return True

        if time.time() - start_time > timeout:
            print("[WARN] Offboard goto timeout - aborting to RTL")
            await stop_offboard_safe(drone)
            await drone.action.set_return_to_launch_altitude(RTL_ALTITUDE)
            await drone.action.return_to_launch()
            return False

        await asyncio.sleep(0.0)  # yield to event loop

async def offboard_precision_land(drone: System, vision_callback):
    """
    Controlled offboard descent that attempts to use vision_callback to center above a marker.
    vision_callback should be an async function that returns (found:bool, tx, ty, tz) or (found, lat, lon, alt)
    The implementation below assumes we get local NED offsets tx (forward/N), ty (right/E), tz (down) in meters.
    Replace transformation logic with your camera-to-vehicle frame math.
    """
    print("[INFO] Starting offboard precision landing routine")
    start_time = time.time()
    last_stream = 0.0
    max_search_time = 120.0
    yaw_deg = 0.0

    while True:
        # battery check
        await check_battery_and_handle(drone)

        pos = await get_current_position(drone)
        if pos is None:
            await asyncio.sleep(1.0 / OFFBOARD_STREAM_HZ)
            continue
        lat, lon, cur_alt_rel = pos

        # call vision detector
        found, tx, ty, tz = await vision_callback()
        if found:
            # tx = forward (north), ty = right (east) relative to vehicle center in meters
            # create small velocity vector to center over marker
            vx_cmd = KP_HORIZONTAL * tx
            vy_cmd = KP_HORIZONTAL * ty
            # limit speeds
            speed = math.hypot(vx_cmd, vy_cmd)
            if speed > 1.5:
                scale = 1.5 / speed
                vx_cmd *= scale
                vy_cmd *= scale
            # descend slowly:
            vz_cmd = PRECISION_DESCEND_RATE  # positive down in NED
            # send command
            try:
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx_cmd, vy_cmd, vz_cmd, yaw_deg))
            except OffboardError as e:
                print(f"[ERROR] Offboard setpoint failed during precision land: {e}")
                await stop_offboard_safe(drone)
                await drone.action.land()
                return False

            print(f"[PRECISION] marker found tx={tx:.2f} ty={ty:.2f} tz={tz:.2f} => vx={vx_cmd:.2f} vy={vy_cmd:.2f} vz={vz_cmd:.2f}")

            # if very low altitude or marker tz indicates ground near, then command FC land
            if cur_alt_rel <= MIN_LANDING_ALT or (tz is not None and abs(tz) <= 0.3):
                print("[INFO] Close to ground - commanding FC LAND for final touchdown")
                await stop_offboard_safe(drone)
                await drone.action.land()
                return True

        else:
            # not found -> descend slowly while sweeping
            vx_cmd = 0.0
            vy_cmd = 0.0
            vz_cmd = PRECISION_DESCEND_RATE  # positive down
            try:
                await drone.offboard.set_velocity_ned(VelocityNedYaw(vx_cmd, vy_cmd, vz_cmd, yaw_deg))
            except OffboardError as e:
                print(f"[ERROR] Offboard setpoint failed during search descend: {e}")
                await stop_offboard_safe(drone)
                await drone.action.land()
                return False
            print(f"[PRECISION] marker NOT found - descending vz={vz_cmd:.2f} m/s alt={cur_alt_rel:.2f}")

            # if too low, give up and land via FC
            if cur_alt_rel <= MIN_LANDING_ALT or (time.time() - start_time) > max_search_time:
                print("[WARN] Marker not found and low/timeout - performing FC LAND")
                await stop_offboard_safe(drone)
                await drone.action.land()
                return False

        await asyncio.sleep(1.0 / OFFBOARD_STREAM_HZ)

async def main():
    drone = System()
    print(f"[INFO] Connecting to {SYSTEM_ADDRESS} ...")
    await drone.connect(system_address=SYSTEM_ADDRESS)

    await wait_for_connection(drone)
    home = await set_home_if_none(drone)

    # set FC settings
    await drone.action.set_return_to_launch_altitude(RTL_ALTITUDE)
    await drone.action.set_maximum_speed(CRUISE_SPEED)

    # Arm and takeoff (high-level action)
    await arm_and_takeoff(drone, TAKEOFF_ALT)

    # Start Offboard
    try:
        await start_offboard_and_stream(drone)
    except Exception as e:
        print(f"[ERROR] Could not start Offboard: {e}. Aborting and switching to RTL.")
        await drone.action.set_return_to_launch_altitude(RTL_ALTITUDE)
        await drone.action.return_to_launch()
        return

    # Fly to target waypoint using Offboard velocity control
    success = await offboard_go_to_waypoint(drone, TARGET_WAYPOINT[0], TARGET_WAYPOINT[1], TARGET_WAYPOINT[2])
    if not success:
        print("[INFO] Offboard transit aborted - awaiting vehicle action to handle fallback (likely RTL).")
        return

    print("[INFO] At target waypoint - hover for 3s")
    for _ in range(int(OFFBOARD_STREAM_HZ * 3)):
        await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        await asyncio.sleep(1.0 / OFFBOARD_STREAM_HZ)

    # Return to home using offboard velocity control: simple P-controller to home coordinates
    print("[INFO] Returning to HOME using Offboard control")
    success = await offboard_go_to_waypoint(drone, home[0], home[1], TAKEOFF_ALT)
    if not success:
        print("[WARN] Offboard return failed or aborted. Exiting.")
        return

    # when close to home and at reasonable low altitude, do precision landing via offboard
    # wait until within horizontal radius and altitude low enough
    print("[INFO] Approaching home - switching to precision landing stage")
    # Keep offboard active and begin precision landing logic
    await offboard_precision_land(drone, vision_marker_detector)

    # Wait for vehicle to report landed
    print("[INFO] Waiting for landed state...")
    async for in_air in drone.telemetry.in_air():
        if not in_air:
            print("[INFO] Landed.")
            break
        await asyncio.sleep(0.5)

    # Disarm if still armed
    try:
        await drone.action.disarm()
    except Exception as e:
        print(f"[INFO] Disarm call returned: {e}")

    print("[INFO] Mission complete. Cleaning up offboard (if active).")
    try:
        await stop_offboard_safe(drone)
    except Exception:
        pass

if __name__ == "__main__":
    asyncio.run(main())
