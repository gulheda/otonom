"""
Servo Test – MAVLink üzerinden servo PWM gönder
================================================
Araç bağlıyken çalıştır, uçuş GEREKMEZ.
"""

from pymavlink import mavutil
import time

# ── Ayarlar ──────────────────────────────────────────────────────────────────
CONNECTION   = "udp:127.0.0.1:14550"
SYSID        = 1
CHANNEL      = 8       # test edilecek servo kanalı
PWM_OPEN     = 1900    # µs – açık konum
PWM_CLOSE    = 1100    # µs – kapalı konum
HOLD_SEC     = 2.0     # açık konumda bekleme süresi


# ── Bağlan ───────────────────────────────────────────────────────────────────
print(f"[Servo] Bağlanılıyor: {CONNECTION}")
vehicle = mavutil.mavlink_connection(CONNECTION, source_system=255, target_system=SYSID)
vehicle.wait_heartbeat(timeout=10)
print(f"[Servo] Heartbeat alındı – SysID:{vehicle.target_system}")


# ── Servo gönder ─────────────────────────────────────────────────────────────
def set_servo(channel, pwm):
    vehicle.mav.command_long_send(
        vehicle.target_system,
        vehicle.target_component,
        mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
        0,
        channel, pwm,
        0, 0, 0, 0, 0
    )
    print(f"[Servo] Kanal {channel} → {pwm} µs")


# ── Test ─────────────────────────────────────────────────────────────────────
input(f"Enter'a bas → kanal {CHANNEL} AÇILACAK ({PWM_OPEN} µs)...")
set_servo(CHANNEL, PWM_OPEN)

time.sleep(HOLD_SEC)

input(f"Enter'a bas → kanal {CHANNEL} KAPANACAK ({PWM_CLOSE} µs)...")
set_servo(CHANNEL, PWM_CLOSE)

print("[Servo] Test tamamlandı.")
