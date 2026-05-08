"""
VTOL İHA – 3 Waypoint Görevi
==============================
Araç  : alti_transition_quad (ArduPlane VTOL)
SysID : 1  |  UDP : 14550
Görev : Kalkış → 3 WP → RTL
"""

import time
import math
import threading
from pymavlink import mavutil


# ═══════════════════════════════════════════════════════════════════════════════
#  AYARLAR – SADECE BURAYA DOKUNUN
# ═══════════════════════════════════════════════════════════════════════════════

VTOL_CONNECTION = "udp:127.0.0.1:14550"
VTOL_SYSID      = 1

TAKEOFF_ALT  = 30.0   # metre
CRUISE_SPEED = 5.0    # m/s
WP_ALT       = 30.0   # metre – waypoint irtifası
WP_ARRIVAL_DIST = 10.0   # metre – bu kadar yaklaşınca "ulaşıldı"
WP_HOVER_TIME   = 3.0    # saniye – waypoint'te bekleme

# ── 3 Hedef Koordinat ────────────────────────────────────────────────────────
# Enlem (lat), Boylam (lon), İrtifa (m) olarak girin.
# Örnek: Kalkış noktasına göre 50m kuzey/doğu/güney.
WAYPOINTS = [
    (-35.3628102, 149.1652074, WP_ALT),   # WP1
    (-35.3632594, 149.1657582, WP_ALT),   # WP2
    (-35.3637086, 149.1652074, WP_ALT),   # WP3
]


# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI
# ═══════════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ═══════════════════════════════════════════════════════════════════════════════
#  VTOL KONTROLCÜSÜ
# ═══════════════════════════════════════════════════════════════════════════════

class VTOLController:

    def __init__(self):
        self.vehicle        = None
        self.mission_active = True

    # ── Bağlantı ─────────────────────────────────────────────────────────────

    def connect(self, retries=5, retry_delay=3):
        for attempt in range(1, retries + 1):
            try:
                print(f"[İHA] MAVLink bağlanılıyor... ({attempt}/{retries})")
                self.vehicle = mavutil.mavlink_connection(
                    VTOL_CONNECTION, source_system=255, target_system=VTOL_SYSID)
                self.vehicle.wait_heartbeat(timeout=10)
                print(f"[İHA] Heartbeat alındı – SysID:{self.vehicle.target_system}")
                return True
            except Exception as exc:
                print(f"[İHA] Bağlantı hatası: {exc}")
                if attempt < retries:
                    time.sleep(retry_delay)
        return False

    # ── MAVLink yardımcıları ──────────────────────────────────────────────────

    def _set_mode(self, mode_name, timeout=15):
        mode_id = self.vehicle.mode_mapping().get(mode_name)
        if mode_id is None:
            print(f"[İHA] Bilinmeyen mod: {mode_name}")
            return False

        def _send():
            self.vehicle.mav.set_mode_send(
                self.vehicle.target_system,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
            self.vehicle.mav.command_long_send(
                self.vehicle.target_system, self.vehicle.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mode_id, 0, 0, 0, 0, 0)

        _send()
        deadline, last_send = time.time() + timeout, time.time()
        while time.time() < deadline:
            if time.time() - last_send >= 3.0:
                _send(); last_send = time.time()
            msg = self.vehicle.recv_match(
                type=["COMMAND_ACK", "HEARTBEAT"], blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.get_type() == "HEARTBEAT":
                if msg.get_srcSystem() != self.vehicle.target_system:
                    continue
                if getattr(msg, "custom_mode", None) == mode_id:
                    print(f"[İHA] Mod → {mode_name}")
                    return True
            elif msg.get_type() == "COMMAND_ACK":
                cmd = getattr(msg, "command", None)
                if cmd not in (mavutil.mavlink.MAV_CMD_DO_SET_MODE, None):
                    continue
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print(f"[İHA] Mod → {mode_name}")
                    return True
        print(f"[İHA] Mod onaylanamadı: {mode_name} – devam ediliyor.")
        return False

    def _wait_prearm(self, timeout=30):
        print("[İHA] Pre-arm bekleniyor...", end="", flush=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.vehicle.recv_match(type="SYS_STATUS", blocking=True, timeout=2)
            if msg is None:
                continue
            hb = self.vehicle.recv_match(type="HEARTBEAT", blocking=False)
            if hb and hb.get_srcSystem() == self.vehicle.target_system:
                if not (hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED):
                    print(".", end="", flush=True)
            time.sleep(0.5)
        print()

    def _arm(self, retries=10, retry_delay=3):
        print("[İHA] ARM ediliyor...")
        for attempt in range(1, retries + 1):
            self.vehicle.mav.command_long_send(
                self.vehicle.target_system, self.vehicle.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0)
            ack = self.vehicle.recv_match(
                type="COMMAND_ACK", blocking=True, timeout=3)
            if ack and ack.command == mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM:
                if ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    self.vehicle.motors_armed_wait()
                    print("[İHA] ARM tamamlandı.")
                    return
                print(f"[İHA] ARM reddedildi (result={ack.result}) – "
                      f"{retry_delay}s sonra tekrar ({attempt}/{retries})")
            else:
                print(f"[İHA] ARM ACK gelmedi – {retry_delay}s sonra ({attempt}/{retries})")
            time.sleep(retry_delay)
        print("[İHA] ARM başarısız – devam ediliyor.")

    def _takeoff(self, altitude):
        print(f"[İHA] TAKEOFF → {altitude} m")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude)
        while True:
            msg = self.vehicle.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
            if msg is None:
                continue
            alt = msg.relative_alt / 1000.0
            print(f"[İHA] İrtifa: {alt:.1f} m / {altitude} m", end="\r")
            if alt >= altitude * 0.95:
                print(f"\n[İHA] {altitude} m irtifasına ulaşıldı.")
                return
            time.sleep(0.5)

    def _get_position(self):
        msg = self.vehicle.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg is None:
            return None, None, None
        return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0

    def _set_speed(self, speed_ms):
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0, 1, speed_ms, -1, 0, 0, 0, 0)

    # ── Mission yükleme ───────────────────────────────────────────────────────

    def _clear_mission(self):
        self.vehicle.mav.mission_clear_all_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
        ack = self.vehicle.recv_match(type="MISSION_ACK", blocking=True, timeout=5)
        if ack:
            print("[İHA] Eski mission temizlendi.")

    def _upload_mission(self, home_lat, home_lon, home_alt, waypoints):
        items = [(0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                  mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                  0, 1, 0.0, 0.0, 0.0, 0.0,
                  int(home_lat * 1e7), int(home_lon * 1e7), float(home_alt))]
        for i, (lat, lon, alt) in enumerate(waypoints):
            items.append((i + 1, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                          mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                          0, 1, 0.0, float(WP_ARRIVAL_DIST), 0.0, float("nan"),
                          int(lat * 1e7), int(lon * 1e7), float(alt)))
        items.append((len(waypoints) + 1, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                      mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                      0, 1, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0))

        total = len(items)
        print(f"[İHA] Mission yükleniyor ({total} item)...")
        self.vehicle.mav.mission_count_send(
            self.vehicle.target_system, self.vehicle.target_component,
            total, mavutil.mavlink.MAV_MISSION_TYPE_MISSION)

        deadline = time.time() + 30
        while time.time() < deadline:
            msg = self.vehicle.recv_match(
                type=["MISSION_REQUEST_INT", "MISSION_REQUEST", "MISSION_ACK"],
                blocking=True, timeout=5)
            if msg is None:
                continue
            if msg.get_type() == "MISSION_ACK":
                ok = msg.type == mavutil.mavlink.MAV_MISSION_ACCEPTED
                print(f"[İHA] Mission {'yüklendi ✓' if ok else 'HATA: ' + str(msg.type)}")
                return ok
            seq = msg.seq
            if seq >= total:
                continue
            it = items[seq]
            self.vehicle.mav.mission_item_int_send(
                self.vehicle.target_system, self.vehicle.target_component,
                it[0], it[1], it[2], it[3], it[4],
                it[5], it[6], it[7], it[8],
                it[9], it[10], it[11],
                mavutil.mavlink.MAV_MISSION_TYPE_MISSION)
        print("[İHA] Mission yükleme timeout!")
        return False

    def _set_current_item(self, seq, retries=5):
        for _ in range(retries):
            self.vehicle.mav.mission_set_current_send(
                self.vehicle.target_system, self.vehicle.target_component, seq)
            msg = self.vehicle.recv_match(
                type="MISSION_CURRENT", blocking=True, timeout=3)
            if msg and msg.seq == seq:
                print(f"[İHA] Mission başlangıç: item {msg.seq}")
                return True
            time.sleep(0.5)
        print(f"[İHA] Mission current {seq} onaylanamadı – devam ediliyor.")
        return False

    def _rtl(self):
        print("[İHA] RTL başlatılıyor...")
        self._set_mode("RTL")
        self.mission_active = False

    # ── Ana görev ─────────────────────────────────────────────────────────────

    def _run_mission(self):
        self._set_speed(CRUISE_SPEED)
        home_lat, home_lon, home_alt = self._get_position()
        if home_lat is None:
            print("[İHA] GPS alınamadı!")
            return

        print(f"\n[İHA] Ev konumu: lat={home_lat:.7f}  lon={home_lon:.7f}")
        print(f"[İHA] Waypoint'ler:")
        for i, (la, lo, al) in enumerate(WAYPOINTS):
            dist = haversine(home_lat, home_lon, la, lo)
            print(f"  WP{i+1}: lat={la:.7f}  lon={lo:.7f}  alt={al:.0f}m  ({dist:.0f}m uzakta)")

        self._clear_mission()
        time.sleep(0.5)
        if not self._upload_mission(home_lat, home_lon, home_alt, WAYPOINTS):
            print("[İHA] Mission yüklenemedi!")
            self._rtl()
            return

        self._set_current_item(1)
        self._set_mode("AUTO")

        n        = len(WAYPOINTS)
        visited  = [None] * n
        next_seq = 1
        deadline = time.time() + 600

        print(f"\n[İHA] Mission izleniyor ({n} waypoint)...")

        while next_seq <= n and self.mission_active and time.time() < deadline:
            mreach = self.vehicle.recv_match(
                type="MISSION_ITEM_REACHED", blocking=True, timeout=5)

            if mreach is None:
                lat, lon, _ = self._get_position()
                if lat is not None:
                    wp_lat, wp_lon, _ = WAYPOINTS[next_seq - 1]
                    dist = haversine(lat, lon, wp_lat, wp_lon)
                    print(f"[İHA] WP{next_seq} bekleniyor – {dist:.0f} m", end="\r")
                    if dist < WP_ARRIVAL_DIST:
                        mreach_seq = next_seq
                    else:
                        continue
                else:
                    continue
            else:
                mreach_seq = mreach.seq

            while next_seq <= mreach_seq and next_seq <= n:
                idx = next_seq - 1
                wp_lat, wp_lon, wp_alt = WAYPOINTS[idx]
                cur_lat, cur_lon, cur_alt = self._get_position()
                if cur_lat is None:
                    cur_lat, cur_lon, cur_alt = wp_lat, wp_lon, wp_alt
                visited[idx] = (cur_lat, cur_lon, cur_alt)
                print(f"\n{'─' * 50}")
                print(f"  [İHA] WAYPOINT {next_seq} ULAŞILDI")
                print(f"  latitude  : {cur_lat:.7f}")
                print(f"  longitude : {cur_lon:.7f}")
                print(f"  altitude  : {cur_alt:.1f} m")
                print(f"{'─' * 50}")
                time.sleep(WP_HOVER_TIME)
                next_seq += 1

        print("\n" + "═" * 50)
        print("  [İHA] TÜM WAYPOINT'LER TAMAMLANDI")
        print("═" * 50)
        for i, v in enumerate(visited):
            if v:
                print(f"  WP{i+1}: lat={v[0]:.7f}  lon={v[1]:.7f}  alt={v[2]:.1f}m")
        print("═" * 50)
        self._rtl()

    # ── Giriş noktası ─────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "═" * 50)
        print("  VTOL İHA – OTONOM GÖREV")
        print("═" * 50)
        print(f"  MAVLink : {VTOL_CONNECTION}")
        print(f"  Kalkış  : {TAKEOFF_ALT} m  |  Hız: {CRUISE_SPEED} m/s")
        print(f"  WP sayısı: {len(WAYPOINTS)}")
        for i, (la, lo, al) in enumerate(WAYPOINTS):
            print(f"    WP{i+1}: ({la}, {lo}, {al}m)")
        print("═" * 50 + "\n")

        if not self.connect():
            print("[İHA] Bağlantı kurulamadı.")
            return

        if not self._set_mode("GUIDED"):
            self._set_mode("QSTABILIZE")
        time.sleep(1)
        self._wait_prearm(timeout=30)
        self._arm()
        self._takeoff(TAKEOFF_ALT)

        try:
            self._run_mission()
        except KeyboardInterrupt:
            print("\n[İHA] Kullanıcı tarafından durduruldu.")
            self._rtl()
        finally:
            self.mission_active = False
            print("[İHA] Script sonlandı.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    VTOLController().run()
