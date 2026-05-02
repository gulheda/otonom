"""
VTOL İHA – Otonom Koordinat Gitme
==================================
Araç    : alti_transition_quad  (ArduPlane VTOL)
SysID   : 1  |  UDP : 14550

Görev:
  Kalkış → Spawn noktasına göre 3 waypoint üret → AUTO mod mission yükle
  → Her noktaya var, koordinatları yazdır → RTL

Bağımlılıklar:
    pip install pymavlink
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import math
import socket
import json
from pymavlink import mavutil


# ═══════════════════════════════════════════════════════════════════════════════
#  AYARLAR
# ═══════════════════════════════════════════════════════════════════════════════

VTOL_CONNECTION = "udp:127.0.0.1:14550"
VTOL_SYSID      = 1

TAKEOFF_ALT     = 30.0   # metre – kalkış irtifası
CRUISE_SPEED    = 5.0    # m/s  – seyir hızı

WP_OFFSET_M     = 50.0   # metre – ev konumuna göre WP uzaklığı
WP_ALT          = 30.0   # metre – waypoint irtifası
WP_ARRIVAL_DIST = 10.0   # metre – "ulaşıldı" toleransı
WP_HOVER_TIME   = 3.0    # saniye – her WP üzerinde bekleme


# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki mesafeyi metre cinsinden döndürür."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
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
                    VTOL_CONNECTION,
                    source_system=255,
                    target_system=VTOL_SYSID,
                )
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
                _send()
                last_send = time.time()
            msg = self.vehicle.recv_match(
                type=["COMMAND_ACK", "HEARTBEAT"], blocking=True, timeout=1)
            if msg is None:
                continue
            if msg.get_type() == "HEARTBEAT":
                if msg.get_srcSystem() != self.vehicle.target_system:
                    continue
                if getattr(msg, "custom_mode", None) == mode_id:
                    print(f"[İHA] Mod → {mode_name} (HEARTBEAT)")
                    return True
            elif msg.get_type() == "COMMAND_ACK":
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print(f"[İHA] Mod → {mode_name} (ACK)")
                    return True

        hb = self.vehicle.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if (hb and hb.get_srcSystem() == self.vehicle.target_system
                and getattr(hb, "custom_mode", None) == mode_id):
            print(f"[İHA] Mod → {mode_name} (gecikmiş)")
            return True

        print(f"[İHA] Mod değiştirme onaylanamadı: {mode_name} – devam ediliyor.")
        return False

    def _arm(self):
        print("[İHA] ARM ediliyor...")
        self.vehicle.arducopter_arm()
        self.vehicle.motors_armed_wait()
        print("[İHA] ARM tamamlandı.")

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
                print("[İHA] GPS bekleniyor...")
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
            return None, None, None, None
        return (msg.lat / 1e7, msg.lon / 1e7,
                msg.relative_alt / 1000.0, msg.hdg / 100.0)

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
        items = []
        # Item 0: Home
        items.append((0, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                       mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                       0, 1, 0.0, 0.0, 0.0, 0.0,
                       int(home_lat * 1e7), int(home_lon * 1e7), float(home_alt)))
        # Items 1..n: Waypoints
        for i, (lat, lon, alt) in enumerate(waypoints):
            items.append((i + 1, mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                           mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                           0, 1, 0.0, float(WP_ARRIVAL_DIST), 0.0, float("nan"),
                           int(lat * 1e7), int(lon * 1e7), float(alt)))
        # RTL
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

    def _set_current_item(self, seq):
        self.vehicle.mav.mission_set_current_send(
            self.vehicle.target_system, self.vehicle.target_component, seq)
        msg = self.vehicle.recv_match(type="MISSION_CURRENT", blocking=True, timeout=5)
        if msg:
            print(f"[İHA] Mission başlangıç item: {msg.seq}")

    # ── RTL ──────────────────────────────────────────────────────────────────

    def _rtl(self):
        print("[İHA] RTL başlatılıyor...")
        self._set_mode("RTL")
        self.mission_active = False

    # ── Ana görev ────────────────────────────────────────────────────────────

    def _waypoint_mission(self):
        self._set_speed(CRUISE_SPEED)

        # Kalkış konumunu oku
        home_lat, home_lon, home_alt, _ = self._get_position()
        if home_lat is None:
            print("[İHA] GPS alınamadı!")
            return

        print(f"\n[İHA] Ev konumu: lat={home_lat:.7f}  lon={home_lon:.7f}")

        # Ev etrafında 3 waypoint üret (kuzey / doğu / güney)
        d    = WP_OFFSET_M
        dlat = d / 111_320.0
        dlon = d / (111_320.0 * math.cos(math.radians(home_lat)))
        waypoints = [
            (home_lat + dlat, home_lon,        WP_ALT),
            (home_lat,        home_lon + dlon, WP_ALT),
            (home_lat - dlat, home_lon,        WP_ALT),
        ]

        print(f"[İHA] Waypoint'ler ({d}m offset):")
        for i, (la, lo, al) in enumerate(waypoints):
            print(f"  WP{i+1}: lat={la:.7f}  lon={lo:.7f}  alt={al:.0f}m")

        # Mission yükle ve AUTO başlat
        self._clear_mission()
        time.sleep(0.5)
        if not self._upload_mission(home_lat, home_lon, home_alt, waypoints):
            print("[İHA] Mission yüklenemedi!")
            self._rtl()
            return

        self._set_current_item(1)
        self._set_mode("AUTO")

        # Waypoint döngüsü
        visited = []
        for idx, (wp_lat, wp_lon, wp_alt) in enumerate(waypoints):
            if not self.mission_active:
                break

            print(f"\n[İHA] ── WAYPOINT {idx+1}/3 ──")
            print(f"[İHA] Hedef: lat={wp_lat:.7f}  lon={wp_lon:.7f}")

            mission_seq = idx + 1
            while self.mission_active:
                # MISSION_ITEM_REACHED önce kontrol et (araç hızlı geçebilir)
                mreach = self.vehicle.recv_match(
                    type="MISSION_ITEM_REACHED", blocking=False)
                if mreach is not None and mreach.seq >= mission_seq:
                    print(f"\n[İHA] Mission item {mreach.seq} tamamlandı.")
                    break
                lat, lon, _, _ = self._get_position()
                if lat is None:
                    time.sleep(0.3)
                    continue
                dist = haversine(lat, lon, wp_lat, wp_lon)
                print(f"[İHA] Mesafe: {dist:.1f} m", end="\r")
                if dist < WP_ARRIVAL_DIST:
                    print()
                    break
                time.sleep(0.3)

            cur_lat, cur_lon, cur_alt, _ = self._get_position()
            if cur_lat is None:
                cur_lat, cur_lon, cur_alt = wp_lat, wp_lon, wp_alt

            visited.append((cur_lat, cur_lon, cur_alt))
            print("─" * 52)
            print(f"  [İHA] WAYPOINT {idx+1} ULAŞILDI")
            print(f"  latitude  : {cur_lat:.7f}")
            print(f"  longitude : {cur_lon:.7f}")
            print(f"  altitude  : {cur_alt:.1f} m")
            print("─" * 52)
            time.sleep(WP_HOVER_TIME)

        # Özet
        print("\n" + "═" * 52)
        print("  [İHA] TÜM WAYPOINT'LER TAMAMLANDI")
        print("═" * 52)
        for i, (la, lo, al) in enumerate(visited):
            print(f"  WP{i+1}: lat={la:.7f}  lon={lo:.7f}  alt={al:.1f}m")
        print("═" * 52)

        self._rtl()

    # ── Giriş noktası ─────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "═" * 52)
        print("  VTOL İHA – OTONOM KOORDİNAT GİTME")
        print("═" * 52)
        print(f"  MAVLink    : {VTOL_CONNECTION}")
        print(f"  Kalkış     : {TAKEOFF_ALT} m")
        print(f"  WP offset  : {WP_OFFSET_M} m  |  WP irtifa: {WP_ALT} m")
        print(f"  Hız        : {CRUISE_SPEED} m/s")
        print("═" * 52 + "\n")

        if not self.connect():
            print("[İHA] Bağlantı kurulamadı.")
            return

        self._set_mode("GUIDED")
        time.sleep(1)
        self._arm()
        self._takeoff(TAKEOFF_ALT)

        try:
            self._waypoint_mission()
        except KeyboardInterrupt:
            print("\n[İHA] Kullanıcı tarafından durduruldu.")
            self._rtl()
        finally:
            self.mission_active = False
            print("[İHA] Script sonlandı.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    vtol = VTOLController()
    vtol.run()
