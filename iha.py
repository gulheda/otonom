"""
VTOL İHA – Otonom Koordinat Gitme + İnsan Tespiti
==================================================
Araç    : alti_transition_quad  (ArduPlane VTOL)
SysID   : 1  |  UDP : 14550

Modlar:
  WAYPOINT – Kalkış → 3 dinamik WP → koordinat yazdır → RTL
  SCAN     – Kalkış → oval tarama → YOLOv8 insan tespiti
             → GPS hesapla → drone'a UDP gönder → RTL

Bağımlılıklar:
    pip install pymavlink ultralytics opencv-python
    (gz.transport13 Gazebo Harmonic ile gelir)
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import time
import math
import threading
import socket
import json
import cv2
import numpy as np
from pymavlink import mavutil

# ── YOLOv8 ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[UYARI] 'ultralytics' bulunamadı – YOLOv8 devre dışı.")

# ── Gazebo Transport ─────────────────────────────────────────────────────────
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.image_pb2 import Image as GzImage
    GZ_AVAILABLE = True
except Exception as exc:
    GZ_AVAILABLE = False
    print(f"[UYARI] gz.transport13 yüklenemedi: {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  AYARLAR
# ═══════════════════════════════════════════════════════════════════════════════

# ── Mod seçimi ────────────────────────────────────────────────────────────────
MISSION_MODE = "SCAN"   # "WAYPOINT" veya "SCAN"

# ── Bağlantı ─────────────────────────────────────────────────────────────────
VTOL_CONNECTION = "udp:127.0.0.1:14550"
VTOL_SYSID      = 1
DRONE_IP        = "127.0.0.1"
DRONE_PORT      = 6000

# ── Uçuş ─────────────────────────────────────────────────────────────────────
TAKEOFF_ALT  = 30.0   # metre
CRUISE_SPEED = 5.0    # m/s

# ── WAYPOINT modu ─────────────────────────────────────────────────────────────
WP_OFFSET_M     = 50.0   # metre – spawn'a göre WP uzaklığı
WP_ALT          = 30.0   # metre
WP_ARRIVAL_DIST = 10.0   # metre
WP_HOVER_TIME   = 3.0    # saniye

# ── SCAN modu ─────────────────────────────────────────────────────────────────
SCAN_ALT        = 30.0   # metre – tarama irtifası
SCAN_SPEED      = 5.0    # m/s
SCAN_RADIUS_M   = 80.0   # metre – oval yarı eksen
SCAN_POINTS     = 16     # oval üzerindeki WP sayısı
SCAN_REPEAT     = 0      # 0 = tespit edilene kadar tekrar

# ── YOLOv8 ───────────────────────────────────────────────────────────────────
YOLO_MODEL      = "yolov8n.pt"
CONFIRM_FRAMES  = 5      # ardışık kaç frame'de tespit = onay
CONFIRM_CONF    = 0.60   # minimum güven skoru

# ── Kamera ───────────────────────────────────────────────────────────────────
CAMERA_TOPIC    = "/camera/image"
CAMERA_HFOV_DEG = 60.0


# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (math.sin(dphi / 2) ** 2
         + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def pixel_to_gps(cx, cy, img_w, img_h, vtol_lat, vtol_lon, vtol_alt, yaw_deg):
    """Piksel koordinatından GPS hesaplar (nadir kamera, sıfır pitch/roll)."""
    hfov = math.radians(CAMERA_HFOV_DEG)
    vfov = hfov * img_h / img_w
    dx_m = vtol_alt * math.tan((cx / img_w - 0.5) * hfov)
    dy_m = vtol_alt * math.tan((cy / img_h - 0.5) * vfov)
    yaw  = math.radians(yaw_deg)
    north =  dx_m * math.sin(yaw) - dy_m * math.cos(yaw)
    east  =  dx_m * math.cos(yaw) + dy_m * math.sin(yaw)
    dlat  = north / 111_320.0
    dlon  = east  / (111_320.0 * math.cos(math.radians(vtol_lat)))
    return vtol_lat + dlat, vtol_lon + dlon


def generate_oval(center_lat, center_lon, radius_m, num_points, alt):
    """Çember üzerinde eşit aralıklı waypoint listesi üretir."""
    dlat = radius_m / 111_320.0
    dlon = radius_m / (111_320.0 * math.cos(math.radians(center_lat)))
    return [
        (center_lat + math.sin(2 * math.pi * i / num_points) * dlat,
         center_lon + math.cos(2 * math.pi * i / num_points) * dlon,
         alt)
        for i in range(num_points)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  GAZEBO KAMERA
# ═══════════════════════════════════════════════════════════════════════════════

class GazeboCamera:
    def __init__(self, topic=CAMERA_TOPIC):
        self.topic    = topic
        self._frame   = None
        self._lock    = threading.Lock()
        self._count   = 0
        self._node    = None
        self._sub     = None

    def start(self):
        if not GZ_AVAILABLE:
            print("[Kamera] gz.transport13 yok.")
            return False
        try:
            self._node = GzNode()
            self._sub  = self._node.subscribe(GzImage, self.topic, self._cb)
            print(f"[Kamera] Abone olundu: {self.topic}")
            threading.Thread(target=self._check_first, daemon=True).start()
            return True
        except Exception as exc:
            print(f"[Kamera] Başlatma hatası: {exc}")
            return False

    def _check_first(self, timeout=8):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._count > 0:
                print(f"[Kamera] İlk frame alındı.")
                return
            time.sleep(0.5)
        print(f"[Kamera] UYARI: {timeout}s içinde frame gelmedi – "
              f"Gazebo çalışıyor mu? Topic: {self.topic}")

    def _cb(self, msg):
        try:
            w, h = msg.width, msg.height
            data = np.frombuffer(msg.data, dtype=np.uint8)
            if msg.pixel_format_type == 3:
                frame = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            else:
                frame = data.reshape(h, w, 3)
            with self._lock:
                self._frame = frame.copy()
                self._count += 1
        except Exception:
            pass

    def get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def frame_count(self):
        return self._count


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLO TESPİT
# ═══════════════════════════════════════════════════════════════════════════════

class PersonDetector:
    def __init__(self):
        self._model = None
        if not YOLO_AVAILABLE:
            return
        try:
            self._model = YOLO(YOLO_MODEL)
            print(f"[YOLO] Model yüklendi: {YOLO_MODEL}")
        except Exception as exc:
            print(f"[YOLO] Model yükleme hatası: {exc}")

    def detect(self, frame):
        """Person (class 0) tespiti. Sonuç: [{'conf', 'cx', 'cy', 'bbox'}]"""
        if self._model is None or frame is None:
            return []
        found = []
        try:
            for r in self._model(frame, verbose=False):
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    if int(box.cls[0]) != 0:
                        continue
                    conf = float(box.conf[0])
                    if conf < CONFIRM_CONF:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    found.append({"conf": conf,
                                  "cx": (x1 + x2) // 2,
                                  "cy": (y1 + y2) // 2,
                                  "bbox": [x1, y1, x2, y2]})
        except Exception as exc:
            print(f"[YOLO] Çıkarım hatası: {exc}")
        return found


# ═══════════════════════════════════════════════════════════════════════════════
#  VTOL KONTROLCÜSÜ
# ═══════════════════════════════════════════════════════════════════════════════

class VTOLController:

    def __init__(self):
        self.vehicle        = None
        self.mission_active = True
        self.camera         = GazeboCamera()
        self.detector       = PersonDetector()

        self._confirm_count = 0
        self._target_found  = False
        self._target_lock   = threading.Lock()
        self._target_lat    = None
        self._target_lon    = None

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
        print(f"[İHA] Mod onaylanamadı: {mode_name} – devam ediliyor.")
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

    def _goto(self, lat, lon, alt):
        """GUIDED modda hedefe git (ArduCopter + ArduPlane komutları birlikte)."""
        self.vehicle.mav.set_position_target_global_int_send(
            0, self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_1111_1111_1000,
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0)
        self.vehicle.mav.command_int_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
            mavutil.mavlink.MAV_CMD_DO_REPOSITION,
            0, 0, -1, 0, 0, float("nan"),
            int(lat * 1e7), int(lon * 1e7), alt)

    def _hover(self):
        lat, lon, alt, _ = self._get_position()
        if lat is not None:
            self._goto(lat, lon, alt)

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
                print(f"[İHA] Mission başlangıç item: {msg.seq}")
                return True
            time.sleep(0.5)
        print(f"[İHA] Mission current {seq} onaylanamadı – devam ediliyor.")
        return False

    def _rtl(self):
        print("[İHA] RTL başlatılıyor...")
        self._set_mode("RTL")
        self.mission_active = False

    # ── Drone'a koordinat gönder ──────────────────────────────────────────────

    def _send_to_drone(self, lat, lon, alt=25.0):
        print("\n" + "═" * 52)
        print("  [İHA] HEDEF TESPİT EDİLDİ")
        print(f"  latitude  : {lat:.7f}")
        print(f"  longitude : {lon:.7f}")
        print(f"  altitude  : {alt:.1f} m")
        print("═" * 52)
        payload = json.dumps({"cmd": "GOTO", "lat": lat, "lon": lon, "alt": alt})
        for attempt in range(3):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                s.sendto(payload.encode(), (DRONE_IP, DRONE_PORT))
                s.close()
                print(f"[İHA→Drone] Koordinat gönderildi ({DRONE_IP}:{DRONE_PORT})")
                return
            except Exception as exc:
                print(f"[İHA→Drone] Hata ({attempt+1}/3): {exc}")
                time.sleep(1)

    # ── Vision thread ─────────────────────────────────────────────────────────

    def _vision_loop(self):
        print("[Görüntü] YOLOv8 döngüsü başladı.")
        prev_count = 0
        warn_ts    = time.time()

        while self.mission_active and not self._target_found:
            if self.camera.frame_count == prev_count:
                if time.time() - warn_ts >= 10:
                    print(f"[Görüntü] Frame bekleniyor "
                          f"(sayaç={self.camera.frame_count})...")
                    warn_ts = time.time()
                time.sleep(0.05)
                continue
            warn_ts = time.time()

            frame = self.camera.get_frame()
            prev_count = self.camera.frame_count
            if frame is None:
                continue

            detections = self.detector.detect(frame)
            if not detections:
                self._confirm_count = 0
                continue

            best = max(detections, key=lambda d: d["conf"])
            self._confirm_count += 1
            print(f"[Görüntü] Tespit {self._confirm_count}/{CONFIRM_FRAMES}  "
                  f"conf={best['conf']:.2f}  piksel=({best['cx']},{best['cy']})",
                  end="\r")

            if self._confirm_count >= CONFIRM_FRAMES:
                lat, lon, alt, yaw = self._get_position()
                if lat is None:
                    self._confirm_count = 0
                    continue
                h, w = frame.shape[:2]
                tgt_lat, tgt_lon = pixel_to_gps(
                    best["cx"], best["cy"], w, h, lat, lon, alt, yaw)
                print(f"\n[Görüntü] {CONFIRM_FRAMES} frame onaylandı! "
                      f"Hedef=({tgt_lat:.6f},{tgt_lon:.6f})")
                with self._target_lock:
                    self._target_lat   = tgt_lat
                    self._target_lon   = tgt_lon
                    self._target_found = True
                self._confirm_count = 0

            time.sleep(0.05)
        print("[Görüntü] Döngü sonlandı.")

    # ── SCAN görev döngüsü ────────────────────────────────────────────────────

    def _scan_loop(self):
        """Oval tarama: her turda yeniden mission yükle, vision thread paralel."""
        home_lat, home_lon, home_alt, _ = self._get_position()
        if home_lat is None:
            print("[İHA] GPS alınamadı!")
            return

        waypoints = generate_oval(
            home_lat, home_lon, SCAN_RADIUS_M, SCAN_POINTS, SCAN_ALT)

        print(f"[İHA] Oval tarama – {SCAN_POINTS} WP, yarıçap={SCAN_RADIUS_M}m")

        tur = 0
        while self.mission_active and not self._target_found:
            tur += 1
            if SCAN_REPEAT > 0 and tur > SCAN_REPEAT:
                print(f"[İHA] {SCAN_REPEAT} tur tamamlandı – tespit yok.")
                break

            print(f"\n[İHA] ── TUR {tur} ──")
            self._set_mode("GUIDED")
            self._set_speed(SCAN_SPEED)

            for wp_lat, wp_lon, wp_alt in waypoints:
                if not self.mission_active or self._target_found:
                    break
                self._goto(wp_lat, wp_lon, wp_alt)

                while self.mission_active and not self._target_found:
                    lat, lon, _, _ = self._get_position()
                    if lat is None:
                        time.sleep(0.3)
                        continue
                    if haversine(lat, lon, wp_lat, wp_lon) < WP_ARRIVAL_DIST:
                        break
                    self._goto(wp_lat, wp_lon, wp_alt)  # ArduPlane: periyodik yenile
                    time.sleep(1.0)

        if self._target_found:
            print("\n[İHA] Hedef onaylandı – hover yapılıyor...")
            self._set_mode("GUIDED")
            self._hover()
            time.sleep(3)
            with self._target_lock:
                tgt_lat, tgt_lon = self._target_lat, self._target_lon
            self._send_to_drone(tgt_lat, tgt_lon)
            time.sleep(5)

        self._rtl()

    # ── WAYPOINT görev döngüsü ────────────────────────────────────────────────

    def _waypoint_mission(self):
        self._set_speed(CRUISE_SPEED)
        home_lat, home_lon, home_alt, _ = self._get_position()
        if home_lat is None:
            print("[İHA] GPS alınamadı!")
            return

        print(f"\n[İHA] Ev konumu: lat={home_lat:.7f}  lon={home_lon:.7f}")
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

        self._clear_mission()
        time.sleep(0.5)
        if not self._upload_mission(home_lat, home_lon, home_alt, waypoints):
            print("[İHA] Mission yüklenemedi!")
            self._rtl()
            return

        self._set_current_item(1)
        self._set_mode("AUTO")

        n, visited, next_seq = len(waypoints), [None] * len(waypoints), 1
        print(f"\n[İHA] Mission izleniyor ({n} waypoint)...")
        deadline = time.time() + 600

        while next_seq <= n and self.mission_active and time.time() < deadline:
            mreach = self.vehicle.recv_match(
                type="MISSION_ITEM_REACHED", blocking=True, timeout=5)
            if mreach is None:
                lat, lon, _, _ = self._get_position()
                if lat is not None:
                    wp_lat, wp_lon, _ = waypoints[next_seq - 1]
                    dist = haversine(lat, lon, wp_lat, wp_lon)
                    print(f"[İHA] WP{next_seq} bekleniyor – {dist:.1f} m", end="\r")
                    if dist >= WP_ARRIVAL_DIST:
                        continue
                    mreach_seq = next_seq
                else:
                    continue
            else:
                mreach_seq = mreach.seq

            while next_seq <= mreach_seq and next_seq <= n:
                idx = next_seq - 1
                wp_lat, wp_lon, wp_alt = waypoints[idx]
                cur_lat, cur_lon, cur_alt, _ = self._get_position()
                if cur_lat is None:
                    cur_lat, cur_lon, cur_alt = wp_lat, wp_lon, wp_alt
                visited[idx] = (cur_lat, cur_lon, cur_alt)
                print(f"\n{'─' * 52}")
                print(f"  [İHA] WAYPOINT {next_seq} ULAŞILDI")
                print(f"  latitude  : {cur_lat:.7f}")
                print(f"  longitude : {cur_lon:.7f}")
                print(f"  altitude  : {cur_alt:.1f} m")
                print(f"{'─' * 52}")
                time.sleep(WP_HOVER_TIME)
                next_seq += 1

        print("\n" + "═" * 52)
        print("  [İHA] TÜM WAYPOINT'LER TAMAMLANDI")
        print("═" * 52)
        for i, v in enumerate(visited):
            if v:
                print(f"  WP{i+1}: lat={v[0]:.7f}  lon={v[1]:.7f}  alt={v[2]:.1f}m")
        print("═" * 52)
        self._rtl()

    # ── Giriş noktası ─────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "═" * 52)
        print("  VTOL İHA – OTONOM GÖREV SİSTEMİ")
        print("═" * 52)
        print(f"  Mod        : {MISSION_MODE}")
        print(f"  MAVLink    : {VTOL_CONNECTION}")
        print(f"  Kalkış     : {TAKEOFF_ALT} m  |  Hız: {CRUISE_SPEED} m/s")
        if MISSION_MODE == "SCAN":
            print(f"  Kamera     : {CAMERA_TOPIC}")
            print(f"  YOLO       : {YOLO_MODEL}  |  Eşik: {CONFIRM_CONF}")
            print(f"  Oval       : r={SCAN_RADIUS_M}m, {SCAN_POINTS} WP")
        print("═" * 52 + "\n")

        if not self.connect():
            print("[İHA] Bağlantı kurulamadı.")
            return

        if MISSION_MODE == "SCAN":
            self.camera.start()
            threading.Thread(target=self._vision_loop, daemon=True).start()

        self._set_mode("GUIDED")
        time.sleep(1)
        self._arm()
        self._takeoff(TAKEOFF_ALT)

        try:
            if MISSION_MODE == "SCAN":
                self._scan_loop()
            else:
                self._waypoint_mission()
        except KeyboardInterrupt:
            print("\n[İHA] Kullanıcı tarafından durduruldu.")
            self._rtl()
        finally:
            self.mission_active = False
            print("[İHA] Script sonlandı.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    VTOLController().run()
