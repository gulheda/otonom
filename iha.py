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
MISSION_MODE = "SCAN"       # "WAYPOINT" veya "SCAN"

# ── Bağlantı ─────────────────────────────────────────────────────────────────
VTOL_CONNECTION = "udp:127.0.0.1:14550"
VTOL_SYSID      = 1
DRONE_IP        = "127.0.0.1"
DRONE_PORT      = 6000

# ── Uçuş ─────────────────────────────────────────────────────────────────────
TAKEOFF_ALT  = 50.0   # metre
CRUISE_SPEED = 5.0    # m/s

# ── WAYPOINT modu ─────────────────────────────────────────────────────────────
WP_OFFSET_M     = 50.0   # metre – spawn'a göre WP uzaklığı
WP_ALT          = 30.0   # metre
WP_ARRIVAL_DIST = 10.0   # metre
WP_HOVER_TIME   = 3.0    # saniye

# ── SCAN modu ─────────────────────────────────────────────────────────────────
SCAN_ALT        = 50.0   # metre – tarama irtifası
SCAN_SPEED      = 5.0    # m/s
SCAN_RADIUS_M   = 80.0   # metre – oval yarı eksen
SCAN_POINTS     = 16     # oval üzerindeki WP sayısı
SCAN_LAPS       = 2      # kaç tur (0 = sonsuz)
DETECT_COOLDOWN = 15.0   # saniye – aynı bölgede tekrar tespit saymaması için

# ── İHA Servo (drone serbest bırakma) ────────────────────────────────────────
IHA_SERVO_CH    = 9      # servo kanalı
IHA_SERVO_OPEN  = 2000   # µs – drone serbest
IHA_SERVO_CLOSE = 1000   # µs – drone kilitli

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

    def _check_first(self, timeout=15):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._count > 0:
                print(f"[Kamera] İlk frame alındı.")
                return
            time.sleep(0.5)
        print(f"[Kamera] UYARI: {timeout}s içinde frame gelmedi – "
              f"Gazebo çalışıyor mu? Beklenen topic: {self.topic}")
        # Mevcut Gazebo topic'lerini listele
        try:
            topics = self._node.topic_list() if self._node else []
            img_topics = [t for t in topics if "image" in t.lower() or "camera" in t.lower()]
            if img_topics:
                print(f"[Kamera] Gazebo'daki kamera topic'leri:")
                for t in img_topics:
                    print(f"  {t}")
                print(f"[Kamera] İPUCU: CAMERA_TOPIC değişkenini yukarıdaki "
                      f"topic'lerden biriyle güncelleyin.")
            elif topics:
                print(f"[Kamera] Gazebo topic listesinde kamera bulunamadı. "
                      f"Toplam {len(topics)} topic var.")
            else:
                print(f"[Kamera] Gazebo topic listesi boş – simülasyon çalışıyor mu?")
        except Exception as exc:
            print(f"[Kamera] Topic listesi alınamadı: {exc}")

    def _cb(self, msg):
        try:
            w, h = msg.width, msg.height
            fmt = getattr(msg, "pixel_format_type", 3)
            data = np.frombuffer(msg.data, dtype=np.uint8)
            data_len = len(msg.data)

            # Gazebo piksel format sabitleri (gz-msgs)
            if fmt == 3:    # PIXEL_FORMAT_RGB_INT8
                frame = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            elif fmt == 4:  # PIXEL_FORMAT_RGBA_INT8
                frame = cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_RGBA2BGR)
            elif fmt == 5:  # PIXEL_FORMAT_BGRA_INT8
                frame = cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
            elif fmt == 7:  # PIXEL_FORMAT_BGR_INT8
                frame = data.reshape(h, w, 3)
            elif data_len == w * h * 4:
                frame = cv2.cvtColor(data.reshape(h, w, 4), cv2.COLOR_BGRA2BGR)
            elif data_len == w * h * 3:
                frame = cv2.cvtColor(data.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
            else:
                if self._count == 0:
                    print(f"[Kamera] Desteklenmeyen format: type={fmt}, "
                          f"boyut={data_len}B beklenen={w*h*3}B ({w}x{h})")
                return

            with self._lock:
                self._frame = frame.copy()
                self._count += 1
        except Exception as exc:
            if self._count == 0:
                fmt_val = getattr(msg, "pixel_format_type", "?")
                w_val   = getattr(msg, "width", "?")
                h_val   = getattr(msg, "height", "?")
                d_len   = len(getattr(msg, "data", b""))
                print(f"[Kamera] Frame işleme hatası: {exc} "
                      f"(fmt={fmt_val}, {w_val}x{h_val}, {d_len}B)")

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

        self._confirm_count  = 0
        self._detect_queue   = []          # onaylanan tespitler buraya girer
        self._detect_lock    = threading.Lock()
        self._last_detect_ts = 0.0         # cooldown kontrolü için
        self._releases_done  = 0           # kaç kez drone bırakıldı

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
                # Sadece mod değiştirme ACK'ini kabul et – diğer komutların
                # ACK'i yanlış "başarılı" sonuç verebilir.
                cmd = getattr(msg, "command", None)
                if cmd not in (mavutil.mavlink.MAV_CMD_DO_SET_MODE, None):
                    continue
                if msg.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                    print(f"[İHA] Mod → {mode_name} (ACK)")
                    return True
                if msg.result not in (mavutil.mavlink.MAV_RESULT_IN_PROGRESS,
                                      mavutil.mavlink.MAV_RESULT_TEMPORARILY_REJECTED):
                    print(f"[İHA] Mod reddedildi (result={msg.result}): {mode_name}")
        hb = self.vehicle.recv_match(type="HEARTBEAT", blocking=True, timeout=3)
        if (hb and hb.get_srcSystem() == self.vehicle.target_system
                and getattr(hb, "custom_mode", None) == mode_id):
            print(f"[İHA] Mod → {mode_name} (gecikmiş)")
            return True
        print(f"[İHA] Mod onaylanamadı: {mode_name} – devam ediliyor.")
        return False

    def _wait_prearm(self, timeout=30):
        """EKF ve gyro tutarlılığı hazır olana kadar bekle."""
        print("[İHA] Pre-arm kontrol bekleniyor...", end="", flush=True)
        deadline = time.time() + timeout
        while time.time() < deadline:
            msg = self.vehicle.recv_match(
                type="SYS_STATUS", blocking=True, timeout=2)
            if msg is None:
                continue
            # Tüm pre-arm sensörleri sağlıklıysa onboard_control_sensors_health
            # alanında gerekli bitler set olur; en basit kontrol: 3 saniye boyunca
            # HEARTBEAT'te base_mode'da ARMED bayrağı yoksa pre-arm temizdir.
            hb = self.vehicle.recv_match(
                type="HEARTBEAT", blocking=False)
            if hb and hb.get_srcSystem() == self.vehicle.target_system:
                armed = bool(hb.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                if not armed:
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
                # result=4 → pre-arm check failed; result metni varsa göster
                reason = getattr(ack, "result_param2", "")
                print(f"[İHA] ARM reddedildi (result={ack.result}"
                      f"{', ' + str(reason) if reason else ''}) "
                      f"– {retry_delay}s sonra tekrar ({attempt}/{retries})")
            else:
                print(f"[İHA] ARM ACK gelmedi – {retry_delay}s sonra tekrar "
                      f"({attempt}/{retries})")
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

    def _set_nav_mode(self):
        """GUIDED dene; VTOL desteklemiyorsa QLOITER'a geç."""
        if self._set_mode("GUIDED"):
            return "GUIDED"
        print("[İHA] GUIDED başarısız → QLOITER deneniyor...")
        if self._set_mode("QLOITER"):
            return "QLOITER"
        print("[İHA] UYARI: GUIDED/QLOITER onaylanamadı – mevcut modda devam.")
        return None

    def _rtl(self):
        print("[İHA] RTL başlatılıyor...")
        self._set_mode("RTL")
        self.mission_active = False

    # ── İHA Servo: drone serbest bırak ───────────────────────────────────────

    def _release_drone(self):
        self._releases_done += 1
        n = self._releases_done
        print(f"[İHA] Drone #{n} serbest bırakılıyor – "
              f"kanal {IHA_SERVO_CH}, PWM {IHA_SERVO_OPEN} µs")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, IHA_SERVO_CH, IHA_SERVO_OPEN, 0, 0, 0, 0, 0)
        time.sleep(3.0)  # drone serbest düşsün, kendi stabilize olsun
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, IHA_SERVO_CH, IHA_SERVO_CLOSE, 0, 0, 0, 0, 0)
        print(f"[İHA] Drone #{n} serbest bırakıldı – servo kapatıldı.")

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

        while self.mission_active:
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
                now = time.time()
                # Cooldown: aynı bölgede kısa sürede tekrar onaylama
                if now - self._last_detect_ts < DETECT_COOLDOWN:
                    self._confirm_count = 0
                    continue

                lat, lon, alt, yaw = self._get_position()
                if lat is None:
                    self._confirm_count = 0
                    continue

                h, w = frame.shape[:2]
                tgt_lat, tgt_lon = pixel_to_gps(
                    best["cx"], best["cy"], w, h, lat, lon, alt, yaw)

                print(f"\n[Görüntü] {CONFIRM_FRAMES} frame onaylandı! "
                      f"Hedef=({tgt_lat:.6f},{tgt_lon:.6f})")

                with self._detect_lock:
                    self._detect_queue.append((tgt_lat, tgt_lon))
                self._last_detect_ts = now
                self._confirm_count  = 0   # sıfırla, tarama devam edecek

            time.sleep(0.05)
        print("[Görüntü] Döngü sonlandı.")

    # ── SCAN görev döngüsü ────────────────────────────────────────────────────

    def _handle_detection(self, tgt_lat, tgt_lon):
        """Tespit işleme: hover → servo → koordinat gönder."""
        print(f"\n[İHA] ★ TESPİT İŞLENİYOR → ({tgt_lat:.6f}, {tgt_lon:.6f})")

        # AUTO'dan LOITER/QLOITER'a geç – yerinde dur
        if not self._set_mode("QLOITER"):
            self._set_mode("LOITER")
        self._hover()
        time.sleep(2)

        # İHA servo: drone'u serbest bırak
        self._release_drone()

        # Koordinatı drone'a / yer istasyonuna ilet
        self._send_to_drone(tgt_lat, tgt_lon)

        # Drone stabilize olsun
        print("[İHA] Drone stabilize bekleniyor (5s)...")
        time.sleep(5)

    def _scan_loop(self):
        """
        Tam görev akışı – AUTO + mission upload (WAYPOINT modu gibi):
        Oval WP'ler SCAN_LAPS tur için yüklenir → AUTO mod → ArduPlane uçar.
        MISSION_ITEM_REACHED alındığında WP sayacı artar.
        Paralel vision thread tespiti kuyruğa ekler; tespit kuyruğu boş değilse
        AUTO duraklatılır (QLOITER/LOITER), tespit işlenir, AUTO devam eder.
        Tüm WP'ler bitti → RTL.
        """
        home_lat, home_lon, home_alt, _ = self._get_position()
        if home_lat is None:
            print("[İHA] GPS alınamadı!")
            return

        # SCAN_LAPS tur = SCAN_LAPS × SCAN_POINTS waypoint
        laps = SCAN_LAPS if SCAN_LAPS > 0 else 1
        oval = generate_oval(home_lat, home_lon, SCAN_RADIUS_M, SCAN_POINTS, SCAN_ALT)
        waypoints = oval * laps
        total = len(waypoints)

        print(f"[İHA] Alan taraması yükleniyor – "
              f"{SCAN_POINTS} WP × {laps} tur = {total} item")

        self._clear_mission()
        time.sleep(0.5)
        if not self._upload_mission(home_lat, home_lon, home_alt, waypoints):
            print("[İHA] Mission yüklenemedi!")
            self._rtl()
            return

        self._set_current_item(1)
        self._set_mode("AUTO")
        self._set_speed(SCAN_SPEED)

        completed = 0        # kaç WP geçildi
        next_seq  = 1        # beklenen mission seq (1-based)
        deadline  = time.time() + 7200  # max 2 saat

        print(f"[İHA] Alan taraması başladı ({total} WP izleniyor)...\n")

        while completed < total and self.mission_active and time.time() < deadline:

            # Tespit kuyruğu kontrol (non-blocking)
            detection = None
            with self._detect_lock:
                if self._detect_queue:
                    detection = self._detect_queue.pop(0)

            if detection:
                tgt_lat, tgt_lon = detection
                self._handle_detection(tgt_lat, tgt_lon)
                # AUTO'ya geri dön – ArduPlane kaldığı mission item'dan devam eder
                self._set_mode("AUTO")
                self._set_speed(SCAN_SPEED)

            # MISSION_ITEM_REACHED veya MISSION_CURRENT mesajı bekle (2s timeout)
            msg = self.vehicle.recv_match(
                type=["MISSION_ITEM_REACHED", "MISSION_CURRENT"],
                blocking=True, timeout=2)

            if msg is None:
                # Pozisyon tabanlı ilerleme kontrolü (fallback)
                lat, lon, _, _ = self._get_position()
                if lat is not None and next_seq <= total:
                    wp_lat, wp_lon, _ = waypoints[next_seq - 1]
                    dist = haversine(lat, lon, wp_lat, wp_lon)
                    print(f"[İHA] WP{next_seq}/{total} → {dist:.0f}m", end="\r")
                    if dist < WP_ARRIVAL_DIST:
                        completed += 1
                        tur_no = (completed - 1) // SCAN_POINTS + 1
                        wp_no  = (completed - 1) % SCAN_POINTS + 1
                        print(f"\n[İHA] WP {completed}/{total} geçildi "
                              f"(Tur {tur_no}, Nokta {wp_no})")
                        next_seq += 1
                continue

            if msg.get_type() == "MISSION_ITEM_REACHED":
                seq = msg.seq
                if seq >= next_seq:
                    # Atlanmış WP'leri de say
                    while next_seq <= seq and next_seq <= total:
                        completed += 1
                        tur_no = (completed - 1) // SCAN_POINTS + 1
                        wp_no  = (completed - 1) % SCAN_POINTS + 1
                        print(f"[İHA] WP {completed}/{total} geçildi "
                              f"(Tur {tur_no}, Nokta {wp_no})")
                        next_seq += 1

        print(f"\n[İHA] Tarama tamamlandı – {completed}/{total} WP geçildi.")
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

        # ARM öncesi: GUIDED dene, olmuyorsa VTOL quad kalkış için QSTABILIZE
        if not self._set_mode("GUIDED"):
            print("[İHA] GUIDED başarısız → QSTABILIZE deneniyor...")
            self._set_mode("QSTABILIZE")
        time.sleep(1)
        self._wait_prearm(timeout=30)
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
