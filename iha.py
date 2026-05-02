"""
VTOL İHA Kontrol Scripti
========================
Araç    : alti_transition_quad
SysID   : 1
UDP     : 14550
Görev   : Kalkış → Oval Alan Taraması → Gazebo Kamerasından YOLOv8 ile
          İnsan Tespiti → GPS Koordinatı Hesaplama → Drone'a İletme

Bağımlılıklar:
    pip install pymavlink ultralytics opencv-python
    (gz.transport13 Gazebo Harmonic ile birlikte kurulur)
"""

import os
import time
import math
import threading
import socket
import json

# gz.msgs10 eski protobuf ile derlendi; yeni protobuf sürümleriyle uyumlu çalışması için
# pure-Python implementasyonu zorunlu kılınır (performans kaybı ihmal edilebilir ölçüde).
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2
import numpy as np
from pymavlink import mavutil

# ── YOLOv8 ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[UYARI] 'ultralytics' paketi bulunamadı. YOLOv8 devre dışı.")

# ── Gazebo Transport (gz-harmonic / gz-transport13) ──────────────────────────
try:
    from gz.transport13 import Node as GzNode
    from gz.msgs10.image_pb2 import Image as GzImage
    GZ_AVAILABLE = True
except ImportError:
    GZ_AVAILABLE = False
    print("[UYARI] 'gz.transport13' bulunamadı. Gazebo kamera devre dışı.")
except TypeError as exc:
    # Protobuf sürüm uyumsuzluğu hâlâ çözülemediyse bilgi ver
    GZ_AVAILABLE = False
    print(f"[UYARI] gz.msgs10 yüklenemedi (protobuf uyumsuzluğu): {exc}")


# ═══════════════════════════════════════════════════════════════════════════════
#  KULLANICI TARAFINDAN DÜZENLENECEk BÖLÜM
# ═══════════════════════════════════════════════════════════════════════════════

# Oval tarama alanı merkezi (lat, lon)
SCAN_CENTER     = (47.3982419, 8.5465938)
SCAN_SEMI_MAJOR = 120      # metre – kuzey-güney yarı eksen (uzun taraf)
SCAN_SEMI_MINOR = 70       # metre – doğu-batı yarı eksen (kısa taraf)
SCAN_NUM_POINTS = 24       # oval üzerindeki waypoint sayısı
SCAN_REPEAT     = 0        # tur sayısı (0 = tespit olana kadar sonsuz)
SCAN_ALT        = 50.0     # metre – tarama irtifası
SCAN_SPEED      = 10.0     # m/s  – tarama hızı

# Kalkış
TAKEOFF_ALT     = 50.0     # metre

# YOLOv8 ayarları
YOLO_MODEL      = "yolov8n.pt"   # model dosyası (yoksa otomatik indirilir)
CONFIRM_FRAMES  = 5              # art arda kaç frame onayı gerekli
CONFIRM_CONF    = 0.70           # minimum güven skoru (0-1)

# Kamera parametreleri (Gazebo varsayılan kamera)
CAMERA_HFOV_DEG = 60.0    # yatay görüş açısı (derece)
CAMERA_TOPIC    = "/camera/image"

# ═══════════════════════════════════════════════════════════════════════════════
#  SİSTEM YAPILANDIRMASI (normalde dokunmayın)
# ═══════════════════════════════════════════════════════════════════════════════

VTOL_CONNECTION = "udp:127.0.0.1:14550"
VTOL_SYSID      = 1
DRONE_IP        = "127.0.0.1"
DRONE_MSG_PORT  = 6000
PERSON_CLASS_ID = 0    # COCO veri setinde "person" sınıfı 0'dır


# ═══════════════════════════════════════════════════════════════════════════════
#  GPS & GEOMETRİ YARDIMCILARI
# ═══════════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki mesafeyi metre cinsinden döndürür."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def pixel_to_gps(cx, cy, img_w, img_h, vtol_lat, vtol_lon, vtol_alt_m, yaw_deg):
    """
    Görüntüdeki piksel koordinatından GPS koordinatı hesaplar.

    Kamera yere dik (nadir) baktığı ve pitch/roll sıfır olduğu varsayılır.
    Büyük roll/pitch açılarında ek düzeltme gerekir.

    cx, cy     : bounding box merkezinin piksel koordinatı
    img_w, h   : görüntü boyutu (piksel)
    vtol_alt_m : VTOL'ün yerden yüksekliği (metre)
    yaw_deg    : VTOL başlık açısı (kuzeyden saat yönünde, 0-360 derece)
    """
    if vtol_alt_m <= 0:
        vtol_alt_m = SCAN_ALT   # yedek değer

    hfov_rad = math.radians(CAMERA_HFOV_DEG)
    vfov_rad = hfov_rad * img_h / img_w   # piksel en/boy oranından dikey FOV

    # Görüntü merkezinden normalize sapma (-1 ile +1)
    dx_norm = (cx - img_w / 2.0) / (img_w / 2.0)
    dy_norm = (cy - img_h / 2.0) / (img_h / 2.0)

    # Yerdeki metrik sapma: tan(açı) × yükseklik
    dx_m = vtol_alt_m * math.tan(dx_norm * hfov_rad / 2.0)
    dy_m = vtol_alt_m * math.tan(dy_norm * vfov_rad / 2.0)

    # Kamera ekseni VTOL başlığına hizalanır
    yaw_rad = math.radians(yaw_deg)
    north_m = -dy_m * math.cos(yaw_rad) - dx_m * math.sin(yaw_rad)
    east_m  = -dy_m * math.sin(yaw_rad) + dx_m * math.cos(yaw_rad)

    # Metrik ofset → derece ofset
    delta_lat = north_m / 111_320.0
    delta_lon = east_m  / (111_320.0 * math.cos(math.radians(vtol_lat)))

    return vtol_lat + delta_lat, vtol_lon + delta_lon


def generate_oval(center_lat, center_lon, semi_major_m, semi_minor_m, num_points, altitude):
    """Elips üzerinde eşit aralıklı waypoint listesi üretir."""
    dlat = 1.0 / 111_320.0
    dlon = 1.0 / (111_320.0 * math.cos(math.radians(center_lat)))
    wps  = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        lat   = center_lat + math.sin(angle) * semi_major_m * dlat
        lon   = center_lon + math.cos(angle) * semi_minor_m * dlon
        wps.append((lat, lon, altitude))
    return wps


# ═══════════════════════════════════════════════════════════════════════════════
#  GAZEBO KAMERA ABONESİ
# ═══════════════════════════════════════════════════════════════════════════════

class GazeboCamera:
    """
    gz.transport13 üzerinden Gazebo kamera görüntüsü alır.
    Gelen her frame OpenCV BGR formatında saklanır.
    Thread-safe: get_frame() herhangi bir thread'den çağrılabilir.
    """

    def __init__(self, topic=CAMERA_TOPIC):
        self.topic         = topic
        self._frame        = None
        self._lock         = threading.Lock()
        self._total_frames = 0
        self._node         = None
        self._sub          = None   # subscription referansı – GC'den korur
        self._running      = False

    def start(self):
        """Gazebo kamera topic'ine abone olur."""
        if not GZ_AVAILABLE:
            print("[Kamera] gz.transport13 yok – kamera başlatılamadı.")
            return False
        try:
            self._node = GzNode()
            # Dönüş değeri saklanmazsa Python GC subscription'ı siler → frame gelmez
            self._sub  = self._node.subscribe(GzImage, self.topic, self._callback)
            self._running = True
            print(f"[Kamera] Abone olundu: {self.topic}")
            # Tanılama: 3 saniye içinde ilk frame gelip gelmediğini kontrol et
            threading.Thread(target=self._check_first_frame, daemon=True).start()
            return True
        except Exception as exc:
            print(f"[Kamera] Başlatma hatası: {exc}")
            return False

    def _check_first_frame(self, timeout=5):
        """Başlangıçta frame gelip gelmediğini kontrol eder, gelmezse uyarır."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._total_frames > 0:
                print(f"[Kamera] İlk frame alındı (toplam: {self._total_frames})")
                return
            time.sleep(0.5)
        print(f"[Kamera] UYARI: {timeout}s içinde frame gelmedi. "
              f"Gazebo çalışıyor mu? Topic: {self.topic}")

    def _callback(self, msg):
        """Gazebo'dan gelen protobuf Image mesajını OpenCV frame'e çevirir."""
        try:
            w    = msg.width
            h    = msg.height
            data = np.frombuffer(msg.data, dtype=np.uint8)

            # PixelFormatType değerleri: RGB_INT8=3, BGR_INT8=4
            if msg.pixel_format_type == 3:        # RGB → BGR
                frame = data.reshape((h, w, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif msg.pixel_format_type == 4:      # Zaten BGR
                frame = data.reshape((h, w, 3))
            else:                                  # Bilinmeyen – ilk 3 kanalı al
                channels = len(data) // (h * w)
                frame = data.reshape((h, w, channels))[:, :, :3]

            with self._lock:
                self._frame        = frame.copy()
                self._total_frames += 1
        except Exception as exc:
            print(f"[Kamera] Frame dönüşüm hatası: {exc}")

    def get_frame(self):
        """
        Son alınan frame'in kopyasını döndürür.
        Henüz frame gelmemişse None döner.
        """
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    @property
    def frame_count(self):
        """Toplam alınan frame sayısı."""
        return self._total_frames

    def stop(self):
        self._running = False


# ═══════════════════════════════════════════════════════════════════════════════
#  YOLOv8 TESPİT MOTORU
# ═══════════════════════════════════════════════════════════════════════════════

class PersonDetector:
    """
    YOLOv8 kullanarak görüntüde yalnızca 'person' (class 0) sınıfını arar.
    YOLO_AVAILABLE=False ise tüm çağrılar boş liste döndürür.
    """

    def __init__(self, model_path=YOLO_MODEL, conf_threshold=CONFIRM_CONF):
        self._model          = None
        self._conf_threshold = conf_threshold

        if not YOLO_AVAILABLE:
            print("[YOLO] ultralytics yok – tespit devre dışı.")
            return

        try:
            self._model = YOLO(model_path)
            print(f"[YOLO] Model yüklendi: {model_path}")
        except Exception as exc:
            print(f"[YOLO] Model yükleme hatası: {exc}")

    def detect(self, frame):
        """
        frame   : OpenCV BGR görüntü (numpy ndarray)
        Döndürür: [{"confidence": float, "bbox": [x1,y1,x2,y2],
                     "cx": int, "cy": int}, ...]
        Sadece "person" tespitleri ve eşik üstü güven skorları döndürülür.
        """
        if self._model is None or frame is None:
            return []

        found = []
        try:
            results = self._model(frame, verbose=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if int(box.cls[0]) != PERSON_CLASS_ID:
                        continue                        # yalnızca person
                    conf = float(box.conf[0])
                    if conf < self._conf_threshold:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    found.append({
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "cx": (x1 + x2) // 2,
                        "cy": (y1 + y2) // 2,
                    })
        except Exception as exc:
            print(f"[YOLO] Çıkarım hatası: {exc}")

        return found


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA VTOL KONTROLCÜSÜ
# ═══════════════════════════════════════════════════════════════════════════════

class VTOLController:
    """
    VTOL İHA görev yöneticisi.

    Akış:
        connect() → kalkış → görüntü thread'i → oval tarama
        → tespit onaylandığında hover + drone'a koordinat gönder → RTL
    """

    def __init__(self):
        self.vehicle         = None
        self.camera          = GazeboCamera()
        self.detector        = PersonDetector()
        self.mission_active  = True

        # Görüntü işleme durumu (vision_loop thread'inde güncellenir)
        self._confirm_count  = 0
        self._target_found   = False

        # Tespit koordinatları (thread-safe)
        self._target_lock    = threading.Lock()
        self._target_lat     = None
        self._target_lon     = None

        # Oval waypoint listesi
        self.waypoints = generate_oval(
            SCAN_CENTER[0], SCAN_CENTER[1],
            SCAN_SEMI_MAJOR, SCAN_SEMI_MINOR,
            SCAN_NUM_POINTS, SCAN_ALT,
        )

    # ─────────────────────────────────────────────────────────────────────────
    #  BAĞLANTI
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self, retries=5, retry_delay=3):
        """MAVLink bağlantısı kurar; başarısız olursa belirtilen sayıda tekrar dener."""
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

    # ─────────────────────────────────────────────────────────────────────────
    #  MAVLink YARDIMCILARI
    # ─────────────────────────────────────────────────────────────────────────

    def _set_mode(self, mode_name, timeout=10):
        """Belirtilen uçuş modunu aktif eder, ACK alana kadar bekler."""
        mode_id = self.vehicle.mode_mapping().get(mode_name)
        if mode_id is None:
            print(f"[İHA] Bilinmeyen mod: {mode_name}")
            return False

        self.vehicle.mav.set_mode_send(
            self.vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id,
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            ack = self.vehicle.recv_match(type="COMMAND_ACK", blocking=True, timeout=2)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"[İHA] Mod → {mode_name}")
                return True
        print(f"[İHA] Mod değiştirme onaylanamadı: {mode_name}")
        return False

    def _arm(self):
        """Motorları ARM eder ve onay bekler."""
        print("[İHA] ARM ediliyor...")
        self.vehicle.arducopter_arm()
        self.vehicle.motors_armed_wait()
        print("[İHA] ARM tamamlandı.")

    def _takeoff(self, altitude):
        """Kalkış komutu gönderir, hedef irtifanın %95'ine ulaşana kadar bekler."""
        print(f"[İHA] TAKEOFF → {altitude} m")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude,
        )
        while True:
            msg = self.vehicle.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
            if msg is None:
                print("[İHA] GPS verisi alınamıyor – bekleniyor...")
                continue
            cur_alt = msg.relative_alt / 1000.0
            print(f"[İHA] İrtifa: {cur_alt:.1f} m / {altitude} m", end="\r")
            if cur_alt >= altitude * 0.95:
                print(f"\n[İHA] {altitude} m irtifasına ulaşıldı.")
                return
            time.sleep(0.5)

    def _get_position(self):
        """
        GLOBAL_POSITION_INT mesajından anlık konum döndürür.
        Döndürür: (lat, lon, alt_m, yaw_deg) – başarısızsa (None, None, None, None)
        """
        msg = self.vehicle.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg is None:
            return None, None, None, None
        lat     = msg.lat / 1e7
        lon     = msg.lon / 1e7
        alt_m   = msg.relative_alt / 1000.0
        yaw_deg = msg.hdg / 100.0   # centi-derece → derece
        return lat, lon, alt_m, yaw_deg

    def _goto(self, lat, lon, alt):
        """GUIDED modda belirtilen GPS noktasına git komutu gönderir."""
        self.vehicle.mav.set_position_target_global_int_send(
            0,
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_1111_1111_1000,   # yalnızca konum (hız/ivme yoksay)
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )

    def _set_speed(self, speed_ms):
        """Araç yatay seyir hızını m/s cinsinden ayarlar."""
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0, 1, speed_ms, -1, 0, 0, 0, 0,
        )

    def _hover(self):
        """Mevcut GPS konumunda askıda kal (GUIDED modda yerinde dur)."""
        lat, lon, alt, _ = self._get_position()
        if lat is not None:
            self._goto(lat, lon, alt)

    # ─────────────────────────────────────────────────────────────────────────
    #  DRONE'A KOORDİNAT GÖNDER
    # ─────────────────────────────────────────────────────────────────────────

    def _send_target_to_drone(self, lat, lon, alt=25.0):
        """
        Hedef koordinatı hem ekrana yazdırır hem de UDP soket üzerinden
        drone.py'a JSON formatında gönderir.
        """
        print("\n" + "═" * 60)
        print("  [İHA] HEDEF TESPİT EDİLDİ – ÇIKTI")
        print("═" * 60)
        print(f"  latitude  : {lat:.7f}")
        print(f"  longitude : {lon:.7f}")
        print(f"  altitude  : {alt:.1f} m (drone seyir irtifası)")
        print("═" * 60)

        payload = json.dumps({"cmd": "GOTO", "lat": lat, "lon": lon, "alt": alt})
        for attempt in range(3):
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(payload.encode(), (DRONE_IP, DRONE_MSG_PORT))
                sock.close()
                print(f"[İHA→Drone] Hedef koordinat gönderildi "
                      f"({DRONE_IP}:{DRONE_MSG_PORT}).")
                return
            except Exception as exc:
                print(f"[İHA→Drone] Soket hatası (deneme {attempt+1}/3): {exc}")
                time.sleep(1)
        print("[İHA→Drone] Koordinat gönderilemedi.")

    # ─────────────────────────────────────────────────────────────────────────
    #  GÖRÜNTÜ İŞLEME THREAD'İ
    # ─────────────────────────────────────────────────────────────────────────

    def _vision_loop(self):
        """
        Gazebo kamerasından sürekli frame alır ve YOLOv8 ile person tespiti yapar.

        Tespit mantığı:
        - Her frame'de person algılanırsa sayaç artar.
        - Ardışık CONFIRM_FRAMES frame boyunca tespit varsa hedef onaylanır.
        - Tek frame kaçırılırsa sayaç sıfırlanır (isteğe göre gevşetilebilir).
        - Tespit onaylandıktan sonra döngü durur (thread arka planda kapanır).
        """
        print("[Görüntü] YOLOv8 işleme döngüsü başladı.")
        prev_frame_count = 0
        no_frame_warn_ts = time.time()

        while self.mission_active and not self._target_found:
            # Kamera bağlı değil veya frame gelmiyorsa bekle
            if self.camera.frame_count == prev_frame_count:
                # Her 10 saniyede bir "frame yok" uyarısı bas
                if time.time() - no_frame_warn_ts >= 10:
                    print(f"[Görüntü] Bekleniyor – frame sayısı: "
                          f"{self.camera.frame_count}  "
                          f"(kamera topic: {CAMERA_TOPIC})")
                    no_frame_warn_ts = time.time()
                time.sleep(0.05)
                continue
            no_frame_warn_ts = time.time()

            frame = self.camera.get_frame()
            prev_frame_count = self.camera.frame_count

            if frame is None:
                time.sleep(0.1)
                continue

            # YOLOv8 çıkarımı – sadece "person" sınıfı
            detections = self.detector.detect(frame)

            if not detections:
                self._confirm_count = 0   # ardışık sayaç sıfırla
                continue

            # En yüksek güven skorlu tespiti seç
            best      = max(detections, key=lambda d: d["confidence"])
            conf      = best["confidence"]
            cx, cy    = best["cx"], best["cy"]
            h, w      = frame.shape[:2]

            self._confirm_count += 1
            print(f"[Görüntü] Tespit #{self._confirm_count}/{CONFIRM_FRAMES}  "
                  f"conf={conf:.2f}  piksel=({cx},{cy})", end="\r")

            if self._confirm_count >= CONFIRM_FRAMES:
                # Hedefi GPS'e çevir
                vtol_lat, vtol_lon, vtol_alt, vtol_yaw = self._get_position()

                if vtol_lat is None:
                    print("\n[Görüntü] GPS verisi yok – tespit yeniden başlıyor.")
                    self._confirm_count = 0
                    continue

                tgt_lat, tgt_lon = pixel_to_gps(
                    cx, cy, w, h, vtol_lat, vtol_lon, vtol_alt, vtol_yaw)

                print(f"\n[Görüntü] {CONFIRM_FRAMES} frame doğrulandı! "
                      f"conf={conf:.2f}  "
                      f"VTOL=({vtol_lat:.5f},{vtol_lon:.5f})  "
                      f"Hedef=({tgt_lat:.6f},{tgt_lon:.6f})")

                with self._target_lock:
                    self._target_lat   = tgt_lat
                    self._target_lon   = tgt_lon
                    self._target_found = True

                self._confirm_count = 0

            time.sleep(0.05)   # ~20 FPS hedeflenir

        print("[Görüntü] İşleme döngüsü sonlandı.")

    # ─────────────────────────────────────────────────────────────────────────
    #  ALAN TARAMA DÖNGÜSÜ
    # ─────────────────────────────────────────────────────────────────────────

    def _scan_loop(self):
        """
        Oval waypoint'leri sırayla gezerek alan taraması yapar.

        SCAN_REPEAT = 0  → insan tespit edilene kadar sonsuz döngü
        SCAN_REPEAT > 0  → belirlenen tur sayısı tamamlanana kadar

        Her waypoint'e giderken _target_found bayrağı kontrol edilir;
        tespit onaylandığında döngü kırılır ve tespit akışına geçilir.
        """
        self._set_mode("GUIDED")
        self._set_speed(SCAN_SPEED)

        total_wps = len(self.waypoints)
        tur       = 0
        print(f"[İHA] Alan taraması başlıyor – "
              f"{total_wps} waypoint, merkez={SCAN_CENTER}, irtifa={SCAN_ALT}m")

        while self.mission_active:
            # Tespit onaylandıysa taramayı kes
            if self._target_found:
                break

            tur += 1
            if SCAN_REPEAT > 0 and tur > SCAN_REPEAT:
                print(f"[İHA] {SCAN_REPEAT} tur tamamlandı – tespit yapılamadı.")
                break

            print(f"\n[İHA] ── TUR {tur} ──")

            for wp_idx, (wp_lat, wp_lon, wp_alt) in enumerate(self.waypoints):
                if not self.mission_active or self._target_found:
                    break

                self._goto(wp_lat, wp_lon, wp_alt)
                print(f"[İHA] WP {wp_idx+1:2d}/{total_wps} "
                      f"→ ({wp_lat:.5f}, {wp_lon:.5f})", end="  ")

                # Waypoint'e ulaşana kadar bekle (5m tolerans)
                while self.mission_active and not self._target_found:
                    lat, lon, _, _ = self._get_position()
                    if lat is None:
                        time.sleep(0.5)
                        continue
                    if haversine(lat, lon, wp_lat, wp_lon) < 5.0:
                        print("✓")
                        break
                    time.sleep(1.0)

        # Tespit varsa işle; yoksa RTL
        if self._target_found:
            self._handle_detection()
        else:
            self._rtl()

    # ─────────────────────────────────────────────────────────────────────────
    #  TESPİT SONRASI AKIŞ
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_detection(self):
        """
        İnsan tespit onaylandıktan sonra:
        1. GUIDED moda geç ve hover yap.
        2. Hedef koordinatı ekrana yazdır.
        3. Drone'a UDP ile gönder.
        4. RTL başlat.
        """
        with self._target_lock:
            tgt_lat = self._target_lat
            tgt_lon = self._target_lon

        print("\n[İHA] İnsan tespit onaylandı – GUIDED / Hover moduna geçiliyor...")
        self._set_mode("GUIDED")
        self._hover()
        time.sleep(3)

        # Koordinatı çıktı ver ve drone'a ilet
        self._send_target_to_drone(tgt_lat, tgt_lon, alt=25.0)

        # Drone'un koordinatı alması için bekle, sonra dön
        time.sleep(5)
        self._rtl()

    # ─────────────────────────────────────────────────────────────────────────
    #  GÖREV SONU
    # ─────────────────────────────────────────────────────────────────────────

    def _rtl(self):
        print("[İHA] RTL başlatılıyor...")
        try:
            self._set_mode("RTL")
        except Exception as exc:
            print(f"[İHA] RTL hatası: {exc}")
        self.mission_active = False

    # ─────────────────────────────────────────────────────────────────────────
    #  ANA GİRİŞ NOKTASI
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        print("\n" + "═" * 60)
        print("  VTOL İHA – OTONOM GÖREV SİSTEMİ")
        print("═" * 60)
        print(f"  MAVLink bağlantısı : {VTOL_CONNECTION}")
        print(f"  Kamera topic       : {CAMERA_TOPIC}")
        print(f"  YOLO modeli        : {YOLO_MODEL}")
        print(f"  Onay frame sayısı  : {CONFIRM_FRAMES}")
        print(f"  Güven eşiği        : {CONFIRM_CONF}")
        print(f"  Tarama irtifası    : {SCAN_ALT} m")
        print(f"  Oval merkezi       : {SCAN_CENTER}")
        print("═" * 60 + "\n")

        # 1. MAVLink bağlantısı
        if not self.connect():
            print("[İHA] MAVLink bağlantısı kurulamadı – çıkılıyor.")
            return

        # 2. Gazebo kamera başlat
        cam_ok = self.camera.start()
        if not cam_ok:
            print("[İHA] Gazebo kamerası başlatılamadı. "
                  "Görüntü işleme çalışmayacak.")

        # 3. Görüntü işleme thread'ini arka planda başlat
        vision_th = threading.Thread(target=self._vision_loop, daemon=True)
        vision_th.start()

        # 4. Kalkış
        self._set_mode("GUIDED")
        time.sleep(1)
        self._arm()
        self._takeoff(TAKEOFF_ALT)

        # 5. Alan taraması + tespit döngüsü
        try:
            self._scan_loop()
        except KeyboardInterrupt:
            print("\n[İHA] Kullanıcı tarafından durduruldu.")
            self.mission_active = False
            self._rtl()
        finally:
            self.mission_active = False
            self.camera.stop()
            print("[İHA] Script sonlandı.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    vtol = VTOLController()
    vtol.run()
