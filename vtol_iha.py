"""
VTOL İHA Kontrol Scripti
========================
Araç   : alti_transition_quad
SysID  : 1
UDP    : 14550
Görev  : Oval alanda uç → 2 tespit noktasında insan bul → drone bırak
"""

import time
import math
import threading
import socket
import json
from pymavlink import mavutil

# ═══════════════════════════════════════════════════════════════
#  ▼▼▼  KULLANICI TARAFINDAN DÜZENLENECEk BÖLÜM  ▼▼▼
# ═══════════════════════════════════════════════════════════════

# ── Oval uçuş parametreleri ──────────────────────────────────
OVAL_CENTER      = (47.3982419, 8.5465938)   # Ovalin merkezi (lat, lon)
OVAL_SEMI_MAJOR  = 120     # metre – kuzey-güney yarı eksen (uzun taraf)
OVAL_SEMI_MINOR  = 70      # metre – doğu-batı yarı eksen (kısa taraf)
OVAL_NUM_POINTS  = 24      # oval üzerindeki waypoint sayısı
OVAL_REPEAT      = 3       # kaç tur atacak (0 = sonsuz)
SCAN_ALT         = 50.0    # metre – tarama irtifası

# ── Tespit noktaları (2 adet insan) ─────────────────────────
#    VTOL bu noktalara DETECTION_RADIUS içine girince YOLO tetiklenir
HUMAN_POINT_1    = (47.3988000, 8.5462000)   # 1. insan konumu
HUMAN_POINT_2    = (47.3976000, 8.5472000)   # 2. insan konumu
DETECTION_RADIUS = 20      # metre – bu mesafe içinde tespit başlar

# ── Kalkış ──────────────────────────────────────────────────
TAKEOFF_ALT      = 50.0    # metre

# ═══════════════════════════════════════════════════════════════
#  SİSTEM YAPILANDIRMASI (normalde dokunma)
# ═══════════════════════════════════════════════════════════════

VTOL_CONNECTION  = "udp:127.0.0.1:14550"
VTOL_SYSID       = 1
DRONE_IP         = "127.0.0.1"
DRONE_MSG_PORT   = 6000

CONFIRM_FRAMES   = 5       # art arda kaç frame tespit gerekli
CONFIRM_CONF     = 0.70    # minimum güven eşiği
HOVER_SPEED      = 2.0     # m/s – tespit anında yavaşlama


# ═══════════════════════════════════════════════════════════════
#  OVAL WAYPOINT ÜRETICI
# ═══════════════════════════════════════════════════════════════

def generate_oval(center_lat, center_lon, semi_major_m, semi_minor_m,
                  num_points, altitude, clockwise=True):
    """
    Elips üzerinde eşit aralıklı waypoint'ler üretir.
    Coğrafi koordinata dönüşüm için küçük açı yaklaşımı kullanılır.

    center_lat/lon : ovalin merkezi
    semi_major_m   : kuzey-güney yarı eksen (metre)
    semi_minor_m   : doğu-batı yarı eksen (metre)
    num_points     : kaç waypoint
    altitude       : uçuş irtifası
    """
    DEG_PER_METER_LAT = 1.0 / 111320.0
    DEG_PER_METER_LON = 1.0 / (111320.0 * math.cos(math.radians(center_lat)))

    waypoints = []
    angles = [2 * math.pi * i / num_points for i in range(num_points)]
    if not clockwise:
        angles = angles[::-1]

    for angle in angles:
        lat = center_lat + math.sin(angle) * semi_major_m * DEG_PER_METER_LAT
        lon = center_lon + math.cos(angle) * semi_minor_m * DEG_PER_METER_LON
        waypoints.append((lat, lon, altitude))

    return waypoints


# ═══════════════════════════════════════════════════════════════
#  YOLO TESPİT SİMÜLATÖRÜ
# ═══════════════════════════════════════════════════════════════

class YOLODetector:
    """
    VTOL'ün mevcut konumunu alır; tespit noktalarına yaklaşınca
    yüksek güvenli tespit döndürür.

    Gerçek entegrasyon:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        results = model(frame)
        dets = [{"confidence": float(r.boxes.conf[i]),
                 "bbox": r.boxes.xyxy[i].tolist()}
                for r in results for i in range(len(r.boxes))]
    """

    def __init__(self):
        self._human_points = [HUMAN_POINT_1, HUMAN_POINT_2]

    def detect(self, vtol_lat, vtol_lon):
        """
        vtol_lat/lon : VTOL'ün anlık GPS konumu
        Dönüş       : [{"confidence": float, "point_idx": int}] veya []
        """
        detections = []
        for idx, (hlat, hlon) in enumerate(self._human_points):
            dist = _haversine(vtol_lat, vtol_lon, hlat, hlon)
            if dist <= DETECTION_RADIUS:
                # Mesafeye göre güven değeri: yakın → yüksek
                conf = round(min(0.99, 0.70 + (1 - dist / DETECTION_RADIUS) * 0.29), 2)
                detections.append({"confidence": conf,
                                   "point_idx": idx,
                                   "point_loc": (hlat, hlon)})
        return detections


# ═══════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi  = math.radians(lat2 - lat1)
    dlam  = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def send_target_to_drone(lat, lon, alt=30.0, point_idx=0):
    """VTOL → Drone: hedef koordinat gönder."""
    payload = json.dumps({
        "cmd"      : "GOTO",
        "lat"      : lat,
        "lon"      : lon,
        "alt"      : alt,
        "point_idx": point_idx
    })
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.sendto(payload.encode(), (DRONE_IP, DRONE_MSG_PORT))
        sock.close()
        print(f"[VTOL→Drone] Hedef #{point_idx+1} gönderildi: "
              f"lat={lat:.6f} lon={lon:.6f}")
    except Exception as e:
        print(f"[VTOL→Drone] Soket hatası: {e}")


# ═══════════════════════════════════════════════════════════════
#  ANA VTOL SINIFI
# ═══════════════════════════════════════════════════════════════

class VTOLController:

    def __init__(self):
        self.vehicle         = None
        self.detector        = YOLODetector()
        self.mission_active  = True

        # Tespit durumu (2 nokta için ayrı sayaç)
        self._confirm_buf    = {0: [], 1: []}    # {point_idx: [conf, ...]}
        self._detected       = {0: False, 1: False}  # bir kez tespit edildiyse True

        # Waypoint listesi
        self.waypoints       = generate_oval(
            OVAL_CENTER[0], OVAL_CENTER[1],
            OVAL_SEMI_MAJOR, OVAL_SEMI_MINOR,
            OVAL_NUM_POINTS, SCAN_ALT
        )

        # Aktif hedef kuyruğu (thread-safe)
        self._pending_targets = []
        self._pending_lock    = threading.Lock()

    # ──────────────────────────────────────────
    #  BAĞLANTI
    # ──────────────────────────────────────────
    def connect(self, retries=5):
        for attempt in range(1, retries + 1):
            try:
                print(f"[VTOL] Bağlanılıyor... ({attempt}/{retries})")
                self.vehicle = mavutil.mavlink_connection(
                    VTOL_CONNECTION,
                    source_system=255,
                    target_system=VTOL_SYSID
                )
                self.vehicle.wait_heartbeat(timeout=10)
                print(f"[VTOL] Heartbeat – SysID:{self.vehicle.target_system}")
                return True
            except Exception as e:
                print(f"[VTOL] Bağlantı hatası: {e}")
                time.sleep(3)
        return False

    # ──────────────────────────────────────────
    #  MAVLink YARDIMCILARI
    # ──────────────────────────────────────────
    def _set_mode(self, mode_name):
        mode_id = self.vehicle.mode_mapping().get(mode_name)
        if mode_id is None:
            print(f"[VTOL] Bilinmeyen mod: {mode_name}")
            return False
        self.vehicle.mav.set_mode_send(
            self.vehicle.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mode_id
        )
        for _ in range(10):
            ack = self.vehicle.recv_match(type="COMMAND_ACK",
                                          blocking=True, timeout=2)
            if ack and ack.result == mavutil.mavlink.MAV_RESULT_ACCEPTED:
                print(f"[VTOL] Mod → {mode_name}")
                return True
        print(f"[VTOL] Mod onaylanamadı: {mode_name}")
        return False

    def _arm(self):
        print("[VTOL] ARM...")
        self.vehicle.arducopter_arm()
        self.vehicle.motors_armed_wait()
        print("[VTOL] ARM tamamlandı.")

    def _takeoff(self, altitude):
        print(f"[VTOL] TAKEOFF → {altitude}m")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
            0, 0, 0, 0, 0, 0, 0, altitude
        )
        while True:
            msg = self.vehicle.recv_match(
                type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
            if msg:
                cur = msg.relative_alt / 1000.0
                print(f"[VTOL] İrtifa: {cur:.1f}m / {altitude}m", end="\r")
                if cur >= altitude * 0.95:
                    print(f"\n[VTOL] {altitude}m irtifaya ulaşıldı.")
                    return
            time.sleep(0.5)

    def _get_position(self):
        msg = self.vehicle.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg:
            return (msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0)
        return None, None, None

    def _goto(self, lat, lon, alt):
        self.vehicle.mav.set_position_target_global_int_send(
            0,
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000111111111000,
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0, 0, 0, 0, 0, 0
        )

    def _set_speed(self, spd):
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0, 1, spd, -1, 0, 0, 0, 0
        )

    # ──────────────────────────────────────────
    #  SERVO – DRONE BIRAKMA
    # ──────────────────────────────────────────
    def _release_drone(self, channel=6, pwm_open=2000, pwm_neutral=1500):
        """Servo ile drone tutucu kilidini açar."""
        print("[VTOL] Drone bırakma – servo açılıyor...")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, channel, pwm_open, 0, 0, 0, 0, 0
        )
        time.sleep(1.5)
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, channel, pwm_neutral, 0, 0, 0, 0, 0
        )
        print("[VTOL] Drone serbest bırakıldı.")

    # ──────────────────────────────────────────
    #  TESPİT THREAD'İ
    # ──────────────────────────────────────────
    def _detection_loop(self):
        """
        VTOL konumunu sürekli okur ve tespit simülatörünü çalıştırır.
        CONFIRM_FRAMES art arda frame sonunda hedefi kuyruğa ekler.
        """
        while self.mission_active:
            lat, lon, _ = self._get_position()
            if lat is None:
                time.sleep(0.2)
                continue

            detections = self.detector.detect(lat, lon)

            for det in detections:
                idx = det["point_idx"]
                if self._detected[idx]:
                    # Bu nokta zaten işlendi
                    continue
                conf = det["confidence"]
                if conf >= CONFIRM_CONF:
                    self._confirm_buf[idx].append(conf)
                else:
                    self._confirm_buf[idx].clear()

                if len(self._confirm_buf[idx]) >= CONFIRM_FRAMES:
                    avg = sum(self._confirm_buf[idx][-CONFIRM_FRAMES:]) / CONFIRM_FRAMES
                    hlat, hlon = det["point_loc"]
                    print(f"\n[VTOL][TESPİT #{idx+1}] "
                          f"İnsan doğrulandı! Güven:{avg:.2f} "
                          f"Konum:({hlat:.6f},{hlon:.6f})")
                    self._detected[idx] = True
                    self._confirm_buf[idx].clear()
                    with self._pending_lock:
                        self._pending_targets.append({
                            "lat": hlat, "lon": hlon, "idx": idx
                        })

            time.sleep(0.1)

    # ──────────────────────────────────────────
    #  TESPİT İŞLEME
    # ──────────────────────────────────────────
    def _handle_detection(self, target):
        lat, lon, idx = target["lat"], target["lon"], target["idx"]
        print(f"\n[VTOL] Tespit #{idx+1} işleniyor...")

        # 1. Hızı düşür – hover
        self._set_speed(HOVER_SPEED)
        cur_lat, cur_lon, cur_alt = self._get_position()
        if cur_lat:
            self._goto(cur_lat, cur_lon, cur_alt)
        time.sleep(3)

        # 2. Drone bırak
        self._release_drone()
        time.sleep(2)

        # 3. Drone'a koordinat gönder
        send_target_to_drone(lat, lon, alt=30.0, point_idx=idx)

        # 4. Normal hıza dön – taramaya devam
        self._set_speed(10.0)
        print(f"[VTOL] Tespit #{idx+1} tamamlandı, oval turuna devam ediyor.")

    # ──────────────────────────────────────────
    #  OVAL TARAMA DÖNGÜSÜ
    # ──────────────────────────────────────────
    def _scan_loop(self):
        """
        Oval waypoint'leri tur sayısı kadar döner.
        Her waypoint'e giderken tespit kuyruğunu kontrol eder.
        İki tespit de tamamlandıktan sonra RTL.
        """
        self._set_mode("GUIDED")
        self._set_speed(10.0)
        total_wps = len(self.waypoints)
        detections_done = 0

        print(f"[VTOL] Oval tarama başlıyor – {total_wps} waypoint, "
              f"{OVAL_REPEAT} tur")
        print(f"[VTOL] Tespit noktası 1: {HUMAN_POINT_1}")
        print(f"[VTOL] Tespit noktası 2: {HUMAN_POINT_2}")

        for tur in range(1, OVAL_REPEAT + 1):
            print(f"\n[VTOL] ── TUR {tur}/{OVAL_REPEAT} ──")
            wp_idx = 0

            while wp_idx < total_wps and self.mission_active:
                wp = self.waypoints[wp_idx]
                print(f"[VTOL] WP {wp_idx+1}/{total_wps} "
                      f"lat:{wp[0]:.5f} lon:{wp[1]:.5f}", end="  ")

                self._goto(wp[0], wp[1], wp[2])

                # Waypoint'e ulaşana kadar bekle + tespit kontrol et
                while self.mission_active:
                    cur_lat, cur_lon, _ = self._get_position()
                    if cur_lat is None:
                        time.sleep(0.5)
                        continue

                    dist = _haversine(cur_lat, cur_lon, wp[0], wp[1])

                    # Tespit kuyruğunda bekleyen var mı?
                    with self._pending_lock:
                        pending = list(self._pending_targets)
                        self._pending_targets.clear()

                    for tgt in pending:
                        self._handle_detection(tgt)
                        detections_done += 1
                        # İşlemden sonra aynı WP'ye devam et
                        self._set_mode("GUIDED")
                        self._goto(wp[0], wp[1], wp[2])

                    if dist < 5.0:
                        print(f"✓")
                        break

                    time.sleep(1)

                wp_idx += 1

            # İki tespit de tamamlandıysa erken çık
            if detections_done >= 2:
                print("\n[VTOL] Her iki tespit tamamlandı – görev bitiyor.")
                break

        self._rtl()

    # ──────────────────────────────────────────
    #  GÖREV SONU
    # ──────────────────────────────────────────
    def _rtl(self):
        print("[VTOL] RTL başlatılıyor...")
        self._set_mode("RTL")

    # ──────────────────────────────────────────
    #  KONUM ÖZET YAZDIR
    # ──────────────────────────────────────────
    def _print_mission_info(self):
        print("\n" + "═"*55)
        print("  VTOL GÖREV PLANI")
        print("═"*55)
        print(f"  Oval Merkezi   : {OVAL_CENTER[0]:.6f}, {OVAL_CENTER[1]:.6f}")
        print(f"  Yarı eksen     : {OVAL_SEMI_MAJOR}m (K-G) × "
              f"{OVAL_SEMI_MINOR}m (D-B)")
        print(f"  Waypoint sayısı: {OVAL_NUM_POINTS}  |  Tur: {OVAL_REPEAT}")
        print(f"  Tarama irtifası: {SCAN_ALT}m")
        print(f"  Tespit yarıçapı: {DETECTION_RADIUS}m")
        print(f"  İnsan Noktası 1: {HUMAN_POINT_1[0]:.6f}, {HUMAN_POINT_1[1]:.6f}")
        print(f"  İnsan Noktası 2: {HUMAN_POINT_2[0]:.6f}, {HUMAN_POINT_2[1]:.6f}")
        print("═"*55 + "\n")

    # ──────────────────────────────────────────
    #  ANA AKIŞ
    # ──────────────────────────────────────────
    def run(self):
        self._print_mission_info()

        if not self.connect():
            print("[VTOL] Bağlantı kurulamadı.")
            return

        self._set_mode("GUIDED")
        time.sleep(1)
        self._arm()
        self._takeoff(TAKEOFF_ALT)

        # Tespit thread'ini başlat
        det_th = threading.Thread(target=self._detection_loop, daemon=True)
        det_th.start()

        try:
            self._scan_loop()
        except KeyboardInterrupt:
            print("\n[VTOL] Kullanıcı durdurdu.")
            self.mission_active = False
            self._rtl()
        finally:
            self.mission_active = False
            print("[VTOL] Script sonlandı.")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    vtol = VTOLController()
    vtol.run()
