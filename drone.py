"""
Drone (Multikopter) Kontrol Scripti
=====================================
Araç    : iris_with_ardupilot
SysID   : 2
UDP     : 14560
Görev   : iha.py'den gelen GPS koordinatına otonom git
          → Servolu payload bırak → Smart Idle Mode

Bağımlılıklar:
    pip install pymavlink
"""

import time
import math
import threading
import socket
import json
from pymavlink import mavutil


# ═══════════════════════════════════════════════════════════════════════════════
#  KULLANICI TARAFINDAN DÜZENLENECEk BÖLÜM
# ═══════════════════════════════════════════════════════════════════════════════

# Görev tamamlandıktan sonra dönülecek güvenli standby noktası
STANDBY_LAT = 47.3977419
STANDBY_LON = 8.5455938
STANDBY_ALT = 25.0        # metre

# ═══════════════════════════════════════════════════════════════════════════════
#  SİSTEM YAPILANDIRMASI (normalde dokunmayın)
# ═══════════════════════════════════════════════════════════════════════════════

DRONE_CONNECTION = "udp:127.0.0.1:14560"
DRONE_SYSID      = 2
MSG_LISTEN_PORT  = 6000       # iha.py'den hedef alınacak UDP portu

CRUISE_ALT       = 25.0       # metre – seyir irtifası
DELIVERY_ALT     = 5.0        # metre – payload bırakma irtifası
ARRIVAL_DIST     = 3.0        # metre – hedefe bu kadar yaklaşınca "ulaşıldı" sayılır

CRUISE_SPEED     = 8.0        # m/s
IDLE_SPEED       = 2.0        # m/s – smart idle sürüklenme hızı
IDLE_DRIFT_DIST  = 20.0       # metre – her idle adımında sürükleme mesafesi
IDLE_TIMEOUT     = 30.0       # saniye – bu süre sonra standby noktasına git

PAYLOAD_SERVO_CH  = 7         # servo çıkış kanalı
PAYLOAD_PWM_OPEN  = 2000      # kilit açma PWM (µs)
PAYLOAD_PWM_CLOSE = 1000      # kilit kapama PWM (µs)


# ═══════════════════════════════════════════════════════════════════════════════
#  YARDIMCI FONKSİYONLAR
# ═══════════════════════════════════════════════════════════════════════════════

def haversine(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki mesafeyi metre cinsinden döndürür."""
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2 - lat1) / 2) ** 2
         + math.cos(phi1) * math.cos(phi2)
         * math.sin(math.radians(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def bearing(lat1, lon1, lat2, lon2):
    """İki GPS noktası arasındaki pusulayı (kuzeyden saat yönü, 0-360) döndürür."""
    dlon  = math.radians(lon2 - lon1)
    la1   = math.radians(lat1)
    la2   = math.radians(lat2)
    x     = math.sin(dlon) * math.cos(la2)
    y     = math.cos(la1) * math.sin(la2) - math.sin(la1) * math.cos(la2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def offset_gps(lat, lon, distance_m, heading_deg):
    """
    Belirtilen noktadan heading_deg yönünde distance_m uzaktaki GPS noktasını döndürür.
    Küçük mesafeler için düz yeryüzü yaklaşımı kullanılır.
    """
    dlat = distance_m * math.cos(math.radians(heading_deg)) / 111_320.0
    dlon = (distance_m * math.sin(math.radians(heading_deg))
            / (111_320.0 * math.cos(math.radians(lat))))
    return lat + dlat, lon + dlon


# ═══════════════════════════════════════════════════════════════════════════════
#  ANA DRONE KONTROLCÜSÜ
# ═══════════════════════════════════════════════════════════════════════════════

class DroneController:
    """
    Multikopter drone görev yöneticisi.

    Akış:
        connect() → UDP dinleyici başlat → hedef bekle
        → ARM → TAKEOFF → hedefe git → payload bırak
        → Smart Idle (yeni hedef yoksa standby)
    """

    def __init__(self):
        self.vehicle          = None
        self.mission_active   = True

        # Hedef kuyruğu – iha.py'den gelen koordinatlar sırayla işlenir
        self._queue           = []
        self._queue_lock      = threading.Lock()

        # Smart idle için son gidiş yönü
        self._last_bearing    = None
        self._last_task_ts    = time.time()

        # Tamamlanan teslimat sayısı
        self._deliveries_done = 0

    # ─────────────────────────────────────────────────────────────────────────
    #  BAĞLANTI
    # ─────────────────────────────────────────────────────────────────────────

    def connect(self, retries=5, retry_delay=3):
        """MAVLink bağlantısı kurar; başarısızsa belirtilen sayıda tekrar dener."""
        for attempt in range(1, retries + 1):
            try:
                print(f"[Drone] MAVLink bağlanılıyor... ({attempt}/{retries})")
                self.vehicle = mavutil.mavlink_connection(
                    DRONE_CONNECTION,
                    source_system=255,
                    target_system=DRONE_SYSID,
                )
                self.vehicle.wait_heartbeat(timeout=10)
                print(f"[Drone] Heartbeat alındı – SysID:{self.vehicle.target_system}")
                return True
            except Exception as exc:
                print(f"[Drone] Bağlantı hatası: {exc}")
                if attempt < retries:
                    time.sleep(retry_delay)
        return False

    # ─────────────────────────────────────────────────────────────────────────
    #  MAVLink YARDIMCILARI
    # ─────────────────────────────────────────────────────────────────────────

    def _set_mode(self, mode_name, timeout=10):
        """Uçuş modunu değiştirir ve ACK bekler."""
        mode_id = self.vehicle.mode_mapping().get(mode_name)
        if mode_id is None:
            print(f"[Drone] Bilinmeyen mod: {mode_name}")
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
                print(f"[Drone] Mod → {mode_name}")
                return True
        print(f"[Drone] Mod değiştirme onaylanamadı: {mode_name}")
        return False

    def _arm(self):
        """Motorları ARM eder ve onay bekler."""
        print("[Drone] ARM ediliyor...")
        self.vehicle.arducopter_arm()
        self.vehicle.motors_armed_wait()
        print("[Drone] ARM tamamlandı.")

    def _takeoff(self, altitude=CRUISE_ALT):
        """Kalkış komutunu gönderir, hedef irtifanın %90'ına ulaşana kadar bekler."""
        print(f"[Drone] TAKEOFF → {altitude} m")
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
                print("[Drone] GPS verisi yok – bekleniyor...")
                continue
            cur_alt = msg.relative_alt / 1000.0
            print(f"[Drone] İrtifa: {cur_alt:.1f} m / {altitude} m", end="\r")
            if cur_alt >= altitude * 0.90:
                print(f"\n[Drone] Seyir irtifasına ulaşıldı.")
                return
            time.sleep(0.5)

    def _get_position(self):
        """
        GLOBAL_POSITION_INT mesajından anlık konumu döndürür.
        Döndürür: (lat, lon, alt_m) – başarısızsa (None, None, None)
        """
        msg = self.vehicle.recv_match(
            type="GLOBAL_POSITION_INT", blocking=True, timeout=5)
        if msg is None:
            return None, None, None
        return msg.lat / 1e7, msg.lon / 1e7, msg.relative_alt / 1000.0

    def _goto(self, lat, lon, alt):
        """GUIDED modda belirtilen konuma git komutu gönderir."""
        self.vehicle.mav.set_position_target_global_int_send(
            0,
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_1111_1111_1000,
            int(lat * 1e7), int(lon * 1e7), alt,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )

    def _set_speed(self, speed_ms):
        """Seyir hızını m/s cinsinden ayarlar."""
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system,
            self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_CHANGE_SPEED,
            0, 1, speed_ms, -1, 0, 0, 0, 0,
        )

    def _change_altitude(self, target_alt, tolerance=1.5):
        """Mevcut yatay konumda irtifa değiştirir, hedefe ulaşana kadar bekler."""
        lat, lon, _ = self._get_position()
        if lat is None:
            return
        self._goto(lat, lon, target_alt)
        while True:
            _, _, cur_alt = self._get_position()
            if cur_alt is not None and abs(cur_alt - target_alt) < tolerance:
                return
            time.sleep(0.5)

    # ─────────────────────────────────────────────────────────────────────────
    #  PAYLOAD BIRAKMA (SERVO)
    # ─────────────────────────────────────────────────────────────────────────

    def _drop_payload(self):
        """
        Servo ile yük tutucuyu açar (PAYLOAD_PWM_OPEN), 2 saniye bekler,
        ardından kapanır (PAYLOAD_PWM_CLOSE).
        """
        print(f"[Drone] Payload bırakılıyor – "
              f"kanal {PAYLOAD_SERVO_CH}, PWM {PAYLOAD_PWM_OPEN} µs")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, PAYLOAD_SERVO_CH, PAYLOAD_PWM_OPEN, 0, 0, 0, 0, 0,
        )
        time.sleep(2.0)   # yükün düşmesi için bekle

        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, PAYLOAD_SERVO_CH, PAYLOAD_PWM_CLOSE, 0, 0, 0, 0, 0,
        )
        print("[Drone] Payload bırakıldı – servo kapatıldı.")

    # ─────────────────────────────────────────────────────────────────────────
    #  UDP HEDEF DİNLEYİCİSİ
    # ─────────────────────────────────────────────────────────────────────────

    def _socket_listener(self):
        """
        UDP port 6000'de iha.py'den JSON mesajı bekler.
        Gelen "GOTO" komutları hedef kuyruğuna eklenir.
        Thread güvenlidir (_queue_lock kullanır).
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind(("0.0.0.0", MSG_LISTEN_PORT))
            sock.settimeout(1.0)
            print(f"[Drone] UDP dinleniyor: 0.0.0.0:{MSG_LISTEN_PORT}")
        except Exception as exc:
            print(f"[Drone] Soket başlatma hatası: {exc}")
            return

        while self.mission_active:
            try:
                data, addr = sock.recvfrom(1024)
                msg = json.loads(data.decode())

                if msg.get("cmd") != "GOTO":
                    continue

                lat = float(msg["lat"])
                lon = float(msg["lon"])
                alt = float(msg.get("alt", CRUISE_ALT))

                print(f"\n[Drone] Hedef alındı: "
                      f"lat={lat:.6f}  lon={lon:.6f}  alt={alt:.1f}m")

                with self._queue_lock:
                    self._queue.append((lat, lon, alt))
                self._last_task_ts = time.time()

            except socket.timeout:
                continue
            except json.JSONDecodeError:
                print("[Drone] Geçersiz JSON formatı – mesaj atlandı.")
            except KeyError as exc:
                print(f"[Drone] Eksik alan: {exc}")
            except Exception as exc:
                print(f"[Drone] Dinleyici hatası: {exc}")

        sock.close()
        print("[Drone] UDP dinleyici kapatıldı.")

    # ─────────────────────────────────────────────────────────────────────────
    #  TESLİMAT GÖREVI
    # ─────────────────────────────────────────────────────────────────────────

    def _deliver(self, lat, lon, alt):
        """
        Tek teslimat sekansı:
        1. Seyir irtifasında hedefe git (yatay yaklaşma).
        2. Payload bırakma irtifasına in.
        3. Servo ile payload bırak.
        4. Seyir irtifasına geri çık.
        5. Teslimatı kaydet.
        """
        self._deliveries_done += 1
        n = self._deliveries_done
        print(f"\n[Drone] ── TESLİMAT #{n} BAŞLADI ──")
        print(f"[Drone] Hedef: ({lat:.6f}, {lon:.6f}, {alt:.1f}m)")

        # Son yönü kaydet (smart idle için)
        cur_lat, cur_lon, _ = self._get_position()
        if cur_lat is not None:
            self._last_bearing = bearing(cur_lat, cur_lon, lat, lon)

        # Seyir irtifasında hedefe yaklaş
        self._set_speed(CRUISE_SPEED)
        self._goto(lat, lon, alt)

        while self.mission_active:
            cur_lat, cur_lon, _ = self._get_position()
            if cur_lat is None:
                time.sleep(0.5)
                continue

            dist = haversine(cur_lat, cur_lon, lat, lon)
            print(f"[Drone] Mesafe: {dist:.1f} m", end="\r")

            if dist <= ARRIVAL_DIST:
                print()
                break
            time.sleep(0.8)

        print(f"[Drone] Hedefe ulaşıldı – irtifa azaltılıyor "
              f"({alt:.0f}m → {DELIVERY_ALT}m)...")
        self._change_altitude(DELIVERY_ALT)

        # Payload bırak
        self._drop_payload()
        time.sleep(1)

        # Seyir irtifasına geri dön
        print(f"[Drone] Seyir irtifasına yükseliyor ({DELIVERY_ALT}m → {CRUISE_ALT}m)...")
        self._change_altitude(CRUISE_ALT)

        print(f"[Drone] Teslimat #{n} tamamlandı.")
        self._last_task_ts = time.time()

    # ─────────────────────────────────────────────────────────────────────────
    #  SMART IDLE MODE
    # ─────────────────────────────────────────────────────────────────────────

    def _smart_idle(self):
        """
        Görev kuyruğu boşken akıllı bekleme modu.

        Davranış:
        - İlk IDLE_TIMEOUT saniye: son hedefe doğru IDLE_SPEED hızda yavaşça sürüklen.
        - IDLE_TIMEOUT geçince: standby noktasına git ve yeni hedef bekle.
        - Herhangi bir anda yeni hedef kuyruğa girerse hemen çık.
        """
        print("[Drone] Smart Idle modu aktif.")
        self._set_speed(IDLE_SPEED)
        idle_start = time.time()

        while self.mission_active:
            # Yeni hedef var mı?
            with self._queue_lock:
                if self._queue:
                    print("[Drone] Smart Idle → yeni hedef var, çıkılıyor.")
                    return

            elapsed = time.time() - idle_start

            if elapsed >= IDLE_TIMEOUT:
                # Standby noktasına git
                print(f"[Drone] Idle timeout ({IDLE_TIMEOUT:.0f}s) – "
                      f"standby noktasına gidiliyor: "
                      f"({STANDBY_LAT:.5f}, {STANDBY_LON:.5f})")
                self._set_speed(CRUISE_SPEED)
                self._goto(STANDBY_LAT, STANDBY_LON, STANDBY_ALT)

                # Standby'da yeni hedef bekle
                while self.mission_active:
                    with self._queue_lock:
                        if self._queue:
                            print("[Drone] Standby'dan ayrılıyor – yeni hedef.")
                            return
                    time.sleep(1.0)
                return

            # Son hedef yönünde küçük adımlarla sürüklen
            if self._last_bearing is not None:
                cur_lat, cur_lon, cur_alt = self._get_position()
                if cur_lat is not None:
                    new_lat, new_lon = offset_gps(
                        cur_lat, cur_lon, IDLE_DRIFT_DIST, self._last_bearing)
                    self._goto(new_lat, new_lon, cur_alt or CRUISE_ALT)

            time.sleep(5.0)

    # ─────────────────────────────────────────────────────────────────────────
    #  ANA GÖREV DÖNGÜSÜ
    # ─────────────────────────────────────────────────────────────────────────

    def _mission_loop(self):
        """
        İha.py'den koordinat gelene kadar smart idle modda bekler.
        Koordinat geldiğinde teslimat görevini çalıştırır.
        Görev tamamlandıktan sonra RTL.
        """
        while self.mission_active:
            target = None
            with self._queue_lock:
                if self._queue:
                    target = self._queue.pop(0)

            if target:
                lat, lon, alt = target
                self._deliver(lat, lon, alt)
                # Tek teslimat sonrası görev bitti – RTL
                print("\n[Drone] Teslimat görevi tamamlandı.")
                break
            else:
                self._smart_idle()

        self._rtl()

    # ─────────────────────────────────────────────────────────────────────────
    #  GÖREV SONU
    # ─────────────────────────────────────────────────────────────────────────

    def _rtl(self):
        print("[Drone] RTL başlatılıyor...")
        try:
            self._set_mode("RTL")
        except Exception as exc:
            print(f"[Drone] RTL hatası: {exc}")
        self.mission_active = False

    # ─────────────────────────────────────────────────────────────────────────
    #  ÖZET BİLGİ
    # ─────────────────────────────────────────────────────────────────────────

    def _print_info(self):
        print("\n" + "═" * 60)
        print("  DRONE – OTONOM TESLİMAT SİSTEMİ")
        print("═" * 60)
        print(f"  MAVLink bağlantısı : {DRONE_CONNECTION}")
        print(f"  UDP dinleme portu  : {MSG_LISTEN_PORT}")
        print(f"  Seyir irtifası     : {CRUISE_ALT} m")
        print(f"  Payload irtifası   : {DELIVERY_ALT} m")
        print(f"  Servo kanalı       : {PAYLOAD_SERVO_CH}")
        print(f"  Idle timeout       : {IDLE_TIMEOUT} s")
        print(f"  Standby noktası    : ({STANDBY_LAT:.5f}, {STANDBY_LON:.5f})")
        print("═" * 60 + "\n")

    # ─────────────────────────────────────────────────────────────────────────
    #  ANA GİRİŞ NOKTASI
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        self._print_info()

        # 1. MAVLink bağlantısı
        if not self.connect():
            print("[Drone] MAVLink bağlantısı kurulamadı – çıkılıyor.")
            return

        # 2. UDP hedef dinleyicisini arka planda başlat
        sock_th = threading.Thread(target=self._socket_listener, daemon=True)
        sock_th.start()

        # 3. Kalkış hazırlığı
        print("[Drone] İHA'dan koordinat bekleniyor ve kalkış hazırlanıyor...")
        self._set_mode("STABILIZE")
        time.sleep(1)
        self._arm()
        self._set_mode("GUIDED")
        self._takeoff(CRUISE_ALT)

        # 4. Görev döngüsü (hedef bekleme + teslimat + smart idle)
        try:
            self._mission_loop()
        except KeyboardInterrupt:
            print("\n[Drone] Kullanıcı tarafından durduruldu.")
            self.mission_active = False
            self._rtl()
        finally:
            self.mission_active = False
            print("[Drone] Script sonlandı.")


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    drone = DroneController()
    drone.run()
