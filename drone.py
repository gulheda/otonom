"""
Drone (Multikopter) Kontrol Scripti
=====================================
Araç   : iris_with_ardupilot
SysID  : 2
UDP    : 14560
Görev  : VTOL'den gelen hedef koordinatlara git → paket bırak
         Smart Idle Mode: görev yoksa standby noktasında bekle
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

# Görev bitince döneceği güvenli standby noktası
STANDBY_LAT  = 47.3977419
STANDBY_LON  = 8.5455938
STANDBY_ALT  = 25.0       # metre

# ═══════════════════════════════════════════════════════════════
#  SİSTEM YAPILANDIRMASI
# ═══════════════════════════════════════════════════════════════

DRONE_CONNECTION  = "udp:127.0.0.1:14560"
DRONE_SYSID       = 2
MSG_LISTEN_PORT   = 6000    # VTOL'den hedef alınacak UDP portu

CRUISE_ALT        = 25.0    # metre – seyir irtifası
DELIVERY_ALT      = 5.0     # metre – paket bırakma irtifası
PAYLOAD_DIST      = 3.0     # metre – hedefe bu kadar yaklaşınca bırak

CRUISE_SPEED      = 8.0     # m/s
IDLE_SPEED        = 3.0     # m/s – smart idle sürükleme hızı
IDLE_TIMEOUT      = 30.0    # saniye – standby'a geçiş süresi


# ═══════════════════════════════════════════════════════════════
#  YARDIMCI
# ═══════════════════════════════════════════════════════════════

def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = (math.sin(math.radians(lat2-lat1)/2)**2 +
         math.cos(phi1)*math.cos(phi2)*math.sin(math.radians(lon2-lon1)/2)**2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))


def _bearing(lat1, lon1, lat2, lon2):
    dlon = math.radians(lon2 - lon1)
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ═══════════════════════════════════════════════════════════════
#  ANA DRONE SINIFI
# ═══════════════════════════════════════════════════════════════

class DroneController:

    def __init__(self):
        self.vehicle         = None
        self.mission_active  = True

        # Hedef kuyruğu – VTOL'den gelen koordinatlar sıraya girer
        self._queue          = []      # [(lat, lon, alt, idx), ...]
        self._queue_lock     = threading.Lock()

        # Smart idle için
        self._last_bearing   = None    # son hedefe gidiş yönü (derece)
        self._last_task_time = time.time()

        # Tamamlanan teslimatlar
        self._deliveries_done = []

    # ──────────────────────────────────────────
    #  BAĞLANTI
    # ──────────────────────────────────────────
    def connect(self, retries=5):
        for attempt in range(1, retries + 1):
            try:
                print(f"[Drone] Bağlanılıyor... ({attempt}/{retries})")
                self.vehicle = mavutil.mavlink_connection(
                    DRONE_CONNECTION,
                    source_system=255,
                    target_system=DRONE_SYSID
                )
                self.vehicle.wait_heartbeat(timeout=10)
                print(f"[Drone] Heartbeat – SysID:{self.vehicle.target_system}")
                return True
            except Exception as e:
                print(f"[Drone] Bağlantı hatası: {e}")
                time.sleep(3)
        return False

    # ──────────────────────────────────────────
    #  MAVLink YARDIMCILARI
    # ──────────────────────────────────────────
    def _set_mode(self, mode_name):
        mode_id = self.vehicle.mode_mapping().get(mode_name)
        if mode_id is None:
            print(f"[Drone] Bilinmeyen mod: {mode_name}")
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
                print(f"[Drone] Mod → {mode_name}")
                return True
        print(f"[Drone] Mod onaylanamadı: {mode_name}")
        return False

    def _arm(self):
        print("[Drone] ARM...")
        self.vehicle.arducopter_arm()
        self.vehicle.motors_armed_wait()
        print("[Drone] ARM tamamlandı.")

    def _takeoff(self, altitude=CRUISE_ALT):
        print(f"[Drone] TAKEOFF → {altitude}m")
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
                print(f"[Drone] İrtifa: {cur:.1f}m / {altitude}m", end="\r")
                if cur >= altitude * 0.90:
                    print(f"\n[Drone] Seyir irtifasına ulaşıldı.")
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

    def _change_alt(self, target_alt, tolerance=1.5):
        """Mevcut yatay konumda irtifa değiştirir."""
        lat, lon, _ = self._get_position()
        if lat is None:
            return
        self._goto(lat, lon, target_alt)
        while True:
            _, _, cur_alt = self._get_position()
            if cur_alt and abs(cur_alt - target_alt) < tolerance:
                return
            time.sleep(0.5)

    # ──────────────────────────────────────────
    #  SERVO – PAKET BIRAKMA
    # ──────────────────────────────────────────
    def _drop_payload(self, channel=7, pwm_open=2000, pwm_close=1000):
        """Yük tutucuyu servo ile açar, sonra kapatır."""
        print("[Drone] Paket bırakılıyor – servo açılıyor...")
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, channel, pwm_open, 0, 0, 0, 0, 0
        )
        time.sleep(2.0)
        self.vehicle.mav.command_long_send(
            self.vehicle.target_system, self.vehicle.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_SERVO,
            0, channel, pwm_close, 0, 0, 0, 0, 0
        )
        print("[Drone] Paket bırakıldı.")

    # ──────────────────────────────────────────
    #  UDP – HEDEF DİNLEYİCİ
    # ──────────────────────────────────────────
    def _socket_listener(self):
        """
        Port 6000'de VTOL'den JSON mesajı bekler.
        Gelen her GOTO komutu kuyruğa eklenir.
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind(("0.0.0.0", MSG_LISTEN_PORT))
        sock.settimeout(1.0)
        print(f"[Drone] UDP dinleniyor: 0.0.0.0:{MSG_LISTEN_PORT}")

        while self.mission_active:
            try:
                data, addr = sock.recvfrom(1024)
                msg = json.loads(data.decode())
                if msg.get("cmd") == "GOTO":
                    lat  = float(msg["lat"])
                    lon  = float(msg["lon"])
                    alt  = float(msg.get("alt", CRUISE_ALT))
                    idx  = int(msg.get("point_idx", 0))
                    print(f"\n[Drone] Hedef #{idx+1} alındı: "
                          f"({lat:.6f}, {lon:.6f}, {alt:.1f}m)")
                    with self._queue_lock:
                        self._queue.append((lat, lon, alt, idx))
                    self._last_task_time = time.time()
            except socket.timeout:
                continue
            except json.JSONDecodeError:
                print("[Drone] Geçersiz mesaj formatı.")
            except Exception as e:
                print(f"[Drone] Soket hatası: {e}")

        sock.close()

    # ──────────────────────────────────────────
    #  TESLİMAT GÖREVİ
    # ──────────────────────────────────────────
    def _deliver(self, lat, lon, alt, point_idx):
        """
        Teslimat sekansı:
        1. Seyir irtifasında hedefe git
        2. İnmeye başla (DELIVERY_ALT)
        3. Paket bırak
        4. Seyir irtifasına geri çık
        """
        print(f"\n[Drone] ── TESLİMAT #{point_idx+1} BAŞLADI ──")
        print(f"[Drone] Hedef: ({lat:.6f}, {lon:.6f})")

        # Son hedefe gidiş yönünü kaydet
        cur_lat, cur_lon, _ = self._get_position()
        if cur_lat:
            self._last_bearing = _bearing(cur_lat, cur_lon, lat, lon)

        self._set_speed(CRUISE_SPEED)
        self._goto(lat, lon, alt)

        # Hedefe yaklaşırken kuyruğu kontrol et
        while self.mission_active:
            cur_lat, cur_lon, cur_alt = self._get_position()
            if cur_lat is None:
                time.sleep(0.5)
                continue

            dist = _haversine(cur_lat, cur_lon, lat, lon)
            print(f"[Drone] Mesafe: {dist:.1f}m", end="\r")

            # Kuyrukta yeni hedef var mı? (direkt yön değiştir)
            with self._queue_lock:
                if len(self._queue) > 0:
                    print(f"\n[Drone] Kuyrukta yeni hedef var – "
                          f"mevcut teslimat tamamlandıktan devralınacak.")

            if dist <= PAYLOAD_DIST:
                print()
                break
            time.sleep(0.8)

        print(f"[Drone] Hedef #{point_idx+1} üzerinde – alçalma başlıyor...")
        self._change_alt(DELIVERY_ALT)
        self._drop_payload()
        time.sleep(1)

        # Seyir irtifasına geri dön
        self._change_alt(CRUISE_ALT)
        self._deliveries_done.append(point_idx)
        print(f"[Drone] Teslimat #{point_idx+1} tamamlandı. "
              f"Toplam: {len(self._deliveries_done)}")

    # ──────────────────────────────────────────
    #  SMART IDLE MODE
    # ──────────────────────────────────────────
    def _smart_idle(self):
        """
        Görev yokken:
        - Son hedefe doğru yavaş drift et
        - IDLE_TIMEOUT sonra standby noktasına git
        - Yeni görev gelirse hemen çık
        """
        print("[Drone] Smart Idle aktif.")
        self._set_speed(IDLE_SPEED)
        idle_start = time.time()

        while self.mission_active:
            # Yeni hedef var mı?
            with self._queue_lock:
                if self._queue:
                    print("[Drone] Idle'dan çıkılıyor – yeni hedef var.")
                    return

            elapsed = time.time() - idle_start

            # Standby noktasına git
            if elapsed >= IDLE_TIMEOUT:
                print("[Drone] Idle timeout – Standby'a gidiliyor...")
                self._set_speed(CRUISE_SPEED)
                self._goto(STANDBY_LAT, STANDBY_LON, STANDBY_ALT)

                # Standby'da yeni hedef bekle
                while self.mission_active:
                    with self._queue_lock:
                        if self._queue:
                            print("[Drone] Standby'dan çıkılıyor – görev geldi.")
                            return
                    time.sleep(1)
                return

            # Son hedef yönünde yavaş sürüklen
            if self._last_bearing is not None:
                cur_lat, cur_lon, cur_alt = self._get_position()
                if cur_lat:
                    ang = math.radians(self._last_bearing)
                    delta = 30.0 / 6371000  # 30m adım
                    new_lat = cur_lat + math.degrees(delta * math.cos(ang))
                    new_lon = cur_lon + math.degrees(
                        delta * math.sin(ang) /
                        math.cos(math.radians(cur_lat))
                    )
                    self._goto(new_lat, new_lon, cur_alt or CRUISE_ALT)

            time.sleep(5)

    # ──────────────────────────────────────────
    #  ANA GÖREV DÖNGÜSÜ
    # ──────────────────────────────────────────
    def _mission_loop(self):
        """
        Kuyruktan hedef al → teslimat yap → smart idle → tekrar
        İki teslimat da tamamlanınca RTL.
        """
        while self.mission_active:
            target = None
            with self._queue_lock:
                if self._queue:
                    target = self._queue.pop(0)

            if target:
                lat, lon, alt, idx = target
                self._deliver(lat, lon, alt, idx)

                # 2 teslimat tamamlandıysa bitir
                if len(self._deliveries_done) >= 2:
                    print("\n[Drone] Her iki teslimat tamamlandı – RTL.")
                    break
            else:
                self._smart_idle()

        self._rtl()

    # ──────────────────────────────────────────
    #  GÖREV SONU
    # ──────────────────────────────────────────
    def _rtl(self):
        print("[Drone] RTL başlatılıyor...")
        self._set_mode("RTL")

    # ──────────────────────────────────────────
    #  ÖZET YAZDIR
    # ──────────────────────────────────────────
    def _print_info(self):
        print("\n" + "═"*50)
        print("  DRONE GÖREV PLANI")
        print("═"*50)
        print(f"  Seyir irtifası : {CRUISE_ALT}m")
        print(f"  Bırakma irtif. : {DELIVERY_ALT}m")
        print(f"  Idle timeout   : {IDLE_TIMEOUT}s")
        print(f"  Standby nokta  : ({STANDBY_LAT:.6f}, {STANDBY_LON:.6f})")
        print(f"  UDP port       : {MSG_LISTEN_PORT}")
        print("═"*50 + "\n")

    # ──────────────────────────────────────────
    #  ANA AKIŞ
    # ──────────────────────────────────────────
    def run(self):
        self._print_info()

        if not self.connect():
            print("[Drone] Bağlantı kurulamadı.")
            return

        # UDP soket dinleyiciyi başlat
        sock_th = threading.Thread(target=self._socket_listener, daemon=True)
        sock_th.start()

        # VTOL bırakmasını bekle
        print("[Drone] VTOL'den bırakılmayı bekliyor (5s)...")
        time.sleep(5)

        # Stabilize → ARM → GUIDED → TAKEOFF
        self._set_mode("STABILIZE")
        time.sleep(1)
        self._arm()
        self._set_mode("GUIDED")
        self._takeoff(CRUISE_ALT)

        # Görev döngüsü
        try:
            self._mission_loop()
        except KeyboardInterrupt:
            print("\n[Drone] Kullanıcı durdurdu.")
            self.mission_active = False
            self._rtl()
        finally:
            self.mission_active = False
            print("[Drone] Script sonlandı.")


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    drone = DroneController()
    drone.run()
