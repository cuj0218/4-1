"""
Ninja Fruit - Hand Gesture Game
손가락으로 과일을 베어라!
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
from collections import deque

# ── 설정 ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
FPS = 60
GRAVITY = 0.35
MIN_SLASH_SPEED = 12       # 이 속도 이상일 때만 칼질 인정
TRAIL_LEN = 18             # 손가락 궤적 길이
FRUIT_SPAWN_INTERVAL = 1.2 # 초
MAX_LIVES = 3

# ── 과일 정의 ────────────────────────────────────────────────────────────────
FRUITS = [
    {"name": "watermelon", "color": (50, 200, 50),  "inner": (50, 50, 220),  "radius": 42, "score": 1},
    {"name": "orange",     "color": (30, 140, 255),  "inner": (30, 190, 255), "radius": 35, "score": 1},
    {"name": "apple",      "color": (40, 40, 220),   "inner": (80, 80, 255),  "radius": 35, "score": 1},
    {"name": "lemon",      "color": (30, 220, 230),  "inner": (80, 240, 255), "radius": 30, "score": 1},
    {"name": "kiwi",       "color": (20, 100, 20),   "inner": (100, 180, 80), "radius": 30, "score": 1},
    {"name": "bomb",       "color": (30, 30, 30),    "inner": (80, 80, 80),   "radius": 33, "score": 0},  # 폭탄
]

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def line_circle_intersect(p1, p2, cx, cy, r):
    """선분 p1-p2 가 원 (cx,cy,r) 과 교차하는지 확인"""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    fx, fy = p1[0] - cx, p1[1] - cy
    a = dx*dx + dy*dy
    if a == 0:
        return False
    b = 2 * (fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0:
        return False
    disc = math.sqrt(disc)
    t1 = (-b - disc) / (2*a)
    t2 = (-b + disc) / (2*a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


# ── 클래스 ───────────────────────────────────────────────────────────────────
class Fruit:
    def __init__(self):
        info = random.choice(FRUITS)
        self.__dict__.update(info)
        self.is_bomb = (info["name"] == "bomb")
        margin = self.radius + 20
        self.x = random.randint(margin, WIDTH - margin)
        self.y = HEIGHT + self.radius
        speed = random.uniform(14, 19)
        angle = random.uniform(60, 120)        # 위쪽 방향
        rad = math.radians(angle)
        self.vx = speed * math.cos(rad) * random.choice([-1, 1])
        self.vy = -speed * math.sin(rad)
        self.rotation = 0
        self.rot_speed = random.uniform(-4, 4)
        self.alive = True
        self.sliced = False
        self.half1_offset = (0, 0)
        self.half2_offset = (0, 0)
        self.slice_vx1 = random.uniform(-3, -1)
        self.slice_vx2 = random.uniform(1, 3)
        self.juice_particles = []
        self.slice_time = 0
        self.missed = False

    def update(self):
        if not self.sliced:
            self.vy += GRAVITY
            self.x += self.vx
            self.y += self.vy
            self.rotation += self.rot_speed
            # 화면 밖으로 나가면 missed
            if self.y > HEIGHT + self.radius * 2:
                if not self.is_bomb:
                    self.missed = True
                self.alive = False
        else:
            # 반쪽들 날아가기
            elapsed = time.time() - self.slice_time
            self.half1_offset = (self.half1_offset[0] + self.slice_vx1,
                                 self.half1_offset[1] + elapsed * 2)
            self.half2_offset = (self.half2_offset[0] + self.slice_vx2,
                                 self.half2_offset[1] + elapsed * 2)
            # 주스 파티클
            for p in self.juice_particles:
                p["x"] += p["vx"]
                p["y"] += p["vy"]
                p["vy"] += 0.3
                p["life"] -= 1
            self.juice_particles = [p for p in self.juice_particles if p["life"] > 0]
            if elapsed > 1.0 and not self.juice_particles:
                self.alive = False

    def do_slice(self):
        self.sliced = True
        self.slice_time = time.time()
        # 주스 파티클 생성
        for _ in range(18):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            self.juice_particles.append({
                "x": self.x, "y": self.y,
                "vx": math.cos(angle) * speed,
                "vy": math.sin(angle) * speed - 3,
                "life": random.randint(20, 40),
            })

    def draw(self, frame):
        ix, iy = int(self.x), int(self.y)
        r = self.radius

        if not self.sliced:
            # 그림자
            shadow = frame.copy()
            cv2.ellipse(shadow, (ix + 6, iy + 6), (r, r // 2), 0, 0, 360,
                        (20, 20, 20), -1)
            cv2.addWeighted(shadow, 0.3, frame, 0.7, 0, frame)
            # 과일 본체
            cv2.circle(frame, (ix, iy), r, self.color, -1)
            cv2.circle(frame, (ix, iy), r - 5, self.inner if not self.is_bomb else (50, 50, 50), -1)
            # 과일 표면 하이라이트
            cv2.circle(frame, (ix - r//4, iy - r//4), r//4, (255, 255, 255), -1)
            if self.is_bomb:
                # 심지
                cv2.line(frame, (ix, iy - r), (ix + 8, iy - r - 15), (100, 100, 200), 3)
                cv2.circle(frame, (ix + 8, iy - r - 15), 5, (0, 200, 255), -1)
        else:
            # 주스 파티클 먼저
            for p in self.juice_particles:
                alpha = max(0, p["life"] / 40)
                color = tuple(int(c * alpha) for c in self.inner)
                cv2.circle(frame, (int(p["x"]), int(p["y"])), 4, color, -1)

            # 반쪽 그리기 (간단히 반원)
            for sign, offset in [(1, self.half1_offset), (-1, self.half2_offset)]:
                hx = int(ix + offset[0])
                hy = int(iy + offset[1])
                if 0 <= hx < WIDTH and 0 <= hy < HEIGHT:
                    cv2.ellipse(frame, (hx, hy), (r, r // 2),
                                self.rotation, 0, 180 if sign == 1 else 180,
                                self.color, -1)
                    cv2.ellipse(frame, (hx, hy), (r - 5, (r - 5) // 2),
                                self.rotation, 0, 180, self.inner, -1)


class SlashTrail:
    def __init__(self):
        self.points = deque(maxlen=TRAIL_LEN)
        self.timestamps = deque(maxlen=TRAIL_LEN)

    def add(self, pt):
        self.points.append(pt)
        self.timestamps.append(time.time())

    def get_speed(self):
        if len(self.points) < 2:
            return 0
        p1 = self.points[-2]
        p2 = self.points[-1]
        return dist(p1, p2)

    def draw(self, frame):
        pts = list(self.points)
        n = len(pts)
        for i in range(1, n):
            alpha = i / n
            thickness = max(1, int(alpha * 6))
            # 흰색 → 붉은 칼날 궤적
            b = int(200 * (1 - alpha))
            g = int(100 * (1 - alpha))
            r = 255
            # 블렌딩으로 그리기
            overlay = frame.copy()
            cv2.line(overlay, pts[i-1], pts[i], (b, g, r), thickness)
            cv2.addWeighted(overlay, alpha * 0.8, frame, 1 - alpha * 0.8, 0, frame)

    def get_recent_segment(self):
        """최근 두 점 반환"""
        pts = list(self.points)
        if len(pts) < 2:
            return None, None
        return pts[-2], pts[-1]


class ComboEffect:
    def __init__(self, x, y, text, color=(0, 255, 255)):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.life = 60
        self.vy = -2

    def update(self):
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, frame):
        alpha = self.life / 60
        scale = 1.2 + (1 - alpha) * 0.5
        thickness = max(1, int(3 * alpha))
        color = tuple(int(c * alpha) for c in self.color)
        cv2.putText(frame, self.text, (self.x, self.y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)


# ── 메인 게임 ────────────────────────────────────────────────────────────────
class NinjaFruitGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FPS)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )

        self.trail = SlashTrail()
        self.fruits = []
        self.effects = []
        self.score = 0
        self.best = 0
        self.lives = MAX_LIVES
        self.combo = 0
        self.last_spawn = time.time()
        self.game_over = False
        self.start_time = time.time()

        # 배경 나무판자 텍스처 생성
        self.bg = self._make_background()

    def _make_background(self):
        bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        bg[:] = (40, 60, 80)   # 기본 갈색 나무
        # 나무 결 라인들
        for y in range(0, HEIGHT, random.randint(60, 120)):
            color = (random.randint(30, 55), random.randint(50, 75), random.randint(65, 90))
            cv2.line(bg, (0, y), (WIDTH, y + random.randint(-10, 10)), color, random.randint(15, 40))
        # 균열
        for _ in range(6):
            x1 = random.randint(0, WIDTH)
            y1 = random.randint(0, HEIGHT)
            x2 = x1 + random.randint(-200, 200)
            y2 = y1 + random.randint(-200, 200)
            cv2.line(bg, (x1, y1), (x2, y2), (20, 35, 50), 2)
        return bg

    def spawn_fruit(self):
        now = time.time()
        if now - self.last_spawn > FRUIT_SPAWN_INTERVAL:
            # 1~3개 한 번에 스폰
            count = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
            for _ in range(count):
                self.fruits.append(Fruit())
            self.last_spawn = now

    def check_slash(self, p1, p2):
        speed = dist(p1, p2)
        if speed < MIN_SLASH_SPEED:
            return
        sliced_now = 0
        for fruit in self.fruits:
            if fruit.sliced or not fruit.alive:
                continue
            if line_circle_intersect(p1, p2, fruit.x, fruit.y, fruit.radius):
                if fruit.is_bomb:
                    self.lives -= 1
                    self.combo = 0
                    self.effects.append(ComboEffect(
                        int(fruit.x) - 40, int(fruit.y),
                        "BOMB!", (0, 0, 255)
                    ))
                    fruit.do_slice()
                else:
                    fruit.do_slice()
                    self.score += fruit.score
                    sliced_now += 1

        if sliced_now >= 2:
            self.combo += sliced_now
            mx = int(sum(f.x for f in self.fruits if f.sliced) / max(1, sliced_now))
            self.effects.append(ComboEffect(
                mx - 60, 200,
                f"{sliced_now}-COMBO +{sliced_now}",
                (0, 255, 200)
            ))
        elif sliced_now == 1:
            self.combo += 1

    def draw_hud(self, frame):
        # 상단 점수
        cv2.putText(frame, str(self.score), (30, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 220, 50), 3, cv2.LINE_AA)
        cv2.putText(frame, f"BEST:{self.best}", (30, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 180, 30), 2, cv2.LINE_AA)

        # 타이머
        elapsed = int(time.time() - self.start_time)
        sec = elapsed % 60
        minute = elapsed // 60
        timer_str = f"{minute}:{sec:02d}"
        tw, _ = cv2.getTextSize(timer_str, cv2.FONT_HERSHEY_DUPLEX, 1.5, 2)[0], 0
        cv2.putText(frame, timer_str, (WIDTH - 140, 60),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 220, 255), 2, cv2.LINE_AA)

        # 목숨 (하트)
        for i in range(MAX_LIVES):
            color = (50, 50, 220) if i < self.lives else (60, 60, 60)
            cx = WIDTH - 50 - i * 45
            cy = 110
            cv2.circle(frame, (cx, cy), 15, color, -1)
            cv2.circle(frame, (cx - 8, cy - 5), 9, color, -1)
            cv2.circle(frame, (cx + 8, cy - 5), 9, color, -1)

        # 콤보
        if self.combo >= 3:
            cv2.putText(frame, f"COMBO x{self.combo}", (WIDTH//2 - 100, 60),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 255, 200), 2, cv2.LINE_AA)

    def draw_game_over(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, "GAME OVER", (WIDTH//2 - 200, HEIGHT//2 - 60),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 50, 255), 4, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {self.score}", (WIDTH//2 - 110, HEIGHT//2 + 20),
                    cv2.FONT_HERSHEY_DUPLEX, 1.8, (255, 220, 50), 3, cv2.LINE_AA)
        cv2.putText(frame, "Press R to restart / Q to quit",
                    (WIDTH//2 - 250, HEIGHT//2 + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)

    def reset(self):
        self.best = max(self.best, self.score)
        self.score = 0
        self.lives = MAX_LIVES
        self.combo = 0
        self.fruits = []
        self.effects = []
        self.trail = SlashTrail()
        self.last_spawn = time.time()
        self.start_time = time.time()
        self.game_over = False

    def run(self):
        print("🍉 Ninja Fruit 시작! 손가락으로 과일을 베세요.")
        print("   Q: 종료 | R: 재시작")
        cv2.namedWindow("Ninja Fruit", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ninja Fruit", WIDTH, HEIGHT)

        prev_time = time.time()

        while True:
            ret, raw_frame = self.cap.read()
            if not ret:
                print("카메라를 읽을 수 없습니다.")
                break

            # 좌우 반전 (거울 모드)
            raw_frame = cv2.flip(raw_frame, 1)
            raw_frame = cv2.resize(raw_frame, (WIDTH, HEIGHT))

            # ── 배경 합성 ────────────────────────────────────────
            # 카메라 영상을 반투명하게 배경에 합성
            frame = self.bg.copy()
            cv2.addWeighted(raw_frame, 0.35, frame, 0.65, 0, frame)

            # ── MediaPipe 손 감지 ────────────────────────────────
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb)

            finger_pos = None
            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    # 검지 손가락 끝 (landmark 8)
                    lm = hand_lm.landmark[8]
                    fx = int(lm.x * WIDTH)
                    fy = int(lm.y * HEIGHT)
                    finger_pos = (fx, fy)

                    # 손가락 시각화 (작은 점)
                    cv2.circle(frame, finger_pos, 10, (255, 255, 255), -1)
                    cv2.circle(frame, finger_pos, 6, (100, 200, 255), -1)
                    break  # 첫 번째 손만 사용

            if finger_pos:
                p1, p2 = self.trail.get_recent_segment()
                self.trail.add(finger_pos)
                if p1 and p2 and not self.game_over:
                    self.check_slash(p1, finger_pos)
            else:
                # 손이 안 보이면 궤적 초기화
                self.trail = SlashTrail()

            # ── 게임 로직 ─────────────────────────────────────────
            if not self.game_over:
                self.spawn_fruit()

                # 과일 업데이트
                for fruit in self.fruits:
                    fruit.update()
                    if fruit.missed:
                        self.lives -= 1
                        self.combo = 0
                        self.effects.append(ComboEffect(
                            int(fruit.x), int(fruit.y),
                            "MISS!", (0, 100, 255)
                        ))
                        fruit.missed = False

                self.fruits = [f for f in self.fruits if f.alive]

                if self.lives <= 0:
                    self.game_over = True
                    self.best = max(self.best, self.score)

            # ── 그리기 ────────────────────────────────────────────
            # 과일
            for fruit in self.fruits:
                fruit.draw(frame)

            # 칼 궤적
            self.trail.draw(frame)

            # 이펙트
            self.effects = [e for e in self.effects if e.update()]
            for e in self.effects:
                e.draw(frame)

            # HUD
            self.draw_hud(frame)

            # FPS
            now = time.time()
            fps = 1 / max(0.001, now - prev_time)
            prev_time = now
            cv2.putText(frame, f"FPS:{fps:.0f}", (30, HEIGHT - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            if self.game_over:
                self.draw_game_over(frame)

            cv2.imshow("Ninja Fruit", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('r'):
                self.reset()

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print(f"게임 종료! 최고 점수: {self.best}")


if __name__ == "__main__":
    game = NinjaFruitGame()
    game.run()
