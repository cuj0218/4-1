"""
Ninja Fruit - Hand Gesture Game
손가락으로 과일을 베어라!
macOS 카메라 호환 버전
"""

import cv2
import mediapipe as mp
import numpy as np
import random
import time
import math
import sys
from collections import deque

# ── 카메라 자동 탐색 ─────────────────────────────────────────────────────────
def find_camera():
    """작동하는 카메라 인덱스를 자동으로 찾습니다."""
    backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
    for backend in backends:
        for idx in range(4):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"카메라 발견: index={idx}, backend={backend}")
                    return cap
                cap.release()
    return None

# ── 설정 ────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
GRAVITY = 0.35
MIN_SLASH_SPEED = 10
TRAIL_LEN = 18
FRUIT_SPAWN_INTERVAL = 1.2
MAX_LIVES = 3

# ── 과일 정의 ────────────────────────────────────────────────────────────────
FRUITS = [
    {"name": "watermelon", "color": (50, 200, 50),  "inner": (50, 50, 220),  "radius": 42, "score": 1},
    {"name": "orange",     "color": (30, 140, 255),  "inner": (30, 190, 255), "radius": 35, "score": 1},
    {"name": "apple",      "color": (40, 40, 220),   "inner": (80, 80, 255),  "radius": 35, "score": 1},
    {"name": "lemon",      "color": (30, 220, 230),  "inner": (80, 240, 255), "radius": 30, "score": 1},
    {"name": "kiwi",       "color": (20, 100, 20),   "inner": (100, 180, 80), "radius": 30, "score": 1},
    {"name": "bomb",       "color": (30, 30, 30),    "inner": (80, 80, 80),   "radius": 33, "score": 0},
]

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def line_circle_intersect(p1, p2, cx, cy, r):
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
        angle = random.uniform(60, 120)
        rad = math.radians(angle)
        self.vx = speed * math.cos(rad) * random.choice([-1, 1])
        self.vy = -speed * math.sin(rad)
        self.rotation = 0
        self.rot_speed = random.uniform(-4, 4)
        self.alive = True
        self.sliced = False
        self.half1_offset = [0, 0]
        self.half2_offset = [0, 0]
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
            if self.y > HEIGHT + self.radius * 2:
                if not self.is_bomb:
                    self.missed = True
                self.alive = False
        else:
            elapsed = time.time() - self.slice_time
            self.half1_offset[0] += self.slice_vx1
            self.half1_offset[1] += elapsed * 1.5
            self.half2_offset[0] += self.slice_vx2
            self.half2_offset[1] += elapsed * 1.5
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
            cv2.circle(frame, (ix, iy), r, self.color, -1)
            cv2.circle(frame, (ix, iy), r - 5, self.inner if not self.is_bomb else (50, 50, 50), -1)
            cv2.circle(frame, (ix - r//4, iy - r//4), r//4, (255, 255, 255), -1)
            if self.is_bomb:
                cv2.line(frame, (ix, iy - r), (ix + 8, iy - r - 15), (100, 100, 200), 3)
                cv2.circle(frame, (ix + 8, iy - r - 15), 5, (0, 200, 255), -1)
        else:
            for p in self.juice_particles:
                alpha = max(0, p["life"] / 40)
                color = tuple(int(c * alpha) for c in self.inner)
                cv2.circle(frame, (int(p["x"]), int(p["y"])), 4, color, -1)
            for offset in [self.half1_offset, self.half2_offset]:
                hx = int(ix + offset[0])
                hy = int(iy + offset[1])
                if -r < hx < WIDTH + r and -r < hy < HEIGHT + r:
                    cv2.ellipse(frame, (hx, hy), (r, max(1, r // 2)),
                                self.rotation, 0, 180, self.color, -1)
                    cv2.ellipse(frame, (hx, hy), (r - 5, max(1, (r - 5) // 2)),
                                self.rotation, 0, 180, self.inner, -1)


class SlashTrail:
    def __init__(self):
        self.points = deque(maxlen=TRAIL_LEN)

    def add(self, pt):
        self.points.append(pt)

    def get_speed(self):
        if len(self.points) < 2:
            return 0
        return dist(self.points[-2], self.points[-1])

    def draw(self, frame):
        pts = list(self.points)
        n = len(pts)
        for i in range(1, n):
            alpha = i / n
            thickness = max(1, int(alpha * 7))
            b = int(180 * (1 - alpha))
            g = int(80 * (1 - alpha))
            overlay = frame.copy()
            cv2.line(overlay, pts[i-1], pts[i], (b, g, 255), thickness)
            cv2.addWeighted(overlay, alpha * 0.75, frame, 1 - alpha * 0.75, 0, frame)

    def get_recent_segment(self):
        pts = list(self.points)
        if len(pts) < 2:
            return None, None
        return pts[-2], pts[-1]


class FloatingText:
    def __init__(self, x, y, text, color=(0, 255, 255), scale=1.2):
        self.x, self.y = x, y
        self.text = text
        self.color = color
        self.scale = scale
        self.life = 60
        self.vy = -2

    def update(self):
        self.y += self.vy
        self.life -= 1
        return self.life > 0

    def draw(self, frame):
        alpha = self.life / 60
        color = tuple(int(c * alpha) for c in self.color)
        cv2.putText(frame, self.text, (self.x, self.y),
                    cv2.FONT_HERSHEY_DUPLEX, self.scale * (1 + (1 - alpha) * 0.3),
                    color, 2, cv2.LINE_AA)


# ── 메인 게임 ────────────────────────────────────────────────────────────────
class NinjaFruitGame:
    def __init__(self, cap):
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.55,
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
        self.bg = self._make_background()

    def _make_background(self):
        bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        bg[:] = (40, 58, 75)
        for _ in range(12):
            y = random.randint(0, HEIGHT)
            color = (random.randint(30, 55), random.randint(48, 70), random.randint(60, 88))
            cv2.line(bg, (0, y), (WIDTH, y + random.randint(-15, 15)), color, random.randint(18, 45))
        for _ in range(6):
            x1, y1 = random.randint(0, WIDTH), random.randint(0, HEIGHT)
            x2 = x1 + random.randint(-200, 200)
            y2 = y1 + random.randint(-200, 200)
            cv2.line(bg, (x1, y1), (x2, y2), (20, 35, 48), 2)
        return bg

    def spawn_fruit(self):
        now = time.time()
        if now - self.last_spawn > FRUIT_SPAWN_INTERVAL:
            count = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
            for _ in range(count):
                self.fruits.append(Fruit())
            self.last_spawn = now

    def check_slash(self, p1, p2):
        if dist(p1, p2) < MIN_SLASH_SPEED:
            return
        sliced_now = 0
        for fruit in self.fruits:
            if fruit.sliced or not fruit.alive:
                continue
            if line_circle_intersect(p1, p2, fruit.x, fruit.y, fruit.radius):
                if fruit.is_bomb:
                    self.lives -= 1
                    self.combo = 0
                    self.effects.append(FloatingText(
                        int(fruit.x) - 40, int(fruit.y), "BOMB!", (0, 0, 255)))
                    fruit.do_slice()
                else:
                    fruit.do_slice()
                    self.score += fruit.score
                    sliced_now += 1
        if sliced_now >= 2:
            self.combo += sliced_now
            self.effects.append(FloatingText(
                WIDTH // 2 - 100, 200,
                f"{sliced_now}-COMBO +{sliced_now}", (0, 255, 200), 1.5))
        elif sliced_now == 1:
            self.combo += 1

    def draw_hud(self, frame):
        # 점수
        cv2.putText(frame, str(self.score), (30, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255, 220, 50), 3, cv2.LINE_AA)
        cv2.putText(frame, f"BEST:{self.best}", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 180, 30), 2, cv2.LINE_AA)
        # 타이머
        elapsed = int(time.time() - self.start_time)
        timer_str = f"{elapsed // 60}:{elapsed % 60:02d}"
        cv2.putText(frame, timer_str, (WIDTH - 150, 65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (50, 220, 255), 2, cv2.LINE_AA)
        # 목숨 (하트)
        for i in range(MAX_LIVES):
            color = (60, 60, 220) if i < self.lives else (50, 50, 50)
            cx = WIDTH - 55 - i * 50
            cy = 115
            pts = np.array([
                [cx, cy + 12],
                [cx - 14, cy - 4],
                [cx - 7, cy - 12],
                [cx, cy - 6],
                [cx + 7, cy - 12],
                [cx + 14, cy - 4],
            ], np.int32)
            cv2.fillPoly(frame, [pts], color)
        # 콤보
        if self.combo >= 3:
            cv2.putText(frame, f"COMBO x{self.combo}",
                        (WIDTH // 2 - 110, 65),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0, 255, 200), 2, cv2.LINE_AA)
        # 안내
        cv2.putText(frame, "Q:quit  R:restart", (10, HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)

    def draw_game_over(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (WIDTH, HEIGHT), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame, "GAME OVER",
                    (WIDTH // 2 - 220, HEIGHT // 2 - 50),
                    cv2.FONT_HERSHEY_DUPLEX, 3, (0, 50, 255), 5, cv2.LINE_AA)
        cv2.putText(frame, f"Score: {self.score}",
                    (WIDTH // 2 - 120, HEIGHT // 2 + 30),
                    cv2.FONT_HERSHEY_DUPLEX, 2, (255, 220, 50), 3, cv2.LINE_AA)
        cv2.putText(frame, "R : Restart    Q : Quit",
                    (WIDTH // 2 - 200, HEIGHT // 2 + 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2, cv2.LINE_AA)

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
        print("게임 시작! 검지 손가락으로 과일을 베세요.")
        print("Q: 종료 | R: 재시작")
        win_name = "Ninja Fruit"
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win_name, WIDTH, HEIGHT)

        while True:
            ret, raw = self.cap.read()
            if not ret or raw is None:
                print("프레임을 읽을 수 없습니다. 카메라를 확인하세요.")
                time.sleep(0.1)
                continue

            # 좌우 반전 + 리사이즈
            raw = cv2.flip(raw, 1)
            raw = cv2.resize(raw, (WIDTH, HEIGHT))

            # 배경: 나무판 + 카메라 반투명 합성
            frame = self.bg.copy()
            cv2.addWeighted(raw, 0.38, frame, 0.62, 0, frame)

            # 손 감지
            rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            result = self.hands.process(rgb)
            rgb.flags.writeable = True

            finger_pos = None
            if result.multi_hand_landmarks:
                for hand_lm in result.multi_hand_landmarks:
                    lm = hand_lm.landmark[8]  # 검지 끝
                    fx = int(lm.x * WIDTH)
                    fy = int(lm.y * HEIGHT)
                    finger_pos = (fx, fy)
                    # 손가락 표시
                    cv2.circle(frame, finger_pos, 14, (255, 255, 255), -1)
                    cv2.circle(frame, finger_pos, 9, (80, 180, 255), -1)
                    break

            if finger_pos:
                p1, _ = self.trail.get_recent_segment()
                self.trail.add(finger_pos)
                if p1 and not self.game_over:
                    self.check_slash(p1, finger_pos)
            else:
                self.trail = SlashTrail()

            # 게임 로직
            if not self.game_over:
                self.spawn_fruit()
                for fruit in self.fruits:
                    fruit.update()
                    if fruit.missed:
                        self.lives -= 1
                        self.combo = 0
                        self.effects.append(FloatingText(
                            int(fruit.x) - 30, int(fruit.y), "MISS!", (0, 80, 255)))
                        fruit.missed = False
                self.fruits = [f for f in self.fruits if f.alive]
                if self.lives <= 0:
                    self.game_over = True
                    self.best = max(self.best, self.score)

            # 그리기
            for fruit in self.fruits:
                fruit.draw(frame)
            self.trail.draw(frame)
            self.effects = [e for e in self.effects if e.update()]
            for e in self.effects:
                e.draw(frame)
            self.draw_hud(frame)
            if self.game_over:
                self.draw_game_over(frame)

            cv2.imshow(win_name, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('r'):
                self.reset()

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print(f"종료! 최고 점수: {self.best}")


# ── 엔트리포인트 ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("카메라 탐색 중...")
    cap = find_camera()
    if cap is None:
        print("카메라를 찾을 수 없습니다.")
        print("시스템 설정 > 개인 정보 보호 > 카메라 에서 터미널 권한을 허용해주세요.")
        sys.exit(1)
    game = NinjaFruitGame(cap)
    game.run()
