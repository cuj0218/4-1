"""
Ninja Fruit - Hand Gesture Game  (성능 최적화 + 입체 과일)
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
    for backend in [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]:
        for idx in range(4):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"카메라 발견: index={idx}")
                    return cap
                cap.release()
    return None

# ── 설정 ─────────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 1280, 720
PROC_W, PROC_H = 640, 360          # MediaPipe 처리 해상도 (성능용)
GRAVITY = 0.35
MIN_SLASH_SPEED = 10
TRAIL_LEN = 20
FRUIT_SPAWN_INTERVAL = 1.2
MAX_LIVES = 3

FRUITS = [
    {"name": "watermelon", "color": (40, 180, 40),   "inner": (40, 40, 200),  "radius": 42, "score": 1},
    {"name": "orange",     "color": (20, 130, 255),  "inner": (20, 180, 255), "radius": 35, "score": 1},
    {"name": "apple",      "color": (30, 30, 210),   "inner": (70, 70, 245),  "radius": 35, "score": 1},
    {"name": "lemon",      "color": (20, 210, 220),  "inner": (60, 230, 245), "radius": 30, "score": 1},
    {"name": "kiwi",       "color": (15, 90, 15),    "inner": (80, 160, 60),  "radius": 32, "score": 1},
    {"name": "bomb",       "color": (25, 25, 25),    "inner": (70, 70, 70),   "radius": 33, "score": 0},
]

# ── 유틸 ─────────────────────────────────────────────────────────────────────
def dist(p1, p2):
    return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def line_circle_intersect(p1, p2, cx, cy, r):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    fx, fy = p1[0]-cx, p1[1]-cy
    a = dx*dx + dy*dy
    if a == 0: return False
    b = 2*(fx*dx + fy*dy)
    c = fx*fx + fy*fy - r*r
    disc = b*b - 4*a*c
    if disc < 0: return False
    sq = math.sqrt(disc)
    t1 = (-b-sq)/(2*a); t2 = (-b+sq)/(2*a)
    return (0<=t1<=1) or (0<=t2<=1)

# ── 입체 과일 그리기 (frame copy 없음) ───────────────────────────────────────
def draw_3d_fruit(frame, cx, cy, r, base_color, inner_color, is_bomb=False):
    # 그림자 (타원, 복사 없이)
    shadow_color = (max(0,base_color[0]-30), max(0,base_color[1]-30), max(0,base_color[2]-30))
    cv2.ellipse(frame, (cx+5, cy+r-4), (r, max(1, r//3)), 0, 0, 360, (10,18,22), -1)

    if is_bomb:
        dark = (15, 15, 15)
        cv2.circle(frame, (cx, cy), r, dark, -1)
        # 그라디언트 레이어
        for s in range(5, 0, -1):
            rr = max(1, r*s//5)
            ox = -r*s//(5*4); oy = -r*s//(5*4)
            shade = min(255, 25 + s*8)
            cv2.circle(frame, (cx+ox, cy+oy), rr, (shade, shade, shade), -1)
        # 심지
        cv2.line(frame, (cx, cy-r), (cx+8, cy-r-14), (80,80,160), 3)
        cv2.circle(frame, (cx+8, cy-r-14), 5, (0,180,255), -1)
        # 하이라이트
        cv2.circle(frame, (cx-r//3, cy-r//3), max(2,r//5), (90,90,90), -1)
        return

    # 어두운 테두리
    dark_edge = tuple(max(0, c-80) for c in base_color)
    cv2.circle(frame, (cx, cy), r, dark_edge, -1)

    # 그라디언트: 바깥→안쪽, 빛 방향 top-left
    steps = 7
    for s in range(steps):
        t = (s+1) / steps
        rr = max(1, int(r * (1 - s*0.10)))
        ox = int(-r * 0.22 * t)
        oy = int(-r * 0.22 * t)
        factor = 0.55 + 0.65*t
        bright = tuple(min(255, int(c * factor)) for c in base_color)
        cv2.circle(frame, (cx+ox, cy+oy), rr, bright, -1)

    # 단면 색(안쪽)
    cv2.circle(frame, (cx - r//6, cy - r//6), max(2, r*5//9), inner_color, -1)

    # 스페큘러 하이라이트 (흰 빛 반사)
    sx, sy = cx - r//3, cy - r//3
    cv2.circle(frame, (sx, sy),     max(2, r//4),  (255,255,255), -1)
    cv2.circle(frame, (sx+3, sy+3), max(1, r//9),  (200,220,255), -1)

    # 테두리 rim
    cv2.circle(frame, (cx, cy), r, dark_edge, 2)

def draw_3d_half(frame, cx, cy, r, base_color, inner_color, flip=False):
    if not (-r < cx < WIDTH+r and -r < cy < HEIGHT+r):
        return
    dark_edge = tuple(max(0, c-80) for c in base_color)
    angle_start = 180 if flip else 0
    angle_end   = 360 if flip else 180
    cv2.ellipse(frame, (cx,cy), (r, max(1,r//2)), 0, angle_start, angle_end, dark_edge, -1)
    for s in range(5):
        t = (s+1)/5
        rr = max(1, int(r*(1-s*0.12)))
        hh = max(1, rr//2)
        ox = int(-r*0.15*t); oy = int(-r*0.15*t)
        factor = 0.55 + 0.55*t
        bright = tuple(min(255, int(c*factor)) for c in base_color)
        cv2.ellipse(frame, (cx+ox, cy+oy), (rr, hh), 0, angle_start, angle_end, bright, -1)
    # 단면
    cv2.ellipse(frame, (cx, cy), (r, max(1,r//2)), 0, angle_start, angle_end, inner_color, 3)

# ── 클래스들 ─────────────────────────────────────────────────────────────────
class Fruit:
    def __init__(self):
        info = random.choice(FRUITS)
        self.__dict__.update(info)
        self.is_bomb = (info["name"] == "bomb")
        m = self.radius + 20
        self.x = float(random.randint(m, WIDTH-m))
        self.y = float(HEIGHT + self.radius)
        speed = random.uniform(14, 19)
        angle = math.radians(random.uniform(60, 120))
        self.vx = speed * math.cos(angle) * random.choice([-1,1])
        self.vy = -speed * math.sin(angle)
        self.rotation = 0.0
        self.rot_speed = random.uniform(-4, 4)
        self.alive = True
        self.sliced = False
        self.h1x = self.h1y = self.h2x = self.h2y = 0.0
        self.sv1 = random.uniform(-4, -1)
        self.sv2 = random.uniform(1, 4)
        self.particles = []
        self.slice_time = 0.0
        self.missed = False

    def update(self):
        if not self.sliced:
            self.vy += GRAVITY
            self.x  += self.vx
            self.y  += self.vy
            self.rotation += self.rot_speed
            if self.y > HEIGHT + self.radius*2:
                if not self.is_bomb: self.missed = True
                self.alive = False
        else:
            self.h1x += self.sv1;  self.h1y += 2.5
            self.h2x += self.sv2;  self.h2y += 2.5
            for p in self.particles:
                p[0] += p[2]; p[1] += p[3]; p[3] += 0.3; p[4] -= 1
            self.particles = [p for p in self.particles if p[4] > 0]
            if time.time()-self.slice_time > 1.0 and not self.particles:
                self.alive = False

    def do_slice(self):
        self.sliced = True
        self.slice_time = time.time()
        for _ in range(20):
            a = random.uniform(0, 2*math.pi)
            s = random.uniform(2, 9)
            self.particles.append([
                self.x, self.y,
                math.cos(a)*s, math.sin(a)*s - 4,
                random.randint(18, 38)
            ])

    def draw(self, frame):
        ix, iy = int(self.x), int(self.y)
        r = self.radius
        if not self.sliced:
            draw_3d_fruit(frame, ix, iy, r, self.color, self.inner, self.is_bomb)
        else:
            # 파티클
            for p in self.particles:
                life_ratio = p[4] / 38
                c = tuple(int(ch * life_ratio) for ch in self.inner)
                cv2.circle(frame, (int(p[0]), int(p[1])), max(1, int(4*life_ratio)), c, -1)
            # 반쪽
            draw_3d_half(frame, int(ix+self.h1x), int(iy+self.h1y), r, self.color, self.inner, flip=False)
            draw_3d_half(frame, int(ix+self.h2x), int(iy+self.h2y), r, self.color, self.inner, flip=True)


class SlashTrail:
    """frame.copy() 없이 직접 색상으로 궤적 표현 → 대폭 빠름"""
    def __init__(self):
        self.pts = deque(maxlen=TRAIL_LEN)

    def add(self, pt):
        self.pts.append(pt)

    def clear(self):
        self.pts.clear()

    def draw(self, frame):
        p = list(self.pts)
        n = len(p)
        if n < 2: return
        # 단일 overlay로 전체 궤적 한 번에 처리 (copy 1회만)
        overlay = frame.copy()
        for i in range(1, n):
            t = i / n
            thick = max(1, int(t * 8))
            b = int(150 * (1-t))
            g = int(60  * (1-t))
            r = min(255, int(180 + 75*t))
            cv2.line(overlay, p[i-1], p[i], (b, g, r), thick, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def last_two(self):
        if len(self.pts) < 2: return None, None
        return self.pts[-2], self.pts[-1]


class FloatingText:
    def __init__(self, x, y, text, color=(0,255,255), scale=1.2):
        self.x=x; self.y=y; self.text=text; self.color=color
        self.scale=scale; self.life=60; self.vy=-2

    def update(self):
        self.y += self.vy; self.life -= 1
        return self.life > 0

    def draw(self, frame):
        a = self.life/60
        c = tuple(int(ch*a) for ch in self.color)
        cv2.putText(frame, self.text, (self.x, self.y),
                    cv2.FONT_HERSHEY_DUPLEX, self.scale, c, 2, cv2.LINE_AA)


# ── 메인 게임 ────────────────────────────────────────────────────────────────
class NinjaFruitGame:
    def __init__(self, cap):
        self.cap = cap
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=0,                 # 가장 빠른 모델
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )

        self.trail = SlashTrail()
        self.fruits = []
        self.effects = []
        self.score = 0; self.best = 0
        self.lives = MAX_LIVES; self.combo = 0
        self.last_spawn = time.time()
        self.game_over = False
        self.start_time = time.time()
        self.bg = self._make_bg()

        # 매 프레임 재사용할 버퍼
        self._proc_buf = np.zeros((PROC_H, PROC_W, 3), dtype=np.uint8)

    def _make_bg(self):
        bg = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        bg[:] = (42, 60, 78)
        for _ in range(14):
            y = random.randint(0, HEIGHT)
            c = tuple(random.randint(a, b) for a,b in [(32,58),(50,72),(62,90)])
            cv2.line(bg,(0,y),(WIDTH, y+random.randint(-20,20)), c, random.randint(20,50))
        for _ in range(7):
            x1,y1 = random.randint(0,WIDTH), random.randint(0,HEIGHT)
            cv2.line(bg,(x1,y1),(x1+random.randint(-250,250), y1+random.randint(-250,250)),
                     (20,35,48), 2)
        return bg

    def spawn(self):
        now = time.time()
        if now - self.last_spawn > FRUIT_SPAWN_INTERVAL:
            for _ in range(random.choices([1,2,3], weights=[50,35,15])[0]):
                self.fruits.append(Fruit())
            self.last_spawn = now

    def check_slash(self, p1, p2):
        if dist(p1,p2) < MIN_SLASH_SPEED: return
        cut = 0
        for f in self.fruits:
            if f.sliced or not f.alive: continue
            if line_circle_intersect(p1,p2, f.x,f.y, f.radius):
                if f.is_bomb:
                    self.lives -= 1; self.combo = 0
                    self.effects.append(FloatingText(int(f.x)-40, int(f.y), "BOMB!", (0,0,255)))
                else:
                    self.score += f.score; cut += 1
                f.do_slice()
        if cut >= 2:
            self.combo += cut
            self.effects.append(FloatingText(WIDTH//2-110, 200,
                f"{cut}-COMBO +{cut}", (0,255,200), 1.6))
        elif cut == 1:
            self.combo += 1

    def draw_hud(self, frame):
        cv2.putText(frame, str(self.score), (30,65),
                    cv2.FONT_HERSHEY_DUPLEX, 2.2, (255,220,50), 3, cv2.LINE_AA)
        cv2.putText(frame, f"BEST:{self.best}", (30,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,180,30), 2, cv2.LINE_AA)
        elapsed = int(time.time()-self.start_time)
        cv2.putText(frame, f"{elapsed//60}:{elapsed%60:02d}", (WIDTH-150,65),
                    cv2.FONT_HERSHEY_DUPLEX, 1.5, (50,220,255), 2, cv2.LINE_AA)
        for i in range(MAX_LIVES):
            c = (55,55,220) if i < self.lives else (45,45,45)
            cx = WIDTH-55-i*52; cy = 115
            pts = np.array([[cx,cy+13],[cx-14,cy-4],[cx-7,cy-13],
                             [cx,cy-7],[cx+7,cy-13],[cx+14,cy-4]], np.int32)
            cv2.fillPoly(frame, [pts], c)
        if self.combo >= 3:
            cv2.putText(frame, f"COMBO x{self.combo}", (WIDTH//2-120,65),
                        cv2.FONT_HERSHEY_DUPLEX, 1.3, (0,255,200), 2, cv2.LINE_AA)
        cv2.putText(frame, "Q:quit  R:restart", (10, HEIGHT-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100,100,100), 1)

    def draw_game_over(self, frame):
        ov = frame.copy()
        cv2.rectangle(ov,(0,0),(WIDTH,HEIGHT),(0,0,0),-1)
        cv2.addWeighted(ov, 0.65, frame, 0.35, 0, frame)
        cv2.putText(frame,"GAME OVER",(WIDTH//2-230,HEIGHT//2-50),
                    cv2.FONT_HERSHEY_DUPLEX,3,(0,50,255),5,cv2.LINE_AA)
        cv2.putText(frame,f"Score: {self.score}",(WIDTH//2-130,HEIGHT//2+35),
                    cv2.FONT_HERSHEY_DUPLEX,2,(255,220,50),3,cv2.LINE_AA)
        cv2.putText(frame,"R : Restart    Q : Quit",(WIDTH//2-210,HEIGHT//2+100),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(200,200,200),2,cv2.LINE_AA)

    def reset(self):
        self.best = max(self.best, self.score)
        self.score=0; self.lives=MAX_LIVES; self.combo=0
        self.fruits=[]; self.effects=[]
        self.trail.clear()
        self.last_spawn=time.time(); self.start_time=time.time()
        self.game_over=False

    def run(self):
        print("게임 시작! 검지로 과일을 베세요.  Q:종료  R:재시작")
        cv2.namedWindow("Ninja Fruit", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Ninja Fruit", WIDTH, HEIGHT)

        while True:
            ret, raw = self.cap.read()
            if not ret or raw is None: continue

            raw = cv2.flip(raw, 1)
            raw = cv2.resize(raw, (WIDTH, HEIGHT))

            # ── 배경 합성 (1회 addWeighted) ───────────────────────
            frame = self.bg.copy()
            cv2.addWeighted(raw, 0.36, frame, 0.64, 0, frame)

            # ── 손 감지: 저해상도로 처리해 속도 향상 ──────────────
            small = cv2.resize(raw, (PROC_W, PROC_H))
            rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = self.hands.process(rgb)
            rgb.flags.writeable = True

            finger_pos = None
            if res.multi_hand_landmarks:
                lm = res.multi_hand_landmarks[0].landmark[8]
                finger_pos = (int(lm.x*WIDTH), int(lm.y*HEIGHT))
                cv2.circle(frame, finger_pos, 14, (255,255,255), -1)
                cv2.circle(frame, finger_pos,  9, (80,180,255), -1)

            if finger_pos:
                p1, _ = self.trail.last_two()
                self.trail.add(finger_pos)
                if p1 and not self.game_over:
                    self.check_slash(p1, finger_pos)
            else:
                self.trail.clear()

            # ── 게임 로직 ─────────────────────────────────────────
            if not self.game_over:
                self.spawn()
                for f in self.fruits:
                    f.update()
                    if f.missed:
                        self.lives -= 1; self.combo = 0
                        self.effects.append(FloatingText(int(f.x)-30, int(f.y), "MISS!", (0,80,255)))
                        f.missed = False
                self.fruits = [f for f in self.fruits if f.alive]
                if self.lives <= 0:
                    self.game_over = True
                    self.best = max(self.best, self.score)

            # ── 렌더링 ────────────────────────────────────────────
            for f in self.fruits:
                f.draw(frame)
            self.trail.draw(frame)
            self.effects = [e for e in self.effects if e.update()]
            for e in self.effects:
                e.draw(frame)
            self.draw_hud(frame)
            if self.game_over:
                self.draw_game_over(frame)

            cv2.imshow("Ninja Fruit", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27): break
            elif key == ord('r'): self.reset()

        self.cap.release()
        cv2.destroyAllWindows()
        self.hands.close()
        print(f"종료! 최고 점수: {self.best}")


if __name__ == "__main__":
    print("카메라 탐색 중...")
    cap = find_camera()
    if cap is None:
        print("카메라를 찾을 수 없습니다. 시스템 설정 > 개인정보보호 > 카메라 확인")
        sys.exit(1)
    NinjaFruitGame(cap).run()
