"""
Pursuit-Evasion Simulation: Sphero vs Cat
==========================================
A 2D simulation environment for learning and experimenting with
pursuit-evasion strategies. Includes:

1. Reactive flee (baseline)
2. Visibility-aware hiding behind obstacles
3. Q-Learning agent that learns evasion strategies
4. Behavior-tree prey (hide → peek → dart → hide)

Controls:
  - Press 1-4 to switch evasion strategy
  - Press R to reset
  - Press SPACE to pause/resume
  - Press H to toggle heatmap (shows learned Q-values)
  - Press T to toggle training mode (fast, no render)
  - Press V to toggle visibility polygon display
  - Click to place the cat manually

Requirements: pip install pygame numpy
"""

import math
import random
from collections import defaultdict
from enum import Enum

import numpy as np
import pygame

# ── CONFIG ──────────────────────────────────────────────────────────────────
WIDTH, HEIGHT = 900, 700
FPS = 60
CELL_SIZE = 20  # for Q-learning grid discretization

# Colors
BG_COLOR = (18, 18, 24)
GRID_COLOR = (30, 30, 40)
OBSTACLE_COLOR = (60, 65, 80)
OBSTACLE_BORDER = (80, 85, 100)
CAT_COLOR = (230, 100, 80)
CAT_EYE = (255, 220, 50)
BALL_COLOR = (60, 180, 255)
BALL_GLOW = (60, 180, 255, 60)
SHADOW_COLOR = (255, 255, 100, 25)
VISIBILITY_COLOR = (255, 200, 50, 15)
SAFE_ZONE_COLOR = (50, 200, 100, 30)
TEXT_COLOR = (200, 200, 210)
PANEL_BG = (25, 25, 35, 220)
ACCENT = (100, 220, 180)
DANGER_COLOR = (255, 80, 80)
HIDE_COLOR = (80, 255, 160)

# Physics
BALL_SPEED = 3.5
CAT_SPEED = 2.8
CAT_ACCEL = 0.15
CAT_POUNCE_SPEED = 6.0
CAT_POUNCE_DIST = 80
CAT_INTEREST_DECAY = 0.998
CAT_INTEREST_BOOST = 0.05
CATCH_DIST = 18
OBSTACLE_MARGIN = 8


# ── OBSTACLES ───────────────────────────────────────────────────────────────
def create_obstacles():
    """Create furniture-like obstacles in the room."""
    return [
        pygame.Rect(150, 150, 120, 80),  # table
        pygame.Rect(400, 100, 60, 160),  # bookshelf
        pygame.Rect(650, 200, 100, 60),  # chair
        pygame.Rect(200, 400, 80, 120),  # ottoman
        pygame.Rect(500, 350, 140, 70),  # couch
        pygame.Rect(100, 550, 60, 80),  # plant pot
        pygame.Rect(700, 450, 90, 100),  # cabinet
        pygame.Rect(350, 550, 100, 50),  # coffee table
    ]


# ── VISIBILITY ──────────────────────────────────────────────────────────────
def compute_visibility_polygon(pos, obstacles, bounds=(WIDTH, HEIGHT), num_rays=120):
    """Cast rays from pos to determine visible area. Returns polygon vertices."""
    ox, oy = pos
    angles = []

    # Cast rays toward obstacle corners (+ slight offsets for accuracy)
    for obs in obstacles:
        corners = [
            (obs.left, obs.top),
            (obs.right, obs.top),
            (obs.left, obs.bottom),
            (obs.right, obs.bottom),
        ]
        for cx, cy in corners:
            angle = math.atan2(cy - oy, cx - ox)
            for offset in [-0.001, 0, 0.001]:
                angles.append(angle + offset)

    # Also cast evenly spaced rays
    for i in range(num_rays):
        angles.append(2 * math.pi * i / num_rays)

    angles.sort()
    polygon = []

    for angle in angles:
        dx = math.cos(angle)
        dy = math.sin(angle)

        # Find closest intersection
        min_t = max(bounds[0], bounds[1]) * 2
        # Check bounds
        if dx > 0:
            min_t = min(min_t, (bounds[0] - ox) / dx)
        elif dx < 0:
            min_t = min(min_t, -ox / dx)
        if dy > 0:
            min_t = min(min_t, (bounds[1] - oy) / dy)
        elif dy < 0:
            min_t = min(min_t, -oy / dy)

        # Check obstacles
        for obs in obstacles:
            # Check all 4 edges
            edges = [
                ((obs.left, obs.top), (obs.right, obs.top)),
                ((obs.right, obs.top), (obs.right, obs.bottom)),
                ((obs.right, obs.bottom), (obs.left, obs.bottom)),
                ((obs.left, obs.bottom), (obs.left, obs.top)),
            ]
            for (x1, y1), (x2, y2) in edges:
                denom = dx * (y2 - y1) - dy * (x2 - x1)
                if abs(denom) < 1e-10:
                    continue
                t = ((x1 - ox) * (y2 - y1) - (y1 - oy) * (x2 - x1)) / denom
                u = ((x1 - ox) * (-dy) - (y1 - oy) * (-dx)) / denom
                if t > 0.1 and 0 <= u <= 1:
                    min_t = min(min_t, t)

        min_t = max(min_t, 0)
        polygon.append((ox + dx * min_t, oy + dy * min_t))

    return polygon


def is_visible(from_pos, to_pos, obstacles):
    """Check if to_pos is visible from from_pos (no obstacles blocking)."""
    fx, fy = from_pos
    tx, ty = to_pos
    dx, dy = tx - fx, ty - fy
    dist = math.hypot(dx, dy)
    if dist < 1:
        return True

    for obs in obstacles:
        # Inflate obstacle slightly
        inflated = obs.inflate(4, 4)
        edges = [
            ((inflated.left, inflated.top), (inflated.right, inflated.top)),
            ((inflated.right, inflated.top), (inflated.right, inflated.bottom)),
            ((inflated.right, inflated.bottom), (inflated.left, inflated.bottom)),
            ((inflated.left, inflated.bottom), (inflated.left, inflated.top)),
        ]
        for (x1, y1), (x2, y2) in edges:
            denom = dx * (y2 - y1) - dy * (x2 - x1)
            if abs(denom) < 1e-10:
                continue
            t = ((x1 - fx) * (y2 - y1) - (y1 - fy) * (x2 - x1)) / denom
            u = ((x1 - fx) * (-dy) - (y1 - fy) * (-dx)) / denom
            if 0.01 < t < 0.99 and 0 <= u <= 1:
                return False
    return True


def find_hiding_spots(ball_pos, cat_pos, obstacles):
    """Find positions behind obstacles that are hidden from the cat."""
    spots = []
    for obs in obstacles:
        # Sample points around the obstacle
        margin = OBSTACLE_MARGIN + 15
        candidates = [
            (obs.left - margin, obs.centery),
            (obs.right + margin, obs.centery),
            (obs.centerx, obs.top - margin),
            (obs.centerx, obs.bottom + margin),
            (obs.left - margin, obs.top - margin),
            (obs.right + margin, obs.top - margin),
            (obs.left - margin, obs.bottom + margin),
            (obs.right + margin, obs.bottom + margin),
        ]
        for cx, cy in candidates:
            if 10 < cx < WIDTH - 10 and 10 < cy < HEIGHT - 10:
                # Check if this spot is hidden from cat
                if not is_visible(cat_pos, (cx, cy), obstacles):
                    # Check if reachable (not inside an obstacle)
                    pt_rect = pygame.Rect(cx - 5, cy - 5, 10, 10)
                    blocked = any(pt_rect.colliderect(o) for o in obstacles)
                    if not blocked:
                        dist = math.hypot(cx - ball_pos[0], cy - ball_pos[1])
                        spots.append(((cx, cy), dist))
    spots.sort(key=lambda x: x[1])
    return spots


# ── EVASION STRATEGIES ──────────────────────────────────────────────────────


class Strategy(Enum):
    REACTIVE = 1
    HIDE = 2
    QLEARNING = 3
    BEHAVIOR_TREE = 4


def strategy_reactive(ball_pos, cat_pos, obstacles):
    """Simple reactive flee: run away + wall avoidance + jitter."""
    bx, by = ball_pos
    cx, cy = cat_pos
    dx, dy = bx - cx, by - cy
    dist = math.hypot(dx, dy)

    if dist < 1:
        dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
    else:
        dx, dy = dx / dist, dy / dist

    # Add wall avoidance
    wall_force_x, wall_force_y = 0, 0
    if bx < 50:
        wall_force_x += (50 - bx) / 50
    if bx > WIDTH - 50:
        wall_force_x -= (bx - (WIDTH - 50)) / 50
    if by < 50:
        wall_force_y += (50 - by) / 50
    if by > HEIGHT - 50:
        wall_force_y -= (by - (HEIGHT - 50)) / 50

    # Add jitter for unpredictability
    jx = random.gauss(0, 0.3)
    jy = random.gauss(0, 0.3)

    dx += wall_force_x * 0.5 + jx
    dy += wall_force_y * 0.5 + jy

    norm = math.hypot(dx, dy)
    if norm > 0:
        dx, dy = dx / norm, dy / norm

    return dx * BALL_SPEED, dy * BALL_SPEED


def strategy_hide(ball_pos, cat_pos, obstacles):
    """Visibility-aware: find and move toward hiding spots behind obstacles."""
    spots = find_hiding_spots(ball_pos, cat_pos, obstacles)

    if spots:
        target, _ = spots[0]  # closest hiding spot
        dx = target[0] - ball_pos[0]
        dy = target[1] - ball_pos[1]
        dist = math.hypot(dx, dy)
        if dist > 2:
            dx, dy = dx / dist, dy / dist
            return dx * BALL_SPEED, dy * BALL_SPEED

    # Fallback to reactive if no hiding spots
    return strategy_reactive(ball_pos, cat_pos, obstacles)


class QLearningAgent:
    """Tabular Q-Learning agent for evasion."""

    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(8))  # 8 directions
        self.lr = 0.1
        self.gamma = 0.95
        self.epsilon = 0.2
        self.episode_reward = 0
        self.total_episodes = 0
        self.best_survival = 0
        self.survival_times = []

        # 8 movement directions
        self.actions = []
        for i in range(8):
            angle = 2 * math.pi * i / 8
            self.actions.append((math.cos(angle), math.sin(angle)))

    def get_state(self, ball_pos, cat_pos, obstacles):
        """Discretize the state space."""
        bx, by = ball_pos
        cx, cy = cat_pos

        # Relative position of cat (discretized angle + distance)
        dx, dy = cx - bx, cy - by
        angle = math.atan2(dy, dx)
        angle_bin = int((angle + math.pi) / (2 * math.pi) * 8) % 8
        dist = math.hypot(dx, dy)
        dist_bin = min(int(dist / 60), 5)

        # Am I visible to the cat?
        visible = 1 if is_visible(cat_pos, ball_pos, obstacles) else 0

        # Nearest wall proximity
        wall_x = 0 if bx < 80 else (2 if bx > WIDTH - 80 else 1)
        wall_y = 0 if by < 80 else (2 if by > HEIGHT - 80 else 1)

        # Nearest obstacle direction
        min_obs_dist = float("inf")
        obs_dir = 0
        for obs in obstacles:
            odist = math.hypot(obs.centerx - bx, obs.centery - by)
            if odist < min_obs_dist:
                min_obs_dist = odist
                obs_angle = math.atan2(obs.centery - by, obs.centerx - bx)
                obs_dir = int((obs_angle + math.pi) / (2 * math.pi) * 4) % 4

        return (angle_bin, dist_bin, visible, wall_x, wall_y, obs_dir)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 7)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        old_val = self.q_table[state][action]
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] += self.lr * (target - old_val)
        self.episode_reward += reward

    def get_move(self, ball_pos, cat_pos, obstacles):
        state = self.get_state(ball_pos, cat_pos, obstacles)
        action = self.choose_action(state)
        dx, dy = self.actions[action]
        return dx * BALL_SPEED, dy * BALL_SPEED, state, action

    def get_q_values_for_display(self):
        """Return Q-values for heatmap visualization."""
        return dict(self.q_table)


class BehaviorTreePrey:
    """State-machine prey: Hide → Peek → Dart → Hide."""

    class State(Enum):
        FLEE = 0
        SEEK_COVER = 1
        HIDING = 2
        PEEKING = 3
        DARTING = 4

    def __init__(self):
        self.state = self.State.FLEE
        self.state_timer = 0
        self.hide_target = None
        self.dart_dir = None
        self.peek_timer = 0

    def get_move(self, ball_pos, cat_pos, obstacles):
        bx, by = ball_pos
        cx, cy = cat_pos
        dist_to_cat = math.hypot(bx - cx, by - cy)
        am_visible = is_visible(cat_pos, ball_pos, obstacles)

        self.state_timer += 1

        # ── State transitions ──
        if self.state == self.State.FLEE:
            if dist_to_cat > 150:
                self.state = self.State.SEEK_COVER
                self.state_timer = 0

        elif self.state == self.State.SEEK_COVER:
            spots = find_hiding_spots(ball_pos, cat_pos, obstacles)
            if spots:
                self.hide_target = spots[0][0]
                if not am_visible and dist_to_cat > 100:
                    self.state = self.State.HIDING
                    self.state_timer = 0

            if dist_to_cat < 80:
                self.state = self.State.FLEE
                self.state_timer = 0

        elif self.state == self.State.HIDING:
            if self.state_timer > 90:  # ~1.5 seconds
                self.state = self.State.PEEKING
                self.state_timer = 0
                self.peek_timer = 0
            if dist_to_cat < 60:
                self.state = self.State.FLEE
                self.state_timer = 0

        elif self.state == self.State.PEEKING:
            self.peek_timer += 1
            if self.peek_timer > 30 and am_visible:
                self.state = self.State.DARTING
                self.state_timer = 0
                angle = math.atan2(by - cy, bx - cx) + random.uniform(-0.8, 0.8)
                self.dart_dir = (math.cos(angle), math.sin(angle))
            if dist_to_cat < 50:
                self.state = self.State.FLEE
                self.state_timer = 0

        elif self.state == self.State.DARTING:
            if self.state_timer > 25:
                self.state = self.State.SEEK_COVER
                self.state_timer = 0
            if dist_to_cat < 60:
                self.state = self.State.FLEE
                self.state_timer = 0

        # ── Movement for each state ──
        if self.state == self.State.FLEE:
            return strategy_reactive(ball_pos, cat_pos, obstacles)

        elif self.state == self.State.SEEK_COVER:
            return strategy_hide(ball_pos, cat_pos, obstacles)

        elif self.state == self.State.HIDING:
            # Stay put, small drift toward cover center
            if self.hide_target:
                dx = self.hide_target[0] - bx
                dy = self.hide_target[1] - by
                dist = math.hypot(dx, dy)
                if dist > 5:
                    return dx / dist * 0.5, dy / dist * 0.5
            return 0, 0

        elif self.state == self.State.PEEKING:
            # Slowly edge toward visibility
            if self.hide_target:
                # Move slightly toward cat to become visible
                dx = cx - bx
                dy = cy - by
                dist = math.hypot(dx, dy)
                if dist > 0:
                    return dx / dist * 0.8, dy / dist * 0.8
            return 0, 0

        elif self.state == self.State.DARTING:
            if self.dart_dir:
                return (
                    self.dart_dir[0] * BALL_SPEED * 1.3,
                    self.dart_dir[1] * BALL_SPEED * 1.3,
                )
            return strategy_reactive(ball_pos, cat_pos, obstacles)

        return 0, 0


# ── CAT AI ──────────────────────────────────────────────────────────────────
class CatAI:
    """Cat pursuer with realistic behaviors."""

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = 0
        self.vy = 0
        self.interest = 1.0
        self.last_seen_pos = None
        self.last_seen_time = 0
        self.pounce_cooldown = 0
        self.is_pouncing = False
        self.wander_angle = random.uniform(0, 2 * math.pi)

    def update(self, ball_pos, obstacles, frame):
        can_see = is_visible((self.x, self.y), ball_pos, obstacles)

        if can_see:
            self.last_seen_pos = ball_pos
            self.last_seen_time = frame
            self.interest = min(1.0, self.interest + CAT_INTEREST_BOOST)
        else:
            self.interest *= CAT_INTEREST_DECAY

        self.pounce_cooldown = max(0, self.pounce_cooldown - 1)

        # Determine target
        if can_see:
            tx, ty = ball_pos
        elif self.last_seen_pos and frame - self.last_seen_time < 120:
            tx, ty = self.last_seen_pos
        else:
            # Wander
            self.wander_angle += random.gauss(0, 0.1)
            tx = self.x + math.cos(self.wander_angle) * 100
            ty = self.y + math.sin(self.wander_angle) * 100
            tx = max(30, min(WIDTH - 30, tx))
            ty = max(30, min(HEIGHT - 30, ty))

        dx = tx - self.x
        dy = ty - self.y
        dist = math.hypot(dx, dy)

        if dist > 1:
            dx, dy = dx / dist, dy / dist

        # Pounce mechanic
        speed = CAT_SPEED * self.interest
        if (
            can_see
            and dist < CAT_POUNCE_DIST
            and self.pounce_cooldown == 0
            and not self.is_pouncing
        ):
            self.is_pouncing = True
            self.vx = dx * CAT_POUNCE_SPEED
            self.vy = dy * CAT_POUNCE_SPEED
            self.pounce_cooldown = 60
        elif self.is_pouncing:
            self.vx *= 0.92
            self.vy *= 0.92
            if math.hypot(self.vx, self.vy) < 1:
                self.is_pouncing = False
        else:
            self.vx += (dx * speed - self.vx) * CAT_ACCEL
            self.vy += (dy * speed - self.vy) * CAT_ACCEL

        # Move with obstacle avoidance
        new_x = self.x + self.vx
        new_y = self.y + self.vy
        cat_rect = pygame.Rect(new_x - 10, new_y - 10, 20, 20)

        blocked = False
        for obs in obstacles:
            if cat_rect.colliderect(obs):
                blocked = True
                break

        if not blocked:
            self.x = max(10, min(WIDTH - 10, new_x))
            self.y = max(10, min(HEIGHT - 10, new_y))
        else:
            # Slide along obstacle
            cat_rect_x = pygame.Rect(self.x + self.vx - 10, self.y - 10, 20, 20)
            cat_rect_y = pygame.Rect(self.x - 10, self.y + self.vy - 10, 20, 20)
            can_x = not any(cat_rect_x.colliderect(o) for o in obstacles)
            can_y = not any(cat_rect_y.colliderect(o) for o in obstacles)
            if can_x:
                self.x = max(10, min(WIDTH - 10, self.x + self.vx))
            if can_y:
                self.y = max(10, min(HEIGHT - 10, self.y + self.vy))


# ── MAIN SIMULATION ────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Pursuit-Evasion: Sphero vs Cat")
    clock = pygame.time.Clock()

    # Fonts
    try:
        font = pygame.font.SysFont("Consolas", 14)
        font_large = pygame.font.SysFont("Consolas", 18, bold=True)
        font_title = pygame.font.SysFont("Consolas", 22, bold=True)
    except:
        font = pygame.font.Font(None, 16)
        font_large = pygame.font.Font(None, 20)
        font_title = pygame.font.Font(None, 24)

    obstacles = create_obstacles()

    # Initialize entities
    def reset():
        ball = [100.0, 600.0]
        cat = CatAI(750.0, 100.0)
        return ball, cat

    ball_pos, cat = reset()
    q_agent = QLearningAgent()
    bt_prey = BehaviorTreePrey()

    strategy = Strategy.REACTIVE
    paused = False
    show_visibility = True
    show_heatmap = False
    training_mode = False
    frame = 0
    survival_time = 0
    best_survival = 0
    catches = 0
    total_runs = 0
    run_history = []

    strategy_names = {
        Strategy.REACTIVE: "1: Reactive Flee",
        Strategy.HIDE: "2: Hide Behind Obstacles",
        Strategy.QLEARNING: "3: Q-Learning Agent",
        Strategy.BEHAVIOR_TREE: "4: Behavior Tree Prey",
    }

    strategy_desc = {
        Strategy.REACTIVE: "Simple repulsion + wall avoidance + jitter",
        Strategy.HIDE: "Computes visibility shadows, seeks occluded positions",
        Strategy.QLEARNING: "Tabular Q-learning, discretized state space",
        Strategy.BEHAVIOR_TREE: "State machine: Flee → Seek Cover → Hide → Peek → Dart",
    }

    running = True
    prev_state = None
    prev_action = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    strategy = Strategy.REACTIVE
                elif event.key == pygame.K_2:
                    strategy = Strategy.HIDE
                elif event.key == pygame.K_3:
                    strategy = Strategy.QLEARNING
                elif event.key == pygame.K_4:
                    strategy = Strategy.BEHAVIOR_TREE
                    bt_prey = BehaviorTreePrey()
                elif event.key == pygame.K_r:
                    ball_pos, cat = reset()
                    survival_time = 0
                    bt_prey = BehaviorTreePrey()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_v:
                    show_visibility = not show_visibility
                elif event.key == pygame.K_h:
                    show_heatmap = not show_heatmap
                elif event.key == pygame.K_t:
                    training_mode = not training_mode
            elif event.type == pygame.MOUSEBUTTONDOWN:
                cat.x, cat.y = event.pos
                cat.vx, cat.vy = 0, 0

        if paused:
            # Draw pause overlay
            if not training_mode:
                surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                surf.fill((0, 0, 0, 100))
                screen.blit(surf, (0, 0))
                txt = font_title.render("PAUSED", True, ACCENT)
                screen.blit(txt, (WIDTH // 2 - txt.get_width() // 2, HEIGHT // 2))
                pygame.display.flip()
            clock.tick(10)
            continue

        # ── Update loop ─────────────────────────────────────────────────
        iterations = 20 if training_mode else 1

        for _ in range(iterations):
            frame += 1
            survival_time += 1

            # Get ball movement
            if strategy == Strategy.REACTIVE:
                vx, vy = strategy_reactive(ball_pos, (cat.x, cat.y), obstacles)
                q_state, q_action = None, None
            elif strategy == Strategy.HIDE:
                vx, vy = strategy_hide(ball_pos, (cat.x, cat.y), obstacles)
                q_state, q_action = None, None
            elif strategy == Strategy.QLEARNING:
                vx, vy, q_state, q_action = q_agent.get_move(
                    ball_pos, (cat.x, cat.y), obstacles
                )
            elif strategy == Strategy.BEHAVIOR_TREE:
                vx, vy = bt_prey.get_move(ball_pos, (cat.x, cat.y), obstacles)
                q_state, q_action = None, None

            # Move ball with collision
            new_bx = ball_pos[0] + vx
            new_by = ball_pos[1] + vy
            ball_rect = pygame.Rect(new_bx - 8, new_by - 8, 16, 16)
            blocked = any(ball_rect.colliderect(o) for o in obstacles)
            if not blocked:
                ball_pos[0] = max(10, min(WIDTH - 10, new_bx))
                ball_pos[1] = max(10, min(HEIGHT - 10, new_by))

            # Update cat
            cat.update(ball_pos, obstacles, frame)

            # Check catch
            dist = math.hypot(ball_pos[0] - cat.x, ball_pos[1] - cat.y)
            caught = dist < CATCH_DIST

            # Q-learning update
            if strategy == Strategy.QLEARNING:
                visible = is_visible((cat.x, cat.y), ball_pos, obstacles)
                reward = 1.0  # survived this frame
                if not visible:
                    reward += 2.0  # bonus for being hidden
                if caught:
                    reward = -50.0

                if prev_state is not None and prev_action is not None:
                    q_agent.update(prev_state, prev_action, reward, q_state, caught)

                prev_state = q_state
                prev_action = q_action

            if caught:
                catches += 1
                total_runs += 1
                if survival_time > best_survival:
                    best_survival = survival_time
                run_history.append(survival_time)
                if len(run_history) > 50:
                    run_history.pop(0)

                if strategy == Strategy.QLEARNING:
                    q_agent.total_episodes += 1
                    q_agent.episode_reward = 0
                    q_agent.epsilon = max(0.05, q_agent.epsilon * 0.995)
                    prev_state, prev_action = None, None

                ball_pos, cat = reset()
                survival_time = 0
                bt_prey = BehaviorTreePrey()

        # ── RENDER ──────────────────────────────────────────────────────
        if training_mode:
            if frame % 600 == 0:  # Render occasionally during training
                pass
            else:
                clock.tick(0)  # Uncapped
                # Minimal display update
                if frame % 100 == 0:
                    screen.fill(BG_COLOR)
                    txt = font_title.render(
                        f"TRAINING MODE - Episodes: {q_agent.total_episodes}  "
                        f"Best: {best_survival}  Epsilon: {q_agent.epsilon:.3f}  "
                        f"States: {len(q_agent.q_table)}",
                        True,
                        ACCENT,
                    )
                    screen.blit(txt, (20, HEIGHT // 2))
                    hint = font.render(
                        "Press T to return to visual mode", True, TEXT_COLOR
                    )
                    screen.blit(hint, (20, HEIGHT // 2 + 30))
                    pygame.display.flip()
                continue

        screen.fill(BG_COLOR)

        # Draw subtle grid
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

        # Q-value heatmap
        if show_heatmap and strategy == Strategy.QLEARNING:
            heatmap_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            for state, values in q_agent.q_table.items():
                max_q = np.max(values)
                # Reconstruct approximate position from state
                # (This is an approximation for visualization)
                angle_bin, dist_bin, visible, wall_x, wall_y, obs_dir = state
                # Color based on max Q
                intensity = max(0, min(255, int((max_q + 10) * 5)))
                color = (0, intensity, intensity // 2, 40)
                # Draw at grid positions
                for x in range(0, WIDTH, CELL_SIZE * 2):
                    for y in range(0, HEIGHT, CELL_SIZE * 2):
                        s = q_agent.get_state((x, y), (cat.x, cat.y), obstacles)
                        if s == state:
                            pygame.draw.rect(
                                heatmap_surf,
                                color,
                                (x, y, CELL_SIZE * 2, CELL_SIZE * 2),
                            )
            screen.blit(heatmap_surf, (0, 0))

        # Visibility polygon
        if show_visibility:
            vis_poly = compute_visibility_polygon(
                (cat.x, cat.y), obstacles, num_rays=80
            )
            if len(vis_poly) > 2:
                vis_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                pygame.draw.polygon(vis_surf, (255, 200, 50, 18), vis_poly)
                screen.blit(vis_surf, (0, 0))

        # Draw hiding spots
        spots = find_hiding_spots(ball_pos, (cat.x, cat.y), obstacles)
        for (sx, sy), _ in spots[:5]:
            spot_surf = pygame.Surface((20, 20), pygame.SRCALPHA)
            pygame.draw.circle(spot_surf, (50, 200, 100, 60), (10, 10), 10)
            screen.blit(spot_surf, (sx - 10, sy - 10))

        # Draw obstacles
        for obs in obstacles:
            pygame.draw.rect(screen, OBSTACLE_COLOR, obs, border_radius=4)
            pygame.draw.rect(screen, OBSTACLE_BORDER, obs, width=1, border_radius=4)

        # Draw line of sight indicator
        ball_visible = is_visible((cat.x, cat.y), ball_pos, obstacles)
        if ball_visible:
            los_color = (255, 80, 80, 40)
        else:
            los_color = (80, 255, 80, 40)
        los_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        pygame.draw.line(los_surf, los_color, (cat.x, cat.y), ball_pos, 2)
        screen.blit(los_surf, (0, 0))

        # Draw ball (Sphero)
        glow_surf = pygame.Surface((40, 40), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (60, 180, 255, 40), (20, 20), 20)
        screen.blit(glow_surf, (ball_pos[0] - 20, ball_pos[1] - 20))
        pygame.draw.circle(screen, BALL_COLOR, (int(ball_pos[0]), int(ball_pos[1])), 10)
        pygame.draw.circle(
            screen, (150, 220, 255), (int(ball_pos[0]) - 3, int(ball_pos[1]) - 3), 3
        )

        # Ball status ring
        ring_color = DANGER_COLOR if ball_visible else HIDE_COLOR
        pygame.draw.circle(
            screen, ring_color, (int(ball_pos[0]), int(ball_pos[1])), 14, 2
        )

        # Draw cat
        cat_body_color = CAT_COLOR if not cat.is_pouncing else (255, 60, 40)
        pygame.draw.circle(screen, cat_body_color, (int(cat.x), int(cat.y)), 14)
        # Ears
        ear_offset = 10
        pygame.draw.polygon(
            screen,
            cat_body_color,
            [
                (int(cat.x) - ear_offset, int(cat.y) - 14),
                (int(cat.x) - ear_offset - 5, int(cat.y) - 24),
                (int(cat.x) - ear_offset + 5, int(cat.y) - 24),
            ],
        )
        pygame.draw.polygon(
            screen,
            cat_body_color,
            [
                (int(cat.x) + ear_offset, int(cat.y) - 14),
                (int(cat.x) + ear_offset - 5, int(cat.y) - 24),
                (int(cat.x) + ear_offset + 5, int(cat.y) - 24),
            ],
        )
        # Eyes
        look_dx = ball_pos[0] - cat.x
        look_dy = ball_pos[1] - cat.y
        look_dist = max(1, math.hypot(look_dx, look_dy))
        ex, ey = look_dx / look_dist * 3, look_dy / look_dist * 3
        pygame.draw.circle(
            screen, CAT_EYE, (int(cat.x) - 5 + int(ex), int(cat.y) - 2 + int(ey)), 3
        )
        pygame.draw.circle(
            screen, CAT_EYE, (int(cat.x) + 5 + int(ex), int(cat.y) - 2 + int(ey)), 3
        )
        pygame.draw.circle(
            screen,
            (20, 20, 20),
            (int(cat.x) - 5 + int(ex), int(cat.y) - 2 + int(ey)),
            1,
        )
        pygame.draw.circle(
            screen,
            (20, 20, 20),
            (int(cat.x) + 5 + int(ex), int(cat.y) - 2 + int(ey)),
            1,
        )

        # Interest indicator
        interest_w = 30
        pygame.draw.rect(
            screen,
            (40, 40, 50),
            (int(cat.x) - interest_w // 2, int(cat.y) + 20, interest_w, 4),
        )
        pygame.draw.rect(
            screen,
            CAT_EYE,
            (
                int(cat.x) - interest_w // 2,
                int(cat.y) + 20,
                int(interest_w * cat.interest),
                4,
            ),
        )

        # ── UI PANEL ────────────────────────────────────────────────────
        panel = pygame.Surface((280, HEIGHT - 20), pygame.SRCALPHA)
        panel.fill((25, 25, 35, 200))
        pygame.draw.rect(
            panel, (50, 50, 65), (0, 0, 280, HEIGHT - 20), 1, border_radius=6
        )

        y_offset = 15
        title = font_title.render("Pursuit-Evasion Sim", True, ACCENT)
        panel.blit(title, (15, y_offset))
        y_offset += 35

        # Strategy
        strat_txt = font_large.render(strategy_names[strategy], True, (255, 255, 255))
        panel.blit(strat_txt, (15, y_offset))
        y_offset += 22
        desc = font.render(strategy_desc[strategy], True, (150, 150, 165))
        # Word wrap description
        words = strategy_desc[strategy].split()
        line = ""
        for w in words:
            test = line + " " + w if line else w
            if font.size(test)[0] > 250:
                panel.blit(font.render(line, True, (150, 150, 165)), (15, y_offset))
                y_offset += 16
                line = w
            else:
                line = test
        if line:
            panel.blit(font.render(line, True, (150, 150, 165)), (15, y_offset))
            y_offset += 25

        # Stats
        y_offset += 5
        pygame.draw.line(panel, (50, 50, 65), (15, y_offset), (265, y_offset))
        y_offset += 10

        stats = [
            ("Survival", f"{survival_time // 60:.0f}s ({survival_time} frames)"),
            ("Best Run", f"{best_survival // 60:.0f}s ({best_survival} frames)"),
            ("Catches", str(catches)),
            ("Visible", "YES" if ball_visible else "HIDDEN"),
            ("Cat Interest", f"{cat.interest:.1%}"),
        ]

        if strategy == Strategy.QLEARNING:
            stats.extend(
                [
                    ("Episodes", str(q_agent.total_episodes)),
                    ("Epsilon", f"{q_agent.epsilon:.3f}"),
                    ("Q-States", str(len(q_agent.q_table))),
                ]
            )

        if strategy == Strategy.BEHAVIOR_TREE:
            stats.append(("BT State", bt_prey.state.name))

        for label, value in stats:
            color = (
                DANGER_COLOR if label == "Visible" and value == "YES" else TEXT_COLOR
            )
            if label == "Visible" and value == "HIDDEN":
                color = HIDE_COLOR
            panel.blit(font.render(f"{label}:", True, (120, 120, 135)), (15, y_offset))
            panel.blit(font.render(value, True, color), (130, y_offset))
            y_offset += 18

        # Survival history sparkline
        if run_history:
            y_offset += 15
            panel.blit(
                font_large.render("Survival History", True, TEXT_COLOR), (15, y_offset)
            )
            y_offset += 22
            max_h = max(run_history) if run_history else 1
            bar_w = max(1, 240 // len(run_history))
            for i, s in enumerate(run_history):
                h = max(1, int(s / max(max_h, 1) * 50))
                color = ACCENT if s == best_survival else (80, 120, 160)
                pygame.draw.rect(
                    panel,
                    color,
                    (15 + i * bar_w, y_offset + 50 - h, max(1, bar_w - 1), h),
                )
            y_offset += 60

        # Controls
        y_offset += 10
        pygame.draw.line(panel, (50, 50, 65), (15, y_offset), (265, y_offset))
        y_offset += 10
        panel.blit(font_large.render("Controls", True, TEXT_COLOR), (15, y_offset))
        y_offset += 22

        controls = [
            "1-4: Switch strategy",
            "R: Reset  SPACE: Pause",
            "V: Toggle visibility",
            "H: Toggle Q-heatmap",
            "T: Training mode (fast)",
            "Click: Move cat",
        ]
        for c in controls:
            panel.blit(font.render(c, True, (120, 120, 135)), (15, y_offset))
            y_offset += 16

        screen.blit(panel, (WIDTH - 290, 10))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
