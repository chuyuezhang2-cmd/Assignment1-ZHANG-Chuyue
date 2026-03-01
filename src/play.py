import pygame
import sys
import time
from maze_env import MazeEnv
from dqn_agent import DQNAgent

# 全局初始化 pygame
pygame.init()

# 颜色定义
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 200)
GREEN = (50, 200, 50)
RED = (220, 50, 50)
GRAY = (200, 200, 200)
LIGHT_BLUE = (230, 240, 255)
DARK_GRAY = (80, 80, 80)
BUTTON_HOVER = (255, 255, 200)

# 字体
FONT_TITLE = pygame.font.SysFont("arial", 52, bold=True)
FONT_RULES = pygame.font.SysFont("arial", 24)
FONT_BUTTON = pygame.font.SysFont("arial", 34, bold=True)
FONT_HEADER = pygame.font.SysFont("arial", 24, bold=True)

def draw_text(surface, text, font, color, x, y, center=False):
    text_surf = font.render(text, True, color)
    text_rect = text_surf.get_rect()
    if center:
        text_rect.center = (x, y)
    else:
        text_rect.topleft = (x, y)
    surface.blit(text_surf, text_rect)

def main_menu():
    screen = pygame.display.set_mode((600, 700))
    pygame.display.set_caption("Dynamic Maze Explorer - Main Menu")
    clock = pygame.time.Clock()

    button_ai = pygame.Rect(150, 450, 300, 70)
    button_manual = pygame.Rect(150, 540, 300, 70)
    button_quit = pygame.Rect(150, 630, 300, 70)

    while True:
        mouse_pos = pygame.mouse.get_pos()
        screen.fill(LIGHT_BLUE)

        draw_text(screen, "Dynamic Maze Explorer", FONT_TITLE, DARK_GRAY, 300, 80, center=True)

        pygame.draw.line(screen, GRAY, (80, 140), (520, 140), 4)

        # 游戏规则说明
        rules = [
            "Game Rules:",
            "  • Reach the GREEN GOAL to WIN! Start with 6 lives!",
            "  • Collect YELLOW coins for +30 points",
            "  • Avoid moving RED traps (-100 & lose life)",
            "  • BLACK walls block your path",
            "  • AI Mode: Watch the agent learn",
            "  • Manual Mode: Arrow keys + Space (stay)",
            "  • Press 'M' to switch modes during play",
        ]
        y = 160
        for line in rules:
            draw_text(screen, line, FONT_RULES, BLACK, 80, y, center=False)
            y += 35

        pygame.draw.line(screen, GRAY, (80, 410), (520, 410), 3)

        # 按钮区域
        ai_hover = button_ai.collidepoint(mouse_pos)
        manual_hover = button_manual.collidepoint(mouse_pos)
        quit_hover = button_quit.collidepoint(mouse_pos)

        # AI 按钮（绿色）
        pygame.draw.rect(screen, BUTTON_HOVER if ai_hover else GREEN, button_ai, border_radius=25)
        pygame.draw.rect(screen, DARK_GRAY, button_ai, 4, border_radius=25)
        draw_text(screen, "Start AI Mode", FONT_BUTTON, WHITE, button_ai.centerx, button_ai.centery, center=True)

        # Manual 按钮（蓝色）
        pygame.draw.rect(screen, BUTTON_HOVER if manual_hover else BLUE, button_manual, border_radius=25)
        pygame.draw.rect(screen, DARK_GRAY, button_manual, 4, border_radius=25)
        draw_text(screen, "Start Manual Mode", FONT_BUTTON, WHITE, button_manual.centerx, button_manual.centery, center=True)

        # Quit 按钮（红色）
        pygame.draw.rect(screen, BUTTON_HOVER if quit_hover else RED, button_quit, border_radius=25)
        pygame.draw.rect(screen, DARK_GRAY, button_quit, 4, border_radius=25)
        draw_text(screen, "Quit", FONT_BUTTON, WHITE, button_quit.centerx, button_quit.centery, center=True)

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if button_ai.collidepoint(mouse_pos):
                    return "ai"
                if button_manual.collidepoint(mouse_pos):
                    return "manual"
                if button_quit.collidepoint(mouse_pos):
                    pygame.quit()
                    sys.exit()

        clock.tick(60)

def play(selected_mode):
    env = MazeEnv()
    state_size = env.grid_size ** 2
    action_size = 5
    agent = DQNAgent(state_size, action_size)
    model_loaded = False
    for model_file in ["trained_model_ep800.pth", "trained_model_final.pth", "trained_model_ep700.pth", "trained_model.pth"]:
        try:
            agent.load(model_file)
            print(f"Model loaded successfully: {model_file}")
            model_loaded = True
            break
        except FileNotFoundError:
            continue
    if not model_loaded:
        print("Warning: No model found, using random policy.")

    screen = pygame.display.set_mode((500, 550))
    pygame.display.set_caption("Dynamic Maze Explorer - Game")
    clock = pygame.time.Clock()

    manual_mode = (selected_mode == "manual")
    running = True
    paused = False
    game_over = False

    state = env.reset()
    total_reward = 0
    lives = 6
    level = 1
    reward_text = "Last Reward: 0"
    action_text = "Action: Stay"
    collision_text = ""
    lives_text = f"Lives: {lives}"
    level_text = f"Level: {level}"
    mode_text = "Mode: Manual (Arrow keys + Space)" if manual_mode else "Mode: AI (Press M to switch)"
    total_reward_text = f"Total: {total_reward:.1f}"
    game_over_text = ""

    print("Game started: Initial lives=6, level=1")

    while running:
        collision = False
        action = None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m and not game_over:
                    manual_mode = not manual_mode
                    mode_text = "Mode: Manual (Arrow keys + Space)" if manual_mode else "Mode: AI (Press M to switch)"
                if event.key == pygame.K_p:
                    paused = not paused
                if event.key == pygame.K_r:
                    state = env.reset()
                    total_reward = 0
                    lives = 6
                    level = 1
                    game_over = False
                    lives_text = f"Lives: {lives}"
                    level_text = f"Level: {level}"
                    print("Game reset: lives=6, level=1")
                if game_over:
                    if event.key == pygame.K_q:
                        running = False
                if manual_mode and not game_over:
                    if event.key == pygame.K_UP: action = 0
                    elif event.key == pygame.K_DOWN: action = 1
                    elif event.key == pygame.K_LEFT: action = 2
                    elif event.key == pygame.K_RIGHT: action = 3
                    elif event.key == pygame.K_SPACE: action = 4

        if paused or game_over:
            if game_over:
                font_big = pygame.font.SysFont(None, 60)
                text_surf = font_big.render(game_over_text, True, RED)
                text_rect = text_surf.get_rect(center=(250, 275))
                screen.blit(text_surf, text_rect)
            pygame.display.flip()
            clock.tick(10)
            continue

        if manual_mode:
            if action is None:
                action = 4
        else:
            action = agent.act(state)

        next_state, reward, done = env.step(action)
        state = next_state
        total_reward += reward

        if reward < -1:
            collision = True

        reward_text = f"Last Reward: {reward:.1f}"
        action_text = f"Action: {['Up', 'Down', 'Left', 'Right', 'Stay'][action]}"
        total_reward_text = f"Total: {total_reward:.1f}"
        collision_text = "Collision!" if collision else ""

        if done:
            print(f"Episode ended | Reward: {total_reward:.1f} | Steps: {env.steps}")
            if reward >= 100:
                level += 1
                env.num_traps += 1
                level_text = f"Level: {level}"
            else:
                lives -= 1
                lives_text = f"Lives: {lives}"
                if lives <= 0:
                    game_over = True
                    game_over_text = "GAME OVER! Press R to Restart, Q to Quit"
            state = env.reset()
            total_reward = 0

        # 渲染头部信息
        screen.fill(LIGHT_BLUE)
        font = FONT_HEADER
        draw_text(screen, mode_text, font, BLACK, 10, 10, center=False)
        draw_text(screen, lives_text, font, RED, 280, 10, center=False)
        draw_text(screen, level_text, font, BLACK, 400, 10, center=False)
        draw_text(screen, total_reward_text, font, GREEN, 10, 40, center=False)
        draw_text(screen, collision_text, font, RED, 280, 40, center=False)

        sub_screen = screen.subsurface((0, 70, 500, 480))
        q_vals = agent.get_q_values(state)
        env.render(sub_screen, collision=collision, show_q_values=q_vals)

        pygame.display.flip()

        if not manual_mode:
            time.sleep(0.12)
        else:
            time.sleep(0.03)
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    mode = main_menu()
    if mode in ["ai", "manual"]:
        play(mode)