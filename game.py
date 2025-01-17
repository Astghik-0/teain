from env import FONT_SIZE, RacingEnv,SCREEN_WIDTH ,SCREEN_HEIGHT
import pygame 
from PIL import Image, ImageSequence
import sys
FONT_SIZE =36
WHITE =(255,255,255)
BLACK=(0,0,0)

def play_gif(screen, gif_path):
    """Play a GIF file on the Pygame screen."""
    # Load GIF using Pillow
    gif = Image.open(gif_path)
    frames = [frame.copy() for frame in ImageSequence.Iterator(gif)]

    clock = pygame.time.Clock()

    for frame in frames:
        # Convert the frame to a surface
        mode = frame.mode
        size = frame.size
        data = frame.tobytes()
        frame_surface = pygame.image.fromstring(data, size, mode)
        
        # Display the frame
        screen.fill((0, 0, 0))
        screen.blit(frame_surface, ((SCREEN_WIDTH - size[0]) // 2, (SCREEN_HEIGHT - size[1]) // 2))
        pygame.display.flip()

        # Wait for the next frame
        clock.tick(2)  # Adjust this to control playback speed (2 FPS in this case)

    # Wait briefly after playing the GIF
    pygame.time.wait(1000)

def starting_screen(screen,font):
    """Display the starting screen with options to start the game or view a video."""
    start_screen_active = True
    while start_screen_active:
        screen.fill(WHITE)
        # Draw title
        title_text = font.render("formula1", True, BLACK)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 4))
        screen.blit(title_text, title_rect)

        # Draw buttons
        start_button = pygame.Rect(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2, 50, 50)
        video_button = pygame.Rect(SCREEN_WIDTH // 4, SCREEN_HEIGHT // 2 + 50 ,50,50)

        pygame.draw.rect(screen, (187, 173, 160), start_button)
        pygame.draw.rect(screen, (187, 173, 160), video_button)

        start_text = font.render("Start Game", True, BLACK)
        video_text = font.render("Watch Video", True, BLACK)
        
        screen.blit(start_text, start_text.get_rect(center=start_button.center))
        screen.blit(video_text, video_text.get_rect(center=video_button.center))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if start_button.collidepoint(event.pos):
                    start_screen_active = False  # Start the game
                elif video_button.collidepoint(event.pos):
                    play_gif(screen, "train.gif")

if __name__ == "__main__":
    env = RacingEnv()
    done = False
    env.reset()

    screen = env.screen
    font = env.font
    starting_screen(screen, font)
    # Game loop
    while not done:
        env.render()
        action = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    action = 2
                if event.key == pygame.K_LEFT:
                    action = 1
        
        obs, reward, done, truncated, info = env.step(action)

    pygame.quit()
