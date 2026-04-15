import pygame
import math
import random

# Physics constants
GRAVITY = 9.8
MASS_CART = 1.0
MASS_POLE = 0.1
TOTAL_MASS = MASS_CART + MASS_POLE
LENGTH = 0.5  # Actually half the pole's length
POLEMASS_LENGTH = MASS_POLE * LENGTH
FORCE_MAG = 10.0  # Standard cartpole force
TAU = 0.01  # Time step in seconds

# Display constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCALE = 100  # Pixels per meter
CART_WIDTH = 50
CART_HEIGHT = 30
POLE_WIDTH = 6

class CartPole:
    """CartPole game environment with physics simulation."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset the cartpole to initial state - challenging but fair."""
        self.x = random.uniform(-2.0, 2.0)  # Random cart position
        self.x_dot = random.uniform(-1.0, 1.0)  # Random cart velocity
        self.theta = random.uniform(-0.15, 0.15)  # ~8.5 degrees - challenging
        self.theta_dot = random.uniform(-1.0, 1.0)  # Random angular velocity
        self.xacc = 0.0  # Cart acceleration
        self.tacc = 0.0  # Angular acceleration
        self.done = False
        self.steps_beyond_done = None
        return self.get_state()
    
    def get_state(self):
        """Get current state as [x, x_dot, theta, theta_dot]."""
        return [self.x, self.x_dot, self.theta, self.theta_dot]
    
    def step(self, action):
        """
        Apply action (0: left, 1: right) and simulate one timestep using leapfrog integration.
        Returns (state, reward, done, info)
        """
        if self.done:
            return self.get_state(), 0.0, True, {}
        
        force = FORCE_MAG if action == 1 else -FORCE_MAG
        
        # Leapfrog integration (more accurate than Euler)
        g = GRAVITY
        mp = MASS_POLE
        mc = MASS_CART
        mt = mp + mc
        L = LENGTH
        dt = TAU
        
        st = math.sin(self.theta)
        ct = math.cos(self.theta)
        
        # Compute accelerations
        theta_acc = (g * st + ct * (-force - mp * L * self.theta_dot ** 2 * st) / mt) / (L * (4.0 / 3 - mp * ct ** 2 / mt))
        x_acc = (force + mp * L * (self.theta_dot ** 2 * st - theta_acc * ct)) / mt
        
        # Update position/angle
        self.x += dt * self.x_dot + 0.5 * self.xacc * dt ** 2
        self.theta += dt * self.theta_dot + 0.5 * self.tacc * dt ** 2
        
        # Store old accelerations
        tacc0 = self.tacc
        xacc0 = self.xacc
        
        # Recompute accelerations at new position
        st = math.sin(self.theta)
        ct = math.cos(self.theta)
        theta_acc = (g * st + ct * (-force - mp * L * self.theta_dot ** 2 * st) / mt) / (L * (4.0 / 3 - mp * ct ** 2 / mt))
        x_acc = (force + mp * L * (self.theta_dot ** 2 * st - theta_acc * ct)) / mt
        
        # Update velocities
        self.x_dot += 0.5 * (xacc0 + x_acc) * dt
        self.theta_dot += 0.5 * (tacc0 + theta_acc) * dt
        
        # Store accelerations
        self.tacc = theta_acc
        self.xacc = x_acc
        
        # Check termination conditions (only cart position, no angle limit)
        self.done = bool(
            self.x < -2.4
            or self.x > 2.4
        )
        
        if not self.done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0
        
        return self.get_state(), reward, self.done, {}
    
    def render(self, screen):
        """Render the cartpole to pygame screen."""
        screen.fill((255, 255, 255))
        
        # Calculate screen coordinates
        center_x = SCREEN_WIDTH // 2
        ground_y = SCREEN_HEIGHT - 100
        
        cart_x = center_x + int(self.x * SCALE)
        cart_y = ground_y - CART_HEIGHT // 2
        
        # Draw ground
        pygame.draw.line(screen, (0, 0, 0), (0, ground_y), (SCREEN_WIDTH, ground_y), 2)
        
        # Draw cart
        cart_rect = pygame.Rect(
            cart_x - CART_WIDTH // 2,
            cart_y,
            CART_WIDTH,
            CART_HEIGHT
        )
        pygame.draw.rect(screen, (50, 50, 200), cart_rect)
        
        # Draw pole
        pole_end_x = cart_x + int(LENGTH * 2 * SCALE * math.sin(self.theta))
        pole_end_y = cart_y - int(LENGTH * 2 * SCALE * math.cos(self.theta))
        pygame.draw.line(screen, (200, 50, 50), (cart_x, cart_y), (pole_end_x, pole_end_y), POLE_WIDTH)
        
        # Draw pivot point
        pygame.draw.circle(screen, (0, 0, 0), (cart_x, cart_y), 5)
        
        # Draw info text
        font = pygame.font.Font(None, 36)
        state_text = f"x: {self.x:.2f}m  θ: {math.degrees(self.theta):.1f}°"
        text = font.render(state_text, True, (0, 0, 0))
        screen.blit(text, (10, 10))
        
        if self.done:
            done_text = font.render("FAILED - Press R to reset", True, (255, 0, 0))
            screen.blit(done_text, (center_x - 150, 50))


def main():
    """Main game loop for manual control."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("CartPole Game")
    clock = pygame.time.Clock()
    
    game = CartPole()
    
    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    game.reset()
        
        # Get manual input
        keys = pygame.key.get_pressed()
        action = 1 if keys[pygame.K_RIGHT] else (0 if keys[pygame.K_LEFT] else None)
        
        # Step game if action provided
        if action is not None and not game.done:
            state, reward, done, info = game.step(action)
        
        # Render
        game.render(screen)
        pygame.display.flip()
        clock.tick(50)  # 50 FPS
    
    pygame.quit()


if __name__ == "__main__":
    main()
