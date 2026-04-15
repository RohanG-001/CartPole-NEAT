import pygame
import neat
import pickle
import random
from pong import CartPole

# Load trained genome
with open("best_genome.pkl", "rb") as f:
    winner = pickle.load(f)

# Load config
config_path = "config.txt"
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    config_path)

# Create neural network from genome
net = neat.nn.FeedForwardNetwork.create(winner, config)

# Initialize pygame
pygame.init()
screen = pygame.display.set_mode((800, 600))
pygame.display.set_caption("CartPole - Trained AI")
clock = pygame.time.Clock()

# Create game instance
game = CartPole()
state = game.reset()

# Main game loop
running = True
while running:
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
    
    # Get action from neural network with noise
    output = net.activate(state)
    noise = random.uniform(-0.8, 0.8)
    action = 1 if output[0] + noise > 0.5 else 0
    
    # Step game
    state, reward, done, info = game.step(action)
    
    # Stop if game over (no auto-reset)
    if done:
        # Game over - pole fell
        running = False
    
    # Render
    game.render(screen)
    pygame.display.flip()
    clock.tick(50)

pygame.quit()