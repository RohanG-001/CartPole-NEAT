import neat
import os
from pong import CartPole
import pickle

def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        # Create neural network from genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        
        # Run multiple episodes and average fitness
        fitness = 0
        num_episodes = 20
        max_steps = 1000
        
        for episode in range(num_episodes):
            # Create game instance and reset
            game = CartPole()
            state = game.reset()
            
            # Run game loop
            episode_fitness = 0
            steps = 0
            
            while steps < max_steps:
                # Feed state to network and get action
                output = net.activate(state)
                action = 1 if output[0] > 0.5 else 0
                
                # Step game
                state, reward, done, info = game.step(action)
                episode_fitness += reward
                steps += 1
                
                # Stop if game over
                if done:
                    break
            
            fitness += episode_fitness
        
        # Assign average fitness to genome
        genome.fitness = fitness / num_episodes

def run_neat():
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Create population
    population = neat.Population(config)
    
    # Add reporter to see progress
    population.add_reporter(neat.StdOutReporter(True))
    
    # Run evolution
    winner = population.run(eval_genomes, 28)
    
    # Save best genome
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    print(f"\nBest genome saved to best_genome.pkl with fitness {winner.fitness}")

if __name__ == "__main__":
    run_neat()