import argparse
import gym
from gym.wrappers import Monitor
import gym_ple
import os
import logging
import numpy as np
import multiprocessing
from copy import deepcopy
from random import randrange, sample
from neat import nn, population, statistics

parser = argparse.ArgumentParser(description='OpenAI Gym Solver')
parser.add_argument('--max-steps', dest='max_steps', type=int, default=5000,
                    help='The max number of steps to take per genome (timeout)')
parser.add_argument('--episodes', type=int, default=1,
                    help="The number of times to run a single genome. This takes the fitness score from the worst run")
parser.add_argument('--generations', type=int, default=50,
                    help="The number of generations to evolve the network")
parser.add_argument('--checkpoint', type=str,
                    help="Uses a checkpoint to start the simulation")
parser.add_argument('--tilde', type=bool, default=True,
                    help="Set False for execution mario with meta inputs. This working more slow and it consumes more processing but evolves better")
args = parser.parse_args()

logger = logging.getLogger()
logger.setLevel(logging.INFO)

my_env = gym.make('FlappyBird-v0')


def simulate_species(net, episodes=1, steps=5000):
    fitnesses = []

    for runs in range(episodes):
        inputs = my_env.reset()
        cum_reward = 0.0
        cont = 0;
        for j in range(steps):
            inputsNeat inputsNeat inputs.flatten()
            print(inputsNeat)
            action = net.serial_activate(inputsNeat)
            action = round(action[0], 1)
            inputs, reward, done, _ = my_env.step(action)
            cum_reward = cum_reward + reward
            if done:
                break
        fitnesses.append(reward)

    fitness = np.array(fitnesses).mean()
    print("Species fitness: %s" % str(fitness))
    return fitness

def worker_evaluate_genome(g):
    net = nn.create_feed_forward_phenotype(g)
    return simulate_species(net, args.episodes, args.max_steps)

def train_network():
    def evaluate_genome(g):
        net = nn.create_feed_forward_phenotype(g)
        return simulate_species(net, args.episodes, args.max_steps)

    def eval_fitness(genomes):
        for g in genomes:
            fitness = evaluate_genome(g)
            g.fitness = fitness

    # Simulation
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'game_config')
    pop = population.Population(config_path)
    # Load checkpoint
    if args.checkpoint:
        pop.load_checkpoint(args.checkpoint)
    # Start simulation
    pop.run(eval_fitness, args.generations)

    pop.save_checkpoint("checkpoint")

    # Log statistics.
    statistics.save_stats(pop.statistics)
    statistics.save_species_count(pop.statistics)
    statistics.save_species_fitness(pop.statistics)

    print('Number of evaluations: {0}'.format(pop.total_evaluations))

    # Show output of the most fit genome against training data.
    winner = pop.statistics.best_genome()

    # Save best network
    import pickle
    with open('winner.pkl', 'wb') as output:
        pickle.dump(winner, output, 1)

    print('\nBest genome:\n{!s}'.format(winner))
    print('\nOutput:')

    raw_input("Press Enter to run the best genome...")
    winner_net = nn.create_feed_forward_phenotype(winner)
    for i in range(100):
        simulate_species(winner_net, 1, args.max_steps)

train_network()
