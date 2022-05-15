"""
This is the representation variant.

# @Create Date: 13/05/2022
# @Author  : Alfie Fields
# @File    : representation_variant.py

"""
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
from configparser import ConfigParser
import matplotlib.pyplot as plt
import numpy
import random
import sys
import copy

TARGET = Image.open("images/darwin.png")
MAX = 255 * TARGET.size[0] * TARGET.size[1]


def random_int(global_min, global_max, local_min, local_max):
    """ Create a random number within local and global range. """
    num = -1000000
    while num < global_min or num > global_max:
        num = random.randint(local_min, local_max)

    return num


def make_ellipse(prob_small, colour):
    """ Create one ellipse to add to an individual. """
    if colour:
        ellipse = [colour]
    else:
        ellipse = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))]

    centre = (random.randint(10, 190), random.randint(10, 190))
    ellipse.append(centre)
    if not random.random() < prob_small:
        ellipse.append((random.randint(0, centre[0]), random.randint(0, centre[1])))
        ellipse.append((random.randint(centre[0], 200), random.randint(centre[1], 200)))
    else:
        ellipse.append((random_int(0, 200, centre[0] - 5, centre[0]),
                        random_int(0, 200, centre[1] - 5, centre[1])))
        ellipse.append((random_int(0, 200, centre[0], centre[0] + 5),
                        random_int(0, 200, centre[1], centre[1] + 5)))

    return ellipse


def draw(solution):
    """ Draw an individual on a canvas. """
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.ellipse(polygon[2:], fill=polygon[0])

    return image


def initialize():
    """ Create initial population. """
    return [make_ellipse(0.5, None) for i in range(random.randint(1, 10))]


def evaluate(x):
    """ Calculate the fitness of an individual. """
    image = draw(x)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def select(population):
    """ Select parents for crossover. """
    return [population[0]]


def combine(*parents):
    """ Generate offspring from parents, by combining left and right. """
    return copy.deepcopy(parents[0])


def mutate(x, add_rate, vertex_rate, prob_small):
    """ Mutate offsprings genome. """
    if random.random() < add_rate:
        if random.choice([True, False]) and len(x) < 100:
            ellipse_colour = random.choice(x)[0]
            x.append(make_ellipse(prob_small, ellipse_colour))
        elif len(x) > 0:
            x.pop(random.randint(0, len(x) - 1))

    for ellipse in x:
        if random.random() < vertex_rate:
            if random.choice([True, False]):
                i = random.randint(2, len(ellipse) - 1)
                point = ellipse[i]
                if i == 2:
                    ellipse[i] = (random_int(0, ellipse[3][0], point[0] - 10, point[0] + 10),
                                  random_int(0, ellipse[3][1], point[1] - 10, point[1] + 10))


                else:
                    ellipse[i] = (random_int(ellipse[2][0], 200, point[0] - 10, point[0] + 10),
                                  random_int(ellipse[2][1], 200, point[1] - 10, point[1] + 10))

                ellipse[1] = ((ellipse[3][0] - ellipse[2][0]) / 2, (ellipse[3][1] - ellipse[2][1]) / 2)

            else:
                colour = ellipse[0]
                ellipse[0] = (random_int(0, 255, colour[0] - 10, colour[0] + 10),
                              random_int(0, 255, colour[1] - 10, colour[1] + 10),
                              random_int(0, 255, colour[2] - 10, colour[2] + 10),
                              random.randint(colour[3] - 10, colour[3] + 10))

    return x



def run(pop_size, maximize, survival_rate, vertex_rate, add_rate, seed, save):
    """ Run evolution and perform analysis. """
    fitness_evaluations = []
    for j in range(1):
        random.seed(int(seed) + j)
        population = Population.generate(initialize, evaluate, size=int(pop_size), maximize=bool(maximize))
        population.evaluate()
        count = 1
        mean = []
        gen = []

        evolution1 = (Evolution().survive(n=int(survival_rate)).
                      breed(parent_picker=select, combiner=combine).
                      mutate(mutate_function=mutate, vertex_rate=float(vertex_rate), add_rate=float(add_rate),
                             prob_small=0.75, elitist=True).
                      evaluate())

        while population.current_best.fitness < 0.95 and count < 6000:
            population = population.evolve(evolution1)
            sd = standard_deviation(population.individuals)

            print("Gen =", count, " Best =", population.current_best.fitness, " Worst =",
                  population.current_worst.fitness, "Polygons =", len(population.current_best.chromosome),
                  "Standard Deviation =", sd)

            gen.append(count)
            mean.append(calc_mean(population.individuals))
            count += 1
            image = draw(population.current_best.chromosome)
            image.save("images/solution.png")

        fitness_evaluations.append(count)

        image = draw(population.current_best.chromosome)
        image.save("images/solution" + str(j + 1) + ".png")
        line_colour = ['b', 'g', 'r', 'c', 'm']
        plt.plot(gen, mean, line_colour[j])

    if save:
        save_test(fitness_evaluations, pop_size, float(survival_rate), float(vertex_rate), float(add_rate))

    plt.ylabel("Mean Fitness")
    plt.xlabel("Generation")
    plt.show()


def read_config(path):
    """ Read values from config file. """
    config = ConfigParser()
    config.read(path)

    values = {section: dict(config.items(section)) for section in config.sections()}

    return values


def calc_mean(population):
    """ Calculate the mean fitness of given population. """
    total = 0
    for individual in population:
        total += individual.fitness

    return total/len(population)


def standard_deviation(population):
    """ Calculate standard deviation of given population. """
    fitness_list = [i.fitness for i in population]
    return numpy.std(fitness_list)


def save_test(generations, pop, survival, vertex, add):
    """ Analise generations and save to markdown. """
    minimum = 100000
    maximum = -100000
    total = 0
    for i in generations:
        if i < minimum:
            minimum = i
        if i > maximum:
            maximum = i
        total += i

    mean = total / len(generations)

    f = open("tests.md", 'a')
    row = "| Representation | {} | {} | {} | {} | {} | {} | {} |\n".format(minimum, maximum, mean, pop, survival,
                                                                           vertex, add)
    f.write(row)
    f.close()


if __name__ == "__main__":
    params = read_config(sys.argv[1])
    run(**params[sys.argv[2]])
