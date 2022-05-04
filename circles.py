from datetime import date
from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
from configparser import ConfigParser
import numpy
import random
import sys
import copy

TARGET = Image.open("images/darwin.png")
MAX = 255 * TARGET.size[0] * TARGET.size[1]


def random_int(global_min, global_max, local_min, local_max):
    num = -1000000
    while num < global_min or num > global_max:
        num = random.randint(local_min, local_max)

    return num


def make_ellipse():
    centre = (random.randint(10, 190), random.randint(10, 190))
    ellipse = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))
               , centre
               , (random.randint(0, centre[0]), random.randint(0, centre[1]))
               , (random.randint(centre[0], 200), random.randint(centre[1], 200))]

    return ellipse


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.ellipse(polygon[2:], fill=polygon[0])

    return image


def initialize():
    return [make_ellipse() for i in range(random.randint(1, 10))]


def evaluate(x):
    image = draw(x)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def select(population):
    return [random.choice(population) for i in range(2)]


def combine(*parents):
    child = []
    for ellipse in parents[0]:
        if ellipse[1][0] <= 100 and len(child) < 100:
            child.append(copy.deepcopy(ellipse))
    for ellipse in parents[1]:
        if ellipse[1][0] > 100 and len(child) < 100:
            child.append(copy.deepcopy(ellipse))

    return child


def mutate(x, add_rate, vertex_rate):
    if random.random() < add_rate:
        if random.choice([True, True, False]) and len(x) < 100:
            x.append(make_ellipse())
        elif len(x) > 0:
            x.pop(random.randint(0, len(x) - 1))

    for ellipse in x:
        if random.random() < vertex_rate:
            if random.choice([True, False]):
                i = random.randint(2, len(ellipse) - 1)
                point = ellipse[i]
                if i == 2:
                    ellipse[i] = (random_int(0, ellipse[1][0], point[0] - 10, point[0] + 10)
                                  , random_int(0, ellipse[1][1], point[1] - 10, point[1] + 10))
                else:
                    ellipse[i] = (random_int(ellipse[1][0], 200, point[0] - 10, point[0] + 10)
                                  , random_int(ellipse[1][1], 200, point[1] - 10, point[1] + 10))


            else:
                colour = ellipse[0]
                ellipse[0] = (random_int(0, 255, colour[0] - 10, colour[0] + 10)
                              , random_int(0, 255, colour[1] - 10, colour[1] + 10)
                              , random_int(0, 255, colour[2] - 10, colour[2] + 10)
                              , random.randint(colour[3] - 10, colour[3] + 10))

    return x



def run(pop_size, maximize, survival_rate, vertex_rate, add_rate, generations, seed, save):
    random.seed(seed)
    population = Population.generate(initialize, evaluate, size=int(pop_size), maximize=bool(maximize))
    population.evaluate()

    evolution1 = (Evolution().survive(fraction=float(survival_rate))
                  .breed(parent_picker=select, combiner=combine)
                  .mutate(mutate_function=mutate, vertex_rate=float(vertex_rate), add_rate=float(add_rate)
                          , elitist=True)
                  .evaluate())

    for i in range(int(generations)):
        population = population.evolve(evolution1)
        sd = standard_deviation(population.individuals)

        print("Gen =", i, " Best =", population.current_best.fitness, " Worst =", population.current_worst.fitness
              , "Polygons =", len(population.current_best.chromosome), "Standard Deviation =", sd)

    mean = calc_mean(population.individuals)

    if save:
        image = draw(population.current_best.chromosome)
        image.save("images/solution.png")

        save_test("basic", population.current_best.fitness, population.current_worst.fitness, mean, pop_size
                  , survival_rate, vertex_rate, add_rate, generations)


def read_config(path):
    config = ConfigParser()
    config.read(path)

    values = {section: dict(config.items(section)) for section in config.sections()}

    return values


def calc_mean(population):
    total = 0
    for individual in population:
        total += individual.fitness

    return total/len(population)


def standard_deviation(population):
    fitness_list = [i.fitness for i in population]
    return numpy.std(fitness_list)


def save_test(evol_type, current_best, current_worst, mean, pop_size, survival_rate, vertex, add, generations):
    current_date = date.today().strftime("%d/%m/%y")
    f = open("test_results.md", 'a')
    row = "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |\n".format(current_date, evol_type, current_best
                                                                         , current_worst, mean, pop_size, survival_rate
                                                                         , vertex, add, generations)
    f.write(row)
    f.close()


if __name__ == "__main__":
    params = read_config(sys.argv[1])
    run(**params[sys.argv[2]])
