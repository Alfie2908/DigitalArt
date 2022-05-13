from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
from configparser import ConfigParser
import matplotlib.pyplot as plt
import math
import numpy
import random
import sys
import copy

TARGET = Image.open("images/medium.png")
MAX = 255 * TARGET.size[0] * TARGET.size[1]


def polygon_centre(polygon):
    sum_x = 0
    sum_y = 0
    count = 0
    for i in range(1, len(polygon)):
        vertex = polygon[i]
        sum_x += vertex[0]
        sum_y += vertex[1]
        count += 1

    return sum_x/count, sum_y/count


def get_angle(point1, point2):
    if point1[1] == point2[1] and point2[0] > point1[0]:
        return 0.5 * math.pi
    elif point1[1] == point2[1] and point2[0] < point1[0]:
        return 0.75 * math.pi
    elif point1 == point2:
        return 0

    theta = math.atan2((point2[0] - point1[0]), (point2[1] - point1[1]))

    if theta < 0:
        return theta + (2 * math.pi)
    else:
        return theta


def dual_insertion_sort(list1, list2):
    for i in range(1, len(list1)):
        key = list1[i]
        item = list2[i + 1]
        j = i - 1

        while j >= 0 and list1[j] > key:
            list1[j + 1] = list1[j]
            list2[j + 2] = list2[j + 1]
            j = j - 1

        list1[j + 1] = key
        list2[j + 2] = item


def order_vertices(polygon):
    centre = polygon_centre(polygon)
    angles = []
    for i in range(1, len(polygon)):
        angles.append(get_angle(centre, polygon[i]))

    dual_insertion_sort(angles, polygon)


def random_int(global_min, global_max, local_min, local_max):
    num = -1000000
    while num < global_min or num > global_max:
        num = random.randint(local_min, local_max)

    return num


def make_polygon(vertices, prob_small, colour):
    if colour:
        polygon = [colour]
    else:
        polygon = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))]

    if not random.random() < prob_small:
        for vertex in range(vertices):
            polygon.append((random.randint(10, 190), random.randint(10, 190)))
    else:
        center = (random.randint(10, 190), random.randint(10, 190))
        for vertex in range(vertices):
            polygon.append((random_int(0, 200, center[0] - 5, center[0] + 5)
                           , random_int(0, 200, center[1] - 5, center[1] + 5)))

    order_vertices(polygon)

    return polygon


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])

    return image


def initialize():
    return [make_polygon(random.randint(3, 6), 0.5, None) for i in range(random.randint(1, 10))]


def evaluate(x):
    image = draw(x)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def select(population):
    return [population[0]]


def combine(*parents):
    return copy.deepcopy(parents[0])


def mutate(x, vertex_rate, add_rate, prob_small):
    if random.random() < add_rate:
        if random.choice([True, False]) and len(x) < 100:
            polygon_colour = random.choice(x)[0]
            x.append(make_polygon(random.randint(3, 6), prob_small, polygon_colour))
        elif len(x) > 0:
            x.pop(random.randint(0, len(x) - 1))

    for polygon in x:
        if random.random() < vertex_rate:
            choice = random.choice([0, 1, 2])
            if choice == 0:
                i = random.randint(1, len(polygon) - 1)
                vertex = polygon[i]
                polygon[i] = (random_int(0, 200, vertex[0] - 10, vertex[0] + 10)
                              , random_int(0, 200, vertex[1] - 10, vertex[1] + 10))

                order_vertices(polygon)

            elif choice == 1:
                colour = polygon[0]
                polygon[0] = (random_int(0, 255, colour[0] - 10, colour[0] + 10)
                              , random_int(0, 255, colour[1] - 10, colour[1] + 10)
                              , random_int(0, 255, colour[2] - 10, colour[2] + 10)
                              , random.randint(colour[3] - 10, colour[3] + 10))

            elif choice == 3:
                i = random.randint(1, len(polygon) - 2)
                point = ((polygon[i][0] + polygon[i + 1][0]) / 2, (polygon[i][1] + polygon[i + 1][1]) / 2)
                polygon.append(point)
                order_vertices(polygon)

    return x



def run(pop_size, maximize, survival_rate, vertex_rate, add_rate, seed, save):
    fitness_evaluations = []
    for j in range(1):
        random.seed(int(seed) + j)
        population = Population.generate(initialize, evaluate, size=int(pop_size), maximize=bool(maximize))
        population.evaluate()
        count = 1
        mean = []
        gen = []

        evolution1 = (Evolution().survive(n=int(survival_rate))
                      .breed(parent_picker=select, combiner=combine)
                      .mutate(mutate_function=mutate, vertex_rate=float(vertex_rate), add_rate=float(add_rate)
                              , prob_small=0.8, elitist=True)
                      .evaluate())

        while population.current_best.fitness < 1:
            population = population.evolve(evolution1)
            sd = standard_deviation(population.individuals)

            print("Gen =", count, " Best =", population.current_best.fitness, " Worst =", population.current_worst.fitness
                  , "Polygons =", len(population.current_best.chromosome), "Standard Deviation =", sd)

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


def save_test(generations, pop, survival, vertex, add):
    minimum = 100000
    maximum = -100000
    total = 0
    for i in generations:
        if i < minimum: minimum = i
        if i > maximum: maximum = i
        total += i

    mean = total / len(generations)

    f = open("tests.md", 'a')
    row = "| Mutation | {} | {} | {} | {} | {} | {} | {} |\n".format(minimum, maximum, mean
                                                                     , pop, survival, vertex, add)
    f.write(row)
    f.close()


if __name__ == "__main__":
    params = read_config(sys.argv[1])
    run(**params[sys.argv[2]])
