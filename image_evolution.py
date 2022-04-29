from datetime import date

from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
from configparser import ConfigParser
import math
import random
import sys
import copy

TARGET = Image.open("images/darwin.png")
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


def make_polygon(vertices, small):
    if not small:
        polygon = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))]
        for vertex in range(vertices):
            polygon.append((random.randint(10, 190), random.randint(10, 190)))
    else:
        polygon = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))]
        center = (random.randint(10, 190), random.randint(10, 190))
        for vertex in range(vertices):
            polygon.append((random_int(0, 200, center[0] - 10, center[0] + 10)
                           , random_int(0, 200, center[1] - 10, center[1] + 10)))

    order_vertices(polygon)

    return polygon


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])

    return image


def initialize():
    return [make_polygon(random.randint(3, 6), False) for i in range(random.randint(1, 10))]


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
    for polygon in parents[0]:
        if polygon_centre(polygon)[0] <= 100:
            child.append(copy.deepcopy(polygon))
    for polygon in parents[1]:
        if polygon_centre(polygon)[0] > 100 and len(child) < 100:
            child.append(copy.deepcopy(polygon))

    return child


def mutate(x, vertex_rate, add_rate, small):
    mutation_type = random.randint(0, 2)
    if mutation_type == 0:
        if random.random() < add_rate and len(x) < 100:
            x.append(make_polygon(random.randint(3, 6), small))
        elif len(x) > 0:
            x.pop(random.randint(0, len(x) - 1))

    elif mutation_type == 1:
        for polygon in x:
            if random.random() < vertex_rate:
                i = random.randint(1, len(polygon) - 1)
                vertex = polygon[i]
                polygon[i] = (random_int(0, 200, vertex[0] - 10, vertex[0] + 10)
                              , random_int(0, 200, vertex[1] - 10, vertex[1] + 10))

            order_vertices(polygon)

    elif mutation_type == 2:
        for polygon in x:
            if random.random() < vertex_rate:
                colour = polygon[0]
                polygon[0] = (random_int(0, 255, colour[0] - 10, colour[0] + 10)
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
                  .mutate(mutate_function=mutate, vertex_rate=float(vertex_rate), add_rate=float(add_rate), small=False
                          , elitist=True)
                  .evaluate())

    evolution2 = (Evolution().survive(fraction=float(survival_rate))
                  .breed(parent_picker=select, combiner=combine)
                  .mutate(mutate_function=mutate, vertex_rate=float(vertex_rate), add_rate=float(add_rate), small=True
                          , elitist=True)
                  .evaluate())

    for i in range(1000):
        population = population.evolve(evolution1)
        print("i =", i, " best =", population.current_best.fitness, " worst =", population.current_worst.fitness, len(population.current_best.chromosome))

    for i in range(1001, 1500):
        population = population.evolve(evolution2)
        print("i =", i, " best =", population.current_best.fitness, " worst =", population.current_worst.fitness,)

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
