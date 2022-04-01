from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
from configparser import ConfigParser
import math
import random
import sys

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

    return math.atan(point2[0] - point1[0]) / (point2[1] - point1[1])


def dual_insertion_sort(list1, list2):
    for i in range(1, len(list1)):
        key = list1[i]
        item = list2[i + 1]
        j = i - 1

        while j > 0 and list1[j] > key:
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


def make_polygon(vertices):
    polygon = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))]
    for vertex in range(vertices):
        polygon.append((random.randint(10, 190), random.randint(10, 190)))

    order_vertices(polygon)

    return polygon


def draw(solution):
    image = Image.new("RGB", (200, 200))
    canvas = ImageDraw.Draw(image, "RGBA")
    for polygon in solution:
        canvas.polygon(polygon[1:], fill=polygon[0])

    image.save("images/solution.png")
    return image


def initialize():
    return [make_polygon(3) for i in range(100)]


def evaluate(x):
    image = draw(x)
    diff = ImageChops.difference(image, TARGET)
    hist = diff.convert("L").histogram()
    count = sum(i * n for i, n in enumerate(hist))
    return (MAX - count) / MAX


def select(population):
    return [random.choice(population) for i in range(2)]


def combine(*parents):
    return [a if random.random() < 0.5 else b for a, b in zip(*parents)]


def mutate(x, rate):
    for polygon in x:
        if random.random() < rate:
            for i in range(1, len(polygon)):
                vertex = polygon[i]
                polygon[i] = (random.randint(vertex[0] - 10, vertex[0] + 10)
                              , random.randint(vertex[1] - 10, vertex[1] + 10))

            order_vertices(polygon)

    return x


def run(pop_size, maximize, survival_rate, mutation_rate, generations, seed):
    random.seed(seed)
    population = Population.generate(initialize, evaluate, size=int(pop_size), maximize=bool(maximize))
    population.evaluate()

    evolution = (Evolution().survive(fraction=float(survival_rate))
                 .breed(parent_picker=select, combiner=combine)
                 .mutate(mutate_function=mutate, rate=float(mutation_rate))
                 .evaluate())

    for i in range(int(generations)):
        population = population.evolve(evolution)
        print("i =", i, " best =", population.current_best.fitness, " worst =", population.current_worst.fitness)


def read_config(path):
    config = ConfigParser()
    config.read(path)

    values = {section: dict(config.items(section)) for section in config.sections()}

    return values


if __name__ == "__main__":
    params = read_config(sys.argv[1])
    run(**params[sys.argv[2]])
