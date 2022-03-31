from PIL import Image, ImageDraw, ImageChops
from evol import Population, Evolution
from configparser import ConfigParser
import random
import sys

TARGET = Image.open("images/darwin.png")
MAX = 255 * TARGET.size[0] * TARGET.size[1]


def make_polygon(vertices):
    polygon = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255), random.randint(30, 60))]
    for vertex in range(vertices):
        polygon.append((random.randint(10, 190), random.randint(10, 190)))

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

    return x


def run(pop_size, maximize, survival_rate, mutation_rate, seed):
    random.seed(seed)
    population = Population.generate(initialize, evaluate, size=int(pop_size), maximize=bool(maximize))
    population.evaluate()

    evolution = (Evolution().survive(fraction=float(survival_rate))
                 .breed(parent_picker=select, combiner=combine)
                 .mutate(mutate_function=mutate, rate=float(mutation_rate))
                 .evaluate())

    for i in range(50):
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
