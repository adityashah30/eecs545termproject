from ga import GA

def main():
    pop_size, gen_count, mutation = 5, 3, 0.3
    solver = GA(gen_count, pop_size, mutation)
    finalPop = solver.search()
    print finalPop[0].fitness

if __name__ == '__main__':
    main()
