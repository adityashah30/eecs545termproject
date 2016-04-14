from ga import GA

def main():
    pop_size = 10
    gen_count = 10
    elitism = 0.15
    mutation = 0.3
    solver = GA(gen_count, pop_size, elitism, mutation)
    finalPop = solver.search()

if __name__ == '__main__':
    main()
