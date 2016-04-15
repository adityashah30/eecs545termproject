from ga import GA

def main():
    pop_size, gen_count, mutation = 250, 100, 0.3
    solve_method = "logistic"
    print "Using solve method: ", solve_method
    solver = GA(gen_count, pop_size, mutation, solve_method)
    finalPop = solver.search()
    print finalPop[0].fitness

if __name__ == '__main__':
    main()
